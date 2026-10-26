"""
BB-8 acoustic beacon toolkit (2-mic stage).

  python beacon.py make-wav chirp.wav            # put this on the phone, play on loop
  python beacon.py meter   --port COM5           # check both mics are alive
  python beacon.py capture --port COM5           # record 0.3 s from the ESP32 and estimate angle
  python beacon.py watch   --port COM5           # live chirp strength, ~1 reading/sec
  python beacon.py analyze rec.npz               # re-run analysis on a saved recording
  python beacon.py simulate --angle 30           # test the algorithm with fake data

Angle convention: 0 deg = straight ahead (broadside), positive = source on the LEFT mic's side.
"""
import argparse
import struct
import sys
import time
import wave

import numpy as np
from scipy.signal import fftconvolve, hilbert, find_peaks
from scipy.signal.windows import tukey

FS = 48000
F0, F1 = 18000.0, 20000.0     # chirp sweep (Hz)
CHIRP_DUR = 0.010             # 10 ms chirp
PERIOD = 0.100                # one chirp every 100 ms
SPEED_OF_SOUND = 343.0        # m/s at ~20 C
FC = 0.5 * (F0 + F1)          # carrier of the matched-filter output


# ---------------------------------------------------------------- chirp
def chirp_template(fs=FS):
    t = np.arange(int(fs * CHIRP_DUR)) / fs
    k = (F1 - F0) / CHIRP_DUR
    c = np.sin(2 * np.pi * (F0 * t + 0.5 * k * t * t))
    return c * tukey(len(c), 0.3)          # soft edges = no audible clicks


def make_wav(path, seconds=60, amplitude=0.8):
    tmpl = chirp_template()
    period = np.zeros(int(FS * PERIOD))
    period[: len(tmpl)] = tmpl
    sig = np.tile(period, int(seconds / PERIOD)) * amplitude
    pcm = (sig * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(pcm.tobytes())
    print(f"wrote {path}: {seconds}s, {F0/1e3:.0f}-{F1/1e3:.0f} kHz chirp every {PERIOD*1e3:.0f} ms")


# ---------------------------------------------------------------- analysis
def _parabolic(y, i):
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        d = a - 2 * b + c
        if d != 0:
            return i + 0.5 * (a - c) / d
    return float(i)


def estimate_angles(left, right, spacing, fs=FS, verbose=True):
    """Return list of (time_s, angle_deg, snr_db) for every chirp found."""
    tmpl = chirp_template(fs)
    left = left - np.mean(left)
    right = right - np.mean(right)
    # matched filter = band-pass + pulse compression in one step
    aL = hilbert(fftconvolve(left, tmpl[::-1], mode="same"))
    aR = hilbert(fftconvolve(right, tmpl[::-1], mode="same"))
    eL, eR = np.abs(aL), np.abs(aR)
    env = eL + eR

    noise = np.median(env) + 1e-12
    peaks, _ = find_peaks(env, height=6 * noise, distance=int(0.05 * fs))

    max_tau = spacing / SPEED_OF_SOUND
    win = int(np.ceil(max_tau * fs)) + 6
    results = []
    for p in peaks:
        lo, hi = max(p - win, 1), min(p + win, len(env) - 1)
        pL = lo + int(np.argmax(eL[lo:hi]))
        pR = lo + int(np.argmax(eR[lo:hi]))
        # coarse delay from envelope peaks (unambiguous, a bit noisy)
        tau_coarse = (_parabolic(eR, pR) - _parabolic(eL, pL)) / fs
        # fine delay from carrier phase (precise, ambiguous by 1/FC)
        c0 = (pL + pR) // 2
        z = np.sum(aR[c0 - 3:c0 + 4] * np.conj(aL[c0 - 3:c0 + 4]))
        tau_fine = -np.angle(z) / (2 * np.pi * FC)
        n = np.round((tau_coarse - tau_fine) * FC)
        tau = np.clip(tau_fine + n / FC, -max_tau, max_tau)
        angle = np.degrees(np.arcsin(tau / max_tau))
        snr = 20 * np.log10(env[p] / noise)
        results.append((p / fs, angle, snr))
        if verbose:
            print(f"  chirp @ {p/fs*1e3:6.1f} ms   angle {angle:+6.1f} deg   "
                  f"(coarse {np.degrees(np.arcsin(np.clip(tau_coarse/max_tau,-1,1))):+6.1f})   SNR {snr:4.1f} dB")
    if verbose:
        if results:
            med = np.median([r[1] for r in results])
            print(f"=> {len(results)} chirps, median angle {med:+.1f} deg "
                  f"({'LEFT' if med > 5 else 'RIGHT' if med < -5 else 'AHEAD'})")
        else:
            print("=> no chirps found. Check phone volume/distance and the spectrogram plot.")
    return results


def plot(left, right, path, fs=FS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for a, x, name in zip(ax, (left, right), ("LEFT mic", "RIGHT mic")):
        a.specgram(x - np.mean(x), NFFT=512, Fs=fs, noverlap=384, cmap="magma")
        a.set_ylabel("Hz")
        a.set_title(name)
        a.axhspan(F0, F1, color="cyan", alpha=0.15)
    ax[1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"spectrogram saved to {path} (chirps should show as short bars in the cyan band)")


# ---------------------------------------------------------------- serial
def _open(port, baud):
    import serial  # pip install pyserial
    s = serial.Serial(port, baud, timeout=3)
    time.sleep(2.0)              # board may reset when the port opens
    s.reset_input_buffer()
    return s


def _read_capture(s):
    s.write(b"r")
    buf = b""
    deadline = time.time() + 10
    while b"AUD0" not in buf:
        buf += s.read(1)
        if time.time() > deadline:
            sys.exit("no data from ESP32 (wrong port? sketch not running?)")
    n, fs = struct.unpack("<II", s.read(8))
    raw = s.read(n * 4)
    if len(raw) != n * 4:
        sys.exit(f"short read: got {len(raw)} of {n*4} bytes")
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(float)
    return pcm[:, 0], pcm[:, 1], fs


def cmd_meter(a):
    s = _open(a.port, a.baud)
    print("Ctrl+C to stop. Tap/scratch near each mic; its number should jump.")
    try:
        while True:
            s.write(b"m")
            print(s.readline().decode(errors="ignore").strip())
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass


def cmd_capture(a):
    s = _open(a.port, a.baud)
    left, right, fs = _read_capture(s)
    np.savez(a.out, left=left, right=right, fs=fs)
    print(f"saved {a.out} ({len(left)} samples @ {fs} Hz)")
    estimate_angles(left, right, a.spacing, fs)
    plot(left, right, a.out.replace(".npz", ".png"), fs)


def chirp_snr(left, right, fs=FS):
    """Best chirp SNR (dB) in a recording, from whichever channel is alive."""
    tmpl = chirp_template(fs)
    best = 0.0
    for x in (left, right):
        if not np.any(x):
            continue
        env = np.abs(hilbert(fftconvolve(x - np.mean(x), tmpl[::-1], mode="same")))
        best = max(best, 20 * np.log10(env.max() / (np.median(env) + 1e-12)))
    return best


def cmd_watch(a):
    s = _open(a.port, a.baud)
    print("Live chirp strength (about 1 reading/sec). Ctrl+C to stop.")
    print("Above ~16 dB = chirp heard clearly; ~11 dB = just room noise.\n")
    try:
        while True:
            left, right, fs = _read_capture(s)
            snr = chirp_snr(left, right, fs)
            bar = "#" * int(max(0, snr - 10))
            tag = "HEARD" if snr >= 16 else "weak " if snr >= 13 else "---  "
            print(f"{snr:5.1f} dB  {tag} {bar}")
    except KeyboardInterrupt:
        pass


def cmd_analyze(a):
    d = np.load(a.file)
    fs = int(d["fs"])
    estimate_angles(d["left"], d["right"], a.spacing, fs)
    plot(d["left"], d["right"], a.file.replace(".npz", ".png"), fs)


# ---------------------------------------------------------------- simulation
def _delay(x, tau, fs):
    n = 2 * len(x)
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * tau), n)[: len(x)]


def simulate(angle, spacing=0.05, snr_db=20, echo=True, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n = int(0.3 * FS)
    src = np.zeros(n)
    tmpl = chirp_template()
    for k in range(3):
        i = int((0.02 + k * PERIOD) * FS)
        src[i:i + len(tmpl)] += tmpl
    tau = spacing * np.sin(np.radians(angle)) / SPEED_OF_SOUND  # >0: left hears first
    left = _delay(src, -tau / 2, FS)
    right = _delay(src, +tau / 2, FS)
    if echo:  # wall reflection: 4 ms later, 50% amplitude, from 60 deg the other way
        te = spacing * np.sin(np.radians(-60)) / SPEED_OF_SOUND
        left += 0.5 * _delay(src, 0.004 - te / 2, FS)
        right += 0.5 * _delay(src, 0.004 + te / 2, FS)
    # loud low-frequency "room noise" (speech/fans) + white noise
    low = fftconvolve(rng.standard_normal(n), np.ones(12) / 12, mode="same") * 3.0
    white = rng.standard_normal(n) * np.std(tmpl) * 10 ** (-snr_db / 20)
    left += low + white
    right += low + rng.standard_normal(n) * np.std(white)
    if verbose:
        print(f"true angle {angle:+.1f} deg, spacing {spacing*100:.0f} cm, SNR {snr_db} dB, echo={echo}")
    return estimate_angles(left, right, spacing, FS, verbose)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-wav"); m.add_argument("path"); m.add_argument("--seconds", type=int, default=60)
    for name in ("meter", "capture", "watch"):
        q = sub.add_parser(name)
        q.add_argument("--port", required=True)
        q.add_argument("--baud", type=int, default=921600)
        q.add_argument("--spacing", type=float, default=0.05, help="mic centre-to-centre distance in metres")
        q.add_argument("--out", default="rec.npz")
    q = sub.add_parser("analyze"); q.add_argument("file"); q.add_argument("--spacing", type=float, default=0.05)
    q = sub.add_parser("simulate"); q.add_argument("--angle", type=float, default=30)
    q.add_argument("--spacing", type=float, default=0.05); q.add_argument("--snr", type=float, default=20)
    a = p.parse_args()

    if a.cmd == "make-wav":
        make_wav(a.path, a.seconds)
    elif a.cmd == "meter":
        cmd_meter(a)
    elif a.cmd == "capture":
        cmd_capture(a)
    elif a.cmd == "watch":
        cmd_watch(a)
    elif a.cmd == "analyze":
        cmd_analyze(a)
    elif a.cmd == "simulate":
        simulate(a.angle, a.spacing, a.snr)


if __name__ == "__main__":
    main()
