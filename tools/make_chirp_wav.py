"""Write the beacon WAVs the phone plays: one 10 ms chirp every 50 ms.

    python tools/make_chirp_wav.py            # -> assets/chirp_us_20-22.5k.wav, assets/chirp_fb_17-19.5k.wav

48 kHz, 16-bit mono, 10 s each (loop it; 10 s is a whole number of periods, so loops are seamless). The waveform is exactly acoustics.chirp(),
the template the simulator, matched filter and network were built around.

Phone caveat: most phone audio paths low-pass near 20 kHz (DAC reconstruction and
anti-imaging filters, speaker roll-off, and some OS resamplers or "audio enhancement"
stages). Many phones therefore emit the 20-22.5 kHz chirp heavily attenuated or not at
all, and some add audible clicks or intermodulation products. Check with a measurement
mic or a second phone running a spectrogram app before trusting it; fall back to the
17-19.5 kHz chirp, which most adults over ~25 cannot hear but some younger people and
pets can. Play at the device's native 48 kHz with every effect disabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.acoustics import chirp  # noqa: E402


def main(seconds: float = 10.0, level: float = 0.7) -> None:
    out = Path(__file__).resolve().parents[1] / "assets"
    out.mkdir(exist_ok=True)
    period = int(round(C.CHIRP_PERIOD * C.FS))
    for name, label in [("us", "20-22.5k"), ("fb", "17-19.5k")]:
        c = chirp(name)
        c = level * c / np.abs(c).max()
        one = np.zeros(period)
        one[: len(c)] = c
        y = np.tile(one, int(seconds / C.CHIRP_PERIOD))
        p = out / f"chirp_{name}_{label}.wav"
        sf.write(p, y, C.FS, subtype="PCM_16")
        print(f"wrote {p}  ({len(y) / C.FS:.0f} s, band {C.CHIRPS[name][0]}-{C.CHIRPS[name][1]} Hz)")


if __name__ == "__main__":
    main()
