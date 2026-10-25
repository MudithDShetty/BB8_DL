"""BB-8 companion app: deep localizer + controller on the laptop, wheel commands to the ESP32 over UDP.

    # simulated replay (closed loop in the simulator; commands also go out over UDP)
    python app/companion.py --source sim --terrain grass --controller dl_pid_res --esp32 127.0.0.1
    # live 4-channel audio from the mic ESP32 (firmware/bb8_mics)
    python app/companion.py --source udp --mics 192.168.4.3 --esp32 192.168.4.1 --calib-range 1.0

Status page with an e-stop button: http://127.0.0.1:8090/

Live-mode range caveat: range comes from time of flight, which needs the chirp
emission time. The phone and robot share no clock, so the app locks a 50 ms grid
to the chirp arrivals while the phone is held at a known distance (--calib-range)
and then free-runs on that grid. Crystal drift between phone and ESP32 (tens of
ppm) moves the grid by tens of microseconds per second, i.e. about 1 cm/s of range
drift; re-lock (button on the status page) when range estimates wander. Bearing
does not depend on this.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl import protocol as P  # noqa: E402
from bb8dl.acoustics import chirp  # noqa: E402
from bb8dl.controllers import build_controller  # noqa: E402

APP_MAX_DUTY = 0.8
STATUS_HTML = (Path(__file__).parent / "status.html").read_text(encoding="utf-8")


def hostport(s: str, default_port: int) -> tuple[str, int]:
    h, _, p = s.partition(":")
    return h, int(p) if p else default_port


# ---------------------------------------------------------------- drive link
class DriveLink:
    """Sends 20-byte command packets; receives telemetry on the same socket."""

    def __init__(self, addr: tuple[str, int] | None):
        self.addr = addr
        self.seq = 0
        self.t0 = time.monotonic()
        self.tel: dict | None = None
        self.tel_time = 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)
        if addr:
            threading.Thread(target=self._rx, daemon=True).start()

    def send(self, uL: float, uR: float, estop: bool = False) -> None:
        if not self.addr:
            return
        uL, uR = (float(np.clip(x, -APP_MAX_DUTY, APP_MAX_DUTY)) for x in (uL, uR))
        self.seq = (self.seq + 1) & 0xFFFF
        t_ms = int((time.monotonic() - self.t0) * 1000)
        self.sock.sendto(P.pack_cmd(self.seq, t_ms, uL, uR, estop), self.addr)

    def _rx(self):
        while True:
            try:
                b, _ = self.sock.recvfrom(64)
            except (socket.timeout, OSError):
                continue
            t = P.unpack_tel(b)
            if t:
                self.tel, self.tel_time = t, time.monotonic()

    def proprio(self) -> np.ndarray:
        """(1,5) [v, yaw rate, pitch, pitch rate, roll] estimated from telemetry.

        With encoders, wheel rates are normalised by the no-load rate; without, the
        applied duties are inverted through the nominal mixer. There is no IMU on the
        reference hardware, so pitch terms are zero (the residual then sees flat ground).
        """
        t = self.tel
        if t is None or time.monotonic() - self.tel_time > 0.5:
            return np.zeros((1, 5))
        if abs(t["wL"]) + abs(t["wR"]) > 1e-3:
            nL, nR = t["wL"] / C.ROBOT.w_noload, t["wR"] / C.ROBOT.w_noload
        else:
            nL, nR = t["uL"], t["uR"]
        v = C.V_NOM_MAX * (nL + nR) / 2
        w = C.W_NOM_MAX * (nR - nL)
        return np.array([[v, w, 0.0, 0.0, 0.0]])


# ---------------------------------------------------------------- audio sources
class SimSource:
    """Closed loop in the simulator: one scripted walk, rendered on the GPU."""

    def __init__(self, terrain: str, channel: str, seed: int, seconds: float, dev):
        from bb8dl.acoustics import Renderer
        from bb8dl.world import World, make_scenarios
        T = int(seconds / C.CTRL_DT)
        self.world = World(make_scenarios(terrain, 1, seed, channel, T), Renderer(dev), seed)
        self.T = T
        self.k = 0

    def observe(self, link):
        if self.k >= self.T:
            return None
        o = self.world.observe()
        o["truth"] = dict(az=float(o["true_az"][0]), range=float(o["true_range"][0]),
                          robot=[float(self.world.s.x[0]), float(self.world.s.y[0])],
                          person=self.world.person_xy()[0].tolist())
        return o

    def after(self, u):
        self.world.step(u)
        self.k += 1


class UdpMicSource:
    """Live 4-channel stream from the mic ESP32, windowed on a locked chirp grid."""

    def __init__(self, mics: tuple[str, int], chirp_name: str, calib_range: float, gain_db: float, dev):
        self.addr = mics
        self.ring = P.AudioRing(seconds=3.0)
        self.ci = 0 if chirp_name == "us" else 1
        self.tmpl = chirp(chirp_name)
        self.calib_range = calib_range
        self.gain = 10 ** (gain_db / 20)
        self.dev = dev
        self.period = int(round(C.CHIRP_PERIOD * C.FS))
        self.grid0: int | None = None
        self.next_start: int | None = None
        self.odom = np.zeros((1, 3))
        self.last_ok = time.monotonic()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.settimeout(0.2)
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._subscribe, daemon=True).start()

    def _subscribe(self):
        while True:
            self.sock.sendto(P.SUBSCRIBE, self.addr)
            time.sleep(1.0)

    def _rx(self):
        while True:
            try:
                b, _ = self.sock.recvfrom(2048)
            except (socket.timeout, OSError):
                continue
            a = P.unpack_audio(b)
            if a:
                self.ring.push(*a)

    def relock(self):
        self.grid0 = None

    def _lock(self) -> bool:
        """Fold the summed matched-filter envelope of the last second on the chirp period."""
        n = C.FS
        head = self.ring.head  # snapshot: the receive thread keeps advancing it
        x = self.ring.window(head - n, n)
        if x is None:
            return False
        L = n + len(self.tmpl)
        X = np.fft.rfft(x, L) * np.conj(np.fft.rfft(self.tmpl, L))[None]
        env = np.abs(np.fft.irfft(X, L)[:, :n]).sum(0)
        fold = env[: (n // self.period) * self.period].reshape(-1, self.period).sum(0)
        arrival = int(np.argmax(fold))
        dh = C.PHONE_HEIGHT_NOMINAL - C.MIC_HEIGHT
        tof = np.hypot(self.calib_range, dh) / C.C_NOMINAL * C.FS
        start0 = head - n + arrival - int(round(tof))
        self.grid0 = start0
        self.next_start = start0 + self.period * ((head - C.N_WIN - start0) // self.period)
        print(f"[lock] arrival phase {arrival} samples, calib range {self.calib_range} m", flush=True)
        return True

    def observe(self, link):
        if self.grid0 is None and not self._lock():
            return {"wait": True}
        newest, head = self.next_start, self.ring.head
        while newest + self.period + C.N_WIN <= head:
            newest += self.period
        w = self.ring.window(newest, C.N_WIN)
        if w is None:
            return {"wait": True}
        self.next_start = newest + self.period
        self.last_start = newest
        self.last_ok = time.monotonic()
        pr = link.proprio()
        self.odom[0, 2] = np.mod(self.odom[0, 2] + pr[0, 1] * C.CTRL_DT + np.pi, 2 * np.pi) - np.pi
        self.odom[0, :2] += pr[0, 0] * C.CTRL_DT * np.array([np.cos(self.odom[0, 2]), np.sin(self.odom[0, 2])])
        audio = torch.from_numpy(w[None] * self.gain).to(self.dev)
        return dict(audio=audio, chirp_idx=np.array([self.ci]), proprio=pr, odom=self.odom.copy())

    def after(self, u):
        pass


# ---------------------------------------------------------------- status server
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.d: dict = {}
        self.estop = False
        self.relock = False

    def update(self, **kw):
        with self.lock:
            self.d.update(kw)

    def json(self) -> bytes:
        with self.lock:
            return json.dumps(dict(self.d, estop=self.estop)).encode()


def serve(state: State, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/state":
                self._send(state.json(), "application/json")
            else:
                self._send(STATUS_HTML.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path == "/estop":
                state.estop = True
            elif self.path == "/arm":
                state.estop = False
            elif self.path == "/relock":
                state.relock = True
            self._send(b"{}", "application/json")

    # bound to localhost only: anyone who can reach this page can drive the robot
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"status page: http://127.0.0.1:{port}/", flush=True)


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sim", "udp"], default="sim")
    ap.add_argument("--controller", default="dl_pid_res",
                    choices=["dl_track", "dl_pid", "dl_pid_res", "gcc_track", "rl", "oracle"])
    ap.add_argument("--ckpt", default=str(C.CHECKPOINTS / "localizer_full.pt"))
    ap.add_argument("--esp32", default="", help="drive ESP32 host[:port]; empty = do not send")
    ap.add_argument("--mics", default="192.168.4.3", help="mic ESP32 host[:port] (udp source)")
    ap.add_argument("--chirp", choices=["us", "fb"], default="us")
    ap.add_argument("--calib-range", type=float, default=1.0)
    ap.add_argument("--gain-db", type=float, default=0.0, help="scale live audio into the simulator's level range")
    ap.add_argument("--terrain", default="flat", choices=C.TERRAINS)
    ap.add_argument("--channel", default="A_us", choices=C.CHANNEL_NAMES)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fast", action="store_true", help="sim: do not pace to real time")
    ap.add_argument("--http", type=int, default=8090)
    ap.add_argument("--log", default="", help="write per-step JSON lines here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    if args.source == "udp" and args.controller == "oracle":
        sys.exit("oracle needs simulator ground truth")
    print(f"device {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}  controller {args.controller}", flush=True)
    ctrl = build_controller(args.controller, dev, args.ckpt)
    ctrl.reset(1)
    link = DriveLink(hostport(args.esp32, P.CMD_PORT) if args.esp32 else None)
    if args.source == "sim":
        src = SimSource(args.terrain, args.channel, args.seed, args.seconds, dev)
    else:
        src = UdpMicSource(hostport(args.mics, P.MIC_PORT), args.chirp, args.calib_range, args.gain_db, dev)
    state = State()
    serve(state, args.http)
    logf = open(args.log, "w") if args.log else None

    k, t_next, lat = 0, time.monotonic(), []
    try:
        while True:
            if args.source == "udp" or not args.fast:
                time.sleep(max(0.0, t_next - time.monotonic()))
            t_next += C.CTRL_DT
            if state.relock and hasattr(src, "relock"):
                src.relock()
                state.relock = False
            obs = src.observe(link)
            if obs is None:
                break
            if obs.get("wait"):
                if args.source == "udp" and time.monotonic() - src.last_ok > 0.5:
                    link.send(0.0, 0.0, state.estop)  # no audio: hold still
                state.update(status="waiting for audio / chirp lock", lost_packets=src.ring.lost_packets)
                continue
            t0 = time.perf_counter()
            u = ctrl.act(obs)
            lat.append(1000 * (time.perf_counter() - t0))
            if state.estop:
                u = np.zeros_like(u)
            link.send(u[0, 0], u[0, 1], state.estop)
            meas = getattr(ctrl, "last_meas", None)
            rec = dict(step=k, t=k * C.CTRL_DT, u=[float(u[0, 0]), float(u[0, 1])],
                       est_az_deg=float(np.rad2deg(meas[0][0])) if meas else None,
                       est_range_m=float(meas[1][0]) if meas else None,
                       ctrl_ms=lat[-1], telemetry=link.tel, status="running")
            if "truth" in obs:
                rec["truth"] = obs["truth"]
                rec["true_az_deg"] = float(np.rad2deg(obs["truth"]["az"]))
            if args.source == "udp":
                rec["lost_packets"] = src.ring.lost_packets
                rec["window_start"] = int(src.last_start)
            state.update(**rec)
            if logf:
                logf.write(json.dumps(rec) + "\n")
            src.after(u)
            k += 1
            if args.source == "udp" and k >= args.seconds / C.CTRL_DT:
                break
            if k % 20 == 0:
                print(f"t {k * C.CTRL_DT:5.1f}s  u ({u[0, 0]:+.2f},{u[0, 1]:+.2f})  est az {rec['est_az_deg']}  "
                      f"r {rec['est_range_m']}  ctrl {np.mean(lat[-20:]):.1f} ms", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            link.send(0.0, 0.0, estop=True)
            time.sleep(0.02)
        if logf:
            logf.close()
        if lat:
            print(f"steps {k}  controller latency mean {np.mean(lat):.1f} ms  p95 {np.percentile(lat, 95):.1f} ms")


if __name__ == "__main__":
    main()
