"""Software stand-in for firmware/bb8_receiver: same parsing, clamp, slew, watchdog and e-stop logic.

    python tools/mock_esp32.py            # listens on 127.0.0.1:5005
    python app/companion.py --source sim --esp32 127.0.0.1

Prints a line per second and writes a summary JSON on Ctrl-C (or after --seconds).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import protocol as P  # noqa: E402

MAX_DUTY, SLEW_PER_S, CMD_TIMEOUT_S, CTRL_PERIOD_S, TEL_PERIOD_S = 0.80, 4.0, 0.300, 0.010, 0.100


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=P.CMD_PORT)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((args.bind, args.port))
    s.setblocking(False)
    print(f"mock ESP32 on {args.bind}:{args.port}", flush=True)
    peer = None
    last_seq, have_seq, estop = 0, False, False
    tgt = [0.0, 0.0]
    app = [0.0, 0.0]
    t_cmd = 0.0
    stats = dict(rx=0, bad=0, stale=0, estop_events=0, watchdog_stops=0, max_abs_applied=0.0)
    t0 = time.monotonic()
    t_ctrl = t_tel = t_print = t0
    timed_out_prev = True
    try:
        while not args.seconds or time.monotonic() - t0 < args.seconds:
            while True:
                try:
                    b, addr = s.recvfrom(64)
                except BlockingIOError:
                    break
                except ConnectionResetError:  # Windows: ICMP port-unreachable from a closed peer
                    continue
                c = P.unpack_cmd(b)
                now = time.monotonic()
                if c is None:
                    stats["bad"] += 1
                    continue
                d = (c["seq"] - last_seq) & 0xFFFF
                fresh = not have_seq or now - t_cmd > CMD_TIMEOUT_S or 0 < d < 0x8000
                if not fresh:
                    stats["stale"] += 1
                    continue
                have_seq, last_seq, peer = True, c["seq"], addr
                stats["rx"] += 1
                if c["estop"]:
                    if not estop:
                        stats["estop_events"] += 1
                    estop = True
                elif estop and c["uL"] == 0.0 and c["uR"] == 0.0:
                    estop = False
                tgt = [clamp(c["uL"], -MAX_DUTY, MAX_DUTY), clamp(c["uR"], -MAX_DUTY, MAX_DUTY)]
                t_cmd = now
            now = time.monotonic()
            if now - t_ctrl >= CTRL_PERIOD_S:
                dt, t_ctrl = now - t_ctrl, now
                timed_out = now - t_cmd > CMD_TIMEOUT_S
                if timed_out and not timed_out_prev and have_seq:
                    stats["watchdog_stops"] += 1
                timed_out_prev = timed_out
                for i in range(2):
                    t = 0.0 if (timed_out or estop) else tgt[i]
                    app[i] = 0.0 if estop else app[i] + clamp(t - app[i], -SLEW_PER_S * dt, SLEW_PER_S * dt)
                stats["max_abs_applied"] = max(stats["max_abs_applied"], abs(app[0]), abs(app[1]))
            if peer and now - t_tel >= TEL_PERIOD_S:
                t_tel = now
                s.sendto(P.pack_tel(last_seq, int((now - t0) * 1000), app[0], app[1], 0.0, 0.0, 7.4, stats["rx"]), peer)
            if now - t_print >= 1.0:
                t_print = now
                print(f"rx {stats['rx']:5d}  applied ({app[0]:+.2f},{app[1]:+.2f})  estop {estop}  "
                      f"watchdog {'STOP' if now - t_cmd > CMD_TIMEOUT_S else 'ok'}", flush=True)
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    print(json.dumps(stats))
    if args.summary:
        Path(args.summary).write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
