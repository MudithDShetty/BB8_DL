"""Software stand-in for firmware/bb8_mics: streams simulated 4-channel audio in the mic-ESP32 UDP format.

    python tools/fake_mic_stream.py --terrain flat --seconds 30
    python app/companion.py --source udp --mics 127.0.0.1 --gain-db <printed value> --calib-range <printed value>

The robot stays put (the stream does not listen to wheel commands) while the
scripted person walks around it. One chirp is emitted every 50 ms at stream sample
index E0 + k*2400; the 2048-sample window after each emission is the simulator's
rendering, the remaining 352 samples are noise at the same level. Ground truth per
chirp goes to --truth as JSON lines (emission sample index, azimuth, range).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl import protocol as P  # noqa: E402
from bb8dl.acoustics import Renderer  # noqa: E402
from bb8dl.world import World, make_scenarios  # noqa: E402

E0 = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=P.MIC_PORT)
    ap.add_argument("--terrain", default="flat", choices=C.TERRAINS)
    ap.add_argument("--channel", default="A_us", choices=C.CHANNEL_NAMES)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--loss", type=float, default=0.0, help="fraction of packets to drop (robustness test)")
    ap.add_argument("--truth", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = int(args.seconds / C.CTRL_DT)
    world = World(make_scenarios(args.terrain, 1, args.seed, args.channel, T), Renderer(dev), args.seed)
    rng = np.random.default_rng(args.seed)
    period = int(round(C.CHIRP_PERIOD * C.FS))
    first = world.observe()
    w0 = first["audio"][0].float().cpu().numpy().T
    scale = 0.25 / np.abs(w0).max()
    print(f"scale {scale:.4g}  ->  use --gain-db {-20 * np.log10(scale):.2f}  "
          f"--calib-range {float(first['true_range'][0]):.2f}  (true range of the first chirp)", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    print(f"waiting for SUB1 on udp {args.port} ...", flush=True)
    while True:
        b, peer = sock.recvfrom(16)
        if b == P.SUBSCRIBE:
            break
    print(f"streaming to {peer}", flush=True)
    sock.setblocking(False)
    truth_f = open(args.truth, "w") if args.truth else None

    seq, sample, sent, dropped = 0, 0, 0, 0
    pending = np.zeros((E0, 4), np.float32)
    t_start = time.monotonic()
    for k in range(T):
        o = first if k == 0 else world.observe()
        w = o["audio"][0].float().cpu().numpy().T          # (2048, 4)
        if truth_f:
            truth_f.write(json.dumps(dict(k=k, emission_sample=E0 + k * period,
                                          az_deg=float(np.rad2deg(o["true_az"][0])),
                                          range_m=float(o["true_range"][0]))) + "\n")
        noise_sd = w[:150].std(0, keepdims=True)
        gap = rng.standard_normal((period - C.N_WIN, 4)).astype(np.float32) * noise_sd
        block = np.concatenate([pending, w, gap]) if k == 0 else np.concatenate([w, gap])
        pending = np.zeros((0, 4), np.float32)
        q = np.clip(np.round(block * scale * 32768), -32768, 32767).astype(np.int16)
        for i in range(0, len(q), P.FRAMES_PER_AUDIO_PACKET):
            frames = q[i:i + P.FRAMES_PER_AUDIO_PACKET]
            due = t_start + (sample + len(frames)) / C.FS
            time.sleep(max(0.0, due - time.monotonic()))
            if rng.random() >= args.loss:
                sock.sendto(P.pack_audio(seq, sample, frames), peer)
                sent += 1
            else:
                dropped += 1
            seq = (seq + 1) & 0xFFFF
            sample += len(frames)
        world.step(np.zeros((1, 2)))
    print(f"done: {T} chirps, {sent} packets sent, {dropped} dropped", flush=True)


if __name__ == "__main__":
    main()
