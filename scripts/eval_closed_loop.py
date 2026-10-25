"""Closed-loop following evaluation: every method x terrain on identical scenarios.

    python scripts/eval_closed_loop.py --n 30 --channel A_us
    python scripts/eval_closed_loop.py --n 30 --channel B_fb

Writes results/closed_loop_<channel>.json, results/traj_<channel>.npz (first
episodes, for figures) and full logs to <BB8DL_DATA>/closed_loop/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.acoustics import Renderer  # noqa: E402
from bb8dl.controllers import build_controller as build  # noqa: E402
from bb8dl.evaluate import aggregate, episode_metrics, run_episodes  # noqa: E402
from bb8dl.world import make_scenarios  # noqa: E402

ALL = ["oracle", "gcc_track", "dl_track", "dl_pid", "dl_pid_res", "rl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--channel", default="A_us")
    ap.add_argument("--methods", nargs="*", default=ALL)
    ap.add_argument("--terrains", nargs="*", default=C.TERRAINS)
    ap.add_argument("--seconds", type=float, default=C.EPISODE_S)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ren = Renderer(dev)
    T = int(args.seconds / C.CTRL_DT)
    out_json = C.RESULTS / f"closed_loop_{args.channel}.json"
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    res.setdefault("config", {})
    res["config"].update(n=args.n, channel=args.channel, seconds=args.seconds, standoff=C.STANDOFF,
                         follow_radius=C.FOLLOW_RADIUS, success_frac=C.SUCCESS_FRAC, warmup_s=C.WARMUP_S,
                         seed_base=C.SEEDS["eval"], device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    res.setdefault("results", {})
    traj_path = C.RESULTS / f"traj_{args.channel}.npz"
    traj = dict(np.load(traj_path)) if traj_path.exists() else {}
    full_dir = C.DATA_ROOT / "closed_loop"
    full_dir.mkdir(parents=True, exist_ok=True)
    for ti, terr in enumerate(args.terrains):
        seed = C.SEEDS["eval"] + 100 * C.TERRAINS.index(terr)
        scen = make_scenarios(terr, args.n, seed, args.channel, T)
        for name in args.methods:
            ctrl = build(name, dev)
            L = run_episodes(ctrl, scen, ren, seed, T)
            m = episode_metrics(L)
            agg = aggregate(m)
            res["results"].setdefault(name, {})[terr] = dict(agg=agg, per_episode={k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in m.items()})
            print(f"[{args.channel}] {terr:9s} {name:11s} success {agg['success']['mean']:.2f}  "
                  f"mean_dist {agg['mean_dist']['mean']:.2f} m  bMAE {agg['loc_bearing_mae']['mean']}  "
                  f"tumbles {agg['tumbles']['mean']:.2f}  ({L['wall_seconds']:.0f}s)", flush=True)
            for k in ["robot", "person", "dist"]:
                traj[f"{name}|{terr}|{k}"] = L[k][:4].astype(np.float32)
            np.savez_compressed(full_dir / f"{args.channel}_{name}_{terr}.npz",
                                **{k: v for k, v in L.items() if isinstance(v, np.ndarray)})
            out_json.write_text(json.dumps(res, indent=1))
            np.savez_compressed(traj_path, **traj)


if __name__ == "__main__":
    main()
