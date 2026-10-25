"""Isolate the PID / residual controllers from localization: run them on oracle bearing and range.

    python scripts/debug_pid.py --terrain flat --n 8 --seconds 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.acoustics import Renderer  # noqa: E402
from bb8dl.controllers import OracleLoc, PIDFollower, TrackPursuit  # noqa: E402
from bb8dl.evaluate import aggregate, episode_metrics, run_episodes  # noqa: E402
from bb8dl.residual import Residual  # noqa: E402
from bb8dl.world import make_scenarios  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--terrain", default="flat")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--seconds", type=float, default=30)
args = ap.parse_args()
T = int(args.seconds / C.CTRL_DT)
seed = C.SEEDS["eval"] + 100 * C.TERRAINS.index(args.terrain)
scen = make_scenarios(args.terrain, args.n, seed, "A_us", T)
ren = Renderer("cpu")
for name, ctrl in [("oracle_pursuit", TrackPursuit(OracleLoc(), "o")),
                   ("oracle_pid", PIDFollower(OracleLoc())),
                   ("oracle_pid_res", PIDFollower(OracleLoc(), Residual(C.CHECKPOINTS / "residual.pt")))]:
    L = run_episodes(ctrl, scen, ren, seed, T)
    a = aggregate(episode_metrics(L))
    u = L["u"]
    print(f"{name:15s} success {a['success']['mean']:.2f}  dist {a['mean_dist']['mean']:.2f}  tumble steps {a['tumbles']['mean']:.1f}  "
          f"|u| mean {np.abs(u).mean():.2f}  frac |u|>0.95 {np.mean(np.abs(u) > 0.95):.2f}  "
          f"max |pitch| {np.abs(L['pitch']).max():.2f} rad  first tumble step {np.argmax(L['tumble'].any(0)) if L['tumble'].any() else -1}")
