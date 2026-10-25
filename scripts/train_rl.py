"""Train the PPO follower (stable-baselines3) on the batched BB-8 environment.

    python scripts/train_rl.py --steps 3000000

Episodes are 30 s; each of the 32 parallel envs samples a terrain uniformly at
every reset. Logs the learning curve to results/train_rl.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.rl_env import BB8VecEnv  # noqa: E402


class CurveLogger(BaseCallback):
    """Learning curve to JSON after every rollout; model checkpoint every `ckpt_every` steps."""

    def __init__(self, out_path: Path, ckpt_path: Path, ckpt_every: int = 250_000, rows=None, wall0: float = 0.0):
        super().__init__()
        self.out_path, self.ckpt_path, self.ckpt_every = out_path, ckpt_path, ckpt_every
        self.rows = rows or []
        self.buf_r, self.buf_l, self.buf_fail = [], [], []
        self.t0 = time.time() - wall0
        self.next_ckpt = None

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode" in info:
                self.buf_r.append(info["episode"]["r"])
                self.buf_l.append(info["episode"]["l"])
                self.buf_fail.append(not info.get("TimeLimit.truncated", False))
        return True

    def _on_rollout_end(self) -> None:
        if self.next_ckpt is None:
            self.next_ckpt = (self.num_timesteps // self.ckpt_every + 1) * self.ckpt_every
        if self.num_timesteps >= self.next_ckpt:
            self.model.save(self.ckpt_path)
            self.next_ckpt += self.ckpt_every
            print(f"checkpoint at {self.num_timesteps} steps", flush=True)
        if not self.buf_r:
            return
        row = dict(timesteps=int(self.num_timesteps), ep_return_mean=float(np.mean(self.buf_r)),
                   ep_len_mean=float(np.mean(self.buf_l)), fail_rate=float(np.mean(self.buf_fail)),
                   n_episodes=len(self.buf_r), wall_s=time.time() - self.t0)
        for k in ["train/value_loss", "train/policy_gradient_loss", "train/entropy_loss", "train/approx_kl", "train/explained_variance"]:
            v = self.logger.name_to_value.get(k)
            if v is not None:
                row[k.split("/")[1]] = float(v)
        self.rows.append(row)
        print(f"{row['timesteps']:>9d}  return {row['ep_return_mean']:8.2f}  len {row['ep_len_mean']:6.1f}  "
              f"fail {row['fail_rate']:.2f}  ({row['wall_s']:.0f}s)", flush=True)
        self.buf_r, self.buf_l, self.buf_fail = [], [], []
        self.out_path.write_text(json.dumps(self.rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--channel", default="A_us")
    ap.add_argument("--resume", action="store_true", help="continue from the periodic checkpoint and curve")
    args = ap.parse_args()
    torch.manual_seed(C.SEEDS["rl"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print(f"env device (acoustic render + features): {dev_name}; policy device: cpu", flush=True)
    env = BB8VecEnv(args.n_envs, C.SEEDS["rl"], channel=args.channel, device=dev)
    curve = C.RESULTS / "train_rl_curve.json"
    ckpt = C.DATA_ROOT / "runs" / "rl_ppo_periodic.zip"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rows, wall0, resumes = None, 0.0, 0
    if args.resume:
        model = PPO.load(ckpt, env=env, device="cpu")
        rows = [r for r in json.loads(curve.read_text()) if r["timesteps"] <= model.num_timesteps]
        wall0 = rows[-1]["wall_s"] if rows else 0.0
        resumes = 1
        print(f"resumed at {model.num_timesteps} steps", flush=True)
    else:
        model = PPO("MlpPolicy", env, n_steps=256, batch_size=1024, n_epochs=8, learning_rate=3e-4, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.7),
                    seed=C.SEEDS["rl"], device="cpu", verbose=0)
    cb = CurveLogger(curve, ckpt, rows=rows, wall0=wall0)
    t0 = time.time()
    model.learn(total_timesteps=args.steps - model.num_timesteps if args.resume else args.steps,
                callback=cb, reset_num_timesteps=not args.resume)
    C.CHECKPOINTS.mkdir(exist_ok=True)
    model.save(C.CHECKPOINTS / "rl_ppo.zip")
    meta = dict(algo="PPO (stable-baselines3)", total_timesteps=int(model.num_timesteps), requested=args.steps,
                n_envs=args.n_envs, episode_s=30.0, channel=args.channel, seconds=time.time() - t0 + wall0,
                resumed_from_checkpoint=bool(resumes),
                env_device=dev_name, policy_device="cpu", seed=C.SEEDS["rl"],
                hyper=dict(n_steps=256, batch_size=1024, n_epochs=8, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                           clip=0.2, net=[256, 256], log_std_init=-0.7))
    (C.RESULTS / "train_rl.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
