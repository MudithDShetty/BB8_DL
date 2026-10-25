"""RL environment: batched SB3 VecEnv plus a single-episode Gymnasium env.

Observation per frame (24): 17 classical acoustic features (6 GCC-PHAT pair lags,
6 pair peak heights, SRP-PHAT azimuth as cos/sin, SRP peak, matched-filter range,
log level), 5 proprioceptive values (speed, yaw rate, pitch, pitch rate, roll)
and the previous action. Four frames are stacked (0.2 s of memory).
Action: wheel duty (u_L, u_R) in [-1, 1].

Reward per 50 ms step (true distance d is privileged, used only for reward):
  +5.0 * (d_prev - d) while d_prev > standoff         progress toward the person
  -0.05 * max(d - standoff, 0)                         distance
  +0.1 if d <= follow radius                           following bonus
  -0.02 * yaw_rate^2                                   excessive spinning
  -0.05 * |u - u_prev|^2                               actuation jerk
  -0.5 if d < 0.4                                      too close to the person
  -1.0 on collision, -0.05 per step while stuck
  -10 and terminate on tumble (cart loops) or if d > 12 m (lost / rolled away)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from . import config as C
from .acoustics import Renderer
from .features import Classical
from .world import World, make_scenarios, splice

FRAME = 24
STACK = 4
OBS_DIM = FRAME * STACK
PROPRIO_SCALE = np.array([C.V_NOM_MAX, C.W_NOM_MAX, 0.5, 2.0, 0.5])


def frame_features(classical: Classical, obs: dict, u_prev: np.ndarray) -> np.ndarray:
    out = classical(obs["audio"], obs["chirp_idx"])
    ac = classical.rl_features(out).cpu().numpy()
    return np.concatenate([ac, obs["proprio"] / PROPRIO_SCALE, u_prev], 1).astype(np.float32)


class RewardState:
    def __init__(self, n):
        self.d_prev = np.zeros(n)
        self.stuck = np.zeros(n, int)

    def reset(self, mask, d):
        self.d_prev[mask] = d[mask]
        self.stuck[mask] = 0

    def __call__(self, d, u, u_prev, yaw_rate, speed, info):
        r = 5.0 * np.where(self.d_prev > C.STANDOFF, self.d_prev - d, 0.0)
        r -= 0.05 * np.maximum(d - C.STANDOFF, 0)
        r += 0.1 * (d <= C.FOLLOW_RADIUS)
        r -= 0.02 * yaw_rate**2
        r -= 0.05 * ((u - u_prev) ** 2).sum(1)
        r -= 0.5 * (d < 0.4)
        r -= 1.0 * info["collided"]
        self.stuck = np.where((np.abs(u).mean(1) > 0.4) & (np.abs(speed) < 0.05), self.stuck + 1, 0)
        r -= 0.05 * (self.stuck > 40)
        fail = info["tumble"] | (d > 12.0)
        r -= 10.0 * fail
        self.d_prev = d.copy()
        return r, fail


class BB8VecEnv(VecEnv):
    def __init__(self, n_envs=32, seed=C.SEEDS["rl"], terrains=None, channel="A_us", T=600, device="cuda"):
        self.terrains = terrains or C.TERRAINS
        self.channel, self.T = channel, T
        self.rng = np.random.default_rng(seed)
        self.next_seed = seed * 1000
        self.ren = Renderer(device)
        self.cl = Classical(device)
        names = [self.terrains[i % len(self.terrains)] for i in range(n_envs)]
        self.world = World(self._mixed(names), self.ren, seed)
        self.rs = RewardState(n_envs)
        self.u_prev = np.zeros((n_envs, 2))
        self.stack = np.zeros((n_envs, STACK, FRAME), np.float32)
        self.ep_ret = np.zeros(n_envs)
        self.ep_len = np.zeros(n_envs, int)
        obs_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        super().__init__(n_envs, obs_space, act_space)

    def _mixed(self, names):
        sc = None
        for i, nm in enumerate(names):
            one = make_scenarios(nm, 1, self._seed(), self.channel, self.T)
            if sc is None:
                sc = {k: np.repeat(v, len(names), axis=0) for k, v in one.items()}
            splice(sc, i, one)
        return sc

    def _seed(self):
        self.next_seed += 1
        return self.next_seed

    def _push(self):
        o = self.world.observe()
        f = frame_features(self.cl, o, self.u_prev)
        self.stack = np.roll(self.stack, -1, axis=1)
        self.stack[:, -1] = f
        return self.stack.reshape(self.num_envs, -1).copy()

    def _restart(self, mask):
        """Fresh observation stacks for envs in `mask` (after their world was reset)."""
        o = self.world.observe()
        f = frame_features(self.cl, o, self.u_prev)
        self.stack[mask] = np.repeat(f[mask][:, None, :], STACK, axis=1)
        self.rs.reset(mask, o["true_range"])
        return self.stack.reshape(self.num_envs, -1).copy()

    def reset(self):
        allm = np.ones(self.num_envs, bool)
        self.world.reset_idx(allm)
        self.u_prev[:] = 0
        self.ep_ret[:] = 0
        self.ep_len[:] = 0
        return self._restart(allm)

    def step_async(self, actions):
        self._a = np.clip(np.asarray(actions, dtype=np.float64), -1, 1)

    def step_wait(self):
        u = self._a
        info = self.world.step(u)
        _, d = self.world.truth()
        rew, fail = self.rs(d, u, self.u_prev, self.world.s.psi_d, self.world.s.v, info)
        self.u_prev = u.copy()
        trunc = self.world.k >= self.T
        done = fail | trunc
        self.ep_ret += rew
        self.ep_len += 1
        infos = [{} for _ in range(self.num_envs)]
        obs = self._push()
        if done.any():
            for i in np.nonzero(done)[0]:
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["TimeLimit.truncated"] = bool(trunc[i] and not fail[i])
                infos[i]["episode"] = {"r": float(self.ep_ret[i]), "l": int(self.ep_len[i])}
                nm = self.terrains[int(self.rng.integers(len(self.terrains)))]
                splice(self.world.sc, i, make_scenarios(nm, 1, self._seed(), self.channel, self.T))
            self.world.reset_idx(done)
            self.u_prev[done] = 0
            self.ep_ret[done] = 0
            self.ep_len[done] = 0
            obs = self._restart(done)
        return obs, rew.astype(np.float32), done, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [seed] * self.num_envs


class BB8FollowEnv(gym.Env):
    """Single-episode Gymnasium wrapper around the same world (for check_env and demos)."""

    metadata = {"render_modes": []}

    def __init__(self, terrain="flat", channel="A_us", T=600, device="cuda"):
        self.terrain, self.channel, self.T = terrain, channel, T
        self.ren = Renderer(device)
        self.cl = Classical(device)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = int(self.np_random.integers(1 << 30))
        self.world = World(make_scenarios(self.terrain, 1, s, self.channel, self.T), self.ren, s)
        self.rs = RewardState(1)
        self.u_prev = np.zeros((1, 2))
        o = self.world.observe()
        f = frame_features(self.cl, o, self.u_prev)
        self.stack = np.repeat(f[:, None], STACK, 1)
        self.rs.reset(np.ones(1, bool), o["true_range"])
        return self.stack.reshape(-1).copy(), {}

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float64)[None], -1, 1)
        info = self.world.step(u)
        _, d = self.world.truth()
        r, fail = self.rs(d, u, self.u_prev, self.world.s.psi_d, self.world.s.v, info)
        self.u_prev = u
        o = self.world.observe()
        f = frame_features(self.cl, o, self.u_prev)
        self.stack = np.roll(self.stack, -1, 1)
        self.stack[:, -1] = f
        return self.stack.reshape(-1).copy(), float(r[0]), bool(fail[0]), bool(self.world.k[0] >= self.T), {}


class RLFollower:
    """Closed-loop wrapper for a trained SB3 policy."""

    name = "rl"

    def __init__(self, model_path, device="cuda"):
        from stable_baselines3 import PPO
        self.policy = PPO.load(model_path, device="cpu")
        self.cl = Classical(device)

    def reset(self, n):
        self.u_prev = np.zeros((n, 2))
        self.stack = None
        self.last_meas = None

    def act(self, obs):
        f = frame_features(self.cl, obs, self.u_prev)
        if self.stack is None:
            self.stack = np.repeat(f[:, None], STACK, 1)
        else:
            self.stack = np.roll(self.stack, -1, 1)
            self.stack[:, -1] = f
        a, _ = self.policy.predict(self.stack.reshape(len(f), -1), deterministic=True)
        self.u_prev = np.clip(a.astype(np.float64), -1, 1)
        return self.u_prev
