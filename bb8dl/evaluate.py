"""Closed-loop episode runner and per-episode following metrics."""

from __future__ import annotations

import copy
import time

import numpy as np

from . import config as C
from .world import World


def run_episodes(ctrl, scen: dict, renderer, seed: int, T: int | None = None) -> dict:
    """Run all episodes in `scen` in parallel with controller `ctrl`. Returns logs."""
    w = World(copy.deepcopy(scen), renderer, seed)
    n = w.n
    T = T or scen["person"].shape[1] - 1
    ctrl.reset(n)
    L = dict(robot=np.zeros((n, T + 1, 2)), person=np.zeros((n, T + 1, 2)), dist=np.zeros((n, T + 1)),
             u=np.zeros((n, T, 2)), coll=np.zeros((n, T), bool), tumble=np.zeros((n, T), bool),
             est_az=np.full((n, T), np.nan), est_r=np.full((n, T), np.nan),
             true_az=np.zeros((n, T)), true_r=np.zeros((n, T)), snr=np.zeros((n, T)),
             speed=np.zeros((n, T)), pitch=np.zeros((n, T)))
    t0 = time.time()
    for t in range(T):
        L["robot"][:, t] = np.stack([w.s.x, w.s.y], 1)
        L["person"][:, t] = w.person_xy()
        obs = w.observe()
        L["dist"][:, t] = obs["true_range"]
        L["true_az"][:, t], L["true_r"][:, t], L["snr"][:, t] = obs["true_az"], obs["true_range"], obs["snr_db"]
        u = ctrl.act(obs)
        if getattr(ctrl, "last_meas", None) is not None:
            L["est_az"][:, t], L["est_r"][:, t] = ctrl.last_meas
        info = w.step(u)
        L["u"][:, t] = u
        L["coll"][:, t], L["tumble"][:, t] = info["collided"], info["tumble"]
        L["speed"][:, t], L["pitch"][:, t] = w.s.v, w.s.phi
    L["robot"][:, T] = np.stack([w.s.x, w.s.y], 1)
    L["person"][:, T] = w.person_xy()
    L["dist"][:, T] = np.hypot(*(L["person"][:, T] - L["robot"][:, T]).T)
    L["wall_seconds"] = time.time() - t0
    return L


def episode_metrics(L: dict) -> dict:
    """Per-episode metrics arrays (n,)."""
    dt = C.CTRL_DT
    w0 = int(C.WARMUP_S / dt)
    d = L["dist"]
    within = (d[:, w0:] <= C.FOLLOW_RADIUS).mean(1)
    tumbles = L["tumble"].sum(1)
    reach = np.argmax(d <= C.FOLLOW_RADIUS, axis=1).astype(float)
    reach[~(d <= C.FOLLOW_RADIUS).any(1)] = np.nan
    steps = np.diff(L["robot"], axis=1)
    ae = np.rad2deg(np.abs(np.mod(L["est_az"] - L["true_az"] + np.pi, 2 * np.pi) - np.pi))
    re = np.abs(L["est_r"] - L["true_r"])
    return dict(
        success=((within >= C.SUCCESS_FRAC) & (tumbles == 0)).astype(float),
        frac_within=within,
        mean_dist=d[:, w0:].mean(1),
        final_dist=d[:, -1],
        time_to_reach=reach * dt,
        path_length=np.linalg.norm(steps, axis=-1).sum(1),
        person_path=np.linalg.norm(np.diff(L["person"], axis=1), axis=-1).sum(1),
        effort=(L["u"] ** 2).mean((1, 2)),
        collisions=L["coll"].sum(1).astype(float),
        tumbles=tumbles.astype(float),
        too_close=(d < 0.4).mean(1),
        loc_bearing_mae=np.nanmean(ae, 1) if np.isfinite(ae).any() else np.full(len(d), np.nan),
        loc_range_mae=np.nanmean(re, 1) if np.isfinite(re).any() else np.full(len(d), np.nan),
    )


def aggregate(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        v = np.asarray(v, float)
        ok = v[np.isfinite(v)]
        if len(ok) == 0:
            out[k] = dict(mean=None, std=None, ci95=None, n=0)
            continue
        out[k] = dict(mean=float(ok.mean()), std=float(ok.std()),
                      ci95=float(1.96 * ok.std(ddof=1) / np.sqrt(len(ok))) if len(ok) > 1 else 0.0, n=int(len(ok)))
    return out
