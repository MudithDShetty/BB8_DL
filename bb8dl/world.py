"""Terrain scenarios, scripted person motion and the batched closed-loop world.

A scenario fixes everything that is not the controller: terrain parameters,
obstacles/room, the person's whole trajectory, acoustic nuisance parameters and
the robot start pose. The same (terrain, seed) gives the same scenario for every
controller, so controllers are compared on identical episodes.
"""

from __future__ import annotations

import numpy as np
import torch

from . import acoustics as A
from . import config as C
from . import robot as Rb

N_ROUGH = 6


def _terrain_surface(rng, name: str, n: int) -> dict:
    base = dict(
        flat=(1.00, 0.020, 0.5, 0.12, 0.0),
        slope=(1.00, 0.025, 0.5, 0.12, 0.0),
        rough=(0.90, 0.040, 0.8, 0.15, 2.0),
        grass=(0.80, 0.080, 3.0, 0.30, 0.8),
        indoor=(1.00, 0.015, 0.3, 0.10, 0.0),
        obstacles=(0.95, 0.025, 0.6, 0.13, 0.0),
    )[name]
    j = lambda v: v * rng.uniform(0.85, 1.15, n)  # noqa: E731  domain randomisation +-15 %
    eta, crr, cd, cyaw, bump = base
    return dict(eta=np.clip(j(eta), 0.5, 1.0), crr=j(crr), cd=j(cd), cyaw=j(cyaw), bump=j(bump))


def _person_track(rng, T, start, heading, room=None, obstacles=None):
    """Scripted walk: piecewise speed/turn-rate targets, pauses, wall/obstacle avoidance."""
    p = np.zeros((T + 1, 2))
    p[0] = start
    h, sp, tgt_sp, turn = heading, 0.0, rng.uniform(0.25, 0.85), 0.0
    t_next = 0
    dt = C.CTRL_DT
    for t in range(T):
        if t >= t_next:
            tgt_sp = 0.0 if rng.random() < 0.15 else rng.uniform(0.25, 0.85)
            turn = rng.uniform(-0.5, 0.5)
            t_next = t + int(rng.uniform(2.0, 6.0) / dt)
        sp += (tgt_sp - sp) * dt / 1.0
        h += turn * dt + rng.normal(0, 0.05)
        nxt = p[t] + sp * dt * np.array([np.cos(h), np.sin(h)])
        if room is not None:
            Lx, Ly = room
            if not (0.8 < nxt[0] < Lx - 0.8 and 0.8 < nxt[1] < Ly - 0.8):
                ctr = np.array([Lx / 2, Ly / 2]) - p[t]
                h = np.arctan2(ctr[1], ctr[0]) + rng.uniform(-0.6, 0.6)
                nxt = p[t] + sp * dt * np.array([np.cos(h), np.sin(h)])
        if obstacles is not None:
            for ox, oy, r, _ in obstacles:
                if r <= 0:
                    continue
                d = nxt - np.array([ox, oy])
                if np.hypot(*d) < r + 0.5:
                    # sidestep tangentially around the obstacle
                    tang = np.array([-d[1], d[0]])
                    if tang @ np.array([np.cos(h), np.sin(h)]) < 0:
                        tang = -tang
                    h = np.arctan2(tang[1], tang[0])
                    nxt = p[t] + sp * dt * np.array([np.cos(h), np.sin(h)])
        p[t + 1] = nxt
    return p


def make_scenarios(terrain: str | list, n: int, seed: int, channel: str = "A_us", T: int | None = None) -> dict:
    """n scenarios. `terrain` may be a name or a list of n names (mixed batches for RL)."""
    rng = np.random.default_rng(seed)
    T = T or int(C.EPISODE_S / C.CTRL_DT)
    names = [terrain] * n if isinstance(terrain, str) else list(terrain)
    tidx = np.array([C.TERRAINS.index(t) for t in names])
    ch = np.full(n, C.CHANNEL_NAMES.index(channel))
    sc = dict(terrain=tidx, seed=np.full(n, seed))
    surf = {k: np.zeros(n) for k in ["eta", "crr", "cd", "cyaw", "bump"]}
    for nm in set(names):
        m = np.array([t == nm for t in names])
        s = _terrain_surface(rng, nm, m.sum())
        for k in surf:
            surf[k][m] = s[k]
    sc.update(surf)

    slope_g = np.zeros((n, 2))
    rA, rk, rph = np.zeros((n, N_ROUGH)), np.zeros((n, N_ROUGH, 2)), np.zeros((n, N_ROUGH))
    room = np.zeros((n, 6))
    obstacles = np.zeros((n, A.K_OBS, 4))
    robot0 = np.zeros((n, 3))
    person = np.zeros((n, T + 1, 2))
    for i, nm in enumerate(names):
        robot0[i, 2] = rng.uniform(-np.pi, np.pi)
        if nm == "slope":
            g = np.tan(np.deg2rad(rng.uniform(5, 10)))
            b = rng.uniform(-np.pi, np.pi)
            slope_g[i] = g * np.array([np.cos(b), np.sin(b)])
        if nm == "rough":
            lam = rng.uniform(1.5, 5.0, N_ROUGH)
            kmag = 2 * np.pi / lam
            ang = rng.uniform(-np.pi, np.pi, N_ROUGH)
            rk[i] = kmag[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
            # amplitude so that each component contributes at most ~0.04 slope
            rA[i] = rng.uniform(0.01, 0.04, N_ROUGH) / kmag
            rph[i] = rng.uniform(0, 2 * np.pi, N_ROUGH)
        rm = None
        if nm == "indoor":
            Lx, Ly = rng.uniform(6, 10), rng.uniform(6, 10)
            room[i] = [Lx, Ly, rng.uniform(2.5, 3.2), rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9)]
            robot0[i, :2] = rng.uniform(1.0, Lx - 1.0), rng.uniform(1.0, Ly - 1.0)
            rm = (Lx, Ly)
        for _ in range(500):
            r0, b0 = rng.uniform(2.0, 4.0), rng.uniform(-np.pi, np.pi)
            pstart = robot0[i, :2] + r0 * np.array([np.cos(robot0[i, 2] + b0), np.sin(robot0[i, 2] + b0)])
            if rm is None or (0.9 < pstart[0] < rm[0] - 0.9 and 0.9 < pstart[1] < rm[1] - 0.9):
                break
        obs_i = None
        if nm == "obstacles":
            k = 0
            while k < A.K_OBS:
                c = robot0[i, :2] + rng.uniform(-7, 7, 2)
                r = rng.uniform(0.2, 0.5)
                if np.hypot(*(c - robot0[i, :2])) > r + 1.0 and np.hypot(*(c - pstart)) > r + 1.0:
                    obstacles[i, k] = [c[0], c[1], r, rng.uniform(0.5, 1.0)]
                    k += 1
            obs_i = obstacles[i]
        person[i] = _person_track(rng, T, pstart, rng.uniform(-np.pi, np.pi), rm, obs_i)

    p = A.sample_scene_params(rng, n, tidx, ch)
    indoor = tidx == C.TERRAINS.index("indoor")
    sc.update(
        slope_g=slope_g, rough_A=rA, rough_k=rk, rough_ph=rph, room=room, obstacles=obstacles,
        occl_db=rng.uniform(8, 20, (n, A.K_OBS)), robot0=robot0, person=person,
        phone_h=rng.uniform(0.9, 1.3, n), tail_amp=np.where(indoor, rng.uniform(0.1, 0.35, n), 0.0),
        wall=A.random_outdoor_wall(rng, robot0[:, :2], person[:, 0], np.where(indoor, 0.0, 0.3)),
        channel=ch, **p,
    )
    return sc


def splice(dst: dict, i: int, src: dict, j: int = 0) -> None:
    for k, v in src.items():
        dst[k][i] = v[j]


class World:
    """n parallel episodes. One `step` = one control period (one chirp)."""

    def __init__(self, sc: dict, renderer: A.Renderer, seed: int = 0):
        self.sc = sc
        self.ren = renderer
        self.n = len(sc["terrain"])
        self.rng = np.random.default_rng(seed)
        self.gen = None
        self.s = Rb.RobotState.zeros(self.n)
        self.k = np.zeros(self.n, int)
        self.odom = np.zeros((self.n, 3))
        self.reset_idx(np.ones(self.n, bool))

    # ---------------------------------------------------------------- terrain
    def terr(self) -> dict:
        sc = self.sc

        def grad(x, y):
            gx = sc["slope_g"][:, 0].copy()
            gy = sc["slope_g"][:, 1].copy()
            arg = sc["rough_k"][..., 0] * x[:, None] + sc["rough_k"][..., 1] * y[:, None] + sc["rough_ph"]
            c = sc["rough_A"] * np.cos(arg)
            gx += (c * sc["rough_k"][..., 0]).sum(1)
            gy += (c * sc["rough_k"][..., 1]).sum(1)
            return gx, gy

        return dict(eta=sc["eta"], crr=sc["crr"], cd=sc["cd"], cyaw=sc["cyaw"], bump=sc["bump"],
                    grad=grad, obstacles=sc["obstacles"], room=sc["room"][:, :2])

    def reset_idx(self, mask: np.ndarray) -> None:
        for f in ["x", "y", "psi", "psi_d", "th_d", "phi", "phi_d", "thl_d", "phil", "phil_d"]:
            getattr(self.s, f)[mask] = 0.0
        self.s.x[mask] = self.sc["robot0"][mask, 0]
        self.s.y[mask] = self.sc["robot0"][mask, 1]
        self.s.psi[mask] = self.sc["robot0"][mask, 2]
        self.k[mask] = 0
        self.odom[mask] = self.sc["robot0"][mask]

    # ---------------------------------------------------------------- geometry
    def person_xy(self) -> np.ndarray:
        T = self.sc["person"].shape[1] - 1
        return self.sc["person"][np.arange(self.n), np.minimum(self.k, T)]

    def truth(self) -> tuple[np.ndarray, np.ndarray]:
        """True bearing (rad, head frame) and horizontal range (m) to the phone."""
        d = self.person_xy() - np.stack([self.s.x, self.s.y], 1)
        az = np.arctan2(d[:, 1], d[:, 0]) - self.s.psi
        return np.mod(az + np.pi, 2 * np.pi) - np.pi, np.hypot(d[:, 0], d[:, 1])

    def scene(self) -> A.Scene:
        sc = self.sc
        pxy = self.person_xy()
        t = self.k * C.CTRL_DT
        h = sc["phone_h"] + 0.03 * np.sin(2 * np.pi * 0.9 * t)
        src = np.concatenate([pxy, h[:, None]], 1)
        head = np.stack([self.s.x, self.s.y], 1)
        # head mast assumed partially levelled: residual tilt is a fraction of the cart lean
        mics = A.head_mics(head, self.s.psi, -C.HEAD_TILT_FRAC * self.s.phi, C.HEAD_TILT_FRAC * self.s.phil)
        if (sc["room"][:, 0] > 0).any():
            # keep the phone strictly inside the room for the image model
            Lx, Ly = sc["room"][:, 0], sc["room"][:, 1]
            inside = sc["room"][:, 0] > 0
            src[inside, 0] = np.clip(src[inside, 0], 0.05, Lx[inside] - 0.05)
            src[inside, 1] = np.clip(src[inside, 1], 0.05, Ly[inside] - 0.05)
        return A.Scene(
            src=src, mics=mics, ground_coef=sc["ground_coef"], wall=sc["wall"], room=sc["room"],
            obstacles=sc["obstacles"], occl_db=sc["occl_db"], tail_amp=sc["tail_amp"],
            excess_db_m=sc["excess_db_m"], temp_c=sc["temp_c"],
            sync=sc["sync_bias"] + self.rng.normal(0, C.SYNC_JITTER_S, self.n),
            chirp_idx=sc["chirp_idx"], real_fe=sc["real_fe"], spk_fc=sc["spk_fc"],
            mic_gain_db=sc["mic_gain_db"], mic_delay=sc["mic_delay"], snr1m_db=sc["snr1m_db"],
            gain_db=sc["gain_db"],
        )

    def proprio(self) -> np.ndarray:
        """Noisy onboard sensing: wheel-odometry speed, gyro yaw rate, IMU pitch, pitch rate, roll."""
        s, r = self.s, self.rng
        return np.stack([
            s.v + r.normal(0, 0.02, self.n),
            s.psi_d + r.normal(0, 0.02, self.n),
            s.phi + r.normal(0, 0.01, self.n),
            s.phi_d + r.normal(0, 0.05, self.n),
            s.phil + r.normal(0, 0.01, self.n),
        ], 1)

    def observe(self) -> dict:
        y, lab = self.ren.render(self.scene(), self.rng, self.gen)
        az, rg = self.truth()
        return dict(audio=y, chirp_idx=self.sc["chirp_idx"], proprio=self.proprio(), odom=self.odom.copy(),
                    true_az=az, true_range=rg, snr_db=lab["snr_db"], los=lab["los"])

    def step(self, u: np.ndarray) -> dict:
        terr = self.terr()
        tumble = np.zeros(self.n, bool)
        coll = np.zeros(self.n, bool)
        for _ in range(C.SUBSTEPS):
            info = Rb.step(self.s, u, terr, self.rng)
            tumble |= info["tumble"]
            coll |= info["collided"]
        pr = self.proprio()
        # dead reckoning from wheel odometry + gyro (does not see lateral drift)
        self.odom[:, 2] = np.mod(self.odom[:, 2] + pr[:, 1] * C.CTRL_DT + np.pi, 2 * np.pi) - np.pi
        self.odom[:, 0] += pr[:, 0] * np.cos(self.odom[:, 2]) * C.CTRL_DT
        self.odom[:, 1] += pr[:, 0] * np.sin(self.odom[:, 2]) * C.CTRL_DT
        self.k += 1
        return dict(tumble=tumble, collided=coll)
