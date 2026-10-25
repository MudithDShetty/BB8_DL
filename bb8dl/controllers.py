"""Localizers (oracle, classical, deep) and following controllers.

Every controller maps the world observation dict to wheel duty u = (u_L, u_R)
in [-1, 1] for a batch of n episodes.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config as C
from .features import Classical, STFTFeatures
from .model import Localizer
from .robot import mix
from .tracker import Tracker


# ---------------------------------------------------------------- localizers
class OracleLoc:
    name = "oracle"

    def __call__(self, obs):
        n = len(obs["true_az"])
        return obs["true_az"], obs["true_range"], np.full(n, 0.01), np.full(n, 0.01)


class GCCLoc:
    """SRP-PHAT azimuth + matched-filter range. Uncertainty from SRP peak height (heuristic)."""
    name = "gcc"

    def __init__(self, device):
        self.cl = Classical(device)

    def __call__(self, obs):
        out = self.cl(obs["audio"], obs["chirp_idx"])
        pk = out["srp_peak"].clamp(0, 1).cpu().numpy()
        s_az = np.deg2rad(3.0 + 40.0 * (1 - pk))
        s_r = 0.15 + 1.0 * (1 - pk)
        return out["az_srp"].cpu().numpy(), out["range"].cpu().numpy(), s_az, s_r


class DLLoc:
    name = "dl"

    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.feat = STFTFeatures(device, ck["mode"], tuple(ck["stats"]))
        self.net = Localizer(in_ch=ck["in_ch"]).to(device)
        self.net.load_state_dict({k: v.float() for k, v in ck["state"].items()})
        self.net.eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, obs):
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            o = self.net(self.feat(obs["audio"], obs["chirp_idx"]))
        az = o["az"].float().cpu().numpy()
        s_az = o["s_az"].float().cpu().numpy()
        r = o["range"].float().cpu().numpy()
        s_r = np.exp(0.5 * o["logvar"].float().cpu().numpy())
        return az, r, s_az, np.maximum(s_r, 0.03)


# ---------------------------------------------------------------- controllers
class TrackPursuit:
    """Localizer -> Kalman tracker -> proportional pursuit law -> nominal mixer."""

    def __init__(self, loc, name=None, k_b=2.0, k_r=1.2):
        self.loc, self.k_b, self.k_r = loc, k_b, k_r
        self.name = name or f"{loc.name}_track"

    def reset(self, n):
        self.trk = Tracker(n)
        self.last_meas = None

    def localize(self, obs):
        az, r, s_az, s_r = self.loc(obs)
        self.last_meas = (az, r)
        self.trk.predict()
        self.trk.update(obs["odom"], az, r, s_az, s_r)
        return self.trk.relative(obs["odom"])

    def person_radial_speed(self, obs):
        """Tracked person velocity projected on the line of sight (>= 0): speed feedforward."""
        d = self.trk.x[:, :2] - obs["odom"][:, :2]
        u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-3)
        return np.clip((self.trk.x[:, 2:] * u).sum(1), 0.0, 1.0)

    def setpoints(self, b, r, obs):
        w = np.clip(self.k_b * b, -C.W_NOM_MAX, C.W_NOM_MAX)
        v = self.k_r * (r - C.STANDOFF) + self.person_radial_speed(obs) * (r > C.STANDOFF)
        v = np.clip(v, -0.2, 1.2) * np.clip(np.cos(b), 0, 1) ** 2
        return v, w

    def act(self, obs):
        b, r = self.localize(obs)
        v, w = self.setpoints(b, r, obs)
        return mix(v, w)


class PIDFollower(TrackPursuit):
    """Cascaded PID: outer PID on (bearing, range) -> (v*, w*); inner PI on measured (v, yaw rate).

    With `residual`, a learned terrain-aware inverse-dynamics model supplies the
    feedforward duty instead of the flat-ground mixer.
    """

    def __init__(self, loc, residual=None, name=None):
        super().__init__(loc, name or (f"{loc.name}_pid_res" if residual is not None else f"{loc.name}_pid"))
        self.res = residual
        self.g = dict(kp_b=2.2, ki_b=0.15, kd_b=0.12, kp_r=0.9, ki_r=0.06, kd_r=0.25,
                      kp_v=0.5, ki_v=0.8, kp_w=0.12, ki_w=0.25)

    def reset(self, n):
        super().reset(n)
        self.ib = np.zeros(n)
        self.ir = np.zeros(n)
        self.iv = np.zeros(n)
        self.iw = np.zeros(n)
        self.pb = None
        self.pr = None
        self.hist = np.zeros((n, 10, 7))
        self.u_prev = np.zeros((n, 2))

    def act(self, obs):
        g, dt = self.g, C.CTRL_DT
        b, r = self.localize(obs)
        er = r - C.STANDOFF
        db = np.zeros_like(b) if self.pb is None else np.mod(b - self.pb + np.pi, 2 * np.pi) - np.pi
        dr = np.zeros_like(r) if self.pr is None else r - self.pr
        self.pb, self.pr = b, r
        self.ib = np.clip(self.ib + b * dt, -0.5, 0.5)
        self.ir = np.clip(self.ir + er * dt, -1.0, 1.0)
        w = np.clip(g["kp_b"] * b + g["ki_b"] * self.ib + g["kd_b"] * db / dt, -C.W_NOM_MAX, C.W_NOM_MAX)
        v = g["kp_r"] * er + g["ki_r"] * self.ir + g["kd_r"] * dr / dt + self.person_radial_speed(obs) * (r > C.STANDOFF)
        v = np.clip(v, -0.2, 1.2) * np.clip(np.cos(b), 0, 1) ** 2

        pr = obs["proprio"]
        ev, ew = v - pr[:, 0], w - pr[:, 1]
        self.iv = np.clip(self.iv + ev * dt, -0.6, 0.6)
        self.iw = np.clip(self.iw + ew * dt, -1.0, 1.0)
        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = np.concatenate([self.u_prev, pr], 1)
        ff = mix(v, w)
        if self.res is not None:
            ff = ff + self.res.residual(self.hist, v, w)
        fb_fwd = g["kp_v"] * ev + g["ki_v"] * self.iv
        fb_turn = g["kp_w"] * ew + g["ki_w"] * self.iw
        u = np.clip(ff + np.stack([fb_fwd - fb_turn, fb_fwd + fb_turn], 1), -1, 1)
        self.u_prev = u
        return u
