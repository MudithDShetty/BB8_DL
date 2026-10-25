"""Learned terrain residual for the PID+DL controller.

What it learns, exactly: a supervised inverse-dynamics map

    u_t = f(h_t, v_target, w_target)

h_t holds the last 10 control steps (0.5 s) of applied wheel duty and noisy
proprioception (speed, yaw rate, pitch, pitch rate, roll). (v_target, w_target)
are the speed and yaw rate measured K=4 steps (0.2 s) after u_t starts being
held. Training data are random piecewise-constant commands on all six terrains.
At run time the PID's setpoints are fed as targets and the residual is
f(h, v*, w*) - mix(v*, w*), i.e. how much the flat-ground mixer must be corrected
on the terrain the recent history implies (more duty uphill or on grass, less
downhill). It does not see the terrain label or any acoustic input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .robot import mix

HIST = 10
K_AHEAD = 4


class ResidualNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIST * 7 + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def make_input(hist: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    h = hist.copy()
    h[..., 2] /= C.V_NOM_MAX
    h[..., 3] /= C.W_NOM_MAX
    return np.concatenate([h.reshape(len(h), -1), (v / C.V_NOM_MAX)[:, None], (w / C.W_NOM_MAX)[:, None]], 1).astype(np.float32)


class Residual:
    def __init__(self, path, device="cpu", clip: float = 0.5):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.net = ResidualNet(ck.get("hidden", 128))
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.clip = clip

    @torch.no_grad()
    def residual(self, hist, v, w):
        u = self.net(torch.from_numpy(make_input(hist, v, w))).numpy()
        return np.clip(u - mix(v, w), -self.clip, self.clip)
