"""Batched constant-velocity Kalman tracker of the phone in the robot's odometry frame.

Measurements are (bearing, range) in the head frame with per-measurement
standard deviations; they are converted to Cartesian with the linearised
polar covariance. Mahalanobis gating rejects outliers (GCC fringe jumps,
echo locks); after `reinit_after` consecutive rejections the track restarts.
"""

from __future__ import annotations

import numpy as np

from . import config as C


class Tracker:
    def __init__(self, n: int, q_acc: float = 0.6, gate: float = 13.8, reinit_after: int = 10):
        self.n, self.q, self.gate, self.reinit_after = n, q_acc, gate, reinit_after
        self.x = np.zeros((n, 4))
        self.P = np.tile(np.eye(4) * 100.0, (n, 1, 1))
        self.init = np.zeros(n, bool)
        self.miss = np.zeros(n, int)

    def reset(self, mask=None):
        mask = np.ones(self.n, bool) if mask is None else mask
        self.init[mask] = False
        self.miss[mask] = 0

    def predict(self, dt: float = C.CTRL_DT):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q**2
        self.x = self.x @ F.T
        self.P = F @ self.P @ F.T + Q

    def update(self, odom: np.ndarray, az: np.ndarray, rng: np.ndarray, s_az: np.ndarray, s_r: np.ndarray):
        th = odom[:, 2] + az
        c, s = np.cos(th), np.sin(th)
        z = odom[:, :2] + rng[:, None] * np.stack([c, s], 1)
        J = np.stack([np.stack([c, -rng * s], 1), np.stack([s, rng * c], 1)], 1)  # d(x,y)/d(r,az)
        Rp = np.zeros((self.n, 2, 2))
        Rp[:, 0, 0] = s_r**2
        Rp[:, 1, 1] = s_az**2
        R = J @ Rp @ J.transpose(0, 2, 1) + np.eye(2) * 1e-4

        new = ~self.init
        if new.any():
            self.x[new, :2] = z[new]
            self.x[new, 2:] = 0.0
            self.P[new] = 0.0
            self.P[new, :2, :2] = R[new]
            self.P[new, 2, 2] = self.P[new, 3, 3] = 1.0
            self.init[new] = True

        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        y = z - self.x[:, :2]
        S = self.P[:, :2, :2] + R
        Si = np.linalg.inv(S)
        d2 = np.einsum("ni,nij,nj->n", y, Si, y)
        ok = (d2 < self.gate) & ~new
        K = self.P[:, :, :2] @ Si
        xn = self.x + np.einsum("nij,nj->ni", K, y)
        Pn = self.P - K @ H @ self.P
        self.x[ok], self.P[ok] = xn[ok], Pn[ok]
        self.miss = np.where(ok | new, 0, self.miss + 1)
        lost = self.miss >= self.reinit_after
        if lost.any():
            self.init[lost] = False
            self.miss[lost] = 0
        return ok

    def relative(self, odom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.x[:, :2] - odom[:, :2]
        b = np.arctan2(d[:, 1], d[:, 0]) - odom[:, 2]
        return np.mod(b + np.pi, 2 * np.pi) - np.pi, np.hypot(d[:, 0], d[:, 1])
