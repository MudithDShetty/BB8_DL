"""Batched planar dynamics of a sphere driven by an internal two-wheel cart.

Model (per env, all arrays shape (n,)):

Longitudinal: shell roll angle theta (v = R theta'), cart pendulum angle phi
(positive = cart swung forward, measured from true vertical). Lagrangian with
no-slip rolling of the shell on the ground:

    [M11 M12] [theta'']   [ tau + m_p R l sin(phi) phi'^2 - R F_res ]
    [M12 M22] [phi''  ] = [ tau - m_p g l sin(phi) - c_phi phi'      ]

    M11 = (m_s + m_p) R^2 + I_s,  M12 = m_p R l cos(phi),  M22 = m_p l^2 + I_p

tau = eta (R / r_w) (tau_L + tau_R) is the torque the cart wheels apply to the
inner shell (and, reversed, to the cart). F_res = rolling resistance + viscous
drag + gravity along the local slope + random bumps.

Lateral: same equations with tau = 0 (the cart cannot push sideways, it only
hangs). This makes the ball drift downhill on a side slope.

Yaw: the wheel speed difference turns the cart (and the head) about the
vertical; spin of the shell against the ground is lumped into c_yaw.

Motors: DC torque-speed line tau_i = tau_stall (u_i - omega_i / omega_0), with
u_i in [-1, 1] the PWM duty sent by the ESP32, clipped to 1.5 tau_stall.

Simplifications (stated in the paper): no shell-ground slip, the cart never
slips on the shell, pitch and roll decoupled, head/array assumed rigidly
aligned with the cart heading, slope modelled only as a gravity term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

P = C.ROBOT


@dataclass
class RobotState:
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    psi_d: np.ndarray
    th_d: np.ndarray
    phi: np.ndarray
    phi_d: np.ndarray
    thl_d: np.ndarray
    phil: np.ndarray
    phil_d: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "RobotState":
        return cls(*[np.zeros(n) for _ in range(10)])

    @property
    def v(self) -> np.ndarray:
        return P.R * self.th_d

    @property
    def v_lat(self) -> np.ndarray:
        return P.R * self.thl_d


def _pendulum_step(th_d, phi, phi_d, tau, F_res, dt):
    M11 = P.m_tot * P.R**2 + P.I_s
    M12 = P.m_p * P.R * P.l * np.cos(phi)
    M22 = P.m_p * P.l**2 + P.I_p
    Q1 = tau + P.m_p * P.R * P.l * np.sin(phi) * phi_d**2 - P.R * F_res
    Q2 = tau - P.m_p * P.g * P.l * np.sin(phi) - P.c_phi * phi_d
    det = M11 * M22 - M12**2
    th_dd = (M22 * Q1 - M12 * Q2) / det
    phi_dd = (M11 * Q2 - M12 * Q1) / det
    th_d = th_d + th_dd * dt
    phi_d = phi_d + phi_dd * dt
    phi = phi + phi_d * dt
    return th_d, phi, phi_d


def step(s: RobotState, u: np.ndarray, terr: dict, rng: np.random.Generator, dt: float = C.PHYS_DT) -> dict:
    """Advance one physics step in place. u: (n,2) = (u_L, u_R) in [-1,1]."""
    u = np.clip(u, -1.0, 1.0)
    N = P.m_tot * P.g
    w_w = P.R * (s.th_d + s.phi_d) / P.r_w
    dw = P.b * s.psi_d / (2 * P.r_w)
    w_L, w_R = w_w - dw, w_w + dw
    lim = 1.5 * P.tau_stall
    tL = np.clip(P.tau_stall * (u[:, 0] - w_L / P.w_noload), -lim, lim)
    tR = np.clip(P.tau_stall * (u[:, 1] - w_R / P.w_noload), -lim, lim)
    eta = terr["eta"]
    tau = eta * (P.R / P.r_w) * (tL + tR)

    gx, gy = terr["grad"](s.x, s.y)
    ch, sh = np.cos(s.psi), np.sin(s.psi)
    s_f = gx * ch + gy * sh
    s_l = -gx * sh + gy * ch
    sin_f = s_f / np.sqrt(1 + s_f**2)
    sin_l = s_l / np.sqrt(1 + s_l**2)
    bump = terr["bump"]
    F_f = (terr["crr"] * N * np.tanh(s.v / 0.05) + terr["cd"] * s.v + P.m_tot * P.g * sin_f
           + bump * rng.normal(size=s.x.shape))
    F_l = (terr["crr"] * N * np.tanh(s.v_lat / 0.05) + terr["cd"] * s.v_lat + P.m_tot * P.g * sin_l
           + bump * rng.normal(size=s.x.shape))

    s.th_d, s.phi, s.phi_d = _pendulum_step(s.th_d, s.phi, s.phi_d, tau, F_f, dt)
    s.thl_d, s.phil, s.phil_d = _pendulum_step(s.thl_d, s.phil, s.phil_d, np.zeros_like(tau), F_l, dt)

    yaw_tau = eta * P.b / (2 * P.r_w) * (tR - tL) - terr["cyaw"] * s.psi_d + 0.004 * bump * rng.normal(size=s.x.shape)
    s.psi_d = s.psi_d + yaw_tau / P.I_z * dt
    s.psi = np.mod(s.psi + s.psi_d * dt + np.pi, 2 * np.pi) - np.pi

    v, vl = s.v, s.v_lat
    ch, sh = np.cos(s.psi), np.sin(s.psi)
    s.x = s.x + (v * ch - vl * sh) * dt
    s.y = s.y + (v * sh + vl * ch) * dt

    tumble = np.zeros(s.x.shape, bool)
    for a, ad in (("phi", "phi_d"), ("phil", "phil_d")):
        ang = getattr(s, a)
        over = np.abs(ang) > P.phi_limit
        if over.any():
            setattr(s, a, np.clip(ang, -P.phi_limit, P.phi_limit))
            getattr(s, ad)[over] = 0.0
            tumble |= over
    collided = _collide(s, terr)
    return dict(tumble=tumble, collided=collided, wheel_tau=np.stack([tL, tR], 1))


def _collide(s: RobotState, terr: dict) -> np.ndarray:
    """Inelastic contact with cylindrical obstacles and room walls: remove the inward normal velocity."""
    hit = np.zeros(s.x.shape, bool)
    ch, sh = np.cos(s.psi), np.sin(s.psi)
    vx = s.v * ch - s.v_lat * sh
    vy = s.v * sh + s.v_lat * ch
    obs = terr["obstacles"]  # (n,K,4)
    if obs is not None and (obs[..., 2] > 0).any():
        dx = s.x[:, None] - obs[..., 0]
        dy = s.y[:, None] - obs[..., 1]
        d = np.hypot(dx, dy)
        rr = obs[..., 2] + P.R
        pen = (d < rr) & (obs[..., 2] > 0)
        if pen.any():
            for i, k in zip(*np.nonzero(pen)):
                nx, ny = dx[i, k] / max(d[i, k], 1e-6), dy[i, k] / max(d[i, k], 1e-6)
                s.x[i] = obs[i, k, 0] + nx * rr[i, k]
                s.y[i] = obs[i, k, 1] + ny * rr[i, k]
                vn = vx[i] * nx + vy[i] * ny
                if vn < 0:
                    vx[i] -= vn * nx
                    vy[i] -= vn * ny
                hit[i] = True
    room = terr["room"]  # (n,2) Lx, Ly ; 0 = no walls
    has = room[:, 0] > 0
    if has.any():
        for axis, (pos, vel, L) in enumerate(((s.x, vx, room[:, 0]), (s.y, vy, room[:, 1]))):
            lo = has & (pos < P.R)
            hi = has & (pos > L - P.R)
            pos[lo] = P.R
            pos[hi] = (L - P.R)[hi]
            vel[lo] = np.maximum(vel[lo], 0)
            vel[hi] = np.minimum(vel[hi], 0)
            hit |= lo | hi
    if hit.any():
        s.th_d = np.where(hit, (vx * ch + vy * sh) / P.R, s.th_d)
        s.thl_d = np.where(hit, (-vx * sh + vy * ch) / P.R, s.thl_d)
    return hit


def mix(v_cmd: np.ndarray, w_cmd: np.ndarray) -> np.ndarray:
    """Nominal flat-ground mixer from (v, omega) setpoints to wheel duty (u_L, u_R)."""
    a = v_cmd / C.V_NOM_MAX
    d = w_cmd / (2 * C.W_NOM_MAX)
    return np.clip(np.stack([a - d, a + d], 1), -1, 1)
