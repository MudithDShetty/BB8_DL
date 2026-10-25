"""Beacon chirp, propagation paths and a batched fractional-delay renderer.

Signal model (per mic m, per path p):

    y_m(t) = sum_p a_pm * h_fe * s(t - d_pm / c - delta_sync - e_m) + n_m(t)

a_pm = rho_p * spread_pm * 10^(-(alpha_air + alpha_terrain) d_pm / 20), s is the
emitted linear chirp, h_fe the speaker/mic/decimation front-end, delta_sync the
residual emission-time error, e_m a per-mic delay mismatch. Delays are applied
exactly in the frequency domain (exp(-j 2 pi f tau)), so they are fractional.

Noise is white and added at the digital output. `snr1m_db` is the in-band SNR an
*ideal flat* front-end would give for the direct path at 1 m; a lossy front-end
therefore lowers the realised SNR by exactly its in-band attenuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import config as C

K_OBS = 8          # max obstacles per scene
N_TAIL = 24        # diffuse late-reverb paths (indoor only)

# Shoebox image-source options per axis: (offset in units of L, sign, n_reflections)
_AXIS_OPTS = [(0.0, 1.0, 0), (0.0, -1.0, 1), (2.0, -1.0, 1), (2.0, 1.0, 2), (-2.0, 1.0, 2)]
_ROOM_COMBOS = [
    (i, j, k)
    for i in range(5) for j in range(5) for k in range(5)
    if 0 < _AXIS_OPTS[i][2] + _AXIS_OPTS[j][2] + _AXIS_OPTS[k][2] <= 2
]
N_ROOM = len(_ROOM_COMBOS)  # 24
SLOT_DIRECT, SLOT_GROUND, SLOT_WALL = 0, 1, 2
SLOT_OBS = 3
SLOT_ROOM = SLOT_OBS + K_OBS
SLOT_TAIL = SLOT_ROOM + N_ROOM
N_PATHS = SLOT_TAIL + N_TAIL  # 59


# ---------------------------------------------------------------- chirp + front-end
def chirp(name: str, fs: int = C.FS, dur: float = C.CHIRP_DUR, taper: float = C.CHIRP_TAPER) -> np.ndarray:
    f0, f1 = C.CHIRPS[name]
    n = int(round(dur * fs))
    t = np.arange(n) / fs
    k = (f1 - f0) / dur
    x = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))
    # Tukey window
    w = np.ones(n)
    m = int(taper * n / 2)
    if m > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m], w[-m:] = ramp, ramp[::-1]
    return (x * w).astype(np.float64)


def frontend_gain(f: np.ndarray, kind: str, spk_fc: float | np.ndarray = 20_000.0) -> np.ndarray:
    """Magnitude response of the transmit+receive chain (zero phase)."""
    f = np.asarray(f, dtype=np.float64)
    if kind == "ideal":
        return np.ones_like(f) if np.ndim(spk_fc) == 0 else np.ones(np.shape(spk_fc) + f.shape)
    fc = np.asarray(spk_fc, dtype=np.float64)[..., None] if np.ndim(spk_fc) else spk_fc
    spk = 1.0 / np.sqrt(1.0 + (f / fc) ** 8)
    mic = 1.0 / np.sqrt(1.0 + (f / C.MIC_LP_FC) ** 4)
    x = np.clip((f - C.DECIM_PASS) / (C.DECIM_STOP - C.DECIM_PASS), 0.0, 1.0)
    dec = 10 ** (-C.DECIM_STOP_DB * x / 20)
    return spk * mic * dec


def inband_frontend_db(chirp_name: str, kind: str, spk_fc: float = 20_000.0) -> float:
    f0, f1 = C.CHIRPS[chirp_name]
    f = np.linspace(f0, f1, 256)
    g = frontend_gain(f, kind, spk_fc)
    return float(10 * np.log10(np.mean(g**2)))


def chirp_power(name: str) -> float:
    s = chirp(name)
    return float(np.mean(s**2))


def speed_of_sound(temp_c: np.ndarray) -> np.ndarray:
    return 331.3 + 0.606 * np.asarray(temp_c)


# ---------------------------------------------------------------- geometry
def head_mics(head_xy: np.ndarray, heading: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """(n,4,3) world mic positions for a head centred at (x, y, MIC_HEIGHT).

    Head frame: x forward, y left, z up. Tilt (pitch about y, roll about x) is
    applied before yaw. Pitch follows the internal cart lean.
    """
    n = head_xy.shape[0]
    local = np.zeros((4, 3))
    local[:, :2] = C.mic_xy()
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    x, y, z = local[:, 0][None], local[:, 1][None], local[:, 2][None]
    # roll about x
    y1 = y * cr[:, None] - z * sr[:, None]
    z1 = y * sr[:, None] + z * cr[:, None]
    # pitch about y (positive pitch tips the front down)
    x2 = x * cp[:, None] + z1 * sp[:, None]
    z2 = -x * sp[:, None] + z1 * cp[:, None]
    ch, sh = np.cos(heading)[:, None], np.sin(heading)[:, None]
    out = np.empty((n, 4, 3))
    out[..., 0] = head_xy[:, 0:1] + x2 * ch - y1 * sh
    out[..., 1] = head_xy[:, 1:2] + x2 * sh + y1 * ch
    out[..., 2] = C.MIC_HEIGHT + z2
    return out


def _seg_circle_hit(a: np.ndarray, b: np.ndarray, o: np.ndarray, r: np.ndarray) -> np.ndarray:
    """2-D segment a->b intersects circle (o, r). a,b: (n,2); o: (n,K,2); r: (n,K)."""
    ab = (b - a)[:, None, :]
    ao = o - a[:, None, :]
    t = np.clip((ao * ab).sum(-1) / np.maximum((ab**2).sum(-1), 1e-9), 0.0, 1.0)
    closest = a[:, None, :] + t[..., None] * ab
    return (np.linalg.norm(o - closest, axis=-1) < r) & (r > 0)


@dataclass
class Scene:
    """Per-example acoustic scene. All arrays have leading batch dim n."""
    src: np.ndarray          # (n,3) phone position
    mics: np.ndarray         # (n,4,3)
    ground_coef: np.ndarray  # (n,) 0 disables (indoor uses room floor instead)
    wall: np.ndarray         # (n,4) nx, ny, d, coef   plane n.x = d (outdoor facade)
    room: np.ndarray         # (n,6) Lx, Ly, H, beta_wall, beta_floor, beta_ceil; Lx=0 -> outdoor
    obstacles: np.ndarray    # (n,K_OBS,4) x, y, radius, scatter coef; radius 0 = unused
    occl_db: np.ndarray      # (n,K_OBS) direct-path loss if that obstacle blocks LOS
    tail_amp: np.ndarray     # (n,) diffuse tail strength (indoor)
    excess_db_m: np.ndarray  # (n,) terrain excess attenuation (grass)
    temp_c: np.ndarray       # (n,) true air temperature
    sync: np.ndarray         # (n,) emission-time error (s)
    chirp_idx: np.ndarray    # (n,) 0 -> "us", 1 -> "fb"
    real_fe: np.ndarray      # (n,) bool, realistic front-end
    spk_fc: np.ndarray       # (n,)
    mic_gain_db: np.ndarray  # (n,4)
    mic_delay: np.ndarray    # (n,4) s
    snr1m_db: np.ndarray     # (n,)
    gain_db: np.ndarray | None = None  # (n,) unknown overall level (phone volume, mic gain)

    @property
    def n(self) -> int:
        return self.src.shape[0]


def build_paths(sc: Scene, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-path per-mic distance (n,P,4), amplitude before air loss (n,P,4), LOS flag (n,)."""
    n = sc.n
    dist = np.full((n, N_PATHS, 4), 1.0)
    amp = np.zeros((n, N_PATHS, 4))
    mics = sc.mics
    src = sc.src
    centre = mics.mean(axis=1)

    def point_image(img: np.ndarray, coef: np.ndarray, slot: int) -> None:
        d = np.linalg.norm(img[:, None, :] - mics, axis=-1)
        dist[:, slot] = d
        amp[:, slot] = coef[:, None] / np.maximum(d, 0.05)

    # direct, with occlusion by obstacles
    blocked = _seg_circle_hit(src[:, :2], centre[:, :2], sc.obstacles[..., :2], sc.obstacles[..., 2])
    occl = np.where(blocked, sc.occl_db, 0.0).sum(-1)
    point_image(src, 10 ** (-occl / 20), SLOT_DIRECT)
    los = ~blocked.any(-1)

    ground_img = src * np.array([1.0, 1.0, -1.0])
    point_image(ground_img, sc.ground_coef, SLOT_GROUND)

    nx, ny, dw, cw = sc.wall.T
    nvec = np.stack([nx, ny, np.zeros(n)], -1)
    s_n = (src * nvec).sum(-1)
    wall_img = src + 2 * (dw - s_n)[:, None] * nvec
    point_image(wall_img, cw, SLOT_WALL)

    o = np.concatenate([sc.obstacles[..., :2], np.full((n, K_OBS, 1), 0.6)], -1)  # scatter at 0.6 m height
    d1 = np.linalg.norm(o - src[:, None, :], axis=-1)                         # (n,K)
    d2 = np.linalg.norm(o[:, :, None, :] - mics[:, None, :, :], axis=-1)      # (n,K,4)
    used = sc.obstacles[..., 2] > 0
    dist[:, SLOT_OBS:SLOT_OBS + K_OBS] = d1[..., None] + d2
    amp[:, SLOT_OBS:SLOT_OBS + K_OBS] = np.where(
        used[..., None],
        sc.obstacles[..., 3:4] * sc.obstacles[..., 2:3] / np.maximum(d1[..., None] * d2, 0.05),
        0.0,
    )

    indoor = sc.room[:, 0] > 0
    if indoor.any():
        Lx, Ly, H, bw, bf, bc = sc.room.T
        for q, (i, j, k) in enumerate(_ROOM_COMBOS):
            ox, sx, rx = _AXIS_OPTS[i]
            oy, sy, ry = _AXIS_OPTS[j]
            oz, sz, rz = _AXIS_OPTS[k]
            img = np.stack([ox * Lx + sx * src[:, 0], oy * Ly + sy * src[:, 1], oz * H + sz * src[:, 2]], -1)
            # z options: 1 mirrors the floor, 2 the ceiling, 3/4 hit both once
            zc = [np.ones(n), bf, bc, bf * bc, bf * bc][k]
            coef = bw ** (rx + ry) * zc
            point_image(img, np.where(indoor, coef, 0.0), SLOT_ROOM + q)
        # diffuse tail: random directions, exponentially decaying with extra path length
        d0 = np.linalg.norm(src - centre, axis=-1)
        u = rng.normal(size=(n, N_TAIL, 3))
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        extra = rng.uniform(3.0, 18.0, size=(n, N_TAIL))
        L = d0[:, None] + extra
        proj = ((mics - centre[:, None, :])[:, None, :, :] * u[:, :, None, :]).sum(-1)  # (n,T,4)
        dist[:, SLOT_TAIL:] = L[..., None] - proj
        a = sc.tail_amp[:, None] * np.exp(-extra / 6.0) * rng.uniform(0.3, 1.0, size=(n, N_TAIL))
        amp[:, SLOT_TAIL:] = np.where(indoor[:, None, None], (a / L)[..., None], 0.0)
    return dist, amp, los


# ---------------------------------------------------------------- renderer
class Renderer:
    """Batched GPU renderer. Holds chirp spectra and the synthesis band."""

    def __init__(self, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        N = C.N_FFT_SYN
        f = np.fft.rfftfreq(N, 1 / C.FS)
        self.k = np.nonzero((f >= C.SYN_BAND[0]) & (f <= C.SYN_BAND[1]))[0]
        self.f = f[self.k]
        spec = []
        for name in ["us", "fb"]:
            s = np.zeros(N)
            c = chirp(name)
            s[: len(c)] = c
            spec.append(np.fft.rfft(s)[self.k])
        self.S = torch.tensor(np.stack(spec), dtype=torch.complex64, device=self.device)  # (2,K)
        self.kt = torch.tensor(self.k, dtype=torch.float64, device=self.device)
        self.pc = {name: chirp_power(name) for name in ["us", "fb"]}

    def render(self, sc: Scene, rng: np.random.Generator, gen: torch.Generator | None = None,
               anechoic: bool = False) -> tuple[torch.Tensor, dict]:
        """Return audio (n,4,N_WIN) float32 on device and a dict of derived labels."""
        dist, amp, los = build_paths(sc, rng)
        if anechoic:
            amp[:, 1:] = 0.0
            amp[:, 0] = 1.0 / np.maximum(dist[:, 0], 0.05)
        c = speed_of_sound(sc.temp_c)
        air = 10 ** (-(C.AIR_ABS_DB_PER_M + sc.excess_db_m)[:, None, None] * dist / 20)
        a = amp * air * 10 ** (sc.mic_gain_db[:, None, :] / 20)
        tau = dist / c[:, None, None] + sc.sync[:, None, None] + sc.mic_delay[:, None, :]
        tau_s = torch.tensor(tau * C.FS, dtype=torch.float64, device=self.device)        # samples
        a_t = torch.tensor(a, dtype=torch.float32, device=self.device)
        frac = torch.remainder(tau_s[..., None] * self.kt / C.N_FFT_SYN, 1.0).float()  # (n,P,4,K)
        ph = torch.polar(torch.ones_like(frac), -2 * np.pi * frac)
        H = (a_t[..., None] * ph).sum(1)                                              # (n,4,K)
        fe = np.stack([frontend_gain(self.f, "real", fc) if r else np.ones_like(self.f)
                       for r, fc in zip(sc.real_fe, sc.spk_fc)])
        fe_t = torch.tensor(fe, dtype=torch.float32, device=self.device)
        ci = torch.tensor(sc.chirp_idx, dtype=torch.long, device=self.device)
        Y = H * self.S[ci][:, None, :] * fe_t[:, None, :]
        full = torch.zeros(sc.n, 4, C.N_FFT_SYN // 2 + 1, dtype=torch.complex64, device=self.device)
        full[..., self.k] = Y
        y = torch.fft.irfft(full, n=C.N_FFT_SYN)[..., : C.N_WIN]

        names = np.where(sc.chirp_idx == 0, "us", "fb")
        pc = np.array([self.pc[nm] for nm in names])
        bw = np.array([C.CHIRPS[nm][1] - C.CHIRPS[nm][0] for nm in names])
        sigma = np.sqrt(pc / (10 ** (sc.snr1m_db / 10) * bw / (C.FS / 2)))
        sig_t = torch.tensor(sigma, dtype=torch.float32, device=self.device)[:, None, None]
        # numpy RNG so that data are identical whether rendered on CPU or GPU
        noise = torch.from_numpy(rng.standard_normal(tuple(y.shape), dtype=np.float32)).to(self.device)
        y = y + sig_t * noise
        if sc.gain_db is not None:
            y = y * torch.tensor(10 ** (sc.gain_db / 20), dtype=torch.float32, device=self.device)[:, None, None]

        d_direct = dist[:, 0].mean(-1)
        fe_db = np.array([inband_frontend_db(nm, "real", fc) if r else 0.0
                          for nm, r, fc in zip(names, sc.real_fe, sc.spk_fc)])
        direct_gain_db = 20 * np.log10(np.maximum(amp[:, 0].mean(-1) * np.maximum(d_direct, 0.05), 1e-6))
        snr = (sc.snr1m_db - 20 * np.log10(np.maximum(d_direct, 0.05))
               - (C.AIR_ABS_DB_PER_M + sc.excess_db_m) * d_direct + direct_gain_db + fe_db)
        return y, {"snr_db": snr, "los": los, "d_direct": d_direct}


# ---------------------------------------------------------------- random scenes for the dataset
def sample_scene_params(rng: np.random.Generator, n: int, terrain: np.ndarray, channel: np.ndarray) -> dict:
    """Per-example nuisance parameters shared by dataset generation and the closed-loop world."""
    chirp_idx = np.array([0 if C.CHANNELS[C.CHANNEL_NAMES[c]][0] == "us" else 1 for c in channel])
    real_fe = np.array([C.CHANNELS[C.CHANNEL_NAMES[c]][1] == "real" for c in channel])
    tname = np.array(C.TERRAINS)[terrain]
    ground = np.select(
        [tname == "flat", tname == "slope", tname == "rough", tname == "grass", tname == "indoor"],
        [rng.uniform(0.6, 0.9, n), rng.uniform(0.5, 0.8, n), rng.uniform(0.3, 0.6, n),
         rng.uniform(0.05, 0.3, n), np.zeros(n)],
        rng.uniform(0.5, 0.8, n),
    )
    excess = np.where(tname == "grass", rng.uniform(0.5, 1.5, n), np.where(tname == "rough", 0.1, 0.0))
    return dict(
        chirp_idx=chirp_idx, real_fe=real_fe, ground_coef=ground, excess_db_m=excess,
        temp_c=rng.uniform(5, 35, n),
        sync_bias=rng.uniform(-C.SYNC_BIAS_S, C.SYNC_BIAS_S, n),
        spk_fc=rng.uniform(*C.SPK_FC_RANGE, n),
        mic_gain_db=np.where(real_fe[:, None], rng.uniform(-C.MIC_GAIN_MISMATCH_DB, C.MIC_GAIN_MISMATCH_DB, (n, 4)), 0.0),
        mic_delay=np.where(real_fe[:, None], rng.uniform(-C.MIC_DELAY_MISMATCH_S, C.MIC_DELAY_MISMATCH_S, (n, 4)), 0.0),
        snr1m_db=rng.uniform(10, 45, n),
        gain_db=rng.uniform(-6, 6, n),
    )


def random_outdoor_wall(rng: np.random.Generator, robot_xy: np.ndarray, src_xy: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Facade plane on the far side of both robot and phone, present with probability p."""
    n = robot_xy.shape[0]
    ang = rng.uniform(-np.pi, np.pi, n)
    nv = np.stack([np.cos(ang), np.sin(ang)], -1)
    far = np.maximum((nv * robot_xy).sum(-1), (nv * src_xy).sum(-1))
    d = far + rng.uniform(1.0, 6.0, n)
    coef = np.where(rng.random(n) < p, rng.uniform(0.4, 0.9, n), 0.0)
    return np.stack([nv[:, 0], nv[:, 1], d, coef], -1)


def random_obstacles(rng: np.random.Generator, n: int, active: np.ndarray, centre: np.ndarray,
                     spread: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    obs = np.zeros((n, K_OBS, 4))
    k = rng.integers(2, K_OBS + 1, n)
    mask = (np.arange(K_OBS)[None] < k[:, None]) & active[:, None]
    obs[..., 0] = centre[:, None, 0] + rng.uniform(-spread, spread, (n, K_OBS))
    obs[..., 1] = centre[:, None, 1] + rng.uniform(-spread, spread, (n, K_OBS))
    obs[..., 2] = np.where(mask, rng.uniform(0.15, 0.5, (n, K_OBS)), 0.0)
    obs[..., 3] = rng.uniform(0.5, 1.0, (n, K_OBS))
    occl = rng.uniform(8.0, 20.0, (n, K_OBS))
    return obs, occl


def sample_dataset_scenes(rng: np.random.Generator, n: int) -> tuple[Scene, dict]:
    """Random single-snapshot scenes. Labels are relative to the robot head frame."""
    terrain = rng.integers(0, len(C.TERRAINS), n)
    channel = rng.choice(len(C.CHANNEL_NAMES), n, p=C.CHANNEL_PROBS)
    p = sample_scene_params(rng, n, terrain, channel)
    tname = np.array(C.TERRAINS)[terrain]
    indoor = tname == "indoor"

    heading = rng.uniform(-np.pi, np.pi, n)
    # covers HEAD_TILT_FRAC * cart lean seen in closed loop
    tilt_scale = np.where(np.isin(tname, ["slope", "rough"]), 0.35, 0.15)
    pitch = rng.uniform(-1, 1, n) * tilt_scale
    roll = rng.uniform(-1, 1, n) * tilt_scale * 0.5
    az = rng.uniform(-np.pi, np.pi, n)
    rng_h = rng.uniform(C.RANGE_MIN, C.RANGE_MAX, n)
    h_src = rng.uniform(*C.PHONE_HEIGHT_RANGE, n)

    room = np.zeros((n, 6))
    robot_xy = np.zeros((n, 2))
    idx = np.nonzero(indoor)[0]
    if len(idx):
        Lx, Ly = rng.uniform(4, 10, len(idx)), rng.uniform(4, 10, len(idx))
        room[idx] = np.stack([Lx, Ly, rng.uniform(2.5, 3.2, len(idx)), rng.uniform(0.5, 0.9, len(idx)),
                              rng.uniform(0.5, 0.9, len(idx)), rng.uniform(0.5, 0.9, len(idx))], -1)
        for t, i in enumerate(idx):
            for _ in range(200):
                rx, ry = rng.uniform(0.4, Lx[t] - 0.4), rng.uniform(0.4, Ly[t] - 0.4)
                sx = rx + rng_h[i] * np.cos(heading[i] + az[i])
                sy = ry + rng_h[i] * np.sin(heading[i] + az[i])
                if 0.3 < sx < Lx[t] - 0.3 and 0.3 < sy < Ly[t] - 0.3:
                    break
                rng_h[i] = max(C.RANGE_MIN, rng_h[i] * 0.85)
            robot_xy[i] = (rx, ry)

    world_az = heading + az
    src = np.stack([robot_xy[:, 0] + rng_h * np.cos(world_az), robot_xy[:, 1] + rng_h * np.sin(world_az), h_src], -1)
    mics = head_mics(robot_xy, heading, pitch, roll)
    wall = random_outdoor_wall(rng, robot_xy, src[:, :2], np.where(indoor, 0.0, np.where(tname == "obstacles", 0.5, 0.3)))
    mid = 0.5 * (robot_xy + src[:, :2])
    obstacles, occl = random_obstacles(rng, n, tname == "obstacles", mid, spread=3.5)
    # keep obstacles off the robot and the phone
    for pt in (robot_xy, src[:, :2]):
        dd = np.linalg.norm(obstacles[..., :2] - pt[:, None], axis=-1)
        obstacles[..., 2] = np.where(dd < obstacles[..., 2] + 0.45, 0.0, obstacles[..., 2])

    jitter = rng.normal(0, C.SYNC_JITTER_S, n)
    sc = Scene(
        src=src, mics=mics, ground_coef=p["ground_coef"], wall=wall, room=room,
        obstacles=obstacles, occl_db=occl, tail_amp=np.where(indoor, rng.uniform(0.1, 0.35, n), 0.0),
        excess_db_m=p["excess_db_m"], temp_c=p["temp_c"], sync=p["sync_bias"] + jitter,
        chirp_idx=p["chirp_idx"], real_fe=p["real_fe"], spk_fc=p["spk_fc"],
        mic_gain_db=p["mic_gain_db"], mic_delay=p["mic_delay"], snr1m_db=p["snr1m_db"], gain_db=p["gain_db"],
    )
    labels = dict(
        azimuth=az.astype(np.float32), range=rng_h.astype(np.float32), terrain=terrain.astype(np.int8),
        channel=channel.astype(np.int8), h_src=h_src.astype(np.float32), pitch=pitch.astype(np.float32),
        temp_c=p["temp_c"].astype(np.float32), sync=sc.sync.astype(np.float32),
        snr1m_db=p["snr1m_db"].astype(np.float32), gain_db=p["gain_db"].astype(np.float32),
    )
    return sc, labels
