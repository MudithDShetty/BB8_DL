"""Single source of truth for every physical, acoustic and training constant.

Anything a reader might want to cite in the paper lives here so the numbers in
the text, the simulator, the dataset and the app cannot drift apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CHECKPOINTS = ROOT / "checkpoints"
SAMPLE_DATA = ROOT / "data" / "sample"
# Large generated data lives outside the (OneDrive-synced) repo by default.
DATA_ROOT = Path(os.environ.get("BB8DL_DATA", Path.home() / "bb8dl_data"))

# ---------------------------------------------------------------- acoustics
FS = 48_000                 # Hz, I2S sample rate assumed for INMP441 capture
N_WIN = 2048                # samples per capture window (42.7 ms)
N_FFT_SYN = 4096            # FFT size used by the frequency-domain renderer
CHIRP_DUR = 0.010           # s
CHIRP_TAPER = 0.15          # Tukey taper fraction (limits spectral splatter)
CHIRP_PERIOD = 0.050        # s, one chirp per control step (20 Hz)
C_NOMINAL = 343.2           # m/s at 20 C; the receiver always assumes this
# Synthesis band. Bins outside it are set to zero (chirp energy there < -40 dB).
SYN_BAND = (16_000.0, 23_800.0)

CHIRPS = {
    # name: (f0, f1)
    "us": (20_000.0, 22_500.0),   # ultrasonic beacon (Daredevil-style, inaudible to most adults)
    "fb": (17_000.0, 19_500.0),   # hardware fallback, faintly audible to some people
}
# Acoustic channel = chirp + front-end. A is the ideal transducer; B is a phone
# speaker into an INMP441-class MEMS mic.
CHANNELS = {
    "A_us": ("us", "ideal"),
    "B_us": ("us", "real"),
    "B_fb": ("fb", "real"),
}
CHANNEL_NAMES = list(CHANNELS)
CHANNEL_PROBS = np.array([0.50, 0.15, 0.35])

# Realistic front-end. INMP441 datasheet specifies 60 Hz - 15 kHz response; its
# I2S decimation filter passband edge is ~0.423 fs with the stopband at 0.5 fs
# (values to be confirmed against the datasheet revision in use).
MIC_LP_FC = 19_000.0        # Hz, 2nd-order roll-off of the MEMS element above its spec
DECIM_PASS = 0.423 * FS     # 20.3 kHz
DECIM_STOP = 0.5 * FS       # 24 kHz
DECIM_STOP_DB = 60.0
SPK_FC_RANGE = (18_500.0, 21_000.0)  # phone speaker 4th-order roll-off corner, per device
MIC_GAIN_MISMATCH_DB = 1.0
MIC_DELAY_MISMATCH_S = 2e-6

AIR_ABS_DB_PER_M = 0.55     # ~20 kHz, 20 C, 50 % RH (ISO 9613-1 order of magnitude)
SYNC_JITTER_S = 100e-6      # per-chirp emission-time error after clock sync
SYNC_BIAS_S = 200e-6        # per-episode residual bias bound (uniform +-)

# Array: 4 mics in the BB-8 head, square on a circle, head stays aligned with the cart.
ARRAY_RADIUS = 0.05         # m
MIC_ANGLES = np.deg2rad([45.0, 135.0, 225.0, 315.0])
MIC_HEIGHT = 0.47           # m above local ground (shell 2R=0.40 + head)
HEAD_TILT_FRAC = 0.25       # head tilt / cart lean (assumed servo- or pivot-levelled head mast)
PHONE_HEIGHT_RANGE = (0.75, 1.45)
PHONE_HEIGHT_NOMINAL = 1.10
RANGE_MIN, RANGE_MAX = 0.5, 7.0  # horizontal range covered by the dataset

# STFT front end for the network
STFT_N = 256
STFT_HOP = 32
BIN_LO, BIN_HI = 90, 122    # 32 bins, 16.88 - 22.69 kHz (bin width 187.5 Hz)
N_FRAMES = N_WIN // STFT_HOP + 1  # 65

TERRAINS = ["flat", "slope", "rough", "grass", "indoor", "obstacles"]

# ---------------------------------------------------------------- robot
@dataclass(frozen=True)
class RobotParams:
    R: float = 0.20          # shell radius (m)
    m_s: float = 1.0         # shell mass (kg), thin shell
    m_p: float = 1.5         # internal cart (pendulum) mass (kg)
    l: float = 0.10          # cart COM distance below shell centre (m)
    I_p: float = 0.008       # cart pitch inertia about its COM (kg m^2)
    r_w: float = 0.05        # cart wheel radius (m), wheels run on the inner shell
    b: float = 0.16          # cart wheel base (m)
    tau_stall: float = 0.25  # per-wheel stall torque after gearbox (N m)
    w_noload: float = 30.0   # wheel no-load speed (rad/s)
    I_z: float = 0.02        # yaw inertia of cart + coupled shell (kg m^2)
    c_phi: float = 0.02      # cart pitch friction (N m s)
    phi_limit: float = 1.40  # rad; beyond this the cart "loops" -> counted as tumble
    g: float = 9.81

    @property
    def I_s(self) -> float:
        return 2.0 / 3.0 * self.m_s * self.R**2

    @property
    def m_tot(self) -> float:
        return self.m_s + self.m_p


ROBOT = RobotParams()
PHYS_DT = 0.005              # s, physics step
CTRL_DT = CHIRP_PERIOD       # s, one control decision per chirp
SUBSTEPS = int(round(CTRL_DT / PHYS_DT))
EPISODE_S = 60.0
STANDOFF = 0.8               # m, desired following distance
FOLLOW_RADIUS = 1.5          # m, "following" if within this
SUCCESS_FRAC = 0.8           # fraction of time within FOLLOW_RADIUS after warm-up
WARMUP_S = 10.0
V_NOM_MAX = 1.30             # m/s, nominal flat-ground speed at full command
W_NOM_MAX = 2.50             # rad/s, nominal yaw rate at full differential command

# ---------------------------------------------------------------- seeds
SEEDS = dict(
    train=1_000, val=2_000, test=3_000, train_anechoic=4_000,
    residual_data=5_000, rl=6_000, eval=7_000, torch=0,
)


@dataclass
class DatasetSizes:
    train: int = 64_000
    val: int = 8_000
    test: int = 8_000
    train_anechoic: int = 64_000
    extra: dict = field(default_factory=dict)


def mic_xy() -> np.ndarray:
    """(4, 2) mic offsets in the head frame (x forward, y left)."""
    return ARRAY_RADIUS * np.stack([np.cos(MIC_ANGLES), np.sin(MIC_ANGLES)], axis=1)
