"""Physics and signal-chain sanity checks. Prints numbers and writes results/sanity.json.

    python scripts/sanity_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import acoustics as A  # noqa: E402
from bb8dl import config as C  # noqa: E402
from bb8dl import robot as Rb  # noqa: E402
from bb8dl.features import Classical  # noqa: E402


def robot_checks() -> dict:
    out = {}
    flat = dict(eta=np.ones(1), crr=np.full(1, 0.02), cd=np.full(1, 0.5), cyaw=np.full(1, 0.12), bump=np.zeros(1),
                grad=lambda x, y: (np.zeros_like(x), np.zeros_like(y)), obstacles=None, room=np.zeros((1, 2)))
    rng = np.random.default_rng(0)
    for name, u in [("full_forward", (1.0, 1.0)), ("half_forward", (0.5, 0.5)), ("turn_du1", (-0.5, 0.5))]:
        s = Rb.RobotState.zeros(1)
        for _ in range(int(6.0 / C.PHYS_DT)):
            Rb.step(s, np.array([u]), flat, rng)
        out[name] = dict(v=float(s.v[0]), yaw_rate=float(s.psi_d[0]), pitch_deg=float(np.rad2deg(s.phi[0])))
    for deg in [8.0, 13.0, 20.0]:
        g = np.tan(np.deg2rad(deg))
        slope = dict(flat, grad=lambda x, y, g=g: (np.full_like(x, g), np.zeros_like(y)))
        s = Rb.RobotState.zeros(1)
        for _ in range(int(6.0 / C.PHYS_DT)):
            Rb.step(s, np.array([[1.0, 1.0]]), slope, rng)
        out[f"uphill_{deg:.0f}deg_full"] = dict(v=float(s.v[0]), pitch_deg=float(np.rad2deg(s.phi[0])))
    grass = dict(flat, eta=np.full(1, 0.8), crr=np.full(1, 0.08), cd=np.full(1, 3.0), cyaw=np.full(1, 0.3))
    s = Rb.RobotState.zeros(1)
    for _ in range(int(6.0 / C.PHYS_DT)):
        Rb.step(s, np.array([[1.0, 1.0]]), grass, rng)
    out["grass_full"] = dict(v=float(s.v[0]), pitch_deg=float(np.rad2deg(s.phi[0])))
    return out


def acoustic_checks(dev) -> dict:
    rng = np.random.default_rng(1)
    ren = A.Renderer(dev)
    cl = Classical(dev)
    out = {}
    n = 256
    sc, lab = A.sample_dataset_scenes(rng, n)
    sc.snr1m_db[:] = 60.0
    for tag, an in [("anechoic_hiSNR", True), ("multipath_hiSNR", False)]:
        y, ex = ren.render(sc, rng, anechoic=an)
        r = cl(y, sc.chirp_idx)
        e = np.rad2deg(np.abs(np.mod(r["az_srp"].cpu().numpy() - lab["azimuth"] + np.pi, 2 * np.pi) - np.pi))
        e_ls = np.rad2deg(np.abs(np.mod(r["az_ls"].cpu().numpy() - lab["azimuth"] + np.pi, 2 * np.pi) - np.pi))
        re = np.abs(r["range"].cpu().numpy() - lab["range"])
        ch = lab["channel"]
        out[tag] = {C.CHANNEL_NAMES[c]: dict(srp_mae=float(e[ch == c].mean()), srp_median=float(np.median(e[ch == c])),
                                             ls_median=float(np.median(e_ls[ch == c])),
                                             range_median=float(np.median(re[ch == c])), n=int((ch == c).sum()))
                    for c in range(3)}
    out["frontend_inband_db"] = {
        "B_us(spk_fc=20k)": A.inband_frontend_db("us", "real", 20_000.0),
        "B_fb(spk_fc=20k)": A.inband_frontend_db("fb", "real", 20_000.0),
    }
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = dict(robot=robot_checks(), acoustics=acoustic_checks(dev))
    print(json.dumps(res, indent=1))
    C.RESULTS.mkdir(exist_ok=True)
    (C.RESULTS / "sanity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
