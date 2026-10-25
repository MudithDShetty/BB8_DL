"""Held-out test-set localization: deep models vs SRP-PHAT and GCC-PHAT LS-TDOA.

    python scripts/eval_localization.py

All methods see the same 8000 test recordings. Writes results/localization.json
and results/localization_errors.npz (per-example errors, for CDF figures).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.controllers import DLLoc  # noqa: E402
from bb8dl.features import Classical  # noqa: E402


def ang_err(a, b):
    return np.rad2deg(np.abs(np.mod(a - b + np.pi, 2 * np.pi) - np.pi))


def stats(ae, re) -> dict:
    d = dict(n=int(len(ae)), bearing_mae_deg=float(np.mean(ae)), bearing_rmse_deg=float(np.sqrt(np.mean(ae**2))),
             bearing_median_deg=float(np.median(ae)), bearing_p90_deg=float(np.percentile(ae, 90)),
             frac_bearing_lt_10deg=float(np.mean(ae < 10)))
    if re is not None:
        d.update(range_mae_m=float(np.mean(re)), range_rmse_m=float(np.sqrt(np.mean(re**2))),
                 range_median_m=float(np.median(re)), range_p90_m=float(np.percentile(re, 90)))
    return d


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = C.DATA_ROOT / "test"
    audio = torch.from_numpy(np.load(d / "audio.npy")).to(dev)
    lab = dict(np.load(d / "labels.npz"))
    n = len(lab["azimuth"])
    preds = {}

    cl = Classical(dev)
    chirp_of = np.array([0 if C.CHANNELS[c][0] == "us" else 1 for c in C.CHANNEL_NAMES])
    ci = chirp_of[lab["channel"].astype(int)]
    az_srp, az_ls, rg = [], [], []
    t0 = time.time()
    for s in range(0, n, 256):
        o = cl(audio[s:s + 256].float(), ci[s:s + 256])
        az_srp.append(o["az_srp"].cpu().numpy())
        az_ls.append(o["az_ls"].cpu().numpy())
        rg.append(o["range"].cpu().numpy())
    torch.cuda.synchronize() if dev == "cuda" else None
    t_cl = (time.time() - t0) / n
    rg = np.concatenate(rg)
    preds["srp_phat"] = (np.concatenate(az_srp), rg)
    preds["gcc_ls_tdoa"] = (np.concatenate(az_ls), rg)

    lat = {"classical_batched_ms_per_example": 1000 * t_cl}
    for tag in ["full", "mag", "anechoic"]:
        p = C.CHECKPOINTS / f"localizer_{tag}.pt"
        if not p.exists():
            continue
        loc = DLLoc(p, dev)
        az, r = [], []
        t0 = time.time()
        for s in range(0, n, 512):
            a, rr, _, _ = loc({"audio": audio[s:s + 512].float(), "chirp_idx": ci[s:s + 512]})
            az.append(a)
            r.append(rr)
        lat[f"dl_{tag}_batched_ms_per_example"] = 1000 * (time.time() - t0) / n
        preds[f"dl_{tag}"] = (np.concatenate(az), np.concatenate(r))
        if tag == "full":
            x = {"audio": audio[:1].float(), "chirp_idx": ci[:1]}
            for _ in range(5):
                loc(x)
            t0 = time.time()
            for _ in range(50):
                loc(x)
            lat["dl_full_single_ms_gpu"] = 1000 * (time.time() - t0) / 50
            loc_cpu = DLLoc(p, "cpu")
            xc = {"audio": audio[:1].float().cpu(), "chirp_idx": ci[:1]}
            loc_cpu(xc)
            t0 = time.time()
            for _ in range(20):
                loc_cpu(xc)
            lat["dl_full_single_ms_cpu"] = 1000 * (time.time() - t0) / 20

    snr_bins = [(-99, 0), (0, 10), (10, 20), (20, 30), (30, 99)]
    rng_bins = [(0.5, 2.0), (2.0, 4.0), (4.0, 7.01)]
    out = dict(n_test=n, device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", latency=lat, methods={})
    errs = {}
    for name, (az, r) in preds.items():
        ae = ang_err(az, lab["azimuth"])
        re = np.abs(r - lab["range"])
        errs[f"{name}_bearing"] = ae.astype(np.float32)
        errs[f"{name}_range"] = re.astype(np.float32)
        m = dict(overall=stats(ae, re))
        m["by_channel"] = {cn: stats(ae[lab["channel"] == i], re[lab["channel"] == i]) for i, cn in enumerate(C.CHANNEL_NAMES)}
        m["by_terrain"] = {tn: stats(ae[lab["terrain"] == i], re[lab["terrain"] == i]) for i, tn in enumerate(C.TERRAINS)}
        m["by_snr"] = {f"{a}..{b}": stats(ae[(lab["snr_db"] >= a) & (lab["snr_db"] < b)], re[(lab["snr_db"] >= a) & (lab["snr_db"] < b)])
                       for a, b in snr_bins if ((lab["snr_db"] >= a) & (lab["snr_db"] < b)).any()}
        m["by_range"] = {f"{a}..{b}": stats(ae[(lab["range"] >= a) & (lab["range"] < b)], re[(lab["range"] >= a) & (lab["range"] < b)])
                         for a, b in rng_bins}
        m["by_los"] = {k: stats(ae[lab["los"] == v], re[lab["los"] == v]) for k, v in [("los", True), ("nlos", False)] if (lab["los"] == v).any()}
        out["methods"][name] = m
        o = m["overall"]
        print(f"{name:12s} bMAE {o['bearing_mae_deg']:6.2f}  bRMSE {o['bearing_rmse_deg']:6.2f}  med {o['bearing_median_deg']:5.2f}  "
              f"rMAE {o['range_mae_m']:.3f}  rRMSE {o['range_rmse_m']:.3f}", flush=True)
    errs.update({k: lab[k] for k in ["snr_db", "range", "terrain", "channel", "los"]})
    np.savez_compressed(C.RESULTS / "localization_errors.npz", **errs)
    (C.RESULTS / "localization.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(lat, indent=1))


if __name__ == "__main__":
    main()
