"""Verify the D4 (square-array) augmentation: permuted channels + transformed azimuth must stay consistent.

Uses the classical SRP-PHAT localizer as the referee: its error on transformed
examples (against transformed labels) must match its error on the originals.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.augment import d4_apply  # noqa: E402
from bb8dl.features import Classical  # noqa: E402

dev = "cuda"
N = 2048
audio = torch.from_numpy(np.load(C.DATA_ROOT / "val" / "audio.npy")[:N]).to(dev).float()
lab = dict(np.load(C.DATA_ROOT / "val" / "labels.npz"))
ci = np.array([0 if C.CHANNELS[C.CHANNEL_NAMES[c]][0] == "us" else 1 for c in lab["channel"][:N]])
az = torch.tensor(lab["azimuth"][:N], device=dev)
cl = Classical(dev)
for g in range(8):
    a2, z2 = d4_apply(audio, az, torch.full((N,), g, device=dev))
    est = torch.cat([cl(a2[s:s + 512], ci[s:s + 512])["az_srp"] for s in range(0, N, 512)])
    e = torch.rad2deg(torch.abs(torch.remainder(est - z2 + np.pi, 2 * np.pi) - np.pi)).cpu().numpy()
    print(f"g={g}: SRP-PHAT median {np.median(e):5.2f}  mean {e.mean():5.2f}")
