"""Which STFT phase normalisation keeps the direction information? Untrained steered response, 360 az x 13 el."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.features import PAIRS, STFTFeatures  # noqa: E402

dev = "cuda"
N = 1024
audio = torch.from_numpy(np.load(C.DATA_ROOT / "val" / "audio.npy")[:N]).to(dev).float()
lab = dict(np.load(C.DATA_ROOT / "val" / "labels.npz"))
X = STFTFeatures(dev, "full").logpow(audio)
i, j = zip(*PAIRS)
G = X[:, list(i)] * X[:, list(j)].conj()                       # (N,6,F,T)
P = G.abs()
f = np.arange(C.BIN_LO, C.BIN_HI) * C.FS / C.STFT_N
mic = np.zeros((4, 3))
mic[:, :2] = C.mic_xy()
az = np.arange(360) * 2 * np.pi / 360 - np.pi
el = np.deg2rad(np.arange(0, 65, 5.0))
EL, AZ = np.meshgrid(el, az, indexing="ij")
u = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
tau = np.stack([-(mic[a] - mic[b]) @ u.T / C.C_NOMINAL for a, b in PAIRS])
Ws = torch.tensor(np.exp(1j * 2 * np.pi * f[None, :, None] * tau[:, None, :]), dtype=torch.complex64, device=dev)

variants = {
    "power-sqrt weighted (current)": G / (P + 1e-12) * torch.sqrt(P / P.amax((2, 3), keepdim=True)),
    "per-TF PHAT": G / (P + 1e-12),
    "per-freq sum-normalised": G / (P.sum(3, keepdim=True) + 1e-12),
    "per-freq PHAT of time sum": (G.sum(3) / (G.sum(3).abs() + 1e-12)).unsqueeze(-1),
    "per-freq sum-norm, top-half frames": None,
}
w = P.sum(1, keepdim=True)
thr = w.median(3, keepdim=True).values
Gm = G * (w >= thr)
variants["per-freq sum-norm, top-half frames"] = Gm / (Gm.abs().sum(3, keepdim=True) + 1e-12)
fb = torch.tensor(lab["channel"][:N] == 2, device=dev)
ft = torch.tensor(f, device=dev, dtype=torch.float32)
in_us = (ft >= C.CHIRPS["us"][0] - 200) & (ft <= C.CHIRPS["us"][1] + 200)
in_fb = (ft >= C.CHIRPS["fb"][0] - 200) & (ft <= C.CHIRPS["fb"][1] + 200)
band = torch.where(fb[:, None], in_fb[None], in_us[None]).float().view(N, 1, -1, 1)
for k in list(variants):
    variants[k + " +band mask"] = variants[k] * band
# data-driven mask: per-frequency in-band energy vs the median bin
fe = P.sum((1, 3), keepdim=True)
variants["sum-norm x soft energy mask"] = variants["per-freq sum-normalised"] * torch.sqrt(fe / fe.amax(2, keepdim=True))
print("labels:", list(lab.keys()), "channel counts", np.bincount(lab["channel"][:N]))
for name, Z in variants.items():
    Zs = Z.sum(3)                                               # (N,6,F)
    beam = torch.einsum("npf,pfd->nd", Zs, Ws).real.reshape(N, len(el), len(az))
    est = az[beam.amax(1).argmax(1).cpu().numpy()]
    e = np.rad2deg(np.abs(np.mod(est - lab["azimuth"][:N] + np.pi, 2 * np.pi) - np.pi))
    print(f"{name:38s} median {np.median(e):5.1f}  mean {e.mean():5.1f}  frac<10 {np.mean(e < 10):.2f}")
