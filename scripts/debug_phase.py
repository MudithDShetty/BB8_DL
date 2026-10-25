"""Diagnostic: how much bearing information survives in different STFT phase representations?

Runs a plain steered-response search on each representation (no learning) and
reports the median bearing error on 1024 validation clips.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.features import PAIRS, Classical  # noqa: E402

dev = "cuda"
d = C.DATA_ROOT / "val"
N = 1024
audio = torch.from_numpy(np.load(d / "audio.npy")[:N]).to(dev).float()
lab = dict(np.load(d / "labels.npz"))
az_t = lab["azimuth"][:N]
ch = lab["channel"][:N]
chirp_of = np.array([0 if C.CHANNELS[c][0] == "us" else 1 for c in C.CHANNEL_NAMES])


def err(est):
    return np.rad2deg(np.abs(np.mod(est - az_t + np.pi, 2 * np.pi) - np.pi))


o = Classical(dev)(audio, chirp_of[ch])
print(f"classical full-window SRP-PHAT: median {np.median(err(o['az_srp'].cpu().numpy())):.1f}")

mic = np.zeros((4, 3))
mic[:, :2] = C.mic_xy()
azg = np.deg2rad(np.arange(-180, 180, 1.0))
u = np.stack([np.cos(azg), np.sin(azg), 0 * azg], 1)
tau = np.stack([-(mic[i] - mic[j]) @ u.T / C.C_NOMINAL for i, j in PAIRS])
for nfft in [256, 512]:
    win = torch.hann_window(nfft, device=dev)
    X = torch.stft(audio.reshape(-1, C.N_WIN), nfft, 32, window=win, center=True, return_complex=True)
    f_all = np.arange(X.shape[1]) * C.FS / nfft
    sel = (f_all >= 16_900) & (f_all <= 22_700)
    X = X[:, sel].reshape(N, 4, sel.sum(), -1)
    f = f_all[sel]
    st = torch.tensor(np.exp(1j * 2 * np.pi * f[None, :, None] * tau[:, None, :]), dtype=torch.complex64, device=dev)
    i, j = zip(*PAIRS)
    G = X[:, list(i)] * X[:, list(j)].conj()
    P = G.abs()
    for name, Z in [("phat_unweighted", G / (P + 1e-12)),
                    ("phat_x_power", G / (P + 1e-12) * P / P.amax((2, 3), keepdim=True)),
                    ("phat_x_sqrtpower", G / (P + 1e-12) * (P / P.amax((2, 3), keepdim=True)).sqrt()),
                    ("raw_cross", G)]:
        score = torch.einsum("npft,pfa->na", Z.to(torch.complex64), st).real
        e = err(azg[score.argmax(1).cpu().numpy()])
        print(f"nfft {nfft:3d} {name:18s}: median {np.median(e):5.1f}  mean {e.mean():5.1f}")
