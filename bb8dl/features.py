"""Network input features and the classical GCC-PHAT / SRP-PHAT / matched-filter baseline.

Everything is batched torch on the same device as the audio.

Sign convention: a plane wave from unit direction u reaches mic i at
t_i = t0 - p_i.u / c, so the GCC-PHAT peak of pair (i, j) is at t_i - t_j.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from . import config as C
from .acoustics import chirp

PAIRS = list(itertools.combinations(range(4), 2))  # 6 pairs


# ---------------------------------------------------------------- network input
class STFTFeatures:
    """(n,4,N_WIN) audio -> (n,Cin,32,65) tensor.

    Channels: 4 log-power maps, cos/sin of the phase of X_m X_0^* for m=1..3,
    and a time-index ramp (absolute time = time of flight, needed for range).
    `mode="mag"` drops the phase channels (ablation).
    """

    def __init__(self, device, mode: str = "full", stats: tuple[float, float] | None = None):
        self.device = torch.device(device)
        self.mode = mode
        self.win = torch.hann_window(C.STFT_N, device=self.device)
        self.ramp = torch.linspace(-1, 1, C.N_FRAMES, device=self.device)
        self.stats = stats or (0.0, 1.0)

    @property
    def in_ch(self) -> int:
        return 11 if self.mode == "full" else 5

    def logpow(self, y: torch.Tensor) -> torch.Tensor:
        n = y.shape[0]
        X = torch.stft(y.reshape(-1, C.N_WIN).float(), C.STFT_N, C.STFT_HOP, window=self.win,
                       center=True, return_complex=True)[:, C.BIN_LO:C.BIN_HI]
        return X.reshape(n, 4, C.BIN_HI - C.BIN_LO, -1)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        X = self.logpow(y)
        lp = torch.log(X.real**2 + X.imag**2 + 1e-10)
        lp = (lp - self.stats[0]) / self.stats[1]
        n, _, F, T = lp.shape
        parts = [lp]
        if self.mode == "full":
            cross = X[:, 1:] * X[:, :1].conj()
            cross = cross / (cross.abs() + 1e-9)
            parts += [cross.real, cross.imag]
        parts.append(self.ramp.view(1, 1, 1, T).expand(n, 1, F, T))
        return torch.cat(parts, 1)


# ---------------------------------------------------------------- classical baseline
class Classical:
    """SRP-PHAT azimuth, pairwise GCC-PHAT TDOA + least squares, matched-filter range."""

    def __init__(self, device, n_az: int = 360, els_deg=tuple(range(0, 65, 5)), lag_res: float = 0.1):
        self.device = torch.device(device)
        N = C.N_FFT_SYN
        f = np.fft.rfftfreq(N, 1 / C.FS)
        self.k = np.nonzero((f >= C.SYN_BAND[0]) & (f <= C.SYN_BAND[1]))[0]
        fk = f[self.k]
        self.masks = torch.tensor(
            np.stack([(fk >= C.CHIRPS[nm][0] - 200) & (fk <= C.CHIRPS[nm][1] + 200) for nm in ["us", "fb"]]),
            dtype=torch.float32, device=self.device)
        self.nbins = self.masks.sum(1)

        mic = np.zeros((4, 3))
        mic[:, :2] = C.mic_xy()
        self.az = np.deg2rad(np.arange(n_az) * 360.0 / n_az - 180.0)
        self.el = np.deg2rad(np.array(els_deg, dtype=float))
        AZ, EL = np.meshgrid(self.az, self.el, indexing="ij")
        u = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
        tau = np.stack([-(mic[i] - mic[j]) @ u.T / C.C_NOMINAL for i, j in PAIRS])  # (6,S)
        E = np.exp(1j * 2 * np.pi * fk[None, :, None] * tau[:, None, :])               # (6,K,S)
        self.E = torch.tensor(E.reshape(-1, E.shape[-1]), dtype=torch.complex64, device=self.device)
        self.n_el = len(self.el)

        max_lag = 2 * C.ARRAY_RADIUS / C.C_NOMINAL * C.FS + 1.0
        self.lags = np.arange(-max_lag, max_lag + 1e-9, lag_res)  # samples
        L = np.exp(1j * 2 * np.pi * fk[:, None] * self.lags[None] / C.FS)
        self.L = torch.tensor(L, dtype=torch.complex64, device=self.device)               # (K,nl)
        D = np.stack([-(mic[i] - mic[j])[:2] for i, j in PAIRS]) / C.C_NOMINAL            # (6,2)
        self.D_pinv = torch.tensor(np.linalg.pinv(D), dtype=torch.float32, device=self.device)
        self.max_lag = max_lag

        spec = []
        for nm in ["us", "fb"]:
            s = np.zeros(N)
            c = chirp(nm)
            s[: len(c)] = c
            spec.append(np.fft.rfft(s)[self.k])
        self.S = torch.tensor(np.stack(spec), dtype=torch.complex64, device=self.device)
        self.kt = torch.tensor(self.k, device=self.device)

    def spectra(self, y: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft(y.float(), n=C.N_FFT_SYN)[..., self.kt]  # (n,4,K)

    def gcc(self, X: torch.Tensor, chirp_idx: torch.Tensor) -> torch.Tensor:
        m = self.masks[chirp_idx]
        G = torch.stack([X[:, i] * X[:, j].conj() for i, j in PAIRS], 1)
        return G / (G.abs() + 1e-12) * m[:, None, :]  # (n,6,K)

    @torch.no_grad()
    def __call__(self, y: torch.Tensor, chirp_idx) -> dict:
        ci = torch.as_tensor(chirp_idx, device=self.device, dtype=torch.long)
        X = self.spectra(y)
        G = self.gcc(X, ci)
        nb = self.nbins[ci]
        n = y.shape[0]
        srp = (G.reshape(n, -1) @ self.E).real / (6 * nb[:, None])           # (n,S)
        best = srp.argmax(1)
        ia, ie = best // self.n_el, best % self.n_el
        az_srp = torch.tensor(self.az, device=self.device, dtype=torch.float32)[ia]
        el_srp = torch.tensor(self.el, device=self.device, dtype=torch.float32)[ie]
        srp_peak = srp.max(1).values

        R = (G @ self.L).real / nb[:, None, None]                              # (n,6,nl)
        il = R.argmax(-1)
        lag = torch.tensor(self.lags, device=self.device, dtype=torch.float32)[il] / C.FS
        peak = R.max(-1).values
        v = lag @ self.D_pinv.T                                                # (n,2)
        az_ls = torch.atan2(v[:, 1], v[:, 0])

        rng, lvl = self.matched_range(X, ci)
        return dict(az_srp=az_srp, el_srp=el_srp, srp_peak=srp_peak, az_ls=az_ls,
                    lag=lag * C.FS / self.max_lag, gcc_peak=peak, range=rng, level=lvl)

    def matched_range(self, X: torch.Tensor, ci: torch.Tensor):
        """First-arrival time of flight from the summed 4-mic matched-filter envelope."""
        n = X.shape[0]
        Z = torch.zeros(n, 4, C.N_FFT_SYN, dtype=torch.complex64, device=self.device)
        Z[..., self.kt] = 2 * X * self.S[ci][:, None, :].conj() * self.masks[ci][:, None, :]
        env = torch.fft.ifft(Z).abs().sum(1)[:, : C.N_WIN]                    # (n,T)
        mx = env.max(1, keepdim=True).values
        above = env >= 0.5 * mx
        first = above.float().argmax(1)
        # refine: local max within the next 24 samples
        idx = first[:, None] + torch.arange(24, device=self.device)[None]
        idx = idx.clamp(max=C.N_WIN - 2)
        loc = env.gather(1, idx).argmax(1)
        t0 = (first + loc).clamp(1, C.N_WIN - 2)
        ym, y0, yp = env.gather(1, (t0 - 1)[:, None]), env.gather(1, t0[:, None]), env.gather(1, (t0 + 1)[:, None])
        denom = (ym - 2 * y0 + yp).squeeze(1)
        off = torch.where(denom.abs() > 1e-9, 0.5 * (ym - yp).squeeze(1) / denom, torch.zeros_like(denom))
        t = (t0.float() + off.clamp(-1, 1)) / C.FS
        d = C.C_NOMINAL * t
        dh = C.PHONE_HEIGHT_NOMINAL - C.MIC_HEIGHT
        r = torch.sqrt(torch.clamp(d**2 - dh**2, min=0.01))
        level = torch.log10(mx.squeeze(1) + 1e-9)
        return r, level

    def rl_features(self, out: dict) -> torch.Tensor:
        """Compact acoustic observation for the RL policy (17 dims)."""
        return torch.cat([
            out["lag"], out["gcc_peak"],
            torch.cos(out["az_srp"])[:, None], torch.sin(out["az_srp"])[:, None],
            out["srp_peak"][:, None], (out["range"][:, None] / 4.0) - 1.0,
            (out["level"][:, None] + 1.0) / 2.0,
        ], 1)
