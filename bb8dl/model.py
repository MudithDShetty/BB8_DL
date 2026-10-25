"""Deep multi-channel localizer (v3).

Input: STFT features (18 x 32 freq x 65 frames, see features.STFTFeatures).

  spatial branch:  12 pair-phasor channels
                   -> per-frequency steering layer (learnable, initialised to the
                      SRP-PHAT steering vectors of 72 azimuths x 7 elevations, summed
                      over frequency) -> beam-time maps (7 elev x 72 az x 65 frames)
                   -> 3 residual conv blocks with circular padding in azimuth
  spectral branch: 4 log-power maps + 2 ramps -> 4 residual conv blocks (pool freq)
  temporal model:  per-frame [spatial summary, spectral features] -> 2-layer BiGRU
  heads:           azimuth = attention over frames pools the spatial maps
                   -> circular 1-D convs -> 72-way logits (5 deg bins);
                   range = second attention pooling of GRU states -> mean, log-variance

The attention over frames lets the net weight early (direct-path) frames over
later echo frames, which a full-window SRP-PHAT cannot do.

Two earlier designs that fed raw phase to a shift-invariant CNN failed to learn
bearing at all (results/aborted_localizer_v1_history.json, _v2_history.json):
with a 5 cm array at 17-23 kHz the phase -> direction map is spatially aliased
and frequency dependent, which a shared-weight convolution cannot represent.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

N_AZ = 72
ELEVS_DEG = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
AZ_GRID = torch.arange(N_AZ) * (2 * math.pi / N_AZ) - math.pi
SIGMA_AZ = math.radians(5.0)


class ConvBlock(nn.Module):
    """Two 3x3 convs with BN/GELU and a 1x1 residual; pools frequency only (time = range)."""

    def __init__(self, cin: int, cout: int, pool: bool = True):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False)
        self.pool = nn.MaxPool2d((2, 1)) if pool else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.pool(F.gelu(h + self.skip(x)))


class CircBlock(nn.Module):
    """Residual 3x3 block on (azimuth, time) maps; azimuth padded circularly."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=(0, 1), bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=(0, 1), bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False)

    @staticmethod
    def cpad(x):
        return torch.cat([x[:, :, -1:], x, x[:, :, :1]], 2)

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(self.cpad(x))))
        h = self.b2(self.c2(self.cpad(h)))
        return F.gelu(h + self.skip(x))


def steering_init(n_freq: int) -> torch.Tensor:
    """(F, 12, 4*72) weights so that sum_f W . [Re Z, Im Z] = Re sum_p,f Z_p(f) exp(+j 2 pi f tau_p(dir))."""
    from .features import PAIRS
    f = np.arange(C.BIN_LO, C.BIN_LO + n_freq) * C.FS / C.STFT_N
    mic = np.zeros((4, 3))
    mic[:, :2] = C.mic_xy()
    az = AZ_GRID.numpy()
    el = np.deg2rad(np.array(ELEVS_DEG))
    EL, AZ = np.meshgrid(el, az, indexing="ij")                     # (E, A)
    u = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
    tau = np.stack([-(mic[i] - mic[j]) @ u.T / C.C_NOMINAL for i, j in PAIRS])  # (6, D)
    th = 2 * np.pi * f[:, None, None] * tau[None]                                # (F, 6, D)
    W = np.concatenate([np.cos(th), -np.sin(th)], 1) / (6 * n_freq) * 8.0
    return torch.tensor(W, dtype=torch.float32)


class Localizer(nn.Module):
    def __init__(self, in_ch: int = 18, n_freq: int = C.BIN_HI - C.BIN_LO, gru: int = 192):
        super().__init__()
        self.full = in_ch == 18
        self.n_el = len(ELEVS_DEG)
        if self.full:
            self.steer = nn.Parameter(steering_init(n_freq))
            self.beam_bn = nn.BatchNorm2d(self.n_el)
            self.spatial = nn.Sequential(CircBlock(self.n_el, 24), CircBlock(24, 32), CircBlock(32, 32))
            sp_ch = 64
        else:
            sp_ch = 0
        self.spectral = nn.Sequential(ConvBlock(6, 32), ConvBlock(32, 64), ConvBlock(64, 96), ConvBlock(96, 128))
        f_out = n_freq // 16
        self.proj = nn.Sequential(nn.Linear(128 * f_out + sp_ch, 256), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(256, gru, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.att_az = nn.Linear(2 * gru, 1)
        self.att_r = nn.Linear(2 * gru, 1)
        self.r_head = nn.Sequential(nn.Linear(2 * gru, 128), nn.GELU(), nn.Linear(128, 2))
        if self.full:
            self.film = nn.Linear(2 * gru, 64)
            self.az_head = nn.Sequential(nn.Conv1d(32, 64, 5, padding=2, padding_mode="circular"), nn.GELU(),
                                         nn.Conv1d(64, 1, 5, padding=2, padding_mode="circular"))
        else:
            self.az_head_mlp = nn.Sequential(nn.Linear(2 * gru, 256), nn.GELU(), nn.Linear(256, N_AZ))

    def forward(self, x):
        B, _, Fq, T = x.shape
        spec_in = torch.cat([x[:, :4], x[:, -2:]], 1)
        s = self.spectral(spec_in)                                   # (B,128,F/16,T)
        s = s.permute(0, 3, 1, 2).reshape(B, T, -1)
        if self.full:
            beam = torch.einsum("bcft,fcd->bdt", x[:, 4:16], self.steer.to(x.dtype))   # (B, E*A, T)
            beam = self.beam_bn(beam.reshape(B, self.n_el, N_AZ, T))
            fa = self.spatial(beam)                                  # (B,32,A,T)
            summ = torch.cat([fa.amax(2), fa.mean(2)], 1).transpose(1, 2)  # (B,T,64)
            s = torch.cat([s, summ], 2)
        h, _ = self.gru(self.proj(s))                                # (B,T,2g)
        a_r = torch.softmax(self.att_r(h).squeeze(-1).float(), 1).to(h.dtype)
        zr = (h * a_r.unsqueeze(-1)).sum(1)
        a_az = torch.softmax(self.att_az(h).squeeze(-1).float(), 1).to(h.dtype)
        za = (h * a_az.unsqueeze(-1)).sum(1)
        if self.full:
            pooled = torch.einsum("bcat,bt->bca", fa, a_az)          # (B,32,A)
            g, b = self.film(za).chunk(2, 1)
            pooled = pooled * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)
            logits = self.az_head(pooled).squeeze(1).float()
        else:
            logits = self.az_head_mlp(za).float()
        ro = self.r_head(zr).float()
        az, s_az = decode_az(logits)
        return dict(logits=logits, az=az, s_az=s_az, range=F.softplus(ro[:, 0]) * 2.0,
                    logvar=ro[:, 1].clamp(-8, 4), att=a_az.float())


def decode_az(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax bin refined by a local circular mean; spread from the global resultant length."""
    p = torch.softmax(logits, 1)
    grid = AZ_GRID.to(logits.device)
    k = p.argmax(1)
    idx = (k[:, None] + torch.arange(-2, 3, device=logits.device)[None]) % N_AZ
    pw = p.gather(1, idx)
    th = grid[idx]
    az = torch.atan2((pw * th.sin()).sum(1), (pw * th.cos()).sum(1))
    Rg = torch.sqrt((p * grid.cos()).sum(1) ** 2 + (p * grid.sin()).sum(1) ** 2).clamp(1e-4, 0.9999)
    s_az = torch.sqrt(-2 * torch.log(Rg)).clamp(min=math.radians(2.0))
    return az, s_az


def loss_fn(out: dict, az: torch.Tensor, rng: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Soft-label cross entropy on azimuth bins + smooth-L1 and Gaussian NLL on range."""
    grid = AZ_GRID.to(az.device)
    d = torch.remainder(grid[None] - az[:, None] + math.pi, 2 * math.pi) - math.pi
    tgt = torch.exp(-0.5 * (d / SIGMA_AZ) ** 2)
    tgt = tgt / tgt.sum(1, keepdim=True)
    ce = -(tgt * F.log_softmax(out["logits"], 1)).sum(1)
    err = out["range"] - rng
    nll_r = 0.5 * (out["logvar"] + err**2 / out["logvar"].exp())
    l_r = F.smooth_l1_loss(out["range"], rng, beta=0.2, reduction="none")
    loss = (ce + 2.0 * l_r + 0.1 * nll_r).mean()
    return loss, dict(ce=ce.mean().item(), rng=l_r.mean().item())


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
