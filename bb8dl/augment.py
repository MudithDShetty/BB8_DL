"""Exact symmetry augmentation for the square 4-mic array (dihedral group D4, 8 elements).

Mics sit at 45, 135, 225, 315 degrees. Rotating the source by k*90 degrees is the same
as relabelling mic i -> mic (i-k) mod 4; reflecting about the forward axis (az -> -az)
relabels mic i -> mic 3-i. Element g in 0..7: reflect if g >= 4, then rotate by
(g mod 4) * 90 degrees. Range is unchanged. Not exact for a tilted head (pitch and roll
swap roles), which is a small effect at the tilts simulated (verified by
scripts/debug_symmetry.py).
"""

from __future__ import annotations

import math

import torch

_PERM = torch.tensor([[(3 - ((i - k) % 4)) if r else ((i - k) % 4) for i in range(4)]
                      for r in (0, 1) for k in range(4)])  # (8,4): new channel i <- old channel _PERM[g, i]


def d4_apply(audio: torch.Tensor, az: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """audio (B,4,N), az (B,) rad, g (B,) int in 0..7 -> transformed audio and azimuth."""
    perm = _PERM.to(audio.device)[g]                                    # (B,4)
    out = torch.gather(audio, 1, perm[:, :, None].expand(-1, -1, audio.shape[2]))
    refl = g >= 4
    az2 = torch.where(refl, -az, az) + (g % 4).to(az.dtype) * (math.pi / 2)
    az2 = torch.remainder(az2 + math.pi, 2 * math.pi) - math.pi
    return out, az2
