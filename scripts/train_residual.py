"""Collect inverse-dynamics data on all terrains and train the PID residual net.

    python scripts/train_residual.py

See bb8dl/residual.py for what the network learns.
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
from bb8dl.residual import HIST, K_AHEAD, ResidualNet, make_input  # noqa: E402
from bb8dl.robot import mix  # noqa: E402
from bb8dl.world import World, make_scenarios  # noqa: E402


def collect(n_per_terrain: int, T: int, seed: int):
    names = [t for t in C.TERRAINS for _ in range(n_per_terrain)]
    sc = None
    for i, t in enumerate(C.TERRAINS):
        one = make_scenarios(t, n_per_terrain, seed + i, "A_us", T)
        if sc is None:
            sc = {k: np.zeros((len(names),) + v.shape[1:], v.dtype) for k, v in one.items()}
        for k, v in one.items():
            sc[k][i * n_per_terrain:(i + 1) * n_per_terrain] = v
    w = World(sc, renderer=None, seed=seed)
    rng = np.random.default_rng(seed)
    n = w.n
    U = np.zeros((n, T, 2))
    Pr = np.zeros((n, T + 1, 5))
    hold = np.zeros(n, int)
    u = np.zeros((n, 2))
    Pr[:, 0] = w.proprio()
    for t in range(T):
        new = hold <= 0
        if new.any():
            # mixture: drive-like commands (forward + turn) and uniform random duties
            v = rng.uniform(-0.3, 1.2, n)
            om = rng.uniform(-2.5, 2.5, n)
            drive = mix(v, om)
            unif = rng.uniform(-1, 1, (n, 2))
            pick = rng.random(n) < 0.7
            cand = np.where(pick[:, None], drive, unif)
            u[new] = cand[new]
            hold[new] = rng.integers(K_AHEAD + 1, 16, new.sum())
        U[:, t] = u
        w.step(u)
        Pr[:, t + 1] = w.proprio()
        hold -= 1
    terr = np.repeat(np.arange(len(C.TERRAINS)), n_per_terrain)
    X, Y, TT = [], [], []
    for t in range(HIST, T - K_AHEAD):
        same = np.all(U[:, t:t + K_AHEAD + 1] == U[:, t:t + 1], axis=(1, 2))  # u held for K steps
        if not same.any():
            continue
        hist = np.concatenate([np.concatenate([np.zeros((n, 1, 2)), U[:, :-1]], 1)[:, t - HIST + 1:t + 1],
                               Pr[:, t - HIST + 1:t + 1]], 2)
        v_t, w_t = Pr[:, t + K_AHEAD, 0], Pr[:, t + K_AHEAD, 1]
        X.append(make_input(hist[same], v_t[same], w_t[same]))
        Y.append(U[same, t].astype(np.float32))
        TT.append(terr[same])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(TT)


def main():
    torch.manual_seed(C.SEEDS["torch"])
    t0 = time.time()
    Xtr, Ytr, Ttr = collect(48, 1200, C.SEEDS["residual_data"])
    Xva, Yva, Tva = collect(12, 1200, C.SEEDS["residual_data"] + 50)
    print(f"samples train {len(Xtr)} val {len(Xva)}  collect {time.time() - t0:.0f}s", flush=True)
    net = ResidualNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xv, Yv = torch.from_numpy(Xva), torch.from_numpy(Yva)
    # baseline: flat-ground mixer applied to the same targets
    vt = Xva[:, -2] * C.V_NOM_MAX
    wt = Xva[:, -1] * C.W_NOM_MAX
    mix_mae = float(np.abs(mix(vt, wt) - Yva).mean())
    hist = []
    epochs = 40
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tl = 0.0
        for b in range(0, len(Xt), 1024):
            idx = perm[b:b + 1024]
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv)
            vl = torch.nn.functional.mse_loss(pv, Yv).item()
            mae = (pv - Yv).abs().mean().item()
        hist.append(dict(epoch=ep + 1, train_mse=tl / len(Xt), val_mse=vl, val_mae=mae))
        print(f"ep {ep + 1:02d} train {tl / len(Xt):.5f} val {vl:.5f} mae {mae:.4f} (mixer mae {mix_mae:.4f})", flush=True)
    with torch.no_grad():
        pv = net(Xv).numpy()
    per_t = {C.TERRAINS[i]: dict(net_mae=float(np.abs(pv[Tva == i] - Yva[Tva == i]).mean()),
                                 mixer_mae=float(np.abs(mix(vt[Tva == i], wt[Tva == i]) - Yva[Tva == i]).mean()))
             for i in range(len(C.TERRAINS))}
    C.CHECKPOINTS.mkdir(exist_ok=True)
    torch.save(dict(state=net.state_dict(), hidden=128), C.CHECKPOINTS / "residual.pt")
    out = dict(n_train=len(Xtr), n_val=len(Xva), history=hist, val_mae=hist[-1]["val_mae"], mixer_val_mae=mix_mae,
               per_terrain=per_t, seconds=time.time() - t0, seed=C.SEEDS["residual_data"],
               params=sum(p.numel() for p in net.parameters()))
    (C.RESULTS / "train_residual.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(per_t, indent=1))


if __name__ == "__main__":
    main()
