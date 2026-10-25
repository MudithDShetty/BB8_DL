"""Train the deep localizer on the simulated dataset (GPU, mixed precision).

    python scripts/train_localizer.py --tag full
    python scripts/train_localizer.py --tag mag --mode mag
    python scripts/train_localizer.py --tag anechoic --train-split train_anechoic

The whole float16 dataset (about 1 GB) is kept resident on the GPU and STFT
features are computed per batch on the GPU, so there is no loader bottleneck.
Each training batch gets a random exact symmetry of the square array
(bb8dl/augment.py), unless --no-aug.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402
from bb8dl.augment import d4_apply  # noqa: E402
from bb8dl.features import STFTFeatures  # noqa: E402
from bb8dl.model import Localizer, loss_fn, n_params  # noqa: E402


def load_split(name: str, device) -> tuple[torch.Tensor, dict]:
    d = C.DATA_ROOT / name
    audio = torch.from_numpy(np.load(d / "audio.npy")).to(device)
    lab = dict(np.load(d / "labels.npz"))
    lab["chirp_idx"] = chirp_of_channel(lab["channel"])
    return audio, lab


def chirp_of_channel(channel: np.ndarray) -> np.ndarray:
    return np.array([0 if C.CHANNELS[C.CHANNEL_NAMES[c]][0] == "us" else 1 for c in channel])


def angle_err_deg(a, b):
    return np.rad2deg(np.abs(np.mod(a - b + np.pi, 2 * np.pi) - np.pi))


@torch.no_grad()
def predict(model, feat, audio, ci, bs=256):
    model.eval()
    outs = []
    ci = torch.as_tensor(ci, device=audio.device)
    for s in range(0, audio.shape[0], bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=audio.is_cuda):
            o = model(feat(audio[s:s + bs].float(), ci[s:s + bs]))
        outs.append({k: v.float().cpu().numpy() for k, v in o.items() if k != "logits"})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def summarize(pred, lab) -> dict:
    ae = angle_err_deg(pred["az"], lab["azimuth"])
    re = np.abs(pred["range"] - lab["range"])
    return dict(bearing_mae_deg=float(ae.mean()), bearing_rmse_deg=float(np.sqrt((ae**2).mean())),
                bearing_median_deg=float(np.median(ae)), range_mae_m=float(re.mean()),
                range_rmse_m=float(np.sqrt((re**2).mean())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--mode", default="full", choices=["full", "mag"])
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many train examples")
    ap.add_argument("--no-aug", action="store_true", help="disable the D4 square-array symmetry augmentation")
    args = ap.parse_args()

    torch.manual_seed(C.SEEDS["torch"])
    np.random.seed(C.SEEDS["torch"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print(f"device: {dev_name}  torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)

    tr_a, tr_l = load_split(args.train_split, dev)
    if args.limit:
        tr_a, tr_l = tr_a[: args.limit], {k: v[: args.limit] for k, v in tr_l.items()}
    va_a, va_l = load_split("val", dev)
    te_a, te_l = load_split("test", dev)
    print(f"train {tr_a.shape[0]}  val {va_a.shape[0]}  test {te_a.shape[0]}", flush=True)

    probe = STFTFeatures(dev, args.mode)
    X = probe.logpow(tr_a[:2048].float())
    lp = torch.log(X.real**2 + X.imag**2 + 1e-10)
    stats = (float(lp.mean()), float(lp.std()))
    feat = STFTFeatures(dev, args.mode, stats)

    model = Localizer(in_ch=feat.in_ch).to(dev)
    print(f"params: {n_params(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = tr_a.shape[0] // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    az_t = torch.tensor(tr_l["azimuth"], device=dev)
    rg_t = torch.tensor(tr_l["range"], device=dev)
    ci_t = torch.tensor(tr_l["chirp_idx"], device=dev)
    run_dir = C.DATA_ROOT / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    hist, best, best_state = [], 1e9, None
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(tr_a.shape[0], device=dev)
        tl, n_b, t0 = 0.0, 0, time.time()
        for b in range(steps_per_epoch):
            idx = perm[b * args.bs:(b + 1) * args.bs]
            xa, za = tr_a[idx].float(), az_t[idx]
            if not args.no_aug:
                xa, za = d4_apply(xa, za, torch.randint(0, 8, (len(idx),), device=dev))
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                out = model(feat(xa, ci_t[idx]))
            loss, _ = loss_fn(out, za, rg_t[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
            n_b += 1
            if ep == 0 and b == 49 and dev.type == "cuda":
                print(f"  {(time.time() - t0) / 50 * 1000:.0f} ms/batch, peak mem "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        pv = predict(model, feat, va_a, va_l["chirp_idx"])
        vm = summarize(pv, va_l)
        # train-set error on a fixed 8k subset, for the train/val curve
        ptr = predict(model, feat, tr_a[:8000], tr_l["chirp_idx"][:8000])
        tm = summarize(ptr, {k: v[:8000] for k, v in tr_l.items()})
        score = vm["bearing_mae_deg"] / 10.0 + vm["range_mae_m"]
        rec = dict(epoch=ep + 1, train_loss=tl / n_b, lr=sched.get_last_lr()[0], sec=time.time() - t0,
                   train=tm, val=vm)
        hist.append(rec)
        print(f"ep {ep + 1:02d} loss {tl / n_b:.4f}  train bMAE {tm['bearing_mae_deg']:.2f} rMAE {tm['range_mae_m']:.3f} | "
              f"val bMAE {vm['bearing_mae_deg']:.2f} rMAE {vm['range_mae_m']:.3f}  ({rec['sec']:.1f}s)", flush=True)
        if score < best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        (run_dir / "history.json").write_text(json.dumps(hist, indent=1))

    model.load_state_dict(best_state)
    pt = predict(model, feat, te_a, te_l["chirp_idx"])
    test = summarize(pt, te_l)
    print(f"TEST  bMAE {test['bearing_mae_deg']:.2f} deg  bRMSE {test['bearing_rmse_deg']:.2f}  "
          f"rMAE {test['range_mae_m']:.3f} m  rRMSE {test['range_rmse_m']:.3f}", flush=True)
    ck = dict(state=best_state, mode=args.mode, in_ch=feat.in_ch, stats=stats, tag=args.tag,
              train_split=args.train_split, epochs=args.epochs, params=n_params(model))
    torch.save(ck, run_dir / "localizer_full.pt")
    C.CHECKPOINTS.mkdir(exist_ok=True)
    small = dict(ck, state={k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()})
    torch.save(small, C.CHECKPOINTS / f"localizer_{args.tag}.pt")
    np.savez(C.DATA_ROOT / "runs" / args.tag / "test_pred.npz", **pt)
    out = dict(tag=args.tag, mode=args.mode, train_split=args.train_split, d4_augmentation=not args.no_aug, device=dev_name,
               torch=torch.__version__, params=n_params(model), epochs=args.epochs, batch=args.bs,
               lr=args.lr, n_train=int(tr_a.shape[0]), n_val=int(va_a.shape[0]), n_test=int(te_a.shape[0]),
               train_seconds=time.time() - t_start, test=test, history=hist, seed=C.SEEDS["torch"])
    C.RESULTS.mkdir(exist_ok=True)
    (C.RESULTS / f"train_localizer_{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
