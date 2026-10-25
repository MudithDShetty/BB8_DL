"""Generate the simulated 4-mic chirp dataset.

    python scripts/gen_dataset.py --split train
    python scripts/gen_dataset.py --split train_anechoic --anechoic

Writes <BB8DL_DATA>/<split>/audio.npy (n,4,2048) float16 and labels.npz.
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
from bb8dl import acoustics as A  # noqa: E402
from bb8dl import config as C  # noqa: E402


def generate(split: str, n: int, seed: int, anechoic: bool, out_dir: Path, batch: int = 256, device="cuda") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = np.lib.format.open_memmap(out_dir / "audio.npy", mode="w+", dtype=np.float16, shape=(n, 4, C.N_WIN))
    ren = A.Renderer(device)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    cols: dict[str, list] = {}
    t0 = time.time()
    for s in range(0, n, batch):
        m = min(batch, n - s)
        sc, lab = A.sample_dataset_scenes(rng, m)
        y, extra = ren.render(sc, rng, gen, anechoic=anechoic)
        audio[s:s + m] = y.cpu().numpy().astype(np.float16)
        lab["snr_db"] = extra["snr_db"].astype(np.float32)
        lab["los"] = extra["los"].astype(np.bool_)
        for k, v in lab.items():
            cols.setdefault(k, []).append(v)
        if (s // batch) % 20 == 0:
            print(f"[{split}] {s + m}/{n}  {time.time() - t0:.1f}s", flush=True)
    audio.flush()
    labels = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(out_dir / "labels.npz", **labels)
    meta = dict(split=split, n=n, seed=seed, anechoic=anechoic, fs=C.FS, n_win=C.N_WIN,
                channels=C.CHANNEL_NAMES, terrains=C.TERRAINS, seconds=time.time() - t0,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{split}] done in {meta['seconds']:.1f}s -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test", "train_anechoic"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--anechoic", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    sizes = C.DatasetSizes()
    n = args.n or getattr(sizes, args.split)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate(args.split, n, C.SEEDS[args.split], args.anechoic or args.split == "train_anechoic",
             C.DATA_ROOT / args.split, args.batch, device)


if __name__ == "__main__":
    main()
