"""Copy a small, stratified slice of the test split into data/sample/ (committed, used by the browser).

    python scripts/export_sample.py --n 144
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=144)
    args = ap.parse_args()
    src = C.DATA_ROOT / "test"
    lab = dict(np.load(src / "labels.npz"))
    audio = np.load(src / "audio.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    per = args.n // (len(C.TERRAINS) * len(C.CHANNEL_NAMES))
    idx = []
    for t in range(len(C.TERRAINS)):
        for c in range(len(C.CHANNEL_NAMES)):
            cand = np.nonzero((lab["terrain"] == t) & (lab["channel"] == c))[0]
            idx += list(rng.choice(cand, min(per, len(cand)), replace=False))
    idx = np.sort(np.array(idx))
    out = Path(__file__).resolve().parents[1] / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "audio.npy", np.asarray(audio[idx]).astype(np.float16))
    np.savez(out / "labels.npz", test_index=idx, **{k: v[idx] for k, v in lab.items()})
    print(f"wrote {len(idx)} examples to {out}")


if __name__ == "__main__":
    main()
