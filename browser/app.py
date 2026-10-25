"""Dataset and results browser (Flask).

    python browser/app.py                 # http://127.0.0.1:5000
    python browser/app.py --split test    # browse a full split from BB8DL_DATA instead of data/sample

Per example: labels, 4-channel waveforms, 4-channel spectrograms, the deep model's and
SRP-PHAT's bearing estimates against the truth, and playback slowed 6x (20-22.5 kHz -> 3.3-3.75 kHz)
so the chirp and its echoes are audible. /results shows every figure and metric file
in results/.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import wave
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flask import Flask, Response, abort, render_template_string, request, send_from_directory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bb8dl import config as C  # noqa: E402

app = Flask(__name__)
DATA: dict = {}
SLOW = 6

BASE = """<!doctype html><html><head><meta charset="utf-8"><title>BB-8 DL browser</title>
<style>body{font:14px system-ui,sans-serif;margin:24px;max-width:1300px}table{border-collapse:collapse}
td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}th{background:#f2f2f2}a{color:#1565c0}
img{max-width:100%;border:1px solid #ddd;margin:6px 0}.nav a{margin-right:14px}pre{background:#f6f6f6;padding:10px;overflow:auto;max-height:500px}</style>
</head><body><div class="nav"><a href="/">examples</a><a href="/results">results</a></div>{{ body|safe }}</body></html>"""


def page(body: str) -> str:
    return render_template_string(BASE, body=body)


def load(split: str) -> None:
    d = ROOT / "data" / "sample" if split == "sample" else C.DATA_ROOT / split
    DATA["audio"] = np.load(d / "audio.npy", mmap_mode="r")
    DATA["lab"] = dict(np.load(d / "labels.npz"))
    DATA["split"] = split
    DATA["pred"] = predict()


def predict() -> dict:
    """DL and SRP-PHAT bearing for every example (CPU; skipped if no checkpoint)."""
    lab, n = DATA["lab"], len(DATA["lab"]["azimuth"])
    if n > 2000:
        return {}
    from bb8dl.features import Classical
    audio = torch.from_numpy(np.array(DATA["audio"], dtype=np.float32))
    ci = np.array([0 if C.CHANNELS[C.CHANNEL_NAMES[c]][0] == "us" else 1 for c in lab["channel"]])
    out = {"srp_az": Classical("cpu")(audio, ci)["az_srp"].numpy()}
    ck = C.CHECKPOINTS / "localizer_full.pt"
    if ck.exists():
        from bb8dl.controllers import DLLoc
        az, r, s_az, _ = DLLoc(ck, "cpu")({"audio": audio, "chirp_idx": ci})
        out.update(dl_az=az, dl_range=r, dl_saz=s_az)
    return out


def fmt(k: str, v) -> str:
    if k in ("azimuth", "srp_az", "dl_az", "dl_saz"):
        return f"{np.rad2deg(v):.2f} deg"
    v = np.asarray(v)
    if v.ndim:
        return str(np.round(v, 3).tolist())
    return f"{float(v):.3f}" if np.issubdtype(v.dtype, np.floating) else str(v)


def png(fig) -> Response:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")


def ex(i: int) -> np.ndarray:
    if not 0 <= i < len(DATA["lab"]["azimuth"]):
        abort(404)
    return np.asarray(DATA["audio"][i], dtype=np.float32)


@app.route("/")
def index():
    lab, pred = DATA["lab"], DATA["pred"]
    t = request.args.get("terrain", "")
    c = request.args.get("channel", "")
    keep = np.ones(len(lab["azimuth"]), bool)
    if t:
        keep &= lab["terrain"] == C.TERRAINS.index(t)
    if c:
        keep &= lab["channel"] == C.CHANNEL_NAMES.index(c)
    rows = []
    for i in np.nonzero(keep)[0][:500]:
        dl = f"{np.rad2deg(pred['dl_az'][i]):.1f}" if "dl_az" in pred else "-"
        srp = f"{np.rad2deg(pred['srp_az'][i]):.1f}" if "srp_az" in pred else "-"
        rows.append(f"<tr><td><a href='/ex/{i}'>{i}</a></td><td>{C.TERRAINS[lab['terrain'][i]]}</td>"
                    f"<td>{C.CHANNEL_NAMES[lab['channel'][i]]}</td><td>{np.rad2deg(lab['azimuth'][i]):.1f}</td>"
                    f"<td>{dl}</td><td>{srp}</td><td>{lab['range'][i]:.2f}</td><td>{lab['snr_db'][i]:.1f}</td>"
                    f"<td>{'yes' if lab['los'][i] else 'no'}</td></tr>")
    filt = " ".join(f"<a href='/?terrain={x}'>{x}</a>" for x in C.TERRAINS) + " | " + \
        " ".join(f"<a href='/?channel={x}'>{x}</a>" for x in C.CHANNEL_NAMES) + " | <a href='/'>all</a>"
    return page(f"<h2>split: {DATA['split']} ({len(lab['azimuth'])} examples)</h2><p>filter: {filt}</p>"
                "<table><tr><th>#</th><th>terrain</th><th>channel</th><th>true az</th><th>DL az</th><th>SRP az</th>"
                "<th>range m</th><th>SNR dB</th><th>LOS</th></tr>" + "".join(rows) + "</table>")


@app.route("/ex/<int:i>")
def example(i: int):
    ex(i)
    lab, pred = DATA["lab"], DATA["pred"]
    items = [(k, v[i]) for k, v in lab.items()]
    if pred:
        items += [(k, v[i]) for k, v in pred.items()]
    rows = "".join(f"<tr><th>{k}</th><td>{fmt(k, v)}</td></tr>" for k, v in items)
    return page(f"<h2>example {i} <small><a href='/ex/{max(i - 1, 0)}'>prev</a> <a href='/ex/{i + 1}'>next</a></small></h2>"
                f"<p>terrain <b>{C.TERRAINS[lab['terrain'][i]]}</b>, channel <b>{C.CHANNEL_NAMES[lab['channel'][i]]}</b></p>"
                f"<audio controls src='/ex/{i}/audio.wav'></audio> (slowed {SLOW}x, mic 0)"
                f"<div style='display:flex;gap:20px'><table>{rows}</table><img src='/ex/{i}/polar.png' width=320></div>"
                f"<img src='/ex/{i}/wave.png'><img src='/ex/{i}/spec.png'>")


@app.route("/ex/<int:i>/wave.png")
def wave_png(i: int):
    y = ex(i)
    t = np.arange(y.shape[1]) / C.FS * 1000
    fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    for m in range(4):
        ax[m].plot(t, y[m], lw=0.5)
        ax[m].set_ylabel(f"mic {m}")
    ax[-1].set_xlabel("time since emission (ms)")
    fig.suptitle("4-channel waveform")
    return png(fig)


@app.route("/ex/<int:i>/spec.png")
def spec_png(i: int):
    y = ex(i)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for m in range(4):
        ax[m].specgram(y[m], NFFT=256, Fs=C.FS, noverlap=224, cmap="magma")
        ax[m].set_ylim(15000, 24000)
        ax[m].set_title(f"mic {m}")
        ax[m].set_xlabel("s")
    ax[0].set_ylabel("Hz")
    return png(fig)


@app.route("/ex/<int:i>/polar.png")
def polar_png(i: int):
    lab, pred = DATA["lab"], DATA["pred"]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.plot([lab["azimuth"][i]] * 2, [0, lab["range"][i]], "g-", lw=3, label="truth")
    if "dl_az" in pred:
        ax.plot([pred["dl_az"][i]] * 2, [0, pred["dl_range"][i]], "-", color="tab:orange", lw=2, label="DL")
    if "srp_az" in pred:
        ax.plot([pred["srp_az"][i]] * 2, [0, lab["range"][i]], "b--", lw=1.5, label="SRP-PHAT")
    ax.set_rmax(7)
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(-0.15, -0.15))
    return png(fig)


@app.route("/ex/<int:i>/audio.wav")
def audio_wav(i: int):
    y = ex(i)[0]
    y = np.tile(np.concatenate([y / (np.abs(y).max() + 1e-9), np.zeros(len(y))]), 3)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(C.FS // SLOW)
        w.writeframes((0.8 * y * 32767).astype("<i2").tobytes())
    return Response(buf.getvalue(), mimetype="audio/wav")


@lru_cache(maxsize=1)
def result_files():
    r = C.RESULTS
    return sorted(r.glob("figures/*.png")), sorted(r.glob("*.json"))


@app.route("/results")
def results():
    result_files.cache_clear()
    figs, js = result_files()
    body = "<h2>figures</h2>" + "".join(f"<h4>{p.name}</h4><img src='/results/file/figures/{p.name}'>" for p in figs)
    body += "<h2>metric files</h2><ul>" + "".join(f"<li><a href='/results/json/{p.name}'>{p.name}</a> "
                                                  f"({p.stat().st_size // 1024} KB)</li>" for p in js) + "</ul>"
    return page(body or "no results yet")


@app.route("/results/json/<name>")
def results_json(name: str):
    p = C.RESULTS / name
    if p.suffix != ".json" or not p.exists() or p.parent != C.RESULTS:
        abort(404)
    d = json.loads(p.read_text())
    if isinstance(d, dict) and "history" in d:
        d = dict(d, history=f"<{len(d['history'])} epochs, see figures>")
    return page(f"<h2>{name}</h2><pre>{json.dumps(d, indent=1)[:200000]}</pre>")


@app.route("/results/file/figures/<name>")
def results_fig(name: str):
    return send_from_directory(C.RESULTS / "figures", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="sample")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    load(args.split)
    print(f"loaded {len(DATA['lab']['azimuth'])} examples from {args.split}; predictions: {list(DATA['pred'])}")
    app.run(host="127.0.0.1", port=args.port, debug=False)
