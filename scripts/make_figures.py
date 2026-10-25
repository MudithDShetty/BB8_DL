"""Build every figure from the saved JSON / NPZ results. No numbers are typed in here.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402

R, F = C.RESULTS, C.FIGURES
LABEL = dict(oracle="Oracle + pursuit", gcc_track="GCC-PHAT + tracker", dl_track="DL + tracker",
             dl_pid="DL + PID", dl_pid_res="DL + PID + residual", rl="RL (PPO)",
             srp_phat="SRP-PHAT", gcc_ls_tdoa="GCC-PHAT LS-TDOA", dl_full="DL (mag+phase)",
             dl_mag="DL (magnitude only)", dl_anechoic="DL (no multipath aug.)")
COL = dict(oracle="0.4", gcc_track="tab:orange", dl_track="tab:blue", dl_pid="tab:green", dl_pid_res="tab:red",
           rl="tab:purple", srp_phat="tab:orange", gcc_ls_tdoa="tab:brown", dl_full="tab:blue",
           dl_mag="tab:cyan", dl_anechoic="tab:olive")


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(F / name, dpi=150)
    plt.close(fig)
    print("wrote", F / name)


def training_curves():
    for tag in ["full", "mag", "anechoic"]:
        p = R / f"train_localizer_{tag}.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())["history"]
        ep = [r["epoch"] for r in h]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        ax[0].plot(ep, [r["train_loss"] for r in h])
        ax[0].set(title="training loss", xlabel="epoch")
        ax[1].plot(ep, [r["train"]["bearing_mae_deg"] for r in h], label="train (8k subset)")
        ax[1].plot(ep, [r["val"]["bearing_mae_deg"] for r in h], label="validation")
        ax[1].set(title="bearing MAE (deg)", xlabel="epoch")
        ax[1].legend()
        ax[2].plot(ep, [r["train"]["range_mae_m"] for r in h], label="train (8k subset)")
        ax[2].plot(ep, [r["val"]["range_mae_m"] for r in h], label="validation")
        ax[2].set(title="range MAE (m)", xlabel="epoch")
        ax[2].legend()
        fig.suptitle(f"Localizer training: {tag}")
        save(fig, f"train_localizer_{tag}.png")
    p = R / "train_rl_curve.json"
    if p.exists():
        rows = json.loads(p.read_text())
        t = np.array([r["timesteps"] for r in rows]) / 1e6
        ret = np.array([r["ep_return_mean"] for r in rows])
        k = max(1, len(ret) // 50)
        sm = np.convolve(ret, np.ones(k) / k, mode="valid")
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        ax[0].plot(t, ret, alpha=0.3, color="tab:purple")
        ax[0].plot(t[k - 1:], sm, color="tab:purple")
        ax[0].set(title="PPO mean episode return", xlabel="env steps (M)")
        ax[1].plot(t, [r["fail_rate"] for r in rows], color="tab:red", alpha=0.6)
        ax[1].set(title="episode failure rate (tumble / lost)", xlabel="env steps (M)")
        ax[2].plot(t, [r.get("explained_variance", np.nan) for r in rows], color="tab:gray")
        ax[2].set(title="value explained variance", xlabel="env steps (M)")
        save(fig, "train_rl.png")
    p = R / "train_residual.json"
    if p.exists():
        h = json.loads(p.read_text())
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot([r["epoch"] for r in h["history"]], [r["train_mse"] for r in h["history"]], label="train MSE")
        ax.plot([r["epoch"] for r in h["history"]], [r["val_mse"] for r in h["history"]], label="val MSE")
        ax.set(title="Residual inverse-dynamics net", xlabel="epoch", yscale="log")
        ax.legend()
        save(fig, "train_residual.png")


def design_history():
    runs = [("aborted_localizer_v1_history.json", "v1: per-bin unit phasors, 2-D CNN (aborted)"),
            ("aborted_localizer_v2_history.json", "v2: revised v1 (aborted)"),
            ("stopped_localizer_v3_noaug_history.json", "v3 without D4 augmentation (stopped)"),
            ("train_localizer_full.json", "v3 final (D4 augmentation)")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for fn, lab in runs:
        p = R / fn
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        h = h["history"] if isinstance(h, dict) else h
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["val"]["bearing_mae_deg"] for r in h], marker=".", label=lab)
        ax[1].plot(ep, [r["val"]["bearing_median_deg"] for r in h], marker=".", label=lab)
    ax[0].set(title="validation bearing MAE (deg)", xlabel="epoch", yscale="log")
    ax[1].set(title="validation bearing median error (deg)", xlabel="epoch", yscale="log")
    for a in ax:
        a.axhline(90, color="k", ls=":", lw=1)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    save(fig, "localizer_design_history.png")


def localization():
    p = R / "localization_errors.npz"
    if not p.exists():
        return
    e = dict(np.load(p))
    loc = json.loads((R / "localization.json").read_text())
    methods = [m for m in ["srp_phat", "gcc_ls_tdoa", "dl_full", "dl_mag", "dl_anechoic"] if f"{m}_bearing" in e]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in methods:
        b = np.sort(e[f"{m}_bearing"])
        ax[0].plot(b, np.arange(1, len(b) + 1) / len(b), label=LABEL[m], color=COL[m])
        r = np.sort(e[f"{m}_range"])
        if m != "gcc_ls_tdoa":
            ax[1].plot(r, np.arange(1, len(r) + 1) / len(r), label=LABEL[m] if m != "srp_phat" else "matched filter ToF", color=COL[m])
    ax[0].set(xscale="log", xlabel="bearing error (deg)", ylabel="CDF", title="Bearing error CDF (test set)")
    ax[1].set(xscale="log", xlabel="range error (m)", ylabel="CDF", title="Range error CDF (test set)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    save(fig, "loc_cdf.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = list(loc["methods"][methods[0]]["by_snr"].keys())
    x = np.arange(len(bins))
    wdt = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ax[0].bar(x + i * wdt, [loc["methods"][m]["by_snr"][b]["bearing_mae_deg"] for b in bins], wdt, label=LABEL[m], color=COL[m])
        ax[1].bar(x + i * wdt, [loc["methods"][m]["by_snr"][b]["range_mae_m"] for b in bins], wdt, label=LABEL[m], color=COL[m])
    for a, t in zip(ax, ["bearing MAE (deg) vs direct-path SNR (dB)", "range MAE (m) vs direct-path SNR (dB)"]):
        a.set_xticks(x + 0.4 - wdt / 2, bins)
        a.set_title(t)
        a.grid(alpha=0.3, axis="y")
    ax[0].legend(fontsize=7)
    save(fig, "loc_vs_snr.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    groups = [("by_terrain", C.TERRAINS), ("by_channel", C.CHANNEL_NAMES)]
    for a, (key, names) in zip(ax, groups):
        x = np.arange(len(names))
        for i, m in enumerate(methods):
            a.bar(x + i * wdt, [loc["methods"][m][key][nm]["bearing_mae_deg"] for nm in names], wdt, label=LABEL[m], color=COL[m])
        a.set_xticks(x + 0.4 - wdt / 2, names)
        a.set_title(f"bearing MAE (deg) {key.replace('by_', 'by ')}")
        a.grid(alpha=0.3, axis="y")
    ax[0].legend(fontsize=7)
    save(fig, "loc_by_terrain_channel.png")


def closed_loop(channel="A_us"):
    p = R / f"closed_loop_{channel}.json"
    if not p.exists():
        return
    res = json.loads(p.read_text())["results"]
    methods = [m for m in ["oracle", "gcc_track", "dl_track", "dl_pid", "dl_pid_res", "rl"] if m in res]
    terrs = [t for t in C.TERRAINS if all(t in res[m] for m in methods)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    x = np.arange(len(terrs))
    wdt = 0.8 / len(methods)
    for i, m in enumerate(methods):
        for a, key in zip(ax, ["success", "mean_dist", "effort"]):
            mu = [res[m][t]["agg"][key]["mean"] for t in terrs]
            ci = [res[m][t]["agg"][key]["ci95"] for t in terrs]
            a.bar(x + i * wdt, mu, wdt, yerr=ci, capsize=2, label=LABEL[m], color=COL[m])
    for a, t in zip(ax, ["follow success rate", "mean distance to person after 10 s (m)", "control effort mean(u^2)"]):
        a.set_xticks(x + 0.4 - wdt / 2, terrs, rotation=20)
        a.set_title(t)
        a.grid(alpha=0.3, axis="y")
    ax[1].axhline(C.STANDOFF, color="k", ls=":", lw=1)
    ax[0].legend(fontsize=7)
    save(fig, f"closed_loop_bars_{channel}.png")

    tr = dict(np.load(R / f"traj_{channel}.npz"))
    fig, axes = plt.subplots(len(methods), len(terrs), figsize=(3 * len(terrs), 3 * len(methods)))
    for i, m in enumerate(methods):
        for j, t in enumerate(terrs):
            a = axes[i, j]
            rb, ps = tr[f"{m}|{t}|robot"][0], tr[f"{m}|{t}|person"][0]
            a.plot(ps[:, 0], ps[:, 1], color="k", lw=1, label="person")
            a.plot(rb[:, 0], rb[:, 1], color=COL[m], lw=1.2, label="robot")
            a.plot(*rb[0], "o", color=COL[m], ms=4)
            a.plot(*ps[0], "o", color="k", ms=4)
            a.set_aspect("equal", "datalim")
            a.tick_params(labelsize=6)
            if i == 0:
                a.set_title(t, fontsize=9)
            if j == 0:
                a.set_ylabel(LABEL[m], fontsize=8)
    save(fig, f"trajectories_{channel}.png")

    fig, axes = plt.subplots(1, len(terrs), figsize=(3.2 * len(terrs), 3), sharey=True)
    tt = None
    for j, t in enumerate(terrs):
        for m in methods:
            d = tr[f"{m}|{t}|dist"]
            tt = np.arange(d.shape[1]) * C.CTRL_DT
            axes[j].plot(tt, d.mean(0), color=COL[m], lw=1, label=LABEL[m])
        axes[j].axhline(C.STANDOFF, color="k", ls=":", lw=1)
        axes[j].set_title(t, fontsize=9)
        axes[j].set_xlabel("time (s)")
    axes[0].set_ylabel("distance to person (m), mean of 4 episodes")
    axes[0].legend(fontsize=6)
    save(fig, f"distance_curves_{channel}.png")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m in methods:
        dd = np.concatenate([np.asarray(res[m][t]["per_episode"]["mean_dist"], float) for t in terrs])
        dd = np.sort(dd[np.isfinite(dd)])
        ax.plot(dd, np.arange(1, len(dd) + 1) / len(dd), color=COL[m], label=LABEL[m])
    ax.axvline(C.FOLLOW_RADIUS, color="k", ls=":", lw=1)
    ax.set(xscale="log", xlabel="per-episode mean distance after 10 s (m)", ylabel="CDF", title=f"All terrains ({channel})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, f"closed_loop_cdf_{channel}.png")


def dataset_overview():
    p = C.DATA_ROOT / "train" / "labels.npz"
    if not p.exists():
        return
    lab = dict(np.load(p))
    fig, ax = plt.subplots(1, 4, figsize=(15, 3))
    ax[0].hist(np.rad2deg(lab["azimuth"]), 36)
    ax[0].set_title("azimuth (deg)")
    ax[1].hist(lab["range"], 40)
    ax[1].set_title("range (m)")
    ax[2].hist(lab["snr_db"], 60)
    ax[2].set_title("direct-path SNR (dB)")
    cnt = np.zeros((len(C.TERRAINS), len(C.CHANNEL_NAMES)))
    for t in range(len(C.TERRAINS)):
        for c in range(len(C.CHANNEL_NAMES)):
            cnt[t, c] = ((lab["terrain"] == t) & (lab["channel"] == c)).sum()
    im = ax[3].imshow(cnt, cmap="Blues")
    ax[3].set_xticks(range(3), C.CHANNEL_NAMES)
    ax[3].set_yticks(range(len(C.TERRAINS)), C.TERRAINS)
    for t in range(len(C.TERRAINS)):
        for c in range(3):
            ax[3].text(c, t, int(cnt[t, c]), ha="center", va="center", fontsize=7)
    ax[3].set_title("train examples")
    fig.colorbar(im, ax=ax[3])
    save(fig, "dataset_overview.png")


def frontend():
    sys.path.insert(0, str(C.ROOT))
    from bb8dl.acoustics import frontend_gain
    f = np.linspace(10_000, 24_000, 600)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for fc in C.SPK_FC_RANGE:
        ax.plot(f / 1e3, 20 * np.log10(frontend_gain(f, "real", fc)), label=f"realistic chain, speaker fc={fc / 1e3:.1f} kHz")
    for nm, (a, b) in C.CHIRPS.items():
        ax.axvspan(a / 1e3, b / 1e3, alpha=0.15, color="tab:blue" if nm == "us" else "tab:green", label=f"chirp {nm}")
    ax.set(xlabel="frequency (kHz)", ylabel="gain (dB)", ylim=(-70, 3), title="Front-end model (channel B)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "frontend.png")


if __name__ == "__main__":
    training_curves()
    design_history()
    localization()
    for ch in ["A_us", "B_fb"]:
        closed_loop(ch)
    dataset_overview()
    frontend()
