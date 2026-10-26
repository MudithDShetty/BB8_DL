"""Generate LaTeX tables and number macros for the paper from results/*.json. Nothing is typed by hand.

    python scripts/paper_tables.py      # -> paper/generated/*.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bb8dl import config as C  # noqa: E402

R = C.RESULTS
OUT = C.ROOT / "paper" / "generated"
LOC_NAMES = dict(srp_phat="SRP-PHAT", gcc_ls_tdoa="GCC-PHAT LS-TDOA", dl_full="DL (mag.+phase)",
                 dl_mag="DL (magnitude only)", dl_anechoic="DL (no multipath in training)")
CL_NAMES = dict(oracle="Oracle + pursuit", gcc_track="GCC-PHAT + tracker", dl_track="DL + tracker",
                dl_pid="DL + PID", dl_pid_res="DL + PID + residual", rl="RL (PPO)")
macros: dict[str, str] = {}


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def mac(name: str, value) -> None:
    macros[name] = str(value)


def f(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", OUT / name)


def tex_name(s: str) -> str:
    return s.replace("_", r"\_")


def localization_tables():
    loc = load("localization.json")
    if not loc:
        return
    mac("NTest", loc["n_test"])
    rows = []
    for m, nm in LOC_NAMES.items():
        if m not in loc["methods"]:
            continue
        o = loc["methods"][m]["overall"]
        rows.append(f"{nm} & {f(o['bearing_mae_deg'])} & {f(o['bearing_rmse_deg'])} & {f(o['bearing_median_deg'])} & "
                    f"{f(o['bearing_p90_deg'])} & {f(100 * o['frac_bearing_lt_10deg'], 0)} & "
                    f"{f(o['range_mae_m'], 2)} & {f(o['range_rmse_m'], 2)} & {f(o['range_median_m'], 2)} \\\\")
        key = m.replace("_", "")
        mac(f"Loc{key}BMAE", f(o["bearing_mae_deg"]))
        mac(f"Loc{key}BMed", f(o["bearing_median_deg"]))
        mac(f"Loc{key}BRMSE", f(o["bearing_rmse_deg"]))
        mac(f"Loc{key}RMAE", f(o["range_mae_m"], 2))
        mac(f"Loc{key}RMed", f(o["range_median_m"], 2))
        mac(f"Loc{key}Lt", f(100 * o["frac_bearing_lt_10deg"], 0))
    write("tab_loc.tex", "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
          "method & \\multicolumn{5}{c}{bearing error (deg)} & \\multicolumn{3}{c}{range error (m)}\\\\\n"
          "\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}\n"
          " & MAE & RMSE & median & p90 & \\% $<10^\\circ$ & MAE & RMSE & median\\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    methods = [m for m in LOC_NAMES if m in loc["methods"]]
    for key, names, fn in [("by_terrain", C.TERRAINS, "tab_loc_terrain.tex"), ("by_channel", C.CHANNEL_NAMES, "tab_loc_channel.tex")]:
        hdr = " & ".join(tex_name(n) for n in names)
        rows = []
        for m in methods:
            g = loc["methods"][m][key]
            rows.append(f"{LOC_NAMES[m]} & " + " & ".join(f"{f(g[n]['bearing_mae_deg'])} / {f(g[n]['bearing_median_deg'])}" for n in names) + " \\\\")
        write(fn, f"\\begin{{tabular}}{{l{'r' * len(names)}}}\n\\toprule\nmethod & {hdr}\\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        for m in methods:
            for n in names:
                g = loc["methods"][m][key][n]
                mac(f"Loc{m.replace('_', '')}{n.replace('_', '')}BMAE", f(g["bearing_mae_deg"]))
                mac(f"Loc{m.replace('_', '')}{n.replace('_', '')}BMed", f(g["bearing_median_deg"]))
    for m in methods:
        for i, (b, g) in enumerate(loc["methods"][m]["by_snr"].items()):
            L = chr(65 + i)
            mac(f"Loc{m.replace('_', '')}Snr{L}BMAE", f(g["bearing_mae_deg"]))
            mac(f"Loc{m.replace('_', '')}Snr{L}BMed", f(g["bearing_median_deg"]))
            mac(f"Loc{m.replace('_', '')}Snr{L}RMAE", f(g["range_mae_m"], 2))
            if m == methods[0]:
                mac(f"Snr{L}N", f"{g['n']:,}".replace(",", "{,}"))
                mac(f"Snr{L}Bin", b.replace("-99..", "$<$").replace("..99", "+").replace("..", "--"))
    lat = loc.get("latency", {})
    for k, v in lat.items():
        mac("Lat" + "".join(w.capitalize() for w in k.split("_") if w.isalpha()), f(v, 2))


def training_macros():
    for tag in ["full", "mag", "anechoic"]:
        t = load(f"train_localizer_{tag}.json")
        if not t:
            continue
        T = tag.capitalize()
        mac(f"Train{T}Epochs", t["epochs"])
        mac(f"Train{T}Minutes", f(t["train_seconds"] / 60, 0))
        mac(f"Train{T}Params", f"{t['params']:,}".replace(",", "{,}"))
        mac(f"Train{T}NTrain", f"{t['n_train']:,}".replace(",", "{,}"))
        best = min(t["history"], key=lambda r: r["val"]["bearing_mae_deg"] / 10 + r["val"]["range_mae_m"])
        mac(f"Train{T}BestEpoch", best["epoch"])
        if tag == "full":
            mac("GPU", t["device"])
            mac("TorchVersion", t["torch"])
            mac("NVal", t["n_val"])
            mac("BatchSize", t["batch"])
    rl = load("train_rl.json")
    if rl:
        mac("RLSteps", f"{rl['total_timesteps']:,}".replace(",", "{,}"))
        mac("RLHours", f(rl["seconds"] / 3600, 1))
        mac("RLEnvs", rl["n_envs"])
    curve = load("train_rl_curve.json")
    if curve:
        mac("RLFirstReturn", f(curve[0]["ep_return_mean"], 0))
        last = curve[-max(1, len(curve) // 20):]
        mac("RLFinalReturn", f(float(np.mean([r["ep_return_mean"] for r in last])), 0))
    for name, key in [("crashed_rl_run2_curve.json", "RLCrashTwoSteps")]:
        c = load(name)
        if c:
            mac(key, f"{c[-1]['timesteps']:,}".replace(",", "{,}"))
    res = load("train_residual.json")
    if res:
        h = res["history"]
        mac("ResValMSE", f"{h[-1]['val_mse']:.4f}")
        mac("ResValMAE", f"{res['val_mae']:.3f}")
        mac("ResMixerMAE", f"{res['mixer_val_mae']:.3f}")
        mac("ResEpochs", len(h))
        mac("ResNTrain", f"{res['n_train']:,}".replace(",", "{,}"))
    old = load("superseded_residual_v1.json")
    if old:
        mac("ResOldValMSE", f"{old['history'][-1]['val_mse']:.4f}")
    for fn, key in [("aborted_localizer_v1_history.json", "AbortOne"), ("aborted_localizer_v2_history.json", "AbortTwo"),
                    ("stopped_localizer_v3_noaug_history.json", "NoAug")]:
        h = load(fn)
        if h:
            mac(f"{key}Epochs", len(h))
            mac(f"{key}ValMAE", f(h[-1]["val"]["bearing_mae_deg"]))
            mac(f"{key}ValMed", f(h[-1]["val"]["bearing_median_deg"]))
            mac(f"{key}TrainMAE", f(h[-1]["train"]["bearing_mae_deg"]))


def dataset_macros():
    for split, key in [("train", "NTrainSet"), ("val", "NValSet"), ("test", "NTestSet"), ("train_anechoic", "NAnechoicSet")]:
        p = C.DATA_ROOT / split / "labels.npz"
        if p.exists():
            mac(key, f"{len(np.load(p)['azimuth']):,}".replace(",", "{,}"))


def closed_loop_tables():
    for ch in ["A_us", "B_fb"]:
        d = load(f"closed_loop_{ch}.json")
        if not d:
            continue
        res, cfg = d["results"], d["config"]
        chm = ch.replace("_", "")
        mac(f"CL{chm}N", cfg["n"])
        mac(f"CL{chm}Seconds", f(cfg["seconds"], 0))
        methods = [m for m in CL_NAMES if m in res]
        terrs = [t for t in C.TERRAINS if all(t in res[m] for m in methods)]
        for metric, fn, nd, scale in [("success", "succ", 0, 100), ("mean_dist", "dist", 2, 1), ("effort", "effort", 3, 1),
                                      ("loc_bearing_mae", "bmae", 1, 1)]:
            rows = []
            for m in methods:
                cells = []
                allv = []
                for t in terrs:
                    a = res[m][t]["agg"][metric]
                    cells.append("--" if a["mean"] is None else f"{a['mean'] * scale:.{nd}f}")
                    mac(f"CL{chm}{m.replace('_', '')}{fn}{t}", cells[-1])
                    allv += [x for x in res[m][t]["per_episode"][metric] if x is not None]
                overall = float(np.mean(allv)) * scale if allv else None
                rows.append(f"{CL_NAMES[m]} & " + " & ".join(cells) + f" & {f(overall, nd)} \\\\")
                if overall is not None:
                    mac(f"CL{chm}{m.replace('_', '')}{fn}", f(overall, nd))
            write(f"tab_cl_{fn}_{ch}.tex", f"\\begin{{tabular}}{{l{'r' * (len(terrs) + 1)}}}\n\\toprule\n"
                  f"method & {' & '.join(tex_name(t) for t in terrs)} & all\\\\\n\\midrule\n" + "\n".join(rows)
                  + "\n\\bottomrule\n\\end{tabular}\n")
        rows = []
        for m in methods:
            def pooled(k):
                v = [x for t in terrs for x in res[m][t]["per_episode"][k] if x is not None]
                return float(np.mean(v)) if v else None
            ttr = pooled("time_to_reach")
            mk = f"CL{chm}{m.replace('_', '')}"
            mac(mk + "tumbles", f(pooled("tumbles"), 2))
            mac(mk + "within", f(pooled("frac_within") * 100, 0))
            mac(mk + "final", f(pooled("final_dist"), 2))
            mac(mk + "close", f(pooled("too_close") * 100, 1))
            for t in terrs:
                mac(mk + "tumbles" + t, f(res[m][t]["agg"]["tumbles"]["mean"], 2))
                mac(mk + "effort" + t, f(res[m][t]["agg"]["effort"]["mean"], 3))
            rows.append(f"{CL_NAMES[m]} & {f(pooled('frac_within') * 100, 0)} & {f(pooled('final_dist'), 2)} & "
                        f"{f(ttr, 1)} & {f(pooled('path_length'), 1)} & {f(pooled('person_path'), 1)} & "
                        f"{f(pooled('collisions'), 1)} & {f(pooled('tumbles'), 2)} & {f(pooled('too_close') * 100, 1)} \\\\")
        write(f"tab_cl_extra_{ch}.tex", "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
              "method & \\% time within & final dist. & time to reach & path & person path & collision & tumbles & \\% time\\\\\n"
              " & radius & (m) & radius (s) & (m) & (m) & steps & /episode & $<0.4$ m\\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def sanity_macros():
    s = load("sanity.json")
    if not s:
        return
    rb = s["robot"]
    mac("VFlatFull", f(rb["full_forward"]["v"], 2))
    mac("VUphillEightFull", f(rb["uphill_8deg_full"]["v"], 2))
    mac("VGrassFull", f(rb["grass_full"]["v"], 2))
    fe = s["acoustics"]["frontend_inband_db"]
    mac("FeBus", f(fe["B_us(spk_fc=20k)"], 1))
    mac("FeBfb", f(fe["B_fb(spk_fc=20k)"], 1))
    a = s["acoustics"]["anechoic_hiSNR"]["A_us"]
    mac("SanitySRPMed", f(a["srp_median"], 1))
    mac("SanityRangeMed", f(a["range_median"] * 100, 0))


def debug_pid_macros():
    for t in C.TERRAINS:
        d = load(f"debug_pid_{t}.json")
        if not d:
            continue
        mac("DbgN", d["n"])
        mac("DbgSeconds", f(d["seconds"], 0))
        mac("PitchLimit", f(d["pitch_limit_rad"], 2))
        for m, v in d["methods"].items():
            k = "Dbg" + "".join(w.capitalize() for w in m.split("_")) + t
            mac(k + "Pitch", f(v["max_abs_pitch_rad"], 2))
            mac(k + "Sat", f(v["frac_saturated"] * 100, 0))
            mac(k + "Succ", f(v["success"] * 100, 0))


def main():
    sanity_macros()
    debug_pid_macros()
    dataset_macros()
    training_macros()
    localization_tables()
    closed_loop_tables()
    mac("FollowRadius", C.FOLLOW_RADIUS)
    mac("Standoff", C.STANDOFF)
    mac("SuccessFrac", int(C.SUCCESS_FRAC * 100))
    mac("WarmupS", int(C.WARMUP_S))
    body = "% generated by scripts/paper_tables.py from results/*.json; do not edit\n" + \
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n"
    write("numbers.tex", body)


if __name__ == "__main__":
    main()
