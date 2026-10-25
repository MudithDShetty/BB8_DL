"""Score a companion-app live-mode log against fake_mic_stream ground truth.

    python tools/compare_live_log.py logs/companion_udp.jsonl logs/fake_truth.jsonl

Matches each control step to the chirp whose emission sample is nearest the window start.
The robot is stationary in the fake stream, so this measures the live localization path
(packetisation, ring buffer, grid lock, windowing, model) and not closed-loop following.
"""

import json
import sys

import numpy as np

PERIOD, E0 = 2400, 1000

log = [json.loads(x) for x in open(sys.argv[1])]
truth = {r["k"]: r for r in map(json.loads, open(sys.argv[2]))}
ae, re, offs = [], [], []
for r in log:
    if r.get("window_start") is None or r.get("est_az_deg") is None:
        continue
    k = round((r["window_start"] - E0) / PERIOD)
    if k not in truth:
        continue
    t = truth[k]
    offs.append(r["window_start"] - (E0 + k * PERIOD))
    ae.append(abs((r["est_az_deg"] - t["az_deg"] + 180) % 360 - 180))
    re.append(abs(r["est_range_m"] - t["range_m"]))
ae, re = np.array(ae), np.array(re)
out = dict(n_matched=len(ae), n_logged=len(log), window_offset_samples_median=float(np.median(offs)),
           tracked_bearing_err_median_deg=float(np.median(ae)), tracked_bearing_err_mean_deg=float(ae.mean()),
           tracked_range_err_median_m=float(np.median(re)), lost_packets=log[-1].get("lost_packets"))
print(json.dumps(out, indent=1))
