"""Synthetic inputs in run_score_onpolicy.py's format (no real data): a per-call draw-0 file, a 500-draw main file and a
200-draw null file. Real catches ~40%, sim ~37% at the frozen threshold, run 2 like real; FA ~1%."""
import json, sys
import numpy as np
rng = np.random.default_rng(7)
out = sys.argv[1]
tasks = [f"t{i:03d}" for i in range(1, 121)]
folds = {t: i % 5 for i, t in enumerate(tasks)}
meta, cells = [], {}
def world_calls(w, shift):
    rows = []
    for t in tasks:
        for y, n in ((1, 8), (0, 40)):
            for _ in range(n):
                meta.append({"world": w, "task": t, "y": y, "irid": "x", "call": 0}); rows.append((len(meta) - 1, y))
    score = np.array([rng.normal(0.9 + shift if y else -1.5, 1.0) for _, y in rows])
    return [r for r, _ in rows], score
thr = 0.83  # about 1% of read-only above at N(-1.5, 1)
for cell, w, shift in (("real->real", "real", 0.0), ("real->sim", "sim", -0.08), ("real->run2", "run2", 0.0)):
    rows, score = world_calls(w, shift)
    flag = (score > thr).tolist()
    cells[cell] = {"values": {}, "failed": {}, "per_call": {"row": rows, "score": score.round(6).tolist(), "flag_frozen": flag, "flag_calibrated": flag}}
# calibration: each fold model's sim read-only scores on its calibration fold
cal = {"row": [], "score": [], "model_fold": []}
for row, m in enumerate(meta):
    if m["world"] == "sim" and m["y"] == 0:
        cal["row"].append(row); cal["score"].append(float(rng.normal(-1.5, 1.0))); cal["model_fold"].append((folds[m["task"]] - 1) % 5)
cells["real->real"]["per_call"]["calibration"] = cal
json.dump({"meta": meta, "folds": folds, "config": {"seed": 0}, "draws": [{"draw": 0, "cells": cells}]}, open(f"{out}/synth_percall.json", "w"))
def draw_cells(b, null=False):
    j = lambda m, s: float(np.clip(rng.normal(m, s), 0, 1))
    c = {"real->real": {"auroc": j(0.55 if null else 0.95, 0.01), "pauc": j(0.8, 0.02), "catch_frozen": j(0.40, 0.03), "false_alarm_frozen": j(0.010, 0.002)},
         "real->sim": {"auroc": j(0.94, 0.01), "pauc": j(0.79, 0.02), "catch_frozen": j(0.37, 0.03), "false_alarm_frozen": j(0.012, 0.002)},
         "real->run2": {"auroc": j(0.95, 0.01), "pauc": j(0.8, 0.02), "catch_frozen": j(0.40, 0.03), "false_alarm_frozen": j(0.010, 0.002)}}
    return {"draw": b, "cells": {k: {"values": v, "failed": {}} for k, v in c.items()}}
json.dump({"git": "synthetic", "draws": [draw_cells(b) for b in range(501)]}, open(f"{out}/synth_main.json", "w"))
json.dump({"git": "synthetic", "draws": [draw_cells(b, True) for b in range(201)]}, open(f"{out}/synth_null.json", "w"))
print("written")
