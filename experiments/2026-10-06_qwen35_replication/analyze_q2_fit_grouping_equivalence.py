"""Do grouped Q2 fits reproduce the committed fits exactly (2026-10-07)?

`fit()` in run_q2_alllayers_qwen35.py now runs every (view, training set, seed) of a position and layer group in one
job, loading the layer's packed rows once, instead of one job per view, training set and seed. This refits one view
of a committed features run (default: honest_only of the matched three-way run, both training sets, real labels, 3
readouts, 8 layers: 6 grouped A10G jobs, 48 fits) into a new fits subdirectory next to the committed fits (nothing is
overwritten), and compares each with the committed fit: max |difference| of the test scores and thresholds, flags that
differ, and per fold the chosen C, validation AUROC, threshold and training size. Also records each fit's seconds and
each layer's load seconds, for the cost of grouping. Modal GPU (A10G).

  uv run python experiments/2026-10-06_qwen35_replication/analyze_q2_fit_grouping_equivalence.py
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_q2_alllayers_qwen35 as q2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-run", default="2026-10-06_q2_qwen35_matched_eb014307fd1aab31_as_labeled")
    ap.add_argument("--out-sub", default="threeway")
    ap.add_argument("--view", default="honest_only")
    ap.add_argument("--layers", default="8,16,24,32,40,48,56,63")
    ap.add_argument("--one-combo-per-job", action="store_true",
                    help="the old job layout (one view, training set and seed per job) through the same fit(), to measure refit noise")
    ap.add_argument("--also-compare", default="", help="another fits subdirectory of the same run to compare with, e.g. a grouped check's")
    a = ap.parse_args()
    base = f"{a.features_run}/{a.out_sub}"
    read = lambda p: b"".join(q2.feats_vol.read_file(p))
    cfg = json.loads(read(f"{base}/result.json"))["config"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg["fits_subdir"] = f"fits_{'ungrouped' if a.one_combo_per_job else 'grouping'}_check_{stamp}"
    committed = {(f.get("view", ""), f["position"], f["train_set"], f["layer"], f["seed"]): f
                 for f in map(json.loads, read(f"{base}/fits_index.jsonl").decode().splitlines())}
    layers = [int(x) for x in a.layers.split(",")]
    groups = [layers[i:i + cfg["layers_per_job"]] for i in range(0, len(layers), cfg["layers_per_job"])]
    combos = [(a.view, t, None) for t in cfg["train_sets"]]
    jobs = ([(cfg, pos, g, [c]) for pos in q2.POSITIONS for g in groups for c in combos] if a.one_combo_per_job else
            [(cfg, pos, g, combos) for pos in q2.POSITIONS for g in groups])
    with q2.app.run():
        new = [f for part in q2.fit.starmap(jobs) for f in part]
    rows, fold_keys = [], ("C", "val_auroc", "threshold", "n_train", "val_fpr", "val_fold")
    for f in new:
        old = committed[(f["view"], f["position"], f["train_set"], f["layer"], f["seed"])]
        n = np.load(io.BytesIO(read(f"{base}/{cfg['fits_subdir']}/{f['file']}")))
        o = np.load(io.BytesIO(read(f"{base}/fits/{old['file']}")))
        folds_same = all(all(fn[k] == fo[k] for k in fold_keys) for fn, fo in zip(sorted(f["folds"], key=lambda x: x["fold"]),
                                                                                   sorted(old["folds"], key=lambda x: x["fold"])))
        other = {}
        if a.also_compare:  # the same fit from another check run
            x = np.load(io.BytesIO(read(f"{base}/{a.also_compare}/{f['file']}")))
            other = {f"vs {a.also_compare}": {"max_abs_score_diff": float(np.abs(n["score"].astype(np.float64) - x["score"]).max()),
                                             "max_abs_threshold_diff": float(np.abs(n["threshold"].astype(np.float64) - x["threshold"]).max()),
                                             "flags_differ": int((n["flag"] != x["flag"]).sum())}}
        rows.append({"file": f["file"], "rows": int(len(n["score"])), "same_rows": len(n["score"]) == len(o["score"]), **other,
                     "max_abs_score_diff": float(np.abs(n["score"].astype(np.float64) - o["score"]).max()),
                     "max_abs_threshold_diff": float(np.abs(n["threshold"].astype(np.float64) - o["threshold"]).max()),
                     "flags_differ": int((n["flag"] != o["flag"]).sum()), "folds_identical": folds_same,
                     "seconds": f["seconds"], "layer_load_seconds": f["layer_load_seconds"], "env": f.get("env")})
    same = lambda r: r["max_abs_score_diff"] == 0 and r["max_abs_threshold_diff"] == 0 and r["flags_differ"] == 0  # scores, thresholds, flags
    summary = {"fits": len(rows), "max_abs_score_diff": max(r["max_abs_score_diff"] for r in rows),
               "max_abs_threshold_diff": max(r["max_abs_threshold_diff"] for r in rows),
               "flags_differ": sum(r["flags_differ"] for r in rows), "folds_identical": all(r["folds_identical"] for r in rows),
               "same_rows": all(r["same_rows"] for r in rows), "jobs": len(jobs), "one_combo_per_job": a.one_combo_per_job,
               "identical_fits": sum(same(r) and r["folds_identical"] for r in rows),
               **({f"vs {a.also_compare}": {"identical_fits": sum(same(r[f"vs {a.also_compare}"]) for r in rows),
                                            "max_abs_score_diff": max(r[f"vs {a.also_compare}"]["max_abs_score_diff"] for r in rows),
                                            "flags_differ": sum(r[f"vs {a.also_compare}"]["flags_differ"] for r in rows)}}
                  if a.also_compare else {}),
               "mean_fit_seconds": round(float(np.mean([r["seconds"] for r in rows])), 1),
               "mean_layer_load_seconds": round(float(np.mean([r["layer_load_seconds"] for r in rows])), 1)}
    path = HERE / "results" / f"q2_fit_grouping_equivalence_{stamp}.json"
    path.write_text(json.dumps({"features_run": a.features_run, "out_sub": a.out_sub, "view": a.view, "layers": layers,
                                "fits_subdir": cfg["fits_subdir"], "summary": summary, "fits": rows}, indent=1))
    print(json.dumps(summary))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
