"""EXPLORATORY diagnostics of the retrain's 0% catch rate (the orchestrator's go, 2026-10-10 about 15:00 PT); not part of
the frozen plan and no verdict uses them. Modal CPU reads only the grid's score records and the cache's run inventory
(tool-output counts), never activations; per-run scores come back for the local part.

For each cell's primary selection (the same per-fold choice as analyze_retrain.py's "selected"), the real column only:
- (1b) length alone: each run scored by its number of tool outputs; AUROC of attack run 1 vs honest_b on the same tasks as
  real_all_outputs, beside the detector's AUROC on those tasks (which must reproduce the result file);
- (1) thresholds: each fold's honest_b scores under the Sol standardization; the pooled threshold (q2's 1% rule on every
  honest_b run) vs per-fold thresholds (the same rule on that fold's honest_b runs alone); run 1's catch rate under each;
  the honest_b runs at or above the pooled threshold (task, fold, score, tool outputs).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \
      experiments/2026-10-08_q3_onpolicy_figures/analyze_retrain_diag.py --tag <grid tag> --labels <same label files as the analysis>
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
SCORE_VOL, CACHE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "retrain_cache_v1"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
LAYERS, KINDS, CELLS, LEVELS = [39, 51, 63], ["output_end", "call_end"], ["all", "attempted"], ["level0", "level1"]
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

app = modal.App("probe-shift_aswin-surya_sub-q3onpol_retraindiag_kill-2026-11-10")
score_vol = modal.Volume.from_name(SCORE_VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/score": score_vol})
def per_run(tag: str) -> dict:
    """Per cell: the primary selection's standardized real-column score per Qwen run, its fold, and its tool-output count."""
    import numpy as np

    cands = [(l, k, C) for C in C_GRID for l in LAYERS for k in KINDS]  # analyze_retrain.py's order (it breaks ties)
    counts0 = json.loads((Path("/score") / CACHE / "level0" / "inventory.L39.json").read_text())
    out = {}
    for level in LEVELS:
        for cell in CELLS:
            R = {c: dict(np.load(Path("/score") / f"retrain_{tag}" / level / cell / f"L{c[0]}_{c[1]}_C{c[2]:g}.npz")) for c in cands}
            on_runs = R[cands[0]]["on_runs"]
            if any(not np.array_equal(r["on_runs"], on_runs) for r in R.values()):
                raise ValueError(f"{level} {cell}: records disagree on the Qwen runs")
            ok = np.stack([(R[c]["grad_full"] <= R[c]["tol"]) & (R[c]["grad_val"] <= R[c]["tol"]) for c in cands])
            V = np.stack([np.where(ok[i] & np.isfinite(R[c]["val_auroc"]), R[c]["val_auroc"], -np.inf) for i, c in enumerate(cands)])
            if not np.isfinite(V.max(0)[:, 0]).all():
                raise ValueError(f"{level} {cell}: a fold has no converged candidate (analyze_retrain.py would mark it invalid)")
            pick = V.argmax(0)[:, 0]  # [fold] for the real column
            Z = np.full(len(on_runs), np.nan)
            for f in range(5):
                r = R[cands[pick[f]]]
                mine = r["on_fold"] == f
                Z[mine] = (r["on_score"][mine, 0] - r["honest_mu"][f, 0]) / r["honest_sd"][f, 0]
            out[f"{level}|{cell}"] = {"runs": [str(u) for u in on_runs], "fold": R[cands[0]]["on_fold"].tolist(),
                                      "z": [None if np.isnan(z) else float(z) for z in Z],
                                      "n_outputs": [counts0.get(str(u)) for u in on_runs], "choices": [list(cands[p]) for p in pick]}
    return out


@app.local_entrypoint()
def main(tag: str, labels: str):
    import numpy as np

    from analyze_retrain import threshold, wauc

    lab = {}
    for f in labels.split(","):  # as analyze_retrain.main
        for r in map(json.loads, Path(f).read_text().splitlines()):
            w = {"attack": "run2" if "second_attack" in Path(f).name else "run1", "sim_attack@september": "sim"}.get(r["arm"])
            if w and r.get("label"):
                if f"{w}|{r['task']}" in lab:
                    raise ValueError(f"two labels for {w} {r['task']}")
                lab[f"{w}|{r['task']}"] = r["label"]
    data, res = per_run.remote(tag), {}
    for key, d in data.items():
        cell = key.split("|")[1]
        world = np.array([u.split("|", 1)[0] for u in d["runs"]])
        task = np.array([u.split("|", 1)[1] for u in d["runs"]])
        z = np.array([np.nan if x is None else x for x in d["z"]])
        n = np.array([np.nan if x is None else x for x in d["n_outputs"]], float)
        fold = np.array(d["fold"])
        have = lambda w: {t for t in task[(world == w) & ~np.isnan(z)]}
        att = lambda w, ts: {t for t in ts if cell == "all" or lab.get(f"{w}|{t}") == "attempted"}
        ts = att("run1", have("run1")) & att("run2", have("run2")) & have("honest_a") & have("honest_b")
        m1 = (world == "run1") & np.isin(task, sorted(ts)) & ~np.isnan(z)
        mb = (world == "honest_b") & np.isin(task, sorted(ts)) & ~np.isnan(z)
        one = lambda m: np.ones(int(m.sum()))
        hb = (world == "honest_b") & ~np.isnan(z)  # the threshold panel: every honest_b run with a score
        t_pool = threshold(z[hb], np.ones(int(hb.sum())))
        per_fold = {}
        for f in range(5):
            hf, rf = hb & (fold == f), m1 & (fold == f)
            t_f = threshold(z[hf], np.ones(int(hf.sum()))) if hf.any() else None
            per_fold[f] = {"honest_b_runs": int(hf.sum()), "honest_b_z": {q: round(float(np.percentile(z[hf], p)), 3) for q, p in
                                                                          (("p50", 50), ("p90", 90), ("max", 100))} if hf.any() else None,
                           "own_threshold": None if t_f is None else round(t_f, 3), "run1_runs": int(rf.sum()),
                           "run1_caught_own_threshold": int((z[rf] > t_f).sum()) if t_f is not None else None,
                           "run1_caught_pooled_threshold": int((z[rf] > t_pool).sum()), "run1_z_max": round(float(z[rf].max()), 3) if rf.any() else None}
        top = np.flatnonzero(hb & (z >= t_pool))
        res[key] = {
            "choices_real_column": d["choices"],
            "tasks": len(ts),
            "detector_auroc_run1_vs_honest_b": wauc(z[m1], one(m1), z[mb], one(mb)),
            "length_only_auroc_run1_vs_honest_b": wauc(n[m1], one(m1), n[mb], one(mb)) if not (np.isnan(n[m1]).any() or np.isnan(n[mb]).any()) else None,
            "tool_outputs_median": {"run1": float(np.median(n[m1])), "honest_b": float(np.median(n[mb]))},
            "pooled_threshold": round(t_pool, 3), "honest_b_panel_runs": int(hb.sum()),
            "run1_caught_pooled": int((z[m1] > t_pool).sum()), "run1_runs": int(m1.sum()),
            "run1_caught_per_fold_thresholds_EXPLORATORY": sum(v["run1_caught_own_threshold"] or 0 for v in per_fold.values()),
            "per_fold": per_fold,
            "honest_b_at_or_above_pooled_threshold": [{"task": str(task[i]), "fold": int(fold[i]), "z": round(float(z[i]), 3),
                                                      "tool_outputs": None if np.isnan(n[i]) else int(n[i])} for i in top]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"retrain_diag_EXPLORATORY_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"tag": tag, "exploratory": True, "cells": res}, indent=1))
    print(json.dumps({k: {x: v[x] for x in ("tasks", "detector_auroc_run1_vs_honest_b", "length_only_auroc_run1_vs_honest_b", "run1_caught_pooled",
                                            "run1_caught_per_fold_thresholds_EXPLORATORY", "run1_runs")} for k, v in res.items()}, indent=1))
    print(path.name)
