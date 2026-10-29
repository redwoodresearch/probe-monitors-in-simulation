"""Planted format's statistics (RULE.md step 3, Amendment 2.7), local, $0, from run_planted.py's per-task run scores.

Per variant (swapped level 1: cells a, b, c; unswapped level 0: a, c; b not scored) and cell:
- Delta catch = catch(planted) - catch(unplanted) for the second and the first real run at the cell's as-is 1% threshold
  (fixed), over the 183-task cohort, with 2,000 paired task resamples (seed 20261012; tasks drawn with replacement).
- Simulated - planted second real run at 1% with thresholds redrawn (task_weights), the plants and the yardstick, by
  analyze_onpolicy_cells.stats on the cell's run scores with the second real run's column-0 scores replaced by the
  planted ones (the null uses other columns and other arms, so it is unchanged).

  uv run --no-project --python 3.14 --with numpy --with modal python -I analyze_planted.py <planted_scores json> <rec dir> <out json>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_onpolicy_cells as A  # noqa: E402

RES = HERE / "results"
STAGES = {"a": ("stage2_cell_a_POSTHOC_2026-10-13_121124.json", "a_{v}"), "b": ("stage1_cell_b_POSTHOC_2026-10-13_120555.json", "b_{v}"),
          "c": ("stage4_cell_c_POSTHOC_2026-10-13_121906.json", "c_{v}")}
RECS = {("b", "swapped_L1"): "claimc_swapped", ("b", "unswapped_L0"): "retrain_L0", ("c", "swapped_L1"): "c_swapped_L1", ("c", "unswapped_L0"): "c_unswapped_L0"}
TRAIN = {"a": None, "b": "run1", "c": "sim"}


def main(planted: str, rec: str, out: str):
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    P = json.loads(Path(planted).read_text())["variants"]
    lab = json.loads((HERE / "data/claimc_labels_reduced.json").read_text())["labels"]
    f196 = A.frozen_folds(json.loads(A.T196.read_text()))
    tasks = sorted(f196)
    ti = {t: i for i, t in enumerate(tasks)}
    W = [A.task_weights([f196[t] for t in tasks], b) for b in range(1, A.B + 1)]
    cell_a = json.loads((RES / "cell_a_run_scores_POSTHOC_2026-10-13_121016.json").read_text())
    res = {}
    for v, cells in P.items():
        for c, r in cells.items():
            name = f"{c}_{v}"
            if "not scored" in r:
                res[name] = r
                continue
            f, key = STAGES[c]
            st = json.loads((RES / f).read_text())["cells"][key.format(v=v)]
            thr, ncoh = st["at_0.01"]["threshold"], st["cohort_tasks"]
            cohort = sorted({t for t in r["run2"]["per_task"]} & {t for t in r["run1"]["per_task"]} & {t for t in tasks
                            if all(lab.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim"))})
            o = {"threshold": thr, "cohort_tasks": len(cohort)}
            rng = np.random.default_rng(20261012)
            draws = [rng.integers(0, len(cohort), len(cohort)) for _ in range(A.B)]
            for w in ("run2", "run1"):
                u = np.array([r[w]["per_task"][t][0] > thr for t in cohort], float)
                p = np.array([r[w]["per_task"][t][1] > thr for t in cohort], float)
                dd = p - u
                bs = [float(dd[i].mean()) for i in draws]
                o[w] = {"catch_unplanted": float(u.mean()), "catch_planted": float(p.mean()), "delta": float(dd.mean()),
                        "delta_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "flags_up": int((dd > 0).sum()),
                        "flags_down": int((dd < 0).sum()), "runs_changed": r[w]["runs_changed"]}
            # simulated - planted second real run at 1%, thresholds redrawn
            if c == "a":
                rs = dict(cell_a["variants"][v]["run_scores"])
                for t, (_, pl) in r["run2"]["per_task"].items():
                    rs[f"run2|{t}"] = pl
                d = {"runs": sorted(rs), "Z": np.array([[rs[k]] for k in sorted(rs)]), "Y": None}
            else:
                d = A.select(Path(rec) / RECS[(c, v)])
                row = {u: i for i, u in enumerate(d["runs"])}
                for t, (_, pl) in r["run2"]["per_task"].items():
                    if f"run2|{t}" in row and np.isfinite(d["Z"][row[f"run2|{t}"], 0]):
                        d["Z"][row[f"run2|{t}"], 0] = pl
            s = A.stats(d, lab, tasks, ti, W, TRAIN[c])
            p1 = s["at_0.01"]
            o["sim_minus_planted_run2"] = {"estimate": p1["sim_minus_run2"], "ci95": p1["sim_minus_run2_interval"]["ci95"],
                                           "as_is_estimate": st["at_0.01"]["sim_minus_run2"], "plants": p1["plants"], "reading": s["reading_at_1pct"],
                                           "cohort_tasks_stats": s["cohort_tasks"], "stage_cohort": ncoh}
            res[name] = o
    Path(out).open("x").write(json.dumps({"label": "POST HOC (RULE.md step 3, Amendment 2.7)", "git": g("rev-parse", "HEAD").strip(),
                                          "planted_scores_sha256": hashlib.sha256(Path(planted).read_bytes()).hexdigest(), "cells": res,
                                          "written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")}, indent=1, default=float))
    pct = lambda x: "  n/a" if x is None else f"{100 * x:6.2f}"  # noqa: E731
    for name, o in res.items():
        if "not scored" in o:
            print(f"== {name}: NOT SCORED ({o['not scored']})")
            continue
        print(f"== {name} (threshold {o['threshold']:.6g}; cohort {o['cohort_tasks']})")
        for w in ("run2", "run1"):
            x = o[w]
            print(f"  {w}: catch unplanted {pct(x['catch_unplanted'])} planted {pct(x['catch_planted'])} | delta {pct(x['delta'])} [{pct(x['delta_ci95'][0])},{pct(x['delta_ci95'][1])}] | "
                  f"flags up {x['flags_up']} down {x['flags_down']} | runs changed {x['runs_changed']}")
        s = o["sim_minus_planted_run2"]
        print(f"  sim - planted run2 @1%: {pct(s['estimate'])} [{pct(s['ci95'][0])},{pct(s['ci95'][1])}] (as-is {pct(s['as_is_estimate'])}) | "
              f"plants {', '.join(k + (' detected' if v['detected'] else ' NOT detected') for k, v in s['plants'].items())} | reading {s['reading']}")
    print("written", Path(out).name)


if __name__ == "__main__":
    main(*sys.argv[1:4])
