"""Quick check, analysis: split the #36 tool-output detector's per-output simulated − e1 score gap (L63, outer model,
original text, logits) by whether the output is an error and whether it contains a timestamp. $0, local.

Per cell (main task + side task), the mean per-output score of the simulated run minus that of its e1 run, within one
category. The difference between categories uses cells that have both categories in both runs. Intervals: 2,000
resamples of main tasks. The share of each category per arm is reported too: an arm with more error outputs can differ
there for that reason alone.

  python analyze_output_flags.py --flags results/output_flags_<stamp>.json.gz --scores-dir <#36 run_scores dir>
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True)
    ap.add_argument("--scores-dir", required=True)
    args = ap.parse_args()
    with gzip.open(args.flags, "rt") as fh:
        F = json.load(fh)
    d = Path(args.scores_dir)
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    s = np.load(d / "scores_L63.npz")["output_end__step__task_outer"]
    run = t["output_end__run"]
    if not (len(s) == len(run) == F["rows"]):
        raise ValueError(f"rows: scores {len(s)}, tables {len(run)}, flags {F['rows']}")
    arm = np.array([r["arm"] for r in runs])
    sims = [i for i, r in enumerate(runs) if r["arm"] == "sim-attack"]
    e1_of = {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == "real-attack-e1"}
    task = {i: runs[i]["main_task"] for i in range(len(runs))}
    rng = np.random.default_rng(0)
    out = {"inputs": {"flags": Path(args.flags).name, "scores": "scores_L63.npz output_end__step__task_outer"}, "results": {}}
    for name in ("error", "timestamp"):
        flag = np.array(F[name], bool)
        means = {}  # (run, category) -> mean score
        for c in (True, False):
            m = flag == c
            sums = np.bincount(run[m], weights=s[m], minlength=len(runs)); n = np.bincount(run[m], minlength=len(runs))
            means[c] = np.where(n > 0, sums / np.maximum(n, 1), np.nan)
        share = {a: float(flag[np.isin(run, np.flatnonzero(arm == a))].mean()) for a in ("sim-attack", "real-attack-e1", "real-attack-e2", "real-honest")}
        rows = []  # (task, gap if flagged, gap if not flagged)
        for i in sims:
            j = e1_of[runs[i]["cell"]]
            rows.append((task[i], means[True][i] - means[True][j], means[False][i] - means[False][j]))
        tk = np.array([r[0] for r in rows]); gf = np.array([r[1] for r in rows]); gn = np.array([r[2] for r in rows])
        tasks = np.unique(tk); tix = np.searchsorted(tasks, tk)

        def boot(v):
            m = ~np.isnan(v)
            b = []
            for _ in range(2000):
                w = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))[tix]
                b.append((v[m] * w[m]).sum() / w[m].sum())
            return {"cells": int(m.sum()), "mean_logits": float(v[m].mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

        out["results"][name] = {"share_of_outputs": share, f"gap_sim_minus_e1 ({name})": boot(gf), f"gap_sim_minus_e1 (no {name})": boot(gn),
                                "difference_of_gaps": boot(gf - gn)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"output_flags_analysis_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    for name, r in out["results"].items():
        print(name, "share", {k: round(100 * v, 1) for k, v in r["share_of_outputs"].items()})
        for k, v in r.items():
            if k != "share_of_outputs":
                print(f"   {k:34s} {v['mean_logits']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] cells={v['cells']}")
    print("->", path)


if __name__ == "__main__":
    main()
