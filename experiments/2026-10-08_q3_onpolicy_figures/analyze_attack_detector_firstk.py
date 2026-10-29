"""The attack detector's matched-length check: each run's score is the mean of its first k tool outputs (in order of
turn, then call index) instead of all of them. $0, local (Python 3.14 reads the .eval logs for the errored runs).
Prediction first: predictions_attack_detector_firstk_2026-10-09.md. k = 27 is the shortest attack run (output_end);
runs with fewer than k outputs use all of theirs. Worlds, errored-run drops, the joint task bootstrap (2,000, seed 0)
and the gaps are analyze_attack_detector.py's.

  uv run --no-project --python 3.14 --with numpy --with scikit-learn python analyze_attack_detector_firstk.py \
      results/attack_detector_per_output_<stamp>.json --run-dir <rerun folder>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_attack_detector import GAPS, WORLDS, aucs, ci, errored_runs

HERE = Path(__file__).parent
KS = [10, 20, 27, None]  # None: every output (the main analysis)


def main(path: str, run_dir: str, cells: list):
    R = json.loads(Path(path).read_text())
    bad = set(errored_runs(Path(run_dir).expanduser()))
    rows = R["rows"]
    out = {"input": Path(path).name, "errored_runs_dropped": sorted(bad), "cells": {}}
    for cell in cells:
        kind = cell.split("_", 1)[1]
        sc = R["scores"][cell]
        per_run = {}
        for r, s in zip(rows, sc):
            if r["kind"] == kind and s is not None:
                per_run.setdefault(f"{r['arm']}|{r['task']}", []).append((r["turn"], r["call_index"], s))
        runs = sorted(r for r in per_run if r not in bad)
        arm = np.array([r.split("|")[0] for r in runs])
        task = np.array([r.split("|")[1] for r in runs])
        tasks = sorted(set(task))
        ti = np.searchsorted(tasks, task)
        rng = np.random.default_rng(0)
        draws = [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(2000)]
        res = {}
        for k in KS:
            s = np.array([np.mean([x[2] for x in sorted(per_run[r])[:k]]) for r in runs])

            def world(wname, w=None):
                a, h = WORLDS[wname]
                m = np.isin(arm, [a, h])
                return aucs((arm[m] == a).astype(int), s[m], None if w is None else w[ti[m]])[0]
            r_k = {w: {"auroc": round(float(world(w)), 4), "ci95": ci([world(w, d) for d in draws])} for w in WORLDS}
            for g, (w1, w0) in GAPS.items():
                r_k[f"gap {g}"] = {"value": round(float(world(w1) - world(w0)), 4), "ci95": ci([world(w1, d) - world(w0, d) for d in draws])}
            res["all" if k is None else f"first {k}"] = r_k
        out["cells"][cell] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attack_detector_firstk_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for cell, res in out["cells"].items():
        for k, r in res.items():
            print(f"{cell} | {k} | real {r['real']['auroc']} {r['real']['ci95']} | sept {r['september']['auroc']} {r['september']['ci95']}"
                  f" | today {r['today']['auroc']} | honest_a {r['real (honest_a replicate)']['auroc']} | gap sept-real {r['gap september - real']}")
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cells", default="L51_output_end,L39_output_end,L63_output_end")
    a = ap.parse_args()
    main(a.path, a.run_dir, a.cells.split(","))
