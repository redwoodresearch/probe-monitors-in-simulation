"""Overlap800-only G3 y (RULE.md here, item 1). POST HOC, descriptive, $0. Per-cell R and S are G3's committed catches
(the x and y records at --ref); the cells' simulated batch comes from #36's pinned runs.json; the bootstrap is G3's
(2,000 draws, numpy default_rng(20261012), main tasks resampled with replacement, cells weighted by their task's count).
Gate first: the full 939 cells must reproduce G3's y and its interval exactly.

  uv run --no-project --python 3.12 --with numpy python -I experiments/2026-10-13_overlap800_rescore/rescore_g3_o800.py \\
      --ref 6daf17d6 --runs <G3 fit runs.json, sha256 5c987d6a…> > experiments/2026-10-13_overlap800_rescore/results/g3_o800_<stamp>.json
"""
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

G3 = "experiments/2026-10-13_g3_new_reader/results/"
RUNS36 = ("c24106ca", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json",
          "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")
G3_RUNS_SHA = "5c987d6a79315344c76ccad0f71c91a4150e762952109838fdbed43819872a00"


def git_show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def interval(R, S, cells, task_of, seed=20261012, B=2000):
    tasks = sorted({task_of[c] for c in cells}); tix = np.array([tasks.index(task_of[c]) for c in cells])
    g = np.random.default_rng(seed); ys = []
    for _ in range(B):
        w = np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks))[tix].astype(float)
        ys.append(100 * (w @ (R - S)) / w.sum())
    return [float(v) for v in np.percentile(ys, [2.5, 97.5])], len(tasks)


def main(a):
    x = json.loads(git_show(a.ref, G3 + "g3_x_2026-10-13_112514.json")); y = json.loads(git_show(a.ref, G3 + "g3_y_2026-10-13_124203.json"))
    r36b = git_show(RUNS36[0], RUNS36[1])
    g3runs_b = open(a.runs, "rb").read()
    if hashlib.sha256(r36b).hexdigest() != RUNS36[2] or hashlib.sha256(g3runs_b).hexdigest() != G3_RUNS_SHA:
        sys.exit("a runs.json differs from its pin")
    r36, g3runs = json.loads(r36b), json.loads(g3runs_b)
    sim_batch = {r["cell"]: r["key"].split("_")[0] for r in r36 if r["arm"] == "sim-attack"}
    task_of = {r["cell"]: r["main_task"] for r in g3runs if r["arm"] == "real-attack-e2"}
    cells = sorted(x["caught, real e2 by cell (R)"])
    R = np.array([x["caught, real e2 by cell (R)"][c] for c in cells]); S = np.array([y["caught, simulated by cell (S)"][c] for c in cells])
    full_y = 100 * (R.mean() - S.mean()); full_ci, n_tasks_full = interval(R, S, cells, task_of)
    if abs(full_y - y["y = R - S (points)"]) > 1e-9 or not np.allclose(full_ci, y["y 95% (points)"], rtol=0, atol=1e-9):
        sys.exit("gate failed: the full 939 cells do not reproduce G3's y or its interval")
    keep = np.array([sim_batch[c] == "overlap800" for c in cells])
    cells_o = [c for c, k in zip(cells, keep) if k]
    yo = 100 * (R[keep].mean() - S[keep].mean()); ci_o, n_tasks_o = interval(R[keep], S[keep], cells_o, task_of)
    yb = 100 * (R[~keep].mean() - S[~keep].mean())
    out = {"rule": "experiments/2026-10-13_overlap800_rescore/RULE.md (7b25c521), item 1", "inputs ref": a.ref,
           "script sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "batch counts (sim)": {b: int(sum(1 for c in cells if sim_batch[c] == b)) for b in ("base200", "overlap800")},
           "gate (939 reproduces G3)": True,
           "full 939": {"R": 100 * R.mean(), "S": 100 * S.mean(), "y": full_y, "y 95%": full_ci, "main tasks": n_tasks_full},
           "overlap800 cells": {"cells": int(keep.sum()), "R": 100 * R[keep].mean(), "S": 100 * S[keep].mean(), "y": yo,
                                "y 95%": ci_o, "main tasks": n_tasks_o},
           "base200 cells (beside)": {"cells": int((~keep).sum()), "R": 100 * R[~keep].mean(), "S": 100 * S[~keep].mean(), "y": yb},
           "change in y, overlap800 - full (points)": yo - full_y, "flag (|change| > 5)": bool(abs(yo - full_y) > 5)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--runs", required=True)
    main(ap.parse_args())
