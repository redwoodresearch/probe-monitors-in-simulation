"""Repaired cross-pipeline diagnostic's numbers (RULE.md Amendment 2.3), local, $0. Threshold-free, no repair-effect reading.

Per swapped cell (a, b, c), on the claim C cohort's tasks that have identity-matched reader-style runs in all three arms:
AUROC (P(score_x > score_y) + half ties) of fixed simulated vs second real run, and of first vs second real run, inside
the reader-style dump; beside it, the replay pipeline's as-is simulated vs second real run on the same tasks (stage
run scores). 2,000 task resamples (task_weights, seed [0, k]) for intervals.

  uv run --no-project --python 3.14 --with numpy --with modal python -I analyze_repaired_diag.py <repaired_diag_scores json> <rec dir> <out json>
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


def main(diag: str, rec: str, out: str):
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    Dg = json.loads(Path(diag).read_text())
    lab = json.loads((HERE / "data/claimc_labels_reduced.json").read_text())["labels"]
    f196 = A.frozen_folds(json.loads(A.T196.read_text()))
    tasks = sorted(f196)
    W = [A.task_weights([f196[t] for t in tasks], b) for b in range(1, A.B + 1)]
    ti = {t: i for i, t in enumerate(tasks)}
    cell_a = json.loads((HERE / "results/cell_a_run_scores_POSTHOC_2026-10-13_121016.json").read_text())["variants"]["swapped_L1"]["run_scores"]
    replay = {"a": cell_a}
    for c, d in (("b", "claimc_swapped"), ("c", "c_swapped_L1")):
        s = A.select(Path(rec) / d)
        replay[c] = {u: float(z) for u, z in zip(s["runs"], s["Z"][:, 0]) if np.isfinite(z)}
    res = {"identity": Dg["identity"], "coverage": Dg["coverage"], "cells": {}}
    for c in ("a", "b", "c"):
        rs = Dg["run_scores"][c]
        ts = sorted(t for t in tasks if all(lab.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim"))
                    and all(f"{w}|{t}" in rs for w in ("run1", "run2", "sim_fixed")) and all(f"{w}|{t}" in replay[c] for w in ("run2", "sim")))
        idx = np.array([ti[t] for t in ts])

        def auc(src, x, y, wt):
            p, n = np.array([src[f"{x}|{t}"] for t in ts]), np.array([src[f"{y}|{t}"] for t in ts])
            return A.wauc(p, wt[idx], n, wt[idx])
        one = np.ones(len(tasks))
        o = {"tasks": len(ts)}
        for name, src, x, y in (("reader_fixed_sim_vs_run2", rs, "sim_fixed", "run2"), ("reader_run1_vs_run2", rs, "run1", "run2"),
                                ("replay_as_is_sim_vs_run2", replay[c], "sim", "run2")):
            o[name] = {"auroc": auc(src, x, y, one), "ci95": A.interval([auc(src, x, y, w) for w in W])["ci95"]}
        res["cells"][c] = o
    Path(out).open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 2.3): cross-pipeline diagnostic, no repair-effect reading",
                                          "git": g("rev-parse", "HEAD").strip(), "diag_sha256": hashlib.sha256(Path(diag).read_bytes()).hexdigest()} | res, indent=1))
    print("identity", json.dumps(res["identity"]), "| coverage", json.dumps(res["coverage"]))
    for c, o in res["cells"].items():
        print(f"== cell {c} ({o['tasks']} tasks): " + " | ".join(f"{k} {o[k]['auroc']:.3f} [{o[k]['ci95'][0]:.3f}, {o[k]['ci95'][1]:.3f}]" for k in o if k != "tasks"))
    print("written", Path(out).name)


if __name__ == "__main__":
    main(*sys.argv[1:4])
