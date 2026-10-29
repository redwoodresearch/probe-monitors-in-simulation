"""I3 analysis (PRICES.md item 1's rule and checklist, fixed before the result; EXPLORATORY: test environments; the
AUROC comparison is UNDERPOWERED as an equivalence yardstick, about 27% at 97 cells). Local and free: per-run scores
only (run_reroll_modal.py fetch). Per cell: R1 (real e1), R2 (real e2), S1 (claim B's fixed sim), S2 (the reroll,
same fixes), each the mean of the frozen probe's output_end scores; L63 primary, step A's layer secondary.

1. Run-to-run spread (the sce squared-error decomposition with no call-noise term; probe-score units squared):
   real (R1-R2)^2/2, sim (S1-S2)^2/2, residual = mean over the four (S-R)^2 minus both (estimates the squared
   sim-real mean gap under conditionally independent, exchangeable rerolls; it can be negative). Reading: residual
   lower bound > 0 = a sim-real mean gap beyond run-to-run spread; the sim/real variance ratio's interval above / below
   / spanning 1 = sim runs vary more / less / unresolved.
2. Drift: D = AUROC(S2 vs S1) - AUROC(R2 vs R1). X = max(0.060, 95th percentile of D under 100 task-grouped label
   shuffles: per main task, one coin swaps S1/S2 and another R1/R2 for all its cells). "As alike as the real epochs"
   if D's one-sided 95% upper bound < X; "more drift" if D's one-sided 95% lower bound > 0; else too few.
3. Holm at 0.05 across the two one-sided tests, both with bootstrap p = (1 + draws <= 0) / (draws + 1): residual > 0
   and D > 0. (The shuffle swaps S1/S2 and R1/R2 independently, a stronger null than D <= 0, so it sets X only; its p is
   reported as descriptive. Each AUROC's own shuffle reference, S-swaps for AUROC(S2,S1) and R-swaps for AUROC(R2,R1),
   is reported beside it.)
4. Positive control: S2 plus half the observed mean(S1) - mean(R1) must give D and the residual estimates > 0 (passed);
   their one-sided 95% lower bounds are reported beside them.
Intervals: 2,000 draws resampling main tasks within environment (all four runs of a cell together), default_rng(20261011).

  uv run --no-project --python 3.13 --with numpy --with scikit-learn python \\
      experiments/2026-10-11_claimB_yardsticks/analyze_reroll.py <scores json>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
SCORES = Path(sys.argv[1]).resolve()
SEED, DRAWS, N_NULL, X_FLOOR = 20261011, 2000, 100, 0.060


def comps(R1, R2, S1, S2):
    real, sim = (R1 - R2) ** 2 / 2, (S1 - S2) ** 2 / 2
    cross = ((S1 - R1) ** 2 + (S1 - R2) ** 2 + (S2 - R1) ** 2 + (S2 - R2) ** 2) / 4
    return {"real_var": real.mean(), "sim_var": sim.mean(), "residual": (cross - real - sim).mean(), "total_cross": cross.mean()}


def auc(a, b):  # AUROC of b over a (b labelled positive)
    return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def D_of(R1, R2, S1, S2):
    return auc(S1, S2) - auc(R1, R2)


def analyse(runs: dict) -> dict:
    cells = sorted(runs)
    R1, R2, S1, S2 = (np.array([runs[c][k] for c in cells]) for k in ("R1", "R2", "S1", "S2"))
    env = np.array([runs[c]["environment"] for c in cells])
    task = np.array([runs[c]["main_task"] for c in cells])
    groups = {e: {t: np.flatnonzero((env == e) & (task == t)) for t in sorted(set(task[env == e]))} for e in sorted(set(env))}
    rng = np.random.default_rng(SEED)
    idxs = []
    for _ in range(DRAWS):
        parts = []
        for e, ts in groups.items():
            names = list(ts)
            parts += [ts[names[j]] for j in rng.choice(len(names), len(names))]
        idxs.append(np.concatenate(parts))
    point = comps(R1, R2, S1, S2)
    boot = [comps(R1[i], R2[i], S1[i], S2[i]) for i in idxs]
    ci = {k: [float(np.percentile([b[k] for b in boot], q)) for q in (2.5, 97.5)] for k in point}
    ratio = [b["sim_var"] / b["real_var"] for b in boot if b["real_var"] > 0]
    res_lo = float(np.percentile([b["residual"] for b in boot], 5))
    p_res = (1 + sum(b["residual"] <= 0 for b in boot)) / (len(boot) + 1)
    D = D_of(R1, R2, S1, S2)
    Db = [D_of(R1[i], R2[i], S1[i], S2[i]) for i in idxs]
    tasks = sorted(set(task))
    null, null_s, null_r = [], [], []
    for k in range(N_NULL):
        r = np.random.default_rng([SEED, k])
        fs, fr = (dict(zip(tasks, r.random(len(tasks)) < 0.5)) for _ in range(2))
        sw_s, sw_r = np.array([fs[t] for t in task]), np.array([fr[t] for t in task])
        null.append(D_of(np.where(sw_r, R2, R1), np.where(sw_r, R1, R2), np.where(sw_s, S2, S1), np.where(sw_s, S1, S2)))
        null_s.append(auc(np.where(sw_s, S2, S1), np.where(sw_s, S1, S2)))
        null_r.append(auc(np.where(sw_r, R2, R1), np.where(sw_r, R1, R2)))
    X = max(X_FLOOR, float(np.percentile(null, 95)))
    d_hi, d_lo = float(np.percentile(Db, 95)), float(np.percentile(Db, 5))
    p_D = (1 + sum(v <= 0 for v in Db)) / (len(Db) + 1)
    p_D_shuffle = (1 + sum(v >= D for v in null)) / (N_NULL + 1)
    ps = {"residual_gt_0": p_res, "D_gt_0": p_D}
    order, holm, stop = sorted(ps, key=ps.get), {}, False
    for i, k in enumerate(order):
        rej = (not stop) and ps[k] <= 0.05 / (len(order) - i)
        stop = stop or not rej
        holm[k] = {"p": ps[k], "threshold": 0.05 / (len(order) - i), "rejects": bool(rej)}
    c = 0.5 * (S1.mean() - R1.mean())
    pc_res_est, pc_D_est = float(comps(R1, R2, S1, S2 + c)["residual"]), float(D_of(R1, R2, S1, S2 + c))
    pc_res = float(np.percentile([comps(R1[i], R2[i], S1[i], S2[i] + c)["residual"] for i in idxs], 5))
    pc_D = float(np.percentile([D_of(R1[i], R2[i], S1[i], S2[i] + c) for i in idxs], 5))
    return {"cells": len(cells), "main_tasks": len(tasks),
            "decomposition": {k: {"estimate": float(v), "ci95": ci[k]} for k, v in point.items()},
            "residual_one_sided_lower95": res_lo,
            "residual_reading": "sim-real mean gap beyond run-to-run spread" if holm["residual_gt_0"]["rejects"] else "not shown",
            "sim_over_real_variance": {"estimate": float(point["sim_var"] / point["real_var"]) if point["real_var"] > 0 else None,
                                       "ci95": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]},
            "auroc": {"S2_vs_S1": {"estimate": float(auc(S1, S2)), "shuffle_p95": float(np.percentile(null_s, 95)),
                                   "shuffle_p": (1 + sum(v >= auc(S1, S2) for v in null_s)) / (N_NULL + 1)},
                      "R2_vs_R1": {"estimate": float(auc(R1, R2)), "shuffle_p95": float(np.percentile(null_r, 95)),
                                   "shuffle_p": (1 + sum(v >= auc(R1, R2) for v in null_r)) / (N_NULL + 1)},
                      "S1_vs_R1": float(auc(R1, S1)), "S2_vs_R1": float(auc(R1, S2))},
            "D": {"estimate": float(D), "one_sided95": [d_lo, d_hi], "X": X, "null_p95": float(np.percentile(null, 95)),
                  "shuffle_p_descriptive": p_D_shuffle,
                  "reading": ("as alike as the real epochs" if d_hi < X else "more drift than the real epochs" if (d_lo > 0 and holm["D_gt_0"]["rejects"])
                              else "too few"), "label": "UNDERPOWERED as an equivalence yardstick (about 27% at 97 cells)"},
            "holm": holm,
            "positive_control": {"shift": float(c), "residual": pc_res_est, "D": pc_D_est, "residual_lower95": pc_res,
                                 "D_lower95": pc_D, "passed": bool(pc_res_est > 0 and pc_D_est > 0)}}


def main():
    raw = SCORES.read_bytes()
    sc = json.loads(raw)
    out = {"label": "I3 reroll yardstick (EXPLORATORY; D underpowered as an equivalence yardstick)",
           "inputs": {str(SCORES): hashlib.sha256(raw).hexdigest()}, "seed": SEED, "draws": DRAWS, "null_draws": N_NULL,
           "score_sanity": sc.get("sanity"), "layers": {k: analyse(v) for k, v in sc["runs"].items()},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"reroll_yardstick_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out["layers"], indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
