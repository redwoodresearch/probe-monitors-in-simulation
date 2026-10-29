"""POST HOC (#858 review thread PRRT_kwDOU9RNXc6rYd2U; $0, stored per-run scores only): I3's drift statistic
D = AUROC(S2 vs S1) - AUROC(R2 vs R1) depends on label orientation. An AUROC below 0.5 is drift in the other direction
(S2 vs S1 = 0.429 is a drift of 0.071), so a negative D does not by itself mean "more alike". Recompute with the
orientation-free form D_abs = |AUROC(S2 vs S1) - 0.5| - |AUROC(R2 vs R1) - 0.5| (identical to using max(AUROC, 1 - AUROC)).

Rule, fixed before running, otherwise as analyze_reroll.py: 2,000 draws resampling main tasks within environment
(default_rng(20261011)); X_abs = max(0.060, 95th percentile of D_abs under the same 100 task-grouped label shuffles,
default_rng([20261011, k])). "As alike as the real epochs" if D_abs's one-sided 95% upper bound < X_abs; "more drift than
the real epochs" if its one-sided 95% lower bound > 0 and its bootstrap p survives Holm with the residual > 0 test, as
registered; else too few. Both layers.
Folded AUROCs are biased upward under no drift, which the shuffle reference carries. EXPLORATORY; underpowered.

  uv run --no-project --python 3.13 --with numpy --with scikit-learn python \\
      experiments/2026-10-11_claimB_yardsticks/analyze_POSTHOC_reroll_orientation.py <I3 reroll_scores json>
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


def auc(a, b):
    return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def residual(R1, R2, S1, S2):
    real, sim = (R1 - R2) ** 2 / 2, (S1 - S2) ** 2 / 2
    cross = ((S1 - R1) ** 2 + (S1 - R2) ** 2 + (S2 - R1) ** 2 + (S2 - R2) ** 2) / 4
    return (cross - real - sim).mean()


def d_signed(R1, R2, S1, S2):
    return auc(S1, S2) - auc(R1, R2)


def d_abs(R1, R2, S1, S2):
    return abs(auc(S1, S2) - 0.5) - abs(auc(R1, R2) - 0.5)


def analyse(runs: dict) -> dict:
    cells = sorted(runs)
    R1, R2, S1, S2 = (np.array([runs[c][k] for c in cells]) for k in ("R1", "R2", "S1", "S2"))
    env = np.array([runs[c]["environment"] for c in cells])
    task = np.array([runs[c]["main_task"] for c in cells])
    groups = {e: {t: np.flatnonzero((env == e) & (task == t)) for t in sorted(set(task[env == e]))} for e in sorted(set(env))}
    rng, idxs = np.random.default_rng(SEED), []
    for _ in range(DRAWS):
        parts = []
        for e, ts in groups.items():
            names = list(ts)
            parts += [ts[names[j]] for j in rng.choice(len(names), len(names))]
        idxs.append(np.concatenate(parts))
    tasks = sorted(set(task))
    null_abs, null_signed = [], []
    for k in range(N_NULL):
        r = np.random.default_rng([SEED, k])
        fs, fr = (dict(zip(tasks, r.random(len(tasks)) < 0.5)) for _ in range(2))
        sw_s, sw_r = np.array([fs[t] for t in task]), np.array([fr[t] for t in task])
        args = (np.where(sw_r, R2, R1), np.where(sw_r, R1, R2), np.where(sw_s, S2, S1), np.where(sw_s, S1, S2))
        null_abs.append(d_abs(*args))
        null_signed.append(d_signed(*args))
    res_b = [residual(R1[i], R2[i], S1[i], S2[i]) for i in idxs]
    p_res = (1 + sum(b <= 0 for b in res_b)) / (len(res_b) + 1)
    out = {}
    for name, f, null in (("signed (as registered)", d_signed, null_signed), ("orientation-free", d_abs, null_abs)):
        D = f(R1, R2, S1, S2)
        Db = [f(R1[i], R2[i], S1[i], S2[i]) for i in idxs]
        X = max(X_FLOOR, float(np.percentile(null, 95)))
        lo, hi = float(np.percentile(Db, 5)), float(np.percentile(Db, 95))
        p_gt0 = (1 + sum(v <= 0 for v in Db)) / (len(Db) + 1)
        ps = {"residual_gt_0": p_res, "D_gt_0": p_gt0}
        holm, stop = {}, False
        for i, k in enumerate(sorted(ps, key=ps.get)):
            rej = (not stop) and ps[k] <= 0.05 / (len(ps) - i)
            stop = stop or not rej
            holm[k] = bool(rej)
        out[name] = {"D": float(D), "one_sided95": [lo, hi], "X": X, "null_p95": float(np.percentile(null, 95)),
                     "p_D_gt_0": p_gt0, "p_residual_gt_0": p_res, "holm_D_rejects": holm["D_gt_0"],
                     "reading": ("as alike as the real epochs" if hi < X else "more drift than the real epochs" if (lo > 0 and holm["D_gt_0"])
                                 else "too few")}
    out["aurocs"] = {"S2_vs_S1": float(auc(S1, S2)), "R2_vs_R1": float(auc(R1, R2)),
                     "S2_vs_S1_folded": float(abs(auc(S1, S2) - 0.5)), "R2_vs_R1_folded": float(abs(auc(R1, R2) - 0.5))}
    return out


def main():
    raw = SCORES.read_bytes()
    sc = json.loads(raw)
    out = {"label": "POST HOC (#858 PRRT_kwDOU9RNXc6rYd2U): I3 drift with and without label orientation (EXPLORATORY; underpowered)",
           "layers": {k: analyse(v) for k, v in sc["runs"].items()},
           "inputs": {str(SCORES): hashlib.sha256(raw).hexdigest()}, "seed": SEED,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"reroll_orientation_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out["layers"], indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
