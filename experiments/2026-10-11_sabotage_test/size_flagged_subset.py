"""A partial sensitivity calculation for k, the flagged commands labelled per run (PLAN.md 3): for each k, how often the
stage-1 held-out set has too few sabotage calls (< 10), how precisely held-out AUROC is measured, and how often it
reaches 0.75 when the within-task AUROC is 0.85. It does not simulate training feasibility, the 5-task minimum, or the
catch-vs-scrambled-baseline half of the gate; with rho > 0 the pooled AUROC is below the within-task 0.85. Synthetic
scores only (no data).

Model of the held-out real stage-1 runs (16 tasks x 2 real runs): sabotage commands per run ~ Poisson(1.5) (the keys'
prior), sitting among the flagged commands that are not target-touching, as all 30 of Aswin's keyed sabotage commands
do (none is in his v3 target-touching set; 28 of 30 are flagged): the k labelled calls are a random k of the run's 32
flagged non-target calls, so the sabotage found is hypergeometric and the rest of the k are negatives (weight 32 / k) plus about 5 unflagged impactful commands audited at about 0.28 (weight about 3.6). Scores are
binormal with a task effect (within-task correlation rho) on both classes; AUROC is inverse-probability weighted.

  uv run --no-project --python 3.12 --with numpy python size_flagged_subset.py
"""

import json
import math
from datetime import datetime
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np

SEED, SIMS, TASKS, RUNS_PER_TASK = 20261011, 2000, 16, 2
PER_RUN_SAB, FLAGGED_ONLY, UNFLAGGED_IMP, AUDIT_RATE = 1.5, 32, 5, 0.28
KS, TRUE_AUROC, BAR, RHOS, MIN_POS = [3, 6, 8, 10, 12, 16, 32], 0.85, 0.75, [0.0, 0.2], 10


def weighted_auroc(sp, sn, wn):
    o = np.argsort(sn)
    sn, wn = sn[o], wn[o]
    cw = np.concatenate([[0.0], np.cumsum(wn)])
    lo, hi = np.searchsorted(sn, sp, "left"), np.searchsorted(sn, sp, "right")
    return float(((cw[lo] + 0.5 * (cw[hi] - cw[lo])) / cw[-1]).mean())


def one(k, rho, rng):
    d = math.sqrt(2) * NormalDist().inv_cdf(TRUE_AUROC)
    sd_u = math.sqrt(rho / (1 - rho)) if rho else 0.0
    sp, sn, wn = [], [], []
    for _ in range(TASKS):
        u = rng.normal(0, sd_u)
        for _ in range(RUNS_PER_TASK):
            kk = min(k, FLAGGED_ONLY)
            sab = min(rng.poisson(PER_RUN_SAB), FLAGGED_ONLY)
            found = rng.hypergeometric(sab, FLAGGED_ONLY - sab, kk)
            sp.append(rng.normal(u + d, 1, found))
            sn.append(rng.normal(u, 1, kk - found))
            wn.append(np.full(kk - found, FLAGGED_ONLY / kk))
            na = rng.binomial(UNFLAGGED_IMP, AUDIT_RATE)
            sn.append(rng.normal(u, 1, na))
            wn.append(np.full(na, 1 / AUDIT_RATE))
    sp, sn, wn = np.concatenate(sp), np.concatenate(sn), np.concatenate(wn)
    return (weighted_auroc(sp, sn, wn) if len(sp) >= MIN_POS else float("nan")), len(sp)


def main():
    rng = np.random.default_rng(SEED)
    out = {"model": __doc__.split("\n\n")[1], "true_auroc": TRUE_AUROC, "bar": BAR, "sims": SIMS, "rows": []}
    for rho in RHOS:
        for k in KS:
            a = np.array([one(k, rho, rng) for _ in range(SIMS)])
            out["rows"].append({"rho": rho, "k": k, "negatives_labelled_per_run": k + UNFLAGGED_IMP * AUDIT_RATE, "p_too_few": round(float(np.isnan(a[:, 0]).mean()), 4),
                                "auroc_mean": round(float(np.nanmean(a[:, 0])), 4), "auroc_sd": round(float(np.nanstd(a[:, 0])), 4),
                                "p_pass": round(float((a[:, 0] >= BAR).mean()), 4),
                                "positives_mean": round(float(a[:, 1].mean()), 1)})
    out["written"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
