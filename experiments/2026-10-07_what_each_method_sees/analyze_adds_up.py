"""Does the evidence add up over a run? Run AUROC from only the first k (or a random k) outputs of each run.

For each level (0, 7), split (env = held-out environment, task = main-task folds), method (tfidf, probe) and
comparison (real e1 vs simulated; the control, real e1 vs real e2), on the outputs both methods scored: a run's score
is the mean of the per-output log-odds of its first min(k, n) outputs in transcript order (or a random min(k, n),
`--draws` draws), k in 1, 2, 3, 5, 10, 20, all. 95% intervals by bootstrapping main tasks (first k only).

Against independent evidence: per-output d' = sqrt(2) * Phi^-1(per-output AUROC, each output weighted 1 / its run's
length, so every run counts equally as in random k; the pooled AUROC weights long runs more); if a run's outputs were independent
given its label, a mean of k has d'_k = d' * sqrt(k) and AUROC Phi(d'_k / sqrt(2)) (k = the mean of min(k, n) over
runs). With outputs equicorrelated within a run at rho (the within-arm share of between-run variance), the mean of k
has d'_k = d' * sqrt(k / (1 + (k - 1) rho)), which caps at d' / sqrt(rho).

  uv run python experiments/2026-10-07_what_each_method_sees/analyze_adds_up.py
Writes results/adds_up_<stamp>.json.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from probe_shift.provenance import capture
from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
E1, E2, SIM = "real-attack-e1", "real-attack-e2", "sim-attack"
POS = {"real_vs_sim": (E1, SIM), "null_e1_vs_e2": (E1, E2)}
METHODS = ("tfidf", "probe")


def icc(score, run, label):
    """Within-arm share of a score's variance that sits between runs (one-way ANOVA estimator), averaged over arms."""
    out = []
    for a in (0, 1):
        m = label == a
        _, inv, n = np.unique(run[m], return_inverse=True, return_counts=True)
        x = score[m]
        mu = np.bincount(inv, x) / n
        within = np.sum((x - mu[inv]) ** 2) / (len(x) - len(n))
        between = max(np.var(mu) - within * np.mean(1 / n), 0.0)
        out.append(between / (between + within))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 0], help="0 = all outputs")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--levels", nargs="+", default=["L0", "L7"])
    ap.add_argument("--splits", nargs="+", default=["env", "task"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    T = dict(np.load(HERE / ".run_cache/joined.npz"))
    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__)]), "args": vars(args),
           "joined_sha256": hashlib.sha256((HERE / ".run_cache/joined.npz").read_bytes()).hexdigest(), "results": {}}
    for level in args.levels:
        for split in args.splits:
            for comp, (neg, pos) in POS.items():
                cols = {m: T[f"{m}_{level}_{split}_{comp}"] for m in METHODS}
                rows = np.isin(T["arm"], [neg, pos]) & ~np.isnan(cols["tfidf"]) & ~np.isnan(cols["probe"])
                idx = np.flatnonzero(rows)
                idx = idx[np.lexsort((T["action"][idx], T["step"][idx], T["key"][idx]))]  # transcript order within run
                keys, run, n = np.unique(T["key"][idx], return_inverse=True, return_counts=True)
                start = np.concatenate([[0], np.cumsum(n)[:-1]])
                ordinal = np.arange(len(idx)) - start[run]  # 0-based position among the run's scored outputs
                y_out = (T["arm"][idx] == pos).astype(int)
                y_run = np.zeros(len(keys), int)
                y_run[run] = y_out
                task_run = np.empty(len(keys), object)
                task_run[run] = T["main_task"][idx]
                res = {"outputs": int(len(idx)), "runs": int(len(keys)), "outputs_per_run": {"mean": round(float(n.mean()), 2),
                       "median": float(np.median(n))}, "by_method": {}}
                for m in METHODS:
                    s = cols[m][idx]
                    a1 = roc_auc_score(y_out, s)
                    a1w = roc_auc_score(y_out, s, sample_weight=1.0 / n[run])  # run-balanced, as random k samples
                    d1 = np.sqrt(2) * norm.ppf(a1w)
                    rho = icc(s, run, y_out)
                    r = {"per_output_auroc": round(float(a1), 4), "per_output_auroc_run_balanced": round(float(a1w), 4),
                         "per_output_d": round(float(d1), 4), "icc_within_arm": round(rho, 4),
                         "first_k": {}, "random_k": {}}
                    for k in args.ks:
                        kk = n if k == 0 else np.minimum(k, n)
                        take = ordinal < kk[run]
                        score = np.bincount(run[take], s[take], len(keys)) / kk
                        k_eff = float(kk.mean())
                        ci = auroc_ci(y_run, score, task_run, n_boot=args.bootstrap)
                        a_these = roc_auc_score(y_out[take], s[take])
                        r["first_k"][str(k or "all")] = ci | {
                            "k_eff": round(k_eff, 2), "runs_with_fewer_than_k": int((n < k).sum()) if k else 0,
                            "per_output_auroc_of_these_outputs": round(float(a_these), 4),
                            "independent_from_these_outputs": round(float(norm.cdf(norm.ppf(a_these) * np.sqrt(k_eff))), 4),
                            "independent": round(float(norm.cdf(d1 * np.sqrt(k_eff) / np.sqrt(2))), 4),
                            "equicorrelated": round(float(norm.cdf(d1 * np.sqrt(k_eff / (1 + (k_eff - 1) * rho)) / np.sqrt(2))), 4)}
                        if k == 0:
                            continue
                        aucs = []
                        for d in range(args.draws):  # a random min(k, n) of each run's outputs: random keys, then take the lowest
                            rng = np.random.default_rng(int(hashlib.md5(f"{level}|{split}|{comp}|{k}|{d}".encode()).hexdigest()[:8], 16))
                            u = rng.random(len(idx))
                            rank = np.empty(len(idx), int)
                            o = np.lexsort((u, run))
                            rank[o] = np.arange(len(idx)) - start[run[o]]
                            take = rank < kk[run]
                            aucs.append(roc_auc_score(y_run, np.bincount(run[take], s[take], len(keys)) / kk))
                        r["random_k"][str(k)] = {"auroc_mean": round(float(np.mean(aucs)), 4), "auroc_sd": round(float(np.std(aucs)), 4),
                                                 "draws": args.draws}
                    res["by_method"][m] = r
                    fk = r["first_k"]
                    print(f"{level} {split} {comp:13} {m:5} out {a1:.3f} rho {rho:.3f} | " +
                          " ".join(f"{k}:{fk[k]['auroc']:.3f}/{r['random_k'].get(k, {}).get('auroc_mean', float('nan')):.3f}/"
                                   f"{fk[k]['independent']:.3f}" for k in fk), flush=True)
                out["results"][f"{level}|{split}|{comp}"] = res
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"adds_up_{stamp}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote results/adds_up_{stamp}.json")


if __name__ == "__main__":
    main()
