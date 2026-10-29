"""Synthetic check for the all-origins row, committed with the code (Append 39 am.1 §5; 09 @ e3a459bd0 item 1): a main in two origins keeps total
weight one, each cell keeps its own origin's calibration (its hit), and every origin uses one multiplicity per canonical main. Toy data only (no
inputs, no Modal): row()'s merged tpr / D arithmetic and pool_reduce_n's X = sum_o X_o D_o / sum_o D_o against the direct main-level average;
merge_counts on fake cohort passes. Exit 1 on any failure.

  <modal env python> experiments/2026-10-15_fpr_sweep_507_prep/test_allorigins_merge.py
"""

import sys

import numpy as np

import run_peraction311 as R
from run_fprsweep_507 import draws_main

bad = []
env = {"A": "e1", "B": "e1", "C": "e2", "D": "e2"}
units, W0 = draws_main(sorted(env), {m: (e,) for m, e in env.items()}, seed=[20261015, 40])
ucol = {m: i for i, m in enumerate(units)}
O = {"O1": [("a1", "A"), ("b1", "B"), ("b2", "B"), ("d1", "D")], "O2": [("a2", "A"), ("a3", "A"), ("c1", "C")]}  # A is in both origins
n = {"A": 3, "B": 2, "C": 1, "D": 1}
rng = np.random.default_rng([20261015, 99])
hit = {o: (rng.random((W0.shape[0], len(c))) < 0.4).astype(float) for o, c in O.items()}  # each origin's own thresholds -> its own hits


def origin_row(o, nn):  # row(): tpr and D under cfg["merge"] (cell weight W[main] / n_main)
    wc = np.array([1.0 / nn[m] for _, m in O[o]])
    Wc = W0[:, [ucol[m] for _, m in O[o]]] * wc
    return (Wc * hit[o]).sum(1) / Wc.sum(1), Wc.sum(1)


X, Dd = zip(*(origin_row(o, n) for o in O))
pooled = sum(np.where(d > 0, x * d, 0.0) for x, d in zip(X, Dd)) / sum(Dd)  # pool_reduce_n's combination for the all-origins combos
cm = {m: [] for m in units}
for o, cs in O.items():
    for j, (_, m) in enumerate(cs):
        cm[m].append(hit[o][:, j])
direct = sum(W0[:, ucol[m]] * np.mean(cm[m], 0) for m in units) / W0.sum(1)  # one main = one main: its cells from every origin averaged first
if not np.allclose(pooled, direct, rtol=0, atol=1e-12):
    bad.append("merged pooled catch != the direct main-level average")
if not np.allclose(sum(Dd), W0.sum(1), rtol=1e-12):
    bad.append("sum over origins of D_o != sum of W over the union's mains (a shared main does not keep total weight one)")
if not np.allclose(Dd[0][0] + Dd[1][0], len(units)):
    bad.append("draw 0: total weight != number of mains")
# negative control: each origin's own equal-main weighting (per-origin n) counts main A twice; it must differ from the one-main rule
n_own = {o: {m: sum(1 for _, mm in cs if mm == m) for _, m in cs} for o, cs in O.items()}
Xs, Ds = zip(*(origin_row(o, n_own[o]) for o in O))
if np.allclose(sum(np.where(d > 0, x * d, 0.0) for x, d in zip(Xs, Ds)) / sum(Ds), direct, atol=1e-6):
    bad.append("negative control: per-origin weighting matched the one-main rule (the test cannot tell them apart)")
# merge_counts: per label and group, cells by canonical main over the group's origins; a main counted differently in two groups STOPs
jobs = [("p C", {"merge": {"groups": ["all", "paper"]}}, None), ("p C", {"merge": {"groups": ["all", "paper"]}}, None), ("p C", {"merge": {"groups": ["all"]}}, None)]
co = [{"cohort_only": True, "cohort_mains": {c: m for c, m in O["O1"]}}, {"cohort_only": True, "cohort_mains": {c: m for c, m in O["O2"]}},
      {"cohort_only": True, "cohort_mains": {"z1": "Z"}}]
got = R.merge_counts(jobs, co)
if got[0][1]["merge"]["n_cells"] != {"A": 3, "B": 2, "D": 1} or got[1][1]["merge"]["n_cells"] != {"A": 3, "C": 1} or got[2][1]["merge"]["n_cells"] != {"Z": 1}:
    bad.append(f"merge_counts: {[g[1]['merge']['n_cells'] for g in got]}")
co_bad = [co[0], co[1], {"cohort_only": True, "cohort_mains": {"a9": "A"}}]  # main A also in the "all"-only row: groups disagree on A's count
try:
    R.merge_counts(jobs, co_bad)
    bad.append("merge_counts: inconsistent groups did not STOP")
except SystemExit:
    pass
w_ = [co[0], {"label": "p C", "withheld": "WITHHELD (x)"}, co[2]]  # a withheld origin contributes no cells and gets no counts
if "n_cells" in R.merge_counts(jobs, w_)[1][1]["merge"] or R.merge_counts(jobs, w_)[0][1]["merge"]["n_cells"]["A"] != 1:
    bad.append("merge_counts: a withheld origin's cells were counted")
print({"checks": 7, "draws_with_an_origin_at_zero_weight": int(sum((d == 0).sum() for d in Dd)), "failed": bad, "pass": not bad})
sys.exit(1 if bad else 0)
