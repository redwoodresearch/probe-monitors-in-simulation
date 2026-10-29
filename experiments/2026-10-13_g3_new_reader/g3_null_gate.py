"""G3, RULE.md Amendment 3 point 1: the selection-aware label-shuffle null gate, after step 4's fit and before step 5's x.
Imported by g3_x.py, which runs it on the fit's stored scores once step 4's layers are chosen and stops G3 as "unevaluable"
unless it passes. The arithmetic is the Olmo-3.1-32B check's (check_label_shuffle_null_olmo.py @ cd205e33, #1231), on real
arms only, attack run 1 against honest: observed = the outer models' scores (output_end__mean__task_outer); each null
AUROC = that null probe's scores (output_end__mean__task_null) against its own shuffled labels (null_labels), same runs.
  (i)   per layer: the observed outer AUROC against the null's mean, 95th percentile and maximum;
  (ii)  the fold-wise detector (step 4's layer per test fold) against the null stitched on the same schedule;
  (iii) per shuffle, the best stitched null AUROC over all 7^5 = 16,807 fold-layer schedules.
(ii) and (iii) use one common mask: honest and e1 runs with finite outer and null scores at every layer.
Gate: (ii)'s observed AUROC must be above the maximum of (iii) over the shuffles; a tie counts as an exceedance;
p = (1 + exceedances) / (1 + shuffles). Any per-fold layer choice picks one of the 16,807 schedules, so this bounds the
selection; it shows the detector isn't task-level label structure, and doesn't test the forecast.
"""
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HONEST, E1 = "real-honest", "real-attack-e1"
_W = {}  # the (iii) arrays, set before the pool forks so the workers inherit them


def auc(s, y):  # tie-aware rank AUROC (mid-ranks), as the Olmo check
    s, y = np.asarray(s, float), np.asarray(y, int)
    order = s.argsort(kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    u, inv = np.unique(s, return_inverse=True); rr = np.zeros(len(u)); np.add.at(rr, inv, r); r = (rr / np.bincount(inv))[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        raise SystemExit("an AUROC needs both classes")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summ(obs, null):
    exc = int((null >= obs).sum())  # a tie counts as an exceedance
    return {"observed AUROC": obs, "null n": int(len(null)), "null mean": float(null.mean()), "null p95": float(np.percentile(null, 95)),
            "null max": float(null.max()), "exceedances": exc, "p": (1 + exc) / (1 + len(null)), "beats the null max": bool(obs > null.max())}


def _best(j):
    NS, FC, YC, sched = _W["NS"], _W["FC"], _W["YC"], _W["sched"]
    cols = np.arange(len(FC))
    return max(auc(NS[[s[f] for f in FC], cols, j], YC[:, j]) for s in sched)


def null_gate(Z: dict, arm: np.ndarray, fold: np.ndarray, chosen: dict, layers: tuple, n_null: int) -> dict:
    """Z: {layer: the fit's scores_L<layer>.npz, loaded}; chosen: {test fold: step 4's layer}. Returns the record, whose
    "gate" is "pass" or an "unevaluable: ..." reason."""
    Y = Z[layers[0]]["null_labels"]
    if any(not np.array_equal(Z[l]["null_labels"], Y, equal_nan=True) for l in layers) or Y.shape[1] != n_null:
        raise SystemExit(f"the stored null labels differ across layers or are not {n_null} shuffles")
    base = np.isin(arm, [HONEST, E1])
    out = {"rule": "RULE.md Amendment 3, point 1 (this folder)", "shuffles": n_null, "per layer (e1 vs honest)": {}}
    for l in layers:  # (i)
        outer, S = Z[l]["output_end__mean__task_outer"], Z[l]["output_end__mean__task_null"]
        u = base & np.isfinite(outer) & np.isfinite(S).all(1) & np.isfinite(Y).all(1)
        out["per layer (e1 vs honest)"][str(l)] = summ(auc(outer[u], arm[u] == E1), np.array([auc(S[u, j], Y[u, j]) for j in range(n_null)])) | {"runs": int(u.sum())}
    common = base & np.isfinite(Y).all(1)
    for l in layers:
        common &= np.isfinite(Z[l]["output_end__mean__task_null"]).all(1) & np.isfinite(Z[l]["output_end__mean__task_outer"])
    li = {l: k for k, l in enumerate(layers)}
    NS = np.stack([Z[l]["output_end__mean__task_null"][common] for l in layers])  # [layers, runs, shuffles]
    OS = np.stack([Z[l]["output_end__mean__task_outer"][common] for l in layers])
    FC, YC = fold[common], Y[common]; cols = np.arange(len(FC)); pick = np.array([li[chosen[int(f)]] for f in FC])
    obs = auc(OS[pick, cols], arm[common] == E1)  # (ii)
    out["fold-wise detector against the same-schedule null"] = summ(obs, np.array([auc(NS[pick, cols, j], YC[:, j]) for j in range(n_null)]))
    _W.update(NS=NS, FC=FC, YC=YC, sched=list(itertools.product(range(len(layers)), repeat=5)))
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as ex:  # (iii)
        mx = np.array(list(ex.map(_best, range(n_null))))
    r = summ(obs, mx) | {"common mask runs": int(common.sum()), "schedules": len(_W["sched"]), "chosen layer by test fold": {str(f): l for f, l in chosen.items()}}
    out["best-of-16,807-schedules null (the gate)"] = r
    out["gate"] = "pass" if r["beats the null max"] else "unevaluable: the fold-wise detector does not beat the selection-aware label-shuffle null"
    return out
