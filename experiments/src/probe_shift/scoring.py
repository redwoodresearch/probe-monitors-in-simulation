"""AUROC and pairwise accuracy with 95% intervals from bootstrapping main tasks (the Q1 pilot's rule).

Every row belongs to a group (its main task). A bootstrap draw resamples groups with replacement and keeps all
rows of each drawn group, so a task's real and simulated runs, and all its side tasks, move together.

All draws are evaluated at once: a draw is how many times each group was drawn, and AUROC on the resample (rows
repeated) is the weighted Mann-Whitney statistic with ties counted 1/2, which is what `roc_auc_score` computes on the
repeated rows.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

CHUNK = 100  # draws evaluated together; memory is CHUNK x distinct scores floats


def _group_counts(groups, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_boot, groups) times each group is drawn, and each row's group. One `rng.integers` call per draw, in order,
    so the draws are the ones the row-copying bootstrap made."""
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(n_boot)])
    return counts.astype(float), inv


def auroc_ci(y, score, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC of `score` for label 1 over label 0. Rows with a NaN score fail loudly; drop them first."""
    y, score = np.asarray(y).astype(bool), np.asarray(score, float)
    assert not np.isnan(score).any(), "NaN scores: drop and count them before scoring"
    counts, inv = _group_counts(groups, n_boot, seed)
    levels, lv = np.unique(score, return_inverse=True)
    shape = (counts.shape[1], len(levels))  # per group and score level: how many positive (negative) rows
    pos_rows = sp.csr_matrix((np.ones(y.sum()), (inv[y], lv[y])), shape=shape)
    neg_rows = sp.csr_matrix((np.ones((~y).sum()), (inv[~y], lv[~y])), shape=shape)
    vals = []
    for c in range(0, n_boot, CHUNK):
        pos, neg = (pos_rows.T @ counts[c:c + CHUNK].T).T, (neg_rows.T @ counts[c:c + CHUNK].T).T  # draw x level weights
        below = np.cumsum(neg, axis=1) - neg  # negative weight strictly below each level
        den = pos.sum(axis=1) * neg.sum(axis=1)
        vals.append(((pos * (below + 0.5 * neg)).sum(axis=1) / np.where(den > 0, den, 1))[den > 0])  # skip one-class draws
    lo, hi = np.percentile(np.concatenate(vals), [2.5, 97.5])
    return {"auroc": round(float(roc_auc_score(y, score)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n": int(len(y)), "n_main_tasks": int(len(np.unique(groups)))}


def pairwise_accuracy_ci(correct, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of per-pair correctness in [0, 1] (a tie or a split verdict counts 0.5)."""
    correct = np.asarray(correct, float)
    assert not np.isnan(correct).any(), "NaN pair scores: drop and count them before scoring"
    counts, inv = _group_counts(groups, n_boot, seed)
    vals = (counts @ np.bincount(inv, correct, counts.shape[1])) / (counts @ np.bincount(inv, minlength=counts.shape[1]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"accuracy": round(float(correct.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_pairs": int(len(correct)), "n_main_tasks": int(len(np.unique(groups)))}
