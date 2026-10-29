"""Review check for #1093 (Codex ru0XP): the Llama part-1 reading gates only on the simulated-vs-e2 swap null. This compares
the detector against the label-shuffle null the runner already stored: 100 probes per layer trained on labels permuted
within main tasks (`output_end__mean__task_null`, `null_labels` in run 2026-10-12_143129's scores_L*.npz). It mirrors
#36's analyze_q3_attack.py (@ main, lines 286-303): observed AUROC = outer scores, real attack e1 vs real honest; each null
AUROC = that null's scores against its own shuffled labels, on the same runs. Only layers 60 and 70 (part 1's chosen
layers) and only real arms are read: no simulated row, no row-2 held-out layer. It also tests the reading's own detector,
which uses the layer chosen per test fold (60, 70, 60, 60, 70, from reply_break_llama's "layer per test fold"): each run's
outer score, and each null probe's score, at its own fold's layer, with the null labels the same across layers (asserted)
(Codex rvIMF on #1154). That fold-wise line is a fixed-schedule diagnostic. The stored null holds outer scores only, so it can't
repeat the reading's nested per-fold selection. So a conservative bound is added: for each shuffle, the BEST stitched pooled null AUROC over
all 7^5 = 16,807 fold-layer schedules of the seven fitted layers (20, 30, 40, 50, 60, 70, 79), which no per-fold selection rule can exceed. Real arms only (honest vs e1);
no simulated row is read at any layer. Exits nonzero if any of the four does not beat its null's maximum.
  uv run --no-project --with numpy python experiments/2026-10-12_q2_llama_reader/check_label_shuffle_null.py <run_scores_2026-10-12_143129 dir>
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
runs = json.loads((d / "runs.json").read_text())
arm = np.array([r["arm"] for r in runs])
use_base = np.isin(arm, ["real-honest", "real-attack-e1"])
assert not np.isin(arm[use_base], ["sim-attack"]).any()


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    order = s.argsort(kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    u, inv = np.unique(s, return_inverse=True); rr = np.zeros(len(u)); np.add.at(rr, inv, r); r = (rr / np.bincount(inv))[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


FOLD_LAYER = {0: 60, 1: 70, 2: 60, 3: 60, 4: 70}  # reply_break_llama_2026-10-12_143812.json, "layer per test fold"
fold = np.array([int(r["fold"]) for r in runs])
bad = 0
Z = {l: np.load(d / f"scores_L{l}.npz") for l in (60, 70)}
assert np.array_equal(Z[60]["null_labels"], Z[70]["null_labels"], equal_nan=True), "null labels differ across layers"
for layer in (60, 70):
    p = d / f"scores_L{layer}.npz"; z = np.load(p)
    outer = z["output_end__mean__task_outer"]; S = z["output_end__mean__task_null"]; Y = z["null_labels"]
    use = use_base & np.isfinite(outer) & np.isfinite(S).all(1) & np.isfinite(Y).all(1)
    obs = auc(outer[use], (arm[use] == "real-attack-e1").astype(int))
    null = np.array([auc(S[use, j], Y[use, j]) for j in range(S.shape[1])])
    beats = obs > null.max(); bad += not beats
    print(f"L{layer}: runs {int(use.sum())} (honest {int((arm[use] == 'real-honest').sum())}, e1 {int((arm[use] == 'real-attack-e1').sum())}); "
          f"observed AUROC e1 vs honest {obs:.4f}; label-shuffle null n {len(null)}, mean {null.mean():.4f}, p95 {np.percentile(null, 95):.4f}, "
          f"max {null.max():.4f}; p = {(1 + (null >= obs).sum()) / (1 + len(null)):.4f}; beats the null's max: {beats}; "
          f"scores_L{layer}.npz sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
# the fold-wise detector, as the reading uses it
lay = np.array([FOLD_LAYER[f] if f in FOLD_LAYER else -1 for f in fold])
outer = np.where(lay == 60, Z[60]["output_end__mean__task_outer"], Z[70]["output_end__mean__task_outer"])
S = np.where((lay == 60)[:, None], Z[60]["output_end__mean__task_null"], Z[70]["output_end__mean__task_null"]); Y = Z[60]["null_labels"]
use = use_base & (lay > 0) & np.isfinite(outer) & np.isfinite(S).all(1) & np.isfinite(Y).all(1)
obs = auc(outer[use], (arm[use] == "real-attack-e1").astype(int))
null = np.array([auc(S[use, j], Y[use, j]) for j in range(S.shape[1])])
beats = obs > null.max(); bad += not beats
print(f"fold-wise (60, 70, 60, 60, 70): runs {int(use.sum())}; observed AUROC e1 vs honest {obs:.4f}; label-shuffle null n {len(null)}, mean {null.mean():.4f}, "
      f"p95 {np.percentile(null, 95):.4f}, max {null.max():.4f}; p = {(1 + (null >= obs).sum()) / (1 + len(null)):.4f}; beats the null's max: {beats}")
# conservative bound for any per-fold layer selection: per shuffle, the best stitched pooled AUROC over ALL 7^5 = 16,807
# fold-layer schedules (Codex's follow-up on rvIMF), on one common real-arm mask for observed and null scores
import itertools
from concurrent.futures import ProcessPoolExecutor
LAYERS = (20, 30, 40, 50, 60, 70, 79)
ZL = {l: np.load(d / f"scores_L{l}.npz") for l in LAYERS}
for l in LAYERS:
    assert np.array_equal(ZL[l]["null_labels"], Y, equal_nan=True), f"null labels differ at L{l}"
common = use_base & (lay > 0) & np.isfinite(Y).all(1)
for l in LAYERS:
    common &= np.isfinite(ZL[l]["output_end__mean__task_null"]).all(1) & np.isfinite(ZL[l]["output_end__mean__task_outer"])
NS = np.stack([ZL[l]["output_end__mean__task_null"][common] for l in LAYERS])  # [7, runs, 100]
FC = fold[common]; YC = Y[common]
SCHED = list(itertools.product(range(len(LAYERS)), repeat=5))
li = {l: k for k, l in enumerate(LAYERS)}
OS = np.stack([ZL[l]["output_end__mean__task_outer"][common] for l in LAYERS])
obs_c = auc(OS[[li[FOLD_LAYER[f]] for f in FC], np.arange(len(FC))], (arm[common] == "real-attack-e1").astype(int))


def best_for(j):
    cols = np.arange(len(FC)); m = 0.0
    for s in SCHED:
        m = max(m, auc(NS[[s[f] for f in FC], cols, j], YC[:, j]))
    return m


if __name__ == "__main__":
    import multiprocessing  # fork: workers inherit the loaded arrays; spawn would re-run this module's top level
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("fork")) as ex:
        mx = np.array(list(ex.map(best_for, range(NS.shape[2]))))
    beats = obs_c > mx.max(); bad += not beats
    print(f"best-of-16,807-schedules null (bound for any per-fold layer selection), common mask {int(common.sum())} runs: n {len(mx)}, "
          f"mean {mx.mean():.4f}, p95 {np.percentile(mx, 95):.4f}, max {mx.max():.4f}; fold-wise detector on the same mask {obs_c:.4f} beats it: {beats}")
    sys.exit(1 if bad else 0)
