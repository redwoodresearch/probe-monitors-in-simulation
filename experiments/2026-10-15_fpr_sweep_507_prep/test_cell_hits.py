"""Toy check for the adds' cell_hits_draw0 weights (09 @ 002c1824b; no inputs, no Modal): the draw-0 effective cell weight w_c that row() writes
(per-origin row: W0[main] / n_cohort_cells(main); all-origins row: W0[main] * 1/n_cells_all_origins(main)) reproduces row()'s tpr() at draw 0
exactly as the in-job assert reads it (sum of w_c over caught cells / sum w_c, from the '0'/'1' string), within 1e-12. Exit 1 on any failure.

  <modal env python> experiments/2026-10-15_fpr_sweep_507_prep/test_cell_hits.py
"""

import sys

import numpy as np

from run_fprsweep_507 import draws_main

bad = []
env = {"A": "e1", "B": "e1", "C": "e2", "D": "e2"}
units, W = draws_main(sorted(env), {m: (e,) for m, e in env.items()}, seed=[20261015, 41])
ucol = {m: i for i, m in enumerate(units)}
coh = ["b1", "a1", "d1", "b2", "a2"]  # cohort order as row() holds it (not sorted)
main_of = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "d1": "D"}
rng = np.random.default_rng([20261015, 98])
hit = rng.random((W.shape[0], len(coh))) < 0.5

mcols = sorted({ucol[main_of[c]] for c in coh}, key=int)  # row(): per-origin tpr
grp = [np.array([i for i, c in enumerate(coh) if ucol[main_of[c]] == m]) for m in mcols]
hm = np.stack([hit[:, g].mean(1) for g in grp], 1)
t_orig = (W[:, mcols] * hm).sum(1) / W[:, mcols].sum(1)
ncm = {m: len(g) for m, g in zip(mcols, grp)}
w_orig = np.array([W[0, ucol[main_of[c]]] / ncm[ucol[main_of[c]]] for c in coh])

n_all = {"A": 3, "B": 2, "D": 1}  # all-origins row: a main's cells over every origin (A has a third cell elsewhere)
ccol = np.array([ucol[main_of[c]] for c in coh])
wcell = np.array([1.0 / n_all[main_of[c]] for c in coh])
Wc = W[:, ccol] * wcell
t_mg = (Wc * hit).sum(1) / Wc.sum(1)
w_mg = W[0, ccol] * wcell

for tag, t, w in (("per-origin", t_orig, w_orig), ("all-origins", t_mg, w_mg)):
    cs = sorted(range(len(coh)), key=lambda i: coh[i])
    s = "".join("1" if hit[0, i] else "0" for i in cs)
    ws = [float(w[i]) for i in cs]
    rc = sum(x for x, ch in zip(ws, s) if ch == "1") / sum(ws)
    if not abs(rc - t[0]) <= 1e-12:
        bad.append(f"{tag}: {rc!r} vs tpr {t[0]!r}")
    if len(s) != len(coh) or set(s) - {"0", "1"}:
        bad.append(f"{tag}: string shape")
print({"pass": not bad, "bad": bad})
sys.exit(1 if bad else 0)
