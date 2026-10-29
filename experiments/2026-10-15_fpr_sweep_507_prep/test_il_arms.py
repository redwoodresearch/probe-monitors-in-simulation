"""Lean checks for il_arms / estimator (B): world parsing (incl. negatives), the CUT key format, and the (B) per-cell arithmetic (equals (A) where a
cell has no extra run; equals q2's comps() design (c) on complete five-arm cells; tpr linearity).  python test_il_arms.py"""
import sys

import numpy as np

sys.path.insert(0, ".")
import il_arms as L  # noqa: E402

ok = []
for f, w in (("il311_q1_run3_s1_v1__attack", "run3"), ("ilnm_s4d_sim2_v1_try2__sim_attack", "sim2"), ("ilold_oldsel_run3_v1__attack", "run3"),
             ("ilnm_s4rgap_sim2_v1__sim_attack", "sim2")):
    ok.append(L.world_of(f) == w)
for bad in ("il311_q1_run3_s1_v1__sim_attack", "ilnm_s5_sim2_v1__attack", "su311_q1_main_v1__attack"):
    try:
        L.world_of(bad)
        ok.append(False)
    except SystemExit:
        ok.append(True)
ok.append(L.cut_key("runs/2026-10-14_il_old/attack/t74_tiktok/x.eval") == "2026-10-14_il_old/attack/t74_tiktok/x.eval#e0")
ok.append(L.cut_key("runs/runs/a.eval") == "runs/a.eval#e0")  # exactly one leading 'runs/' removed

ok.append(L.cell_of_task("t444_file_indexing_redraw1", "t444_file_indexing") == "t444_file_indexing")
ok.append(L.cell_of_task("t444_file_indexing", "t444_file_indexing") == "t444_file_indexing")
for bad_task in ("t444_file_indexing_redrawX", "t445_file_indexing", "t444_file_indexing_r2"):  # never a raw-string join
    try:
        L.cell_of_task(bad_task, "t444_file_indexing")
        ok.append(False)
    except SystemExit:
        ok.append(True)
man_ = {"files": [{"world": "run3", "task": "t444_file_indexing_redraw1", "launch_id": "L", "path": "p", "sha256": "h"}]}
ok.append(L._eval_of(man_, "run3", "t444_file_indexing", "runs/L")["path"] == "p")
try:
    L._eval_of({"files": [{"world": "run3", "task": "t444_file_indexing_redrawX", "launch_id": "L", "path": "p", "sha256": "h"}]}, "run3", "t444_file_indexing", "L")
    ok.append(False)
except SystemExit:
    ok.append(True)

rng = np.random.default_rng(0)
B, n = 4, 9
h = {a: rng.random((B, n)) > 0.5 for a in ("sim", "sim2", "run1", "run2", "run3")}
av = {"sim2": (rng.random(n) > 0.4).astype(float), "run3": (rng.random(n) > 0.4).astype(float)}
hs = (h["sim"].astype(float) + np.where(av["sim2"] > 0, h["sim2"], False).astype(float)) / (1 + av["sim2"])
hr = (h["run1"].astype(float) + h["run2"] + np.where(av["run3"] > 0, h["run3"], False).astype(float)) / (2 + av["run3"])
none_ = (av["sim2"] == 0) & (av["run3"] == 0)
ok.append(np.allclose((hs - hr)[:, none_], (h["sim"] - (h["run1"].astype(float) + h["run2"]) / 2)[:, none_]))
c5 = (av["sim2"] > 0) & (av["run3"] > 0)
q2c = (h["sim"].astype(float) + h["sim2"]) / 2 - (h["run1"].astype(float) + h["run2"] + h["run3"]) / 3
ok.append(np.allclose((hs - hr)[:, c5], q2c[:, c5]))
both_hit = h["sim"] & h["sim2"] & (av["sim2"] > 0)
ok.append(np.all(hs[both_hit] == 1.0))  # two sim hits average to 1.0, never 0.5 (the boolean-addition bug Codex r2 found)
W = rng.random((B, n)) + 0.1
tpr = lambda x: (W * x).sum(1) / W.sum(1)  # noqa: E731
av0 = np.zeros(n)
hr0 = (h["run1"].astype(float) + h["run2"] + 0.0) / (2 + av0)
ok.append(np.allclose(tpr(hr0), (tpr(h["run1"]) + tpr(h["run2"])) / 2))
print({"passed": sum(ok), "of": len(ok)})
sys.exit(0 if all(ok) else 1)
