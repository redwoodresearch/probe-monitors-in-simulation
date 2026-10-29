"""Stub tests for the decisive ΔG pure functions (cutfull_decisive.py): synthetic per-draw catches, no Modal, no real data.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-15_fpr_sweep_507_prep/tests_cutfull/test_decisive_stub.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cutfull_decisive as CD  # noqa: E402

rng = np.random.default_rng(0)
N = 10001
results, fails = [], 0


def check(name, cond, detail=""):
    global fails
    fails += not cond
    results.append({"case": name, "pass": bool(cond), "detail": detail})
    print(("PASS " if cond else "FAIL ") + name, detail)


def out_for(label, sim, rm, r2, r1, ra, k=5):
    return {"label": label, "_draws": {"joint": {"0.01": {"sim@S|primary": sim, "real_mean@B": rm, "run2@B": r2, "run1@B": r1, "run2@A": ra, "D": np.ones(N)}}},
            "pools_at_draw0": {"0.01": {"B": {"T": 1.0, "k": k}, "S": {"T": 1.0, "k": k}, "A": {"T": 1.0, "k": k}}}}


def window(shift):
    o = {}
    for m in CD.MEMBERS:
        base = 0.5 + rng.normal(0, 0.02, N)
        sim = base + shift + rng.normal(0, 0.01, N)
        r1, r2 = base + rng.normal(0, 0.01, N), base + rng.normal(0, 0.01, N)
        o[m] = out_for(m, sim, (r1 + r2) / 2, r2, r1, r2)
    return o


# 1. a clear reduction with residual (cut gap +10 pts, uncut +5 pts) -> "Reduction detected; positive residual remains"
cut_o, unc_o = window(0.10), window(0.05)
cut = {m: CD.member_arrays(o, np) for m, o in cut_o.items()}
unc = {m: CD.member_arrays(o, np) for m, o in unc_o.items()}
est = {(w, m): CD.t_estimable(o)[0] for w, oo in (("cut", cut_o), ("uncut", unc_o)) for m, o in oo.items()}
res, arr = CD.decide(cut, unc, est, np)
check("verdict reduction+residual", res.get("verdict") == "Reduction detected; positive residual remains", json.dumps({k: res.get(k) for k in ("G32", "Gfull", "dG", "verdict")}))
check("group mean: 6 groups, own-real averaged first", len(res["groups"]) == 6 and abs(res["G32"]["point"] - 100 * np.mean([np.mean([cut[m]["delta_full"][0] for m in CD.MEMBERS if CD.GROUP_OF[m] == g]) for g in res["groups"]])) < 1e-9)
check("sensitivity present and labelled", res["sensitivity_own_real_run2"]["label"] == CD.SENS_LABEL)
# 2. no change -> "inconclusive" (Gfull above 0 and dG includes 0 -> "Positive residual remains; change unresolved")
res2, _ = CD.decide(cut, {m: dict(v) for m, v in cut.items()}, est, np)
check("identical windows -> positive residual, change unresolved", res2.get("verdict") == "Positive residual remains; change unresolved", res2.get("verdict"))
# 3. too few finite draws -> unavailable
bad = {m: {k: (v.copy() if hasattr(v, "copy") else v) for k, v in a.items()} for m, a in unc.items()}
bad["FIBS R"]["delta_full"][1:600] = np.nan
res3, _ = CD.decide(cut, bad, est, np)
check("draws < 9,500 -> unavailable", res3["status"] == "unavailable" and "9500" in res3.get("why", ""), res3.get("why"))
# 4. non-estimable T at draw 0 -> unavailable
est4 = dict(est)
est4[("uncut", "Sol (a) R")] = CD.t_estimable(out_for("Sol (a) R", *(np.zeros(N),) * 5, k=0))[0]
res4, _ = CD.decide(cut, unc, est4, np)
check("non-estimable T -> unavailable", res4["status"] == "unavailable" and "non-estimable" in res4["why"], res4.get("why"))
# 5. verdict boundaries (endpoints; equality satisfies none)
check("boundary: dG upper == 0 -> not a reduction", CD.verdict((1.0, 2.0), (-1.0, 0.0)) == "Positive residual remains; change unresolved")
check("boundary: Gfull lower == 0 with dG<0 -> residual unresolved", CD.verdict((0.0, 2.0), (-2.0, -1.0)) == "Reduction detected; residual unresolved")
check("both intervals include 0 -> inconclusive", CD.verdict((-1.0, 1.0), (-1.0, 1.0)) == "inconclusive")
# 6. apply_override: fail closed on a missing S_new arm file; replaces only S_new arms; H_sim roots must match
probes = [{"label": m, "new": f"x/{i}.npz"} for i, m in enumerate(CD.MEMBERS)]
cfg = {"probes": probes, "arms": [{"arm": "su311_q1_main_v1__attack", "files": {p["new"]: {"path": "scaleup/a", "sha256": "c"} for p in probes}},
                                  {"arm": "t196_other__attack", "files": {p["new"]: {"path": "old", "sha256": "o"} for p in probes}}],
       "simh_files": {p["new"]: [{"batch": "simh311_q1", "root": "r1", "path": "simh/c", "sha256": "s"}] for p in probes}}
ov = {"arms": {"su311_q1_main_v1__attack": {p["new"]: {"path": "cutfull/x", "sha256": "u"} for p in probes}},
      "simh": {p["new"]: [{"batch": "simh311_q1", "root": "r1", "path": "cutfull/s", "sha256": "v"}] for p in probes}}
cu, applied = CD.apply_override(cfg, ov, CD.MEMBERS)
check("override replaces S_new arm files only", all(cu["arms"][0]["files"][p["new"]]["path"] == "cutfull/x" for p in probes)
      and all(cu["arms"][1]["files"][p["new"]]["path"] == "old" for p in probes) and cfg["arms"][0]["files"][probes[0]["new"]]["path"] == "scaleup/a")
check("override keeps the cut path beside each H_sim entry", all(a["cut"]["path"] == "simh/c" for a in applied if "simh" in a))
ov_missing = {"arms": {}, "simh": ov["simh"]}
try:
    CD.apply_override(cfg, ov_missing, CD.MEMBERS)
    check("override missing S_new file -> stop", False)
except SystemExit as e:
    check("override missing S_new file -> stop", "uncut files missing" in str(e), str(e)[:100])
ov_roots = {"arms": ov["arms"], "simh": {p["new"]: [{"batch": "simh311_q1", "root": "r2", "path": "cutfull/s", "sha256": "v"}] for p in probes}}
try:
    CD.apply_override(cfg, ov_roots, CD.MEMBERS)
    check("override H_sim roots differ -> stop", False)
except SystemExit as e:
    check("override H_sim roots differ -> stop", "roots differ" in str(e), str(e)[:100])
check("npz key matches the producer's sanitizer", CD.npz_key("Sol (a) R", "S_new", "sim@S|primary") == "Sol_a_R__joint__0.01__S_new__sim_at_S__primary")
# 7. pools_equal: equal keys pass; a different H_sim fails the decisive pools; a different H_A only drops the yardstick
pk = {"cohort_cells": {"n": 272, "sha256": "a"}, "H_B": {"n": 306, "sha256": "b"}, "H_sim": {"n": 305, "sha256": "c"}, "H_A": {"n": 290, "sha256": "d"}}
co = [{"label": m, "pool_keys": dict(pk)} for m in CD.MEMBERS]
eq = CD.pools_equal(co, [{"label": m, "pool_keys": dict(pk)} for m in CD.MEMBERS])
check("pools equal by key -> pass", eq["decisive_pass"] and eq["H_A_pass"])
uo = [{"label": m, "pool_keys": dict(pk, H_sim={"n": 304, "sha256": "x"})} for m in CD.MEMBERS]
check("H_sim differs -> decisive fails (unavailable, no shrink)", not CD.pools_equal(co, uo)["decisive_pass"])
uo2 = [{"label": m, "pool_keys": dict(pk, H_A={"n": 290, "sha256": "y"})} for m in CD.MEMBERS]
e2 = CD.pools_equal(co, uo2)
check("H_A differs -> decisive passes, yardstick off", e2["decisive_pass"] and not e2["H_A_pass"])
check("pool_keys missing -> fail", not CD.pools_equal(co, [{"label": m} for m in CD.MEMBERS])["decisive_pass"])
# 8. t_estimable rejects an infinite T and a missing pool
o_inf = out_for("FIBS R", *(np.zeros(N),) * 5)
o_inf["pools_at_draw0"]["0.01"]["S"]["T"] = float("inf")
check("infinite T -> not estimable", not CD.t_estimable(o_inf)[0])
o_miss = out_for("FIBS R", *(np.zeros(N),) * 5)
del o_miss["pools_at_draw0"]["0.01"]["B"]
check("missing pool -> not estimable", not CD.t_estimable(o_miss)[0])
# 9. yardstick: unavailable unless H_A passes; arrays carry it only when available
r_y, a_y = CD.decide(cut, unc, est, np, yardstick=False)
check("yardstick off -> labelled unavailable, no uncut array", isinstance(r_y["per_member"]["FIBS R"]["yardstick_delta_rr_uncut"], str) and "FIBS_R__delta_rr_uncut" not in a_y)
r_y2, a_y2 = CD.decide(cut, unc, est, np, yardstick=True)
check("yardstick on -> both windows", isinstance(r_y2["per_member"]["FIBS R"]["yardstick_delta_rr_uncut"], float) and "FIBS_R__delta_rr_uncut" in a_y2)
# 10. ood_by_member fails closed; accepts rows keyed by probe without readout
tbl = {"classes": {"ii": "(ii) wording", "iii": "(iii) wording"}, "pending": [],
       "rows": [{"probe": ("Sol (a) (R; a_tight)" if m.startswith("Sol") else m.rsplit(" ", 1)[0] + " (R)"), "class": "x", "class_key": "ii"} for m in CD.MEMBERS]
       + [{"probe": "c_asis (STRESS; not a decisive member)", "class_key": "ii"}]}
o_ = CD.ood_by_member(tbl)
check("OOD table (09's schema) -> all members, caveat from the class wording", len(o_) == len(CD.MEMBERS) and o_["Sol (a) R"]["caveat"] == "(ii) wording")
try:
    CD.ood_by_member(dict(tbl, pending=["FIBS (R)"]))
    check("OOD table with a pending member -> stop", False)
except SystemExit as e:
    check("OOD table with a pending member -> stop", "FIBS R (pending" in str(e))
try:
    CD.ood_by_member(dict(tbl, rows=tbl["rows"][1:]))
    check("OOD table missing a member -> stop", False)
except SystemExit as e:
    check("OOD table missing a member -> stop", "no class/caveat" in str(e))
check("cut-side block shas found by file name", CD.block_shas({"files": [{"path": "scaleup/x/sol/a_tight__avg_top2.npz", "sha256": "a" * 64}]}, "sol/a_tight__avg_top2.npz") == ["a" * 64]
      and CD.block_shas(None, "sol/a_tight__avg_top2.npz") is None)
# 11. boundary list from a step-0 record
s0 = {"runs": [{"key": "L/r#e0", "checks": {"units": {"excluded_boundary": [[3, "output_end", 3, 0]]}}}]}
check("step-0 boundary units listed by key", CD.boundary_units(s0) == [{"run_key": "L/r#e0", "unit": "3|output_end|3|0"}])
check("apply_override marks the uncut cfg", cu.get("cutfull_uncut") is True and "cutfull_uncut" not in cfg)
# 12. uncut sensitivity (09 @ 669ae8e09): C members (no Sol (a)), five groups; Y = run2@B - run1@B; change = uncut - cut
def window_c(shift, yshift):
    o = {}
    for m in CD.MEMBERS_C:
        base = 0.5 + rng.normal(0, 0.02, N)
        r1 = base + rng.normal(0, 0.01, N)
        r2 = r1 + yshift
        o[m] = out_for(m, base + shift, (r1 + r2) / 2, r2, r1, r2)
    return o


cc = {m: CD.member_arrays(o, np) for m, o in window_c(0.10, 0.0).items()}
uc = {m: CD.member_arrays(o, np) for m, o in window_c(0.04, 0.02).items()}
sr, sa = CD.sens_summary(cc, uc, CD.MEMBERS_C, np)
sb = sr["subsets"]["all_members"]
check("uncutsens C: five groups (no Sol (a)), only one subset", sb["groups"] == sorted({CD.group_of(m) for m in CD.MEMBERS_C}) and len(sb["groups"]) == 5
      and list(sr["subsets"]) == ["all_members"])
check("uncutsens: Y_uncut = run2@B - run1@B (2 pts), Y_cut 0", abs(sb["Y_uncut"]["point"] - 2.0) < 1e-9 and abs(sb["Y_cut"]["point"]) < 1e-9, json.dumps(sb.get("Y_uncut")))
exp_s = 100 * np.mean([np.mean([uc[m]["delta_full"][0] for m in CD.MEMBERS_C if CD.group_of(m) == g]) for g in sb["groups"]])
check("uncutsens: S_uncut is the group-first mean", abs(sb["S_uncut"]["point"] - exp_s) < 1e-9)
check("uncutsens: change = uncut - cut", abs(sb["S_change"]["point"] - (sb["S_uncut"]["point"] - sb["S_cut"]["point"])) < 1e-9)
check("uncutsens: no verdict key", "verdict" not in sr and "verdict" not in sb)
sr_r, _ = CD.sens_summary(cut, unc, CD.MEMBERS, np)
check("uncutsens R: six groups plus the without-Sol (a) subset", len(sr_r["subsets"]["all_members"]["groups"]) == 6 and len(sr_r["subsets"]["without_sol_a (like C)"]["groups"]) == 5)
Path(Path(__file__).with_suffix(".result.json")).write_text(json.dumps({"cases": results, "failures": fails}, indent=1))
print("failures:", fails)
sys.exit(1 if fails else 0)
