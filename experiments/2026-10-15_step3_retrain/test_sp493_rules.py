"""Positive / negative tests of sp493_rules.py on the pinned inputs (strings and JSON only; no Modal). Exits nonzero on any failure.

  python3 -I test_sp493_rules.py
"""

import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sp493_rules", HERE / "sp493_rules.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
P = R.load()
rx = R.pinned(R.REGEX_PIN)
fails = []


def check(name, ok):
    if not ok:
        fails.append(name)


# grammar: 09's own positives and hand negatives, verbatim
for k, xs in rx["positives"].items():
    check(f"positive {k}", all(R.kind_of(x) == k for x in xs))
check("hand negatives", all(R.kind_of(x) is None for x in rx["negatives_hand"]))
# 09's sim_attack v2 cases, read from the pinned test file
v2 = re.findall(r"'(sp493_[^']+)' => ([01])", R.pinned(R.REGEX_V2_PIN, raw=True))
check("v2: 14 cases", len(v2) == 14)
check("v2 cases", all((R.kind_of(n) == "sim_attack") == (e == "1") for n, e in v2))
# refuse-list: the _try rule AND d9's names record
cases = {"sp493_a_main_v1_try3__attack": None, "sp493_a_main_v1_try2__attack": "list", "sp493_a_main_v1__attack": "list",
         "sp493_b_main_v1_try3__honest_a": "list", "sp493_b_main_v1_try4__honest_a": "list", "sp493_b_run2_v1_try2__attack": "list",
         "sp493_b_main_v1_try5__honest_a": None, "sp493_b_run2_v1_try3__attack": None, "sp493_c_main_v1_try3__attack": "list", "sp493_c_main_v1_try4__attack": None,
         "sp493_c_run2_v1_try3__attack": "list", "sp493_c_run2_v1_try4__attack": None,
         "sp493_a_main_v1_try5__attack": None, "sp493_g1_sim_s1_v1__sim_attack": "list", "sp493_g1_sim_s1_v1_try2__sim_attack": None,
         "sp493_g2_sim_s3_v1__sim_attack": "list", "sp493_g1_sim_s6_v1__sim_attack": "try", "sp493_g1_sim_v1__sim_attack": None, "sp493_g2_sim_v1__sim_attack": None,
         "sp493_g3_sim_s1_v1__sim_attack": None, "sp493_g4_simh_v1__sim_honest": "list", "sp493_g4_simh_v1_try2__sim_honest": None,
         "sp493_g5_simh_v1__sim_honest": None, "sp493_a_main_v1_dry__attack": "grammar", "sp493_d_main_v1_try3__attack": "grammar",
         "sp493_g6_sim_x_v1__sim_attack": None, "sp493_g7_sim_y_s2_v1_try2__sim_attack": None, "sp493_g3_sim_x_s1_v1__sim_attack": None,
         "sp493_g6_sim_z_v1__sim_attack": "grammar"}
for f, want in cases.items():
    got = R.refuse_folder(f, P)
    check(f"refuse {f}", (got is None) == (want is None))
# cells: t987 by name; outside the list; g4's T196 rule (positive and negative); folds
t196_main_cell = next(k for k, r in P["facts"].items() if r["main_origin"] == "t196" and not k.startswith("t987"))
p311_cell = next(k for k, r in P["facts"].items() if r["main_origin"] == "p311")
pnew_cell = next(k for k, r in P["facts"].items() if r["main_origin"] == "new")
m, f, why = R.admit_cell("run1", t196_main_cell, P)
check("T196-origin cell mapped to its runs36 fold", why is None and f == P["r36"][m] and f >= 0)
check("p311 cell fold -1", R.admit_cell("sim", p311_cell, P)[1] == -1)
check("P_new cell fold -1", R.admit_cell("honest_a", pnew_cell, P)[1] == -1)
check("t987 refused by name", R.admit_cell("run1", "t987_model_registry", P)[2] == R.T987_REASON)
check("outside the list refused", R.admit_cell("run1", "t000_not_a_cell", P)[2] is not None)
check("b_run2 t615_r2 refused by key", R.admit_cell("run2", "t615_media_processing_r2", P)[2] is not None)
t196_mains_in_facts = {(r["env"], r["main"]) for r in P["facts"].values() if r["main_origin"] == "t196"}
g4_yes = next(k for k, c in P["t196"].items() if (c["env"], c["main"]) in t196_mains_in_facts)
g4_no = next((k for k, c in P["t196"].items() if (c["env"], c["main"]) not in t196_mains_in_facts), None)
check("g4 admits a T196 key of an S_paper493 t196 main", R.admit_cell("sp493_t196", g4_yes, P)[2] is None)
check("g4 refuses another T196 key", g4_no is None or R.admit_cell("sp493_t196", g4_no, P)[2] is not None)
check("g4 refuses a non-T196 key", R.admit_cell("sp493_t196", p311_cell, P)[2] is not None)
# held-out gate and its controls
rows = [R.admit_cell("run1", k, P)[:2] for k in P["facts"] if not k.startswith("t987")]
g = R.heldout_gate(rows, P)
check("held-out gate passes with all controls", g["pass"] and all(g["controls"].values()))
check("held-out gate catches a wrong fold", bool(R.heldout_violations([(m, (f + 1) % 5)], P)))
# census, one-set assert, sample_error identity
c = R.census({"S_paper493": rows})
check("census counts every cell", c["S_paper493"]["mapped"] + c["S_paper493"]["fold_minus1"] == len(rows))
check("one set per g1/g2 cell", not R.one_set_per_cell({"sp493_g1_sim_v1": {"sim|a"}, "sp493_g1_sim_s1_v1_try2": {"sim|b"}}))
check("a g1 cell in two sets is caught", bool(R.one_set_per_cell({"sp493_g1_sim_v1": {"sim|a"}, "sp493_g1_sim_s1_v1_try2": {"sim|a"}})))
check("a cell in a g1 and a g2 set is caught", bool(R.one_set_per_cell({"sp493_g1_sim_v1": {"sim|a"}, "sp493_g2_sim_v1": {"sim|a"}})))
check("a cell in a g6 x and a g6 y set is caught", bool(R.one_set_per_cell({"sp493_g6_sim_x_v1": {"sim|a"}, "sp493_g6_sim_y_s1_v1": {"sim|a"}})))
check("a cell in g1 and g6 (one group, L1) is caught", bool(R.one_set_per_cell({"sp493_g1_sim_s1_v1_try2": {"sim|a"}, "sp493_g6_sim_x_v1": {"sim|a"}})))
full = {}
for tc, tg in P["tail_group"].items():
    full.setdefault(f"sp493_{tg}_sim_{P['tail_launch'][tc]}_v1", set()).add(f"sim|{tc}")
check("tail pins: 137 + 121 + 37", sorted(__import__("collections").Counter(P["tail_group"].values()).items()) == [("g3", 37), ("g6", 137), ("g7", 121)])
check("the full tail release passes the per-set assert", not R.tail_assert(full, P))
one = next(iter(full))
check("a partial tail release passes the per-set assert (09 01:2x)", not R.tail_assert({one: full[one]}, P))
cells_all = sorted(P["tail_group"])
check("final union: all admitted passes", not R.tail_union_assert(cells_all, {}, P))
check("final union: admitted + classed gaps passes", not R.tail_union_assert(cells_all[2:], {cells_all[0]: "setup failure", cells_all[1]: "no clean donor"}, P))
check("final union: an unclassed gap STOPs", bool(R.tail_union_assert(cells_all[1:], {}, P)))
check("final union: admitted and classed STOPs", bool(R.tail_union_assert(cells_all, {cells_all[0]: "errored run"}, P)))
check("final union: an empty class STOPs", bool(R.tail_union_assert(cells_all[1:], {cells_all[0]: " "}, P)))
check("final union: a cell outside the 295 STOPs", bool(R.tail_union_assert(cells_all + ["t000_x"], {}, P)))
c0 = next(iter(full[one]))
wrong = one.replace("_x_", "_y_") if "_x_" in one else one.replace("_y_", "_x_")
check("a tail cell under the wrong launch token fails", bool(R.tail_assert({k: v - {c0} for k, v in full.items()} | {wrong: full.get(wrong, set()) | {c0}}, P)))
check("a tail set without an x/y token fails", bool(R.tail_assert({"sp493_g6_sim_v1": {"sim|a"}}, P)))
check("disjoint x/y tails pass", not R.one_set_per_cell({"sp493_g6_sim_x_v1": {"sim|a"}, "sp493_g6_sim_y_v1": {"sim|b"}, "sp493_g3_sim_x_v1": {"sim|c"}}))
row = lambda r, e: {"launch": "L", "rel": r, "sample_error": e} if e != "missing" else {"launch": "L", "rel": r}  # noqa: E731
check("sample_error null on admitted rows passes", R.sample_error_ok([row("a", None), row("x", "EmptyPolicyCompletionError")], ["L/a#e0"]))
check("sample_error missing on any row refuses", not R.sample_error_ok([row("a", None), row("x", "missing")], ["L/a#e0"]))
check("sample_error set on an admitted row refuses", not R.sample_error_ok([row("a", "EmptyPolicyCompletionError")], ["L/a#e0"]))
check("an admitted row absent from the manifest refuses", not R.sample_error_ok([row("a", None)], ["L/b#e0"]))
print({"cases": len(cases), "census": c, "controls": g["controls"], "failures": fails})
if fails:
    raise SystemExit(f"FAIL: {fails}")
print("PASS")
