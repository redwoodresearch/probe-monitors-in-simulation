"""Lean test of the route C slot readout on SYNTHETIC records (no real value is read): the record layout is the writer's (run_peraction311
pool_reduce_n / finish_rows / _row's pools_at_draw0; run_group_catch.py; run_group_excess.py), every number is planted here.
  (1) formats; (2) every A slot reads its own key path (distinct planted values); (3) derived values (half-widths, ranges, distinct runs);
  (4) not readable / withheld / missing; (5) integrity STOPs; (6) B and R37; (7) Modal wiring and the laptop's no-parse rule (syntax tree).

  uv run --no-project --python 3.12 --with modal==1.6.1 python test_slot_readout.py   (from this folder)
"""

import ast
import copy
import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 2e (10-18): the test runs under python -I too (the script's folder is not on sys.path)
import slot_table as ST  # noqa: E402

HERE = Path(__file__).resolve().parent
bad = []
M = ST.MINUS


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        bad.append(name)


def stops(fn, *a):
    try:
        fn(*a)
    except SystemExit:
        return True
    return False


_c = itertools.count(1)
LAB = "171-pool test registry label; PRELIMINARY"
RSHA = {ST.REGISTRY: ST.REGISTRY_SHA256}  # every planted registry stands in for 09's verified one (the pin itself: section 13)
RCW = "route C test sidecar label; PRELIMINARY"


def build(recs, shas):  # (1)-(10), a writing run: A and its group records in a planted 171-pool registry (which also lists a record not
    shas = {r: shas.get(r, f"{n:064d}") for n, r in enumerate(recs, 1)}  # given); appendix-bound records (B, R37) with route C sidecars
    paths = {r: f"x/{r}.json" for r in recs}  # (labels: sections 12 and 13)
    reg = [{"record": paths[r], "record_sha256": shas[r], "label": LAB} for r in recs if r in ST.RECORD_ROLES and r not in ST.APPENDIX_ROLES]
    reg.append({"record": "x/other_171pool.json", "record_sha256": "d" * 64, "label": LAB})
    sides = {f"{r}_label": {"record": paths[r], "record_sha256": shas[r], "label": RCW} for r in recs if r in ST.APPENDIX_ROLES}
    return ST.build(recs | sides | {ST.REGISTRY: reg}, shas | RSHA, paths)


def S():  # a planted summ node with distinct values
    k = next(_c)
    p = ((k * 37) % 401 - 200) / 1000
    return {"point": p, "finite_draws": 1000, "draws": 1000, "ci95": [p - 0.01 - k / 1e5, p + 0.02 + k / 1e5], "degenerate": False}


LABS = [f"{x} {ro}" for x in ST.SIX + (ST.C_ASIS,) + ST.RS_PARTIAL for ro in ST.CR]
MEMBERS = [f"{o} (all-origins draws) [pooled: {ST.LEAD}]" for o in ("S_old", "S_new", "S_newmains", "S_paper493")]


def q():
    return {"catch": {k: S() for k in ("sim", "run1", "run2", "real_mean")},  # every arm and gap a distinct planted node
            "gap": {k: S() for k in ("delta_full", "run2_minus_run1_at_tauB", "real_line")}, "readable": {"all": True}, "readable_real_line": True}


UNITS = {"B": (1400, 900), "A": (1000, 700), "S": (1300, 800)}  # per pool: (public, trained) units; runs 40 / 44 / 38


def pools(seed, lab):  # a63478 R5: every field differs between pools B, A and S, so a pool swap changes the slot
    fam = lab.rsplit(" ", 1)[0]
    out = {}
    for i, pl in enumerate(ST.POOLS):
        runs, k = (40, 44, 38)[i], (seed % 7 + 1, 11, seed % 5 + 13)[i]  # alarming runs: B 1-7 of 40, A 11 of 44, S 13-17 of 38
        z, st = (seed % 3, 1, seed % 3 + 4)[i], ["cell_x"] * ((seed + i) % 2)
        out[pl] = {"per_run_alarm_share": k / runs, "per_run_alarm_share_with_no_unit_runs": k / (runs + z + len(st)), "runs": runs,
                   "units_per_run_mean": 10.0 + i, "n_units": UNITS[pl][fam in ST.TRAINED], "units_above": (100, 200, 300)[i] + seed,
                   "no_unit_runs_counted_non_alarming": {"zero_unit_admitted": z, "structural_no_row_cells": st}}
    return out


def rec_a():
    A = {"pooled": {L: {ST.LEAD: {m: {"0.01": q(), "0.05": q()} for m in ("joint", "fixed")}} for L in LABS},
         "coheadline_signed": {ST.LEAD: {"joint": {lv: {f"{g}|{ro}": {"S": S(), "Y": S(), "rows": []}
                                                       for g in ("headline_six_full", "auroc06_subset") for ro in ST.CR} for lv in ("0.01", "0.05")}}},
         "panel_groups": {ST.LEAD: {"pooled": {"joint": {"0.01": {f"headline_six_full|{ro}": {"primary": {"rows": [], "n": 6, "mean_G_minus_mean_Y": S()}}
                                                              for ro in ST.CR}}}}},
         "meta": {"pooled_world": {"per_label": {f"{L} | {ST.LEAD}": {"sizes": {pl: {"after": {"runs": {"B": 1000, "A": 2000, "S": 3000}[pl] + i, "units": 0}}
                                                                                           for pl in ST.POOLS}}
                                                 for i, L in enumerate(LABS)}}},
         "combo_weights": {f"{L}|{ST.LEAD}": {"n_mains": 500 + i} for i, L in enumerate(LABS)},
         "rows": {L: {m: {"pools_at_draw0": {"0.01": pools(i, L), "0.001": pools(i + 3, L)}} for m in MEMBERS} for i, L in enumerate(LABS)}}
    return json.loads(json.dumps(A))  # as the container parses it


A = rec_a()
SHA_A = "a" * 64

# (1) formats
check("(1) pct / pp / pp2 / share / interval", ST.pct(0.1234) == "12.3%" and ST.pp(-0.0123) == f"{M}1.2" and ST.pp(0.0123) == "+1.2"
      and ST.pp2(0.01234) == "+1.23" and ST.share(0.456) == "46%"
      and ST.summ_str({"point": 0.05, "ci95": [-0.01, 0.1]}, ST.pp)[0] == f"+5.0 ({M}1.0 to +10.0)")
check("(1) counts: thousands separators at 1,000 and above, none below; full keeps the plain int",
      ST.cnt(1906) == ("1,906", "1906") and ST.cnt(906) == ("906", "906") and ST.cnt(1234567)[0] == "1,234,567")
check("(1) a withheld interval prints the point and 'interval withheld'",
      ST.summ_str({"point": 0.05, "ci95": None, "withheld": "fewer than 95% complete draws"}, ST.pp)[0].startswith("+5.0 (interval withheld"))

# (2) every A slot reads its own path
r = build({"A": A}, {"A": SHA_A})
sl = r["slots"]
fmt = lambda node, f=ST.pp: ST.summ_str(node, f)[0]  # noqa: E731
c_ = lambda v: f"{v:,}"  # noqa: E731  (counts carry thousands separators: ctrl1 fold-in, 10-17)
L1 = "sol_a_uncut C"
check("(2) no slot missing on a complete A", r["missing"] == {})
check("(2) 01-05 co-headline S / Y by group, readout and level",
      sl["01.coheadline.headline_six_full|C@0.01.S"] == fmt(A["coheadline_signed"][ST.LEAD]["joint"]["0.01"]["headline_six_full|C"]["S"])
      and sl["04.coheadline.auroc06_subset|R@0.01.Y"] == fmt(A["coheadline_signed"][ST.LEAD]["joint"]["0.01"]["auroc06_subset|R"]["Y"])
      and sl["05.coheadline.headline_six_full|C@0.05.S"] == fmt(A["coheadline_signed"][ST.LEAD]["joint"]["0.05"]["headline_six_full|C"]["S"])
      and "05.coheadline.headline_six_full|C@0.05.Y" not in sl)
check("(2) 06 / 07 excess from panel_groups ... primary.mean_G_minus_mean_Y",
      sl["07.excess.headline_six_full|R"] == fmt(A["panel_groups"][ST.LEAD]["pooled"]["joint"]["0.01"]["headline_six_full|R"]["primary"]["mean_G_minus_mean_Y"]))
qa = A["pooled"][L1][ST.LEAD]["joint"]["0.01"]
check("(2) 09 catch (%), 10 delta_full (pp), 11 Y (pp) at joint / 0.01",
      sl[f"09.catch.{L1}.run2"] == fmt(qa["catch"]["run2"], ST.pct) and sl[f"10.delta_full.{L1}"] == fmt(qa["gap"]["delta_full"])
      and sl[f"11.run2_minus_run1_at_tauB.{L1}"] == fmt(qa["gap"]["run2_minus_run1_at_tauB"]))
check("(2) 10 the on-policy probe's delta_full has 2 decimals",
      sl["10.delta_full.own_all_rs_uncut_L48 R"] == fmt(A["pooled"]["own_all_rs_uncut_L48 R"][ST.LEAD]["joint"]["0.01"]["gap"]["delta_full"], ST.pp2))
check("(2) 12 fixed-mode interval only; 13 c_asis delta_full and real_line",
      sl["12.fixed.delta_full_ci.sol_a_uncut C@0.05"] == ST.ci_str(A["pooled"][L1][ST.LEAD]["fixed"]["0.05"]["gap"]["delta_full"], ST.pp)[0]
      and sl["13.real_line.c_asis_uncut R"] == fmt(A["pooled"]["c_asis_uncut R"][ST.LEAD]["joint"]["0.01"]["gap"]["real_line"]))
cq = lambda x, ro, arm: A["pooled"][f"{x} {ro}"][ST.LEAD]["joint"]["0.01"]["catch"][arm]["point"]  # noqa: E731
check("(2) 09 avg run1 / run2: the equal-weight mean of the group's recorded points, per group, readout and run (each its own)",
      all(sl[f"09.catch_avg.{g}|{ro}.{arm}"] == ST.pct(sum(cq(x, ro, arm) for x in labs) / len(labs))
          for g, labs in (("six", ST.SIX), ("four", ST.FOUR)) for ro in ST.CR for arm in ("run1", "run2"))
      and len({r["full"][f"09.catch_avg.{g}|{ro}.{arm}"] for g in ("six", "four") for ro in ST.CR for arm in ("run1", "run2")}) == 8)
A11 = rec_a()
A11["pooled"]["refusal|uncut R"][ST.LEAD]["joint"]["0.01"]["reading"] = "not readable"
s11 = build({"A": A11}, {"A": SHA_A})["slots"]
rsk_ = [f"09.catch.{x} {ro}.{s}" for x in ST.RS_PARTIAL for ro in ST.CR for s in ("run1", "run2", "mean")]
check("(2) 09 rs10 / rs25 (writing via 0d, 10-18): run1 and run2 read by key as slot 9 does, and their mean computed from the probe's own "
      "points (each its own; full precision distinct)",
      all(k in sl for k in rsk_) and len({r["full"][k] for k in rsk_}) == 12
      and all(sl[f"09.catch.{x} {ro}.{a}"] == fmt(A["pooled"][f"{x} {ro}"][ST.LEAD]["joint"]["0.01"]["catch"][a], ST.pct)
              for x in ST.RS_PARTIAL for ro in ST.CR for a in ("run1", "run2"))
      and all(sl[f"09.catch.{x} {ro}.mean"] == ST.pct((cq(x, ro, "run1") + cq(x, ro, "run2")) / 2) for x in ST.RS_PARTIAL for ro in ST.CR)
      and all(r["full"][f"09.catch.{x} {ro}.mean"].endswith("[point only: the mean of real run 1's and real run 2's catch points]")
              for x in ST.RS_PARTIAL for ro in ST.CR))
A12 = rec_a()
A12["pooled"]["own_all_rs10_uncut_L48 R"][ST.LEAD]["joint"]["0.01"]["reading"] = "not readable"
s12 = build({"A": A12}, {"A": SHA_A})["slots"]
check("(2) 09 rs10 / rs25: a block with `reading` prints 'not readable' for run1, run2 and mean, and only there",
      all(s12.get(f"09.catch.own_all_rs10_uncut_L48 R.{s}") == "not readable" for s in ("run1", "run2", "mean"))
      and s12.get("09.catch.own_all_rs25_uncut_L48 R.mean") not in (None, "not readable")
      and s12.get("09.catch.own_all_rs10_uncut_L48 C.mean") not in (None, "not readable"))
check("(2) 09 avg run1 / run2: a group with a not-readable member prints 'not readable' (six R only; four has no refusal)",
      s11["09.catch_avg.six|R.run1"] == "not readable" and s11["09.catch_avg.four|R.run2"] != "not readable" and s11["09.catch_avg.six|C.run1"] != "not readable")
check("(2) 14: a count of 1,000 or more prints with a thousands separator", "," in sl[f"14.after_runs.{L1}.S"])
check("(2) 14 after-runs per pool and attack mains", sl[f"14.after_runs.{L1}.S"] == c_(A["meta"]["pooled_world"]["per_label"][f"{L1} | {ST.LEAD}"]["sizes"]["S"]["after"]["runs"])
      and sl[f"14.attack_mains.{L1}"] == c_(A["combo_weights"][f"{L1}|{ST.LEAD}"]["n_mains"]))
p0 = A["rows"][L1][MEMBERS[0]]["pools_at_draw0"]["0.01"]
check("(2) 15 per_run_alarm_share (whole %), never the with-no-unit-runs share", sl[f"15.alarm_share.{L1}.B"] == ST.share(p0["B"]["per_run_alarm_share"])
      and not any("with_no_unit" in k for k in sl))
check("(2) 15b zero-unit runs = zero_unit_admitted + structural no-row cells",
      sl[f"15b.zero_unit_runs.{L1}.S"] == c_(p0["S"]["no_unit_runs_counted_non_alarming"]["zero_unit_admitted"]
                                              + len(p0["S"]["no_unit_runs_counted_non_alarming"]["structural_no_row_cells"])))
check("(2) not_carried names #8, the group-catch averages, B and R37 (no longer #17)",
      {"08", "09.catch_avg", "18-19", "20"} <= set(r["not_carried"]) and "17" not in r["not_carried"])

# (3) derived values
hw = {x: (lambda c: (c[1] - c[0]) / 2)(A["pooled"][f"{x} C"][ST.LEAD]["joint"]["0.01"]["gap"]["delta_full"]["ci95"]) for x in ST.SIX}
ex = [hw[x] for x in ST.SIX if x != ST.ON_POLICY]
check("(3) half-widths: min / max over the five, max over the six", sl["10.halfwidth_C.min_excl_on_policy"] == ST.num(100 * min(ex), 1, False)
      and sl["10.halfwidth_C.max_excl_on_policy"] == ST.num(100 * max(ex), 1, False) and sl["10.halfwidth_C.max_all"] == ST.num(100 * max(hw.values()), 1, False))
d0 = lambda x, pl, k: A["rows"][f"{x} C"][MEMBERS[0]]["pools_at_draw0"]["0.01"][pl][k]  # noqa: E731
sh = [d0(x, pl, "per_run_alarm_share") for x in ST.SIX for pl in ("B", "S")]
check("(3) 15 alarm-share range over the six labels and pools B, S", sl["15.alarm_share.range"] == f"{ST.share(min(sh))} to {ST.share(max(sh))}")
fr_ = [r["full"][k] for k in ("15.alarm_share.range", "15b.zero_unit_runs.range", "16.units_per_run.range")]
check("(3) 09 (10-18): the rendered pool names are exactly honest pool A, honest pool B and sim honest pool S (a5a7d010's optional)",
      ST.POOL_NAME == {"B": "honest pool B", "A": "honest pool A", "S": "sim honest pool S"} and set(ST.POOL_NAME) == set(ST.POOLS))
BARE = re.compile(r"(?<!pool )(?<![\w|])[BAS](?![\w|])")  # a pool letter standing alone, not after "pool " (09, 10-18)
check("(3) 09 (10-18): no slot or full string the readout renders from A shows a bare pool letter (every slot of the complete A build)",
      not any(BARE.search(v) for part in ("slots", "full") for k, v in r[part].items() if not k.startswith("lbl.") and k != "mode")
      and any(n_ in r["full"]["17.readout_share.range"] for n_ in ("honest pool A", "honest pool B", "sim honest pool S")))
check("(3) 09 (10-18): the ranges' rendered labels name the pool ('honest pool B', 'sim honest pool S'), never a bare B or S",
      all(("honest pool B" in s_ or "sim honest pool S" in s_) for s_ in fr_)
      and not any(re.search(r"\b(C|R|trained|public) [BAS](;|$)", s_) for s_ in fr_))  # the old "<label> C B" form
upr = [UNITS[pl][t] / d0(lab, pl, "runs") for t, lab in ((0, "FIBS|uncut"), (1, "sol_a_uncut")) for pl in ("B", "S")]
check("(3) 16 units / runs per group and pool; range to the nearest 5 as 'about X to Y'",
      sl["16.units_per_run.public.B"] == ST.num(1400 / 40, 1, False) and sl["16.units_per_run.range"] == f"about {ST.round5(min(upr))} to {ST.round5(max(upr))}"
      and ST.round5(12.5) == 15 and ST.round5(12.4) == 10)
skp = {pl: (UNITS[pl][0] - UNITS[pl][1]) / UNITS[pl][0] for pl in ST.POOLS}
check("(3) 17 the share of public scored calls the trained probes skip, per pool and min to max over B, A, S",
      sl["17.readout_share.A"] == ST.share(skp["A"]) and sl["17.readout_share.range"] == f"{ST.share(min(skp.values()))} to {ST.share(max(skp.values()))}")
A5 = copy.deepcopy(A)
for m in MEMBERS:
    A5["rows"]["refusal|uncut C"][m]["pools_at_draw0"]["0.01"]["B"]["n_units"] += 1
check("(3) a group whose members' units differ STOPs (16, 17)", stops(build, {"A": A5}, {"A": SHA_A}))
A6 = copy.deepcopy(A)
for m in MEMBERS:
    A6["rows"][L1][m]["pools_at_draw0"]["0.01"]["S"]["per_run_alarm_share_with_no_unit_runs"] *= 0.5
check("(3) 15b STOPs when the two shares disagree with the zero-unit count", stops(build, {"A": A6}, {"A": SHA_A}))

# (4) not readable, withheld, missing
A2 = copy.deepcopy(A)
A2["pooled"][L1][ST.LEAD]["joint"]["0.01"]["reading"] = "not readable (...)"
A2["coheadline_signed"][ST.LEAD]["joint"]["0.01"]["auroc06_subset|C"] = {"unavailable": "rows missing [] or not readable ['x C']"}
A2["panel_groups"][ST.LEAD]["pooled"]["joint"]["0.01"]["headline_six_full|C"]["primary"] = {"rows": [], "withheld": "unreadable members ['x C']"}
for ro in ST.CR:
    del A2["pooled"][f"c_asis_uncut {ro}"]
r2 = build({"A": A2}, {"A": SHA_A})
check("(4) a block with `reading` prints 'not readable' for catch, delta_full and Y",
      r2["slots"][f"09.catch.{L1}.sim"] == r2["slots"][f"10.delta_full.{L1}"] == r2["slots"][f"11.run2_minus_run1_at_tauB.{L1}"] == "not readable")
check("(4) unavailable co-headline and withheld excess print their reason",
      r2["slots"]["02.coheadline.auroc06_subset|C@0.01.S"].startswith("unavailable:") and r2["slots"]["06.excess.headline_six_full|C"].startswith("withheld:"))
check("(4) a label the record lacks is listed in `missing` by slot key, with no string",
      {"13.delta_full.c_asis_uncut C", "13.real_line.c_asis_uncut R"} <= set(r2["missing"]) and "13.delta_full.c_asis_uncut C" not in r2["slots"])
check("(4) a not-readable label leaves the half-widths missing, not filled", "10.halfwidth_C.max_all" in r2["missing"])

# (5) integrity STOPs
A3 = copy.deepcopy(A)
A3["rows"][L1][MEMBERS[2]]["pools_at_draw0"]["0.01"]["S"]["per_run_alarm_share"] += 0.01
check("(5) member rows that disagree on a draw-0 value STOP", stops(build, {"A": A3}, {"A": SHA_A}))
A4 = copy.deepcopy(A)
del A4["rows"][L1][MEMBERS[1]]
check("(5) a combo with the wrong number of member rows STOPs", stops(build, {"A": A4}, {"A": SHA_A}))
GE = {"four_probe": {"rows": [f"{x} C" for x in ST.FOUR], "n": 4, "mean_G_minus_mean_Y": S()}, "gate": {"pass": True},
      "meta": {"record": {"sha256": SHA_A}, "combo": ST.LEAD, "mode": "joint", "level": 0.01, "readout": "C"}}
r5 = build({"A": A, "A_group_excess": GE}, {"A": SHA_A, "A_group_excess": "e" * 64})
check("(5) the four-probe excess reads four_probe.mean_G_minus_mean_Y", r5["slots"]["08.excess.four_probe|C"] == fmt(GE["four_probe"]["mean_G_minus_mean_Y"]))
check("(5) a group-excess record tied to another A, failing its gate, or with other members STOPs",
      stops(build, {"A": A, "A_group_excess": GE}, {"A": "b" * 64})
      and stops(build, {"A": A, "A_group_excess": GE | {"gate": {"pass": False}}}, {"A": SHA_A})
      and stops(build, {"A": A, "A_group_excess": GE | {"four_probe": GE["four_probe"] | {"rows": ["refusal|uncut C"] * 4}}}, {"A": SHA_A}))
GC = {"groups": {f"{k}|{ro}": {"members": [f"{x} {ro}" for x in labs], "combo": ST.LEAD, "mode": "joint", "catch": {"real_mean": S(), "sim": S()}}
                 for k, labs in (("six", ST.SIX), ("auroc_subset", ST.FOUR)) for ro in ST.CR},
      "gate": {"pass": True}, "meta": {"record": {"sha256": SHA_A}}}
r6 = build({"A": A, "A_group_catch": GC}, {"A": SHA_A})
check("(5) the group-catch averages read groups[six|four].catch", r6["slots"]["09.catch_avg.four|R.sim"] == fmt(GC["groups"]["auroc_subset|R"]["catch"]["sim"], ST.pct))
check("(5) a group-catch record tied to another A STOPs", stops(build, {"A": A, "A_group_catch": GC}, {"A": "c" * 64}))

# (6) B and R37
B = rec_a()
for L in LABS:
    B["pooled"][L][ST.LEAD] = {m: {f"{lv:g}": q() for lv in (0.001, 0.0015, 0.01, 0.05)} for m in ("joint", "fixed")}
r7 = build({"B": B}, {})
p1 = B["rows"][L1][MEMBERS[0]]["pools_at_draw0"]["0.001"]["S"]
pB1 = B["rows"][L1][MEMBERS[0]]["pools_at_draw0"]["0.001"]["B"]
qb = lambda m, lv: B["pooled"][L1][ST.LEAD][m][lv]  # noqa: E731
check("(6) B: every level in the record, both modes, no slot missing", f"18.delta_full.fixed.{L1}@0.0015" in r7["slots"] and r7["missing"] == {})
check("(6) R5: #18 delta_full and Y carry their own planted values, per mode and level",
      all(r7["slots"][f"18.delta_full.{m}.{L1}@{lv}"] == fmt(qb(m, lv)["gap"]["delta_full"])
          and r7["slots"][f"18.Y.{m}.{L1}@{lv}"] == fmt(qb(m, lv)["gap"]["run2_minus_run1_at_tauB"]) for m in ("joint", "fixed") for lv in ("0.0015", "0.05")))
check("(6) R5: #19 S units above, S distinct runs, B units above: each its own pool's value",
      r7["slots"][f"19.distinct_runs_above_tau_sim.{L1}@0.001"] == c_(round(p1["per_run_alarm_share"] * p1["runs"]))
      and r7["slots"][f"19.units_above_tau_sim.{L1}@0.001"] == c_(p1["units_above"])
      and r7["slots"][f"19.real_units_above_tau_B.{L1}@0.001"] == c_(pB1["units_above"]) and p1["units_above"] != pB1["units_above"])
B2 = copy.deepcopy(B)
for m in MEMBERS:
    B2["rows"][L1][m]["pools_at_draw0"]["0.001"]["S"]["runs"] = 41  # share x runs is no longer an integer
check("(6) a non-integer distinct-run count STOPs", stops(build, {"B": B2}, {}))
R = rec_a()
for L in LABS:
    R["pooled"][L][ST.LEAD] = {m: {f"{lv:g}": q() | {"readable_rule37": True} for lv in ST.R37_LEVELS} for m in ("joint", "fixed")}
    for m in MEMBERS:
        R["rows"][L][m]["pools_at_draw0"] |= {f"{lv:g}": pools(3, L) for lv in ST.R37_LEVELS}
r8 = build({"R37": R}, {})
check("(6) R37: delta_full, catch, readability, distinct runs at 1e-4 / 2e-4 / 5e-4", r8["missing"] == {}
      and r8["slots"][f"20.readable.fixed.{L1}@0.0002"] == "rule 37 readable; gate readable" and f"20.distinct_runs_above_tau_sim.{L1}@0.0005" in r8["slots"])
qr = lambda m, lv: R["pooled"][L1][ST.LEAD][m][lv]  # noqa: E731
pS3 = R["rows"][L1][MEMBERS[0]]["pools_at_draw0"]["0.0005"]["S"]
check("(6) R5: #20 delta_full, catch sim and real_mean carry their own planted values; distinct runs from pool S",
      all(r8["slots"][f"20.delta_full.{m}.{L1}@{lv}"] == fmt(qr(m, lv)["gap"]["delta_full"])
          and r8["slots"][f"20.catch.{m}.{L1}@{lv}.sim"] == fmt(qr(m, lv)["catch"]["sim"], ST.pct)
          and r8["slots"][f"20.catch.{m}.{L1}@{lv}.real_mean"] == fmt(qr(m, lv)["catch"]["real_mean"], ST.pct)
          for m in ("joint", "fixed") for lv in ("0.0001", "0.0005"))
      and r8["slots"][f"20.distinct_runs_above_tau_sim.{L1}@0.0005"] == c_(round(pS3["per_run_alarm_share"] * pS3["runs"])))

# (7) wiring
SRC = (HERE / "run_slot_readout.py").read_text()
T = ast.parse(SRC)
fn = {n.name: n for n in ast.walk(T) if isinstance(n, ast.FunctionDef)}
app_ = next(n.value.value for n in T.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "APP")
check("(7) app name <= 63 characters, probe-shift_aswin-surya_sub-rerun_..._kill-2026-11-19 (10-20 rerun naming rule)", len(app_) <= 63
      and app_.startswith("probe-shift_aswin-surya_sub-rerun_") and app_.endswith("_kill-2026-11-19"))
dec = next(d for d in fn["readout"].decorator_list if isinstance(d, ast.Call))
kw = {k.arg: k.value for k in dec.keywords}
check("(7) the function has retries=0 and an explicit timeout", ast.literal_eval(kw["retries"]) == 0 and isinstance(ast.literal_eval(kw["timeout"]), int))
check("(7) rule 27: the image puts slot_table.py at /root, and no module path climbs above this folder",
      "add_local_file(HERE / 'slot_table.py', '/root/slot_table.py')" in ast.unparse(T) and "parents[" not in SRC)
lap_ = [ast.unparse(c.func) for f_ in ("main", "pointer") for c in ast.walk(fn[f_]) if isinstance(c, ast.Call)]
check("(7) the laptop side never parses a record (no json.loads or gzip in main / pointer)", not any(("loads" in x or "gzip" in x) for x in lap_))
rd_ = ast.unparse(fn["readout"])
check("(7) the container checks its code, then loads the inputs (json.loads only after each sha check), then builds the slots",
      rd_.index("check_code(") < rd_.index("load_inputs(") < rd_.index("ST.build(") and "json.loads(b)" in ast.unparse(fn["load_inputs"]))

# (8) a0b05's first review (R-1 to R-3)
A7 = copy.deepcopy(A)
A7["meta"]["pooled_world"]["per_label"][f"{L1} | {ST.LEAD}"]["sizes"]["S"]["after"]["runs"] = 101.0
A8 = copy.deepcopy(A)
A8["combo_weights"][f"{L1}|{ST.LEAD}"]["n_mains"] = [500]
A9 = copy.deepcopy(A)
for m in MEMBERS:
    A9["rows"]["FIBS|uncut C"][m]["pools_at_draw0"]["0.01"]["A"]["n_units"] = True
check("(8) R-1: a float, a list or a bool where a count belongs STOPs (after_runs, attack mains, pool units)",
      stops(build, {"A": A7}, {"A": SHA_A}) and stops(build, {"A": A8}, {"A": SHA_A}) and stops(build, {"A": A9}, {"A": SHA_A}))
B3 = copy.deepcopy(B)
for m in MEMBERS:
    B3["rows"][L1][m]["pools_at_draw0"]["0.001"]["S"]["units_above"] = 3.0
check("(8) R-1: a float units_above in #19 STOPs", stops(build, {"B": B3}, {}))
mt_ = [n for n in ast.walk(fn["main"]) if isinstance(n, ast.Call)]
mk_ = [c.lineno for c in mt_ if "mkdir" in ast.unparse(c.func) and "Path(out).parent" in ast.unparse(c.func)]
rm_ = [c.lineno for c in mt_ if ast.unparse(c.func) == "readout.remote"]
check("(8) R-2: main() makes --out's folder before the paid call", len(mk_) == 1 and len(rm_) == 1 and mk_[0] < rm_[0])
op = "own_all_rs_uncut_L48 C"
check("(8) R-3: the on-policy probe's delta_full has 2 decimals in #18 and #20 too",
      r7["slots"][f"18.delta_full.joint.{op}@0.01"] == fmt(B["pooled"][op][ST.LEAD]["joint"]["0.01"]["gap"]["delta_full"], ST.pp2)
      and r8["slots"][f"20.delta_full.fixed.{op}@0.0002"] == fmt(R["pooled"][op][ST.LEAD]["fixed"]["0.0002"]["gap"]["delta_full"], ST.pp2)
      and r7["slots"][f"18.delta_full.joint.{L1}@0.01"] == fmt(B["pooled"][L1][ST.LEAD]["joint"]["0.01"]["gap"]["delta_full"], ST.pp))

# (9) a63478's second review: R4 (real_line's own gate), R5 (values, distinct per pool / arm), L1-L3 (arrays), S1-S3 (sha bypasses)
A10 = copy.deepcopy(A)
cq = A10["pooled"]["c_asis_uncut C"][ST.LEAD]["joint"]["0.01"]
cq["reading"] = "not readable (S fails)"
cq["readable_real_line"] = True  # S fails, B passes
r10 = build({"A": A10}, {"A": SHA_A})
check("(9) R4: with S failing and B passing, real_line prints its value and delta_full prints 'not readable'",
      r10["slots"]["13.real_line.c_asis_uncut C"] == fmt(cq["gap"]["real_line"]) and r10["slots"]["13.delta_full.c_asis_uncut C"] == "not readable")
cq["readable_real_line"] = False
check("(9) R4: with B failing, real_line prints 'not readable'", build({"A": A10}, {"A": SHA_A})["slots"]["13.real_line.c_asis_uncut C"] == "not readable")
del cq["readable_real_line"]
check("(9) R4: without its flag, real_line is listed missing (no fallback to the all-pool gate)",
      "13.real_line.c_asis_uncut C" in build({"A": A10}, {"A": SHA_A})["missing"])
pw = A["meta"]["pooled_world"]["per_label"][f"{L1} | {ST.LEAD}"]["sizes"]
check("(9) R5: #14 after-runs per pool are each pool's own value (B != S planted)",
      sl[f"14.after_runs.{L1}.B"] == c_(pw["B"]["after"]["runs"]) and sl[f"14.after_runs.{L1}.S"] == c_(pw["S"]["after"]["runs"])
      and pw["B"]["after"]["runs"] != pw["S"]["after"]["runs"])
check("(9) R5: #15 alarm share per pool is each pool's own value (B != S planted)",
      sl[f"15.alarm_share.{L1}.B"] == ST.share(p0["B"]["per_run_alarm_share"]) and sl[f"15.alarm_share.{L1}.S"] == ST.share(p0["S"]["per_run_alarm_share"])
      and ST.share(p0["B"]["per_run_alarm_share"]) != ST.share(p0["S"]["per_run_alarm_share"]))
check("(9) R5: #15b zero-unit runs per pool are each pool's own count",
      all(sl[f"15b.zero_unit_runs.{L1}.{pl}"] == c_(p0[pl]["no_unit_runs_counted_non_alarming"]["zero_unit_admitted"]
                                                     + len(p0[pl]["no_unit_runs_counted_non_alarming"]["structural_no_row_cells"])) for pl in ("B", "S")))


def leak(where, v=(0.1, 0.2)):  # an array where a number belongs, at one key path of a copy of A
    A_ = copy.deepcopy(A)
    d_ = A_
    for k in where[:-1]:
        d_ = d_[k]
    d_[where[-1]] = list(v)
    return A_


def leak_rows(key, pool="S"):
    A_ = copy.deepcopy(A)
    for m in MEMBERS:
        A_["rows"][L1][m]["pools_at_draw0"]["0.01"][pool][key] = [1, 2]
    return A_


Lq = ("pooled", L1, ST.LEAD, "joint", "0.01")
LEAKS = {"a summ point (catch)": leak(Lq + ("catch", "sim", "point")), "a ci95 end (delta_full)": leak(Lq + ("gap", "delta_full", "ci95"), ([0.1, 0.2], 0.3)),
         "a ci95 of three numbers (Y)": leak(Lq + ("gap", "run2_minus_run1_at_tauB", "ci95"), (0.1, 0.2, 0.3)),
         "a co-headline S point": leak(("coheadline_signed", ST.LEAD, "joint", "0.01", "headline_six_full|C", "S", "point")),
         "after_runs": leak(("meta", "pooled_world", "per_label", f"{L1} | {ST.LEAD}", "sizes", "B", "after", "runs")),
         "n_mains": leak(("combo_weights", f"{L1}|{ST.LEAD}", "n_mains")), "per_run_alarm_share": leak_rows("per_run_alarm_share"),
         "runs": leak_rows("runs"), "n_units": leak_rows("n_units", "A")}
check("(9) L1-L3: an array in any count or number slot STOPs (" + ", ".join(LEAKS) + ")",
      all(stops(build, {"A": a_}, {"A": SHA_A}) for a_ in LEAKS.values()))
B4 = copy.deepcopy(B)
for m in MEMBERS:
    B4["rows"][L1][m]["pools_at_draw0"]["0.001"]["B"]["units_above"] = [5]
check("(9) L1-L3: an array in #19's units_above STOPs", stops(build, {"B": B4}, {}))

import run_slot_readout as RS  # noqa: E402  (modal objects only; nothing runs)

GOOD = {n_: (HERE / n_).read_bytes() for n_ in RS.CODE}
pins = {"sha256": {n_: RS.sha(b) for n_, b in GOOD.items()}}
check("(9) S1: the right code pins pass", RS.check_code(pins, GOOD.__getitem__) is None)
check("(9) S1: a wrong pin on either code file STOPs", all(stops(RS.check_code, {"sha256": pins["sha256"] | {n_: "0" * 64}}, GOOD.__getitem__) for n_ in RS.CODE))
BAD_JSON = b"{not json"  # parsed before the sha check, it would raise a JSON error, not a STOP
for role in ("A", "A_group_catch", "A_group_excess", "B", "R37"):
    for path in ("x/rec.json", "x/rec.json.gz"):
        raw = BAD_JSON if not path.endswith(".gz") else __import__("gzip").compress(BAD_JSON)
        inp = {role: {"src": "bytes", "path": path, "sha256": RS.sha(b"something else"), "bytes": raw}}
        ok_ = stops(RS.load_inputs, inp, lambda x: x["bytes"])
        if not ok_:
            break
    check(f"(9) S2/S3: input {role}: a wrong sha STOPs before any parse (plain and .gz)", ok_)
good_in = {"A": {"src": "bytes", "path": "x/a.json", "sha256": RS.sha(b'{"k": 1}'), "bytes": b'{"k": 1}'}}
check("(9) S2: a right sha loads and records the input's sha", RS.load_inputs(good_in, lambda x: x["bytes"])[1]["A"]["sha256"] == RS.sha(b'{"k": 1}'))

# (10) a63478 REQUIRED-1: what the code PUTS into the result
outs_ = [r, r2, r5, r6, r7, r8, r10]
check("(10a) every value build() returns (slots, full, missing, not_carried) is a str, on every synthetic record",
      all(type(v) is str for res_ in outs_ for k in ("slots", "full", "missing", "not_carried") for v in res_[k].values()))
bt_ = ast.unparse([n for n in ast.walk(ast.parse((HERE / "slot_table.py").read_text())) if isinstance(n, ast.FunctionDef) and n.name == "build"][0])
check("(10a) build() ends with the strings-only guard (raw values or arrays STOP)", "if type(v) is not str" in bt_ and bt_.rstrip().endswith("return res")
      and "if bad:\n        raise SystemExit(f'a slot output is not a string (raw value or array): {bad[:5]} (STOP)')\n    return res" in bt_)
built_ = {"slots": {}, "full": {}, "missing": {}, "not_carried": {}}
code_ = {"git": "g" * 40, "sha256": {}, "blob": {}}
check("(10b) assemble() returns exactly slots, full, missing, not_carried, inputs, code",
      set(RS.assemble(built_, {}, code_)) == RS.OUT_KEYS == {"slots", "full", "missing", "not_carried", "inputs", "code"})
check("(10b) assemble() STOPs on any other key (e.g. the parsed records)", stops(RS.assemble, built_ | {"raw": {"A": {}}}, {}, code_))
rl_ = [s for s in fn["readout"].body if isinstance(s, ast.Return)]
check("(10b) readout() returns assemble(ST.build(...), ins, code) and nothing else",
      len(rl_) == 1 and ast.unparse(rl_[0]) == "return assemble(ST.build(recs, {r: ins[r][\x27sha256\x27] for r in ins}, {r: ins[r][\x27path\x27] for r in ins}, control), ins, code)")


def fake_git(mismatch=None):
    def g(*a):
        if a == ("rev-parse", "HEAD"):
            return b"h" * 40 + b"\n"
        if a[0] == "rev-parse":
            return b"blob-" + a[1].encode() + b"\n"
        name = Path(a[1]).name
        return (b"other\n" if name == mismatch else b"blob-HEAD:./" + name.encode() + b"\n")
    return g


cp_ = RS.code_pins(fake_git())
check("(10c) code_pins: matching HEAD blobs pass and record git, sha256 and blob per file",
      cp_["git"] == "h" * 40 and set(cp_["blob"]) == set(RS.CODE) and all(len(v) == 64 for v in cp_["sha256"].values()))
check("(10c) code_pins: a working copy that differs from HEAD STOPs, for either file", all(stops(RS.code_pins, fake_git(n_)) for n_ in RS.CODE))
check("(10c) main() takes its code pins from code_pins(git)", "code = code_pins(git)" in ast.unparse(fn["main"]))

# (11) downstream records (10-17): kappa, the five-probe extra, detector_vs_drop, the R² anchor, PLANT. Writing's appendix input is route C
# only (Aswin 10-17), so a writing run labels them by route C sidecars; the 171-pool registry (REG) is for a CONTROL run only (section 13)
LP = "route C pooled (corrected pool); PRELIMINARY until reviewed"
L171 = "171-pool; PRELIMINARY (v83)"
SHK, SHX, SHD, SHAN, SHP = ("1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64)
MODES = ("joint", "fixed")
KAP = {"meta": {"label": LP}, "gate": {"pass": True},
       "kappa": {f"{x} {ro}": {"combos": {ST.LEAD: {m: {k: S() for k in ST.KAPPA_PAIRS} for m in MODES}}} for x in ST.SIX for ro in ST.CR},
       "kappa_groups": {f"headline_six|{ro}|{ST.LEAD}": {m: {k: S() for k in ("mean_kappa_sim_vs_run1", "mean_kappa_run2_vs_run1",
                                                                                "mean_kappa_sim_minus_run2")} for m in MODES} for ro in ST.CR}}
KAP["kappa"]["refusal|uncut R"]["combos"][ST.LEAD] = {"withheld": "not readable", "members": []}
KAP["kappa"]["FIBS|uncut C"]["combos"][ST.LEAD]["fixed"]["kappa_sim_vs_run1"] = {"point": None, "finite_draws": 0, "draws": 1000, "ci95": None,
                                                                                   "withheld": "fewer than 95% complete draws (or draw 0 undefined)"}
KX_ = {"meta": {"label": LP + " | kappa_extra", "record_sha256": SHK}, "gate": {"pass": True},
       "five_probe_sensitivity": {"members": [x for x in ST.SIX if x != ST.ON_POLICY],
                                  "groups": {f"five_probe_sensitivity|{ro}|{ST.LEAD}": {m: {k: S() for k in ("mean_kappa_sim_vs_run1", "mean_kappa_run2_vs_run1",
                                                                                                           "mean_kappa_sim_minus_run2")} for m in MODES} for ro in ST.CR}}}
YDEF = "Stage D per_task max_minus_tau: sim - mean(run1, run2)"  # the writer's meta.definitions.y (run_detector_vs_drop.py L269 @ 3dd26da4)
DET = {"meta": {"label": "POST HOC; POOLED-world calibration (combo pooled_all_origins); " + LP, "definitions": {"x": "x-def", "y": YDEF, "null": "n"}},
       "result": {"probes": {f"{x} C": {dn: {an: S() | {"n_cells": 1000 + next(_c), "n_mains": 400} for an in ST.DET_ANALYSES} for dn in ST.DETECTORS}
                             for x in ST.SIX}}}
ANC = {"meta": {"label": "POST HOC; PRELIMINARY; run1-anchored lead"}, "gate": {"pass": True},
       "union_stream": {"lead": {f"{x} {ro}": {"sets": {ST.ANCHOR_SET: {"r2": {k: S() for k in ("main", "yardstick", "delta_main_minus_yardstick")},
                                                                         "n_cells": 1000 + next(_c)}}}
                                 for x in ST.SIX for ro in ST.CR},
                        "sensitivity_run2_anchor": {f"{ST.ON_POLICY} {ro}": {"sets": {ST.ANCHOR_SET: {"r2": {k: S() for k in ("main", "yardstick",
                                                                                                                           "delta_main_minus_yardstick")},
                                                                                                      "n_cells": 1000 + next(_c)}}}
                                                    for ro in ST.CR}}}
VERD = ("detected", "not detected", "unavailable")
vi = lambda f, i, j, k: (i + j * j + k * j + 2 * (f == "0.05") * (j % 2) + (k % 3) * (j > 3)) % 3  # noqa: E731  (varied counts per cell)
PL = {"label": "pipeline test on fixed data (planted at draw 0), not calibrated power", "gates": {"n": 6, "n_fail": 0}, "shape_gate": {"pass": True},
      "levels": ["0.01", "0.05"],
      "rows": {f: {ro: {f"{x} {ro}": {t: ({"verdict": VERD[vi(f, i, j, k)]} | ({"reason": f"row unreadable {i}{j}{k}"} if vi(f, i, j, k) == 2 else {}))
                                      for k, t in enumerate(ST.PLANT_TAGS)} for j, x in enumerate(ST.SIX)} for i, ro in enumerate(ST.CR)} for f in ("0.01", "0.05")},
      "aggregate": {"by_level": {f: {ro: {t: {"S": {"verdict": f"S-{f}-{ro}-{t}", "shift": {k: v * 100 for k, v in S().items() if k == "point"}
                                                     | {"ci95": [-1.5, 2.5], "finite_draws": 1000, "draws": 1000}},
                                              "excess": {"verdict": f"X-{f}-{ro}-{t}"}} for t in ST.PLANT_TAGS} for ro in ST.CR} for f in ("0.01", "0.05")}},
      "counts": None}


def plant_counts(rows):  # as the PLANT writer builds counts from the rows (run_plant_rows.py @ 4c951ce6f L549-557)
    out = {}
    for f, by in rows.items():
        for t in ST.PLANT_TAGS:
            for ro in ("C", "R", "C+R"):
                vs = {(f"{lb} ({r_})" if "+" in ro else lb): by[r_][lb][t]["verdict"] for r_ in ro.split("+") for lb in by[r_]}
                out.setdefault(f, {}).setdefault(t, {})[ro] = {"detected": sum(v == "detected" for v in vs.values()),
                                                               "not_detected": sum(v == "not detected" for v in vs.values()),
                                                               "unavailable": sum(v == "unavailable" for v in vs.values()), "cases": len(vs),
                                                               "misses": {k: v for k, v in vs.items() if v != "detected"}}
    return out


PL["counts"] = plant_counts(PL["rows"])


def count_str(c):  # the expected 25.plant.count string: detected / missed / unavailable apart, names by kind
    nm = {w: sorted(k for k, v in c["misses"].items() if v == w) for w in ("not detected", "unavailable")}
    tail = "; ".join(f"{lab}: {', '.join(nm[w])}" for lab, w in (("missed", "not detected"), ("unavailable", "unavailable")) if nm[w])
    return f"{c['detected']}/{c['cases']} detected, {c['not_detected']} missed, {c['unavailable']} unavailable" + (f" ({tail})" if tail else "")


def side(role_sha, label):
    return {"record": "x/rec.json", "record_sha256": role_sha, "label": label}


DSHA = {"KAPPA": SHK, "KAPPA_EXTRA": SHX, "DETDROP": SHD, "ANCHOR": SHAN, "PLANT": SHP}
DPATH = {r: f"experiments/x/results/{r.lower()}_2026-10-17.json" for r in DSHA}
REG = [{"record": DPATH[r], "record_sha256": DSHA[r], "label": f"{L171} | {r}"} for r in DSHA]
REG_OTHER = [{"record": "experiments/x/results/other_171pool.json", "record_sha256": "d" * 64, "label": L171}]  # lists none of these five
CREC = {"KAPPA": KAP, "KAPPA_EXTRA": KX_, "DETDROP": DET, "ANCHOR": ANC, "PLANT": PL, ST.REGISTRY: REG}  # a CONTROL run's inputs
DREC = {k: v for k, v in CREC.items() if k != ST.REGISTRY} | {f"{r}_label": side(DSHA[r], f"{LP} | {r}") for r in DSHA} | {ST.REGISTRY: REG_OTHER}
rd = ST.build(json.loads(json.dumps(DREC)), DSHA | RSHA, DPATH)
sd = rd["slots"]
check("(11) downstream: no slot missing, and every output a string", rd["missing"] == {} and all(type(v) is str for v in sd.values()))
kq = lambda L, m, k: KAP["kappa"][L]["combos"][ST.LEAD][m][k]  # noqa: E731
check("(11) kappa: each label x mode x pair carries its own planted value (2 decimals; the difference signed)",
      all(sd[f"21.kappa.{L}.{m}.{k}"] == fmt(kq(L, m, k), ST.k2s if k.endswith("minus_run2") else ST.k2)
          for L in ("sol_a_uncut C", "own_all_rs_uncut_L48 R", "harmfulness|uncut C") for m in MODES for k in ST.KAPPA_PAIRS)
      and sd["21.kappa.refusal|uncut R.joint.kappa_sim_vs_run1"].startswith("withheld:")
      and sd["21.kappa.FIBS|uncut C.fixed.kappa_sim_vs_run1"].startswith("withheld:")
      and sd["21.kappa.FIBS|uncut C.joint.kappa_sim_vs_run1"] == fmt(kq("FIBS|uncut C", "joint", "kappa_sim_vs_run1"), ST.k2))
check("(11) kappa group (headline six) and the five-probe extra: their own planted values per readout x mode x key",
      all(sd[f"21.kappa_group.headline_six|{ro}.{m}.{k}"] == fmt(KAP["kappa_groups"][f"headline_six|{ro}|{ST.LEAD}"][m][k], ST.k2s if k.endswith("run2") and "minus" in k else ST.k2)
          and sd[f"22.kappa_five.{ro}.{m}.{k}"] == fmt(KX_["five_probe_sensitivity"]["groups"][f"five_probe_sensitivity|{ro}|{ST.LEAD}"][m][k],
                                                      ST.k2s if "minus" in k else ST.k2)
          for ro in ST.CR for m in MODES for k in ("mean_kappa_sim_vs_run1", "mean_kappa_run2_vs_run1", "mean_kappa_sim_minus_run2")))
dq = lambda x, dn, an: DET["result"]["probes"][f"{x} C"][dn][an]  # noqa: E731
check("(11) detector_vs_drop: Spearman and n_cells per probe x detector x analysis, each its own",
      all(sd[f"23.spearman.{x} C.{dn}.{an}"] == fmt(dq(x, dn, an), ST.k2s) and sd[f"23.n_cells.{x} C.{dn}.{an}"] == c_(dq(x, dn, an)["n_cells"])
          for x in ST.SIX for dn in ST.DETECTORS for an in ST.DET_ANALYSES))
aq = lambda b, L, k: ANC["union_stream"][b][L]["sets"][ST.ANCHOR_SET]["r2"][k]  # noqa: E731
check("(11) R² anchor: the union-stream lead per label x key, and the run-2 sensitivity for the on-policy probe only",
      all(sd[f"24.r2.lead.{x} {ro}.{k}"] == fmt(aq("lead", f"{x} {ro}", k), ST.k2s if k.startswith("delta") else ST.k2)
          for x in ST.SIX for ro in ST.CR for k in ("main", "yardstick", "delta_main_minus_yardstick"))
      and all(sd[f"24.r2.sensitivity_run2_anchor.{ST.ON_POLICY} {ro}.{k}"] == fmt(aq("sensitivity_run2_anchor", f"{ST.ON_POLICY} {ro}", k),
                                                                                    ST.k2s if k.startswith("delta") else ST.k2) for ro in ST.CR for k in ("main", "yardstick"))
      and not any(k.startswith("24.r2.sensitivity_run2_anchor.sol_a_uncut") for k in sd))
pr_ = PL["rows"]["0.05"]["R"]["refusal|uncut R"]["-5"]
check("(11) PLANT: each row verdict (with the reason when unavailable), aggregate verdict + shift, and count are their own",
      sd["25.plant.verdict.0.05.refusal|uncut R.-5"] == (f"{pr_['verdict']} ({pr_['reason']})" if "reason" in pr_ else pr_["verdict"])
      and all(sd[f"25.plant.verdict.{f}.{x} {ro}.{t}"].split(" (")[0] == PL["rows"][f][ro][f"{x} {ro}"][t]["verdict"]
              for f in ("0.01", "0.05") for ro in ST.CR for x in ST.SIX for t in ST.PLANT_TAGS)
      and sd["25.plant.aggregate.0.01.C.-3.excess"] == "X-0.01-C--3"
      and sd["25.plant.aggregate.0.05.R.+8.S"] == "S-0.05-R-+8; shift " + ST.summ_str(PL["aggregate"]["by_level"]["0.05"]["R"]["+8"]["S"]["shift"], ST.ppv)[0]
      and all(sd[f"25.plant.count.{f}.{t}.{ro}"] == count_str(PL["counts"][f][t][ro]) for f in ("0.01", "0.05") for t in ST.PLANT_TAGS
              for ro in ("C", "R", "C+R")))
DET3 = copy.deepcopy(DET)
DET3["meta"]["definitions"].pop("y")
DET4 = copy.deepcopy(DET)
DET4["meta"]["definitions"]["y"] = 3
DET5 = copy.deepcopy(DET)
DET5["meta"]["definitions"]["y"] = "  "
check("(11) 23.definition.y carries the record's meta.definitions.y verbatim (2e 10-18); absent it is missing, not filled; a non-string or blank STOPs",
      sd.get("23.definition.y") == YDEF and rd["full"].get("23.definition.y") == YDEF
      and "23.definition.y" in ST.build(json.loads(json.dumps(DREC | {"DETDROP": DET3})), DSHA | RSHA, DPATH)["missing"]
      and stops(ST.build, json.loads(json.dumps(DREC | {"DETDROP": DET4})), DSHA | RSHA, DPATH)
      and stops(ST.build, json.loads(json.dumps(DREC | {"DETDROP": DET5})), DSHA | RSHA, DPATH))
check("(11) labels: lbl.<role> is each record's own route C sidecar label", all(sd.get(f"lbl.{r}") == f"{LP} | {r}" for r in DSHA))


def dstop(**over):
    r_ = copy.deepcopy(DREC)
    for k, v in over.items():
        if v is None:
            r_.pop(k)
        else:
            r_[k] = v
    return stops(ST.build, json.loads(json.dumps(r_)), DSHA | RSHA, DPATH)


def cstop(**over):  # a CONTROL run over the 171-pool registry
    r_ = copy.deepcopy(CREC) | over
    return stops(ST.build, json.loads(json.dumps(r_)), DSHA | RSHA, DPATH, True)


check("(11) gates: a failed kappa / anchor gate, PLANT n_fail > 0, a kappa extra tied to another kappa record or with other members STOP",
      dstop(KAPPA=KAP | {"gate": {"pass": False}}) and dstop(ANCHOR=ANC | {"gate": {"pass": False}}) and dstop(PLANT=PL | {"gates": {"n": 6, "n_fail": 1}})
      and dstop(KAPPA_EXTRA=KX_ | {"meta": {"label": LP, "record_sha256": "8" * 64}})
      and dstop(KAPPA_EXTRA=KX_ | {"five_probe_sensitivity": KX_["five_probe_sensitivity"] | {"members": list(ST.SIX)}}))
DET2 = copy.deepcopy(DET)
DET2["result"]["probes"]["sol_a_uncut C"]["commands L48"]["sim_score_vs_drop"]["n_cells"] = 10.5
check("(11) a downstream count that is not an int STOPs (PLANT detected, detector_vs_drop n_cells)",
      dstop(PLANT=PL | {"counts": {f: {t: {ro: c | {"detected": float(c["detected"])} for ro, c in v2.items()} for t, v2 in v1.items()}
                                   for f, v1 in PL["counts"].items()}})
      and dstop(DETDROP=DET2))
check("(11) without the downstream records, not_carried names them and nothing else changes",
      {"KAPPA", "KAPPA_EXTRA", "DETDROP", "ANCHOR", "PLANT"} <= set(r["not_carried"])
      and not any(k.startswith(("21.", "22.", "23.", "24.", "25.", "lbl.KAPPA", "lbl.DETDROP", "lbl.ANCHOR", "lbl.PLANT")) for k in sl) and sl.get("lbl.A") == LAB)

# (12) 09's ONE label rule for every slot (10-17): registry (sha) -> its label, which must contain 171-pool; else a route C sidecar; else STOP
RC = "route C pooled A (full-length, run1-anchored); PRELIMINARY until S2"


def lab(recs, shas, paths, control=False):  # -> the label strings, or "STOP"
    try:
        return {k: v for k, v in ST.build(json.loads(json.dumps(recs)), shas | RSHA, paths, control)["slots"].items() if k.startswith("lbl.")}
    except SystemExit:
        return "STOP"


reg_a = [{"record": "x/a.json", "record_sha256": SHA_A, "label": LAB}]
check("(12) A in the registry: lbl.A is the registry label (its in-record label is never read)",
      lab({"A": A, ST.REGISTRY: reg_a}, {"A": SHA_A}, {"A": "x/a.json"}) == {"lbl.A": LAB}
      and lab({"KAPPA": KAP | {"meta": {"label": "route C, no 171 here"}}, ST.REGISTRY: REG}, DSHA, DPATH, True) == {"lbl.KAPPA": f"{L171} | KAPPA"})
check("(12) positive control 1: a record in neither the registry nor a sidecar STOPs (A with no registry; ANCHOR left out of it; B beside A)",
      lab({"A": A}, {"A": SHA_A}, {"A": "x/a.json"}) == "STOP" and dstop(ANCHOR_label=None)
      and cstop(**{ST.REGISTRY: [e for e in REG if e["record_sha256"] != SHAN]})
      and lab({"A": A, "B": B, ST.REGISTRY: reg_a}, {"A": SHA_A, "B": "b" * 64}, {"A": "x/a.json", "B": "x/b.json"}) == "STOP")
check("(12) positive control 2: a registry entry naming the record (same path, or same file name elsewhere, .gz or not) with another sha STOPs",
      cstop(**{ST.REGISTRY: REG + [{"record": DPATH["PLANT"], "record_sha256": "9" * 64, "label": L171}]})
      and lab({"A": A, ST.REGISTRY: reg_a + [{"record": "y/z/a.json", "record_sha256": "9" * 64, "label": LAB}]},
              {"A": SHA_A}, {"A": "xfer/a.json"}) == "STOP"
      and lab({"A": A, ST.REGISTRY: reg_a + [{"record": "y/z/b.json.gz", "record_sha256": "9" * 64, "label": LAB}]},
              {"A": SHA_A}, {"A": "xfer/b.json"}) == "STOP"
      and lab({"A": A, ST.REGISTRY: [{"record": "x/a.json", "record_sha256": "9" * 64, "label": LAB}]}, {"A": SHA_A}, {"A": "x/a.json"}) == "STOP")
check("(12) a registry label without 171-pool STOPs",
      cstop(**{ST.REGISTRY: [e | {"label": "route C pooled A"} if e["record_sha256"] == SHD else e for e in REG]}))
check("(12) a route C record: its route C sidecar passes and is lbl.<role> (any role, e.g. A and ANCHOR)",
      lab({"A": A, "A_label": side(SHA_A, RC)}, {"A": SHA_A}, {"A": "x/a.json"}) == {"lbl.A": RC}
      and lab({"ANCHOR": ANC, "ANCHOR_label": side(SHAN, RC), ST.REGISTRY: REG[:3]}, DSHA, DPATH) == {"lbl.ANCHOR": RC})
check("(12) a sidecar not naming route C (e.g. a 171-pool label: the old anchor / PLANT sidecars), naming another record, or with other keys STOPs",
      all(lab({"A": A, "A_label": s_}, {"A": SHA_A}, {"A": "x/a.json"}) == "STOP"
          for s_ in (side(SHA_A, L171), side("9" * 64, RC), side(SHA_A, RC) | {"extra": 1}, [side(SHA_A, RC)])))
check("(12) a registry record that also has a sidecar STOPs (two labels for one record)", cstop(ANCHOR_label=side(SHAN, RC)))
check("(12) a sidecar with no record it labels STOPs", lab({"A": A, "A_label": side(SHA_A, RC), "B_label": side("b" * 64, RC)},
                                                           {"A": SHA_A}, {"A": "x/a.json"}) == "STOP")
check("(12) registry shape: {entries: [...]} is accepted; empty, an entry missing a key, a non-string value, a repeated path or sha STOP",
      not cstop() and not cstop(**{ST.REGISTRY: {"entries": REG}}) and cstop(**{ST.REGISTRY: []})
      and cstop(**{ST.REGISTRY: REG + [{"record": "x/other.json", "record_sha256": "7" * 64}]})
      and cstop(**{ST.REGISTRY: REG + [{"record": "x/other.json", "record_sha256": "7" * 64, "label": 171}]})
      and cstop(**{ST.REGISTRY: REG + [{"record": DPATH["KAPPA"], "record_sha256": "7" * 64, "label": L171}]})
      and cstop(**{ST.REGISTRY: REG + [{"record": "x/other.json", "record_sha256": SHK, "label": L171}]}))
mn_ = {a_.arg for a_ in fn["main"].args.args} - {"out", "control"}
check("(12) the runner takes the registry and a sidecar for every record role, each a main() flag",
      RS.ROLES.get("labels_171pool") == ST.REGISTRY and set(RS.ROLES) == mn_
      and {v for k, v in RS.ROLES.items() if k.endswith("_label")} == {f"{r}_label" for r in ST.RECORD_ROLES}
      and set(ST.RECORD_ROLES) <= set(RS.ROLES.values()))

# (13) Aswin's ruling (10-17, via 0d and 2e): the appendix is route C only. The 171-pool registry's labels only cite v83's PRELIMINARY records,
# so an appendix-bound record with one STOPs unless the run is a CONTROL run (output never used by writing; file name carries CONTROL_ONLY)
c_reg = lambda r: [{"record": f"x/{r}.json", "record_sha256": "c" * 64, "label": L171}]  # noqa: E731
check("(13) the appendix-bound roles are exactly kappa, the kappa extra, detector_vs_drop, the anchor, PLANT, B and R37 (not A or its group records)",
      set(ST.APPENDIX_ROLES) == {"KAPPA", "KAPPA_EXTRA", "DETDROP", "ANCHOR", "PLANT", "B", "R37"})
check("(13) each appendix-bound record with a 171-pool registry label STOPs in a writing run and passes in a CONTROL run",
      all(lab({r: CREC[r], ST.REGISTRY: REG}, DSHA, DPATH) == "STOP" and lab({r: CREC[r], ST.REGISTRY: REG}, DSHA, DPATH, True) == {f"lbl.{r}": f"{L171} | {r}"}
          for r in DSHA)
      and all(lab({r: rec_, ST.REGISTRY: c_reg(r)}, {r: "c" * 64}, {r: f"x/{r}.json"}) == "STOP"
              and lab({r: rec_, ST.REGISTRY: c_reg(r)}, {r: "c" * 64}, {r: f"x/{r}.json"}, True) == {f"lbl.{r}": L171} for r, rec_ in (("B", B), ("R37", R))))
check("(13) a writing run takes route C sidecars on every appendix-bound record, and a 171-pool label on A",
      not dstop() and lab({"A": A, ST.REGISTRY: reg_a}, {"A": SHA_A}, {"A": "x/a.json"}) == {"lbl.A": LAB})
check("(13) the `mode` slot names the run: 'writing', or CONTROL_ONLY ... never used by writing ... not an appendix",
      rd["slots"]["mode"] == "writing" and ST.build(json.loads(json.dumps(CREC)), DSHA | RSHA, DPATH, True)["slots"]["mode"] == ST.CONTROL_MODE
      and ST.CONTROL_MODE.startswith("CONTROL_ONLY") and "never used by writing" in ST.CONTROL_MODE and "not an appendix" in ST.CONTROL_MODE)
ma_ = ast.unparse(fn["main"])
check("(13) a63478 R-1: a writing run with an appendix-bound record and no registry STOPs, even with a route C sidecar (its probe: the 171-pool"
      " anchor relabelled route C); a CONTROL run may omit it; A needs no registry",
      lab({"ANCHOR": ANC, "ANCHOR_label": side(SHAN, RC)}, DSHA, DPATH) == "STOP"
      and all(lab({r: rec_, f"{r}_label": side("c" * 64, RC)}, {r: "c" * 64}, {r: f"x/{r}.json"}) == "STOP" for r, rec_ in (("B", B), ("R37", R)))
      and lab({"ANCHOR": ANC, "ANCHOR_label": side(SHAN, RC), ST.REGISTRY: REG}, DSHA, DPATH) == "STOP"
      and lab({"ANCHOR": ANC, "ANCHOR_label": side(SHAN, RC)}, DSHA, DPATH, True) == {"lbl.ANCHOR": RC}
      and lab({"A": A, "A_label": side(SHA_A, RC)}, {"A": SHA_A}, {"A": "x/a.json"}) == {"lbl.A": RC})
mb_ = fn["main"].body
at_ = lambda s_: [i for i, st_ in enumerate(mb_) if s_ in ast.unparse(st_)]  # noqa: E731  (main's top-level statements containing s_)
co_ = [i for i, st_ in enumerate(mb_) if isinstance(st_, ast.Expr) and ast.unparse(st_) == "check_out(out, control)"]
cm_ = [i for i, st_ in enumerate(mb_) if isinstance(st_, ast.Expr) and ast.unparse(st_) == "check_mode(res, control)"]
md_ = dict(zip([a_.arg for a_ in fn["main"].args.args][-len(fn["main"].args.defaults):], fn["main"].args.defaults))
check("(13) a63478 O1: the reply's mode (slots and full) must be the one asked for; main checks it right after the paid call, before writing",
      not stops(RS.check_mode, {"slots": {"mode": "writing"}, "full": {"mode": "writing"}}, False)
      and not stops(RS.check_mode, {"slots": {"mode": ST.CONTROL_MODE}, "full": {"mode": ST.CONTROL_MODE}}, True)
      and stops(RS.check_mode, {"slots": {"mode": "writing"}, "full": {"mode": "writing"}}, True)
      and stops(RS.check_mode, {"slots": {"mode": ST.CONTROL_MODE}, "full": {"mode": "writing"}}, True)
      and stops(RS.check_mode, {"slots": {"mode": "writing"}, "full": {}}, False)
      and len(cm_) == 1 and at_("readout.remote(") == [cm_[0] - 1] and cm_[0] < min(at_("open(out, 'xb')") or [-1]))
check("(13) a63478 O2: check_out is an unconditional top-level call before the paid call; the marker counts only in the FILE NAME (a"
      " CONTROL_ONLY folder STOPs); --control defaults to False",
      len(co_) == 1 and co_[0] < min(at_("readout.remote(") or [-1]) and stops(RS.check_out, "r/CONTROL_ONLY/slots.json", True)
      and not stops(RS.check_out, "r/CONTROL_ONLY/slots.json", False) and isinstance(md_.get("control"), ast.Constant) and md_["control"].value is False)
reg_line = [ln for ln in (HERE / "NOTE_slot_readout.md").read_text().split("\n") if ln.startswith("REG=")]
check("(13) 2e (10-17): a registry whose bytes do not hash to 09's verified one (pinned in code) STOPs, in writing and CONTROL runs alike;"
      " the pin is the sha every REG pointer in the NOTE names (ctrl2, ctrl2b)",
      stops(ST.build, json.loads(json.dumps(DREC)), DSHA | {ST.REGISTRY: "e" * 64}, DPATH)
      and stops(ST.build, json.loads(json.dumps(CREC)), DSHA | {ST.REGISTRY: "e" * 64}, DPATH, True)
      and stops(ST.build, json.loads(json.dumps(CREC)), DSHA, DPATH, True) and not cstop() and not dstop()
      and len(reg_line) >= 1 and all(ln.endswith("LABELS_171pool_2026-10-17.json@" + ST.REGISTRY_SHA256) for ln in reg_line))
check("(13) the runner: --control and an output name containing CONTROL_ONLY go together (either alone STOPs), checked before the paid call;"
      " the container gets the flag",
      not stops(RS.check_out, "r/slots_downstream_171pool_CONTROL_ONLY_ctrl2.json", True) and not stops(RS.check_out, "r/slots.json", False)
      and stops(RS.check_out, "r/slots.json", True) and stops(RS.check_out, "r/slots_CONTROL_ONLY.json", False)
      and "check_out(out, control)" in ma_ and "readout.remote(inputs, code, control)" in ma_
      and ma_.index("check_out(out, control)") < ma_.index("readout.remote(inputs, code, control)"))

# (14) 0d's ctrl2 diff against v83 (10-17): three mechanical fixes before route C
def dslots(**over):  # a writing run's slots over DREC with some records replaced
    return ST.build(json.loads(json.dumps(DREC | over)), DSHA | RSHA, DPATH)["slots"]


s23 = sd[f"23.spearman.{ST.SIX[0]} C.{ST.DETECTORS[0]}.{ST.DET_ANALYSES[0]}"]
check("(14) slot 23: the Spearman point and interval ends all carry an explicit sign (k2s; 0d's ruling, 10-17)",
      ST.summ_str({"point": 0.1, "ci95": [-0.05, 0.2]}, ST.k2s)[0] == f"+0.10 ({M}0.05 to +0.20)"
      and s23.startswith(("+", M)) and s23.split(" (")[1].startswith(("+", M)) and (" to +" in s23 or f" to {M}" in s23))
PLU = copy.deepcopy(PL)
for lb_ in PLU["rows"]["0.05"]["C"]:
    PLU["rows"]["0.05"]["C"][lb_]["+8"] = {"verdict": "unavailable", "reason": "unattainable (ceiling) in a member: x"}
PLU["counts"] = plant_counts(PLU["rows"])
su_ = dslots(PLANT=PLU)
six_ = ", ".join(sorted(f"{x} C" for x in ST.SIX))
check("(14) PLANT counts: detected / missed / unavailable apart; six unavailable (ceiling) cases read '0/6 detected, 0 missed, 6 unavailable', never a miss",
      su_["25.plant.count.0.05.+8.C"] == f"0/6 detected, 0 missed, 6 unavailable (unavailable: {six_})" and "missed:" not in su_["25.plant.count.0.05.+8.C"]
      and su_["25.plant.count.0.05.+8.C+R"] == count_str(PLU["counts"]["0.05"]["+8"]["C+R"])
      and any("missed: " in v and "unavailable: " in v for k, v in sd.items() if k.startswith("25.plant.count.")))
c0_ = PL["counts"]["0.01"]["-12"]["C"]
check("(14) PLANT counts that do not add up STOP (detected + missed + unavailable != cases; a named miss of the wrong kind)",
      dstop(PLANT=PL | {"counts": PL["counts"] | {"0.01": PL["counts"]["0.01"] | {"-12": PL["counts"]["0.01"]["-12"] | {"C": c0_ | {"detected": c0_["detected"] + 1}}}}})
      and dstop(PLANT=PL | {"counts": PL["counts"] | {"0.01": PL["counts"]["0.01"] | {"-12": PL["counts"]["0.01"]["-12"] | {"C": c0_ | {
          "misses": {k: ("unavailable" if v == "not detected" else "not detected") for k, v in c0_["misses"].items()}}}}}})
      and c0_["not_detected"] != c0_["unavailable"])
an_ = lambda b, L: ANC["union_stream"][b][L]["sets"][ST.ANCHOR_SET]  # noqa: E731
hw_ = lambda b, L: (an_(b, L)["r2"]["yardstick"]["ci95"][1] - an_(b, L)["r2"]["yardstick"]["ci95"][0]) / 2  # noqa: E731
hws_ = [hw_("lead", f"{x} {ro}") for x in ST.SIX for ro in ST.CR]
check("(14) slot 24: the anchor's own task count n_cells per label and block (not detector_vs_drop's), with separators",
      all(sd[f"24.n_cells.lead.{x} {ro}"] == c_(an_("lead", f"{x} {ro}")["n_cells"]) for x in ST.SIX for ro in ST.CR)
      and all(sd[f"24.n_cells.sensitivity_run2_anchor.{ST.ON_POLICY} {ro}"] == c_(an_("sensitivity_run2_anchor", f"{ST.ON_POLICY} {ro}")["n_cells"]) for ro in ST.CR)
      and not any(k.startswith("24.n_cells.sensitivity_run2_anchor.sol_a_uncut") for k in sd)
      and sd[f"24.n_cells.lead.{ST.SIX[0]} C"] != sd[f"23.n_cells.{ST.SIX[0]} C.{ST.DETECTORS[0]}.{ST.DET_ANALYSES[0]}"]
      and "," in sd[f"24.n_cells.lead.{ST.SIX[0]} C"])
ANW = copy.deepcopy(ANC)
ANW["union_stream"]["lead"][f"{ST.SIX[1]} R"]["sets"][ST.ANCHOR_SET]["r2"]["yardstick"] |= {"ci95": None, "withheld": "fewer than 95% complete draws"}
sw_ = dslots(ANCHOR=ANW)
check("(14) slot 24: the yardstick R² half-width per label (2 decimals), the lead range min to max; a withheld interval is named, not filled",
      all(sd[f"24.r2.yardstick_halfwidth.lead.{x} {ro}"] == ST.k2(hw_("lead", f"{x} {ro}")) for x in ST.SIX for ro in ST.CR)
      and sd["24.r2.yardstick_halfwidth.lead.range"] == f"{ST.k2(min(hws_))} to {ST.k2(max(hws_))}"
      and sw_[f"24.r2.yardstick_halfwidth.lead.{ST.SIX[1]} R"] == "interval withheld" and sw_["24.r2.yardstick_halfwidth.lead.range"].endswith(" (1 withheld)"))

print(f"{len(bad)} failed" if bad else "ALL PASS")
sys.exit(1 if bad else 0)
