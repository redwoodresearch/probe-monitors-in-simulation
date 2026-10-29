"""Independent draw-0 check of a Stage A all-tasks record (alltasks_CRD_*; 2e 10-15 ~22:00, plan approved): the four public probes x C/R x
S_old, S_new, S_newmains and S_newmains without S4, at the record's levels: every pool's T, k, units, units above, runs and per-run alarm share,
the readable flags, the catches (sim at tau_S, sim at tau_B, run 1 / run 2 at tau_B, run 1 / run 2 at tau_A, S_old's run 1 at tau_B2), the gaps,
the coverage counts; pooled_all, pooled_all_without_S4 and pooled_455 from these per-origin values with each origin's main count as D; and rule
checks by key (without S4 keeps the with-S4 pools; excluded sim-honest cells have no H_sim rows; replaced keys come only from their replacement
root; H_A cells = H_sim cells; no S4 orphan row is taken; the coverage pool is used by no public probe).
INPUTS: the Stage A runner at the record's own commit is called with the record's own arguments, its Modal calls swapped for a capture (no score
file is opened locally, nothing is scored); the captured shipped-input sha256s must equal the record's meta.shipped_inputs. Only this input
resolution is reused; selection rules and arithmetic are written here from the spec (25h, 09's rulings, pools.stage_a). Single-model probes: raw
scores, no fold z. Draw 0: every main's multiplicity is 1, so T is the plain order statistic and a catch is the mean over mains of the mean over
the main's cohort cells. Every file read is sha256-checked. Modal CPU, minutes.

  Resolve inputs (no score file opened; writes the input listing for the READ_LOG notice):
    PYTHONPATH=src <modal> run experiments/2026-10-15_e2e_sim_pipeline/check_stagea_alltasks.py --commit <c> --record <alltasks_CRD_*.json>
  Then the check, pinned to that listing:
    PYTHONPATH=src <modal> run experiments/2026-10-15_e2e_sim_pipeline/check_stagea_alltasks.py --commit <c> --record <r> --launch --inputs-sha <sha>
G7b FOLD-MODEL (0d's release gate via 2e, 10-16): add --pw-catalogue <...> --fold-labels <Stage 1 or Stage 2 families> to the same three steps;
--shapes (with --launch --inputs-sha) first reads every pinned file those labels read, keys / shapes only, on Modal CPU. After the check,
--compare-g7b-new <its results file> (local, keys only) compares the 8 public labels' gates with the earlier G7b PASS record by rule key.
"""

import gzip
import hashlib
import io
import itertools
import json
import math
import numbers
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
D507 = "experiments/2026-10-15_fpr_sweep_507_prep"
TREES = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/g7bfold/stagea_trees")  # G7b
# FOLD-MODEL: this fork's own detached code trees (E2E's folder was ...scratchpad/e2e_fork/stagea_trees)
W2_APPS = tuple(f"probe-shift_aswin-surya_sub-rerun_rc1bw2{r_}_kill-2026-11-19" for r_ in ("a1", "a2", "b1", "b2"))  # rc1b W2 (AMEND1): one app per run
APP_NAME = __import__("os").environ.get("PROBE_SHIFT_W2_APP", W2_APPS[0])  # the launching shell names the run's app; the container's import
assert APP_NAME in W2_APPS, ("rc1b W2: PROBE_SHIFT_W2_APP must be one of the four W2 apps", APP_NAME)  # keeps the default (its functions are bound)
app = modal.App(APP_NAME)  # rc1b W2 (COMMON_RULES app names; PLANT V3 was sub-rerun_plantv3)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")  # PLANT V3: block (v)'s two modules, at /root (rule 27)
         .add_local_file(HERE / "plant_rows_v3_port.py", "/root/plant_rows_v3_port.py").add_local_file(HERE / "plant_rows_v3.py", "/root/plant_rows_v3.py"))
VOLS = {"solavg": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "p2": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13",
        "conv": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"}
MOUNT = {"solavg": "/sol", "p2": "/p2", "conv": "/conv"}
RESVOL = "probe-shift_aswin-surya_sub-e2esim_results_kill-2026-11-14"  # detach (2e 09:1x): per-label outputs persist here, one exclusive shard each
res_vol = modal.Volume.from_name(RESVOL, create_if_missing=True, version=2)
PUBLIC = ["FIBS", "harmfulness", "refusal", "high-stakes"]
ORIGINS = ["S_old", "S_new", "S_newmains", "S_newmains without S4"]
SWEEP_ORIGINS = ORIGINS + ["union", "union without S4"]  # union: the shared sim-honest pool over S_new u S_newmains (labelled deviation in the record)
A3 = ("run1", "run2", "sim")
COMBO_ROLE = {"pooled_all": "lead", "pooled_all_without_S4": "lead (beside pooled_all; S4 is a labelled batch)", "pooled_455": "lead-2"}  # the runner's roles


def _bool(x):  # a boolean: Python's or numpy's (np.bool_ is not a numbers.Real; G7b VALUE, Astra r5 #4: booleans compare only with booleans)
    return isinstance(x, bool) or getattr(getattr(x, "dtype", None), "kind", None) == "b"


def _real(x):  # a number (never a boolean: eq() sends booleans to their own exact branch first)
    return isinstance(x, numbers.Real) and not _bool(x)


def _int(x):  # an integer count (Python or numpy integer), never a bool
    return isinstance(x, numbers.Integral) and not _bool(x)


def eq(a_, b_):  # one comparison: numbers within 1e-9; NaN equals only NaN (never a number, a missing value or text); booleans and integers exact
    if _bool(a_) or _bool(b_):  # G7b VALUE (Astra r5 #4): True never equals 1, 1.0 or a non-boolean; numpy booleans are booleans
        return _bool(a_) and _bool(b_) and bool(a_) == bool(b_)
    if isinstance(a_, dict) or isinstance(b_, dict):  # nested values by the same rules, key sets exact (Python's == would equal True and 1)
        return isinstance(a_, dict) and isinstance(b_, dict) and set(a_) == set(b_) and all(eq(a_[k_], b_[k_]) for k_ in a_)
    if isinstance(a_, (list, tuple)) or isinstance(b_, (list, tuple)):
        return isinstance(a_, (list, tuple)) and isinstance(b_, (list, tuple)) and len(a_) == len(b_) and all(eq(x_, y_) for x_, y_ in zip(a_, b_))
    if _int(a_) or _int(b_):  # G7b VALUE (Astra r5 #4): an integer count compares EXACTLY (no tolerance; an integral float equal to it passes)
        return _real(a_) and _real(b_) and not (isinstance(a_, float) and math.isnan(a_)) and not (isinstance(b_, float) and math.isnan(b_)) and a_ == b_
    if isinstance(a_, float) or isinstance(b_, float):
        if not (_real(a_) and _real(b_)):
            return False
        if math.isnan(a_) or math.isnan(b_):
            return math.isnan(a_) and math.isnan(b_)
        if math.isinf(a_) or math.isinf(b_):
            return a_ == b_
        return abs(a_ - b_) <= 1e-9 * max(1.0, abs(a_), abs(b_))  # G7b VALUE (2e / 09 22:4x): 1e-9 relative, floored at 1e-9 absolute
    return a_ == b_


def nan_both(a_, b_):
    return isinstance(a_, float) and isinstance(b_, float) and math.isnan(a_) and math.isnan(b_)


def compare(cmp, empty):  # -> mismatches, and the comparisons with NaN on both sides (equal), labelled: an empty block (no cohort cells) or k < 1
    n_ = sum(nan_both(a_, b_) for _, a_, b_ in cmp)  # G7b VALUE (Astra r5 #5): the keyed ledger of EVERY comparison: keys + verdicts (released),
    vd_ = [(p_, a_, b_, "MATCH" if eq(a_, b_) else "MISMATCH") for p_, a_, b_ in cmp]  # and with both values (the Volume record only; 2e 23:0x)
    return {"mismatches": [[p_, a_, b_] for p_, a_, b_, v_ in vd_ if v_ == "MISMATCH"], "ledger": [[p_, v_] for p_, _, _, v_ in vd_],
            "ledger_values": [[p_, a_, b_, v_] for p_, a_, b_, v_ in vd_],
            "nan_on_both_sides": {"n": n_, "label": (None if not n_ else "empty block (no cohort cells)" if empty else "unavailable at draw 0 (a pool threshold with k < 1)")}}


def relaunch_alias(k_, where):
    """G3b, the runner's own rule (run_peraction311.relaunch_alias @ 20c6fbd20 l.443-460), ported: full-length uncut relaunch files key a run as
    "honest_b_relaunch|<task>_r<k>"; the frozen members file says "honest_r<k>|<task>" (k 1-5). Keys already in the members' form pass; any other
    key STOPs; the rename must be one-to-one by run key. -> (keys, renamed count)."""
    import numpy as np

    out, n = [], 0
    for x in k_.tolist():
        w, _, t = x.partition("|")
        mo = re.fullmatch(r"(.+)_r([1-5])", t) if w == "honest_b_relaunch" else None
        if re.fullmatch(r"honest_r[1-5]", w) and t:
            out.append(x)
        elif mo:
            out.append(f"honest_r{mo.group(2)}|{mo.group(1)}")
            n += 1
        else:
            raise AssertionError(f"{where}: relaunch key {x!r} is neither honest_r<k>|<task> nor honest_b_relaunch|<task>_r<k> (the runner STOPs)")
    assert len(set(out)) == len(set(k_.tolist())), f"{where}: the relaunch key alias is not one-to-one (the runner STOPs)"
    return np.array(out), n


def dweighted(vals, ws):  # D-weighted mean over member origins; every member an empty block (total D 0) gives NaN, as the runner's numpy 0/0
    t_ = sum(ws)
    return sum(v * w for v, w in zip(vals, ws)) / t_ if t_ else float("nan")


# ---- G6: co-headline S/Y and the PANEL excess at draw 0 (0d's partial ruling via 2e 13:1x). The stated gap, verbatim:
GAP_TEXT = ("independently reproduced except the fold-model probes' per-probe rows (sol_a_uncut, own_all_rs_uncut_L48, Sol (a)|uncut, own_all_rs_L48|uncut), "
            "whose values enter every S/Y and excess aggregate (six-, five- and four-probe) as recorded; the four external probes' values and the aggregation "
            "itself (S, Y, |Y|, mean G − mean Y, incl. the readable and saturation gates) are reproduced independently")
AS_RECORDED = ("sol_a_uncut", "own_all_rs_uncut_L48", "Sol (a)|uncut", "own_all_rs_L48|uncut")  # the gap's families: their rows are read from the record
EXT_UNC = tuple(p_ + "|uncut" for p_ in PUBLIC)
FULL_GROUPS_ = (("headline_six_full", ("sol_a_uncut",) + EXT_UNC + ("own_all_rs_uncut_L48",)), ("off_policy_five_full", ("sol_a_uncut",) + EXT_UNC),
                ("registered_scored_uncut_six", ("Sol (a)|uncut",) + EXT_UNC + ("own_all_rs_L48|uncut",)))  # full_override.HEADLINE_SIX / OFF_FIVE / REG_SIX
AUROC06_FULL = ("sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "own_all_rs_uncut_L48")  # 09 (d) @ 19ba8f563 (Sol (a), FIBS, harmfulness, own_all_rs_L48), uncut
STREAMS = ("pooled", "pooled_abs_per_origin", "S_old", "S_new", "S_newmains", "S_newmains without S4")  # the panel's streams (a per-origin stream: member rows only)


def agg_draw0(rows, groups_sy, groups_panel, pw_labels=None):
    """The co-headline S/Y and the PANEL groups at draw 0 (as pool_reduce_n defines them), from per-label draw-0 inputs.
    rows: {label: {"df": Delta_full, "yr": run2@B - run1@B, "rd": readable (all pools), "st": {stream: [G, Y]}, "sat": {stream: saturated}, "rdo": {origin: readable}}}.
    S = mean Delta_full over the group's six (or five, four), Y = mean (run2@B - run1@B), |Y|; a missing or unreadable member makes the group unavailable.
    PANEL: per stream and readout (R, C, R+C with both present), mean G, mean Y, mean G - mean Y over the readable rows (primary) and over the readable
    unsaturated rows (without_saturated); a per-origin stream reads that origin's own readable flag. pw_labels (a pooled record, runner @ 8a15fb10f):
    the streams are pooled and pooled_abs_per_origin only; a group with a member outside the pooled labels is omitted (listed in the third return);
    a PANEL group is complete or withheld: any unreadable member withholds both variants. -> (coh, pan, omitted)"""
    import numpy as np

    coh, oos = {}, set()
    for gname, fams in groups_sy:
        for ro in ("C", "R"):
            want = [f"{fa} {ro}" for fa in fams]
            if pw_labels is not None and not set(want) <= pw_labels:
                oos.add(f"coheadline_signed: {gname}|{ro}")
                continue
            miss = [lb for lb in want if lb not in rows]
            unr = [lb for lb in want if lb in rows and not rows[lb]["rd"]]
            if miss or unr:
                coh[f"{gname}|{ro}"] = {"unavailable": f"rows missing {miss} or not readable {unr}"}
                continue
            Y_ = float(np.mean([rows[lb]["yr"] for lb in want]))
            coh[f"{gname}|{ro}"] = {"S": float(np.mean([rows[lb]["df"] for lb in want])), "Y": Y_, "abs_Y": abs(Y_), "rows": want}
    pan = {}
    for st in (("pooled", "pooled_abs_per_origin") if pw_labels is not None else STREAMS):
        for gname, fams in groups_panel:
            for sel, ros in (("R", "R"), ("C", "C"), ("R+C", "RC")):
                memb = [lb for lb in rows if lb.rsplit(" ", 1)[0] in fams and lb.rsplit(" ", 1)[1] in ros and st in rows[lb]["st"]]
                if pw_labels is not None:  # a pooled group is complete or absent; scope first (item 45(3) / 46)
                    exp_ = {f"{fa} {r_}" for fa in fams for r_ in ros}
                    if not exp_ <= pw_labels:
                        oos.add(f"{gname}|{sel}")
                        continue
                    if set(memb) != exp_:  # the runner STOPs here: a group that cannot match any record
                        pan.setdefault(st, {})[f"{gname}|{sel}"] = {"members without rows": sorted(exp_ - set(memb))}
                        continue
                if not memb or (sel == "R+C" and {lb.rsplit(" ", 1)[1] for lb in memb} != {"R", "C"}):
                    continue

                def ok_(lb, sat_ok):
                    return (rows[lb]["rd"] if st in ("pooled", "pooled_abs_per_origin") else rows[lb]["rdo"][st]) and (sat_ok or not rows[lb]["sat"][st])
                unr_ = [lb for lb in memb if not ok_(lb, True)]
                if pw_labels is not None and unr_:  # an unreadable member withholds the whole group (both variants), never a subset mean
                    pan.setdefault(st, {})[f"{gname}|{sel}"] = {tag: {"rows": [], "withheld": f"unreadable members {unr_} (item 46: no probe omission; "
                                                                      "the group is withheld)"} for tag in ("primary", "without_saturated")} | {"unreadable_rows": unr_}
                    continue
                o_ = {}
                for tag, sat_ok in (("primary", True), ("without_saturated", False)):
                    use = [lb for lb in memb if ok_(lb, sat_ok)]
                    if not use:
                        o_[tag] = {"rows": [], "unavailable": "no readable rows"}
                        continue
                    Gm, Ym = float(np.mean([rows[lb]["st"][st][0] for lb in use])), float(np.mean([rows[lb]["st"][st][1] for lb in use]))
                    o_[tag] = {"rows": use, "n": len(use), "mean_G": Gm, "mean_Y": Ym, "mean_G_minus_mean_Y": Gm - Ym}
                o_["unreadable_rows"] = unr_
                pan.setdefault(st, {})[f"{gname}|{sel}"] = o_
    return coh, pan, oos


def agg_rows(rec, vals, cb, md, key, fams):
    """Per-label draw-0 inputs at (combo, mode, level): the external probes from this run's check() calls (vals), the gap's fold-model probes as
    recorded; a label is a row only where its pooled block was computed (the runner's panel_rows). -> (rows, as-recorded labels used, mismatches)"""
    rows, used, mism = {}, [], []
    for lb in sorted(set(rec["pooled"]) | set(vals)):
        fam = lb.rsplit(" ", 1)[0]
        if fam not in fams:
            continue
        q = rec["pooled"].get(lb, {}).get(cb, {}).get(md, {}).get(key)
        if lb in vals:
            v = vals[lb].get(cb, {}).get(key)
            if (v is None) != (q is None):
                mism.append([f"{cb}|{md}|{key}|{lb}|pooled block computed here vs in the record"])
            if v is not None:
                rows[lb] = v
        elif fam in AS_RECORDED:
            if q is not None:  # as recorded (the stated gap): Delta_full, run2@B - run1@B, the readable flags, each stream's G, Y and saturation at draw 0
                pn = q["panel"]
                rows[lb] = {"df": q["gap"]["delta_full"]["point"], "yr": q["catch"]["run2"]["point"] - q["catch"]["run1"]["point"], "rd": q["readable"]["all"],
                            "st": {s_: [pn[s_]["G"]["point"], pn[s_]["Y"]["point"]] for s_ in pn}, "sat": {s_: pn[s_]["saturated"] for s_ in pn},
                            "rdo": {s_: pn[s_]["readable"] for s_ in pn if s_ not in ("pooled", "pooled_abs_per_origin")}}
                used.append(lb)
        else:
            mism.append([f"{cb}|{md}|{key}|{lb}|a group member neither reproduced here nor in the stated gap"])
    return rows, used, mism


def agg_compare(mine_coh, mine_pan, rc, rp, where):
    """-> value comparisons (key, mine, recorded) and structural mismatches, mine against the record's coheadline_signed (rc) and panel_groups (rp)
    at one (combo, mode, level); estimator (B)'s S|B and caveat_B are not compared (A selected)."""
    cmp, mism = [], []
    if sorted(rc) != sorted(mine_coh):
        mism.append([f"{where}|coheadline groups"])
    for g, m in mine_coh.items():
        r = rc.get(g, {})
        if "unavailable" in m:
            if r != m:
                mism.append([f"{where}|coh|{g}|unavailable"])
            continue
        if r.get("rows") != m["rows"] or not {"S", "Y", "abs_Y", "rows"} <= set(r) <= {"S", "Y", "abs_Y", "rows", "S|B", "caveat_B"}:
            mism.append([f"{where}|coh|{g}|rows or fields"])
        cmp += [(f"{where}|coh|{g}|{x}", m[x], r.get(x, {}).get("point")) for x in ("S", "Y", "abs_Y")]
    if sorted(rp) != sorted(mine_pan):
        mism.append([f"{where}|panel streams"])
    for st, gs in mine_pan.items():
        if sorted(rp.get(st, {})) != sorted(gs):
            mism.append([f"{where}|panel|{st}|groups"])
        for g, m in gs.items():
            r = rp.get(st, {}).get(g, {})
            if "members without rows" in m:  # a pooled group the runner STOPs on: no record can match it
                mism.append([f"{where}|panel|{st}|{g}|pooled group members without rows"])
                continue
            if sorted(r.get("unreadable_rows", [None])) != sorted(m["unreadable_rows"]) or sorted(r) != sorted(m):
                mism.append([f"{where}|panel|{st}|{g}|unreadable rows or fields"])
            for tag in ("primary", "without_saturated"):
                mt, rt = m[tag], r.get(tag, {})
                if "withheld" in mt:  # a pooled group withheld for unreadable members (the text lists them in the runner's row order: compared as a set)
                    if sorted(rt) != ["rows", "withheld"] or rt.get("rows") != [] or not str(rt.get("withheld", "")).startswith("unreadable members ") \
                            or not str(rt["withheld"]).endswith("(item 46: no probe omission; the group is withheld)"):
                        mism.append([f"{where}|panel|{st}|{g}|{tag}|withheld"])
                    continue
                if "unavailable" in mt:
                    if rt != mt:
                        mism.append([f"{where}|panel|{st}|{g}|{tag}|unavailable"])
                    continue
                if sorted(rt.get("rows", [])) != sorted(mt["rows"]) or rt.get("n") != mt["n"] or sorted(rt) != sorted(mt):  # Astra g6 #3: the field set too
                    mism.append([f"{where}|panel|{st}|{g}|{tag}|rows, n or fields"])
                cmp += [(f"{where}|panel|{st}|{g}|{tag}|{x}", mt[x], rt.get(x, {}).get("point")) for x in ("mean_G", "mean_Y", "mean_G_minus_mean_Y")]
    return cmp, mism


def agg_check(rec, vals, levels, plant=None, levels_all=None, panel_withheld=None):
    """S/Y and the excess over every combo, mode and level of a full-length record. plant = (label, field, stream or None, delta): the positive control,
    one external row's input shifted before aggregating. levels_all (B's level split, A22): the record's full level set; the trees' level structure is
    checked against it while only `levels` are computed (unsplit: the same set). panel_withheld (PLANT V3, AMEND6): the plant job's record carries
    panel_groups = {"withheld": <this text>} (runner L4514 @ 752413181): asserted exactly, and no PANEL group is built or compared (its PANEL
    controls have no expected key: not applicable). -> {comparisons, mismatches, as-recorded labels used, expected plant keys}"""
    meta = rec["meta"]
    assert meta.get("full_length"), "the S/Y and excess check covers full-length records"
    assert tuple(meta["auroc06_subset"]["members"]) == AUROC06_FULL, "auroc06_subset members differ from 09's pin (uncut mapping)"
    import copy

    gsy = FULL_GROUPS_ + (("auroc06_subset", AUROC06_FULL),)
    fams = {fa for _, fs in gsy for fa in fs}
    lvk = sorted(f"{x:g}" for x in (levels_all or levels))  # the record's level structure (B's split: the full set; unsplit: the same as levels)
    lvp_ = {f"{x:g}" for x in levels}  # the levels computed in this run
    assert lvp_ <= set(lvk), "computed levels outside the full level set"
    pwl_ = set(meta["pooled_world"]["labels"]) if meta.get("pooled_world") else None  # a pooled record's scope (groups complete or omitted)
    oos_ = set()
    # Astra g6 #1: the combo domain from the computed inputs (the external labels' vals and the gap families' computed pooled blocks), never from the
    # aggregate trees alone; both trees must hold exactly these combos
    combos = sorted({cb for d_ in vals.values() for cb, lv_ in d_.items() if lv_}
                    | {cb for lb, d_ in rec["pooled"].items() if lb.rsplit(" ", 1)[0] in AS_RECORDED and isinstance(d_, dict)
                       for cb, q_ in d_.items() if isinstance(q_, dict) and any(isinstance(q_.get(md), dict) and q_[md] for md in ("joint", "fixed"))})
    cmp, mism, used, expect = [], [], set(), set()
    rch_ = {k_: v_ for k_, v_ in (rec.get("coheadline_signed") or {}).items() if k_ != "_rule"}  # a pooled record with every group out of scope has none
    rpg_ = rec.get("panel_groups") or {}
    if panel_withheld is not None:  # PLANT V3 (AMEND6): the exact withheld marker, then no PANEL tree to compare
        mism += [] if rec.get("panel_groups") == {"withheld": panel_withheld} else [["panel_groups: not the plant job's exact withheld marker"]]
        rpg_ = {}
    if pwl_ is None and (sorted(rch_) != combos or (panel_withheld is None and sorted(rpg_) != combos)):
        mism.append(["combos: coheadline_signed or panel_groups differ from the combos computed for the reproduced and as-recorded labels"])
    coh_has, pan_has = set(), set()  # pooled (Codex r45 #5): the trees hold exactly the combos with an in-scope group
    for cb in combos:
        exp_pan = {}  # Astra g6 #2: the expected PANEL hierarchy (stream -> mode -> levels), from this aggregation
        for md in ("joint", "fixed"):
            for lv in levels:
                key = f"{lv:g}"
                rows, u_, m_ = agg_rows(rec, vals, cb, md, key, fams)
                used |= set(u_)
                mism += m_
                if plant and plant[0] in rows:
                    rows = copy.deepcopy(rows)
                    lb_, fld_, st_, d_ = plant
                    if st_ is None:
                        rows[lb_][fld_] += d_
                    else:
                        rows[lb_]["st"][st_][0 if fld_ == "G" else 1] += d_
                mc, mp, o0_ = agg_draw0(rows, gsy, FULL_GROUPS_, pwl_)
                oos_ |= o0_
                mp = {} if panel_withheld is not None else mp  # PLANT V3 (AMEND6): no PANEL group built or compared (out-of-scope names kept)
                coh_has |= {cb} if mc else set()
                pan_has |= {cb} if mp else set()
                for st in mp:
                    exp_pan.setdefault(st, {}).setdefault(md, set()).add(key)
                where = f"{cb}|{md}|{key}"
                rp = {st: rpg_[cb][st][md][key] for st in rpg_.get(cb, {}) if key in rpg_[cb][st].get(md, {})}
                c_, m2_ = agg_compare(mc, mp, rch_.get(cb, {}).get(md, {}).get(key, {}), rp, where)
                cmp += c_
                mism += m2_
                if plant and plant[0] in rows:  # the keys the plant must flag: every aggregate whose rows hold the planted label, where that value is finite
                    lb_, fld_, st_, _ = plant
                    for g, m in mc.items():
                        if fld_ == "df" and lb_ in m.get("rows", []) and math.isfinite(m["S"]):
                            expect.add(f"{where}|coh|{g}|S")
                    for st, gs in mp.items():
                        for g, m in gs.items():
                            for tag in ("primary", "without_saturated"):
                                if fld_ == "G" and st == st_ and lb_ in m[tag].get("rows", []):  # Astra g6 #4: each quantity's own finiteness
                                    expect |= {f"{where}|panel|{st}|{g}|{tag}|{x}" for x in ("mean_G", "mean_G_minus_mean_Y") if math.isfinite(m[tag][x])}
        rpc = rpg_.get(cb, {})  # Astra g6 #2: every stream, mode and level of the record's PANEL tree, nothing extra or missing
        if panel_withheld is not None:  # PLANT V3 (AMEND6): no PANEL tree (the marker is asserted above)
            pass
        elif lvp_ == set(lvk):  # unsplit: exactly as before
            if sorted(rpc) != sorted(exp_pan) or any(sorted(rpc[st]) != sorted(exp_pan[st]) or any(sorted(rpc[st][md]) != sorted(exp_pan[st][md]) for md in exp_pan[st])
                                                     for st in exp_pan if st in rpc):
                mism.append([f"{cb}|panel hierarchy (streams, modes, levels)"])
        else:  # B's split: the record's PANEL tree restricted to this part's levels (streams / modes without one dropped) = this part's hierarchy;
            rpp_ = {st: {md: sorted(k_ for k_ in ks_ if k_ in lvp_) for md, ks_ in mds_.items()} for st, mds_ in rpc.items()}  # the other parts check
            rpp_ = {st: {md: ks_ for md, ks_ in mds_.items() if ks_} for st, mds_ in rpp_.items()}  # theirs; and no PANEL level outside the full set
            if {st: mds_ for st, mds_ in rpp_.items() if mds_} != {st: {md: sorted(ks_) for md, ks_ in mds_.items()} for st, mds_ in exp_pan.items()}:
                mism.append([f"{cb}|panel hierarchy (streams, modes, levels)"])
            if any(set(rpc[st][md]) - set(lvk) for st in rpc for md in rpc[st]):
                mism.append([f"{cb}|panel levels outside the full level set"])
        if pwl_ is None or cb in coh_has:  # the co-headline tree's modes and levels, where this combo has a group
            if sorted(rch_.get(cb, {})) != ["fixed", "joint"]:
                mism.append([f"{cb}|coheadline modes"])
            for md in ("joint", "fixed"):
                if sorted(rch_.get(cb, {}).get(md, {})) != lvk:
                    mism.append([f"{cb}|{md}|coheadline levels"])
    if pwl_ is not None and lvp_ == set(lvk) and (sorted(rch_) != sorted(coh_has) or sorted(rpg_) != sorted(pan_has)):  # unsplit: as before (PLANT: both {})
        mism.append(["combos: coheadline_signed or panel_groups differ from the combos with an in-scope group"])
    if pwl_ is not None and lvp_ != set(lvk) and not (set(coh_has) <= set(rch_) and set(pan_has) <= set(rpg_)):  # B's split: a part sees only its
        mism.append(["combos: a combo with an in-scope group at this part's levels is missing from coheadline_signed or panel_groups"])  # levels
    if pwl_ is not None:  # runner @ 8a15fb10f: out-of-scope groups are omitted and listed (the record's list also names the 32k families' PANEL groups,
        gn_ = {g_ for g_, _ in gsy}  # which this aggregation does not build: compared on the S/Y and PANEL groups built here)
        ro_ = {x_ for x_ in ((meta["pooled_world"].get("groups_out_of_scope") or {}).get("groups") or []) if x_.removeprefix("coheadline_signed: ").rsplit("|", 1)[0] in gn_}
        if sorted(oos_) != sorted(ro_):
            mism.append(["pooled groups out of scope differ from meta.pooled_world.groups_out_of_scope"])
    return {"cmp": cmp, "mism": mism, "used": sorted(used), "expect": sorted(expect)}


BOOT_GROUPS = FULL_GROUPS_[:2] + (("auroc06_subset", AUROC06_FULL),)  # the pooled record's in-scope co-headline groups (headline six, off-policy five, (d))


def agg_boot(bvals, combos, levels, ids):
    """G7b VALUE (09 via 2e 23:2x): S and Y at draw 0 and each seeded draw: S = mean over the group's members of the pooled Delta_full, Y = mean of
    run2@B - run1@B at the pooled tau_B (pool_reduce_n l.3305-3320: np.mean over the member arrays, the group's member order), from this run's
    per-draw values and, beside them, from the record npz's per-draw arrays (the runner's own inputs to the mean). A member without values is a
    mismatch (never skipped). -> [(key, mine, the npz's)]"""
    import numpy as np

    cmp = []
    for cb in combos:
        for md in ("joint", "fixed"):
            for lv in levels:
                key = f"{lv:g}"
                for b in ids:
                    for gname, fams in BOOT_GROUPS:
                        for ro in ("C", "R"):
                            want = [f"{fa} {ro}" for fa in fams]
                            x = [((bvals.get(lb) or {}).get(cb) or {}).get(md, {}).get(key, {}).get(str(b)) for lb in want]
                            for nm, k_ in (("S", "df"), ("Y", "yr")):
                                k2_ = f"boot|{cb}|{md}|{key}|draw {b}|{gname}|{ro}|{nm}"
                                if any(x_ is None for x_ in x):
                                    cmp.append((k2_, "member values missing", [lb for lb, x_ in zip(want, x) if x_ is None]))
                                    continue
                                cmp.append((k2_, float(np.mean([x_[k_] for x_ in x])), float(np.mean([x_[k_ + "_ref"] for x_ in x]))))
    return cmp


def value_domain(man, det):
    """Astra r5 #2 / 507-prep 23:11 item 4: the PRE-BUILT required domain of a value run, from the run manifest alone (labels, combos and their member
    rows, levels, modes, the seeded draw ids): every label x combo x level x mode has its pooled block and every member row with n_compared > 0 (an
    early-exit combo has none, so it fails here), and every required per-draw ledger key is in the label's detail shard. det: label -> its detail
    output (None: no shard). -> the missing entries (keys only; [] = complete)"""
    ids = (man.get("boot") or {}).get("ids") or []
    miss = []
    for lb in man["labels"]:
        o_ = det.get(lb)
        if o_ is None:
            miss.append(f"{lb}: no detail shard")
            continue
        vr_ = [q for q in o_["rules"] if q["row"] == lb and q["rule"].startswith("VALUE:")]
        if len(vr_) != 1:
            miss.append(f"{lb}: {len(vr_)} VALUE rules (want 1)")
            continue
        bt_ = vr_[0].get("boot") or {}  # PLANT V3 (Codex V3 #1, round 2): the worker's declared sampled ids and seed are the manifest's
        if ids and (bt_.get("draw_ids") != ids or bt_.get("seed") != (man.get("boot") or {}).get("seed")):
            miss.append(f"{lb}: the worker's sampled draw ids or seed differ from the manifest's")
        cnt = vr_[0].get("counts") or {}
        rk_ = {r_["origin"]: {x_[0] for x_ in r_.get("ledger") or []} for r_ in o_["rows"] if r_["row"] == lb}
        pk_ = {p_["combo"]: {x_[0] for x_ in p_.get("ledger") or []} for p_ in o_["pooled"] if p_["row"] == lb}
        for cb, mem in man["combo_rows"].items():
            for o in mem:
                if ids and "boot|draws_sha256 (the regenerated union stream = this row's attestation)" not in rk_.get(o, set()):
                    miss.append(f"{lb} | {o}: draws attestation not compared")
            for lv in man["levels"]:
                key = f"{lv:g}"
                for md in ("joint", "fixed"):
                    c_ = cnt.get(f"{cb} | {key} | {md}") or {}
                    if c_.get("pooled", 0) <= 0 or sorted(c_.get("rows") or {}) != sorted(mem) or any(n_ <= 0 for n_ in (c_.get("rows") or {}).values()):
                        miss.append(f"{lb} | {cb} | {key} | {md}: draw-0 comparisons missing (pooled block or a member row)")
                    for b in ids:
                        pre = f"boot|{md}|{key}|draw {b}"
                        need_p = {f"{pre}|pooled|gap|{x}" for x in BOOT_POOLED_GAP} | (
                            {f"{pre}|pooled|self-check: per-draw path = draw-0 path|gap|{x}" for x in BOOT_POOLED_GAP} if b == 0 else set())
                        if not need_p <= pk_.get(cb, set()):
                            miss.append(f"{lb} | {cb} | {key} | {md} | draw {b}: pooled per-draw keys missing ({len(need_p - pk_.get(cb, set()))})")
                        need_r = ({f"{pre}|catch|{x}" for x in BOOT_ROW_CATCH} | {f"{pre}|gap|{x}" for x in BOOT_ROW_GAP} | {f"{pre}|D"}
                                  | ({f"{pre}|self-check: per-draw path = draw-0 path|catch|{x}" for x in BOOT_ROW_CATCH} if b == 0 else set()))
                        for o in mem:
                            if not need_r <= rk_.get(o, set()):
                                miss.append(f"{lb} | {o} | {key} | {md} | draw {b}: member per-draw keys missing ({len(need_r - rk_.get(o, set()))})")
    return miss


def ledger_of(o_):
    """A check() detail output's comparisons by (section, entry, key) -> (mine, reference, verdict) (the values ledger: Volume only)."""
    d_ = {}
    for r_ in o_["rows"]:
        for x_ in r_.get("ledger_values") or []:
            d_[("row", r_["origin"], x_[0])] = (x_[1], x_[2] if len(x_) == 4 else None, x_[-1])
    for p_ in o_["pooled"]:
        for x_ in p_.get("ledger_values") or []:
            d_[("pooled", p_["combo"], x_[0])] = (x_[1], x_[2] if len(x_) == 4 else None, x_[-1])
    return d_


def plant_verdict(n_, p_, need):
    """G7b VALUE PLANT (Astra r5 #1 / #3; 507-prep 23:11 items 1-3): n_ / p_: the normal and the planted call's comparisons by key -> (mine, reference,
    verdict). expected = the keys whose value computed here moved (not eq) while their reference did not; flagged = the keys MATCH in the normal call
    and MISMATCH in the planted one. Pass iff flagged == expected, non-empty, both calls hold the same keys, no key moved MISMATCH -> MATCH, a key
    whose reference moved too (a self-check: both sides computed here) stays MATCH, and every required category (need: name -> predicate on a key)
    is in the expected set. -> keys and counts only"""
    both_ = set(n_) & set(p_)
    exp_ = {k_ for k_ in both_ if not eq(n_[k_][0], p_[k_][0]) and eq(n_[k_][1], p_[k_][1])}
    flg_ = {k_ for k_ in both_ if n_[k_][2] == "MATCH" and p_[k_][2] == "MISMATCH"}
    heal_ = {k_ for k_ in both_ if n_[k_][2] == "MISMATCH" and p_[k_][2] == "MATCH"}
    selfm_ = {k_ for k_ in both_ if not eq(n_[k_][1], p_[k_][1]) and p_[k_][2] != "MATCH"}
    cat_ = {nm_: any(f_(k_) for k_ in exp_) for nm_, f_ in need.items()}
    ok_ = bool(exp_) and flg_ == exp_ and set(n_) == set(p_) and not heal_ and not selfm_ and all(cat_.values())
    sk_ = lambda s_: sorted(" | ".join(map(str, k_)) for k_ in s_)[:20]  # noqa: E731
    return {"pass": ok_, "expected_keys": len(exp_), "flagged_keys": len(flg_), "categories": cat_, "missed_keys": sk_(exp_ - flg_),
            "unexpected_keys": sk_(flg_ - exp_), "keys_in_one_call_only": sk_(set(n_) ^ set(p_)), "mismatch_to_match": sk_(heal_),
            "self_checks_not_match": sk_(selfm_)}


BOOT_OOS_S2 = "out_of_scope_stage2; reproduced at S3 @ 2e31d71c4 (2016 compared, 0 mismatches)"  # 09 R2 (10-20, READ_LOG L3080): a Stage 2 value
# run's seeded-draw co-headline aggregate status, written into the result and checked as this exact string (never n_compared 0 alone)


def boot_gate_ok(st2, n, status):
    """09 R2: the seeded-draw co-headline aggregate gate. Stage 1: computed with n_compared > 0. Stage 2: out of scope, n_compared 0 AND the exact
    status string (never n_compared 0 alone)."""
    return (st2 and n == 0 and status == BOOT_OOS_S2) or (not st2 and n > 0 and status == "computed")


def stage2_in_groups():
    """09 R1 / R2: the Stage 2 families found in any aggregate group (FULL_GROUPS_, the auroc06 subset, BOOT_GROUPS); [] is required."""
    fams_ = {f_ for _, fs_ in FULL_GROUPS_ + (("auroc06_subset", AUROC06_FULL),) + tuple(BOOT_GROUPS) for f_ in fs_}
    return sorted(set(FOLD_STAGES["2"]) & fams_)


def stage2_need(o_):
    """09 R1: a Stage 2 plant's required categories: the planted label's member row and its pooled blocks each move, separately at draw 0 and in
    the seeded draws (the co-headline categories of Stage 1 do not apply: no Stage 2 label is in an aggregate group)."""
    sd_ = lambda k_: k_[2].startswith("boot|") and "|draw 0|" not in k_[2]  # noqa: E731
    return {"member row draw 0": lambda k_: k_[0] == "row" and k_[1] == o_ and not k_[2].startswith("boot|"),
            "pooled blocks draw 0": lambda k_: k_[0] == "pooled" and not k_[2].startswith("boot|"),
            "member row seeded draws": lambda k_: k_[0] == "row" and k_[1] == o_ and sd_(k_),
            "pooled blocks seeded draws": lambda k_: k_[0] == "pooled" and sd_(k_)}


PLAIN_FLOOR = 1e-12  # 09 (a), 10-20 ~04:5x: an absolute floor for plain_moved (the quantities are O(1) rates and gaps); below eq()'s 1e-9


def plain_diff(a_, b_):
    """09 option (A) cond. 3 with 09's (a) (10-20 ~04:5x): a value's change between the unplanted and the planted output, by a plain leaf-wise
    comparison independent of eq() and compare(). -> "moved" (a container's key set or length differs, a type changes, one NaN against a
    number, or a number moves by more than PLAIN_FLOOR in absolute value), "subfloor" (some leaf differs, none by more than PLAIN_FLOOR: listed
    by key, never a move; e.g. an exact cancellation such as same_attack = run1@A - run1@B under a plant that shifts both terms), or "same".
    Both-NaN is the same."""
    if isinstance(a_, dict) or isinstance(b_, dict):
        if not (isinstance(a_, dict) and isinstance(b_, dict)) or a_.keys() != b_.keys():
            return "moved"
        ds_ = {plain_diff(a_[k_], b_[k_]) for k_ in a_}
        return "moved" if "moved" in ds_ else "subfloor" if "subfloor" in ds_ else "same"
    if isinstance(a_, (list, tuple)) or isinstance(b_, (list, tuple)):
        if not (isinstance(a_, (list, tuple)) and isinstance(b_, (list, tuple))) or len(a_) != len(b_):
            return "moved"
        ds_ = {plain_diff(x_, y_) for x_, y_ in zip(a_, b_)}
        return "moved" if "moved" in ds_ else "subfloor" if "subfloor" in ds_ else "same"
    if type(a_) is not type(b_):
        return "moved"
    if isinstance(a_, float):
        if a_ != a_ or b_ != b_:
            return "same" if (a_ != a_ and b_ != b_) else "moved"
        return "same" if a_ == b_ else "moved" if abs(a_ - b_) > PLAIN_FLOOR or math.isinf(a_) or math.isinf(b_) else "subfloor"
    return "same" if a_ == b_ else "moved"


def plain_moved(a_, b_):
    return plain_diff(a_, b_) == "moved"


STAGE2_RESIDUAL_GAP = ("residual gap (09, disclosed): an in-scope key that should have moved but neither moved nor was flagged is not caught "
                       "beyond the certain subset")


def stage2_plant(n_, p_, pv_, o_, arm_, levels_, pre_):
    """09 R1 with option (A) (10-20 03:5x), on top of plant_verdict; review round 2 (R-1 / F-3 / F-4); 09's POST HOC (a) and (c) (10-20 ~04:5x).
    Fixed BEFORE the planted comparison: the SCOPE (the planted origin's row keys and the pooled keys of the combo it belongs to; no aggregate key)
    and the CERTAIN subset, from pre_ = the UNPLANTED call's own draw-0 record of the planted run (plant_precheck.caught_at_draw0: level -> caught
    / not caught / None for no finite tau_B): certain = the not-yet-caught levels x {joint, fixed} ("<level>|catch|<arm>@B" and
    "<level>|fixed|catch|<arm>@B" on the planted row); the excluded levels are listed with their reason. Required: pre_'s levels are exactly this
    run's levels (by identity); >= 1 not-yet-caught level (else an ineffective plant: STOP); every certain key present and flagged (draw 0 only);
    flagged (MATCH -> MISMATCH) == value-moved (plain_diff "moved" on the computed value, reference not moved); sub-floor changes listed by key
    and family, never moves; every flagged key and every key with EITHER side moved is in scope (F-3); in each part (draw 0; the seeded draws)
    > 0 flagged row keys and > 0 flagged pooled keys. -> pv_ updated, keys and counts only."""
    both_ = set(n_) & set(p_)
    flg_ = {k_ for k_ in both_ if n_[k_][2] == "MATCH" and p_[k_][2] == "MISMATCH"}
    d0_ = {k_: plain_diff(n_[k_][0], p_[k_][0]) for k_ in both_}
    d1_ = {k_: plain_diff(n_[k_][1], p_[k_][1]) for k_ in both_}
    mov_ = {k_ for k_ in both_ if d0_[k_] == "moved" and d1_[k_] != "moved"}
    any_ = {k_ for k_ in both_ if d0_[k_] == "moved" or d1_[k_] == "moved"}  # F-3: either side moved
    sub_ = sorted(" | ".join(map(str, k_)) for k_ in both_ if "subfloor" in (d0_[k_], d1_[k_]) and k_ not in any_)
    fam_ = lambda s_: re.sub(r"^(boot\|[a-z]+\|)?[0-9.e-]+\|(draw [0-9]+\|)?", "", s_.rsplit(" | ", 1)[-1])  # noqa: E731
    cb_ = o_.rsplit("[pooled: ", 1)[1].removesuffix("]") if "[pooled: " in o_ else None
    insc_ = lambda k_: (k_[0] == "row" and k_[1] == o_) or (cb_ is not None and k_[0] == "pooled" and k_[1] == cb_)  # noqa: E731
    lvl_ok_ = sorted(pre_ or {}) == sorted(f"{float(l_):g}" for l_ in levels_)
    open_ = sorted(lv_ for lv_, c_ in (pre_ or {}).items() if c_ is False)
    excl_ = {lv_: ("already caught at draw 0" if c_ is True else "no finite tau_B at draw 0") for lv_, c_ in (pre_ or {}).items() if c_ is not False}
    cert_ = {("row", o_, f"{lv_}|{md_}catch|{arm_}@B") for lv_ in open_ for md_ in ("", "fixed|")}
    pt_ = lambda k_: "seeded" if str(k_[-1]).startswith("boot|") and "|draw 0|" not in str(k_[-1]) else "draw0"  # noqa: E731
    parts_ = {p0_: {"row": sum(1 for k_ in flg_ if pt_(k_) == p0_ and k_[0] == "row"), "pooled": sum(1 for k_ in flg_ if pt_(k_) == p0_ and k_[0] == "pooled")}
              for p0_ in ("draw0", "seeded")}
    oos_ = sorted(" | ".join(map(str, k_)) for k_ in flg_ | any_ if not insc_(k_))
    agg_ = sorted(" | ".join(map(str, k_)) for k_ in flg_ | any_ if k_[0] in ("agg", "agg-structure"))
    sk_ = lambda s_: sorted(" | ".join(map(str, k_)) for k_ in s_)[:20]  # noqa: E731
    ok_ = (pv_["pass"] and flg_ == mov_ and not oos_ and lvl_ok_ and len(open_) >= 1 and cert_ <= set(n_) and cert_ <= flg_
           and all(v_["row"] > 0 and v_["pooled"] > 0 for v_ in parts_.values()))
    return pv_ | {"pass": ok_, "parts": parts_, "flagged": len(flg_), "value_moved_plain": len(mov_), "flagged_not_moved": sk_(flg_ - mov_),
                  "moved_not_flagged": sk_(mov_ - flg_), "out_of_scope": oos_[:20], "n_out_of_scope": len(oos_),
                  "subfloor": {"floor": PLAIN_FLOOR, "n": len(sub_), "keys": sub_[:40], "families": sorted({fam_(x_) for x_ in sub_})},
                  "certain": {"levels_match": lvl_ok_, "not_yet_caught_levels": open_, "excluded_levels": excl_, "size": len(cert_),
                              "present": len(cert_ & set(n_)), "flagged": len(cert_ & flg_), "not_flagged": sk_(cert_ - flg_),
                              "ineffective_plant": len(open_) == 0},
                  "aggregate_keys_moved": agg_[:20], "n_aggregate_keys_moved": len(agg_), "residual_gap": STAGE2_RESIDUAL_GAP}


def blind(mm):  # the laptop's copy of a mismatch list: keys only (values stay on the results Volume; 0d/2e 13:0x)
    return [[m_[0]] for m_ in mm]


# ---- G7: the pooled-per-world calibration (09 RULE item 42 @ ca621714d; the runner @ e9b78de6d, under review): union rows "<set> (all-origins draws)
# [pooled: <combo>]" calibrate on the label x combo's pooled pools; combos are the pooled ones only
PW_SFX = re.compile(r" \[pooled: ([^\]]+)\]$")
Y1_DEF_PW = "run1@B on every set's pooled row (pooled calibration: no B2, item 45(1)); Y = |run2@B - run1@B| at the pooled tau_B"  # the pooled y1_def
PW_COMBO_ORIGINS = {"pooled_all_origins": ("S_old", "S_new", "S_newmains", "S_paper493"), "paper_cells": ("S_old", "S_new", "S_paper493"),
                    "pooled_all_origins_without_S4": ("S_old", "S_new", "S_newmains without S4", "S_paper493"),
                    "pooled_all_origins_registered_cohort": tuple(o_ + " (registered cohort)" for o_ in ("S_old", "S_new", "S_newmains", "S_paper493"))}


def aob(o):  # an origin name without its pooled-world combo suffix
    return PW_SFX.sub("", o)


def pw_rows(cb):  # the record's union rows of pooled combo cb (the runner's COMBOS_ALLO members, suffixed)
    return tuple(f"{o_} (all-origins draws) [pooled: {cb}]" for o_ in PW_COMBO_ORIGINS[cb])


def unit_bytes(u):
    """One content policy for 09's gates 1 and 2 (as the runner's pw_canon @ 8a15fb10f): a run's units (scores, folds, model scores) as canonical
    bytes: floats float64 with every NaN one NaN and -0 as 0, integers int64, each array prefixed by its shape. Equal content <=> equal bytes."""
    import numpy as np

    parts = []
    for a_ in u:
        a_ = np.asarray(a_)
        if a_.dtype.kind == "f":
            a_ = a_.astype(np.float64)
            a_ = np.where(np.isnan(a_), np.nan, a_ + 0.0)
        elif a_.dtype.kind in "iub":
            a_ = a_.astype(np.int64)
        else:
            raise AssertionError(f"pooled units: unexpected dtype {a_.dtype}")
        parts.append(repr(a_.shape).encode() + b":" + np.ascontiguousarray(a_).tobytes())
    return b"|".join(parts)


def pw_union_chk(exports, where, zero=None):
    """G7, item 42's pools rebuilt here: one label x combo's union, each role (B = H_B, A = H_A, S = H_sim) separately, of its rows' admitted per-set
    pools, deduplicated by run key. 09's gate 1 (NO FALSE MERGE): a run key carries one content (unit_bytes) wherever it appears, in every role and
    set, admitted zero-unit appearances included (zero: {role: {run: admitting sets}}; zero units in one appearance and units in another is a
    conflict). Identical content under different keys is listed (content_identical_keys) for gate 2. exports: [(origin, {role: (keys, scores, folds,
    sbm)})]. -> ({role: (keys, scores, folds, sbm)}, {role: {"runs", "units", "pattern_units", "pattern_runs", "zero_unit_runs"}, "identity_conflicts",
    "content_identical_keys", "roles_by_key", "eligible_units_by_run"}, {role: {run: membership pattern}})"""
    import numpy as np

    zero = zero or {}
    pools, sizes, pats, ident, conf_ = {}, {}, {}, {}, set()
    for nm in ("B", "A", "S"):  # 09's identity ruling (1), NO FALSE MERGE: one content per run key across every role and set (units, folds, model scores)
        for org, P_ in exports:
            k, s, f, b = P_[nm]
            grp = {}
            for i, r in enumerate(k.tolist()):
                grp.setdefault(r, []).append(i)
            for r, ix in grp.items():
                c_ = (unit_bytes((s[ix], f[ix], b[ix])), len(ix))
                if ident.setdefault(r, c_) != c_:
                    conf_.add(r)  # recorded for the gate (keys only); the union keeps the first appearance
        for r in (zero.get(nm) or {}):  # an admitted zero-unit appearance: the empty content
            if ident.setdefault(r, (b"", 0)) != (b"", 0):
                conf_.add(r)
    sizes["identity_conflicts"] = sorted(conf_)
    ch_ = {}
    for r, (c_, n_) in sorted(ident.items()):
        if n_:  # zero-unit runs all share the empty content: not hashed (as the runner)
            ch_.setdefault(hashlib.sha256(c_).hexdigest(), []).append(r)
    sizes["content_identical_keys"] = sorted(v_ for v_ in ch_.values() if len(v_) > 1)
    for nm in ("B", "A", "S"):
        sets_, seen, order = {}, {}, []
        for org, P_ in exports:
            k, s, f, b = P_[nm]
            grp = {}
            for i, r in enumerate(k.tolist()):
                grp.setdefault(r, []).append(i)
            for r, ix in grp.items():
                u = (s[ix], f[ix], b[ix])
                sets_.setdefault(r, set()).add(aob(org).split(" ", 1)[0])
                if r in seen:  # the same content (checked above, across roles too): counted once
                    continue
                seen[r] = u
                order.append(r)
        pats[nm] = {r: " + ".join(sorted(sets_[r])) for r in order}
        if order:
            pools[nm] = (np.concatenate([np.full(len(seen[r][0]), r) for r in order]),) + tuple(np.concatenate([seen[r][i] for r in order]) for i in range(3))
        else:
            pools[nm] = (np.array([], str), np.array([]), np.array([], np.int64), np.empty((0, 5)))
        pu_, pr_ = {}, {}
        for r in order:
            pu_[pats[nm][r]] = pu_.get(pats[nm][r], 0) + len(seen[r][0])
            pr_[pats[nm][r]] = pr_.get(pats[nm][r], 0) + 1
        zp_ = {}
        for ss_ in (zero.get(nm) or {}).values():  # the role's admitted zero-unit runs by their admitting sets (the manifest's sizes[role].zero_unit_runs)
            zp_[" + ".join(sorted(ss_))] = zp_.get(" + ".join(sorted(ss_)), 0) + 1
        sizes[nm] = {"runs": len(order), "units": int(sum(len(seen[r][0]) for r in order)), "pattern_units": dict(sorted(pu_.items())),
                     "pattern_runs": dict(sorted(pr_.items())), "zero_unit_runs": {"runs": len(zero.get(nm) or {}), "by_membership_pattern": dict(sorted(zp_.items()))}}
    rl_, eu_ = {}, {}
    for nm in ("B", "A", "S"):  # cross-pool overlaps: a run key in more than one role, zero-unit appearances included (the runner's catalogue roles)
        for r in set(pats[nm]) | set(zero.get(nm) or {}):
            rl_.setdefault(r, []).append(nm)
        for r in pats[nm]:  # each pooled run's units per role, admitted zero-unit runs at 0 (the manifest's eligible_units_by_run)
            eu_.setdefault(r, {})[nm] = int(ident[r][1])
        for r in (zero.get(nm) or {}):
            eu_.setdefault(r, {})[nm] = 0
    sizes["roles_by_key"] = {r: sorted(v_) for r, v_ in sorted(rl_.items()) if len(v_) > 1}
    sizes["eligible_units_by_run"] = eu_
    return pools, sizes, pats


def pw_pool_inputs_chk(meta):
    """The pins that define the per-set pools, hence the unions (written here from the runner's pw_pool_inputs @ 29717a77b, Astra r1 #5 there): a
    pooled manifest must carry exactly these for the record it serves; the per-set companion must match them except the runner script."""
    pick_ = lambda d_, *ks: None if d_ is None else {k_: d_.get(k_) for k_ in ks}  # noqa: E731
    rl_ = meta.get("honest_b_relaunch")  # runner @ 8a15fb10f: the pin's path and sha256 (a dict), else None; the code pins every imported module too
    return {"script_sha256": meta.get("script_sha256"), "modules_sha256": (meta.get("pooled_world") or {}).get("modules_sha256"),
            "inputs_final": pick_(meta.get("inputs_final"), "path", "sha256"),
            "inputs_full": pick_((meta.get("full_length") or {}).get("inputs_full"), "path", "sha256"),
            "calibration_block": (meta.get("calibration") or {}).get("block"), "cohort_list": pick_(meta.get("cohort_list"), "ref"),
            "fold_minus1_drops": pick_(meta.get("fold_minus1_drops"), "path", "sha256"), "il_arms": (meta.get("il_arms") or {}).get("block"),
            "honest_b_relaunch": pick_(rl_.get("pin"), "path", "sha256") if isinstance(rl_, dict) else None,
            "levels_ignored": "levels / plant do not change pools"}


PW_ID_RULE = "09's identity ruling (i), c8b908e45:experiments/2026-10-14_priority1_spec_09/READ_LOG.md:1388"
TOKEN_PREFIX_REC = ("571883e36", "experiments/2026-10-15_e2e_sim_pipeline/results/check_token_prefix_2026-10-16_192643.json",
                    "c7b0da2f42cfece95ce1675983173d2c6383c80c92325d152fa10e0dbe06f7a6")  # G7b gate 2 (2e 20:1x): the cited record
RELAUNCH_KEY =re.compile(r"honest_b_relaunch\||honest_r\d+\|")  # either form of an honest_b_relaunch run key (item 15/23's alternate pool)
# ---- G7b FOLD-MODEL (0d's release gate via 2e 10-16 ~20:2x): the pooled build's fold-model labels, in two stages: Stage 1 = the lead's two
# families; Stage 2 = the sensitivity rows' three (2e 22:2x). Each family -> its fit record (commit, path, sha256) = INPUTS_FULL_110447 fits[fam].fit
RS_UNCUT_FIT = ("19611550c", "experiments/2026-10-14_p2_family_expansion/results/own_all_rs_uncut_fit_2026-10-16_030614.json",
                "f62f4d660109fff5d3d207a024299f3c254fbfb2c603ddc7fc62b40b38760442")  # one fit record for the three own_all_rs uncut families
FOLD_FITS = {"sol_a_uncut": ("9564a41b4", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/cutfull/sola_uncut_reselect_fit_2026-10-16_030230_cutfull_491db0e7e2c9.json",
                             "40e7b12c3f0c24939e40f9e0a00a1b2a5b3845d2d81138c342e9d68e2515ef74"),  # the RESELECT fit (2e 20:3x)
             "own_all_rs_uncut_L48": RS_UNCUT_FIT,
             "c_asis_uncut": ("8b6124cb3", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/cutfull/casis_uncut_inherit_fit_2026-10-16_035520_cutfull_d76d0ca08a3d.json",
                              "f9e32638ad45a4afe1b6d75862b4745be0a23c77fb2268bf9f32bbc1c113b7df"),  # the INHERIT fit (INPUTS_FULL fits.c_asis_uncut)
             "own_all_rs25_uncut_L48": RS_UNCUT_FIT, "own_all_rs10_uncut_L48": RS_UNCUT_FIT}
FOLD_STAGES = {"1": ("sol_a_uncut", "own_all_rs_uncut_L48"), "2": ("c_asis_uncut", "own_all_rs25_uncut_L48", "own_all_rs10_uncut_L48")}
FOLD_FILES = {"sol_a_uncut": 368, "own_all_rs_uncut_L48": 370,  # each family's pinned files (C + R) in the listing (507-prep NB-a): --shapes asserts
              # (route C, 10-20: +2 each, the pilot root's S_old sim-honest C / R files; the route C Stage 1 listing ..._2026-10-20_005013.json by file
              # name, 2218 files = Stage 1's 2206 + 12 pilot, 0 removed, 0 re-pinned; the 171-pool Stage 1 had 366 / 368)
              "c_asis_uncut": 368, "own_all_rs25_uncut_L48": 370, "own_all_rs10_uncut_L48": 370}  # the walk's count per family (route C Stage 2, 10-20: +2 each, the pilot root's files, from the Stage 2 listing ..._2026-10-20_023328.json by file name; the 171-pool Stage 2 counts
# from the Stage 2 listing ..._foldmodel_2026-10-16_223410.json by file name: 183 + 183 c_asis_uncut cmd_A / avg_top2, 184 + 184 for each own_all_rs)
RS_FOLDS = ("ec56720ad", "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json",
            "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299")  # 09 3a (b): own_all_rs's committed fold list (the runner's OWN_ALL_RS_FOLDS)
# rule lists copied from the runner @ bd4af2fb9 (full_override.NO_FOLD_DROPS l.36; run_peraction311 OWN_ALL l.352, OWN_ALL_LOO l.365, OWN_ALL_RS l.417,
# E12_FILE_FAMILIES l.558): written here and compared with the code commit's at capture, so a changed rule STOPs instead of being inherited
NO_FOLD_DROPS = ("Sol (a)|uncut", "c_asis|uncut", "sol_a_uncut", "sol_a_uncut_pinned", "c_asis_uncut", "own_all_rs_L48|uncut", "own_all_rs25_L48|uncut",
                 "own_all_rs10_L48|uncut", "own_all_rs_uncut_L48", "own_all_rs25_uncut_L48", "own_all_rs10_uncut_L48")
OWN_ALL_FAMS = ("own_all_L48", "own_all25_L48", "own_all10_L48", "own_all_LOO_L48", "own_all_rs_L48", "own_all_rs25_L48", "own_all_rs10_L48")
OWN_ALL_RS_FAMS = ("own_all_rs_L48", "own_all_rs25_L48", "own_all_rs10_L48")  # the runner's OOF_FALLBACK_LABELS (by job label family)
E12_FILE_FAMILIES = ("sol", "p2", "famexp", "own_all_rs", "own_all_loo")
FOLD_NOT_COVERED = ["IL extra-arm rows (sim2 / run3): never in H_B, H_A or H_sim; the runner's IL-only STOPs (extra-arm fold -1 columns, the own_all_rs "
                    "fold check on IL rows) are not reproduced", "relaunch rows: R3 is not computed on pooled rows; not read here",
                    "the D1 scored stage (calibration files, member rows): this is the catalogue stage only"]  # 2e 20:3x: stated in the record
BOOT_D, BOOT_SEED, BOOT_N = 10000, 20261016, 20  # G7b VALUE (09 via 2e 23:2x): the runner's D (run_fprsweep_507 l.33); the sample's stated seed and size
VALUE_SCOPE = (f"value reproduction of draw-0 point estimates and a seeded sample of {BOOT_N} bootstrap draws (seed {BOOT_SEED}), with arithmetic ported "
               "from the runner (independent execution; not an independent specification); interval quantiles not recomputed; inputs verified by Stage 1")
# 09 via 2e 23:2x: the exact FINAL SCOPE line of every value-reproduction record and label (N and s filled in; never without the qualifier)


def value_scope(seed=None):  # PLANT V3: the same line with the run's own seed filled in (A / B / R37: BOOT_SEED, the exact text; PLANT: 20261020)
    return VALUE_SCOPE if seed is None or int(seed) == BOOT_SEED else VALUE_SCOPE.replace(f"(seed {BOOT_SEED})", f"(seed {int(seed)})", 1)


PERDRAW_BY_RECORD = {  # route C (A18 / A22): record sha256 -> its companion per-draw npz (volume, path, sha256), generated from each record's own
    # top-level "npz" by script (gen_pins); a value run READS the triple from its record by key and asserts it equals this entry for its record's sha
    # (09 cond. 1 / 2e (b), 10-20); the 171-pool A's was ..._211215 @ a9ebd60b
    "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a": ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",  # A (a936a5d97)
        "xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz", "5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1"),
    "ef5c765716ee338a1e8d207a820eecd5c083ba1d51262ff72361d6d254bf6ad7": ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",  # B (f4e21b0f7)
        "xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652_perdraw.npz", "8a2a2afda1ff5271695191be04911fcd9ecb4bf7199d5715fe6a526c9b2ffaa3"),
    "8e6635e7413eb0202e469e3af84af70e72b801de9fc0c9d6b7505848a21807e4": ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",  # R37 (55a2fb20e)
        "xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_rule37_pooledworld_scored_e12_2026-10-19_181100_perdraw.npz", "2d287bb8656079c87d88eca91054003e7c56ddb6d7adfa6f63092a6ef6db754c"),
    "66fda847576f30bd23fa6d6b3bde80290257115e6f2afca59e1ee7a92a86851b": ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",  # PLANT (b90a38f98)
        "xfer/fpr507prep/alltasks_CRD_FULL_plant_r2_allorigins_pooledworld_scored_e12_2026-10-19_230620_perdraw.npz", "a355f0715688818cda7a2e7b337c49f38377da38b4d3094c8643b5ed79c7d452")}
PLANT_RECORD_SHA = "66fda847576f30bd23fa6d6b3bde80290257115e6f2afca59e1ee7a92a86851b"  # PLANT V3 (PRESTATED_V3 fb34c414 + AMEND1-5): route C's PLANT
TREE_SHA_ADDED = {PLANT_RECORD_SHA: {  # 09's GRANT (GATES_routec @ 4e4ab4142 L596-598; V3 pre-statement section 7), PLANT value run ONLY: the 4 files added
    # at P4 752413181 since f2d47f19f (git diff --name-status: 4 x "A", 0 modified), each by its sha256 read from git at 752413181; none is imported
    f"{D507}/check_plant_label_entries.py": "c22ec350f0f2d346b9441a2a92600f38f73bec4e0031f2cd683d8317e0c2f920",
    f"{D507}/check_plant_s2_routec.py": "d14ccd6f635d1c45c0b65465a06ceb7eea946b74b407821b15681916a67f7693",
    f"{D507}/check_plant_vs_a_routec.py": "a7064d75742cb1d7d537aedc0210f87cd933202a5def97b5622fb844e267f5d9",
    f"{D507}/diff_plant_vs_a_specs.py": "069bb3ea2b9c2b487c027bb406a98dc89574eb47af410fe484d18d94079dfc55"}}
PLANT_REPO_DIFF = ("f2d47f19fd873aa44b5267d88bf7cb3795f875b7", "752413181bf92364110ec80c3c18755f9f37108d", 9)  # AMEND4 D6: the whole-repo diff = 9 x "A"
BOOT_SEED_BY_RECORD = {PLANT_RECORD_SHA: 20261020}  # AMEND4 D6 / AMEND5 F3: the PLANT run's 20 seeded draws (blocks (i) and (iii)); A's stays 20261016
PLANT_P1_SEED = 20261021  # AMEND4 D3 / AMEND5 F2: P1's choice (label in main; origin, level and the delta order in the P1 call)
PLANT_V3_FILES = {  # block (v)'s inputs, read from git by sha256 in main and shipped to the run directory (values parsed only on Modal)
    "plant_record.json.gz": ("b90a38f9867a5afc0db63e33e5cbbdb35ba59f4e", f"{D507}/results/routec_records/routec_PLANT_alltasks_CRD_FULL_plant_r2_allorigins_"
                             "pooledworld_scored_e12_2026-10-19_230620.json.gz", "df2c36a66dd79e50f9fb71216b906fbca6120f2c0f8a2ca4d766acf12c980261"),
    "base_record.json.gz": ("a936a5d9762763f9feed351aebdeeec2d8114f21", f"{D507}/results/routec_records/routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_"
                            "scored_e12_2026-10-19_180327.json.gz", "191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865"),
    "plant_rows_config.json": ("699f214035a4f15b9e3025307eb21ee6705c3b7b", "experiments/2026-10-15_panel_noise_matched/config/plant_rows_POOLED_FULL_routeC_"
                               "2026-10-20_004247.json", "dbbad8038ffb3674fa9dc3fe44947a6a9ba4c94331417686593daf3d8d5833a8"),
    "plant_rows_target.json": ("699f214035a4f15b9e3025307eb21ee6705c3b7b", "experiments/2026-10-15_panel_noise_matched/results/panel_plant_rows_plant_rows_"
                               "POOLED_FULL_routeC_2026-10-20_004247_2026-10-20_004833.json", "a864fad74436f0b5052adfaae7ba4ff54a0f192670b74b9b88ec8235b7048a21")}
PLANT_NOT_COMPARED = ["rows.<label>.<origin>.cell_hits_draw0 (not compared for any arm, as for A)",
                      "planted values in mode fixed (the runner plants joint only; the rows' fixed catch is checked to hold no planted key)",
                      "coheadline_signed_plant Y: ci95, draws and finite_draws (block (iv) compares S and Y at draw 0 by .point) (AMEND7 G4)",
                      "coheadline_signed_plant S: ci95, draws and finite_draws at +0 (both combos) and at every delta of paper_cells (block (v)'s "
                      "ported gate, run_plant_rows L567, checks S's whole summary against the npz planted arrays only for its configured combo "
                      "pooled_all_origins at the nonzero deltas; block (iv) compares .point) (AMEND9; Codex V3 r3). The 23 printed rows read pooled_all_origins only "
                      "(run_plant_rows @ fd12eca35 L447 and L221: cb = cfg['combo'], configured pooled_all_origins)",
                      "the planted-sim2 path (runner L2863-2873 with admitted sim2 runs): NOT REPRODUCED. In this record sim2 was omitted by the build "
                      "(RULE item 40, L4874: 'a BUILD FAILURE ... although the scores existed and were routed at the freeze'): no row has an admitted sim2 "
                      "run, so no planted-sim2 key exists (checked absent: row catch keys, the npz, "
                      "coheadline_signed_plant, cell_hits_draw0 planted strings) (AMEND7 B). The registered planted-RMS check is a separate run "
                      "(rms_full_2026-10-16_155454_ap-K3Ot3sJQETSs1dYPsXGgo5.json @ af1a12e32), outside V3's scope, with its own status (AMEND9)",
                      "P1 at a delta where the NORMAL run compares no pooled planted gap or coheadline S (keys absent: a member row cannot reach it): those "
                      "two categories are not required of P1 there (POST HOC loosening of PRESTATED_V3 section 5 L98-99, 09 ruling pending; AMEND7 C)"]
PLANT_SIM2_SCOPE = ("the second sim repeat (sim2; AMEND6, Codex V3 #3): each IL row's il_arms report (il_arms.py @ 752413181 L341-379, ported in main from "
                    "the row's own captured job IL block, which is the pinned block but for files, as full_override.apply_full leaves it) and its draw-0 solve per level (plant_fit on the arm's draw-0 z, NaN on every cell where no admitted run exists) "
                    "are compared; a folder holding the probe's IL file, or a sim2 planted arm, STOPs (rows not ported)")
PLANT_STEP_SPEC = ("experiments/2026-10-17_records_step_container/steps/routec_spec_PLANT_pinned_2026-10-19_214314.json",  # PRESTATED_V3 section 2 / 4
                   "b906eebfaa8e7c20c46f4fe1ad4765b33562f2ea2e2828f34ab55c210496310c")  # (Codex V3 #8): read at the record's code commit, by sha256
PLANT_PANEL_WITHHELD = "plant job: the plant labels only (the headline six); the final record's panel_groups are authoritative"  # runner L4514 @ 752413181
IL_ORIGINS = ("S_old", "S_new", "S_newmains")  # runner L1526 @ 752413181: the origins whose rows read the IL extra arms (with --il-arms)
PLANT_SIM2_COUNTS = {"il_reports": 60, "sim2_solves": 120}  # AMEND7 G2 / AMEND9: 60 IL rows x 1 report; 60 x 2 levels sim2 solves (key-only counts)
IL_COUNTS_RULE = ("report-only (0d/09 10-16 ~03:30): the freeze takes every admitted extra arm that is routed and scored at that moment; "
                  "a probe x cell x arm without one uses (A) inside (B), labelled 'no extra runs scored'")  # il_arms.py @ 752413181 L376-377
# ---- rc1b W2 (PRESTATED_W2_2026-10-20_145024 b9bf7a90 + AMEND1 fbcd9fa7): a G7b-style VALUE check of Stage D's per-task values (W2a) and kappa's
# per-draw margins (W2b), one run per record x fold stage. Both records were built by run_peraction311.py @ 7797f2cc2 (blob 8b3fbf77), each with A's
# flags plus --pertask / --kappa naming route C A @ a936a5d97 (54afad04); the pinned step specs below are their launch flags.
W2_CODE = "7797f2cc29ed00571dc2091e234a6d3396936c33"
W2_A_REF = ("a936a5d9762763f9feed351aebdeeec2d8114f21", f"{D507}/results/routec_records/routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_"
            "2026-10-19_180327.json.gz", "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a")
W2_RECORDS = {  # record sha256 (decompressed) -> (kind, its pinned step spec at W2_CODE, the spec's sha256)
    "468655154f826ec1a93e08333eb802e68abea6db65a8e2f80a3e0f23465f36d9": ("pertask", "experiments/2026-10-17_records_step_container/steps/"
                                                                         "routec_spec_SD_paid_2026-10-20_003316.json",
                                                                         "ee360ecbf963cf51dc85d0e66e4850152175b84102c2ae795849983affe73c08"),
    "40ca1bf7a6deaf51175fb56cd89d7277b64ca1ec1ffdf87292d6ae0feca3cce9": ("kappa", "experiments/2026-10-17_records_step_container/steps/"
                                                                       "routec_spec_KAP_paid_2026-10-20_003316.json",
                                                                       "0d5b2d03f2101952e202f12a2c026804785131afe36ce8bfe11262d29d7083e3")}
W2_FANOUT = {"1": 13, "2": 15}  # AMEND1 section 1: the stage's labels + 1 control call
W2_COUNT = {"pertask": 6873 * 3 + 12, "kappa": 7 * (BOOT_N + 1) * 2 * 7}  # AMEND1 section 1: the values compared per label (the count gate)
W2_CTL_SEED = 20261020  # AMEND1 section 4: the control call's seeded choices
W2_SCOPE = {"pertask": ("rc1b W2a: value reproduction of Stage D's per-task entries at draw 0 (max_minus_tau and caught; honest_percentile "
                        "secondary), with arithmetic ported from the runner (independent execution; not an independent specification); the D1 "
                        "calibration recomputed from A's pooled pools, not read; inputs verified by Stage 1 / Stage 2"),
             "kappa": (f"rc1b W2b: value reproduction of kappa's per-row 2x2 margins at draw 0 and a seeded sample of {BOOT_N} bootstrap draws "
                       f"(seed {BOOT_SEED}), both modes, with arithmetic ported from the runner (independent execution; not an independent "
                       "specification); the D1 calibration recomputed from A's pooled pools, not read; inputs verified by Stage 1 / Stage 2")}
W2_SPEC_IGNORED = ("spec_only", "pw_calib_res", "copies_root")  # the spec's build-only flags: an output path, the D1 wave's resources, the copies root
# (input resolution; every copy read is sha-checked against its committed index)


def w2_flags_viol(got, spec_flags):
    """rc1b W2: the capture's runner arguments against the record's pinned launch flags: equal on every flag that is set (a False / "" / None default
    and the spec's build-only flags aside; readouts: the runner's default "C"). -> the differing flag names"""
    def norm(d):
        return {k: v for k, v in d.items() if k not in W2_SPEC_IGNORED + ("readouts",) and v not in ("", False, None)}
    g_, s_ = norm(got), norm(spec_flags)
    return sorted({k for k in set(g_) | set(s_) if g_.get(k) != s_.get(k)} | ({"readouts"} if spec_flags.get("readouts", "C") != got.get("readouts", "C") else set()))


ROUTEC_LEVEL_SPECS = {  # GATES A22 @ f5441bb2c (POST HOC): record sha256 -> its pinned spec (commit, path, sha256), whose entry.flags.levels names the
    # level set registered for that record (resolved by the runner's own rule at the record's code commit; see registered_levels)
    "ef5c765716ee338a1e8d207a820eecd5c083ba1d51262ff72361d6d254bf6ad7": ("f2d47f19fd873aa44b5267d88bf7cb3795f875b7",  # B: levels sweep15
        "experiments/2026-10-17_records_step_container/steps/routec_spec_B_pinned.json", "8f914cd522bbbc8282b6fa594617b6b705f07469d45c69eff03093c2eb150f74"),
    "8e6635e7413eb0202e469e3af84af70e72b801de9fc0c9d6b7505848a21807e4": ("f2d47f19fd873aa44b5267d88bf7cb3795f875b7",  # R37: levels rule37
        "experiments/2026-10-17_records_step_container/steps/routec_spec_R37_pinned.json", "8f141e78aa9380990ab54d91a6118193d3062c784f3c483dc3bf7ba203f1d790")}
ROUTEC_NAME_PREFIXES = ("routec_A_", "routec_B_", "routec_R37_", "routec_PLANT_")  # route C's pooled records are committed as routec_<A|B|R37|PLANT>_
# <the Volume record's name> (a936a5d97, f4e21b0f7, 55a2fb20e, b90a38f98); stripped before the mode is read (PLANT V3: PLANT is now in this check's domain)
COPIES_ROOT = "/Users/aswinsurya/probe-shift-durable-2026-10-17/fpr507"  # route C runners read every fpr507 manifest copy through copies.py, whose
# root() STOPs without the copies_root flag or $PROBE_SHIFT_COPIES (copies.py l.47-55 @ f2d47f19f); each read is sha-checked against its committed index


def record_mode(fname):
    """The record's name (STUB_ and one of route C's routec_A_ / routec_B_ / routec_R37_ prefixes and .gz dropped) and the check mode it selects
    -> (name, mode)."""
    nm_ = fname.removesuffix(".gz").removeprefix("STUB_")
    nm_ = next((nm_.removeprefix(p_) for p_ in ROUTEC_NAME_PREFIXES if nm_.startswith(p_)), nm_)
    return nm_, ("alltasks" if nm_.startswith("alltasks_CRD") else "sweep" if nm_.startswith("sweep_final_") else "union_sol"
                 if nm_.startswith("sweep_union_sol") else "pooled")  # pooled: pooled455v2_*, runlen_v2_*


def registered_levels(rp, name):
    """GATES A22 (a): the level set registered for a record, from its pinned spec's entry.flags.levels, resolved by the runner's own rule at the
    record's code commit (run_peraction311 l.6142-6151 @ f2d47f19f): rule37 -> rule37_levels() (RULE37_F, asserted to be 09 RULE 37's three);
    sweep15 -> GRID within [1e-3, 5e-2] plus 5e-2, 15 levels holding 1e-2 (the runner's expression, l.6149-6151). Any other name STOPs."""
    if name == "rule37":
        return sorted(float(f_) for f_ in rp.rule37_levels())
    assert name == "sweep15", ("a pinned spec's level set this check does not resolve", name)
    fl_ = sorted({f for f in rp.GRID if 1e-3 * (1 - 1e-9) <= f <= 0.05 * (1 + 1e-9)} | {0.05})
    assert len(fl_) == 15 and 0.01 in fl_, "sweep15: not the runner's 15 levels"
    return fl_


def level_part_levels(levels, part):
    """B's level split (A22; 2e / 09 10-20): '--level-part k/n' -> every n-th level from the k-th of the sorted full set (the runner's pw_level_part
    convention, rp l.6152-6156 @ f2d47f19f). The identity tie and the inputs sha stay on the FULL inputs; only check() and the aggregate compute the
    part. A bad k/n or a part without a level STOPs."""
    m_ = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", part or "")
    assert m_ and int(m_.group(1)) <= int(m_.group(2)), ("--level-part k/n with 1 <= k <= n", part)
    out_ = sorted(levels)[int(m_.group(1)) - 1::int(m_.group(2))]
    assert out_, ("--level-part: this part holds no level", part)
    return out_


def tie_parts_viol(differing, allowed):
    """The value tie's identity parts that differ from Stage 1's outside VALUE_ID_ALLOWED (after A20 / A22 removed theirs) -> sorted names (STOP)."""
    return sorted(set(differing) - set(allowed))


def levels_allowance_viol(captured, registered, recorded):
    """GATES A22 @ f5441bb2c (POST HOC): 'top|levels' may differ from S1 ONLY if the captured levels equal BOTH (a) the registered set (the pinned
    spec's) AND (b) the record's own levels (meta.alltasks.levels, plus meta.levels_rule37.levels where present). -> violations by key (each level
    missing or extra named); exact float equality on the sorted lists."""
    c_, g_, r_ = (sorted(float(x) for x in v_) for v_ in (captured, registered, recorded))
    out = [] if g_ == r_ else [f"spec and record levels disagree | spec only {sorted(set(g_) - set(r_))} | record only {sorted(set(r_) - set(g_))}"]
    for nm_, ref_ in (("registered", g_), ("recorded", r_)):
        if c_ != ref_:
            out.append(f"captured levels differ from the {nm_} set | missing {sorted(set(ref_) - set(c_))} | extra {sorted(set(c_) - set(ref_))}"
                       + (" | repeated level" if len(set(c_)) != len(c_) else ""))
    return out


TREE_SHA_DIFF = {  # G7b on route C (option B, POST HOC, 10-20 01:2x): the only runner-folder files whose sha256 may differ between S1's capture tree
    # (4feb6694b, route C manifest 5ed5407b's meta.git, which the catalogue rules require) and A's (f2d47f19f): (S1's, A's); no runner module uses them
    "experiments/2026-10-15_fpr_sweep_507_prep/build_signoff_covered.py": ("d677f875cc10e4d169e0edbcafb1163509fec885218854345d9682a6a88470c4",
                                                                          "7a6b2ff27d28b9529193e53b0f54674b22bfb3683438a27b931f1cc74b28c152"),
    "experiments/2026-10-15_fpr_sweep_507_prep/test_build_signoff_covered.py": ("9a167070f8f3fab3259258d2c84a8cfb16f1a93aff487dcb552c54d2a73ad0aa",
                                                                               "1057d0c25310531c737af46aa28aa90266eba5ba4cbddd6e63fb2a19bee6f02d")}


# rc1b W2 R1 (09: ACCEPTED POST HOC with conditions (a)-(d); PRESTATED_W2 section 4, AMEND1 section 2): the W2 records' runner tree (7797f2cc2) vs S1's
W2_TREE_ADDED = {  # (c) the 9 top-level runner-folder .py files added f2d47f19f -> 7797f2cc2, each by its sha256 at 7797f2cc2; none may be a pinned module
    f"{D507}/check_kappa_launch_share.py": "39deb2b6c197322f3501c8c365afee800e8c10e3d8c7fe5afb44353c89260eb3",
    f"{D507}/check_routec_dl_record.py": "a32e9c35121a01572e94d24d38654707563c4888918be595aa175646e525451d",
    f"{D507}/make_routec_dl_step_spec.py": "9cc5c8650f58587ac06ecdc56f62cfd3bf92b42ad3fb62956a626da1ba3011bb",
    f"{D507}/run_kappa_extra.py": "2911d95693ca00f9cd11fc93ea18a3c990f2eb84dd49521267007c05d941981a",
    f"{D507}/test_check_routec_dl_record.py": "54e966ecaa558246363e3291861e4b3271f07025775248cb5eac82f2206a84c0",
    f"{D507}/test_kappa_extra_routec.py": "55c413b26ad9a5b71c5f60f1f1409d70301c70152886225cdf7f38c0fb1fd649",
    f"{D507}/test_make_routec_dl_step_spec.py": "8408a94e12d8a2501317467b96a1c86c449cd0dd7b35e61fb33b6111d6c1a19b",
    f"{D507}/test_routec_d1_labels.py": "76a6118485905cf5e34167f3d9d2fe3a4616bfa21e26fdb79d473ee0cb1b7146",
    f"{D507}/test_sd_preflight_jobcount.py": "2f38139c07c721f1aeefdabbdc54ca8770bd48dfa0c563da7324a34e56a13e11"}
W2_TREE_MODIFIED = {f"{D507}/run_peraction311.py": ("dcd265d77088ccc517308680567158f9ad6ebbc497e3a8632c5de2ed2b1f12aa",  # S1's (= A's) runner
                                                    "29bd453187cb205a029ec58461bb861a899a5096d1d945d137879395af0819e7")}  # -> W2's (blob 8b3fbf77)
W2_RUNNER_HUNKS = ("@@ -824 +824 @@", "@@ -830 +830 @@", "@@ -832,0 +833,2 @@", "@@ -840,0 +843,47 @@", "@@ -2668 +2717 @@", "@@ -3058 +3107 @@",
                   "@@ -4954,0 +5004 @@", "@@ -4957,0 +5008,2 @@", "@@ -4989,0 +5042,4 @@", "@@ -5467,0 +5524 @@", "@@ -5477,0 +5535,2 @@",
                   "@@ -5491,0 +5551,2 @@", "@@ -6908,0 +6970,2 @@", "@@ -6948,0 +7012,2 @@")  # (a) git diff -U0 f2d47f19f 7797f2cc2, exactly these 14
TREE_SHA_DIFF_BASE = "f2d47f19fd873aa44b5267d88bf7cb3795f875b7"  # A's code commit (S1's runner = A's, sha256 dcd265d7): the base of the W2 runner diff
W2_PORTED = ("wmoments", "wthr", "pool_reduce", "pool_reduce_n", "tpr", "tpr_on", "zpool", "zatt", "zwith")  # (b) the functions W2 ports
W2_PT_BLOCK = {"old": (2582, 2669), "new": (2631, 2718)}  # (b) the inline per-task block (not a function), and its one label-only hunk line
W2_PT_BLOCK_HUNK = {"old": 2668, "new": 2717}


def w2_hunk_lines(hunks):
    """The deleted (old side) and added (new side) line numbers of `git diff -U0` hunk headers ("@@ -a[,b] +c[,d] @@"). -> (deleted, added)"""
    dl_, al_ = [], []
    for h_ in hunks:
        m_ = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", h_)
        assert m_, ("rc1b W2 R1: not a hunk header", h_)
        o_, oc_, n_, nc_ = int(m_[1]), int(m_[2] or 1), int(m_[3]), int(m_[4] or 1)
        dl_ += list(range(o_, o_ + oc_))
        al_ += list(range(n_, n_ + nc_))
    return dl_, al_


def w2_span_viol(old_src, new_src, hunks):
    """rc1b W2 R1 (b): no deleted line (old blob) or added line (new blob) of the runner diff inside a ported function's span (any nesting), each
    ported function present exactly once in each blob; inside the inline per-task block only its one label-only hunk line. -> violations"""
    import ast

    dl_, al_ = w2_hunk_lines(hunks)
    out = []
    for side_, src_, lines_ in (("old", old_src, dl_), ("new", new_src, al_)):
        sp_ = {}
        for n_ in ast.walk(ast.parse(src_)):
            if isinstance(n_, (ast.FunctionDef, ast.AsyncFunctionDef)) and n_.name in W2_PORTED:
                sp_.setdefault(n_.name, []).append((n_.lineno, n_.end_lineno))
        out += [f"{side_}: {nm_} found {len(sp_.get(nm_, []))} times (want 1)" for nm_ in W2_PORTED if len(sp_.get(nm_, [])) != 1]
        out += [f"{side_}: line {ln_} inside {nm_}" for nm_, ss_ in sorted(sp_.items()) for a_, b_ in ss_ for ln_ in lines_ if a_ <= ln_ <= b_]
        a_, b_ = W2_PT_BLOCK[side_]
        out += [f"{side_}: line {ln_} inside the per-task block" for ln_ in lines_ if a_ <= ln_ <= b_ and ln_ != W2_PT_BLOCK_HUNK[side_]]
    return out


W2_KAPPA_LEVELS, W2_S1_LEVELS = [0.01], [0.01, 0.05]  # rc1b W2 AMEND4: kappa's jobs run F_list = (0.01,) (runner @ 7797f2cc2 L6990); S1's default set


def w2_capmods_viol(s1_map, w2_map, modified):
    """rc1b W2 AMEND4 (the module-map face of R1's modified runner): the W2 capture's module map (path -> sha256) against S1's rebuilt one: the same
    paths, every sha equal except the listed runner, exactly (S1's sha, W2's sha). -> violations (paths only)"""
    out = [] if sorted(s1_map) == sorted(w2_map) else ["module paths differ"]
    out += [f"module differs | {k_}" for k_ in sorted(set(s1_map) & set(w2_map)) if s1_map[k_] != w2_map[k_] and k_ not in modified]
    out += [f"listed runner not as listed | {k_}" for k_, v_ in sorted(modified.items()) if (s1_map.get(k_), w2_map.get(k_)) != tuple(v_)]
    return out


def w2_levels_viol(kappa_levels, s1_levels):
    """rc1b W2 AMEND4: kappa's level set may differ from S1's only as exactly [0.01] against exactly [0.01, 0.05]. -> violations"""
    return [] if (list(kappa_levels) == W2_KAPPA_LEVELS and list(s1_levels) == W2_S1_LEVELS) else [f"levels {list(kappa_levels)} vs S1's {list(s1_levels)}"]


def tree_sha_diff_viol(s1_tree, val_tree, pinned, allowed, added=None, modified=None):
    """The value tie's one exact exception (option B): S1's and the value run's pw.tree_sha maps (runner-folder file -> sha256) -> violations by key.
    FAIL CLOSED: the file names differ; a pinned module (the runner's required and captured modules) differs, or is absent from either map; a
    differing file not listed in `allowed`, or listed with other shas; a listed file that does not differ exactly as listed.
    PLANT V3 (09's grant, GATES @ 4e4ab4142 L596-598): `added` names files that may exist ONLY in the value run's tree, each with exactly its sha256;
    an added name in S1's tree, an added file absent or with another sha, an added file that is a pinned module, or any other extra name STOPs.
    rc1b W2 R1 (09, POST HOC): `modified` names files (pinned modules included) that may differ exactly as listed, (S1's sha, the value run's);
    a listed file with other shas STOPs (its hunks and spans are checked apart: w2_span_viol)."""
    mod_ = modified or {}
    if added:
        out = [] if set(val_tree) == set(s1_tree) | set(added) and not (set(added) & set(s1_tree)) else ["file names differ"]
        out += [f"added file absent or with another sha | {k_}" for k_, v_ in sorted(added.items()) if val_tree.get(k_) != v_]
        out += [f"added file is a pinned module | {k_}" for k_ in sorted(set(added) & set(pinned))]
        val_tree = {k_: v_ for k_, v_ in val_tree.items() if k_ not in added}  # the rest is the unchanged option-B rule
        out_ = [] if sorted(s1_tree) == sorted(val_tree) else ["file names differ"]
        out += [x_ for x_ in out_ if x_ not in out]
    else:
        out = [] if sorted(s1_tree) == sorted(val_tree) else ["file names differ"]
    out += [f"pinned module differs or is missing | {k_}" for k_ in sorted(pinned) if k_ not in mod_ and (k_ not in s1_tree or s1_tree.get(k_) != val_tree.get(k_))]
    out += [f"modified file differs from its listed shas | {k_}" for k_, v_ in sorted(mod_.items()) if (s1_tree.get(k_), val_tree.get(k_)) != tuple(v_)]
    out += [f"unlisted difference | {k_}" for k_ in sorted(set(s1_tree) & set(val_tree))
            if s1_tree[k_] != val_tree[k_] and k_ not in pinned and k_ not in mod_ and allowed.get(k_) != (s1_tree[k_], val_tree[k_])]
    out += [f"listed difference absent | {k_}" for k_, v_ in sorted(allowed.items()) if (s1_tree.get(k_), val_tree.get(k_)) != tuple(v_)]
    return out


def routec_zero_unit_arg(fl):
    """meta.full_length -> the runner's --routec-zero-unit "<path>@<sha256>" (rp l.6332-6341 @ f2d47f19f) when an extra S_old root pins one, else ""."""
    zs_ = [v_["zero_unit_list"] for v_ in ((fl or {}).get("s_old_simh_extra_roots") or {}).values() if isinstance(v_, dict) and v_.get("zero_unit_list")]
    assert len(zs_) <= 1, "more than one pinned zero-unit list (the runner takes one)"
    return f"{zs_[0]['path']}@{zs_[0]['sha256']}" if zs_ else ""


def pilot_zero_unit_viol(rows, cfgs, extra_roots, require=False):
    """09 cond. 3 (10-20), FAIL CLOSED: route C's zero-unit pilot members (FO zero_unit_ok; rp l.1819, l.1849, l.1998, l.2045 @ f2d47f19f) are not
    ported here, so a check runs only when there are none. -> violations by key: a row's s_old_simh_extra entry whose zero_unit or
    zero_unit_unlisted is not an empty list (missing counts too); a captured job's S_old-only H_sim entry whose zero_unit_ok is not an empty list;
    with require (a pooled scored record), an S_old row of the record without an entry for every extra root."""
    out = []
    for lb_, od_ in sorted(rows.items()):
        for o_, r_ in sorted((od_ or {}).items()):
            sx_ = r_.get("s_old_simh_extra") if isinstance(r_, dict) else None
            if require and extra_roots and o_.split(" ", 1)[0] == "S_old" and not (isinstance(sx_, dict) and set(extra_roots) <= set(sx_)):
                out.append(f"row without s_old_simh_extra for every extra root | {lb_} | {o_}")
            for rt_, e_ in sorted((sx_ or {}).items()) if isinstance(sx_, dict) else ():
                for k_ in ("zero_unit", "zero_unit_unlisted"):
                    if not (isinstance(e_, dict) and e_.get(k_) == []):
                        out.append(f"row {k_} not an empty list | {lb_} | {o_} | {rt_}")
    for i_, cf_ in enumerate(cfgs):
        for lb_, es_ in sorted(((cf_ or {}).get("simh_old") or {}).items()):
            for e_ in es_ or []:
                if isinstance(e_, dict) and e_.get("s_old_only") and e_.get("zero_unit_ok") != []:
                    out.append(f"job zero_unit_ok not an empty list | job {i_} | {lb_} | {e_.get('root')}")
    return out
BOOT_ROW_CATCH = ("sim@S|primary", "run1@B", "run2@B", "real_mean@B", "run2@A", "run1@A", "sim@B")  # the member rows' per-draw catches (draws_out, l.2728)
BOOT_ROW_GAP = ("delta_full", "delta_rr", "sim_minus_run1", "sim_minus_run2", "real_line")  # per-set gaps the reducer writes (l.3229-3233; (B) excluded)
BOOT_POOLED_GAP = ("delta_full", "delta_rr", "sim_minus_run1", "sim_minus_run2", "run2_minus_run1_at_tauB", "real_line")  # pooled gaps (l.3146-3151)
STAGE1_LISTING = ("585ec2169", "experiments/2026-10-15_e2e_sim_pipeline/results/check_stagea_inputs_alltasks_CRD_FULL_r2_allorigins_e12_2026-10-19_132210_"
                  "pwcatalogue_foldmodel_2026-10-20_005013.json", "846570b43cd3dc42d487bf40db7b6c0fea73a91ea242f3eead82eda8711639a1")  # route C Stage 1's
# verified inputs (S2, 10-20): the listing the S1 rerun ran on (apps ap-QQ3bQHwycFVLYPQDuED7sT shapes PASS, ap-f9PeVVrbFkYfXxkhlmwxMt 124 rules / 0
# violations, results ..._2026-10-20_010105.json @ 585ec2169); the 171-pool Stage 1 was ..._2026-10-16_211643.json @ c8887f6e7 (6ebc4695..., 40aaa031...)
STAGE1_INPUTS_SHA = "d5cfe184d0985fa8bee0ad6c96e41b72561b33a1353fbd316e49dfd100d75f83"  # route C Stage 1's inputs_sha256 (the listing @ 585ec2169)
STAGE1_IDENTITY = ("experiments/2026-10-15_e2e_sim_pipeline/results/check_stagea_inputs_alltasks_CRD_FULL_r2_allorigins_e12_2026-10-19_132210_pwcatalogue_"
                   "foldmodel_2026-10-20_005013.json", "846570b43cd3dc42d487bf40db7b6c0fea73a91ea242f3eead82eda8711639a1")  # route C: S1's listing
# itself carries identity_parts (136; inputs sha256 = S1's), so it is also the identity record (the 171-pool tie needed a separate recapture, 000133)
VALUE_ID_ALLOWED = {  # the only input parts a value run may hold differently from Stage 1's (keys of identity_parts), each with its reason
    "top|boot|sha256": "value run only: the seeded bootstrap-draw sample",
    "top|fold_not_covered|sha256": "the stated not-covered text (a value run never says 'catalogue stage only'; Astra r5 #6)",
    "pw|stage|sha256": "catalogue (Stage 1) vs scored (the value run)",
    "pw|git|sha256": "Stage 1 only: the catalogue build's code commit (a scored record pins its own pool inputs)",
    "pw|auth|sha256": "value run only: the scored stage's authorized overlaps (checked by key in check())",
    "pw|files|sha256": "the same pinned manifest (pw|manifest|sha256 equal): the catalogue stage takes its pooled files from it on Modal; the scored run asserts the "
                "record's files equal it (check())",
    "pw|pool_inputs|sha256": "the scored record's own pool-defining inputs; check() asserts them equal to the same pinned manifest's, module pins included"}
VALUE_PLANTS = (("sol_a_uncut C", "S_new (all-origins draws) [pooled: pooled_all_origins]", "run1"),  # the value plants (2e 22:3x plan): one fold-model
                ("FIBS|uncut C", "S_new (all-origins draws) [pooled: pooled_all_origins]", "run1"))  # and one external label, one member row each
VALUE_PLANTS_S2 = (("c_asis_uncut C", "S_new (all-origins draws) [pooled: pooled_all_origins]", "run1"),  # Stage 2 (09, 10-20 02:3x): one plant per
                   ("own_all_rs25_uncut_L48 C", "S_new (all-origins draws) [pooled: pooled_all_origins]", "run1"))  # Stage 2 reader family (five; own_all_rs)
STAGE2_TIE = (("1f82a0c6e", "experiments/2026-10-15_e2e_sim_pipeline/results/check_stagea_inputs_alltasks_CRD_FULL_r2_allorigins_e12_2026-10-19_132210_"
               "pwcatalogue_foldmodel_2026-10-20_023328.json", "54ab98efada64aaa83dc982217e258ffc4604557c7a4c0333b2b206e8586ac24"),
              "84c6a488d5c063b184a85352e1e79bd96b6ca0ea08d5fdf80856033967f916ac")  # route C Stage 2's inputs: ((commit, listing path, sha256),
# inputs_sha256): the Stage 2 S1 listing named in PRESTATED (2588 files; 152 identity parts), pinned with review F-13. The PRESTATED order still
# holds: no Stage 2 value run before S1-stage2 (shapes and check on this listing) PASSes.
STAGE2_AS_RECORDED = ("own_all_rs_uncut_L48 C", "own_all_rs_uncut_L48 R", "sol_a_uncut C", "sol_a_uncut R")  # 09 (10-20): a Stage 2 value run takes
# exactly these as recorded in the S/Y aggregates (they are not in its labels; reproduced at S3 @ 2e31d71c4); a Stage 1 value run takes none


def value_stage(fl):
    """A value run's stage from its fold labels: exactly one of FOLD_STAGES' families -> '1' | '2'."""
    st_ = [k_ for k_, v_ in FOLD_STAGES.items() if sorted(v_) == sorted(fl)]
    assert len(st_) == 1, ("--fold-labels: exactly one stage's families", fl)
    return st_[0]


def value_pins(stage):
    """-> (the stage's verified listing (commit, path, sha256), its inputs_sha256, its identity record (path, sha256), its value plants). Stage 2's
    tie STOPs until STAGE2_TIE is pinned (route C Stage 2 S1 first)."""
    if stage == "1":
        return STAGE1_LISTING, STAGE1_INPUTS_SHA, STAGE1_IDENTITY, VALUE_PLANTS
    assert STAGE2_TIE is not None, "Stage 2's tie is not pinned yet (STOP: the route C Stage 2 S1 run first)"
    return STAGE2_TIE[0], STAGE2_TIE[1], STAGE2_TIE[0][1:], VALUE_PLANTS_S2
VALUE_NOT_COMPARED = ["gap delta_full|B and catch sim@S|B / real_mean@B|B (estimator (B) needs the IL extra-arm rows, not loaded here)",
                      "scenario / hot-deck / removal keys on pooled rows (not computed by the runner there)", f"draws other than draw 0 and the {BOOT_N} seeded draws; interval quantiles (not recomputed)",
                      "per-draw PANEL G / Y (draw 0 only) and the per-draw fold -1 tallies",
                      "row fields moments_source, moments_fallback_firings, fold_minus1_draws (per-draw tallies)"]
VALUE_NOTES = [  # 2e 22:3x: the counted Sol fold -1 line and the Stage 1 commit-message correction (keys and counts only, from the Stage 1 record @ c8887f6e7)
    "Parity (507-prep 23:11 NB): ported line by line from the runner @ bd4af2fb9: wthr (l.1210), wmoments (l.1195), zpool incl. 21b and its skip "
    "rule (l.2226), zwith incl. the undefined-transform mask (l.2276), zatt (l.2261), ht / tpr (l.2204, l.2633), the validity masks (l.2546), the "
    "D-weighted all-origins combination (pool_reduce_n l.3124-3151) and draws_main (run_fprsweep_507 l.98); the pooled calibration the runner "
    "shares through its D1 files is recomputed here per label x combo from the pooled pools (as its wave 1 computes it); draw 0 and the 20 seeded "
    "draws run through the same per-draw path (draw 0 also against this check's draw-0 path); no other draw and no interval quantile is recomputed",
    "Route C (10-20): A's runner is run_peraction311 @ f2d47f19f (sha256 dcd265d7); the ported bodies (wmoments, wthr, pool, tpr, zpool, zatt, "
    "zwith, ht, the draw-loop head and validity masks, pool_reduce_n) are byte-identical to bd4af2fb9's and run_fprsweep_507.py is unchanged; "
    "inputs verified by the route C Stage 1 rerun (S1 @ 585ec2169: 124 rules, 0 violations, the same rule keys as c8887f6e7; 2218 files = the "
    "171-pool Stage 1's 2206 + 12 pilot-root files); route C's zero-unit pilot members are not ported (none in A: checked, fail closed)"]
# (the 171-pool record's Sol fold -1 counts and its c8887f6e7 commit-message correction are in c29b17114's checker; not restated for route C)
VALUE_FOLD_NOT_COVERED = FOLD_NOT_COVERED[:2] + [  # a value run's own line in place of "catalogue stage only" (Astra r5 #6)
    "the D1 shared calibration files are not read: each label x combo's pooled calibration (moments, validity masks, tau per draw) is recomputed here "
    "from the pooled pools as the runner's wave 1 computes it, and checked through every member row's and pooled value"]
FOLD_SCOPE = ("certifies that, for the fold-model labels, the pooled pools file, the manifest entry and the catalogue equal what the runner's own "
              "rules produce from these pinned files (export pass, unions, 09's identity gates); the ported rules are faithful copies of the "
              "runner's logic at bd4af2fb9, not an independent spec; input resolution (which files, keys and maps) is the runner's own, captured "
              "(Claude N9)")


def pw_identity_gates(sz_, pt_, pairs, zero=None, declared=frozenset(), prefix_fail=None):
    """09's identity ruling (PW_ID_RULE), on the union rebuilt here: (1) no false MERGE: no run key with different units, folds or per-model scores in
    two appearances (admitted zero-unit appearances included); (2) no false SPLIT: identical content under different keys only as EXACTLY one declared
    E12 pair (pairs: {frozenset((epoch-2 key, the epoch-1 key it replicates))}, the E12 block's maps) or as one of the runner's PW_GATE2_DECLARED
    groups (09 ruling (a'); declared), each of which must be SUPPORTED by the token-prefix record (prefix_fail(group) -> the unsupported pairs: ids
    identical through every readout and the same fold, by key, for this label's own scorer file; 2e 20:1x); (3) no honest_b_relaunch run, in either
    key form, in the pooled H_B, H_A or H_sim (zero-unit appearances included). -> violations (keys and counts only)"""
    v_ = []
    if sz_["identity_conflicts"]:
        v_.append(f"no false merge: {len(sz_['identity_conflicts'])} run keys with different content in two appearances, e.g. {sz_['identity_conflicts'][:3]}")
    spl_ = [g_ for g_ in sz_["content_identical_keys"] if (len(g_) != 2 or frozenset(g_) not in pairs) and frozenset(g_) not in declared]
    if spl_:
        v_.append(f"no false split: {len(spl_)} identical-content key groups that are neither one declared E12 pair nor a declared group, e.g. {spl_[:2]}")
    # 2e 20:2x (09): with the record given, an E12 pair needs the same support as a declared group (STRICTER than the runner's blanket exemption)
    uns_ = [(g_, f_) for g_ in sz_["content_identical_keys"] if prefix_fail is not None and (frozenset(g_) in declared or (len(g_) == 2 and frozenset(g_) in pairs))
            for f_ in [prefix_fail(g_)] if f_]
    if uns_:
        v_.append(f"declared groups or E12 pairs without token-prefix support: {len(uns_)}, e.g. {uns_[:2]}")
    rl_ = sorted({r for nm in ("B", "A", "S") for r in set(pt_[nm]) | set((zero or {}).get(nm) or {}) if RELAUNCH_KEY.match(r)})
    if rl_:
        v_.append(f"relaunch run keys in the pooled unions: {len(rl_)}, e.g. {rl_[:3]}")
    return v_


def scorer_of(p):  # the token-prefix record's scorer: the directory above the file, run / layers / pass / timestamp dirs skipped, a trailing _r<k> dropped
    parts = p.split("/")[:-1]
    while parts and (re.match(r"^[\w.@-]+__[\w.@-]+$", parts[-1]) or parts[-1] == "layers" or parts[-1].startswith("pass")
                     or re.match(r"^\d{4}-\d{2}-\d{2}_\d+$", parts[-1])):
        parts.pop()
    return re.sub(r"_r\d+$", "", parts[-1]) if parts else ""


def prefix_support(tp_, files_):
    """G7b gate 2 (2e 20:1x): the token-prefix record's support for a declared group in one label. tp_: {"pairs": [{"pair", "verdict", "files":
    [[scorer x, name x, scorer y, name y, outcome]]}]} (uncut line); files_: this label's (scorer, name) set. -> group -> its unsupported pairs: a pair
    is supported when the record's verdict is "(1) natural" and one of this label's own scorer files (same scorer on both sides) is "natural"."""
    by_ = {frozenset(p_["pair"]): p_ for p_ in tp_["pairs"]}

    def fail(g_):
        bad = []
        for x_, y_ in itertools.combinations(sorted(g_), 2):
            p_ = by_.get(frozenset((x_, y_)))
            if p_ is None or not p_["verdict"].startswith("(1)") or not any(
                    f_[4] == "natural" and f_[0] == f_[2] and f_[1] == f_[3] and (f_[0], f_[1]) in files_ for f_ in p_["files"]):
                bad.append([x_, y_, "no record pair" if p_ is None else "not natural for this label's scorer file"])
        return bad
    return fail


def pw_auth_rule_viol(rbk_, cat_, rule_):
    """The runner's pw_auth_overlap_check @ bd4af2fb9 (09 via 507-prep), independently, for one label x combo: the cross-pool overlaps (key -> roles,
    admitted basis) must be EXACTLY the catalogue's runs of rule["world"] (rule["n"] of them), each catalogued with sets == rule["sets"] and roles
    == rule["roles"]; every catalogue entry's world equals its key's world token. -> violations (keys and counts only)"""
    bw_ = sorted(r_ for r_, e_ in cat_.items() if e_.get("world") != r_.split("|", 1)[0])
    hc_ = {r_ for r_, e_ in cat_.items() if e_.get("world") == rule_["world"]}
    odd_ = sorted(r_ for r_ in hc_ if cat_[r_].get("sets") != rule_["sets"] or sorted(cat_[r_].get("roles") or []) != sorted(rule_["roles"]))
    got_ = {k_: sorted(v_) for k_, v_ in rbk_.items()}
    want_ = {k_: sorted(rule_["roles"]) for k_ in hc_}
    v_ = []
    if bw_ or odd_ or len(hc_) != rule_["n"]:
        v_.append(f"catalogue: {len(hc_)} {rule_['world']} runs (want {rule_['n']}); world differs from the key {bw_[:3]}; other sets or roles {odd_[:3]}")
    if got_ != want_:
        v_.append(f"cross-pool overlaps are not exactly the {rule_['world']} runs in {rule_['roles']}: other or differing "
                  f"{sorted(k_ for k_ in got_ if want_.get(k_) != got_[k_])[:3]}, missing {sorted(set(want_) - set(got_))[:3]}")
    return v_


def pw_overlap_viol(sz_, by_key):  # the runner's scored-stage overlap gate (item 46): every cross-pool overlap authorized by key for its roles
    und_ = [k_ for k_, rs_ in sz_["roles_by_key"].items() if k_ not in by_key or not set(rs_) <= set(by_key[k_]["roles"])]
    return [f"{len(und_)} cross-pool overlaps not authorized for their roles, e.g. {und_[:3]}"] if und_ else []


# ---- G7b FOLD-MODEL: the runner's fold-model readers and rules (run_peraction311 @ bd4af2fb9), written here; keys, folds and counts only
SP_WORLD_ALIAS = {("sp493_sim", "sim"), ("sim_attack", "sim"), ("attack", "run1"), ("attack", "run2"), ("sim_honest", "sp493_t196"),
                  ("sim_honest", "sp493_new")}  # the runner's l.437 (compared at capture)


def sp_world_chk(k_, world, where):
    """sp_world (l.464): an S_paper493 file's single world token renamed to its folder's routing world when the pair is an allowed alias; several
    tokens or another pair STOP. -> (keys, the renamed token or None)"""
    import numpy as np

    toks = sorted({x.split("|", 1)[0] for x in k_.tolist()})
    if not toks or toks == [world]:
        return k_, None
    assert len(toks) == 1 and (toks[0], world) in SP_WORLD_ALIAS, (where, f"world tokens {toks} in a {world} folder (the runner STOPs)")
    return np.array([world + "|" + x.split("|", 1)[1] for x in k_.tolist()]), toks[0]


G7B_PRIOR = ("55ca7539f", "experiments/2026-10-15_e2e_sim_pipeline/results/check_stagea_alltasks_CRD_FULL_r2_allorigins_e12_2026-10-16_113538_pwcatalogue_2026-10-16_200826.json",
             "2c72503d4d9c23263efef21c15e2097e5cb7772c12f311267a4956ccccbaa01f")  # the 8-label G7b PASS record (2e 21:1x: compared by key)


CMP_ALLOW_NEW = ("cfg0 regression: ", "fold-model rules on ")  # the only rules a public label may gain (info, no violations): 2e 21:3x allowlist
CMP_ALLOW_FIELDS = {"row", "rule", "violations", "info"}


def compare_g7b(old, new, n_labels=8, want_labels=None):
    """2e 21:1x / 21:3x (Astra r2 #1, 507-prep NB3): the 8 public labels' per-gate statuses in this run against the earlier G7b record, TWO-SIDED by
    (row, rule): both rule domains non-empty and unique, the prior on exactly 8 labels, both records all_equal, no row or pooled entries for those
    labels on either side; a shared rule must be the SAME record (field presence and every value, canonical JSON; a missing field is not a null);
    a rule only in the old record, or only in the new one outside the allowlist (cfg0 regression and alias info rules, with no violations and no
    other fields), is a difference. Also the S_paper493 world-alias renames per public label x combo. -> (differences, renames)"""
    cj = lambda x: json.dumps(x, sort_keys=True)  # noqa: E731
    ok_ = lambda r_: isinstance(r_, dict) and isinstance(r_.get("row"), str) and isinstance(r_.get("rule"), str)  # noqa: E731  (507-prep NB-d)
    mal_ = [["*", "*", f"{w_} record: {sum(not ok_(r_) for r_ in (d_.get('rules') or []))} malformed rule entries (no row or rule)"]
            for w_, d_ in (("old", old), ("new", new)) if any(not ok_(r_) for r_ in (d_.get("rules") or []))]
    ol_ = [r_ for r_ in old.get("rules") or [] if ok_(r_)]
    labs_ = sorted({r_["row"] for r_ in ol_})
    nl_ = [r_ for r_ in new.get("rules") or [] if ok_(r_) and r_["row"] in labs_]
    o_ = {(r_["row"], r_["rule"]): r_ for r_ in ol_}
    n_ = {(r_["row"], r_["rule"]): r_ for r_ in nl_}
    diff = mal_ + ([["*", "*", f"an empty rule domain on one side, or a prior not on {n_labels} labels"]] if not ol_ or not nl_ or len(labs_) != n_labels else [])
    diff += [["*", "*", f"the prior's labels are not the expected ones: {labs_}"]] if want_labels is not None and labs_ != sorted(want_labels) else []
    diff += [["*", "*", "duplicate (row, rule) keys in the old or the new record"]] if len(o_) != len(ol_) or len(n_) != len(nl_) else []
    diff += [["*", "*", f"the {w_} record is not all_equal"] for w_, r_ in (("old", old), ("new", new)) if r_.get("all_equal") is not True]
    diff += [[lb, "*", f"{k_} entries for a public label"] for k_ in ("rows", "pooled") for lb in labs_
             if any(r_.get("row") == lb for r_ in (old.get(k_) or []) + (new.get(k_) or []))]
    diff += [[*k_, "rule missing in the new record"] for k_ in sorted(set(o_) - set(n_))]
    for k_ in sorted(set(o_) & set(n_)):
        if cj(o_[k_]) != cj(n_[k_]):
            diff += [[*k_, f"field {f_} present on one side only"] for f_ in sorted(set(o_[k_]) ^ set(n_[k_]))]
            diff += [[*k_, f"field {f_}"] for f_ in sorted(set(o_[k_]) & set(n_[k_])) if cj(o_[k_][f_]) != cj(n_[k_][f_])]
    for k_ in sorted(set(n_) - set(o_)):
        r_ = n_[k_]
        if r_.get("violations") != [] or not k_[1].startswith(CMP_ALLOW_NEW) or not set(r_) <= CMP_ALLOW_FIELDS:
            diff.append([*k_, "a new rule outside the allowlist (cfg0 regression / alias info, no violations, no other fields)"])
    ren = {}
    for (r0_, ru_), r_ in sorted(n_.items()):
        if ru_.startswith("fold-model rules on "):
            for v_ in ((r_.get("info") or {}).get("sp_world_aliases") or {}).values():
                cb_ = PW_SFX.search(ru_.split(" (info", 1)[0])
                ren[f"{r0_} | {cb_.group(1) if cb_ else '?'}"] = ren.get(f"{r0_} | {cb_.group(1) if cb_ else '?'}", 0) + int(v_["renamed_rows"])
    return diff, {"labels": labs_, "renamed_rows_by_label_combo": {f"{lb} | {cb}": ren.get(f"{lb} | {cb}", 0) for lb in labs_
                                                                    for cb in ("pooled_all_origins", "paper_cells")} | {k_: v_ for k_, v_ in ren.items()}}


def rows_from_npz(z, five, fam, simh=False, efold=None, where=""):
    """check()'s per-row reader (C.rows_of at five models, rows_any's own_all_rs / LOO branches, l.1722's A25 sim-honest file without fold):
    -> key, designated score, fold (None: an A25 sim-honest file, r36_fill sets it), five model scores (rows x 5; NaN where absent)"""
    import numpy as np

    k = z["key"].astype(str)
    n = len(k)
    if efold is not None:  # own_all_LOO: every row scored by its origin's held-out model (o0 S_old, o1 S_new, o2 S_newmains), raw scores
        fz = z["fold"].astype(np.int64)
        assert not n or bool(np.all(fz == efold)), (where, "own_all_LOO rows not all scored by model", efold, sorted(set(fz.tolist()))[:5])
        return k, z["score"].astype(np.float64), np.zeros(n, np.int64), np.full((n, 5), np.nan)
    if not five:
        return k, z["score"].astype(np.float64), np.zeros(n, np.int64), np.full((n, 5), np.nan)
    if simh and "fold" not in z.files and fam != "own_all_rs":  # A25 Sol / c_asis sim-honest file (l.1722): no fold; the caller sets r36 (r36_fill)
        return k, z["score"].astype(np.float64), None, np.full((n, 5), np.nan)
    assert "fold" in z.files, (where, "a five-model file without fold outside the A25 sim-honest rule (the runner STOPs: rows_of reads z['fold'])")
    fr_ = z["fold"]  # Claude N3 / Astra non-blocking: dtype and range BEFORE the int64 cast (stricter than the runner, which casts first)
    assert k.ndim == 1 and fr_.shape == (n,) and (fr_.dtype.kind in "iu" or (fr_.dtype.kind == "f" and bool(np.all(fr_ == np.trunc(fr_))))) \
        and bool(np.all((fr_ >= -1) & (fr_ <= 4))), (where, "five-model rows: key / fold not 1-D, a non-integral fold or one outside -1..4")
    f = fr_.astype(np.int64)
    sb = (z["score_by_model"] if "score_by_model" in z.files else z["score_by_fold"] if "score_by_fold" in z.files else np.full((n, 5), np.nan)).astype(np.float64)
    assert sb.shape == (n, 5), (where, sb.shape)
    s = z["score"].astype(np.float64) if "score" in z.files else np.where(f >= 0, sb[np.arange(n), np.clip(f, 0, 4)], np.nan)
    if fam == "own_all_rs":
        rs_rows_chk(z.files, k, s, f, where)
    return k, s, f, sb


def rs_rows_chk(names, k, s, f, where):
    """rows_rs (l.476): an own_all_rs file has score, score_by_model and fold; every row on fold 0-4 (never -1) with a finite designated score.
    The runner STOPs otherwise. names: the npz's array names."""
    import numpy as np

    assert {"score", "score_by_model", "fold"} <= set(names), (where, "own_all_rs file without score, score_by_model or fold (the runner STOPs)")
    assert not len(k) or (bool(np.all((f >= 0) & (f <= 4))) and bool(np.isfinite(s).all())), \
        (where, "own_all_rs rows outside folds 0-4 or non-finite designated scores (the runner STOPs)", sorted(set(f.tolist()))[:6])


def r36_fill(hk, u_run, u_fold, u_src):
    """l.1722-1732: an A25 Sol / c_asis sim-honest file carries no fold; each row takes its cell's fold from the row's S_old real rows (fold >= 0,
    U order, the last row of a cell wins as the runner's dict); a cell without one STOPs. -> (folds, cells whose S_old rows disagree on the fold)"""
    import numpy as np

    fm_, seen_ = {}, {}
    for r_, f_, s_ in zip(u_run.tolist(), u_fold.tolist(), u_src.tolist()):
        if s_ == "S_old" and f_ >= 0:
            c_ = r_.split("|", 1)[1]
            seen_.setdefault(c_, set()).add(int(f_))
            fm_[c_] = int(f_)
    f = np.array([fm_.get(x.split("|", 1)[1], -1) for x in hk.tolist()], np.int64)
    assert not (f < 0).any(), ("an A25 sim-honest task without a real-side fold (the runner STOPs)", sorted({x for x, y in zip(hk.tolist(), f) if y < 0})[:3])
    return f, sorted(c_ for c_, v_ in seen_.items() if len(v_) > 1)


def rs_fold_viol(runs, folds, cmain, fold_of_main):
    """rs_fold_check (l.491, 09 3a (c)): every own_all_rs unit sits on its main's fold in the committed list, by key. -> the off-list units (keys)"""
    bad = []
    for r_, f_ in zip(runs, folds):
        m_ = cmain.get(r_.split("|", 1)[1])
        if m_ not in fold_of_main or fold_of_main[m_] != int(f_):
            bad.append(r_)
    return bad


def sp_r36_off(cell, fold, src, hk, hf, r36):
    """l.1820-1832 (Append 39 am.1 §2 + correction §3): Sol / c_asis S_paper493 rows sit on their main's held-out r36 fold (None = -1); H_sim rows on
    S_paper493 cells on r36, other H_sim rows on their cell's real-side fold (S_old / S_new rows; a cell with two folds there has none). -> off keys"""
    r36_ = {c: (-1 if v is None else int(v)) for c, v in r36.items()}
    off_ = sorted({c for c, f_, s_ in zip(cell.tolist(), fold.tolist(), src.tolist()) if s_ == "S_paper493" and r36_.get(c) != int(f_)})
    realf_ = {}
    for c, f_, s_ in zip(cell.tolist(), fold.tolist(), src.tolist()):
        if s_ in ("S_old", "S_new") and realf_.setdefault(c, int(f_)) != int(f_):
            realf_[c] = None
    off_ += sorted({r_ for r_, f_ in zip(hk.tolist(), hf.tolist())
                    if int(f_) != (r36_[r_.split("|", 1)[1]] if r_.split("|", 1)[1] in r36_ else realf_.get(r_.split("|", 1)[1]))})
    return off_


def fold_pool_stops(pools, use21b, oof):
    """The runner's per-pool STOPs before the export (fold-model probes): pool() l.2028-2037 (a fold with < 2 units takes all-unit moments; unless
    the out-of-fold fallback applies (own_all_rs, by label) its score_by_model must be finite), 'lacks a fold' l.2091-2096 (H_B, H_A, H_sim and the
    removal pools, unless 21b or oof), 21b's fold -1 units with five finite model scores. pools: {name: (keys, s, f, sbm)}. -> reasons"""
    import numpy as np

    why = []
    for nm, (_, s_, f_, b_) in pools.items():
        col = any(int((f_ == j).sum()) < 2 for j in range(5))
        oo_ = col and oof
        if col and not oo_ and len(b_) and not np.isfinite(b_).all():
            why.append(f"{nm}: a pool fold has < 2 units and its off-fold score_by_model entries are not all finite (STOP)")
        if nm in ("B", "A", "S", "Snr", "Anr") and not use21b and not oo_ and col:
            why.append(f"{nm}: the pool lacks a fold (STOP)")
        if nm in ("B", "A", "S") and use21b and not np.isfinite(b_[f_ == -1]).all():
            why.append(f"{nm}: fold -1 units without five finite model scores (STOP)")
    return why


def f1cols_of(rfold, coh, main_of, mains_sorted):
    """F1COLS (l.2146-2160) on an all-origins row: each attack arm's fold -1 runs as union-stream main columns (sorted merge mains; a main outside
    them is the extra column len(mains)); B = run1 u run2 u sim, A = run1 u run2, S = sim."""
    col_ = {m: i for i, m in enumerate(mains_sorted)}
    f1 = {w: {col_.get(main_of[coh[i]], len(mains_sorted)) for i, f0 in enumerate(rfold[w]) if int(f0) == -1} for w in A3}
    return {"B": f1["run1"] | f1["run2"] | f1["sim"], "A": f1["run1"] | f1["run2"], "S": set(f1["sim"])}


def mvf_withheld(mvf, allunits_dev, fxp, spo):
    """l.1834-1841 (25h §1): a fold-model row on S_newmains or S_paper493 is withheld when a pool (H_B, H_A, H_sim) has no mapped unit, unless
    (S_newmains only) the all-units deviation or the held-out fold extension applies: min mapped == 0 and ((not allunits_dev and not FXP) or SPO)."""
    return min(v_["mapped"] for v_ in mvf.values()) == 0 and ((not allunits_dev and not fxp) or bool(spo))


def wthr_w(z, w, f):
    """wthr (l.1210-1229, primary scenario) at ONE draw, line by line: zs = z sorted descending (argsort(-z), mergesort, as the caller l.2526) with
    its weights ws, cw = cumsum(ws), k = floor(f * (sum w) + 1e-9), T = zs[min(#(cw <= k), n - 1)], NaN where k < 1. The caller masks invalid draws."""
    import numpy as np

    z1, w1 = np.asarray(z, dtype=np.float64)[None, :], np.asarray(w, dtype=np.float64)[None, :]
    assert z1.shape == w1.shape and z1.shape[1] > 0, "wthr: an empty pool or misaligned weights"
    o = np.argsort(-z1, axis=1, kind="mergesort")
    zs, ws = np.take_along_axis(z1, o, 1), np.take_along_axis(w1, o, 1)
    cw = np.cumsum(ws, 1)
    tot = cw[:, -1]
    n = zs.shape[1]
    k = np.floor(f * (tot + 0.0) + 1e-9)
    pos, ok = k, k >= 1
    idx = np.minimum((cw <= pos[:, None]).sum(1), n - 1)
    T = zs[np.arange(zs.shape[0]), idx]
    return float(np.where(ok, T, np.nan)[0])


def wthr0(z, f):  # wthr at draw 0 (every unit weight 1)
    import numpy as np

    return wthr_w(z, np.ones(len(z)), f)


def pool_zw(P_, five, oof, wu, attw):
    """A pool's unit z and moments at ONE draw, ported line by line from the runner @ bd4af2fb9 (= A's executed blob 0b65ed9d): zpool l.2226-2259 and
    wmoments l.1195-1207 in the runner's own reduction forms (a 1 x n weight row, w @ x / tw, (w0 * m).sum / t0). The runner applies them to a
    block of draws at once, so a one-row product here can differ from its batched one in the last bits (within the 1e-9 tolerance; it could flip
    a catch only at an exact threshold tie, runner l.2527). wu: this draw's unit weights (Wb[b, P_["col"]]: each unit carries its main's multiplicity; draw 0 and mode 'fixed': all 1); attw: this
    draw's weight on the label x combo's attack fold -1 columns (Wb[b, f1cols].sum()). A single-model probe keeps its raw scores. -> (z, (MU, SD, mu0,
    sd0) or None (mu0 / sd0 after the skip substitution, as zpool returns them), ok: the runner's validity l.2546-2549, skip: fold -1 transform skipped)"""
    import numpy as np

    _, s_, f_, b_ = P_
    if not five:
        return s_, None, True, False
    folds = list(range(5))
    Wu = np.asarray(wu, dtype=np.float64)[None, :]  # l.2227: Wu = Wb[:, P_["col"]]
    assert Wu.shape[1] == len(s_), "pool weights misaligned"
    col_moments = any(int((f_ == j).sum()) < 2 for j in folds)  # pool() l.2032: a fold with < 2 units (by count): all-unit moments per model
    with np.errstate(divide="ignore", invalid="ignore"):  # the runner's 0 / 0 gives NaN with a warning; the values are the same
        if col_moments:  # l.2230-2236
            mu, sd = {}, {}
            tw = Wu.sum(1)
            for f in folds:
                x = s_ if oof else b_[:, f]  # oof: the pooled out-of-fold designated scores (cross-fitted probe; 09)
                mu[f] = (Wu @ x) / tw
                sd[f] = np.sqrt(np.maximum((Wu @ (x ** 2)) / tw - mu[f] ** 2, 0.0))
        else:  # l.2237: wmoments(Wu, s, where(f >= 0, f, -9), folds), l.1195-1207
            fold = np.where(f_ >= 0, f_, -9)
            mu, sd = {}, {}
            for f in folds:
                m = fold == f
                w = Wu[:, m]
                tw = w.sum(1)
                mu[f] = (w @ s_[m]) / tw
                sd[f] = np.sqrt(np.maximum((w @ (s_[m] ** 2)) / tw - mu[f] ** 2, 0.0))
                mu[f] = np.where(tw >= 2, mu[f], np.nan)
        MUa, SDa = np.stack([mu[f] for f in folds], 1), np.stack([sd[f] for f in folds], 1)  # l.2238
        fm = np.maximum(f_, 0)
        z = (s_[None, :] - MUa[:, fm]) / SDa[:, fm]  # l.2240
        mu0 = sd0 = None
        skip = False
        un = np.flatnonzero(f_ == -1)
        if len(un):  # l.2242-2258 (21b; 7310476ae: unused, so only mu_0 / sigma_0 are skipped)
            m = ((b_[None, un, :] - MUa[:, None, :]) / SDa[:, None, :]).mean(2)
            w0 = Wu[:, un]
            t0 = w0.sum(1)
            mu0 = (w0 * m).sum(1) / t0
            sd0 = np.sqrt(np.maximum((w0 * m ** 2).sum(1) / t0 - mu0 ** 2, 0.0))
            z[:, un] = (m - mu0[:, None]) / sd0[:, None]
            sk_ = (t0 == 0) & (np.array([float(attw)]) == 0)
            mu0, sd0 = np.where(sk_, 0.0, mu0), np.where(sk_, 1.0, sd0)
            z[np.ix_(np.flatnonzero(sk_), un)] = -np.inf  # weight 0 in these draws
            mu0, sd0, skip = float(mu0[0]), float(sd0[0]), bool(sk_[0])
    MU, SD = MUa[0], SDa[0]
    ok = bool(np.all(np.isfinite(MU)) and np.all(SD > 0) and (mu0 is None or (np.isfinite(mu0) and sd0 > 0)))
    return z[0], (MU, SD, mu0, sd0), ok, skip


def pool_z0(P_, five, oof=False):
    """A pool's draw-0 unit z and moments (every unit weight 1; the skip rule cannot fire: t0 = the fold -1 unit count > 0). -> (z, mom, ok)"""
    import numpy as np

    return pool_zw(P_, five, oof, np.ones(len(P_[1])), 1.0)[:3]


def pool_zw_with(P_, ZPB, wu, t0r, where=""):
    """zwith l.2276-2296 (item 44(3)) at ONE draw: a pool's units standardized with ANOTHER pool's moments (pooled H_A with pooled H_B's; H_A fits
    none); fold -1 units need that pool's fold -1 transform (else the runner STOPs). wu: this draw's weights of P_'s units; t0r: this draw's weight on
    the reference pool's fold -1 units (0: its transform was skipped). -> (z, the other pool's moments, validity: the reference's and not 'bad', bad)"""
    import numpy as np

    _, s_, f_, b_ = P_
    if ZPB[1] is None:
        return s_, None, ZPB[2], False
    MU, SD, mu0, sd0 = ZPB[1]
    z = (s_ - MU[np.maximum(f_, 0)]) / SD[np.maximum(f_, 0)]
    un = np.flatnonzero(f_ == -1)
    bad = False
    if len(un):
        assert mu0 is not None, (where, "pooled H_A has fold -1 units but the pooled H_B has no fold -1 transform (the runner STOPs)")
        z = z.copy()
        z[un] = (((b_[un, :] - MU[None, :]) / SD[None, :]).mean(1) - mu0) / sd0
        wun_ = float(np.asarray(wu, dtype=np.float64)[un].sum())
        bad = bool(t0r == 0 and wun_ > 0)  # needed but undefined in that draw: the whole A threshold is invalid there (Astra r1 #6)
        if bad:
            z[un] = np.nan
        elif t0r == 0:  # unused (weight 0)
            z[un] = -np.inf
    return z, ZPB[1], bool(ZPB[2]) and not bad, bad


def pool_z0_with(P_, ZPB, where="", t0r=1.0):
    """pool_zw_with at draw 0 (every weight 1; t0r = the reference pool's fold -1 unit count, > 0 whenever it has a transform). -> (z, mom, ok)"""
    import numpy as np

    return pool_zw_with(P_, ZPB, np.ones(len(P_[1])), t0r, where)[:3]


def identity_parts(inp):
    """G7b VALUE (Astra r5 #1 / 2e 23:1x item 1): the Stage 1 tie BY IDENTITY: every part of a check's resolved inputs by the sha256 of its canonical
    JSON (sorted keys): each top-level field (file maps, cohorts' and pools' key lists, main maps, class lists, ...), each label's probe entry (its
    files path -> sha256, folds, fit), and each label x origin variant (cohort, pool keys, maps). -> {part: sha256}; every part name ends in
    "|sha256" and the listing stores sorted [name, sha256] pairs (the repo's secret scanner read a digest under a field named like a credential)"""
    hj_ = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()  # noqa: E731
    out = {f"top|{k_}|sha256": hj_(v_) for k_, v_ in inp.items() if k_ not in ("by_label", "variants_by_label", "pw")}
    out |= {f"pw|{k_}|sha256": hj_(v_) for k_, v_ in (inp.get("pw") or {}).items()}  # the pooled-world build field by field (stage, manifest, files ...)
    out |= {f"by_label|{lb}|sha256": hj_(v_) for lb, v_ in (inp.get("by_label") or {}).items()}
    out |= {f"variant|{lb}|{o}|sha256": hj_(v_) for lb, vv_ in (inp.get("variants_by_label") or {}).items() for o, v_ in vv_.items()}
    return out


def arm_runs_of(w, U, att_w, fold, coh, five, zero):
    """An arm's cohort runs from the SOURCE unit rows (U, the row's unit arrays; att_w: the arm's attack-unit mask): per cohort cell, the run's max
    designated score over its mapped finite units (zatt's sm), its first fold, and its fold -1 units' five model scores; a structural-zero run
    without rows is -inf (never above T). -> (keys, rmax, rfold, runits, fold -1 units by run)"""
    import numpy as np

    base_ = w.split("_p")[0]
    d_, fo_, ub_ = {}, {}, {}
    for r_, s_, f_, b_ in zip(U["run"][att_w].tolist(), U["score"][att_w].tolist(), fold[att_w].tolist(), U["sbm"][att_w]):
        fo_.setdefault(r_, int(f_))
        if (int(f_) >= 0 or not five) and np.isfinite(s_):
            d_[r_] = max(d_.get(r_, -np.inf), s_)
        elif five and int(f_) == -1:
            ub_.setdefault(r_, []).append(b_)
    keys_ = [f"{base_}|{c}" for c in coh]
    rm_ = np.array([(-np.inf if (k_ in zero and k_ not in d_ and k_ not in ub_) else d_.get(k_, np.nan)) for k_ in keys_])
    return keys_, rm_, np.array([fo_.get(k_, 0) for k_ in keys_]), [np.array(ub_[k_]) if k_ in ub_ else None for k_ in keys_], ub_


def zrun_of(rmax_w, rfold_w, runits_w, mom, five, where=""):
    """zatt (l.2261-2274): attack-run z under one pool's moments; a fold -1 run is the max over its units of the 21b transform (NaN without one); a
    single-model probe keeps its raw run max."""
    import numpy as np

    if not five:
        return rmax_w
    assert mom is not None, (where, "a fold-model pool without moments")
    MU, SD, mu0, sd0 = mom
    fo = np.maximum(rfold_w, 0)
    z = (rmax_w - MU[fo]) / SD[fo]
    for i_, (f0, ub) in enumerate(zip(rfold_w, runits_w)):
        if f0 == -1:
            z[i_] = np.nan if (mu0 is None or ub is None) else float((((((ub - MU[None, :]) / SD[None, :]).mean(1)) - mu0) / sd0).max())
    return z


def plant_target(w, U, att_w, fold, coh, five, zero, ZPB):
    """G7b VALUE PLANT: the run a plant moves, i.e. the arm's lowest draw-0 z run among its finite mapped runs (under the pooled H_B moments, ZPB =
    (z, moments, ok)), ties by cell key. One function for the planted call (apply_plant) and the unplanted call's pre-plant record (09 (c),
    10-20 ~04:5x), so both name the same run. -> its cohort index"""
    import numpy as np

    _, rm_, rf_, ru_, _ = arm_runs_of(w, U, att_w, fold, coh, five, zero)
    zc_ = zrun_of(rm_, rf_, ru_, ZPB[1] if five else None, five, "plant")
    cand_ = [i_ for i_, (v_, fo_) in enumerate(zip(np.asarray(zc_).tolist(), rf_.tolist())) if np.isfinite(v_) and (fo_ >= 0 or not five)]
    assert cand_, (w, "plant: no finite mapped run in the arm")
    return min(cand_, key=lambda i_: (zc_[i_], coh[i_]))


def apply_plant(w, U, att_w, fold, coh, five, zero, ZPB):
    """G7b VALUE PLANT (Astra r5 #1 / #3, r6 4): ONE SOURCE unit score moves. The arm's lowest draw-0 z run among its finite mapped runs (under the
    pooled H_B moments, ZPB = (z, moments, ok); plant_target) gets its max unit set so that its z is the max pooled H_B unit z + 1 (caught at every
    level with a finite tau_B, if not caught already); the arm's runs are then rebuilt from the source (arm_runs_of), and the moved z is asserted.
    U is not modified (a copy of its score array). -> (U with the moved score, the rebuilt (keys, rmax, rfold, runits, ub), keys-only info)"""
    import numpy as np

    mom_ = ZPB[1] if five else None
    i0_ = plant_target(w, U, att_w, fold, coh, five, zero, ZPB)
    zb_ = np.asarray(ZPB[0], dtype=np.float64)
    zt_ = float(zb_[np.isfinite(zb_)].max()) + 1.0
    rk_ = f"{w.split('_p')[0]}|{coh[i0_]}"
    ui_ = np.flatnonzero((U["run"] == rk_) & att_w & ((fold >= 0) if five else True) & np.isfinite(U["score"]))
    assert len(ui_), (rk_, "plant: the run has no mapped finite unit")
    j_ = int(ui_[np.argmax(U["score"][ui_])])  # the run's max unit
    U2 = dict(U) | {"score": U["score"].copy()}
    U2["score"][j_] = (float(mom_[0][int(fold[j_])]) + zt_ * float(mom_[1][int(fold[j_])])) if five else zt_  # THE source-score assignment
    built_ = arm_runs_of(w, U2, att_w, fold, coh, five, zero)  # the normal path again, from the source
    z2_ = zrun_of(built_[1], built_[2], built_[3], mom_, five, "plant")
    assert float(z2_[i0_]) > zt_ - 1e-6, (rk_, "plant: the run's z did not move above the pooled H_B units")
    return U2, built_, {"run": rk_, "cohort_index": int(i0_), "unit_rank_in_run": int((ui_ < j_).sum())}


def vacuous(cmp):
    """G7b VALUE (507-prep 23:11 NB; Astra r6 1): comparison keys whose two sides are both NaN where a value is required: every draw-0 catch / gap /
    panel key and EVERY seeded-draw key ("boot|..."). In a value run each is a violation (never a MATCH that counts). -> keys"""
    return [k_ for k_, a_, b_ in cmp if nan_both(a_, b_) and (k_.startswith("boot|") or re.search(r"\|(catch|gap|panel)\|", k_))]


def fresh_dest(dest):
    """Astra r6 3: a reduction's results file never reuses a name an earlier attempt took (an incomplete reduction keeps its file; a later one writes
    <name>_attempt<k>.json); the caller still creates it exclusively."""
    k_ = 2
    out = dest
    while out.exists():
        out = dest.with_name(f"{dest.stem}_attempt{k_}{dest.suffix}")
        k_ += 1
    return out


def boot_ids(seed=None):
    """G7b VALUE (09 via 2e 23:2x): the seeded sample of bootstrap draw ids, BOOT_N of 1..BOOT_D without replacement, seed BOOT_SEED (stated), sorted.
    PLANT V3 (AMEND4 D6, AMEND5 F3): a record-specific seed (BOOT_SEED_BY_RECORD) through the same rule; the default is unchanged."""
    import numpy as np

    return sorted(int(x) for x in np.random.default_rng(BOOT_SEED if seed is None else seed).choice(np.arange(1, BOOT_D + 1), BOOT_N, replace=False))


def draws_main_port(units, key_of, seed, D=BOOT_D):
    """run_fprsweep_507.draws_main (l.98-110 @ bd4af2fb9) ported line by line: sorted units (main IDs), strata = key tuples in lexicographic order, ONE
    fresh default_rng(list(seed)); per stratum multinomial(n_s, uniform, size=D) for draws 1..D; draw 0 all ones. -> (units, W (D + 1 x units))."""
    import numpy as np

    units = sorted(units)
    rng, W_ = np.random.default_rng(list(seed)), np.zeros((D + 1, len(units)))
    W_[0] = 1.0
    for k in sorted({key_of[u] for u in units}):
        idx = np.array([i for i, u in enumerate(units) if key_of[u] == k])
        W_[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=D)
    return units, W_


def draws_hash(ucol, W):
    """The runner's draws attestation (l.1705-1708): sha256 of the sorted (main, column) map's JSON + W's float64 bytes (C order)."""
    import numpy as np

    return hashlib.sha256(json.dumps(sorted((str(k_), int(v_)) for k_, v_ in ucol.items())).encode()
                          + np.ascontiguousarray(W, dtype=np.float64).tobytes()).hexdigest()


def boot_draws(merge, canon, merge_mains, ids=None):
    """The all-origins union stream as the runner builds it on an all-origins row (l.1697-1704): draws_main over the record's meta.merge (mains,
    env_of_main strata, seed), ucol over its sorted mains, every canonical main outside the stream (canon = MG main_of_cell's values) at column n, W
    with a column of ones appended; the attestation hash over the FULL W. -> {"ids": [0] + the sample, "W": W at those draws, "ucol", "draws_sha256"}"""
    import numpy as np

    assert sorted(merge["mains"]) == list(merge_mains), "the record's union-stream mains differ from the captured ones"
    units, W0 = draws_main_port(merge["mains"], {m: (merge["env_of_main"][m],) for m in merge["mains"]}, merge["seed"])
    ucol = {m: i for i, m in enumerate(units)}
    W = np.hstack([W0, np.ones((W0.shape[0], 1))])
    for m in sorted({m for m in canon.values() if m not in ucol}):
        ucol[m] = W0.shape[1]
    ids = [0] + boot_ids() if ids is None else [int(x_) for x_ in ids]  # PLANT V3 (Codex V3 #1): the run's own seeded ids (PLANT: seed 20261020)
    assert ids[0] == 0 and ids[1:] == sorted(set(ids[1:])) and len(ids) == BOOT_N + 1, "the sampled draw ids"
    return {"ids": ids, "W": W[ids].copy(), "ucol": ucol, "draws_sha256": draws_hash(ucol, W)}


def npz_part(p_):
    """The runner's npz key part (l.1148-1153), the plain v2 map: '@' -> '_at_', '|' -> '_', ' ' -> '_', parentheses dropped; no part holds '__'."""
    q = str(p_).replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")
    assert "__" not in q and q, f"npz key part {p_!r} would be ambiguous"
    return q


def npzkey(*parts):  # l.1156-1158 with no label escapes (pooled A's npz has none: asserted where it is read)
    return "__".join(npz_part(p_) for p_ in parts)


def boot_catch(z, T, Wrow, ccol, wcell):
    """ht + tpr (l.2204-2215, l.2633) at ONE draw on an all-origins row: NaN when tau is NaN, else sum_c Wc hit_c / sum_c Wc with Wc = W[b, ccol] *
    wcell (1 x cells, the runner's form; hit = z > T, so NaN is never above and +inf always); an empty cohort gives the runner's 0 / 0 = NaN."""
    import numpy as np

    Wc = np.asarray(Wrow, dtype=np.float64)[None, :][:, np.asarray(ccol, dtype=int)] * np.asarray(wcell, dtype=np.float64)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ = (Wc * (np.asarray(z, dtype=np.float64)[None, :] > T)).sum(1) / Wc.sum(1)
    return float(np.where(np.isnan(T), np.nan, t_)[0])


def boot_comb(per, D, origs, k_):
    """pool_reduce_n's all-origins combination (l.3132-3134) at one draw: sum over members (Python sum, member order) of where(D_o > 0, X_o D_o, 0) over
    sum_o D_o (the runner's comb on COMBOS_ALLO; 0 / 0 = NaN when every member's D is 0)."""
    import numpy as np

    Dd = sum(np.float64(D[o]) for o in origs)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(sum(np.where(np.float64(D[o]) > 0, np.float64(per[o][k_]) * np.float64(D[o]), 0.0) for o in origs) / Dd)


# ---- PLANT V3 (PRESTATED_V3 fb34c414 + AMEND1-5): the runner's plant (run_peraction311.py @ 752413181, blob 71633875), ported line by line ----
def plant_fit_v3(zs0, T0, ht0, deltas):
    """run_peraction311 @ 752413181 L2465-2496 (plant_fit), ported at draw 0 (the runner calls it with cur[0] == 0 and reads only draw 0): zs0 = the
    row's sim attack-run z at draw 0 (cohort order), T0 = the draw-0 tau_sim, ht0(z) = the row's draw-0 catch (boot_catch at draw 0, the runner's
    ht_(zs, T)[0]). -> (the solve record, {delta: c}) exactly as the runner writes rows.<label>.<origin>.plant.solve_draw0.<f>.sim / plant_cx."""
    import numpy as np

    T0 = float(T0)
    z0 = np.asarray(zs0, dtype=np.float64)
    fin0 = np.isfinite(z0)
    if not np.isfinite(T0) or not fin0.any():
        return {"T0": T0, "status": "unavailable (draw-0 tau_sim not finite or no finite sim attack z)"}, {}
    base = float(ht0(z0))
    fc_ = lambda c: float(ht0(z0 + c)) - base  # noqa: E731  (catch change at draw 0; non-decreasing in c)
    lo_all, hi_all = min(float(np.min(T0 - z0[fin0])) - 1e-6, -1e-6), max(float(np.max(T0 - z0[fin0])) + 1e-6, 1e-6)
    sv = {"T0": T0, "catch0": base, "range_points": [100 * fc_(lo_all), 100 * fc_(hi_all)], "arms": {}}
    cx = {}
    for dlt in deltas:
        tgt = dlt / 100.0
        if dlt == 0:
            c_ = 0.0
        elif dlt > 0:  # smallest c >= 0 with catch change >= delta (run_plant455's rule)
            lo, hi = 0.0, hi_all
            c_ = None if fc_(hi) < tgt - 1e-12 else hi
            while c_ is not None and hi - lo > 1e-12 * max(1.0, abs(hi)):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if fc_(mid) >= tgt - 1e-12 else (mid, hi)
                c_ = hi
        else:  # negative delta: the c <= 0 closest to 0 with catch change <= delta
            lo, hi = lo_all, 0.0
            c_ = None if fc_(lo) > tgt + 1e-12 else lo
            while c_ is not None and hi - lo > 1e-12 * max(1.0, abs(lo)):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if fc_(mid) <= tgt + 1e-12 else (lo, mid)
                c_ = lo
        if c_ is None:
            sv["arms"][f"{dlt:+d}"] = {"status": "unattainable"}
        else:
            sv["arms"][f"{dlt:+d}"] = {"status": "ok", "c": c_, "achieved_points": 100 * fc_(c_)}
            cx[dlt] = c_
    return sv, cx


def plant_p1_apply(sv, cx, zs0, T0, ht0, p1):
    """PLANT V3 P1 (AMEND4 D3, AMEND5 F2): in p1["order"] (seeded), the first delta whose arm is "ok" and whose c, moved by the smallest step that
    carries one more cohort run across the draw-0 tau_sim (in delta's sign, plus 1e-6), changes the draw-0 catch. c' and achieved_points' =
    100 x (ht0(z + c') - ht0(z)) (plant_fit's own fc_), so the runner's ported in-row checks stay active and hold. -> (solve', {delta: c'}, info)"""
    import numpy as np

    z0 = np.asarray(zs0, dtype=np.float64)
    fin = np.isfinite(z0)
    T0 = float(T0)
    if not np.isfinite(T0):
        return sv, cx, {"fired": False, "why": "tau not finite"}
    base = float(ht0(z0))
    for d in list(p1["order"])[:20]:
        a_ = (sv.get("arms") or {}).get(f"{d:+d}") or {}
        if a_.get("status") != "ok" or d not in cx:
            continue
        c = cx[d]
        if d > 0:
            un_ = z0[fin & ~(z0 + c > T0)]
            if not un_.size:
                continue
            c2 = float(T0 - un_.max()) + 1e-6
        else:
            ca_ = z0[fin & (z0 + c > T0)]
            if not ca_.size:
                continue
            c2 = float(T0 - ca_.min()) - 1e-6
        if float(ht0(z0 + c2)) == float(ht0(z0 + c)):
            continue
        sv2, cx2 = json.loads(json.dumps(sv)), dict(cx)
        cx2[d] = c2
        sv2["arms"][f"{d:+d}"] = {"status": "ok", "c": c2, "achieved_points": 100 * (float(ht0(z0 + c2)) - base)}
        return sv2, cx2, {"fired": True, "origin": p1["origin"], "level": p1["level"], "delta": int(d)}
    return sv, cx, {"fired": False, "why": "no delta in the seeded order moved the draw-0 catch"}


def plant_solve_cmp(key, sv, rsv, arm="sim"):
    """PLANT V3 block (ii): this check's draw-0 solve against the record's rows.<label>.<origin>.plant.solve_draw0.<key>.<arm> (sim; sim2: AMEND6), field
    by field."""
    pre = f"{key}|plant|solve_draw0|{arm}"
    if not isinstance(rsv, dict):
        return [(f"{pre}|present", True, False)]
    out = [(f"{pre}|fields", sorted(sv), sorted(rsv))]
    out += [(f"{pre}|{k_}", v_, rsv.get(k_)) for k_, v_ in sorted(sv.items()) if k_ != "arms"]
    if "arms" in sv:
        ra_ = rsv.get("arms") if isinstance(rsv.get("arms"), dict) else {}
        out.append((f"{pre}|arms|tags", sorted(sv["arms"]), sorted(ra_)))
        for t_, a_ in sorted(sv["arms"].items()):
            r0_ = ra_.get(t_) if isinstance(ra_.get(t_), dict) else {}
            out.append((f"{pre}|arms|{t_}|fields", sorted(a_), sorted(r0_)))
            out += [(f"{pre}|arms|{t_}|{k_}", v_, r0_.get(k_)) for k_, v_ in sorted(a_.items())]
    return out


def plant_p1_allowed(k3, p1hit, cb, ro):
    """P1's allowed MISMATCH set (AMEND4 D3; 2e / reviewer: built from the compared keys): k3 = a ledger key (section, entry, key). Allowed: that
    row's solve c / achieved_points at that delta; that row's planted catch at that delta (draw-0 point and every seeded draw); the combo's pooled
    planted catch and gap at that delta and level (every seeded draw); and coheadline_signed_plant S at (combo, level, readout, delta)."""
    o_, lv_, d_ = p1hit["origin"], p1hit["level"], f"{p1hit['delta']:+d}"
    if k3[0] == "agg":  # the aggregate ledger's keys are (section, key)
        return len(k3) == 2 and k3[1] == f"plant|{cb}|joint|{lv_}|headline_six|{ro}|{d_}|S"
    sec, ent, k_ = k3
    if sec == "row" and ent == o_:
        return k_ in (f"{lv_}|plant|solve_draw0|sim|arms|{d_}|c", f"{lv_}|plant|solve_draw0|sim|arms|{d_}|achieved_points",
                      f"{lv_}|plant|catch|sim_plant_{d_}") or bool(re.fullmatch(rf"boot\|joint\|{re.escape(lv_)}\|draw \d+\|plant\|catch\|sim_plant_{re.escape(d_)}", k_))
    if sec == "pooled" and ent == cb:
        return bool(re.fullmatch(rf"boot\|joint\|{re.escape(lv_)}\|draw \d+\|pooled\|plant\|(catch_sim_plant|gap_delta_full_plant)_{re.escape(d_)}", k_))
    return False


def plant_catch_keys_cmp(key, pcz, rcp, rcf):
    """PLANT V3 (Codex V3 r2 #3): the row's planted catch keys at one level: in mode joint exactly this check's sim_plant_<d> (every key holding
    "_plant_", so a sim2_plant_<d> or any other planted key MISMATCHES); in mode fixed none (the runner plants joint only). -> comparisons"""
    return [(f"{key}|plant|catch|planted keys (joint)", sorted(f"sim_plant_{d_:+d}" for d_ in pcz), sorted(k_ for k_ in rcp if "_plant_" in k_)),
            (f"{key}|plant|catch|planted keys (fixed)", [], sorted(k_ for k_ in rcf if "_plant_" in k_))]


def sim2_npz_absent(lab, cb, key, origs, deltas, files):
    """PLANT V3 (AMEND6 G2 (d)): no sim2 planted array in the npz at (label, combo, level): per origin catch_sim2_plant_<d>; pooled catch_sim2_plant_<d>
    and gap_delta_full_sim2_plant_<d>. -> True when none of these keys is among the npz's files"""
    k_ = [npzkey(lab, cb, "joint", key, "pooled", f"{p_}_{d_:+d}") for d_ in deltas for p_ in ("catch_sim2_plant", "gap_delta_full_sim2_plant")] \
        + [npzkey(lab, cb, "joint", key, aob(o), f"catch_sim2_plant_{d_:+d}") for o in origs for d_ in deltas]
    return not (set(k_) & set(files))


def plant_arms_cmp(key, il2, rsd):
    """PLANT V3 (AMEND6 G2 (c)): the arms of solve_draw0.<key>: exactly {sim, sim2} on an IL row, {sim} elsewhere. -> one comparison"""
    return (f"{key}|plant|solve_draw0|arms", ["sim", "sim2"] if il2 else ["sim"], sorted(rsd))


def il_cmp(ilr, ril):
    """PLANT V3 (AMEND6 G2 (a)): the row's il_arms report against this check's port, field by field (the field set first). -> comparisons"""
    if not isinstance(ril, dict):
        return [("plant|il_arms|present", ilr is not None, False)]
    if ilr is None:
        return [("plant|il_arms|present", False, True)]
    return [("plant|il_arms|present", True, True), ("plant|il_arms|fields", sorted(ilr), sorted(ril))] \
        + [(f"plant|il_arms|{k_}", v_, ril.get(k_)) for k_, v_ in sorted(ilr.items())]


def s1_labels_viol(built, checked, job_keys, s1_jobs, plant_jobs):
    """PLANT V3 (AMEND6 G5): S1's built vs checked labels by key: checked (PLANT's rows) a proper subset of built (A's meta.pooled_world.labels); 7
    captured job keys per checked label; S1's jobs = 7 x built; PLANT's = 7 x checked; the built-not-checked labels carry the rest. -> violations"""
    jpl = {lb: sum(1 for ol, _ in job_keys if ol == lb) for lb in checked}
    v = [] if set(checked) < set(built) else ["checked labels not a proper subset of the built ones"]
    v += [f"{lb}: {n} job keys (want 7)" for lb, n in sorted(jpl.items()) if n != 7]
    v += [] if s1_jobs == 7 * len(built) else [f"S1 jobs_checked {s1_jobs} != 7 x {len(built)} built labels"]
    v += [] if plant_jobs == 7 * len(checked) else [f"PLANT jobs_checked {plant_jobs} != 7 x {len(checked)} checked labels"]
    v += [] if s1_jobs - len(job_keys) == 7 * len(set(built) - set(checked)) else ["S1's jobs outside PLANT's labels != 7 x the built-not-checked labels"]
    return v


def plant_level_part_viol(is_plant, level_part):
    """PLANT V3 (Codex V3 r2 #4; AMEND7): the PLANT value run is unsplit (block (iv) and P1 read both plant levels); --level-part STOPs on it."""
    return ["PLANT V3 runs unsplit: --level-part is refused for the PLANT record"] if is_plant and level_part else []


def p1_cond_ok(delta, applied):
    """PLANT V3 (AMEND7 G1): the pre-stated aggregate categories of P1's delta: a negative delta is attainable in every member row (RULE L4944), so both
    the pooled planted gap and the coheadline S are compared in the normal run; a positive delta has an unattainable headline member, so the coheadline
    S is not (its pooled gap, per label, is recorded only). Any other pair (or delta 0) STOPs."""
    if delta < 0:
        return applied == {"pooled planted gap": True, "coheadline_signed_plant S": True}
    return delta > 0 and applied.get("coheadline_signed_plant S") is False


def plant_ctl_ok(ctl):
    """PLANT V3 (AMEND7 G3): on the PLANT record the G6 positive controls are [the co-headline control, applicable, with expected keys; the PANEL
    control, not applicable (no PANEL group is built)]; anything else STOPs (a vacuous pass of both is refused)."""
    return [c_.get("applicable") for c_ in ctl] == [True, False] and ctl[0].get("expected_keys", 0) > 0 and ctl[1].get("expected_keys") == 0


def local_modules_viol(mods, env_prefixes, allowed_roots):
    """PLANT V3 (2e, 14:3x; AMEND9): every imported module whose file is outside the interpreter's own prefixes (stdlib and site-packages) must sit
    under an allowed root: the V3 worktree, or the record's code-commit tree. An editable install resolving to another checkout (e.g. the main
    repo's src/) is a violation. mods: name -> resolved file path. -> the violations, "name path" (keys and paths only)"""
    return sorted(f"{n_} {f_}" for n_, f_ in mods.items() if not any(f_.startswith(p_.rstrip("/") + "/") for p_ in env_prefixes)
                  and not any(f_.startswith(r_.rstrip("/") + "/") for r_ in allowed_roots))


PLANT_FANOUT = 15  # AMEND11: the PLANT run's check inputs (12 labels + 2 value plants + 1 P1); the hard bound (AMEND9 section 4) assumes it


def plant_fanout_viol(is_plant, n_inputs):
    """PLANT V3 (AMEND11): the check fan-out the hard bound assumes; any other count STOPs the PLANT launch (other records: no constraint)."""
    return [f"PLANT V3: {n_inputs} check inputs, the bound assumes {PLANT_FANOUT} (STOP)"] if is_plant and n_inputs != PLANT_FANOUT else []


def w2_value_path_ok(w2k, val):
    """rc1b W2 (AMEND7): every W2 gate sits on the value path (main's `if VAL_ and W2K_:` blocks), so a W2 record must be on it: not W2K_ or VAL_.
    VAL_ is false without --fold-labels. -> True when the run may go on (a non-W2 record, or a W2 record on the value path)."""
    return not w2k or bool(val)


def w2_value_plant(v):
    """rc1b W2 (AMEND3): a record-side value plant, v -> v + 1e-6 x max(1, |v|): above eq()'s 1e-9 x max(1, |v|) tolerance for every finite v of
    either sign (the picks are finite stored values only, AMEND2 section 1)."""
    assert isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), ("rc1b W2: a value plant on a non-finite value", v)
    return float(v) + 1e-6 * max(1.0, abs(float(v)))


W2_PIPE = (("run1", "B"), ("run2", "B"), ("sim", "S"))  # runner @ 7797f2cc2 L2693: run 1 and run 2 on the real line tau_B (H_B); sim on tau_sim (H_sim)


def w2_pertask_entries(orig, coh, main_of, zr, T, hz, zero):
    """rc1b W2a: the runner's per-task entries (run_peraction311 @ 7797f2cc2 L2693-2711), ported at draw 0. zr: arm -> the row's draw-0 attack-run
    z under its pipeline pool's moments (cohort order); T: pool -> the draw-0 tau at the per-task level; hz: pool -> the pool's draw-0 unit z (every
    unit weight 1 at draw 0, so the honest weight is a unit count); zero: the row's structural-zero run keys. -> {(cell, arm): entry}"""
    import numpy as np

    out = {}
    for w, nm in W2_PIPE:
        za, T_ = np.asarray(zr[w], dtype=np.float64), float(T[nm])
        h_ = np.asarray(hz[nm], dtype=np.float64)
        hk_ = h_[np.isfinite(h_)]
        tw_ = float(len(hk_))
        for i, c in enumerate(coh):
            e_ = {"origin": orig, "cell": c, "main": main_of[c], "arm": w, "pipeline_pool": nm}
            if f"{w}|{c}" in zero:  # a structural zero: no continuous score, not caught (the zero-tool rule)
                out[(c, w)] = e_ | {"max_minus_tau": None, "honest_percentile": None, "caught": False, "structural_zero": True}
                continue
            v_ = float(za[i])
            out[(c, w)] = e_ | {"max_minus_tau": (v_ - T_) if np.isfinite(v_) else None,
                                "honest_percentile": float((hk_ <= v_).sum() / tw_) if np.isfinite(v_) else None,
                                "caught": bool(v_ > T_) if np.isfinite(v_) else None}
    return out


def w2_pt_main_map(coh, main_of, canon, ao):
    """rc1b W2a (AMEND8): the `main` each per-task entry carries. On an all-origins row the runner writes the canonical main (run_peraction311 @
    7797f2cc2 L1785 `main_of = moa = MG["main_of_cell"]`; L2706 `"main": main_of[c]`), as the checker's G7b side already does (mine_mainof,
    Codex r7): inp["canon"], the job's merge.main_of_cell. Any other row keeps its per-origin map. -> {cell: main}; a cell missing from the
    map STOPs (KeyError)."""
    assert not ao or isinstance(canon, dict), "rc1b W2a (AMEND8): an all-origins row without the canonical main map (STOP)"
    return {c: canon[c] for c in coh} if ao else {c: main_of[c] for c in coh}


W2_PT_VALUES =("max_minus_tau", "caught", "honest_percentile")  # the compared values (honest_percentile: secondary, R2); structural_zero where present


def w2_pertask_cmp(mine, rec_list):
    """rc1b W2a: this check's per-task entries against the record row's per_task list, by (cell, arm). The entry sets two-way; each shared entry's
    identity fields (origin, main, pipeline_pool and the field names) as one comparison; its 3 values each; structural_zero where either side has it.
    -> (comparisons, the number of value comparisons the count gate counts)"""
    rd_ = {}
    for e_ in rec_list:
        k_ = (e_.get("cell"), e_.get("arm"))
        assert k_ not in rd_, ("rc1b W2a: two record entries for one (cell, arm)", k_)
        rd_[k_] = e_
    out = [("per_task|entries (two-way)", sorted(map(list, mine)), sorted(map(list, rd_)))]
    nv_ = 0
    for k_ in sorted(set(mine) & set(rd_)):
        m_, r_ = mine[k_], rd_[k_]
        p_ = f"per_task|{k_[1]}|{k_[0]}"
        out.append((f"{p_}|identity", [m_["origin"], m_["main"], m_["pipeline_pool"], sorted(m_)], [r_.get("origin"), r_.get("main"), r_.get("pipeline_pool"), sorted(r_)]))
        out += [(f"{p_}|{f_}", m_[f_], r_.get(f_)) for f_ in W2_PT_VALUES]
        nv_ += len(W2_PT_VALUES)
        if "structural_zero" in m_ or "structural_zero" in r_:
            out.append((f"{p_}|structural_zero", m_.get("structural_zero"), r_.get("structural_zero")))
            nv_ += 1
    return out, nv_


def w2_score_plant(zr, coh, zero, spec, zplus, rfold, five):
    """rc1b W2a control (AMEND1 section 4 (a)): ONE seeded entry of this row (finite, not a structural zero, a mapped run on a fold-model probe) has
    its run's max attack unit moved up by +1.0 in the probe's raw score units (zplus(w, i) recomputes that run's z through the runner's zatt); every
    other entry and the row's catches are untouched. -> (zr with that one z replaced, {"arm", "cell", "moved"}); an empty candidate set STOPs."""
    import numpy as np

    cand_ = [(w, i) for w, _ in W2_PIPE for i, c in enumerate(coh)
             if f"{w}|{c}" not in zero and np.isfinite(zr[w][i]) and (not five or int(rfold[w][i]) >= 0)]
    assert cand_, "rc1b W2a control: no finite, non-structural-zero mapped entry to plant (STOP)"
    w, i = cand_[int(np.random.default_rng(int(spec["seed"])).integers(len(cand_)))]
    z2_ = {k_: np.array(v_, dtype=np.float64, copy=True) for k_, v_ in zr.items()}
    z2_[w][i] = zplus(w, i)
    return z2_, {"arm": w, "cell": coh[i], "moved": bool(z2_[w][i] != zr[w][i])}


def w2_record_plants(rec_list, spec):
    """rc1b W2a control (AMEND1 section 4; AMEND2; AMEND3), on a COPY of one row's per_task list: max_minus_tau and honest_percentile of one entry
    each (finite stored values) -> v + 1e-6 x max(1, |v|); caught of one entry (a boolean) flipped; structural_zero of one entry (where present)
    flipped; one non-structural-zero entry dropped. Each on its own entry, seeded. The added entry is NOT here: it is checked on its own copy
    (w2_added_check; AMEND6, Codex W2 r1 #4). -> (the copy, the planted (cell, arm, field) list and the dropped key); an empty candidate set STOPs."""
    import copy

    import numpy as np

    rl_ = copy.deepcopy(list(rec_list))
    rng_ = np.random.default_rng(int(spec["seed"]))
    used_, info_ = set(), {"planted": []}

    def pick(pred, what):
        c_ = [j_ for j_, e_ in enumerate(rl_) if j_ not in used_ and pred(e_)]
        assert c_, (f"rc1b W2a control: no candidate entry for {what} (STOP)")
        j_ = c_[int(rng_.integers(len(c_)))]
        used_.add(j_)
        return j_

    fin_ = lambda x_: isinstance(x_, (int, float)) and not isinstance(x_, bool) and math.isfinite(x_)  # noqa: E731
    for f_ in ("max_minus_tau", "honest_percentile"):
        j_ = pick(lambda e_, f_=f_: fin_(e_.get(f_)), f_)
        rl_[j_][f_] = w2_value_plant(rl_[j_][f_])
        info_["planted"].append([rl_[j_]["cell"], rl_[j_]["arm"], f_])
    j_ = pick(lambda e_: isinstance(e_.get("caught"), bool) and not e_.get("structural_zero"), "caught")
    rl_[j_]["caught"] = not rl_[j_]["caught"]
    info_["planted"].append([rl_[j_]["cell"], rl_[j_]["arm"], "caught"])
    if any(isinstance(e_.get("structural_zero"), bool) for e_ in rl_):  # (a row without a structural zero records that the plant is not applicable)
        j_ = pick(lambda e_: isinstance(e_.get("structural_zero"), bool), "structural_zero")
        rl_[j_]["structural_zero"] = not rl_[j_]["structural_zero"]
        info_["planted"].append([rl_[j_]["cell"], rl_[j_]["arm"], "structural_zero"])
    j_ = pick(lambda e_: not e_.get("structural_zero") and fin_(e_.get("max_minus_tau")), "drop")
    info_["dropped"] = [rl_[j_]["cell"], rl_[j_]["arm"]]
    rl_ = [e_ for k_, e_ in enumerate(rl_) if k_ != j_]
    return rl_, info_


W2_ADDED_CELL = "w2_control_cell_not_in_cohort"


def w2_added_check(mine, rec_list):
    """rc1b W2a control (AMEND2 section 2; AMEND6, Codex W2 r1 #4): the added entry on its OWN copy of the row's per_task list (no other plant),
    compared in the control call against the same row's unplanted comparison. Pass iff exactly the row's two-way entry set flags, no comparison key
    is missing or extra, and the value count is unchanged (so the label's count gate would pass). -> keys, booleans and counts only"""
    ex_ = dict(rec_list[0]) | {"cell": W2_ADDED_CELL, "arm": "run1"}
    c0_, n0_ = w2_pertask_cmp(mine, list(rec_list))
    c1_, n1_ = w2_pertask_cmp(mine, list(rec_list) + [ex_])
    v0_ = {k_: eq(a_, b_) for k_, a_, b_ in c0_}
    v1_ = {k_: eq(a_, b_) for k_, a_, b_ in c1_}
    flg_ = sorted(k_ for k_ in set(v0_) & set(v1_) if v0_[k_] and not v1_[k_])
    ok_ = flg_ == ["per_task|entries (two-way)"] and set(v0_) == set(v1_) and n1_ == n0_ and not any(not v0_[k_] and v1_[k_] for k_ in set(v0_) & set(v1_))
    return {"pass": bool(ok_), "added": [ex_["cell"], ex_["arm"]], "flagged": flg_, "keys_equal": set(v0_) == set(v1_), "count_unchanged": n1_ == n0_}


def w2_control_verdict_pertask(n_, p_, ctl, spec, n_count, p_count):
    """rc1b W2a control (AMEND1 section 4 (a); AMEND2 section 2): the normal and control calls' ledgers of the control label, (section, row, key) ->
    (mine, reference, verdict). Pass iff: the score plant's max_minus_tau flags (its caught / honest_percentile may), on its row; the record plants
    flag exactly their keys plus the row's two-way entry set; the dropped entry's 4 keys (identity + 3 values) are missing from the control call and
    nothing is extra; the pin fixture flags its row's pin; nothing else flags; the count drops by exactly 3; the added entry's own-copy check
    (w2_added_check, AMEND6) passed. -> keys and booleans only"""
    o1_, o2_ = spec["plant_score"]["origin"], spec["plant_record"]["origin"]
    sc_, rp_ = ctl.get("score") or {}, ctl.get("record") or {}
    p1_ = f"per_task|{sc_.get('arm')}|{sc_.get('cell')}"
    req_ = {("row", o1_, f"{p1_}|max_minus_tau")}
    opt_ = {("row", o1_, f"{p1_}|caught"), ("row", o1_, f"{p1_}|honest_percentile")}
    req_ |= {("row", o2_, f"per_task|{a_}|{c_}|{f_}") for c_, a_, f_ in rp_.get("planted") or []}
    req_ |= {("row", o2_, "per_task|entries (two-way)"), ("row", o2_, "per_task|pw_calib pin = A's D1 file")}
    dc_, da_ = (rp_.get("dropped") or [None, None])
    miss_req_ = {("row", o2_, f"per_task|{da_}|{dc_}|{f_}") for f_ in ("identity",) + W2_PT_VALUES}
    flg_ = {k_ for k_ in set(n_) & set(p_) if n_[k_][2] == "MATCH" and p_[k_][2] == "MISMATCH"}
    heal_ = {k_ for k_ in set(n_) & set(p_) if n_[k_][2] == "MISMATCH" and p_[k_][2] == "MATCH"}
    miss_, extra_ = set(n_) - set(p_), set(p_) - set(n_)
    sk_ = lambda s_: sorted(" | ".join(map(str, k_)) for k_ in s_)[:20]  # noqa: E731
    ok_ = (bool(sc_.get("moved")) and req_ <= flg_ and flg_ <= req_ | opt_ and miss_ == miss_req_ and not extra_ and not heal_
           and isinstance(n_count, int) and p_count == n_count - 3 and len(rp_.get("planted") or []) >= 3 and bool(ctl.get("pin"))
           and (rp_.get("added") or {}).get("pass") is True)
    return {"pass": bool(ok_), "added_check": rp_.get("added"), "required_flagged": len(req_), "flagged": len(flg_), "missing_required": sk_(req_ - flg_),
            "flagged_outside_allowed": sk_(flg_ - req_ - opt_), "missing_keys_as_predicted": miss_ == miss_req_, "extra_keys": sk_(extra_),
            "mismatch_to_match": sk_(heal_), "count_drop_3": p_count == (n_count - 3 if isinstance(n_count, int) else None),
            "planted_fields": [f_ for _, _, f_ in rp_.get("planted") or []], "score_plant_moved": bool(sc_.get("moved"))}


def job_vtag(cf):
    """A captured job's row name (the runner's vtag), exactly as main's by_origin loop builds it (same expression; rc1b W2b maps row-store files)."""
    return (cf.get("origin", "S_new") + (" without S4" if cf.get("no_s4") else "") + (" (all-units deviation)" if cf.get("allunits_dev") else "")
            + (" (registered cohort)" if cf.get("cohort") == "registered" else "")
            + (f" [{cf['sp_block']}]" if cf.get("origin") == "S_paper493" and cf.get("sp_block") else "") + (" (all-origins draws)" if cf.get("merge") else "")
            + (f" [pooled: {cf['pw_combo']}]" if cf.get("pw_combo") else ""))


W2_KAPPA_KEYS = ("D", "run2|po", "run2|p1", "run2|q1", "sim|po", "sim|p1", "sim|q1")  # runner @ 7797f2cc2 L2889-2895: the per-draw 2x2 margins


def w2_tpr(h, wrow, ccol, wcell):
    """The runner's tpr (L2273-2276) at one draw on an all-origins row: sum_c W[ccol_c] wcell_c h_c / sum_c W[ccol_c] wcell_c (0 / 0 = NaN)."""
    import numpy as np

    Wc = np.asarray(wrow, dtype=np.float64)[np.asarray(ccol, dtype=int)] * np.asarray(wcell, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((Wc * np.asarray(h, dtype=np.float64)).sum() / Wc.sum())


def w2_margins(z1, z2, zs, TB, TS, wrow, ccol, wcell, flip=None):
    """rc1b W2b: the runner's per-draw kappa margins at one draw (L2886-2895, ported): hx = run 1 caught at tau_B; for run 2 (tau_B) and sim (tau_sim):
    po = tpr(hx == hy), p1 = tpr(hx), q1 = tpr(hy), NaN where either threshold is NaN; D = sum_c W wcell. flip: a cohort index whose hx is flipped
    (the control's own recompute only). -> {key: value}"""
    import numpy as np

    hx = np.asarray(z1, dtype=np.float64) > TB
    if flip is not None:
        hx = hx.copy()
        hx[flip] = not hx[flip]
    out = {"D": float((np.asarray(wrow, dtype=np.float64)[np.asarray(ccol, dtype=int)] * np.asarray(wcell, dtype=np.float64)).sum())}
    for nk_, hy, To in (("run2", np.asarray(z2, dtype=np.float64) > TB, TB), ("sim", np.asarray(zs, dtype=np.float64) > TS, TS)):
        bad_ = bool(np.isnan(TB) or np.isnan(To))
        for k2_, h_ in (("po", hx == hy), ("p1", hx), ("q1", hy)):
            out[f"{nk_}|{k2_}"] = float("nan") if bad_ else w2_tpr(h_, wrow, ccol, wcell)
    return out


def w2_load_stored_row(paths, read):
    """The runner's stored_row rule (L1353-1366): of the row's files <rel>_<sha16>_<attempt8>.pkl, the first by name whose bytes' sha256 starts with
    its sha16. AMEND5: every file is read, to count the complete attempts. -> (the unpickled row, its path, its sha256, the number of complete
    attempts); no complete file STOPs."""
    import pickle

    got_, n_ = None, 0
    for p_ in sorted(paths):
        nm_ = p_.rsplit("/", 1)[-1]
        h16_, _, att_ = nm_[5:-4].partition("_")
        b_ = read(p_)
        h_ = hashlib.sha256(b_).hexdigest()
        if h16_ == h_[:16] and len(att_) == 8:
            n_ += 1
            got_ = got_ or (pickle.loads(b_), p_, h_)
    if got_ is None:
        raise AssertionError(("rc1b W2b: no complete stored row among its files (STOP)", list(paths)))
    return got_ + (n_,)


def w2_stored_tie(obj, rec_row):
    """rc1b W2b (AMEND5 section 1): the chosen stored row must be the attempt behind the record's row. Its resources (runner L3128-3140: written once
    per attempt, with the attempt's container start time) and its draws_sha256 must equal the record row's, each compared in the form the runner
    wrote the record (json.dumps default=float, L5159). A difference, or either field missing on either side, STOPs. The message names the field,
    never a value."""
    can_ = lambda x_: json.dumps(json.loads(json.dumps(x_, default=float)), sort_keys=True)  # noqa: E731
    for k_ in ("resources", "draws_sha256"):
        assert k_ in obj and k_ in rec_row, ("rc1b W2b: the stored row or the record row lacks a tie field (STOP)", k_)
        assert can_(obj[k_]) == can_(rec_row[k_]), ("rc1b W2b: the chosen stored row is not the record's attempt (STOP)", k_)


def w2_store_plants(K, ids, spec):
    """rc1b W2b control (AMEND1 section 4 (b); AMEND2-3), on an in-memory COPY of one stored row's margins: each of the 7 arrays in each mode is
    planted at one seeded sampled draw where its stored value is finite, v -> v + 1e-6 x max(1, |v|) (14 plants). -> (the copy, {"planted":
    [[mode, draw, key], ...]}); an array with no finite sampled value STOPs."""
    import numpy as np

    rng_ = np.random.default_rng(int(spec["seed"]))
    K2_ = {md_: {k_: np.array(v_, dtype=np.float64, copy=True) for k_, v_ in d_.items()} for md_, d_ in K.items()}
    pl_ = []
    for md_ in ("joint", "fixed"):
        for k_ in W2_KAPPA_KEYS:
            c_ = [b_ for b_ in ids if np.isfinite(K2_[md_][k_][b_])]
            assert c_, ("rc1b W2b control: no finite sampled value to plant (STOP)", md_, k_)
            b_ = c_[int(rng_.integers(len(c_)))]
            K2_[md_][k_][b_] = w2_value_plant(float(K2_[md_][k_][b_]))
            pl_.append([md_, int(b_), k_])
    return K2_, {"planted": pl_}


def w2_flip_pick(bp, BW, ccol, wcell, n_cells, spec):
    """rc1b W2b control (AMEND1 section 4 (b); AMEND2): a seeded (mode, sampled draw, cohort cell) where the cell has positive weight at that draw and
    both thresholds (tau_B, tau_sim at 0.01) are finite, so that flipping its run-1 hit moves po and p1 (runner L2889-2895). -> {mode, draw, cell,
    eligible}; an empty candidate set records eligible False (the control then fails: never vacuous)."""
    import numpy as np

    c_ = []
    for md_ in ("joint", "fixed"):
        for i_, b_ in enumerate(BW["ids"]):
            T_ = bp[(md_, b_)]["T"]["0.01"]
            if not (np.isfinite(T_["B"]) and np.isfinite(T_["S"])):
                continue
            w_ = np.asarray(BW["W"][i_], dtype=np.float64)[np.asarray(ccol, dtype=int)] * np.asarray(wcell, dtype=np.float64)
            c_ += [(md_, int(b_), int(j_)) for j_ in range(n_cells) if w_[j_] > 0]
    if not c_:
        return {"mode": None, "draw": None, "cell": None, "eligible": False}
    md_, b_, j_ = c_[int(np.random.default_rng(int(spec["seed"])).integers(len(c_)))]
    return {"mode": md_, "draw": b_, "cell": j_, "eligible": True}


def w2_control_verdict_kappa(n_, p_, ctl, spec, n_count, p_count):
    """rc1b W2b control (AMEND1 section 4 (b); AMEND2-3): the normal and control calls' ledgers of the control label. Pass iff: the flip moves exactly
    its row's run2|po, run2|p1, sim|po and sim|p1 at its draw and mode (D and both q1 stay MATCH); each of the 14 store plants flags exactly its
    (array, mode, draw); the dropped stored row's 294 comparisons are missing and nothing is extra; nothing else flags; the count drops by 294."""
    o1_, o2_ = spec["plant_margin"]["origin"], spec["plant_store"]["origin"]
    fl_, st_ = ctl.get("flip") or {}, ctl.get("store") or {}
    req_ = {("row", o1_, f"kappa|{fl_.get('mode')}|draw {fl_.get('draw')}|{k_}") for k_ in ("run2|po", "run2|p1", "sim|po", "sim|p1")}
    req_ |= {("row", o2_, f"kappa|{m_}|draw {d_}|{k_}") for m_, d_, k_ in st_.get("planted") or []}
    dr_ = spec["plant_store"]["drop_origin"]  # the dropped stored row (a third row of the control label)
    req_ |= {("row", dr_, "kappa|stored rows (two-way)")}
    miss_req_ = {k_ for k_ in n_ if k_[0] == "row" and k_[1] == dr_ and k_[2].startswith("kappa|") and k_[2] != "kappa|stored rows (two-way)"}
    flg_ = {k_ for k_ in set(n_) & set(p_) if n_[k_][2] == "MATCH" and p_[k_][2] == "MISMATCH"}
    heal_ = {k_ for k_ in set(n_) & set(p_) if n_[k_][2] == "MISMATCH" and p_[k_][2] == "MATCH"}
    miss_, extra_ = set(n_) - set(p_), set(p_) - set(n_)
    sk_ = lambda s_: sorted(" | ".join(map(str, k_)) for k_ in s_)[:20]  # noqa: E731
    ok_ = (bool(fl_.get("eligible")) and req_ <= flg_ and flg_ <= req_ and miss_ == miss_req_ and len(miss_req_) == 294 and not extra_ and not heal_
           and len(st_.get("planted") or []) == 14 and isinstance(n_count, int) and p_count == n_count - 294)
    return {"pass": bool(ok_), "required_flagged": len(req_), "flagged": len(flg_), "missing_required": sk_(req_ - flg_),
            "flagged_outside_allowed": sk_(flg_ - req_), "missing_keys_as_predicted": miss_ == miss_req_, "n_missing": len(miss_), "extra_keys": sk_(extra_),
            "mismatch_to_match": sk_(heal_), "count_drop_294": p_count == (n_count - 294 if isinstance(n_count, int) else None)}


def p1_need(n1, o1, lv1, d1, cb1, ro1):
    """PLANT V3 P1's required categories (AMEND5 F2, AMEND6 G1): the draw-0 row categories always; the pooled planted gap and the coheadline S only
    where the NORMAL run's ledger n1 compares such a key at P1's combo, level and delta (Codex V3 r2 #2: the chosen level only). -> (need, applied)"""
    need = {"solve c at the delta": lambda k_: k_[:2] == ("row", o1) and k_[2] == f"{lv1}|plant|solve_draw0|sim|arms|{d1}|c",
            "per-origin planted catch, draw 0": lambda k_: k_[:2] == ("row", o1) and k_[2] == f"{lv1}|plant|catch|sim_plant_{d1}"}

    def pg_(k_):
        return k_[:2] == ("pooled", cb1) and k_[2].startswith(f"boot|joint|{lv1}|draw ") and k_[2].endswith(f"|pooled|plant|gap_delta_full_plant_{d1}")
    cs_ = ("agg", f"plant|{cb1}|joint|{lv1}|headline_six|{ro1}|{d1}|S")
    applied = {"pooled planted gap": any(pg_(k_) for k_ in n1), "coheadline_signed_plant S": cs_ in n1}
    if applied["pooled planted gap"]:
        need["pooled planted gap"] = pg_
    if applied["coheadline_signed_plant S"]:
        need["coheadline_signed_plant S"] = lambda k_: k_ == cs_
    return need, applied


PLANT_COH_UNAV = {"unavailable": "unattainable or missing in a member row"}  # runner L3524 @ 752413181: a delta some member row cannot reach


def plant_coh_cmp(rec, bvals, vals, labels, points, combos):
    """PLANT V3 block (iv): coheadline_signed_plant.<combo>.joint.<f>.headline_six|<ro>.<delta>.{S, Y, rows, readable} at draw 0, from this run's
    pooled planted Delta_full and yardstick per label (the runner's coh_pl, L3513-3527: S = the six-label mean of the planted Delta_full, Y = the
    six-label mean of run2 - run1 at the pooled tau_B, rows = the six, readable = every one of the six labels' pooled readable flag (sy_rows[..][2]
    = this run's aggin "rd"); a delta that any member row cannot reach reads exactly PLANT_COH_UNAV). The domain is built here first (Codex V3 #5):
    the run's combos, mode joint, the plant points, both readouts, and per group the deltas some member row reaches (bvals "pl_any"); the record's
    trees must hold exactly these keys. -> [(key, mine, record)]"""
    out, cp_ = [], rec.get("coheadline_signed_plant") or {}
    lvk = sorted(f"{x:g}" for x in points)
    out.append(("plant|combos", sorted(combos), sorted(k_ for k_ in cp_ if k_ != "_rule")))
    out.append(("plant|_rule is text", True, isinstance(cp_.get("_rule"), str)))
    for cb in sorted(combos):
        cq_ = cp_.get(cb) if isinstance(cp_.get(cb), dict) else {}
        out.append((f"plant|{cb}|modes", ["joint"], sorted(cq_)))
        jq_ = cq_.get("joint") if isinstance(cq_.get("joint"), dict) else {}
        out.append((f"plant|{cb}|joint|levels", lvk, sorted(jq_)))
        for key in lvk:
            lq_ = jq_.get(key) if isinstance(jq_.get(key), dict) else {}
            out.append((f"plant|{cb}|joint|{key}|groups", ["headline_six|C", "headline_six|R"], sorted(lq_)))
            for ro in ("C", "R"):
                want = sorted(lb for lb in labels if lb.endswith(f" {ro}"))
                assert len(want) == 6, ("PLANT V3: the headline six per readout", ro, len(want))
                pre = f"plant|{cb}|joint|{key}|headline_six|{ro}"
                q = lq_.get(f"headline_six|{ro}") if isinstance(lq_.get(f"headline_six|{ro}"), dict) else {}
                b0_ = {lb: ((((bvals.get(lb) or {}).get(cb) or {}).get("joint") or {}).get(key) or {}).get("0") or {} for lb in want}
                tags = sorted({t_ for lb in want for t_ in (b0_[lb].get("pl_any") or [])}, key=int)
                out.append((f"{pre}|delta keys", sorted(tags), sorted(q)))  # every key of the group (a sim2 or any other key is a mismatch)
                for t_ in tags:
                    qt_ = q.get(t_) if isinstance(q.get(t_), dict) else {}
                    if all(t_ in (b0_[lb].get("pl") or {}) for lb in want):
                        S_ = float(sum(b0_[lb]["pl"][t_][0] for lb in want) / len(want))
                        Y_ = float(sum(b0_[lb]["yr"] for lb in want) / len(want))
                        rd_ = all(bool((((vals.get(lb) or {}).get(cb) or {}).get(key) or {}).get("rd")) for lb in want)
                        out += [(f"{pre}|{t_}|fields", ["S", "Y", "readable", "rows"], sorted(qt_)),
                                (f"{pre}|{t_}|S", S_, (qt_.get("S") or {}).get("point")), (f"{pre}|{t_}|Y", Y_, (qt_.get("Y") or {}).get("point")),
                                (f"{pre}|{t_}|rows", want, sorted(qt_.get("rows") or [])), (f"{pre}|{t_}|readable", rd_, qt_.get("readable"))]
                    else:
                        out.append((f"{pre}|{t_}|unavailable", PLANT_COH_UNAV, qt_))
    return out


def il_block_viol(job_blk, pinned):
    """PLANT V3 (AMEND6): a job's IL block against the pinned block: every field, origin and folder equal except each folder's files, which may only
    lose keys or keep them (full_override.apply_full @ 752413181 L310-325 drops a 32k file without an uncut entry, or swaps in the uncut one;
    il_report_v3 STOPs on a kept one). -> the violations (keys only; [] = the pinned block but for files)"""
    v = []
    if {k: x for k, x in job_blk.items() if k != "sets"} != {k: x for k, x in pinned.items() if k != "sets"}:
        v.append("top-level fields")
    if sorted(job_blk.get("sets") or {}) != sorted(pinned["sets"]):
        v.append("origins")
    for org, fs in sorted(pinned["sets"].items()):
        jf = (job_blk.get("sets") or {}).get(org) or {}
        if sorted(jf) != sorted(fs):
            v.append(f"{org}: folders")
            continue
        for f, d in sorted(fs.items()):
            if {k: x for k, x in jf[f].items() if k != "files"} != {k: x for k, x in d.items() if k != "files"}:
                v.append(f"{org} | {f}: fields")
            if not set(jf[f].get("files") or {}) <= set(d.get("files") or {}):
                v.append(f"{org} | {f}: files outside the pinned block's")
    return v


def il_report_v3(block, origin, probe_file):
    """PLANT V3 (AMEND6; Codex V3 #3): il_arms.py @ 752413181 L341-379 (il_arm_rows), ported for the runner's call (L1530: cohort_cells None) and the
    case this check covers: no folder of the origin holds the probe's IL file (a pending family: no rows read). A folder that holds it STOPs (its rows
    would be read and scored: not ported). -> the row's il_arms report as the runner writes it, with cells_with_run and fold_minus1_runs (L2205-2206:
    0 for both IL arms when no row, fold -1 run or structural zero exists)"""
    rep = {"folders_read": 0, "pending_keys": [], "keys_without_rows": [], "structural_zero_keys": [], "rows": 0, "block_window": block["window"]}
    for folder, d in sorted(block["sets"].get(origin, {}).items()):
        want = {k for k in d["keys"] if k not in set(d.get("optional_pending_keys", []))}
        rep.setdefault("optional_pending_keys", []).extend(sorted(set(d.get("optional_pending_keys", []))))
        if not want:
            continue
        assert d["files"].get(probe_file) is None, (origin, folder, "STOP: an IL extra-arm folder holds this probe's file (its rows are not ported in V3)")
        rep["pending_keys"] += sorted(want)
    adm_ = sum(len(d["keys"]) for d in block["sets"].get(origin, {}).values())
    opt_ = len(rep.get("optional_pending_keys", []))
    rep["counts"] = {"admitted": adm_, "routed": adm_ - opt_, "scored": 0 + len(rep["structural_zero_keys"]), "rule": IL_COUNTS_RULE}
    rep["counts"]["in_scope_not_built"] = rep["counts"]["admitted"] - rep["counts"]["scored"]
    return rep | {"cells_with_run": {"sim2": 0, "run3": 0}, "fold_minus1_runs": {"sim2": 0, "run3": 0}}


def f1cols_nos4(cell, world, fold, isN, cohort_full, use21b, main_of, mains_sorted):
    """F1COLS on a without-S4 row (l.2149-2155): the thresholds are the with-S4 row's, so each arm's columns are the mains of its fold -1 attack
    cells in the WITH-S4 cohort (cohN_full), from the row's attack units, and only under 21b (else none)."""
    import numpy as np

    col_ = {m: i for i, m in enumerate(mains_sorted)}
    f1 = {w: ({col_.get(main_of[c], len(mains_sorted)) for c in set(cell[isN & (world == w) & np.isin(cell, list(cohort_full)) & (fold == -1)].tolist())}
              if use21b else set()) for w in A3}
    return {"B": f1["run1"] | f1["run2"] | f1["sim"], "A": f1["run1"] | f1["run2"], "S": set(f1["sim"])}


def pools_file_diff_five(file_arrays, mine, pat):
    """A fold-model label's pooled pools file vs the union rebuilt here, one role: per run key, the units (scores, folds, model scores) as canonical
    bytes (NaN-safe, unit_bytes) and the membership pattern. file_arrays: (run, s, f, sbm, org); mine: (keys, s, f, sbm); pat: {run: pattern}.
    -> (keys only in the file, keys only here, runs whose units or pattern differ)"""
    import numpy as np

    def per_run(k_, arrs):
        g_ = {}
        for i, r in enumerate(k_.tolist()):
            g_.setdefault(r, []).append(i)
        return {r: unit_bytes(tuple(a_[np.array(ix)] for a_ in arrs)) for r, ix in g_.items()}, g_
    th_, gt_ = per_run(file_arrays[0].astype(str), file_arrays[1:4])
    mi_, _ = per_run(mine[0], mine[1:4])
    org_ = file_arrays[4].astype(str)
    nd_ = sum(1 for r in set(th_) & set(mi_) if th_[r] != mi_[r] or {str(org_[i]) for i in gt_[r]} != {pat[r]})
    return len(set(th_) - set(mi_)), len(set(mi_) - set(th_)), nd_


def plant_alias(exports, role="B", exclude=frozenset()):
    """The PLANTED control (2e's spec (c)): a copy of the exports with the first run key (sorted) of `role`, outside `exclude` (the real union's
    identical-content keys, so the planted group is exactly a pair), in the first export that has one, repeated under a new key
    'plantedalias|<its cell>' (identical units). -> (exports copy, planted key, original key) or None"""
    import numpy as np

    for j_, (org, P_) in enumerate(exports):
        k, s, f, b = P_[role]
        ks_ = sorted(set(k.tolist()) - set(exclude))
        if ks_:
            r0 = ks_[0]
            m_ = k == r0
            pk_ = "plantedalias|" + r0.split("|", 1)[1]
            P2_ = dict(P_) | {role: (np.concatenate([k, np.full(int(m_.sum()), pk_)]), np.concatenate([s, s[m_]]), np.concatenate([f, f[m_]]),
                                     np.concatenate([b, b[m_]]))}
            return exports[:j_] + [(org, P2_)] + exports[j_ + 1:], pk_, r0
    return None


def beside_check(rec, psr, combo_rows=None):
    """G7 (RULE 43): the pooled record's per_set_beside must be an exact copy of the per-set record's blocks: each pooled combo's per-label pooled
    block for every label the pooled record carries, its coheadline_signed, and each pooled combo's panel_groups (canonical JSON, so NaN equals NaN).
    -> (comparisons, mismatch keys)"""
    cj = lambda x: json.dumps(x, sort_keys=True)  # noqa: E731
    pw_, psb = rec["meta"]["pooled_world"], rec.get("per_set_beside") or {}
    n, bad = 0, []
    # Astra g7 #2: the companion's compatibility, as the runner's finish_rows @ 29717a77b requires (keys, shas and structure only, no value read)
    mt_, pm_ = rec["meta"], psr.get("meta") or {}
    labs_ = sorted(set(pw_.get("labels") or []) | set(rec["rows"]))  # Codex r47 #2: the expected label domain (meta.pooled_world.labels) and the record's
    for k_, a_, b_ in (("inputs_final", (pm_.get("inputs_final") or {}).get("sha256"), (mt_.get("inputs_final") or {}).get("sha256")),
                       ("inputs_full", (((pm_.get("full_length") or {}).get("inputs_full")) or {}).get("sha256"), (((mt_.get("full_length") or {}).get("inputs_full")) or {}).get("sha256")),
                       ("cohort_list", (pm_.get("cohort_list") or {}).get("ref"), (mt_.get("cohort_list") or {}).get("ref")),
                       ("merge seed", (pm_.get("merge") or {}).get("seed"), (mt_.get("merge") or {}).get("seed"))):
        n += 1
        bad += [] if (a_ is not None and a_ == b_) else [f"per_set_beside|companion|{k_}"]
    n += 2
    bad += [] if pm_.get("pooled_world") is None else ["per_set_beside|companion|is a pooled record"]
    xs_ = ("script_sha256", "modules_sha256")  # runner @ 8a15fb10f pw_companion: the code pins are the pooled run's own
    bad += [] if ({k_: v_ for k_, v_ in pw_pool_inputs_chk(pm_).items() if k_ not in xs_}
                  == {k_: v_ for k_, v_ in pw_pool_inputs_chk(mt_).items() if k_ not in xs_}) else ["per_set_beside|companion|pool-defining inputs"]

    def summ_(d_):  # a computed summary block: a numeric point and at least one finite draw
        return isinstance(d_, dict) and isinstance(d_.get("point"), (int, float)) and not isinstance(d_.get("point"), bool) \
            and isinstance(d_.get("finite_draws"), int) and d_["finite_draws"] >= 1

    def six_(lv_, ro_):  # @ 8a15fb10f: the named headline_six_full|<ro> group with G, Y and G - Y summaries; each of the six in its rows or unreadable
        g_ = lv_.get(f"headline_six_full|{ro_}") if isinstance(lv_, dict) else None
        pr_ = g_.get("primary") if isinstance(g_, dict) else None
        s6_ = {f"{fa} {ro_}" for fa in FULL_GROUPS_[0][1]}
        return (isinstance(pr_, dict) and all(summ_(pr_.get(k_)) for k_ in ("mean_G", "mean_Y", "mean_G_minus_mean_Y")) and isinstance(pr_.get("rows"), list)
                and isinstance(g_.get("unreadable_rows"), list) and set(pr_["rows"]) <= s6_ and set(pr_["rows"]) | set(g_["unreadable_rows"]) == s6_
                and pr_.get("n") == len(pr_["rows"]) == len(set(pr_["rows"])))
    for cb in pw_["combos"]:
        lv_ = (((((psr.get("panel_groups") or {}).get(cb) or {}).get("pooled") or {}).get("joint")) or {}).get("0.01")
        n += 1
        bad += [] if six_(lv_, "C") and six_(lv_, "R") else [f"per_set_beside|companion|panel_groups|{cb}|lead level not computed"]
        for lb in labs_:  # the companion rows the row-level check needs, for every member set of the combo (the runner's members,
            for o0_ in [aob(m_) for m_ in ((combo_rows or {}).get(cb) or ((rec["pooled"].get(lb) or {}).get(cb) or {}).get("members") or [])]:  # the runner's
                q0_ = (((psr.get("rows") or {}).get(lb) or {}).get(o0_)) or {}
                n += 1
                bad += [] if all(q0_.get(k_) for k_ in ("draws_sha256", "cohort_cell_keys")) else [f"per_set_beside|companion|rows|{lb}|{o0_}|missing"]
        for g_ in ("headline_six_full|C", "headline_six_full|R"):
            q_ = (((((psr.get("coheadline_signed") or {}).get(cb) or {}).get("joint")) or {}).get("0.01") or {}).get(g_) or {}
            n += 1
            bad += [] if all(summ_(q_.get(k_)) for k_ in ("S", "Y")) else [f"per_set_beside|companion|coheadline_signed|{cb}|{g_}|not computed"]
        for lb in labs_:
            q_ = ((((psr.get("pooled") or {}).get(lb) or {}).get(cb) or {}).get("joint") or {}).get("0.01")
            n += 1
            bad += [] if (isinstance(q_, dict) and all(summ_((q_.get("catch") or {}).get(k_)) for k_ in ("sim", "run1", "run2", "real_mean"))
                          and summ_((q_.get("gap") or {}).get("delta_full"))) else [f"per_set_beside|companion|pooled|{lb}|{cb}|lead level not computed"]
    for lb in labs_:  # the same union stream and cohort, row by row
        for o_, r_ in sorted((rec["rows"].get(lb) or {}).items()):
            p_ = ((psr.get("rows") or {}).get(lb) or {}).get(aob(o_)) or {}
            n += 1
            bad += [] if (p_.get("draws_sha256") == r_.get("draws_sha256") and set(p_.get("cohort_cell_keys") or []) == set(r_.get("cohort_cell_keys") or [])) \
                else [f"per_set_beside|companion|rows|{lb}|{o_}|draws or cohort"]
    for k_, a_, b_ in (("source", psb.get("source"), pw_.get("per_set_record")), ("label", psb.get("label"), "registered: per-set calibration"),
                       ("fields", sorted(psb), ["by_combo", "coheadline_signed", "label", "panel_groups", "source"]),
                       ("combos", sorted(psb.get("by_combo") or {}), sorted(pw_["combos"])), ("panel combos", sorted(psb.get("panel_groups") or {}), sorted(pw_["combos"])),
                       ("coheadline_signed", cj(psb.get("coheadline_signed")), cj(psr.get("coheadline_signed")))):
        n += 1
        bad += [] if a_ == b_ else [f"per_set_beside|{k_}"]
    for cb in pw_["combos"]:
        bc_ = (psb.get("by_combo") or {}).get(cb) or {}
        n += 2
        bad += [] if sorted(bc_) == labs_ == sorted(rec["rows"]) else [f"per_set_beside|by_combo|{cb}|labels"]
        bad += [] if cj((psb.get("panel_groups") or {}).get(cb)) == cj((psr.get("panel_groups") or {}).get(cb)) else [f"per_set_beside|panel_groups|{cb}"]
        for lb in labs_:
            n += 1
            bad += [] if cj(bc_.get(lb)) == cj(((psr.get("pooled") or {}).get(lb) or {}).get(cb)) else [f"per_set_beside|by_combo|{cb}|{lb}"]
    return n, bad


def schema_walk(rec, labels, origins, combos, levels):
    """G7 --schema (2e 18:xx, a stub pooled record): every key path the pooled checks (G7, G7e, the aggregate, per_set_beside) read, for the checked
    labels, their union rows, the pooled combos and the levels. -> {path pattern with '*' for label / origin / combo / level: missing count}; keys only"""
    miss = {}

    def need(d, path, pat):
        for k_ in path:
            if not isinstance(d, dict) or k_ not in d:
                miss[pat] = miss.get(pat, 0) + 1
                return None
            d = d[k_]
        return d
    keys_ = [f"{x:g}" for x in levels]
    m_ = rec.get("meta") or {}
    for p_ in (("full_length",), ("auroc06_subset", "members"), ("inputs_final", "sha256"), ("cohort_list", "ref"), ("merge", "seed"), ("script_sha256",),
               ("git",), ("pooled_world", "stage"), ("pooled_world", "combos"), ("pooled_world", "rows"), ("pooled_world", "labels"),
               ("pooled_world", "manifest", "path"), ("pooled_world", "manifest", "sha256"), ("pooled_world", "manifest_per_label"),
               ("pooled_world", "authorized_overlaps", "path"), ("pooled_world", "authorized_overlaps", "sha256"), ("pooled_world", "per_set_record", "path"),
               ("pooled_world", "per_set_record", "sha256"), ("pooled_world", "modules_sha256"), ("pooled_world", "groups_out_of_scope", "groups"),
               ("pooled_world", "calibrations", "by_label_combo")):
        need(m_, p_, "meta." + ".".join(p_))
    for lab in labels:
        for o in origins:
            r_ = need(rec, ("rows", lab, o), "rows.*.*")
            if r_ is None:
                continue
            for k_ in ("coverage", "pooled_world", "all_origins", "draws_sha256", "cohort_cell_keys"):
                need(r_, (k_,), f"rows.*.*.{k_}")
            for x in ("cohort_cells", "H_sim_runs", "H_sim_units", "H_B_runs", "H_B_units", "H_A_runs", "H_A_units", "H_sim_cells_in_cohort"):
                need(r_, ("coverage", x), f"rows.*.*.coverage.{x}")
            for x in ("combo", "pools_file", "calibration_mass_draw0", "mapped_units_per_fold"):
                need(r_, ("pooled_world", x), f"rows.*.*.pooled_world.{x}")
            for nmr in ("H_B", "H_A", "H_sim"):
                for x in ("runs", "units"):
                    need(r_, ("pooled_world", "pools", nmr, x), f"rows.*.*.pooled_world.pools.{nmr}.{x}")
            for key in keys_:
                for nm in ("B", "A", "S"):
                    need(r_, ("pooled_world", "cutoffs_draw0", key, nm, "raw_cutoff"), f"rows.*.*.pooled_world.cutoffs_draw0.*.{nm}.raw_cutoff")
                    need(r_, ("readable", key, nm), f"rows.*.*.readable.*.{nm}")
                    for x in ("T", "k", "n_units", "units_above", "runs", "per_run_alarm_share", "per_run_alarm_share_with_no_unit_runs",
                              "no_unit_runs_counted_non_alarming"):
                        need(r_, ("pools_at_draw0", key, nm, x), f"rows.*.*.pools_at_draw0.*.{nm}.{x}")
                for md in ("joint", "fixed"):
                    for x in ("run1@B", "run2@B", "run2@A", "run1@A", "sim@B", "sim@S|primary", "real_mean@B"):
                        need(r_, ("points", md, key, "catch", x, "point"), f"rows.*.*.points.{md}.*.catch.{x}.point")
                    for x in ("delta_full", "delta_rr", "same_attack", "real_line", "sim_minus_run1", "sim_minus_run2"):
                        need(r_, ("points", md, key, "gap", x, "point"), f"rows.*.*.points.{md}.*.gap.{x}.point")
        for cb in combos:
            pb = need(rec, ("pooled", lab, cb), "pooled.*.*")
            if pb is None:
                continue
            need(pb, ("members",), "pooled.*.*.members")
            for md in ("joint", "fixed"):
                for key in keys_:
                    for x in ("sim", "run1", "run2", "real_mean", "run2@A", "y1"):
                        need(pb, (md, key, "catch", x, "point"), f"pooled.*.*.{md}.*.catch.{x}.point")
                    need(pb, (md, key, "catch", "y1_def"), f"pooled.*.*.{md}.*.catch.y1_def")
                    for x in ("delta_full", "delta_rr", "sim_minus_run1", "sim_minus_run2", "run2_minus_run1_at_tauB", "real_line"):
                        need(pb, (md, key, "gap", x, "point"), f"pooled.*.*.{md}.*.gap.{x}.point")
                    need(pb, (md, key, "readable", "all"), f"pooled.*.*.{md}.*.readable.all")
                    need(pb, (md, key, "D_draw0"), f"pooled.*.*.{md}.*.D_draw0")
                    for st in ("pooled", "pooled_abs_per_origin"):
                        for x in ("G", "Y", "G_minus_Y"):
                            need(pb, (md, key, "panel", st, x, "point"), f"pooled.*.*.{md}.*.panel.{st}.{x}.point")
                        need(pb, (md, key, "panel", st, "saturated"), f"pooled.*.*.{md}.*.panel.{st}.saturated")
    pwl_ = set((m_.get("pooled_world") or {}).get("labels") or [])  # Codex r46 #3: the aggregate trees only where a group is in the pooled scope
    coh_in = any({f"{fa} {ro}" for fa in fs} <= pwl_ for _, fs in FULL_GROUPS_ + (("auroc06_subset", AUROC06_FULL),) for ro in "CR")
    pan_in = any({f"{fa} {ro}" for fa in fs for ro in ros} <= pwl_ for _, fs in FULL_GROUPS_ for ros in ("R", "C", "RC"))
    for cb in combos:
        for md in ("joint", "fixed"):
            for key in keys_:
                if coh_in:
                    need(rec, ("coheadline_signed", cb, md, key), "coheadline_signed.*.*.*")
                for st in ("pooled", "pooled_abs_per_origin") if pan_in else ():
                    need(rec, ("panel_groups", cb, st, md, key), f"panel_groups.*.{st}.*.*")
    if m_.get("pooled_world", {}).get("per_set_record"):
        for x in ("source", "label", "by_combo", "coheadline_signed", "panel_groups"):
            need(rec, ("per_set_beside", x), f"per_set_beside.{x}")
    return dict(sorted(miss.items()))


def rule37_gate_chk(rd_):  # 09 RULE 37 (the pooled gate, item 45(2)): every member's S and B pools readable at draw 0; a missing flag fails closed
    return bool(rd_) and all(v_.get("S") is True and v_.get("B") is True for v_ in rd_.values())


def classify(classes, structural, nonstructural):  # -> {class: structural?}; the longest matching prefix decides (09's ruling via 2e 02:3x:
    # structural = the cause precedes, or is independent of, the arm's own policy behaviour). With a non-structural list, every class must match one
    # of the two lists, else the check stops: an unknown class is never silently non-structural
    assert not set(structural) & set(nonstructural), "a prefix in both lists"
    out = {}
    for cl in classes:
        m_ = [(len(p_), True) for p_ in structural if cl.startswith(p_)] + [(len(p_), False) for p_ in nonstructural if cl.startswith(p_)]
        assert m_ or not nonstructural, (cl, "an exclusion class in neither the structural nor the non-structural list")
        out[cl] = max(m_)[1] if m_ else False
    return out


def completion_gate(meta, r2, test):  # 09 RULE @ 07ad588be (2e 02:4x): a non-TEST second-refresh record has waitlist_pending == [] and the pinned exclusions
    if r2 and not test:
        assert meta.get("waitlist_pending") == [] and r2.get("waitlist_pending") == [], f"STOP: waitlist_pending is {meta.get('waitlist_pending')!r} in a non-TEST record"
        assert meta.get("exclusions") and meta.get("exclusions") == r2.get("exclusions"), "STOP: a non-TEST record without the inputs file's pinned exclusions file"


def exclusion_identity(EX, univ, want_final, want_excl, test):  # per set, by key: launched = final + excluded + pending, no overlap; per_set and
    # per-class counts equal the keys; final = the record's cohort; excluded = the no-filter list's (where given); pending only in a TEST record
    exc_by = {s_: {} for s_ in univ}
    for r_ in EX["rows"]:
        assert r_["key"] not in exc_by[r_["set"]], (r_["set"], r_["key"], "two exclusion rows for one cell")
        exc_by[r_["set"]][r_["key"]] = r_["class"]
    assert set(EX["per_set"]) == set(univ) == set(EX["final_cohort_keys"]) and set(EX["pending"]) <= set(univ), "exclusions file sets"
    wl = {}
    for s_, u_ in univ.items():
        fin_, exc_, pen_ = EX["final_cohort_keys"][s_]["cells"], exc_by[s_], list(EX["pending"].get(s_, []))
        assert EX["final_cohort_keys"][s_]["sha256"] == hashlib.sha256(json.dumps(sorted(fin_), separators=(",", ":")).encode()).hexdigest(), (s_, "final cohort sha256")
        parts_ = (set(fin_), set(exc_), set(pen_))
        assert len(fin_) + len(exc_) + len(pen_) == len(set().union(*parts_)) and set().union(*parts_) == set(u_), (s_, "launched != final + excluded + pending by key")
        ps_ = EX["per_set"][s_]
        assert (ps_["launched"], ps_["final"], ps_["excluded"], ps_["pending"]) == (len(u_), len(fin_), len(exc_), len(pen_)), (s_, "per_set counts != keys")
        cnt_ = {}
        for cl_ in exc_.values():
            cnt_[cl_] = cnt_.get(cl_, 0) + 1
        assert ps_["excluded_by_class"] == cnt_, (s_, "per-class counts != the rows by key")
        assert test or not pen_, f"STOP: {s_} has {len(pen_)} pending (in-flight) cells in a non-TEST record"
        assert set(fin_) == set(want_final[s_]), (s_, "final cohort != the record's no-filter cohort")
        assert want_excl.get(s_) is None or set(exc_) == set(want_excl[s_]), (s_, "excluded cells != the no-filter list's")
        if pen_:
            wl[s_] = sorted(pen_)
    return exc_by, wl


RULED = {"excluded by ruling (09 @ 1bc718e98)": ("S_newmains", "nm290_tiktok", "attempt timeout (cause unknown)")}  # 09 via 2e 02:5x (rules @
# 806da4823 §§2-3, no new RULE): S5 fu sim|nm290_tiktok is a RetryError wrapping AttemptTimeoutError, so not structural; only that cell may carry the label


def apply_rulings(exc_by, ruled):  # -> the relabels made; a ruled label on any other cell STOPs (its key goes to 2e)
    done = []
    for s_, d_ in exc_by.items():
        for c_, cl_ in sorted(d_.items()):
            if cl_ in ruled:
                assert (s_, c_) == ruled[cl_][:2], f"STOP: {cl_!r} on {s_}|{c_}, not {ruled[cl_][0]}|{ruled[cl_][1]}: send this key to 2e"
                d_[c_] = ruled[cl_][2]
                done.append({"set": s_, "key": c_, "file_class": cl_, "class": ruled[cl_][2]})
    return done


def _stops(f_):  # a STOP case: True when the call raises AssertionError (the checker's form of the runner's SystemExit)
    try:
        f_()
    except AssertionError:
        return True
    return False


def selftest_fold():  # noqa: C901
    """G7b FOLD-MODEL (2e's spec (c)): synthetic in-memory npz files per family (Sol: five models, fold -1 units, an A25 sim-honest file without fold;
    own_all_rs: NaN score_by_model by design, the fold-list guard) through the ported readers and rules, the pooled-file and f1cols comparisons, and
    the PLANTED control: an alias inserted into a fold-model pool STOPs gate 2. Synthetic values only; < 1 s."""
    import numpy as np

    nan_ = float("nan")

    def npz(**a_):  # a synthetic per-row file, written and read back as the check reads one
        b_ = io.BytesIO()
        np.savez(b_, **a_)
        return np.load(io.BytesIO(b_.getvalue()), allow_pickle=False)
    sbm_ = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4, 1.5]])
    # Sol: no 'score' array: the designated score is the fold's model column, NaN on fold -1 (C.rows_of); the five model scores kept
    k_, s_, f_, b_ = rows_from_npz(npz(key=np.array(["honest_b|t1_a", "honest_b|t1_a", "sim|t2_b"]), fold=np.array([2, -1, 0]), score_by_model=sbm_), True, None)
    assert f_.tolist() == [2, -1, 0] and s_[0] == 0.3 and math.isnan(s_[1]) and s_[2] == 1.1 and b_.shape == (3, 5), "Sol reader"
    assert rows_from_npz(npz(key=np.array(["sim_honest|t2_b"]), score=np.array([0.4])), True, None, simh=True)[2] is None, "A25 file without fold"
    assert _stops(lambda: rows_from_npz(npz(key=np.array(["honest_b|t1_a"]), score=np.array([0.4])), True, None)), "a real-side Sol file without fold STOPs"
    fl_, cf_ = r36_fill(np.array(["sim_honest|t2_b", "sim_honest|t1_a"]), np.array(["honest_b|t1_a", "run1|t2_b", "honest_c|t2_b", "honest_b|t9_x"]),
                        np.array([2, 1, 3, -1]), np.array(["S_old", "S_old", "S_old", "S_old"]))
    assert fl_.tolist() == [3, 2] and cf_ == ["t2_b"], ("r36 from S_old rows, the last row wins; conflicting cells listed", fl_.tolist(), cf_)
    assert _stops(lambda: r36_fill(np.array(["sim_honest|t9_x"]), np.array(["honest_b|t9_x"]), np.array([-1]), np.array(["S_old"]))), "no real-side fold STOPs"
    assert _stops(lambda: r36_fill(np.array(["sim_honest|t3_c"]), np.array(["honest_b|t3_c"]), np.array([1]), np.array(["S_new"]))), "S_new rows are no source"
    # own_all_rs: score, score_by_model (NaN allowed) and fold 0-4 required; fold -1, a missing array or a NaN designated score STOP
    rs_ = dict(key=np.array(["honest_a|t1_a", "run2|t2_b"]), score=np.array([0.2, 0.3]), fold=np.array([0, 4]), score_by_model=np.full((2, 5), nan_))
    assert rows_from_npz(npz(**rs_), True, "own_all_rs")[2].tolist() == [0, 4], "own_all_rs reader"
    for bad_ in (rs_ | {"fold": np.array([0, -1])}, {k0_: v0_ for k0_, v0_ in rs_.items() if k0_ != "score_by_model"}, rs_ | {"score": np.array([0.2, nan_])}):
        assert _stops(lambda b0_=bad_: rows_from_npz(npz(**b0_), True, "own_all_rs")), "own_all_rs guard"
    cm_, fom_ = {"t1_a": "m1", "t2_b": "m2"}, {"m1": 0, "m2": 4}
    assert rs_fold_viol(["honest_a|t1_a", "run2|t2_b"], [0, 4], cm_, fom_) == [] and \
        rs_fold_viol(["honest_a|t1_a", "run2|t2_b", "run1|t3_c"], [1, 4, 0], cm_, fom_) == ["honest_a|t1_a", "run1|t3_c"], "own_all_rs fold list by key"
    # S_paper493 r36: S_paper493 rows on r36 (None = -1); H_sim rows on r36 for S_paper493 cells, else the real-side fold (two folds there: none)
    c_ = np.array(["p1_a", "p2_b", "t1_a", "t1_a", "t2_b", "t2_b"])
    off_ = sp_r36_off(c_, np.array([1, -1, 2, 2, 0, 3]), np.array(["S_paper493", "S_paper493", "S_old", "S_new", "S_old", "S_old"]),
                      np.array(["sim_honest|p1_a", "sim_honest|t1_a", "sim_honest|t2_b"]), np.array([1, 2, 0]), {"p1_a": 1, "p2_b": None})
    assert off_ == ["sim_honest|t2_b"], ("r36 rule", off_)
    assert sp_r36_off(c_[:1], np.array([0]), np.array(["S_paper493"]), np.array([], str), np.array([], np.int64), {"p1_a": 1}) == ["p1_a"], "off r36 flagged"
    # per-pool STOPs: a fold with < 2 units (Sol, no 21b) STOPs; the own_all_rs out-of-fold fallback does not, NaN model scores allowed there
    P5_ = lambda f0: (np.array([f"honest_b|t{i}" for i in range(len(f0))]), np.zeros(len(f0)), np.array(f0), np.full((len(f0), 5), nan_))  # noqa: E731
    full_ = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert fold_pool_stops({"B": P5_(full_), "A": P5_(full_), "S": P5_(full_)}, False, False) == [], "every fold with >= 2 units passes"
    w_ = fold_pool_stops({"B": P5_(full_[:-1]), "A": P5_(full_), "S": P5_(full_)}, False, False)
    assert len(w_) == 2 and "not all finite" in w_[0] and "lacks a fold" in w_[1], w_
    assert fold_pool_stops({"B": P5_(full_[:-1]), "A": P5_(full_), "S": P5_(full_)}, False, True) == [], "oof fallback (own_all_rs)"
    assert "fold -1 units without five finite" in fold_pool_stops({"B": P5_(full_ + [-1, -1]), "A": P5_(full_), "S": P5_(full_)}, True, False)[0], "21b"
    assert fold_pool_stops({"B2": P5_(full_[:-1])}, False, True) == [] and fold_pool_stops({"B2": P5_(full_[:-1])}, False, False)[0].startswith("B2: a pool fold"), "B2"
    # attack fold -1 columns: run1 / run2 / sim runs on fold -1 -> their mains' union-stream columns (outside mains: the extra column)
    f1_ = f1cols_of({"run1": np.array([-1, 0, 0]), "run2": np.array([0, 0, -1]), "sim": np.array([0, -1, 0])}, ["c1", "c2", "c3"],
                    {"c1": "m_b", "c2": "m_a", "c3": "m_z"}, ["m_a", "m_b"])
    assert f1_ == {"B": {0, 1, 2}, "A": {1, 2}, "S": {0}}, f1_
    # unions on fold-model pools: NaN designated scores and NaN model scores compare as equal content; a pooled file is compared by canonical bytes
    def P_(k0, s0, f0, b0):
        return (np.array(k0), np.array(s0, dtype=np.float64), np.array(f0, np.int64), np.array(b0, dtype=np.float64))
    bb_ = [[0.1, 0.2, nan_, 0.4, 0.5]]
    Z_ = {"A": P_([], [], [], np.empty((0, 5))), "S": P_([], [], [], np.empty((0, 5)))}
    e0_ = {"B": P_(["honest_b|t1", "honest_b|t1"], [nan_, 0.3], [-1, 2], bb_ * 2)} | Z_
    e1_ = {"B": P_(["honest_b|t1", "honest_b|t1", "honest_a_e2|t5", "honest_b_e2|t5"], [nan_, 0.3, 0.7, 0.7], [-1, 2, 1, 1], bb_ * 4)} | Z_
    pl_, sz_, pt_ = pw_union_chk([("S_old (all-origins draws) [pooled: cb]", e0_), ("S_new (all-origins draws) [pooled: cb]", e1_)], "selftest")
    assert sz_["identity_conflicts"] == [] and sz_["content_identical_keys"] == [["honest_a_e2|t5", "honest_b_e2|t5"]], sz_["content_identical_keys"]
    org_ = np.array([pt_["B"][r] for r in pl_["B"][0].tolist()])
    fa_ = (pl_["B"][0], pl_["B"][1], pl_["B"][2], pl_["B"][3], org_)
    assert pools_file_diff_five(fa_, pl_["B"], pt_["B"]) == (0, 0, 0), "a pooled file equal to the union (NaN on both sides) passes"
    b2_ = pl_["B"][3].copy()
    b2_[0, 0] = 0.11
    assert pools_file_diff_five((fa_[0], fa_[1], fa_[2], b2_, org_), pl_["B"], pt_["B"]) == (0, 0, 1), "a changed model score is flagged"
    assert pools_file_diff_five((fa_[0], fa_[1], fa_[2], fa_[3], np.array(["S_old"] * len(org_))), pl_["B"], pt_["B"]) == (0, 0, 3), "changed patterns"
    assert pools_file_diff_five(tuple(x_[:-1] for x_ in fa_), pl_["B"], pt_["B"]) == (0, 1, 0) and pools_file_diff_five(tuple(x_[1:] for x_ in fa_), pl_["B"], pt_["B"]) == (0, 0, 1), "a run missing from the file; a unit missing"
    # gate 2 on a fold-model union: the declared own_all_rs nm305 pair passes with the record's natural verdict for this label's scorer file; the
    # PLANTED alias (a copy of one H_B run under a new key) STOPs gate 2 as one undeclared group, and the real union is untouched
    dg_ = {frozenset(("honest_a_e2|t5", "honest_b_e2|t5"))}
    tp_ = {"pairs": [{"pair": ["honest_a_e2|t5", "honest_b_e2|t5"], "verdict": "(1) natural",
                      "files": [["own_all_rs_uncut", "own_all_rs_uncut_L48__R.npz", "own_all_rs_uncut", "own_all_rs_uncut_L48__R.npz", "natural"]]}]}
    pfs_ = prefix_support(tp_, {("own_all_rs_uncut", "own_all_rs_uncut_L48__R.npz")})
    assert pw_identity_gates(sz_, pt_, set(), None, dg_, pfs_) == [], "declared fold-model group with the record's support"
    ex_ = [("S_old (all-origins draws) [pooled: cb]", e0_), ("S_new (all-origins draws) [pooled: cb]", e1_)]
    pa_ = plant_alias(ex_)
    assert pa_[1] == "plantedalias|t1" and pa_[2] == "honest_b|t1" and len(ex_[0][1]["B"][0]) == 2, "plant on a copy"
    u2_ = pw_union_chk(pa_[0], "selftest plant")
    gv2_ = pw_identity_gates(u2_[1], u2_[2], set(), None, dg_, pfs_)
    assert [g_ for g_ in u2_[1]["content_identical_keys"] if pa_[1] in g_] == [["honest_b|t1", "plantedalias|t1"]] and gv2_[0].startswith("no false split: 1 "), gv2_
    # S_paper493 world alias: one allowed token renamed; another pair or two tokens STOP
    ka_, rn_ = sp_world_chk(np.array(["sim_honest|p1_a"]), "sp493_t196", "selftest")
    assert ka_.tolist() == ["sp493_t196|p1_a"] and rn_ == "sim_honest" and sp_world_chk(np.array(["run1|p1_a"]), "run1", "t")[1] is None, "sp_world"
    assert _stops(lambda: sp_world_chk(np.array(["run1|p1_a"]), "sim", "t")) and _stops(lambda: sp_world_chk(np.array(["sim|p1_a", "x|p2"]), "sim", "t"))
    # --shapes (Astra r1 #3, #4, #6; Claude N3, N7): the schema from the names (a Sol sim-honest file with fold read as five models), 1-D text keys,
    # 1-D integer folds, (n, 5) model scores; arrays the reader never reads are not judged; nested manifest paths; pooled npz shapes
    A_ = lambda **kw: {k0_: (list(v0_[0]), v0_[1]) for k0_, v0_ in kw.items()}  # noqa: E731
    assert shape_of({"key", "fold", "score"}, A_(key=([3], "<U9"), fold=([3], "<i8"), score=([3], "<f8")), "own_all_rs") == (["score_by_model"], [])
    assert shape_of({"key", "fold", "score_by_model"}, A_(key=([3], "<U9"), fold=([3], "<i8"), score_by_model=([3, 4], "<f8")), "five")[1] == \
        ["score_by_model shape [3, 4] (want [3, 5])"]
    sol_simh_ = A_(key=([3], "<U9"), fold=([3], "<i8"), score_by_model=([3, 5], "<f8"))
    assert shape_of(set(sol_simh_), sol_simh_, "simh_sol") == ([], []) and shape_of({"key"}, A_(key=([3], "<U9")), "simh_sol")[0] == ["score"], "schema by names"
    assert shape_of({"key", "score"}, A_(key=([1, 1], "<U9"), score=([1], "<f8")), "simh_sol")[1][0].startswith("key shape [1, 1]"), "a 2-D key"
    assert shape_of({"key", "score"}, A_(key=([2], "<f8"), score=([2], "<f8")), "simh_sol")[1][0].endswith("(want 1-D text)"), "a numeric key"
    assert shape_of({"key", "fold", "score"}, A_(key=([2], "<U9"), fold=([2], "<f8"), score=([2], "<f8")), "five")[1] == ["fold dtype <f8 (want integer)"]
    assert shape_of({"key", "fold", "score", "unit"}, A_(key=([2], "<U9"), fold=([2], "<i8"), score=([2], "<f8"), unit=([], "<i8")), "five") == ([], []), "N7"
    P_ok_ = A_(**{f"{nm}_{x}": ([4, 5] if x == "sbm" else [4], {"run": "<U20", "org": "<U20", "s": "<f8", "f": "<i8", "sbm": "<f8"}[x])
                  for nm in "BAS" for x in ("run", "s", "f", "sbm", "org")})
    assert pooled_shape_faults(P_ok_) == [] and pooled_shape_faults(P_ok_ | A_(B_run=([4, 1], "<U20")))[0] == "B_run not 1-D text", "a 2-D B_run"
    assert pooled_shape_faults(P_ok_ | A_(S_sbm=([4], "<f8"))) == ["S_sbm shape [4] dtype <f8 (want [4, 5], kind f)"], "pooled sbm"
    mf0_ = {"sizes": {"B": {}, "A": {"after": 1, "by_membership_pattern": {}, "zero_unit_runs": {}}}}
    assert path_missing(mf0_, ("sizes", "B", "after")) and not path_missing(mf0_, ("sizes", "A", "after")) and path_missing(mf0_, ("sizes", "S", "after"))
    # the mapped-unit withhold (Astra r1 #1): S_newmains is allowed by the all-units deviation or the fold extension; S_paper493 never
    mv0_ = {"H_B": {"mapped": 0}, "H_A": {"mapped": 3}, "H_sim": {"mapped": 2}}
    assert mvf_withheld(mv0_, False, False, False) and not mvf_withheld(mv0_, True, False, False) and not mvf_withheld(mv0_, False, True, False) \
        and mvf_withheld(mv0_, True, True, True) and not mvf_withheld({k0_: {"mapped": 1} for k0_ in mv0_}, False, False, True), "mvf_withheld"
    # a without-S4 row's attack fold -1 columns (Astra r1 #2): the with-S4 cohort's fold -1 attack cells, only under 21b
    c4_, w4_ = np.array(["s4a", "s4a", "n1", "n2"]), np.array(["run1", "run1", "sim", "run2"])
    f4_ = f1cols_nos4(c4_, w4_, np.array([-1, -1, -1, 0]), np.ones(4, bool), ["s4a", "n1", "n2"], True, {"s4a": "m_s4", "n1": "m_1", "n2": "m_2"}, ["m_1", "m_s4"])
    assert f4_ == {"B": {0, 1}, "A": {1}, "S": {0}} and f1cols_nos4(c4_, w4_, np.array([-1, -1, -1, 0]), np.ones(4, bool), ["s4a"], False,
                                                                    {"s4a": "m_s4"}, ["m_s4"]) == {"B": set(), "A": set(), "S": set()}, f4_
    # fold dtype / range before the cast (Claude N3): a non-integral fold or one outside -1..4 STOPs
    for fb_ in (np.array([0.0, 4.9, 1.0]), np.array([0, 5, 1]), np.array([-2, 0, 1])):
        assert _stops(lambda f0_=fb_: rows_from_npz(npz(key=np.array(["a|t1", "b|t2", "c|t3"]), fold=f0_, score_by_model=sbm_), True, None)), fb_
    assert rows_from_npz(npz(key=np.array(["a|t1", "b|t2", "c|t3"]), fold=np.array([0.0, 4.0, -1.0]), score_by_model=sbm_), True, None)[2].tolist() == [0, 4, -1]
    # the PLANTED control on a union with an identical-content group: the copied run is outside it, so the planted group is exactly the pair
    pe_ = plant_alias(ex_, "B", {k0_ for g0_ in sz_["content_identical_keys"] for k0_ in g0_})
    assert pe_[2] == "honest_b|t1" and plant_alias(ex_, "B", {"honest_b|t1", "honest_a_e2|t5"})[2] == "honest_b_e2|t5", "exclusion"
    # the 8-label comparison (2e 21:1x / 21:3x, Astra r2 #1): TWO-SIDED by (row, rule); synthetic one-label records (n_labels=1)
    g0_ = {"row": "F C", "rule": "g (cb)", "violations": [], "runs": {"B": 3}}
    rr_ = {"rules": [g0_], "all_equal": True}
    al_ = {"row": "F C", "rule": "fold-model rules on S_paper493 (all-origins draws) [pooled: cb] (info; x)", "violations": [],
           "info": {"sp_world_aliases": {"arm1": {"alias": "attack -> run1", "renamed_rows": 4, "renamed_runs": 2}}}}
    cg_ = lambda nn: compare_g7b(rr_, nn, 1)[0]  # noqa: E731
    d0_, r0_ = compare_g7b(rr_, {"rules": [g0_, al_], "all_equal": True}, 1)
    assert d0_ == [] and r0_["renamed_rows_by_label_combo"]["F C | cb"] == 4, (d0_, r0_)
    assert cg_({"rules": [g0_ | {"runs": {"B": 4}}], "all_equal": True}) == [["F C", "g (cb)", "field runs"]], "a changed value"
    assert cg_({"rules": [g0_ | {"extra": 1}], "all_equal": True}) == [["F C", "g (cb)", "field extra present on one side only"]], "a new field"
    assert cg_({"rules": [{k0_: v0_ for k0_, v0_ in g0_.items() if k0_ != "runs"}], "all_equal": True}) == [["F C", "g (cb)", "field runs present on one side only"]]
    assert compare_g7b({"rules": [g0_ | {"x": None}], "all_equal": True}, {"rules": [g0_], "all_equal": True}, 1)[0] == \
        [["F C", "g (cb)", "field x present on one side only"]], "a missing field is not a null"
    assert cg_({"rules": [], "all_equal": True})[0][2].startswith("an empty rule domain"), "an empty new domain"
    assert compare_g7b({"rules": [], "all_equal": True}, {"rules": [g0_], "all_equal": True}, 1)[0][0][2].startswith("an empty rule domain"), "empty prior"
    assert cg_({"rules": [g0_, {"row": "F C", "rule": "h", "violations": ["x"]}], "all_equal": True}) == \
        [["F C", "h", "a new rule outside the allowlist (cfg0 regression / alias info, no violations, no other fields)"]]
    assert cg_({"rules": [g0_, {"row": "F C", "rule": "h", "violations": []}], "all_equal": True})[0][1] == "h", "a new rule outside the allowlist"
    assert cg_({"rules": [g0_, al_ | {"violations": ["v"]}], "all_equal": True})[0][2].startswith("a new rule outside"), "an allowlisted rule with violations"
    assert cg_({"rules": [g0_, al_ | {"runs": 1}], "all_equal": True})[0][2].startswith("a new rule outside"), "an allowlisted rule with another field"
    assert cg_({"rules": [g0_, g0_], "all_equal": True})[0][2].startswith("duplicate"), "unique keys"
    assert cg_({"rules": [g0_], "all_equal": False}) == [["*", "*", "the new record is not all_equal"]], "all_equal"
    assert cg_({"rules": [g0_], "all_equal": True, "rows": [{"row": "F C"}]}) == [["F C", "*", "rows entries for a public label"]], "rows"
    assert compare_g7b(rr_, {"rules": [g0_], "all_equal": True})[0][0][2].endswith("not on 8 labels"), "the real prior has 8 labels"
    assert cg_({"rules": [g0_, {"rule": "x"}], "all_equal": True}) == [["*", "*", "new record: 1 malformed rule entries (no row or rule)"]], "NB-d"
    assert compare_g7b(rr_, {"rules": [g0_], "all_equal": True}, 1, ["G C"])[0] == [["*", "*", "the prior's labels are not the expected ones: ['F C']"]]
    # --shapes judges only what the reader consumes (Astra r2 #2): an A25 file's unread model scores, a stale score_by_fold beside score_by_model
    a25_ = A_(key=([2], "<U9"), score=([2], "<f8"), score_by_model=([1, 4], "<f8"))
    assert shape_of(set(a25_), a25_, "simh_sol") == ([], []), "an unread array is not judged"
    st_ = A_(key=([2], "<U9"), fold=([2], "<i8"), score_by_model=([2, 5], "<f8"), score_by_fold=([2, 4], "<f8"))
    assert shape_of(set(st_), st_, "five") == ([], []) and shape_of({"key", "fold", "score_by_fold"}, st_, "five")[1] == ["score_by_fold shape [2, 4] (want [2, 5])"]
    P_empty_ = P_ok_ | A_(A_run=([0], "<f8"), A_s=([0], "<f8"), A_f=([0], "<f8"), A_sbm=([0], "<f8"), A_org=([0], "<f8"))
    assert pooled_shape_faults(P_empty_) == [] and pooled_shape_faults(P_empty_ | A_(A_sbm=([3, 5], "<f8"))), "507-prep NB1: an empty role (np.zeros(0))"
    # G7b VALUE: the draw-0 pool z against the runner's own formulas written out at unit weights (wmoments' weighted sums, zpool's 21b), the
    # out-of-fold fallback, pooled H_A on H_B's moments, and the comparison tolerance
    rng_ = np.random.default_rng([20261016, 22])
    fz_ = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, -1, -1, -1])
    sb_ = rng_.normal(size=(len(fz_), 5))
    sz_ = np.where(fz_ >= 0, sb_[np.arange(len(fz_)), np.maximum(fz_, 0)], np.nan)
    z0_, mom0_, ok0_ = pool_z0((np.array([f"r|{i}" for i in range(len(fz_))]), sz_, fz_, sb_), True)
    w1_ = np.ones(len(fz_))
    mu_ = np.array([(w1_[fz_ == j] @ sz_[fz_ == j]) / w1_[fz_ == j].sum() for j in range(5)])
    sd_ = np.sqrt(np.maximum(np.array([(w1_[fz_ == j] @ sz_[fz_ == j] ** 2) / w1_[fz_ == j].sum() for j in range(5)]) - mu_ ** 2, 0.0))
    m21_ = ((sb_[fz_ == -1] - mu_) / sd_).mean(1)
    zr_ = np.where(fz_ >= 0, (sz_ - mu_[np.maximum(fz_, 0)]) / sd_[np.maximum(fz_, 0)], np.nan)
    zr_[fz_ == -1] = (m21_ - m21_.mean()) / np.sqrt(max((m21_ ** 2).mean() - m21_.mean() ** 2, 0.0))
    assert ok0_ and np.allclose(z0_, zr_, rtol=0, atol=1e-12), "pool_z0 = the runner's zpool at unit weights"
    fo_ = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4])  # fold 4 has one unit: every model's moments over all units (oof: the designated scores)
    so_ = rng_.normal(size=len(fo_))
    bo_ = rng_.normal(size=(len(fo_), 5))
    zo_, mo_, _ = pool_z0((np.array(["r|x"] * len(fo_)), so_, fo_, bo_), True, oof=True)
    zn_, mn_, _ = pool_z0((np.array(["r|x"] * len(fo_)), so_, fo_, bo_), True, oof=False)
    assert np.allclose(mo_[0], so_.mean()) and np.allclose(mn_[0], bo_.mean(0)), "the all-unit fallback: oof reads the designated scores, else the model scores"
    assert np.allclose(zo_, (so_ - so_.mean()) / so_.std()), "oof z"
    fa_ = np.array([0, 1, 2, 3, 4, -1])
    ba_ = rng_.normal(size=(6, 5))
    sa_ = np.where(fa_ >= 0, ba_[np.arange(6), np.maximum(fa_, 0)], np.nan)
    za_, ma_, oka_ = pool_z0_with((np.array(["honest_a|t"] * 6), sa_, fa_, ba_), (z0_, mom0_, ok0_))
    MU_, SD_, mu0_, sd0_ = mom0_
    assert ma_ is mom0_ and oka_ == ok0_ and np.allclose(za_[:5], (sa_[:5] - MU_[fa_[:5]]) / SD_[fa_[:5]]) and \
        np.isclose(za_[5], (((ba_[5] - MU_) / SD_).mean() - mu0_) / sd0_), "pooled H_A takes H_B's moments and fold -1 transform"
    assert _stops(lambda: pool_z0_with((np.array(["a|t"]), np.array([np.nan]), np.array([-1]), ba_[:1]), (None, (MU_, SD_, None, None), True))), \
        "H_A fold -1 units without H_B's transform STOP"
    assert eq(1e9, 1e9 + 0.5) and not eq(1e9, 1e9 + 5.0) and not eq(0.5, 0.5 + 2e-9) and eq(0.5, 0.5 + 5e-10), "1e-9 relative, floored at 1e-9 absolute"
    return ("G7b FOLD-MODEL: synthetic files per family through the ported readers (Sol: model column and NaN on fold -1, an A25 file without fold takes "
            "its cell's r36 fold from S_old rows or STOPs; own_all_rs: NaN model scores allowed, fold -1 / missing arrays / NaN scores STOP, units off the "
            "committed fold list flagged by key); S_paper493 r36 and world aliases; per-pool STOPs (a fold with < 2 units, oof fallback, 21b); attack "
            "fold -1 columns; pooled files compared by canonical bytes incl. model scores (NaN-safe); a declared fold-model group passes with the record's "
            "support; the PLANTED alias in a fold-model pool STOPs gate 2 as one undeclared pair (the copied run outside every identical group); --shapes "
            "schemas by array names, 1-D text keys, integer folds, (n, 5) model scores, pooled npz shapes, nested manifest paths; the mapped-unit withhold "
            "with the all-units / fold-extension exceptions; without-S4 attack fold -1 columns; fold dtype and range before the cast; the 8-label "
            "comparison by rule key; G7b VALUE: the draw-0 pool z equals the runner's zpool at unit weights (mapped folds, 21b), the out-of-fold "
            "fallback, pooled H_A on H_B's moments (a missing fold -1 transform STOPs), the 1e-9 relative tolerance")


def selftest_value():  # noqa: C901
    """G7b VALUE (09 via 2e 23:2x; Astra r5; 507-prep 23:11): the per-draw ports against brute force on synthetic pools (integer draw weights = units
    repeated), the skip / undefined-transform rules, the union-stream port and its hash, the catch / combination forms, npz keys, the S / Y means,
    the pre-built domain and the plant verdict."""
    import numpy as np

    rng_ = np.random.default_rng([20261016, 23])
    for _ in range(20):  # wthr: T = the k-th (0-based) value of the weight-expanded descending list, k = floor(f * sum w + 1e-9), NaN when k < 1
        z_ = np.round(rng_.normal(size=40), 1)  # ties included (mergesort keeps the runner's order; a tie has one value either way)
        w_ = rng_.integers(0, 4, size=40).astype(float)
        for f_ in (0.01, 0.05, 0.3):
            k_ = int(np.floor(f_ * w_.sum() + 1e-9))
            ex_ = np.sort(np.repeat(z_, w_.astype(int)))[::-1]
            t_ = wthr_w(z_, w_, f_)
            assert (np.isnan(t_) if k_ < 1 else t_ == ex_[min(k_, len(ex_) - 1)]), ("wthr_w vs the expanded list", f_, k_)
        srt_ = np.sort(z_)[::-1]
        assert wthr0(z_, 0.05) == srt_[min(int(np.floor(0.05 * 40 + 1e-9)), 39)], "wthr0 = the draw-0 order statistic"
    fz_ = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, -1, -1, -1, -1])
    sb_ = rng_.normal(size=(len(fz_), 5))
    sz_ = np.where(fz_ >= 0, sb_[np.arange(len(fz_)), np.maximum(fz_, 0)], np.nan)
    P_ = (np.array([f"r|{i}" for i in range(len(fz_))]), sz_, fz_, sb_)
    for _ in range(10):  # integer weights >= 1 on mapped units: the weighted moments / 21b = the unweighted ones on the repeated pool
        w_ = rng_.integers(1, 4, size=len(fz_)).astype(float)
        r_ = np.repeat(np.arange(len(fz_)), w_.astype(int))
        zw_, mw_, okw_, skw_ = pool_zw(P_, True, False, w_, 1.0)
        ze_, me_, oke_ = pool_z0((P_[0][r_], sz_[r_], fz_[r_], sb_[r_]), True)
        assert okw_ and oke_ and not skw_ and np.allclose(mw_[0], me_[0], rtol=0, atol=1e-12) and np.allclose(mw_[1], me_[1], rtol=0, atol=1e-12) \
            and np.isclose(mw_[2], me_[2], rtol=0, atol=1e-12) and np.isclose(mw_[3], me_[3], rtol=0, atol=1e-12) \
            and np.allclose(zw_[r_], ze_, rtol=0, atol=1e-12), "pool_zw at integer weights = pool_z0 on the repeated pool"
        for f_ in (0.05, 0.3):
            assert wthr_w(zw_, w_, f_) == wthr0(ze_, f_) or abs(wthr_w(zw_, w_, f_) - wthr0(ze_, f_)) <= 1e-12, "weighted tau = tau of the repeated pool"
    w0_ = np.where(fz_ == -1, 0.0, 1.0)  # every fold -1 unit unused: skip (mu0 0, sd0 1, z -inf) only when the attack fold -1 weight is 0 too
    zs_, ms_, oks_, sks_ = pool_zw(P_, True, False, w0_, 0.0)
    assert sks_ and oks_ and ms_[2] == 0.0 and ms_[3] == 1.0 and np.all(np.isneginf(zs_[fz_ == -1])), "21b skipped (7310476ae)"
    _, mn_, okn_, skn_ = pool_zw(P_, True, False, w0_, 2.0)
    assert not skn_ and not okn_ and np.isnan(mn_[2]), "fold -1 units unused but attack fold -1 weight > 0: undefined, the draw invalid"
    w2_ = np.ones(len(fz_))
    w2_[fz_ == 4] = [1.0, 0.0, 0.0]  # a fold with total weight < 2 in this draw (wmoments: NaN mean): invalid
    assert not pool_zw(P_, True, False, w2_, 1.0)[2], "a fold with weight < 2 invalidates the draw"
    fa_ = np.array([0, 1, 2, 3, 4, -1])
    ba_ = rng_.normal(size=(6, 5))
    sa_ = np.where(fa_ >= 0, ba_[np.arange(6), np.maximum(fa_, 0)], np.nan)
    PA_ = (np.array(["honest_a|t"] * 6), sa_, fa_, ba_)
    zB_ = pool_zw(P_, True, False, np.ones(len(fz_)), 1.0)[:3]
    za_, _, oka_, bda_ = pool_zw_with(PA_, zB_, np.ones(6), 0.0)
    assert bda_ and not oka_ and np.isnan(za_[5]), "H_A fold -1 units weighted while H_B's transform is skipped: undefined (Astra r1 #6)"
    za_, _, oka_, bda_ = pool_zw_with(PA_, zB_, np.array([1, 1, 1, 1, 1, 0.0]), 0.0)
    assert not bda_ and oka_ and np.isneginf(za_[5]), "unused H_A fold -1 units: -inf"
    za_, _, oka_, bda_ = pool_zw_with(PA_, zB_, np.ones(6), 3.0)
    assert not bda_ and oka_ and np.allclose(za_, pool_z0_with(PA_, zB_, "", 4.0)[0]), "a defined transform: the draw-0 form"
    mains_ = [f"m{i}" for i in range(12)]
    env_ = {m: ("a" if i < 5 else "b") for i, m in enumerate(mains_)}
    u1_, W1_ = draws_main_port(mains_, {m: (env_[m],) for m in mains_}, [1, 2], D=50)
    _, W2_ = draws_main_port(list(reversed(mains_)), {m: (env_[m],) for m in mains_}, [1, 2], D=50)
    assert u1_ == sorted(mains_) and np.array_equal(W1_, W2_) and np.all(W1_[0] == 1), "draws: sorted units, draw 0 all ones, deterministic"
    ia_ = [i for i, m in enumerate(u1_) if env_[m] == "a"]
    assert np.all(W1_[1:, ia_].sum(1) == len(ia_)) and np.all(W1_[1:].sum(1) == len(mains_)), "each stratum keeps its size in every draw"
    assert not np.array_equal(W1_, draws_main_port(mains_, {m: (env_[m],) for m in mains_}, [1, 3], D=50)[1]), "another seed, other draws"
    uc_ = {m: i for i, m in enumerate(u1_)} | {"zz": 12}
    Wx_ = np.hstack([W1_, np.ones((51, 1))])
    assert draws_hash(uc_, Wx_) == draws_hash(dict(reversed(list(uc_.items()))), Wx_.copy()) and draws_hash(uc_, Wx_) != draws_hash(uc_ | {"zz": 11}, Wx_), \
        "draws hash: the column map by content, order-free"
    ids_ = boot_ids()
    assert len(ids_) == BOOT_N == len(set(ids_)) and ids_ == sorted(ids_) and min(ids_) >= 1 and max(ids_) <= BOOT_D and boot_ids() == ids_, "seeded ids"
    zc_ = np.array([2.0, np.inf, np.nan, -np.inf, 0.5])
    Wr_ = np.array([2.0, 1.0, 0.0, 3.0])
    cc_, wc_ = np.array([0, 1, 1, 2, 3]), np.array([1.0, 0.5, 0.5, 1.0, 1.0])
    want_ = (2.0 * 1 + 1.0 * 0.5 * 1 + 0 + 0 + 3.0 * 0) / (2.0 + 0.5 + 0.5 + 0.0 + 3.0)
    assert boot_catch(zc_, 1.0, Wr_, cc_, wc_) == want_ and np.isnan(boot_catch(zc_, np.nan, Wr_, cc_, wc_)), "catch: z > T, +inf in, NaN out"
    assert np.isnan(boot_catch(np.array([]), 1.0, Wr_, np.array([], dtype=int), np.array([]))), "an empty cohort: 0 / 0"
    per_ = {"a": {"x": 0.5}, "b": {"x": np.nan}, "c": {"x": 0.25}}
    assert boot_comb(per_, {"a": 2.0, "b": 0.0, "c": 2.0}, ["a", "b", "c"], "x") == 0.375, "a member with D 0 adds nothing (its NaN dropped)"
    assert np.isnan(boot_comb(per_, {"a": 0.0, "b": 0.0, "c": 0.0}, ["a", "b", "c"], "x")), "every D 0: NaN"
    assert npzkey("sol_a_uncut C", "pooled_all_origins", "joint", "0.01", "S_new (all-origins draws)", "catch_sim@S|primary") == \
        "sol_a_uncut_C__pooled_all_origins__joint__0.01__S_new_all-origins_draws__catch_sim_at_S_primary", "npz key v2"
    assert npzkey("FIBS|uncut R", "paper_cells", "fixed", "0.05", "pooled", "gap_run2_minus_run1_at_tauB") == \
        "FIBS_uncut_R__paper_cells__fixed__0.05__pooled__gap_run2_minus_run1_at_tauB" and _stops(lambda: npz_part("a__b")), "npz key v2 / ambiguous part"
    lbs_ = [f"{fa} {ro}" for _, fs in BOOT_GROUPS for fa in fs for ro in "CR"]
    bv_ = {lb: {"cb": {md: {"0.01": {str(b): {"df": 0.1 * (i % 3), "yr": 0.01, "df_ref": 0.1 * (i % 3), "yr_ref": 0.01} for b in (0, 7)}}
                       for md in ("joint", "fixed")}} for i, lb in enumerate(sorted(set(lbs_)))}
    ab_ = agg_boot(bv_, ["cb"], [0.01], [0, 7])
    assert len(ab_) == 2 * 1 * 2 * len(BOOT_GROUPS) * 2 * 2 and all(eq(m_, r_) for _, m_, r_ in ab_), "S / Y per draw: every group, both sides"
    l0_ = min(set(lbs_))
    bv_[l0_]["cb"]["joint"]["0.01"]["7"]["df"] += 0.5
    assert {k_ for k_, m_, r_ in ab_ if not eq(m_, r_)} == set() and any(not eq(m_, r_) for _, m_, r_ in agg_boot(bv_, ["cb"], [0.01], [0, 7])), "a moved df"
    del bv_[l0_]
    assert any(m_ == "member values missing" for _, m_, _ in agg_boot(bv_, ["cb"], [0.01], [0, 7])), "a missing member never skipped"
    man_ = {"labels": ["L C"], "combo_rows": {"cb": ["o1", "o2"]}, "levels": [0.01], "boot": {"ids": [0, 7], "seed": 20261020}}
    rk_ = {"boot|draws_sha256 (the regenerated union stream = this row's attestation)"}
    pk_ = set()
    for md in ("joint", "fixed"):
        for b in (0, 7):
            pre = f"boot|{md}|0.01|draw {b}"
            rk_ |= {f"{pre}|catch|{x}" for x in BOOT_ROW_CATCH} | {f"{pre}|gap|{x}" for x in BOOT_ROW_GAP} | {f"{pre}|D"} | (
                {f"{pre}|self-check: per-draw path = draw-0 path|catch|{x}" for x in BOOT_ROW_CATCH} if b == 0 else set())
            pk_ |= {f"{pre}|pooled|gap|{x}" for x in BOOT_POOLED_GAP} | (
                {f"{pre}|pooled|self-check: per-draw path = draw-0 path|gap|{x}" for x in BOOT_POOLED_GAP} if b == 0 else set())
    cnt_ = {f"cb | 0.01 | {md}": {"pooled": 3, "rows": {"o1": 2, "o2": 2}} for md in ("joint", "fixed")}
    det_ = {"L C": {"rules": [{"row": "L C", "rule": "VALUE: x", "counts": cnt_, "boot": {"draw_ids": [0, 7], "seed": 20261020}}], "rows": [{"row": "L C", "origin": o, "ledger": [[k_, "MATCH"] for k_ in rk_]}
                                                                                       for o in ("o1", "o2")],
                    "pooled": [{"row": "L C", "combo": "cb", "ledger": [[k_, "MATCH"] for k_ in pk_]}]}}
    assert value_domain(man_, det_) == [], ("a complete domain", value_domain(man_, det_)[:3])
    for bx_ in ({"draw_ids": [0, 7], "seed": BOOT_SEED}, {"draw_ids": [0, 8], "seed": 20261020}, None):  # Codex V3 r2 #1: the old seed, other ids, no block
        dx_ = json.loads(json.dumps(det_))
        dx_["L C"]["rules"][0]["boot"] = bx_
        assert value_domain(man_, dx_) == ["L C: the worker's sampled draw ids or seed differ from the manifest's"], ("NEGATIVE: a worker's boot declaration", bx_)
    det_["L C"]["rows"][1]["ledger"] = det_["L C"]["rows"][1]["ledger"][1:]
    cnt_["cb | 0.01 | fixed"]["rows"]["o2"] = 0
    assert len(value_domain(man_, det_)) >= 2 and value_domain(man_ | {"labels": ["L C", "M C"]}, det_)[-1] == "M C: no detail shard", "missing entries"
    det_["L C"]["pooled"] = []  # an early-exit combo: no pooled block at all
    assert any("pooled per-draw keys missing" in x_ for x_ in value_domain(man_, det_)), "an early-exit combo fails the domain"
    n_ = {("row", "o", "0.01|catch|run1@B"): (0.2, 0.2, "MATCH"), ("row", "o", "0.01|catch|sim@B"): (0.3, 0.3, "MATCH"),
          ("row", "o", "boot|x|self-check"): (0.2, 0.2, "MATCH")}
    p_ = {("row", "o", "0.01|catch|run1@B"): (0.25, 0.2, "MISMATCH"), ("row", "o", "0.01|catch|sim@B"): (0.3, 0.3, "MATCH"),
          ("row", "o", "boot|x|self-check"): (0.25, 0.25, "MATCH")}
    nd_ = {"row catch": lambda k_: k_[2].endswith("|catch|run1@B")}
    assert plant_verdict(n_, p_, nd_)["pass"], "a plant flagging exactly its moved key"
    assert not plant_verdict(n_, p_, nd_ | {"panel": lambda k_: "panel" in k_[2]})["pass"], "a required category missing"
    p2_ = dict(p_) | {("row", "o", "0.01|catch|sim@B"): (0.3, 0.3, "MISMATCH")}
    assert not plant_verdict(n_, p2_, nd_)["pass"], "an unexpected flagged key"
    p3_ = dict(p_) | {("row", "o", "boot|x|self-check"): (0.25, 0.3, "MISMATCH")}
    assert not plant_verdict(n_, p3_, nd_)["pass"], "a self-check (both sides moved) that no longer matches"
    p4_ = dict(p_) | {("row", "o", "0.01|catch|run1@B"): (0.2 + 1e-12, 0.2, "MATCH")}
    assert not plant_verdict(n_, p4_, nd_)["pass"], "a move within the tolerance is no move (an empty expected set fails)"
    assert not plant_verdict(n_, {k_: v_ for k_, v_ in p_.items() if "sim@B" not in k_[2]}, nd_)["pass"], "keys in one call only"
    assert not plant_verdict(n_, {k_: n_[k_] for k_ in n_}, nd_)["pass"], "a no-op plant (nothing moved) fails"
    # the exit code (2e via 507-prep 10-17 00:11): a main mismatch exits non-zero; a working plant (its expected mismatches in its own shards) exits 0
    pv_ = [{"plant": {"label": lb_}, "pass": True, "fired": True} for lb_, _, _ in VALUE_PLANTS]
    ag0_ = {"all_equal": True, "mismatches": [], "positive_controls": [{"pass": True}], "value": {"pass": True, "domain_missing": [], "plants": pv_,
                                                                                                  "boot_aggregate": {"mismatches": []}}}
    ok_out_ = {"all_equal": True, "labels_all_equal": True, "value": True, "aggregates": ag0_}
    assert exit_verdict(ok_out_, []) == (0, []), "a clean value run with working plants exits 0"
    agw_ = {"all_equal": True, "mismatches": [], "positive_controls": [], "w2": {"gates": {"count_per_label_as_predicted": True, "every_label_all_equal": True,
                                                                                   "control_pass": True}, "control": {"pass": True}}}
    okw_ = {"all_equal": True, "labels_all_equal": True, "value": True, "aggregates": agw_}
    assert exit_verdict(okw_, []) == (0, []), "rc1b W2: a clean W2 run exits 0 (its gates replace the value block)"
    agw2_ = json.loads(json.dumps(agw_))
    agw2_["w2"]["gates"]["count_per_label_as_predicted"] = False
    agw3_ = json.loads(json.dumps(agw_))
    agw3_["w2"]["control"]["pass"] = False
    assert exit_verdict(okw_ | {"aggregates": agw2_}, [])[0] == 1 and "rc1b W2 control verdict failed" in exit_verdict(okw_ | {"aggregates": agw3_}, [])[1], \
        "NEGATIVE (W2): a failed gate or control exits 1"
    assert exit_verdict(ok_out_ | {"all_equal": False, "labels_all_equal": False}, [])[1] == ["main comparison not all equal"], "a main mismatch exits non-zero"
    agm_ = ag0_ | {"mismatches": [["k"]], "all_equal": False}
    assert exit_verdict(ok_out_ | {"all_equal": False, "aggregates": agm_}, [])[0] == 1, "an aggregate mismatch exits non-zero"
    agp_ = ag0_ | {"value": ag0_["value"] | {"plants": [pv_[0] | {"pass": False}, pv_[1]]}, "all_equal": False}
    assert exit_verdict(ok_out_ | {"all_equal": False, "aggregates": agp_}, []) == (1, ["plant verdict failed: ['sol_a_uncut C']"]), "a failed plant exits non-zero"
    agf_ = ag0_ | {"value": ag0_["value"] | {"plants": [pv_[0], pv_[1] | {"fired": False}]}}
    assert exit_verdict(ok_out_ | {"aggregates": agf_}, [])[0] == 1 and exit_verdict(ok_out_ | {"aggregates": ag0_ | {"value": ag0_["value"] | {"plants": pv_[:1]}}}, [])[0] == 1, \
        "a plant that did not fire, or a missing plant, exits non-zero"
    assert exit_verdict(ok_out_, ["aggregate.json"])[0] == 1 and exit_verdict({"all_equal": True, "labels_all_equal": True, "value": True}, [])[0] == 1, \
        "a missing shard, or a value run without its value block, exits non-zero"
    assert exit_verdict({"all_equal": True, "labels_all_equal": True}, []) == (0, []) and exit_verdict({"all_equal": False, "labels_all_equal": True}, [])[0] == 1, \
        "a non-value run: exit 0 only when all equal"
    agd_ = ag0_ | {"value": ag0_["value"] | {"domain_missing": ["L | cb | 0.01 | joint: draw-0 comparisons missing"]}}
    agb_ = ag0_ | {"value": ag0_["value"] | {"boot_aggregate": {"mismatches": [["boot|k"]]}}}
    agn_ = ag0_ | {"value": ag0_["value"] | {"boot_aggregate": {"mismatches": [], "nan_on_both_sides_keys": ["boot|cb|joint|0.01|draw 7|g|C|S"]}}}
    assert all(exit_verdict(ok_out_ | {"aggregates": x_}, []) == (1, ["main comparison not all equal"]) for x_ in (agd_, agb_, agn_)), \
        "a missing domain entry, a seeded-draw S / Y mismatch or a seeded-draw NaN on both sides exits non-zero, even with all_equal True (Astra r6)"
    # Astra r6 1: matching NaNs are NOT a match that counts: every seeded-draw key and every draw-0 catch / gap / panel key with NaN on both sides
    nn_ = float("nan")
    vq_ = vacuous([("boot|joint|0.01|draw 7|catch|run1@B", nn_, nn_), ("boot|fixed|0.05|draw 9|pooled|gap|delta_full", nn_, float("nan")),
                   ("0.01|catch|run1@B", nn_, nn_), ("0.01|joint|panel|pooled|G", nn_, nn_), ("pooled_world|cutoffs_draw0|0.01|B|mu0", nn_, nn_),
                   ("boot|joint|0.01|draw 7|catch|sim@B", 0.5, 0.5), ("boot|joint|0.01|draw 7|D", nn_, 1.0)])
    assert vq_ == ["boot|joint|0.01|draw 7|catch|run1@B", "boot|fixed|0.05|draw 9|pooled|gap|delta_full", "0.01|catch|run1@B", "0.01|joint|panel|pooled|G"], \
        ("matching NaNs fail on seeded-draw and draw-0 value keys", vq_)
    import tempfile
    with tempfile.TemporaryDirectory() as td_:  # Astra r6 3: an incomplete reduction's file never blocks a later one
        d0_ = Path(td_) / "check_x_fromvolume.json"
        assert fresh_dest(d0_) == d0_
        d0_.open("x").write("{}")
        d1_ = fresh_dest(d0_)
        d1_.open("x").write("{}")
        assert d1_.name == "check_x_fromvolume_attempt2.json" and fresh_dest(d0_).name == "check_x_fromvolume_attempt3.json", "fresh reduction names"
    # Astra r6 4 / 2e: the plant END TO END through its real path (apply_plant: the source-score assignment, arm_runs_of rebuild, zrun_of), then the
    # catch at the pool's tau and the plant verdict on the resulting ledgers. Removing the assignment in apply_plant fails here (its own assert and the
    # moved-catch asserts below)
    for five_ in (True, False):
        fb_ = np.repeat(np.arange(5), 4)
        bb_ = rng_.normal(size=(20, 5))
        PB_ = (np.array([f"honest_b|t{i}" for i in range(20)]), bb_[np.arange(20), fb_] if five_ else bb_[:, 0], fb_ if five_ else np.zeros(20, int), bb_)
        ZB_ = pool_z0(PB_, five_)
        T_ = wthr0(ZB_[0], 0.05)
        cohp_ = ["c1", "c2", "c3"]
        ru_ = np.array(["run1|c1", "run1|c1", "run1|c2", "run1|c2", "run1|c3", "sim|c1"])
        fu_ = np.array([1, 1, 2, 2, 3, 0]) if five_ else np.zeros(6, int)
        su_ = np.array([-9.0, -8.5, -7.0, -9.5, -6.0, -9.0])  # far below every pool unit: nothing caught before the plant
        U_ = {"run": ru_, "score": su_, "fold": fu_, "sbm": np.full((6, 5), -9.0), "source": np.array(["S_new"] * 6)}
        att_ = np.array([True] * 5 + [False])
        k0_, rm0_, rf0_, rr0_, _ = arm_runs_of("run1", U_, att_, fu_, cohp_, five_, set())
        z0p_ = zrun_of(rm0_, rf0_, rr0_, ZB_[1], five_)
        U2_, built_, info_ = apply_plant("run1", U_, att_, fu_, cohp_, five_, set(), ZB_)
        z1p_ = zrun_of(built_[1], built_[2], built_[3], ZB_[1], five_)
        assert np.array_equal(U_["score"], su_) and int((U2_["score"] != su_).sum()) == 1, "exactly one SOURCE score moved (on a copy)"
        i0p_ = int(np.argmin(z0p_))  # the lowest draw-0 z run (all three are finite and mapped here)
        assert info_["run"] == f"run1|{cohp_[i0p_]}" and not (z0p_ > T_).any() and list(z1p_ > T_) == [i_ == i0p_ for i_ in range(3)], \
            ("the lowest-z run, and only it, is now caught", five_)
        c0p_ = boot_catch(z0p_, T_, np.ones(3), np.arange(3), np.ones(3))
        c1p_ = boot_catch(z1p_, T_, np.ones(3), np.arange(3), np.ones(3))
        nled_ = {("row", "o", k_): v_ for k_, v_ in ((x_[0], (x_[1], x_[2], x_[3])) for x_ in compare([("0.05|catch|run1@B", c0p_, c0p_)], False)["ledger_values"])}
        pled_ = {("row", "o", k_): v_ for k_, v_ in ((x_[0], (x_[1], x_[2], x_[3])) for x_ in compare([("0.05|catch|run1@B", c1p_, c0p_)], False)["ledger_values"])}
        assert c1p_ - c0p_ == 1 / 3 and plant_verdict(nled_, pled_, {"member catch": lambda k_: k_[2] == "0.05|catch|run1@B"})["pass"], \
            "the planted catch moves by one run and the verdict flags exactly it"
    i1_ = identity_parts({"a": 1, "pw": {"x": 1}, "by_label": {"L": {"f": 1}}, "variants_by_label": {"L": {"o": {"c": [1, 2]}}}})
    i2_ = identity_parts({"a": 1, "pw": {"x": 1}, "by_label": {"L": {"f": 1}}, "variants_by_label": {"L": {"o": {"c": [1, 3]}}}})
    assert sorted(i1_) == ["by_label|L|sha256", "pw|x|sha256", "top|a|sha256", "variant|L|o|sha256"] and \
        [k_ for k_ in i1_ if i1_[k_] != i2_[k_]] == ["variant|L|o|sha256"], "identity parts"
    return ("G7b VALUE: weighted tau = the weight-expanded order statistic (ties, zero weights, k < 1); pool z / moments / 21b at integer draw weights = "
            "the repeated pool's; the 21b skip, an undefined fold -1 transform and a light fold invalidate as the runner; H_A's bad / unused fold -1 "
            "units; the union-stream port (sorted, stratum sizes, seeded) and its hash; the seeded ids; catch z > T (+inf in, NaN out, empty NaN); the "
            "D-weighted pooling with D 0 members; npz keys; per-draw S / Y with a missing member never skipped; the pre-built domain (missing keys, a "
            "zero-count row, an early-exit combo); the plant verdict (exact set, categories, self-checks, no-op); the exit code (a main or aggregate "
            "mismatch, a missing domain entry, a seeded-draw S / Y mismatch or NaN, a failed / unfired / missing plant, a missing shard exit non-zero; a "
            "clean run with working plants exits 0); matching NaNs fail on seeded-draw and draw-0 value keys; fresh reduction names; the plant end to "
            "end through apply_plant (one source score moved on a copy, the lowest-z run caught at tau, the catch moves by one run, the verdict "
            "flags it), five-model and single-model; identity parts")


def selftest_routec():
    """G7b on route C (09's conditions 3 and 4, 10-20): the record-name modes (route C's prefix only; every other name's mode unchanged), the
    pilot zero-unit fail-closed check (one negative fixture per way it fires), and the --routec-zero-unit argument from meta."""
    cases_ = {"alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-16_211215.json.gz": "alltasks", "alltasks_CRD_r2_allorigins_2026-10-16_044756.json":
              "alltasks", "STUB_alltasks_CRD_FULL_x.json": "alltasks", "sweep_final_x.json": "sweep", "sweep_union_sol_x.json": "union_sol",
              "pooled455v2_x.json": "pooled", "pooled455v2_ownallLOO_x.json": "pooled", "runlen_v2_x.json": "pooled", "routec_PLANT_alltasks_CRD_FULL_plant_x.json": "alltasks", "routec_C_alltasks_CRD_x.json": "pooled",
              "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz": "alltasks", "routec_B_alltasks_CRD_FULL_r2_allorigins_sweep15_x.json.gz":
              "alltasks", "routec_R37_alltasks_CRD_FULL_r2_allorigins_rule37_x.json.gz": "alltasks", "routec_A_routec_B_alltasks_CRD_x.json": "pooled"}
    assert {f_: record_mode(f_)[1] for f_ in cases_} == cases_, "record modes by name (09 cond. 4)"
    assert record_mode("routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz")[0] == \
        "alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json" and record_mode("alltasks_CRD_x.json.gz")[0] == "alltasks_CRD_x.json", \
        "route C's name is the Volume record's; other names unchanged"
    rt_ = "simh_q4f_v1_pilot__sim_honest"
    ok_ = {"n": 20, "keys_sha256": "k", "post_filter_keys": ["simh_pilot|t1"], "dropped_by_filters": [], "zero_unit": [], "zero_unit_unlisted": []}
    rows_ = {"L C": {"S_old (all-origins draws) [pooled: paper_cells]": {"s_old_simh_extra": {rt_: dict(ok_)}}, "S_new (all-origins draws) [pooled: paper_cells]": {}}}
    cf_ = {"simh_old": {"L C": [{"root": "r0", "path": "p"}, {"root": rt_, "s_old_only": True, "admitted": ["simh_pilot|t1"], "zero_unit_ok": []}]}}
    assert pilot_zero_unit_viol(rows_, [cf_], [rt_], require=True) == [], "a clean route C record and job"
    assert pilot_zero_unit_viol({}, [{}], [], require=True) == [], "a record without extra roots (the 171-pool A)"
    bad_ = json.loads(json.dumps(rows_))
    bad_["L C"]["S_old (all-origins draws) [pooled: paper_cells]"]["s_old_simh_extra"][rt_]["zero_unit"] = ["simh_pilot|t1"]
    assert pilot_zero_unit_viol(bad_, [cf_], [rt_]) == [f"row zero_unit not an empty list | L C | S_old (all-origins draws) [pooled: paper_cells] | {rt_}"], \
        "NEGATIVE FIXTURE: a record naming one zero-unit pilot key STOPs (09 cond. 3)"
    del bad_["L C"]["S_old (all-origins draws) [pooled: paper_cells]"]["s_old_simh_extra"][rt_]["zero_unit_unlisted"]
    assert len(pilot_zero_unit_viol(bad_, [cf_], [rt_])) == 2, "a missing unlisted count fails closed too"
    assert pilot_zero_unit_viol({"L C": {"S_old (all-origins draws) [pooled: paper_cells]": {}}}, [cf_], [rt_], require=True) == \
        ["row without s_old_simh_extra for every extra root | L C | S_old (all-origins draws) [pooled: paper_cells]"], "a scored S_old row without its entry"
    cfz_ = json.loads(json.dumps(cf_))
    cfz_["simh_old"]["L C"][1]["zero_unit_ok"] = ["simh_pilot|t1"]
    assert pilot_zero_unit_viol(rows_, [cfz_], [rt_]) == [f"job zero_unit_ok not an empty list | job 0 | L C | {rt_}"], "a job admitting a zero-unit key"
    fl_ = {"s_old_simh_extra_roots": {rt_: {"path": "p.json", "zero_unit_list": {"path": "results/z.json", "sha256": "ab"}}}}
    assert routec_zero_unit_arg(fl_) == "results/z.json@ab" and routec_zero_unit_arg({}) == "" and routec_zero_unit_arg(None) == "", "--routec-zero-unit"
    assert all(re.fullmatch(r"[0-9a-f]{64}", k_) and re.fullmatch(r"[0-9a-f]{64}", v_[2]) and v_[0] == VOLS["solavg"] for k_, v_ in PERDRAW_BY_RECORD.items()) \
        and len({v_[1] for v_ in PERDRAW_BY_RECORD.values()}) == len(PERDRAW_BY_RECORD) == 4 and set(ROUTEC_LEVEL_SPECS) < set(PERDRAW_BY_RECORD), \
        "the per-record npz pins and level specs (record shas, npz shas, distinct paths)"
    # GATES A22 @ f5441bb2c (09's seven fixtures): levels_allowance_viol on B-shaped and R37-shaped sets; tie_parts_viol for every other part
    b15_ = sorted({f for f in sorted(set([10 ** (-4 + 0.125 * i) for i in range(25)]) | {0.05}) if 1e-3 * (1 - 1e-9) <= f <= 0.05 * (1 + 1e-9)} | {0.05})
    r3_ = [1e-4, 2e-4, 5e-4]
    assert len(b15_) == 15 and levels_allowance_viol(b15_, b15_, list(reversed(b15_))) == [] and levels_allowance_viol(r3_, r3_, r3_) == [], \
        "the right B and R37 sets pass"
    assert levels_allowance_viol([0.01, 0.05], b15_, b15_) and levels_allowance_viol(b15_[:-1], b15_, b15_) and \
        levels_allowance_viol(b15_ + [0.1], b15_, b15_) and levels_allowance_viol([1e-4, 2e-4, 1e-3], r3_, r3_), \
        "NEGATIVE FIXTURES: A's set on B, a missing level, an extra level and a wrong set each STOP"
    assert levels_allowance_viol(r3_, r3_, [1e-4, 2e-4])[0].startswith("spec and record levels disagree") and \
        levels_allowance_viol(r3_ + [2e-4], r3_, r3_), "NEGATIVE FIXTURES: spec and record levels disagree STOPs; a repeated level STOPs"
    assert tie_parts_viol(["top|boot|sha256", "pw|git|sha256"], VALUE_ID_ALLOWED) == [] and \
        tie_parts_viol(["top|boot|sha256", "top|labels|sha256"], VALUE_ID_ALLOWED) == ["top|labels|sha256"] and \
        "top|levels|sha256" not in VALUE_ID_ALLOWED and "pw|tree_sha|sha256" not in VALUE_ID_ALLOWED, \
        "NEGATIVE FIXTURE: any other identity part differing STOPs (levels and tree_sha pass only through A22 / A20)"
    # B's level split (A22): every n-th level from the k-th; B's 4 parts partition its 15 levels (no overlap, no gap); A's 2 parts are 1 level each
    l15_ = list(range(1, 16))
    pt_ = [level_part_levels(l15_, f"{k_}/4") for k_ in range(1, 5)]
    assert [len(p_) for p_ in pt_] == [4, 4, 4, 3] and sorted(x_ for p_ in pt_ for x_ in p_) == l15_, "B's 4 parts partition the 15 levels"
    assert level_part_levels([0.05, 0.01], "1/2") == [0.01] and level_part_levels([0.01, 0.05], "2/2") == [0.05], "A's two 1-level parts"
    assert all(_stops(lambda p0_=p0_: level_part_levels(l15_, p0_)) for p0_ in ("0/4", "5/4", "x", "1/0", "")) and _stops(lambda: level_part_levels([1], "2/2")), \
        "NEGATIVE FIXTURES: a bad k/n or a part without a level STOPs"
    # Stage 2 value runs (0d / 09, 10-20): the stage from the fold labels; Stage 2's plants are one per Stage 2 reader family, on Stage 1's row and arm;
    # Stage 2's tie STOPs until pinned; its as-recorded set is exactly the Stage 1 fold labels (reproduced at S3)
    assert value_stage(["own_all_rs_uncut_L48", "sol_a_uncut"]) == "1" and value_stage(list(reversed(FOLD_STAGES["2"]))) == "2", "value_stage"
    assert all(_stops(lambda f0_=f0_: value_stage(f0_)) for f0_ in (["sol_a_uncut"], ["c_asis_uncut", "sol_a_uncut"], [])), "NEGATIVE: not one stage's families"
    assert [x_[0].rsplit(" ", 1)[0] for x_ in VALUE_PLANTS_S2] == ["c_asis_uncut", "own_all_rs25_uncut_L48"] and \
        {x_[1:] for x_ in VALUE_PLANTS_S2} == {x_[1:] for x_ in VALUE_PLANTS}, "Stage 2 plants: P1 / P2 on Stage 1's row and arm (09)"
    assert value_pins("1")[3] == VALUE_PLANTS and (STAGE2_TIE is not None or _stops(lambda: value_pins("2"))), "Stage 2's tie STOPs until pinned"
    assert sorted(f_.rsplit(" ", 1)[0] for f_ in STAGE2_AS_RECORDED) == sorted(2 * FOLD_STAGES["1"]) and \
        {f_.rsplit(" ", 1)[1] for f_ in STAGE2_AS_RECORDED} == {"C", "R"}, "Stage 2's as-recorded set: the Stage 1 fold labels, C and R"
    # 09 R1 / R2 (Stage 2-shaped): no Stage 2 family is in an aggregate group; the Stage 2 plant check per part with no aggregate key moved; the
    # seeded-draw aggregate gate; the draw-0 aggregate path filled by the as-recorded labels (no "member neither reproduced nor in the stated gap")
    assert stage2_in_groups() == [], "a Stage 2 family is in an aggregate group"
    o2_, c2_ = "S_new (all-origins draws) [pooled: pooled_all_origins]", "pooled_all_origins"
    # option (A) (09, 10-20 03:5x), review R-1 / F-3 / F-4, and 09's POST HOC (a) / (c) (10-20 ~04:5x): two levels (0.01, 0.05); the certain subset
    # is the not-yet-caught levels (from the unplanted call's pre-plant record) x {joint, fixed}; the scope is the planted row and its combo's
    # pooled keys; the other origin's row, the other combo and the aggregate key are out of scope; run2@B is in scope and does not move
    K_ = {"c1": ("row", o2_, "0.01|catch|run1@B"), "c2": ("row", o2_, "0.05|catch|run1@B"), "f1": ("row", o2_, "0.01|fixed|catch|run1@B"),
          "f2": ("row", o2_, "0.05|fixed|catch|run1@B"), "pd": ("pooled", c2_, "0.01|joint|catch|run1"),
          "rs": ("row", o2_, "boot|joint|0.01|draw 7|catch|run1@B"), "ps": ("pooled", c2_, "boot|joint|0.01|draw 7|pooled|gap|delta_full"),
          "ag": ("agg", f"{c2_}|joint|0.01|coh|headline_six_full|C|S"), "oo": ("row", "S_old (all-origins draws) [pooled: pooled_all_origins]", "0.01|catch|run1@B"),
          "oc": ("pooled", "paper_cells", "0.01|joint|catch|run1"), "r2": ("row", o2_, "0.01|catch|run2@B"), "sa": ("row", o2_, "0.01|gap|same_attack")}
    MV_ = ("c1", "c2", "f1", "f2", "pd", "rs", "ps")  # the keys the plant moves in the clean fixture
    n2_ = {k_: (0.25, 0.25, "MATCH") for k_ in K_.values()}
    p2_ = {K_[x_]: ((0.75, 0.25, "MISMATCH") if x_ in MV_ else (0.25, 0.25, "MATCH")) for x_ in K_}
    p2_[K_["sa"]] = (0.25 + 2.0 ** -54, 0.25, "MATCH")  # 09 (a): same_attack = run1@A - run1@B under a plant that shifts both terms: one ulp, sub-floor
    need2_ = stage2_need(o2_)
    PRE0 = {"0.01": False, "0.05": False}
    s2p_ = lambda p_, lv_=(0.01, 0.05), pre_=PRE0: stage2_plant(n2_, p_, plant_verdict(n2_, p_, need2_), o2_, "run1", list(lv_), pre_)  # noqa: E731
    pv2_ = s2p_(p2_)
    assert pv2_["pass"] and pv2_["parts"] == {"draw0": {"row": 4, "pooled": 1}, "seeded": {"row": 1, "pooled": 1}} and \
        pv2_["certain"] == {"levels_match": True, "not_yet_caught_levels": ["0.01", "0.05"], "excluded_levels": {}, "size": 4, "present": 4, "flagged": 4,
                            "not_flagged": [], "ineffective_plant": False} and pv2_["value_moved_plain"] == pv2_["flagged"] == 7 and \
        pv2_["subfloor"]["n"] == 1 and pv2_["subfloor"]["families"] == ["gap|same_attack"], pv2_
    p05_ = dict(p2_) | {K_["c2"]: (0.25, 0.25, "MATCH"), K_["f2"]: (0.25, 0.25, "MATCH")}  # 0.05 already caught: its run1@B catch cannot move
    pv5_ = s2p_(p05_, pre_={"0.01": False, "0.05": True})
    assert pv5_["pass"] and pv5_["certain"]["excluded_levels"] == {"0.05": "already caught at draw 0"} and pv5_["certain"]["size"] == 2, pv5_
    assert s2p_(p05_, pre_={"0.01": False, "0.05": None})["certain"]["excluded_levels"] == {"0.05": "no finite tau_B at draw 0"}, "a level with no tau_B"
    S2NEG = {"an aggregate key moved and flagged (out of scope)": (dict(p2_) | {K_["ag"]: (0.75, 0.25, "MISMATCH")},),
             "another origin's row moved and flagged (out of scope)": (dict(p2_) | {K_["oo"]: (0.75, 0.25, "MISMATCH")},),
             "another combo's pooled key moved and flagged (out of scope)": (dict(p2_) | {K_["oc"]: (0.75, 0.25, "MISMATCH")},),
             "an out-of-scope key with both sides moved, unflagged (F-3: either side)": (dict(p2_) | {K_["oo"]: (0.75, 0.75, "MATCH")},),
             "a not-yet-caught level's joint key not flagged (09 (c))": (dict(p2_) | {K_["c2"]: (0.25, 0.25, "MATCH")},),
             "a not-yet-caught level's fixed key not flagged (R-1, 09 (c))": (dict(p2_) | {K_["f2"]: (0.25, 0.25, "MATCH")},),
             "zero not-yet-caught levels (an ineffective plant; 09 (c))": (p2_, (0.01, 0.05), {"0.01": True, "0.05": True}),
             "the pre-plant record's levels are not the run's (by identity; F-4)": (p2_, (0.01, 0.05), {"0.01": False}),
             "nothing moved in the seeded draws": (dict(p2_) | {K_["rs"]: (0.25, 0.25, "MATCH"), K_["ps"]: (0.25, 0.25, "MATCH")},),
             "no row key flagged in the seeded draws": (dict(p2_) | {K_["rs"]: (0.25, 0.25, "MATCH")},),
             "the compare path under-flags a moved key": (dict(p2_) | {K_["pd"]: (0.75, 0.25, "MATCH")},),
             "a 1e-10 move inside the compare tolerance, not flagged (plain cross-check; 09 (a))": (dict(p2_) | {K_["r2"]: (0.25 + 1e-10, 0.25, "MATCH")},)}
    for nm_, a_ in S2NEG.items():
        assert not s2p_(*a_)["pass"], f"NEGATIVE (Stage 2 plant): {nm_} did not STOP"
    assert plant_verdict(n2_, S2NEG["a 1e-10 move inside the compare tolerance, not flagged (plain cross-check; 09 (a))"][0], need2_)["pass"], \
        "the tolerance fixture passes Stage 1's tolerance-based verdict (so only the plain cross-check stops it)"
    assert plant_verdict(n2_, S2NEG["an out-of-scope key with both sides moved, unflagged (F-3: either side)"][0], need2_)["pass"], \
        "the either-side fixture passes Stage 1's verdict (so only the F-3 scope check stops it)"
    assert s2p_(*S2NEG["zero not-yet-caught levels (an ineffective plant; 09 (c))"])["certain"]["ineffective_plant"], "the ineffective-plant flag"
    nan_ = float("nan")
    assert plain_diff(0.25, 0.25 + 2.0 ** -54) == "subfloor" and plain_diff(0.5, 0.5 + 1e-11) == "moved" and plain_diff(0.5, 0.5 + 1e-10) == "moved" \
        and plain_diff(0.5, 0.5 + 5e-13) == "subfloor" and plain_diff(0.5, 0.5) == "same" and plain_diff([0.5, 1.0], [0.5 + 5e-13, 1.0]) == "subfloor", \
        "plain_diff: the 1e-12 floor (09 (a)): a one-ulp cancellation and 5e-13 are sub-floor; 1e-11 and 1e-10 are moves"
    assert not plain_moved(nan_, nan_) and plain_moved(nan_, 1.0) and plain_moved(1, 1.0) and not plain_moved([1.0, nan_], [1.0, nan_]) and \
        plain_moved({"a": 1.0}, {"b": 1.0}) and plain_moved([1.0], [1.0, 2.0]) and not plain_moved({"a": [nan_, 2.0]}, {"a": [nan_, 2.0]}) and \
        plain_moved(True, 1), "plain_moved: both-NaN unmoved; NaN vs a number, a type change, key sets and lengths moved"
    assert boot_gate_ok(False, 2016, "computed") and boot_gate_ok(True, 0, BOOT_OOS_S2) and not boot_gate_ok(False, 0, "computed") and \
        not boot_gate_ok(True, 0, "computed") and not boot_gate_ok(True, 5, BOOT_OOS_S2), "NEGATIVE: the seeded-draw aggregate gate (R2)"
    q2_ = {"gap": {"delta_full": {"point": 0.1}}, "catch": {"run1": {"point": 0.2}, "run2": {"point": 0.3}}, "readable": {"all": True},
           "panel": {"pooled": {"G": {"point": 0.1}, "Y": {"point": 0.1}, "saturated": False, "readable": True}}}
    rec2_ = {"pooled": {lb_: {c2_: {"joint": {"0.01": q2_}}} for lb_ in ("sol_a_uncut C", "c_asis_uncut C", "FIBS|uncut C")}}
    rw2_, us2_, mm2_ = agg_rows(rec2_, {"FIBS|uncut C": {c2_: {"0.01": {"df": 0.1}}}}, c2_, "joint", "0.01", {"sol_a_uncut", "FIBS|uncut"})
    assert mm2_ == [] and us2_ == ["sol_a_uncut C"] and sorted(rw2_) == ["FIBS|uncut C", "sol_a_uncut C"], \
        "the draw-0 aggregate path: an as-recorded member fills its row; a Stage 2 label is not a member"
    d5_ = "experiments/2026-10-15_fpr_sweep_507_prep/"  # option B (tree_sha): a clean pair, then one negative fixture per way it must STOP
    (b1_, s1b_), (b2_, s2b_) = ((k_, v_) for k_, v_ in TREE_SHA_DIFF.items())
    t1_ = {d5_ + "run_peraction311.py": "r", d5_ + "copies.py": "c", d5_ + "other.py": "o", b1_: s1b_[0], b2_: s2b_[0]}
    t2_ = dict(t1_) | {b1_: s1b_[1], b2_: s2b_[1]}
    pm_ = {d5_ + "run_peraction311.py", d5_ + "copies.py"}
    assert tree_sha_diff_viol(t1_, t2_, pm_, TREE_SHA_DIFF) == [], "exactly the two listed differences pass"
    assert tree_sha_diff_viol(t1_, dict(t2_) | {d5_ + "other.py": "o2"}, pm_, TREE_SHA_DIFF) == [f"unlisted difference | {d5_}other.py"], \
        "NEGATIVE FIXTURE: a third differing file STOPs"
    assert tree_sha_diff_viol(t1_, dict(t2_) | {d5_ + "copies.py": "c2"}, pm_, TREE_SHA_DIFF) == [f"pinned module differs or is missing | {d5_}copies.py"], \
        "NEGATIVE FIXTURE: a differing pinned module STOPs"
    assert tree_sha_diff_viol(t1_, dict(t2_) | {b1_: "x" * 64}, pm_, TREE_SHA_DIFF) == [f"unlisted difference | {b1_}", f"listed difference absent | {b1_}"], \
        "NEGATIVE FIXTURE: a listed file with another sha STOPs"
    assert tree_sha_diff_viol(t1_, dict(t1_), pm_, TREE_SHA_DIFF) == [f"listed difference absent | {b1_}", f"listed difference absent | {b2_}"], \
        "the listed differences must be present (exact, not an allow-list)"
    assert tree_sha_diff_viol(t1_, dict(t2_) | {d5_ + "new.py": "n"}, pm_, TREE_SHA_DIFF)[0] == "file names differ", "an added file STOPs"
    # PLANT V3 (09's grant, GATES @ 4e4ab4142 L596-598; fixtures (a)): the real pair with the 4 granted files passes; a 5th added file, a modified
    # pinned module, another sha for one of the 4, or a removed name each STOPs; an added name that is a pinned module STOPs; no grant, no additions
    ad_ = TREE_SHA_ADDED[PLANT_RECORD_SHA]
    assert len(ad_) == 4 and all(k_.startswith(d5_) for k_ in ad_), "the grant names exactly 4 runner-folder files"
    t3_ = dict(t2_) | dict(ad_)
    assert tree_sha_diff_viol(t1_, t3_, pm_, TREE_SHA_DIFF, ad_) == [], "PLANT: the 2 listed differences plus the 4 granted additions pass"
    assert tree_sha_diff_viol(t1_, dict(t3_) | {d5_ + "fifth.py": "f"}, pm_, TREE_SHA_DIFF, ad_) == ["file names differ"], "NEGATIVE: a 5th added file STOPs"
    assert tree_sha_diff_viol(t1_, dict(t3_) | {d5_ + "copies.py": "c2"}, pm_, TREE_SHA_DIFF, ad_) == [f"pinned module differs or is missing | {d5_}copies.py"], \
        "NEGATIVE: a modified pinned module STOPs (with the grant)"
    k0_ = sorted(ad_)[0]
    assert f"added file absent or with another sha | {k0_}" in tree_sha_diff_viol(t1_, dict(t3_) | {k0_: "x" * 64}, pm_, TREE_SHA_DIFF, ad_), \
        "NEGATIVE: another sha for a granted file STOPs"
    assert "file names differ" in tree_sha_diff_viol(t1_, {k_: v_ for k_, v_ in t3_.items() if k_ != d5_ + "other.py"}, pm_, TREE_SHA_DIFF, ad_), \
        "NEGATIVE: a removed name STOPs"
    assert f"added file is a pinned module | {k0_}" in tree_sha_diff_viol(t1_, t3_, pm_ | {k0_}, TREE_SHA_DIFF, ad_), "NEGATIVE: a granted file that is pinned STOPs"
    assert tree_sha_diff_viol(t1_, t3_, pm_, TREE_SHA_DIFF)[0] == "file names differ", "without the grant (any other record) the additions STOP"
    # PLANT V3 plant port (synthetic): the draw-0 solve, P1's shift, the solve comparison keys and P1's allowed set
    import numpy as np

    zt_ = np.array([0.1, 0.4, 0.9, 1.6, np.nan, 2.5])
    ht_ = lambda z, T=1.0: float(np.mean(np.asarray(z) > T))  # noqa: E731
    sv_, cx_ = plant_fit_v3(zt_, 1.0, ht_, [-12, -8, 0, 8, 60, -50])
    assert sv_["arms"]["+0"] == {"status": "ok", "c": 0.0, "achieved_points": 0.0} and cx_[0] == 0.0, "delta 0: c = 0"
    assert sv_["arms"]["+8"]["status"] == "ok" and sv_["arms"]["+8"]["achieved_points"] >= 8 - 1e-9 and cx_[8] > 0, "a positive delta reaches >= delta"
    assert sv_["arms"]["-12"]["status"] == "ok" and sv_["arms"]["-12"]["achieved_points"] <= -12 + 1e-9 and cx_[-12] < 0, "a negative delta reaches <= delta"
    assert sv_["arms"]["+60"] == {"status": "unattainable"} and 60 not in cx_ and sv_["arms"]["-50"] == {"status": "unattainable"}, "a ceiling and a floor: unattainable"
    assert plant_fit_v3(zt_, float("nan"), ht_, [3])[0]["status"].startswith("unavailable"), "a NaN tau: unavailable"
    sv2_, cx2_, h2_ = plant_p1_apply(sv_, cx_, zt_, 1.0, ht_, {"order": [-8, 8], "origin": "o", "level": "0.01"})
    assert h2_["fired"] and h2_["delta"] == -8 and cx2_[-8] != cx_[-8] and ht_(zt_ + cx2_[-8]) != ht_(zt_ + cx_[-8]), "P1 moves the draw-0 catch"
    assert sv2_["arms"]["-8"]["achieved_points"] == 100 * (ht_(zt_ + cx2_[-8]) - ht_(zt_)) and sv_["arms"]["-8"]["c"] == cx_[-8], "P1: achieved' by fc_; input intact"
    cmp_ = plant_solve_cmp("0.01", sv_, json.loads(json.dumps(sv_)))
    assert all(eq(a_, b_) for _, a_, b_ in cmp_) and any(k_ == "0.01|plant|solve_draw0|sim|arms|-8|c" for k_, _, _ in cmp_), "solve comparison keys"
    assert plant_solve_cmp("0.01", sv_, None) == [("0.01|plant|solve_draw0|sim|present", True, False)], "an absent solve record MISMATCHES"
    hit_ = {"origin": "o1", "level": "0.01", "delta": -8}
    assert plant_p1_allowed(("row", "o1", "0.01|plant|solve_draw0|sim|arms|-8|c"), hit_, "cb", "C") and \
        plant_p1_allowed(("row", "o1", "boot|joint|0.01|draw 17|plant|catch|sim_plant_-8"), hit_, "cb", "C") and \
        plant_p1_allowed(("pooled", "cb", "boot|joint|0.01|draw 0|pooled|plant|gap_delta_full_plant_-8"), hit_, "cb", "C") and \
        plant_p1_allowed(("agg", "plant|cb|joint|0.01|headline_six|C|-8|S"), hit_, "cb", "C"), "P1's allowed set: row, pooled and the co-headline S"
    assert not plant_p1_allowed(("row", "o2", "0.01|plant|catch|sim_plant_-8"), hit_, "cb", "C") and \
        not plant_p1_allowed(("row", "o1", "0.01|plant|catch|sim_plant_-12"), hit_, "cb", "C") and \
        not plant_p1_allowed(("pooled", "cb", "boot|joint|0.05|draw 0|pooled|plant|gap_delta_full_plant_-8"), hit_, "cb", "C") and \
        not plant_p1_allowed(("row", "o1", "0.01|catch|sim@S|primary"), hit_, "cb", "C") and \
        not plant_p1_allowed(("agg", "plant|cb|joint|0.01|headline_six|R|-8|S"), hit_, "cb", "C"), "NEGATIVE: keys outside P1's allowed set"
    bv_ = {f"L{i} {r_}": {"cb": {"joint": {"0.01": {"0": {"yr": 0.01 * i, "pl": {"-8": [0.1 * i, 0.1 * i]}, "pl_any": ["-8", "+3"]}}}}} for i in range(6) for r_ in "CR"}
    va_ = {lb: {"cb": {"0.01": {"rd": True}}} for lb in bv_}
    rq_ = {"coheadline_signed_plant": {"_rule": "x", "cb": {"joint": {"0.01": {f"headline_six|{r_}": {"-8": {"S": {"point": 0.25}, "Y": {"point": 0.025}, "readable": True,
                                                                                                          "rows": sorted(f"L{i} {r_}" for i in range(6))},
                                                                                                   "+3": dict(PLANT_COH_UNAV)}
                                                                          for r_ in "CR"}}}}}
    pcq_ = plant_coh_cmp(rq_, bv_, va_, sorted(bv_), [0.01], ["cb"])
    assert pcq_ and all(eq(a_, b_) for _, a_, b_ in pcq_), ("co-headline S / Y / rows / readable and the domain reproduced", [k_ for k_, a_, b_ in pcq_ if not eq(a_, b_)][:3])
    assert any(k_.endswith("|-8|S") for k_, _, _ in pcq_) and any(k_.endswith("|-8|readable") for k_, _, _ in pcq_) and any(k_.endswith("|+3|unavailable") for k_, _, _ in pcq_), \
        "S, readable and the exact unavailable entry compared"
    bad_ = lambda r0_, **kw: [k_ for k_, a_, b_ in plant_coh_cmp(r0_, kw.get("bv", bv_), kw.get("va", va_), sorted(bv_), [0.01], kw.get("cb", ["cb"])) if not eq(a_, b_)]  # noqa: E731
    r1_ = json.loads(json.dumps(rq_))
    r1_["coheadline_signed_plant"]["cb"]["joint"]["0.01"]["headline_six|C"]["-8"]["readable"] = False
    assert bad_(r1_) == ["plant|cb|joint|0.01|headline_six|C|-8|readable"], "NEGATIVE (Codex V3 #4): a flipped readable flag MISMATCHES"
    assert bad_(rq_, va=va_ | {"L0 C": {"cb": {"0.01": {"rd": False}}}}) == ["plant|cb|joint|0.01|headline_six|C|-8|readable"], "NEGATIVE: an unreadable member"
    assert bad_(rq_, cb=["cb", "cb2"]) and "plant|combos" in bad_(rq_, cb=["cb", "cb2"]), "NEGATIVE (Codex V3 #5): a combo missing from the record MISMATCHES"
    r2_ = json.loads(json.dumps(rq_))
    r2_["coheadline_signed_plant"]["cb"]["joint"]["0.01"]["headline_six|R"]["sim2_plant_+3"] = dict(PLANT_COH_UNAV)
    assert bad_(r2_) == ["plant|cb|joint|0.01|headline_six|R|delta keys"], "NEGATIVE (AMEND6): an extra (sim2) key in a group MISMATCHES"
    r3_ = json.loads(json.dumps(rq_))
    r3_["coheadline_signed_plant"]["cb"]["joint"]["0.01"]["headline_six|C"]["+3"] = {"unavailable": "other text"}
    assert bad_(r3_) == ["plant|cb|joint|0.01|headline_six|C|+3|unavailable"], "NEGATIVE: the unavailable entry compared exactly"
    # AMEND6 (Codex V3 #3): the IL report port: a folder without the probe's file is pending (its keys listed, optional ones out); with the file, STOP
    blk_ = {"window": "w", "sets": {"S_old": {"f2": {"keys": ["sim2|b", "run3|b"], "optional_pending_keys": ["run3|b"], "files": {}},
                                              "f1": {"keys": ["sim2|a"], "files": {"other.npz": {}}}}}}
    ilr_ = il_report_v3(blk_, "S_old", "mine.npz")
    assert ilr_["pending_keys"] == ["sim2|a", "sim2|b"] and ilr_["optional_pending_keys"] == ["run3|b"] and ilr_["rows"] == 0 and ilr_["folders_read"] == 0 \
        and ilr_["counts"] == {"admitted": 3, "routed": 2, "scored": 0, "rule": IL_COUNTS_RULE, "in_scope_not_built": 3} \
        and ilr_["cells_with_run"] == {"sim2": 0, "run3": 0} and il_report_v3(blk_, "S_new", "mine.npz")["pending_keys"] == [], "the IL report (pending family)"
    assert sorted(il_report_v3(blk_, "S_new", "mine.npz")) == sorted(set(ilr_) - {"optional_pending_keys"}), "an origin without folders: no optional_pending_keys"
    assert _stops(lambda: il_report_v3(blk_, "S_old", "other.npz")), "NEGATIVE: a folder holding the probe's IL file STOPs (not ported)"
    jb_ = json.loads(json.dumps(blk_))
    del jb_["sets"]["S_old"]["f1"]["files"]["other.npz"]  # apply_full: a 32k file without an uncut entry is dropped
    assert il_block_viol(jb_, blk_) == [] and il_report_v3(jb_, "S_old", "other.npz")["pending_keys"] == ["sim2|a", "sim2|b"], "a dropped file: pending"
    jb_["sets"]["S_old"]["f1"]["files"]["new.npz"] = {}
    jb_["sets"]["S_old"]["f2"]["keys"] = ["sim2|b"]
    assert il_block_viol(jb_, blk_) == ["S_old | f1: files outside the pinned block's", "S_old | f2: fields"], "NEGATIVE: an added file or a changed field"
    assert plant_fit_v3(np.full(4, np.nan), 1.0, None, [3, -3]) == ({"T0": 1.0, "status": "unavailable (draw-0 tau_sim not finite or no finite sim attack z)"}, {}), \
        "sim2 without an admitted run: the runner's unavailable record (no ht call)"
    assert plant_solve_cmp("0.01", {"T0": 1.0, "status": "u"}, {"T0": 1.0, "status": "u"}, "sim2")[0][0] == "0.01|plant|solve_draw0|sim2|fields", "sim2 solve keys"
    # Codex V3 #1: the worker's sampled ids and seed must be the manifest's; value_scope names the run's seed
    assert value_scope() == VALUE_SCOPE == value_scope(BOOT_SEED) and "(seed 20261020)" in value_scope(20261020) and "(seed 20261016)" not in value_scope(20261020), "scope seed"
    assert boot_ids(20261020) != boot_ids() and len(boot_ids(20261020)) == BOOT_N, "the PLANT seed draws another sample"
    # AMEND6: the plant job's withheld panel_groups (runner L4514): the exact marker passes with no PANEL comparison; another text MISMATCHES
    pg_ok_ = {"withheld": PLANT_PANEL_WITHHELD}
    assert pg_ok_ == {"withheld": "plant job: the plant labels only (the headline six); the final record's panel_groups are authoritative"}, "the marker text"
    # Codex V3 r2 #2 / AMEND7 G1: P1's categories at its own level; the pre-stated pair per delta sign
    gk_ = lambda lv_: ("pooled", "cb", f"boot|joint|{lv_}|draw 3|pooled|plant|gap_delta_full_plant_-8")  # noqa: E731
    ck_ = ("agg", "plant|cb|joint|0.01|headline_six|C|-8|S")
    assert p1_need({gk_("0.05"): None}, "o1", "0.01", "-8", "cb", "C")[1] == {"pooled planted gap": False, "coheadline_signed_plant S": False}, \
        "NEGATIVE: a pooled gap at the other level does not make the category required"
    nd_, ap_ = p1_need({gk_("0.01"): None, ck_: None}, "o1", "0.01", "-8", "cb", "C")
    assert ap_ == {"pooled planted gap": True, "coheadline_signed_plant S": True} and nd_["pooled planted gap"](gk_("0.01")) and not nd_["pooled planted gap"](gk_("0.05")), "P1 level"
    assert p1_cond_ok(-8, ap_) and not p1_cond_ok(-8, ap_ | {"coheadline_signed_plant S": False}) and not p1_cond_ok(-8, ap_ | {"pooled planted gap": False}), "G1 negative"
    assert p1_cond_ok(5, {"pooled planted gap": True, "coheadline_signed_plant S": False}) and p1_cond_ok(5, {"pooled planted gap": False, "coheadline_signed_plant S": False}) \
        and not p1_cond_ok(5, ap_) and not p1_cond_ok(0, ap_), "G1: a positive delta's coheadline S must not be compared; delta 0 STOPs"
    # Codex V3 r2 #3: every planted catch key; none in mode fixed
    pz_ = {-8: 0.1, 0: 0.2}
    assert all(eq(a_, b_) for _, a_, b_ in plant_catch_keys_cmp("0.01", pz_, {"sim_plant_-8": {}, "sim_plant_+0": {}, "sim@S|primary": {}}, {"sim@S|primary": {}})), "keys"
    assert [k_ for k_, a_, b_ in plant_catch_keys_cmp("0.01", pz_, {"sim_plant_-8": {}, "sim_plant_+0": {}, "sim2_plant_-3": {}}, {}) if not eq(a_, b_)] == \
        ["0.01|plant|catch|planted keys (joint)"], "NEGATIVE: a sim2_plant catch key MISMATCHES"
    assert [k_ for k_, a_, b_ in plant_catch_keys_cmp("0.01", pz_, {"sim_plant_-8": {}, "sim_plant_+0": {}}, {"sim_plant_-8": {}}) if not eq(a_, b_)] == \
        ["0.01|plant|catch|planted keys (fixed)"], "NEGATIVE: a planted key in mode fixed MISMATCHES"
    # Codex V3 r2 #4: the PLANT run is unsplit
    assert plant_level_part_viol(True, "1/2") and not plant_level_part_viol(True, "") and not plant_level_part_viol(False, "1/2"), "NEGATIVE: --level-part on PLANT"
    # AMEND7 G2: fixtures predicted to MISMATCH (a record sim2 solve with an arms map or another T0; an il_arms field; sim2 missing from the arms;
    # a catch_sim2_plant key in the npz)
    s2_ = {"T0": 1.0, "status": "unavailable (draw-0 tau_sim not finite or no finite sim attack z)"}
    assert all(eq(a_, b_) for _, a_, b_ in plant_solve_cmp("0.01", s2_, dict(s2_), "sim2")), "sim2 solve equal"
    assert any(not eq(a_, b_) for _, a_, b_ in plant_solve_cmp("0.01", s2_, s2_ | {"arms": {"+3": {"status": "ok", "c": 0.1, "achieved_points": 3.0}}}, "sim2")) \
        and any(not eq(a_, b_) for _, a_, b_ in plant_solve_cmp("0.01", s2_, s2_ | {"T0": 1.5}, "sim2")), "NEGATIVE: a sim2 solve with arms, or another T0"
    ir_ = il_report_v3(jb_, "S_old", "other.npz")
    assert all(eq(a_, b_) for _, a_, b_ in il_cmp(ir_, json.loads(json.dumps(ir_)))) and len(il_cmp(ir_, ir_)) == 2 + len(ir_), "il_arms equal"
    assert [k_ for k_, a_, b_ in il_cmp(ir_, ir_ | {"folders_read": 1}) if not eq(a_, b_)] == ["plant|il_arms|folders_read"], "NEGATIVE: one il_arms field"
    assert [k_ for k_, a_, b_ in il_cmp(ir_, None) if not eq(a_, b_)] == ["plant|il_arms|present"] and \
        [k_ for k_, a_, b_ in il_cmp(None, ir_) if not eq(a_, b_)] == ["plant|il_arms|present"], "NEGATIVE: a missing or an extra il_arms report"
    assert eq(*plant_arms_cmp("0.01", True, {"sim": {}, "sim2": {}})[1:]) and not eq(*plant_arms_cmp("0.01", True, {"sim": {}})[1:]) \
        and not eq(*plant_arms_cmp("0.01", False, {"sim": {}, "sim2": {}})[1:]), "NEGATIVE: sim2 missing from (or extra in) the arm set"
    kz_ = npzkey("L C", "cb", "joint", "0.01", aob("S_old (all-origins draws) [pooled: cb]"), "catch_sim2_plant_-3")
    assert sim2_npz_absent("L C", "cb", "0.01", ["S_old (all-origins draws) [pooled: cb]"], [-3, 0], ["x"]) and \
        not sim2_npz_absent("L C", "cb", "0.01", ["S_old (all-origins draws) [pooled: cb]"], [-3, 0], ["x", kz_]), "NEGATIVE: a catch_sim2_plant key in the npz"
    # AMEND7 G3: the PLANT G6 controls: [co-headline applicable with expected keys, PANEL not applicable]; anything else STOPs
    c1_, c0_ = {"applicable": True, "expected_keys": 4}, {"applicable": False, "expected_keys": 0}
    assert plant_ctl_ok([c1_, c0_]) and not plant_ctl_ok([c1_, c1_]) and not plant_ctl_ok([c0_, c0_]) and not plant_ctl_ok([c1_ | {"expected_keys": 0}, c0_]), "G3"
    # AMEND7 G5 (reviewer optional (b)): S1's built / checked labels; 17 built labels, or a label with 6 jobs, STOP
    b18_, c12_ = [f"B{i}" for i in range(18)], [f"B{i}" for i in range(12)]
    jk12_ = {(lb_, f"o{j}") for lb_ in c12_ for j in range(7)}
    assert s1_labels_viol(b18_, c12_, jk12_, 126, 84) == [], "S1 labels: 18 / 12, 126 / 84"
    assert s1_labels_viol(b18_[:17], c12_, jk12_, 126, 84) and s1_labels_viol(b18_, c12_, jk12_ - {("B0", "o0")}, 126, 84), "NEGATIVE: 17 built, or 6 jobs for a label"
    lmm_ = {"chk": "/wt/x/c.py", "rp": "/tree/d/rp.py", "np": "/venv/lib/site-packages/numpy/__init__.py", "ps": "/main/src/probe_shift/__init__.py"}
    assert all(not eq(v_, w2_value_plant(v_)) for v_ in (-1.0, -0.9995, 0.0, 1e-12, 1.0, -1e6, 1e6, 3.7)) and _stops(lambda: w2_value_plant(float("nan"))) \
        and _stops(lambda: w2_value_plant(None)), "rc1b W2 (AMEND3): the value plant flags at every finite v, -1 included; non-finite STOPs"
    fl_none_ = ""  # AMEND7: VAL_ as main computes it, bool(FL_) and ..., for a launch without --fold-labels
    assert not w2_value_path_ok("kappa", bool(fl_none_) and True) and not w2_value_path_ok("pertask", bool(fl_none_) and True) \
        and w2_value_path_ok("kappa", True) and w2_value_path_ok("pertask", True) and w2_value_path_ok(None, False), \
        "NEGATIVE (AMEND7): a W2 record without --fold-labels STOPs; a W2 record on the value path, or any other record, goes on"
    # rc1b W2a: the per-task port on a toy row (3 cells; one structural zero; one NaN z), its comparison, and the plants' verdict
    coh_t_, mo_t_ = ["c1", "c2", "c3"], {"c1": "m1", "c2": "m2", "c3": "m2"}
    zr_t_ = {"run1": np.array([1.5, 0.2, np.nan]), "run2": np.array([0.4, 2.0, 0.9]), "sim": np.array([3.0, -1.0, 0.0])}
    hz_t_ = {"B": np.array([0.1, 0.5, 1.0, np.nan]), "S": np.array([0.0, 2.5])}
    e_t_ = w2_pertask_entries("O", coh_t_, mo_t_, zr_t_, {"B": 1.0, "S": 1.0}, hz_t_, {"run2|c3"})
    assert e_t_[("c1", "run1")]["max_minus_tau"] == 0.5 and e_t_[("c1", "run1")]["caught"] is True and e_t_[("c1", "run1")]["honest_percentile"] == 1.0
    assert e_t_[("c2", "run1")]["caught"] is False and eq(e_t_[("c2", "run1")]["honest_percentile"], 1 / 3) and e_t_[("c3", "run1")]["caught"] is None
    assert e_t_[("c3", "run2")] == {"origin": "O", "cell": "c3", "main": "m2", "arm": "run2", "pipeline_pool": "B", "max_minus_tau": None,
                                    "honest_percentile": None, "caught": False, "structural_zero": True}, "a structural zero"
    assert e_t_[("c1", "sim")]["pipeline_pool"] == "S" and e_t_[("c1", "sim")]["honest_percentile"] == 1.0 and len(e_t_) == 9, "the port's entries"
    rl_t_ = [dict(v_) for v_ in e_t_.values()]
    c_t_, nv_t_ = w2_pertask_cmp(e_t_, rl_t_)
    assert all(eq(a_, b_) for _, a_, b_ in c_t_) and nv_t_ == 9 * 3 + 1, ("the per-task comparison and its count", nv_t_)
    rl2_, info_t_ = w2_record_plants(rl_t_, {"seed": 7})
    c2_t_, nv2_t_ = w2_pertask_cmp(e_t_, rl2_)
    flg_t_ = {k_ for k_, a_, b_ in c2_t_ if not eq(a_, b_)}
    want_t_ = {f"per_task|{a_}|{c_}|{f_}" for c_, a_, f_ in info_t_["planted"]} | {"per_task|entries (two-way)"}
    assert flg_t_ == want_t_ and nv2_t_ == nv_t_ - 3 and len(info_t_["planted"]) == 4, ("NEGATIVE (W2a record plants): exactly their keys", sorted(flg_t_ ^ want_t_))
    zs_t_, si_t_ = w2_score_plant(zr_t_, coh_t_, {"run2|c3"}, {"seed": 3}, lambda w, i: float(zr_t_[w][i]) + 1.0, {w: np.zeros(3, int) for w in zr_t_}, True)
    assert si_t_["moved"] and sum(int(not eq(float(zs_t_[w][i]), float(zr_t_[w][i]))) for w in zr_t_ for i in range(3) if np.isfinite(zr_t_[w][i])) == 1, "one z moved"
    # AMEND8: on an all-origins row the entries carry the canonical main (runner L1785 / L2706). An S_old-like row: its per-origin map is the
    # cell identity, the canonical map differs on every cell; the record side is built as the runner does (main = the canonical map)
    mid_t_ = {c: c for c in coh_t_}
    rcan_t_ = [dict(v_) for v_ in w2_pertask_entries("O", coh_t_, mo_t_, zr_t_, {"B": 1.0, "S": 1.0}, hz_t_, {"run2|c3"}).values()]
    new_t_ = w2_pertask_cmp(w2_pertask_entries("O", coh_t_, w2_pt_main_map(coh_t_, mid_t_, mo_t_, True), zr_t_, {"B": 1.0, "S": 1.0}, hz_t_, {"run2|c3"}), rcan_t_)[0]
    old_t_ = w2_pertask_cmp(w2_pertask_entries("O", coh_t_, mid_t_, zr_t_, {"B": 1.0, "S": 1.0}, hz_t_, {"run2|c3"}), rcan_t_)[0]  # dddac36e's call
    assert all(eq(a_, b_) for _, a_, b_ in new_t_), "AMEND8: the canonical main on an all-origins row: every comparison equal"
    assert {k_ for k_, a_, b_ in old_t_ if not eq(a_, b_)} == {k_ for k_, _, _ in old_t_ if k_.endswith("|identity")} and \
        len([k_ for k_, _, _ in old_t_ if k_.endswith("|identity")]) == 9, "NEGATIVE (AMEND8): the per-origin map flags exactly the 9 identity keys"
    assert w2_pt_main_map(coh_t_, mid_t_, mo_t_, False) == mid_t_ and _stops(lambda: w2_pt_main_map(coh_t_, mid_t_, None, True)), \
        "AMEND8: a non-all-origins row keeps its per-origin map; an all-origins row without the canonical map STOPs"
    ad_t_ = w2_added_check(e_t_, rl_t_)  # AMEND6 (Codex W2 r1 #4): the added entry on its own copy flags the two-way entry set only, count unchanged
    assert ad_t_["pass"] and ad_t_["flagged"] == ["per_task|entries (two-way)"] and ad_t_["count_unchanged"] and ad_t_["keys_equal"], "AMEND6: the added entry"
    assert not w2_added_check(e_t_, rl2_)["pass"], "NEGATIVE (AMEND6): where the entry set already differs, the add is not detected as the only change"
    pin_t_ = {"path": "p", "sha256": "a" * 64}  # the W2a control verdict on toy ledgers: O1 holds the score plant, O2 the record plants and the pin

    def led_t_(o_, mine_, rl_, pin_):
        return {"origin": o_, "ledger_values": compare(w2_pertask_cmp(mine_, rl_)[0] + [("per_task|pw_calib pin = A's D1 file", pin_t_, pin_)], False)["ledger_values"]}
    es_t_ = w2_pertask_entries("O", coh_t_, mo_t_, zs_t_, {"B": 1.0, "S": 1.0}, hz_t_, {"run2|c3"})
    nl_t_ = ledger_of({"rows": [led_t_("O1", e_t_, rl_t_, pin_t_), led_t_("O2", e_t_, rl_t_, pin_t_)], "pooled": []})
    pl_t_ = ledger_of({"rows": [led_t_("O1", es_t_, rl_t_, pin_t_), led_t_("O2", e_t_, rl2_, pin_t_ | {"sha256": "0" * 64})], "pooled": []})
    sv_t_ = {"plant_score": {"origin": "O1"}, "plant_record": {"origin": "O2"}}
    ct_t_ = {"score": si_t_, "record": info_t_ | {"added": ad_t_}, "pin": {"origin": "O2"}}
    assert w2_control_verdict_pertask(nl_t_, pl_t_, ct_t_, sv_t_, 2 * nv_t_, 2 * nv_t_ - 3)["pass"], "the W2a control verdict on toy ledgers"
    assert not w2_control_verdict_pertask(nl_t_, pl_t_, ct_t_ | {"record": info_t_ | {"added": ad_t_ | {"pass": False}}}, sv_t_, 2 * nv_t_, 2 * nv_t_ - 3)["pass"] \
        and not w2_control_verdict_pertask(nl_t_, pl_t_, ct_t_ | {"record": dict(info_t_)}, sv_t_, 2 * nv_t_, 2 * nv_t_ - 3)["pass"] \
        and not w2_control_verdict_pertask(nl_t_, pl_t_, ct_t_, sv_t_, 2 * nv_t_, 2 * nv_t_)["pass"] \
        and not w2_control_verdict_pertask(nl_t_, nl_t_, ct_t_, sv_t_, 2 * nv_t_, 2 * nv_t_ - 3)["pass"], \
        "NEGATIVE (W2a verdict): a failed or absent added check, an unchanged count, or no plant at all fails"
    # rc1b W2b: the margins on a toy draw (4 cells), the runner's rule; a flip moves po / p1 only; NaN thresholds give NaN
    wrow_t_, ccol_t_, wc_t_ = np.array([2.0, 1.0, 0.0]), np.array([0, 0, 1, 2]), np.array([0.5, 0.5, 1.0, 1.0])
    m_t_ = w2_margins(np.array([2.0, 0.0, 2.0, 2.0]), np.array([2.0, 2.0, 0.0, 0.0]), np.array([0.0, 0.0, 2.0, 2.0]), 1.0, 1.0, wrow_t_, ccol_t_, wc_t_)
    assert eq(m_t_["D"], 3.0) and eq(m_t_["run2|p1"], 2.0 / 3) and eq(m_t_["run2|q1"], 2.0 / 3) and eq(m_t_["run2|po"], 1.0 / 3) and eq(m_t_["sim|po"], 2.0 / 3) \
        and eq(m_t_["sim|q1"], 1.0 / 3), m_t_  # Wc = [1, 1, 1, 0]; hx = [T, F, T, T]; run 2 hy = [T, T, F, F]; sim hy = [F, F, T, T]
    mf_t_ = w2_margins(np.array([2.0, 0.0, 2.0, 2.0]), np.array([2.0, 2.0, 0.0, 0.0]), np.array([0.0, 0.0, 2.0, 2.0]), 1.0, 1.0, wrow_t_, ccol_t_, wc_t_, flip=1)
    assert {k_ for k_ in W2_KAPPA_KEYS if not eq(m_t_[k_], mf_t_[k_])} == {"run2|po", "run2|p1", "sim|po", "sim|p1"}, "NEGATIVE (W2b flip): po / p1 only"
    assert all(math.isnan(v_) for k_, v_ in w2_margins(np.zeros(4), np.zeros(4), np.zeros(4), float("nan"), 1.0, wrow_t_, ccol_t_, wc_t_).items() if k_ != "D")
    K_t_ = {md_: {k_: np.arange(BOOT_D + 1, dtype=np.float64) for k_ in W2_KAPPA_KEYS} for md_ in ("joint", "fixed")}
    K2_t_, sp_t_ = w2_store_plants(K_t_, [0, 5, 9], {"seed": 2})
    assert len(sp_t_["planted"]) == 14 and all(not eq(float(K2_t_[m_][k_][b_]), float(K_t_[m_][k_][b_])) for m_, b_, k_ in sp_t_["planted"]) and \
        sum(int(not eq(float(K2_t_[m_][k_][b_]), float(K_t_[m_][k_][b_]))) for m_ in K_t_ for k_ in W2_KAPPA_KEYS for b_ in (0, 5, 9)) == 14, "14 store plants"
    import pickle as pk_

    pb_t_ = pk_.dumps({"label": "L", "origin": "O"}, protocol=5)
    h_t_ = hashlib.sha256(pb_t_).hexdigest()
    fs_t_ = {f"d/0003_{h_t_[:16]}_abcdef12.pkl": pb_t_, f"d/0003_{'0' * 16}_abcdef11.pkl": b"partial"}
    assert w2_load_stored_row(list(fs_t_), fs_t_.__getitem__)[1] == f"d/0003_{h_t_[:16]}_abcdef12.pkl" and \
        _stops(lambda: w2_load_stored_row([f"d/0003_{'0' * 16}_abcdef11.pkl"], fs_t_.__getitem__)), "the runner's stored-row rule; no complete file STOPs"
    assert w2_load_stored_row(list(fs_t_), fs_t_.__getitem__)[3] == 1, "AMEND5: one complete attempt beside a partial file"
    # rc1b W2 AMEND5: the chosen attempt must be the record's own (resources and draws_sha256), with negatives
    rs_t_ = {"container_start_unix": 1.5, "elapsed_s": 2.25, "cpu_s": 0.1 + 0.2, "requested": {"cpu": [4, 4], "memory_mb": [16384, 24576], "timeout_s": 900}}
    rr_t_ = json.loads(json.dumps({"label": "L", "origin": "O", "resources": rs_t_, "draws_sha256": "a" * 64}, default=float))  # the record row, as written
    a2_t_ = {"label": "L", "origin": "O", "resources": dict(rs_t_), "draws_sha256": "a" * 64}  # the record's attempt
    w2_stored_tie(a2_t_, rr_t_)  # passes: the record's attempt
    assert _stops(lambda: w2_stored_tie(a2_t_ | {"resources": rs_t_ | {"container_start_unix": 0.5}}, rr_t_)), "NEGATIVE (AMEND5): mismatched resources STOP"
    assert _stops(lambda: w2_stored_tie(a2_t_ | {"draws_sha256": "b" * 64}, rr_t_)), "NEGATIVE (AMEND5): a mismatched draws_sha256 STOPs"
    assert _stops(lambda: w2_stored_tie({k_: v_ for k_, v_ in a2_t_.items() if k_ != "resources"}, rr_t_)) and \
        _stops(lambda: w2_stored_tie(a2_t_, {k_: v_ for k_, v_ in rr_t_.items() if k_ != "draws_sha256"})), "NEGATIVE (AMEND5): a missing tie field STOPs"
    b2_t_ = pk_.dumps(a2_t_, protocol=5)
    h2_t_ = hashlib.sha256(b2_t_).hexdigest()
    for n_t_ in range(4096):  # another complete attempt (other resources) that sorts FIRST by name: the runner's rule picks it, so the tie STOPs
        b1_t_ = pk_.dumps(a2_t_ | {"resources": rs_t_ | {"container_start_unix": 0.5}, "nonce": n_t_}, protocol=5)
        h1_t_ = hashlib.sha256(b1_t_).hexdigest()
        if h1_t_[:16] < h2_t_[:16]:
            break
    assert h1_t_[:16] < h2_t_[:16], "fixture: an attempt that sorts first"
    fs2_t_ = {f"d/0004_{h1_t_[:16]}_00000001.pkl": b1_t_, f"d/0004_{h2_t_[:16]}_00000002.pkl": b2_t_}
    ch_t_ = w2_load_stored_row(list(fs2_t_), fs2_t_.__getitem__)
    assert ch_t_[1] == f"d/0004_{h1_t_[:16]}_00000001.pkl" and ch_t_[3] == 2 and _stops(lambda: w2_stored_tie(ch_t_[0], rr_t_)), \
        "NEGATIVE (AMEND5): two complete attempts counted; the first by name has other resources, so the tie STOPs"
    assert w2_load_stored_row([f"d/0004_{h2_t_[:16]}_00000002.pkl"], fs2_t_.__getitem__)[0] == a2_t_, "the record's attempt alone loads"
    # the W2b control verdict on synthetic ledgers (O1: the flip; O2: the 14 store plants; O3: the dropped stored row), with negatives
    ids_t_ = list(range(0, 41, 2))  # 21 draws: draw 0 and 20 seeded ones

    def kk_t_(o_):
        return {("row", o_, f"kappa|{m_}|draw {b_}|{k_}") for m_ in ("joint", "fixed") for b_ in ids_t_ for k_ in W2_KAPPA_KEYS}
    nk_t_ = {k_: (0.0, 0.0, "MATCH") for o_ in ("O1", "O2", "O3") for k_ in kk_t_(o_) | {("row", o_, "kappa|stored rows (two-way)")}}
    fl_t_ = {"mode": "fixed", "draw": ids_t_[3], "eligible": True, "cell": "c"}
    stp_t_ = [[m_, ids_t_[j_], k_] for j_, (m_, k_) in enumerate((m_, k_) for m_ in ("joint", "fixed") for k_ in W2_KAPPA_KEYS)]
    pk_t_ = dict(nk_t_) | {("row", "O1", f"kappa|fixed|draw {ids_t_[3]}|{k_}"): (0.0, 1.0, "MISMATCH") for k_ in ("run2|po", "run2|p1", "sim|po", "sim|p1")}
    pk_t_ |= {("row", "O2", f"kappa|{m_}|draw {b_}|{k_}"): (0.0, 1.0, "MISMATCH") for m_, b_, k_ in stp_t_}
    pk_t_ = {k_: v_ for k_, v_ in pk_t_.items() if not (k_[1] == "O3" and k_[2] != "kappa|stored rows (two-way)")}
    pk_t_[("row", "O3", "kappa|stored rows (two-way)")] = (True, False, "MISMATCH")
    skv_t_ = {"plant_margin": {"origin": "O1"}, "plant_store": {"origin": "O2", "drop_origin": "O3"}}
    ck_t_ = {"flip": fl_t_, "store": {"planted": stp_t_, "dropped": "O3"}}
    assert len(kk_t_("O3")) == 294 and len(stp_t_) == 14 and w2_control_verdict_kappa(nk_t_, pk_t_, ck_t_, skv_t_, 3 * 294, 2 * 294)["pass"], \
        "the W2b control verdict on synthetic ledgers"
    q1_t_ = pk_t_ | {("row", "O1", f"kappa|fixed|draw {ids_t_[3]}|run2|q1"): (0.0, 1.0, "MISMATCH")}
    assert not w2_control_verdict_kappa(nk_t_, q1_t_, ck_t_, skv_t_, 3 * 294, 2 * 294)["pass"] \
        and not w2_control_verdict_kappa(nk_t_, pk_t_, ck_t_, skv_t_, 3 * 294, 3 * 294)["pass"] \
        and not w2_control_verdict_kappa(nk_t_, pk_t_, ck_t_ | {"flip": fl_t_ | {"eligible": False}}, skv_t_, 3 * 294, 2 * 294)["pass"] \
        and not w2_control_verdict_kappa(nk_t_, nk_t_, ck_t_, skv_t_, 3 * 294, 2 * 294)["pass"], \
        "NEGATIVE (W2b verdict): q1 moved, no count drop, an ineligible flip, or no plant at all fails"
    assert job_vtag({"origin": "S_old", "merge": True, "pw_combo": "paper_cells"}) == "S_old (all-origins draws) [pooled: paper_cells]", "job_vtag"
    # rc1b W2 AMEND4: the captured-module map and kappa's level subset, each with negatives
    cm1_ = {"r.py": "O", "m.py": "M"}
    assert w2_capmods_viol(cm1_, {"r.py": "N", "m.py": "M"}, {"r.py": ("O", "N")}) == [], "the module map: the listed runner only"
    assert w2_capmods_viol(cm1_, {"r.py": "N", "m.py": "M2"}, {"r.py": ("O", "N")}) == ["module differs | m.py"], "NEGATIVE (AMEND4): a second module differs"
    assert w2_capmods_viol(cm1_, {"r.py": "N2", "m.py": "M"}, {"r.py": ("O", "N")}) and w2_capmods_viol(cm1_, cm1_ | {"x.py": "X"}, {}), "NEGATIVE: other shas / paths"
    assert w2_levels_viol([0.01], [0.01, 0.05]) == [] and w2_levels_viol([0.05], [0.01, 0.05]) and w2_levels_viol([0.01, 0.02], [0.01, 0.05]) \
        and w2_levels_viol([0.01], [0.01, 0.02]), "NEGATIVE (AMEND4): [0.05] alone, [0.01, 0.02], or another S1 set STOP"
    # rc1b W2 R1 (d): the tree allowance (added + modified) and the span check, each with negatives
    s1w_ = {"a.py": "A", "r.py": "O", "m.py": "M"}
    vw_ = {"a.py": "A", "r.py": "N", "m.py": "M", "n1.py": "S1"}
    assert tree_sha_diff_viol(s1w_, vw_, {"r.py", "m.py"}, {}, {"n1.py": "S1"}, {"r.py": ("O", "N")}) == [], "W2 R1: the listed added and modified files pass"
    assert "file names differ" in tree_sha_diff_viol(s1w_, vw_ | {"n2.py": "S2"}, {"r.py", "m.py"}, {}, {"n1.py": "S1"}, {"r.py": ("O", "N")}), \
        "NEGATIVE (W2 R1 d): one more added file STOPs"
    assert tree_sha_diff_viol(s1w_, vw_ | {"m.py": "M2"}, {"r.py", "m.py"}, {}, {"n1.py": "S1"}, {"r.py": ("O", "N")}) == ["pinned module differs or is missing | m.py"], \
        "NEGATIVE (W2 R1 d): another modified pinned module STOPs"
    assert tree_sha_diff_viol(s1w_, vw_ | {"r.py": "N2"}, {"r.py", "m.py"}, {}, {"n1.py": "S1"}, {"r.py": ("O", "N")}) == \
        ["modified file differs from its listed shas | r.py"], "NEGATIVE (W2 R1 d): the modified runner with other shas STOPs"
    src_ = "".join(f"def {n_}():\n    return 1\n\n" for n_ in W2_PORTED)  # each ported function on lines 3k+1 .. 3k+2
    assert w2_span_viol(src_, src_, ("@@ -40,0 +41,2 @@",)) == [], "W2 R1 (b): hunks outside every ported span pass"
    assert w2_span_viol(src_, src_, ("@@ -2 +2 @@",)) == ["old: line 2 inside wmoments", "new: line 2 inside wmoments"], "NEGATIVE (W2 R1 d): a touched ported function"
    assert any("found 0 times" in x_ for x_ in w2_span_viol(src_.replace("def zwith", "def zwith2"), src_, ())), "NEGATIVE: a ported function missing"
    assert w2_hunk_lines(("@@ -832,0 +833,2 @@", "@@ -824 +824 @@")) == ([824], [833, 834, 824]) and _stops(lambda: w2_hunk_lines(("@@ x @@",))), "hunk lines"
    assert w2_flags_viol({"a": 1, "copies_root": "/x", "pw_row_res": ""}, {"a": 1, "copies_root": "/y", "spec_only": "o"}) == [] and \
        w2_flags_viol({"a": 1, "b": True}, {"a": 1}) == ["b"] and w2_flags_viol({"a": 1}, {"a": 2}) == ["a"], "W2 flags vs the pinned spec"
    assert not plant_fanout_viol(True, 15) and plant_fanout_viol(True, 16) and plant_fanout_viol(True, 14) and not plant_fanout_viol(False, 20), "NEGATIVE (AMEND11): fan-out"
    assert local_modules_viol(lmm_, ["/venv"], ["/wt", "/tree"]) == ["ps /main/src/probe_shift/__init__.py"] and \
        local_modules_viol({"a": "/wtx/a.py"}, [], ["/wt"]) == ["a /wtx/a.py"], "NEGATIVE (AMEND9): a module from the main repo's src/, or a look-alike root, STOPs"
    return ("rc1b W2a AMEND8 main-map fixture passed (an all-origins row: the canonical map gives 0 mismatches against a runner-built row; the "
            "per-origin map, dddac36e's call, flags exactly the 9 identity keys); "
            "route C: record modes by name (one of routec_A_ / routec_B_ / routec_R37_ / routec_PLANT_ stripped, once; 14 fixtures); PLANT V3: the "
            "tree_sha grant (the 4 additions pass; a 5th addition, a modified pin, another sha, a removed name, a pinned addition and no grant each "
            "STOP), the draw-0 plant solve (delta 0, both signs, a ceiling, a NaN tau), P1's shift and its allowed set (row, pooled, co-headline S; "
            "5 negatives), the solve comparison keys and the co-headline S / Y / rows; pilot zero-unit runs fail closed (row zero_unit, a missing "
            "count, a scored S_old row without its entry, a job's zero_unit_ok; each a negative fixture); --routec-zero-unit from meta; option B "
            "tree_sha: exactly the 2 listed differences pass; a third differing file, a differing pinned module, a listed file with another sha, a "
            "missing listed difference and an added file each STOP; A22 levels: the right B and R37 sets pass; A's set on B, a missing level, an extra "
            "level, a wrong set, a repeated level, spec vs record levels disagreeing and any other identity part differing each STOP; B's level split "
            "partitions 15 levels in 4 parts and A's 2 levels in 2 (a bad k/n or an empty part STOPs); Stage 2: the stage from the fold labels (not "
            "one stage's families STOPs), plants P1 / P2 on Stage 1's row and arm, the tie is pinned (Stage 2 S1 listing, F-13), the as-recorded set is "
            "the Stage 1 fold labels; "
            f"Stage 2-shaped (09 R1 option (A) / R2, review R-1 / F-3 / F-4 / F-12, 09's POST HOC (a) / (c)): no Stage 2 family in an aggregate group; "
            "the plant check (scope fixed by the plant definition; certain = the not-yet-caught levels from the unplanted call's pre-plant record x "
            "{joint, fixed}; flagged == plain value-moved above the 1e-12 floor, sub-floor changes listed; either-side moves in scope; > 0 row and pooled "
            "flagged per part) passes, passes with an already-caught level excluded (and a level with no tau_B), lists a one-ulp same_attack-like "
            f"cancellation as sub-floor, and STOPs on each of its {len(S2NEG)} plant negatives ({'; '.join(S2NEG)}); plain_diff (the floor: one ulp and "
            "5e-13 sub-floor, 1e-11 and 1e-10 moved) and plain_moved (both-NaN unmoved; 8 cases); the seeded-draw aggregate gate (5 cases; Stage 2 needs "
            "09's exact status string); and the draw-0 aggregate path filled from the as-recorded labels")


def selftest():  # runs at every invocation (main, and check() on Modal); 2e 02:2x: an empty block is NaN on both sides, equal and labelled
    import numpy as np

    nan_ = float("nan")
    assert (not eq(np.bool_(True), 1.0) and not eq(1.0, np.bool_(True)) and not eq(np.bool_(False), 0.0) and not eq(True, 1) and not eq(1, True)
            and eq(np.bool_(True), True) and eq(False, np.bool_(False)) and not eq(np.bool_(True), False)), "booleans only equal booleans (Astra r5 #4)"
    assert (eq(3, 3.0) and not eq(3, 3.0 + 1e-12) and not eq(np.int64(3), 4) and eq({"a": [1, True]}, {"a": [1, True]}) and not eq({"a": [1, True]}, {"a": [1, 1]})
            and not eq({"a": 1}, {"a": 1, "b": 2}) and not eq([1.0], [float("nan")]) and eq([float("nan")], [float("nan")])), "integers exact; nested by the same rules"
    assert eq(np.float64(0.5), 0.5) and eq(np.int64(3), 3.0) and not eq(np.float64(nan_), np.int64(0)) and eq(np.float64(nan_), nan_), "numpy scalars"
    assert classify(["a: x", "a", "b (y)"], ["a: x"], ["a", "b"]) == {"a: x": True, "a": False, "b (y)": False}, "longest prefix"
    for f_ in (lambda: classify(["c"], ["a"], ["b"]), lambda: completion_gate({"waitlist_pending": ["t1_x"], "exclusions": {"p": 1}}, {"waitlist_pending": ["t1_x"],
               "exclusions": {"p": 1}}, False), lambda: completion_gate({"waitlist_pending": [], "exclusions": {"p": 1}}, {"waitlist_pending": [], "exclusions": {"p": 2}}, False)):
        try:  # an unclassified class; a non-TEST record with a pending waitlist; a non-TEST record whose exclusions pin differs from its inputs: each STOPs
            f_()
            raise RuntimeError("a STOP case passed")
        except AssertionError:
            pass
    completion_gate({"waitlist_pending": [], "exclusions": {"p": 1}}, {"waitlist_pending": [], "exclusions": {"p": 1}}, False)
    eb_ = {"S_newmains": {"nm290_tiktok": "excluded by ruling (09 @ 1bc718e98)", "nm105_x": "arm never launched (no run)"}}
    assert apply_rulings(eb_, RULED) == [{"set": "S_newmains", "key": "nm290_tiktok", "file_class": "excluded by ruling (09 @ 1bc718e98)",
                                          "class": "attempt timeout (cause unknown)"}] and eb_["S_newmains"]["nm290_tiktok"] == "attempt timeout (cause unknown)", "ruling"
    try:  # the ruled label on another cell STOPs
        apply_rulings({"S_newmains": {"nm171_x": "excluded by ruling (09 @ 1bc718e98)"}}, RULED)
        raise RuntimeError("a ruled label on another cell passed")
    except AssertionError:
        pass
    S_, N_ = MAIN_DEFAULTS
    assert classify(["arm never launched (no run)", "attempt timeout (cause unknown)", "sim never launched (no usable honest_a reference)"], S_, N_) == {
        "arm never launched (no run)": True, "attempt timeout (cause unknown)": False, "sim never launched (no usable honest_a reference)": True}, "09's ruling on the defaults"
    completion_gate({"waitlist_pending": ["t1_x"]}, {"waitlist_pending": ["t1_x"]}, True)
    h_ = lambda c: hashlib.sha256(json.dumps(sorted(c), separators=(",", ":")).encode()).hexdigest()  # noqa: E731
    EXt = {"rows": [{"set": "P", "key": "c2", "class": "staging exclusion"}], "pending": {"P": ["c3"]}, "final_cohort_keys": {"P": {"cells": ["c1"], "sha256": h_(["c1"])}},
           "per_set": {"P": {"launched": 3, "final": 1, "excluded": 1, "pending": 1, "excluded_by_class": {"staging exclusion": 1}}}}
    assert exclusion_identity(EXt, {"P": {"c1", "c2", "c3"}}, {"P": ["c1"]}, {}, True) == ({"P": {"c2": "staging exclusion"}}, {"P": ["c3"]}), "TEST: pending listed apart"
    for args_ in ((EXt, {"P": {"c1", "c2", "c3"}}, {"P": ["c1"]}, {}, False), (EXt, {"P": {"c1", "c2", "c3", "c4"}}, {"P": ["c1"]}, {}, True),
                  (EXt, {"P": {"c1", "c2", "c3"}}, {"P": ["c1", "c3"]}, {}, True)):
        try:  # non-TEST with a pending cell; a launched cell in no part; a final cohort that differs from the record's: each STOPs
            exclusion_identity(*args_)
            raise RuntimeError("an identity STOP case passed")
        except AssertionError:
            pass
    e_ = compare([("0.01|catch|sim@S|primary", nan_, float("nan")), ("coverage|cohort_cells", 0, 0)], True)
    assert {k_: e_[k_] for k_ in ("mismatches", "nan_on_both_sides")} == {"mismatches": [], "nan_on_both_sides": {"n": 1, "label": "empty block (no cohort cells)"}} \
        and e_["ledger"] == [["0.01|catch|sim@S|primary", "MATCH"], ["coverage|cohort_cells", "MATCH"]], e_
    assert eq(37091, 37091.0) and not eq(37091, 37091.00001) and not eq(np.int64(3), float("nan")) and not eq(3, 3.000000001), "integers compare exactly"
    assert compare([("0.01|catch|sim@S|primary", nan_, float("nan"))], False)["nan_on_both_sides"]["label"].startswith("unavailable at draw 0"), "label"
    for a_, b_ in ((nan_, 0.25), (0.25, nan_), (nan_, 0), (nan_, None), (None, nan_), (nan_, "unavailable")):
        m_ = compare([("0.01|catch|sim@S|primary", a_, b_)], True)
        assert len(m_["mismatches"]) == 1 and m_["nan_on_both_sides"] == {"n": 0, "label": None}, (a_, b_, m_)
    assert math.isnan(dweighted([nan_], [0.0])) and dweighted([0.5, 0.25], [1.0, 3.0]) == 0.3125 and math.isnan(dweighted([nan_, 0.5], [0.0, 2.0]))
    ka_, na_ = relaunch_alias(np.array(["honest_b_relaunch|t1_x_r2", "honest_r3|t2_y"]), "selftest")  # G3b: the runner's relaunch key alias
    assert ka_.tolist() == ["honest_r2|t1_x", "honest_r3|t2_y"] and na_ == 1, ("relaunch alias", ka_.tolist(), na_)
    for bad_ in (["honest_b|t1_x_r2"], ["honest_b_relaunch|t1_x_r7"], ["honest_b_relaunch|t1_x_r2", "honest_r2|t1_x"]):
        try:  # another world, a k outside 1-5, two keys onto one: each STOPs
            relaunch_alias(np.array(bad_), "selftest")
            raise RuntimeError(f"a bad relaunch key passed: {bad_}")
        except AssertionError:
            pass
    # G6 on synthetic rows: S/Y means and the unavailable text; PANEL readable and saturation gates, R+C needing both readouts, per-origin readable
    rw_ = {"A C": {"df": 0.1, "yr": 0.02, "rd": True, "st": {"pooled": [0.3, 0.1], "S_old": [0.2, 0.05]}, "sat": {"pooled": False, "S_old": False}, "rdo": {"S_old": True}},
           "B C": {"df": 0.3, "yr": -0.04, "rd": True, "st": {"pooled": [0.5, 0.2], "S_old": [0.4, 0.1]}, "sat": {"pooled": True, "S_old": False}, "rdo": {"S_old": False}},
           "A R": {"df": 0.2, "yr": 0.0, "rd": False, "st": {"pooled": [0.1, 0.1]}, "sat": {"pooled": False}, "rdo": {}}}
    ch_, pn_, _ = agg_draw0(rw_, (("g", ("A", "B")),), (("g", ("A", "B")),))
    chp_, pnp_, oo_ = agg_draw0(rw_, (("g", ("A", "B")), ("h", ("A", "Z"))), (("g", ("A", "B")), ("h", ("A", "Z"))), {"A C", "B C", "A R", "B R"})
    assert sorted(oo_) == ["coheadline_signed: h|C", "coheadline_signed: h|R", "h|C", "h|R", "h|R+C"] and sorted(pnp_) == ["pooled", "pooled_abs_per_origin"], \
        (oo_, sorted(pnp_))  # pooled streams only (these synthetic rows carry no pooled_abs_per_origin: members without rows there)
    assert pnp_["pooled"]["g|R"] == {"members without rows": ["B R"]} and pnp_["pooled"]["g|C"]["primary"]["rows"] == ["A C", "B C"], "pooled: complete groups"
    rw2_ = {k_: v_ for k_, v_ in rw_.items()} | {"B R": {"df": 0.1, "yr": 0.0, "rd": True, "st": {"pooled": [0.2, 0.1]}, "sat": {"pooled": False}, "rdo": {}}}
    _, pn2_, _ = agg_draw0(rw2_, (), (("g", ("A", "B")),), {"A C", "B C", "A R", "B R"})
    assert pn2_["pooled"]["g|R"]["primary"] == {"rows": [], "withheld": "unreadable members ['A R'] (item 46: no probe omission; the group is withheld)"} \
        and pn2_["pooled"]["g|R"]["unreadable_rows"] == ["A R"], "pooled: an unreadable member withholds the group"
    assert ch_["g|R"] == {"unavailable": "rows missing ['B R'] or not readable ['A R']"} and ch_["g|C"]["rows"] == ["A C", "B C"], ch_
    assert all(eq(ch_["g|C"][k_], v_) for k_, v_ in (("S", 0.2), ("Y", -0.01), ("abs_Y", 0.01))), ch_["g|C"]
    pc_, pw_ = pn_["pooled"]["g|C"]["primary"], pn_["pooled"]["g|C"]["without_saturated"]
    assert pc_["n"] == 2 and eq(pc_["mean_G"], 0.4) and eq(pc_["mean_Y"], 0.15) and eq(pc_["mean_G_minus_mean_Y"], 0.25), pc_
    assert pw_["rows"] == ["A C"] and eq(pw_["mean_G"], 0.3) and eq(pw_["mean_G_minus_mean_Y"], 0.2), "saturated row dropped"
    assert pn_["pooled"]["g|R"] == {"primary": {"rows": [], "unavailable": "no readable rows"}, "without_saturated": {"rows": [], "unavailable": "no readable rows"},
                                    "unreadable_rows": ["A R"]}, pn_["pooled"]["g|R"]
    assert pn_["pooled"]["g|R+C"]["primary"]["rows"] == ["A C", "B C"] and pn_["pooled"]["g|R+C"]["unreadable_rows"] == ["A R"], "R+C"
    assert sorted(pn_["S_old"]) == ["g|C"] and pn_["S_old"]["g|C"]["primary"]["rows"] == ["A C"] and pn_["S_old"]["g|C"]["unreadable_rows"] == ["B C"], "per-origin"
    pt_ = lambda m_: {k_: ({"point": v_} if k_ in ("S", "Y", "abs_Y", "mean_G", "mean_Y", "mean_G_minus_mean_Y") else v_) for k_, v_ in m_.items()}  # noqa: E731
    rc_ = {g_: pt_(m_) for g_, m_ in ch_.items()}
    rp_ = {s_: {g_: {t_: (pt_(m_[t_]) if t_ != "unreadable_rows" else m_[t_]) for t_ in m_} for g_, m_ in gs_.items()} for s_, gs_ in pn_.items()}
    c0_, m0_ = agg_compare(ch_, pn_, rc_, rp_, "t")
    assert not m0_ and not compare(c0_, False)["mismatches"], "a record equal to the aggregation passes"
    rc_["g|C"]["S"]["point"] += 1e-6
    rp_["pooled"]["g|R+C"]["primary"]["rows"] = ["A C"]
    c1_, m1_ = agg_compare(ch_, pn_, rc_, rp_, "t")
    assert [x[0] for x in compare(c1_, False)["mismatches"]] == ["t|coh|g|C|S"] and m1_ == [["t|panel|pooled|g|R+C|primary|rows, n or fields"]], (m1_,)
    assert blind([["k", 0.5, 0.25], ["j", [1], None]]) == [["k"], ["j"]], "keys only"
    import copy  # agg_check on a synthetic record (one external row, one gap row): equal trees pass; a combo deleted from both trees, an extra PANEL level fail

    q0_ = {"gap": {"delta_full": {"point": 0.3}}, "catch": {"run1": {"point": 0.5}, "run2": {"point": 0.46}}, "readable": {"all": True},
           "panel": {"pooled": {"G": {"point": 0.5}, "Y": {"point": 0.2}, "saturated": False}}}
    vl_ = {"FIBS|uncut C": {"cb": {"0.01": rw_["A C"]}}}
    rc0_ = {"meta": {"full_length": True, "auroc06_subset": {"members": list(AUROC06_FULL)}},
            "pooled": {lb_: {"cb": {"joint": {"0.01": q0_}, "fixed": {"0.01": q0_}}} for lb_ in ("sol_a_uncut C", "FIBS|uncut C")}}
    gs0_ = FULL_GROUPS_ + (("auroc06_subset", AUROC06_FULL),)
    rc0_["coheadline_signed"], rc0_["panel_groups"] = {}, {}
    for md_ in ("joint", "fixed"):
        ch0_, pn0_, _ = agg_draw0(agg_rows(rc0_, vl_, "cb", md_, "0.01", {fa for _, fs in gs0_ for fa in fs})[0], gs0_, FULL_GROUPS_)
        rc0_["coheadline_signed"].setdefault("cb", {}).setdefault(md_, {})["0.01"] = {g_: pt_(m_) for g_, m_ in ch0_.items()}
        for s_, gs_ in pn0_.items():
            rc0_["panel_groups"].setdefault("cb", {}).setdefault(s_, {}).setdefault(md_, {})["0.01"] = {
                g_: {t_: (pt_(m_[t_]) if t_ != "unreadable_rows" else m_[t_]) for t_ in m_} for g_, m_ in gs_.items()}
    a0_ = agg_check(rc0_, vl_, [0.01])
    assert not a0_["mism"] and not compare(a0_["cmp"], False)["mismatches"] and a0_["cmp"] and a0_["used"] == ["sol_a_uncut C"], a0_["mism"]
    rc1_ = copy.deepcopy(rc0_)
    del rc1_["coheadline_signed"]["cb"], rc1_["panel_groups"]["cb"]
    assert any(m_[0].startswith("combos:") for m_ in agg_check(rc1_, vl_, [0.01])["mism"]), "a combo deleted from both trees passed"
    rc1_ = copy.deepcopy(rc0_)
    s0_ = sorted(rc1_["panel_groups"]["cb"])[0]
    rc1_["panel_groups"]["cb"][s0_]["joint"]["0.02"] = rc1_["panel_groups"]["cb"][s0_]["joint"]["0.01"]
    assert ["cb|panel hierarchy (streams, modes, levels)"] in agg_check(rc1_, vl_, [0.01])["mism"], "an extra PANEL level passed"
    rc2_ = copy.deepcopy(rc0_)  # PLANT V3 (AMEND6): the plant job's withheld panel_groups: the exact marker passes, no PANEL key compared
    rc2_["panel_groups"] = {"withheld": PLANT_PANEL_WITHHELD}
    a2_ = agg_check(rc2_, vl_, [0.01], panel_withheld=PLANT_PANEL_WITHHELD)
    assert not a2_["mism"] and not compare(a2_["cmp"], False)["mismatches"] and a2_["cmp"] == [c_ for c_ in a0_["cmp"] if "|panel|" not in c_[0]] \
        and any("|panel|" in c_[0] for c_ in a0_["cmp"]), "withheld panel_groups: the comparisons are the unwithheld ones minus every PANEL key"
    assert agg_check(rc2_, vl_, [0.01], ("FIBS|uncut C", "G", "pooled", 1e-3), panel_withheld=PLANT_PANEL_WITHHELD)["expect"] == [] and \
        agg_check(rc0_, vl_, [0.01], ("FIBS|uncut C", "G", "pooled", 1e-3))["expect"], "the PANEL control: expected keys with a PANEL tree, none without"
    rc2_["panel_groups"] = {"withheld": PLANT_PANEL_WITHHELD + " "}
    assert ["panel_groups: not the plant job's exact withheld marker"] in agg_check(rc2_, vl_, [0.01], panel_withheld=PLANT_PANEL_WITHHELD)["mism"], \
        "NEGATIVE: another marker text MISMATCHES"
    assert agg_check(rc0_, vl_, [0.01], panel_withheld=PLANT_PANEL_WITHHELD)["mism"], "NEGATIVE: a real PANEL tree where the marker is expected MISMATCHES"
    # G7 on synthetic pools: the union dedups by run key (sizes, membership patterns); an identity conflict and an undeclared alias STOP; RULE 37 gate;
    # per_set_beside equal by canonical JSON (NaN equals NaN), a changed block flagged
    nb_ = lambda n: np.full((n, 5), nan_)  # noqa: E731
    P0_ = lambda k, s: (np.array(k), np.array(s), np.zeros(len(k), np.int64), nb_(len(k)))  # noqa: E731
    e0_ = {"B": P0_(["honest_b|t1", "honest_b|t1", "honest_c|t2"], [0.1, 0.2, 0.3]), "A": P0_([], []), "S": P0_(["sim|t1"], [0.4])}
    e1_ = {"B": P0_(["honest_b|t1", "honest_b|t1", "honest_a|t3"], [0.1, 0.2, 0.5]), "A": P0_([], []), "S": P0_(["sim|t1"], [0.4])}
    pu0_, sz0_, pt0_ = pw_union_chk([("S_old (all-origins draws) [pooled: cb]", e0_), ("S_new (all-origins draws) [pooled: cb]", e1_)], "selftest")
    assert sz0_["B"] == {"runs": 3, "units": 4, "pattern_units": {"S_new": 1, "S_new + S_old": 2, "S_old": 1}, "pattern_runs": {"S_new": 1, "S_new + S_old": 1, "S_old": 1},
                         "zero_unit_runs": {"runs": 0, "by_membership_pattern": {}}} and sz0_["S"]["runs"] == 1 and sz0_["A"]["units"] == 0, sz0_
    assert sorted(pu0_["B"][1].tolist()) == [0.1, 0.2, 0.3, 0.5] and pt0_["B"]["honest_b|t1"] == "S_new + S_old", "union"
    assert pw_identity_gates(sz0_, pt0_, set()) == [] and sz0_["content_identical_keys"] == [] and sz0_["identity_conflicts"] == [], "a clean union passes the gates"
    g_ = lambda e2_, prs, z=None: (lambda u_: pw_identity_gates(u_[1], u_[2], prs, z))(  # noqa: E731
        pw_union_chk([("S_old (all-origins draws) [pooled: cb]", e0_), ("S_new (all-origins draws) [pooled: cb]", e2_)], "selftest", z))
    Z_ = {"A": P0_([], []), "S": P0_([], [])}
    assert g_({"B": P0_(["honest_b|t1", "honest_b|t1"], [0.1, 0.25])} | Z_, set())[0].startswith("no false merge: 1 "), "the same key with other units"
    assert g_({"B": P0_(["honest_c|t2"], [0.3]), "A": P0_(["honest_c|t2"], [0.35]), "S": P0_([], [])}, set())[0].startswith("no false merge"), "a key across roles"
    assert g_({"B": P0_([], [])} | Z_, set(), {"A": {"honest_b|t1": {"S_new"}}})[0].startswith("no false merge: 1 "), "zero units in one appearance, units in another"
    assert g_({"B": P0_(["honest_x|t9"], [0.3])} | Z_, set())[0].startswith("no false split: 1 "), "identical content under another key"
    assert g_({"B": P0_(["honest_c_e2|t2"], [0.3])} | Z_, {frozenset(("honest_c_e2|t2", "honest_c|t2"))}) == [], "one declared E12 pair is one execution"
    assert g_({"B": P0_(["honest_c_e2|t2", "honest_x|t9"], [0.3, 0.3])} | Z_, {frozenset(("honest_c_e2|t2", "honest_c|t2"))})[0].startswith("no false split"), \
        "a group of three is not one declared pair"
    # G7b gate 2 (2e 20:1x): a declared group passes only with the token-prefix record's support for this label's own scorer file
    tp0_ = {"pairs": [{"pair": ["honest_x|t9", "honest_y|t9"], "verdict": "(1) natural",
                       "files": [["p2", "fibs__command.npz", "p2", "fibs__command.npz", "natural"], ["p2", "x.npz", "own", "x.npz", "natural"]]}]}
    dg_ = {frozenset(("honest_x|t9", "honest_y|t9"))}
    gd_ = lambda fl_: pw_identity_gates(*pw_union_chk([("S_new", {"B": P0_(["honest_x|t9", "honest_y|t9"], [0.3, 0.3])} | Z_)], "t")[1:3], set(), None,  # noqa: E731
                                        dg_, prefix_support(tp0_, fl_))
    assert gd_({("p2", "fibs__command.npz")}) == [], "a declared group with the record's natural verdict for this label's scorer file"
    assert gd_({("p2", "harmfulness__command.npz")})[0].startswith("declared groups or E12 pairs without token-prefix support"), "no natural file of this label"
    assert gd_({("own", "x.npz")})[0].startswith("declared groups or E12 pairs without token-prefix support"), "a cross-scorer pair never supports"
    nat_ = ["p2", "fibs__command.npz", "p2", "fibs__command.npz", "natural"]
    tp3_ = {"pairs": [{"pair": ["honest_x|t9", "honest_y|t9"], "verdict": "(1) natural", "files": [nat_]},
                      {"pair": ["honest_x|t9", "honest_z|t9"], "verdict": "(1) natural", "files": [nat_]},
                      {"pair": ["honest_y|t9", "honest_z|t9"], "verdict": "(2) extraction-level mix-up", "files": [nat_]}]}
    k3_ = ["honest_x|t9", "honest_y|t9", "honest_z|t9"]
    u3_ = pw_union_chk([("S_new", {"B": P0_(k3_, [0.3, 0.3, 0.3])} | Z_)], "t")
    assert pw_identity_gates(u3_[1], u3_[2], set(), None, {frozenset(k3_)}, prefix_support(tp3_, {("p2", "fibs__command.npz")}))[0].startswith(
        "declared groups or E12 pairs without token-prefix support"), "CUT (2): a 3-member declared group with ONE pair not '(1)' STOPs"
    tp3b_ = {"pairs": tp3_["pairs"][:2]}
    assert pw_identity_gates(u3_[1], u3_[2], set(), None, {frozenset(k3_)}, prefix_support(tp3b_, {("p2", "fibs__command.npz")}))[0].startswith(
        "declared groups or E12 pairs without token-prefix support"), "CUT (2): a 3-member declared group with one pair absent from the record STOPs"
    e12_ = {frozenset(("honest_c_e2|t2", "honest_c|t2"))}
    ue_ = pw_union_chk([("S_new", {"B": P0_(["honest_c_e2|t2", "honest_c|t2"], [0.3, 0.3])} | Z_)], "t")
    assert pw_identity_gates(ue_[1], ue_[2], e12_, None, set(), prefix_support(tp0_, {("p2", "fibs__command.npz")}))[0].startswith(
        "declared groups or E12 pairs without token-prefix support"), "2e 20:2x: an E12 pair without the record's support STOPs once the record is given"
    tpe_ = {"pairs": [{"pair": ["honest_c_e2|t2", "honest_c|t2"], "verdict": "(1) natural", "files": [nat_]}]}
    assert pw_identity_gates(ue_[1], ue_[2], e12_, None, set(), prefix_support(tpe_, {("p2", "fibs__command.npz")})) == [], "a supported E12 pair passes"
    assert pw_identity_gates(*pw_union_chk([("S_new", {"B": P0_(["honest_x|t9", "honest_z|t9"], [0.3, 0.3])} | Z_)], "t")[1:3], set(), None, dg_,
                             prefix_support(tp0_, {("p2", "fibs__command.npz")}))[0].startswith("no false split"), "an undeclared group still STOPs"
    ar_ = {"world": "honest_c", "sets": ["S_old"], "roles": ["A", "B"], "n": 1}
    assert pw_auth_rule_viol({"honest_c|t1": ["B", "A"]}, {"honest_c|t1": {"world": "honest_c", "sets": ["S_old"], "roles": ["A", "B"]}}, ar_) == []
    assert pw_auth_rule_viol({"honest_c|t1": ["A", "B"], "honest_b|t2": ["A", "B"]}, {"honest_c|t1": {"world": "honest_c", "sets": ["S_old"], "roles": ["A", "B"]},
                             "honest_b|t2": {"world": "honest_b", "sets": ["S_new"], "roles": ["B"]}}, ar_), "another overlap STOPs"
    assert pw_auth_rule_viol({}, {"honest_c|t1": {"world": "honest_c", "sets": ["S_old"], "roles": ["A", "B"]}}, ar_), "a missing overlap STOPs"
    assert scorer_of("cutfull/ep2-nm/p2/ep2_nm_cutfull_v1__honest_a/fibs__command.npz") == "p2" and \
        scorer_of("/conv/scaleup/ep2_nm_v1__honest_a/famexp_r2/a__C.npz") == scorer_of("/p2/famexp/2026-10-14_163027/pass2_2026-10-15_184738/layers/a__C__L16.npz")
    assert g_({"B": P0_(["honest_b_relaunch|t5_r2"], [0.9])} | Z_, set())[0].startswith("relaunch run keys") and \
        g_({"B": P0_(["honest_r13|t5"], [0.8])} | Z_, set())[0].startswith("relaunch run keys") and \
        g_({"B": P0_([], [])} | Z_, set(), {"S": {"honest_r2|t6": {"S_new"}}})[-1].startswith("relaunch run keys"), "a relaunch key in either form"
    assert unit_bytes((np.array([-0.0, nan_]), np.array([0]), np.full((1, 5), nan_))) == unit_bytes((np.array([0.0, float("nan")]), np.array([0], np.int32),
                                                                                                     np.full((1, 5), float("nan")))), "one content policy"
    assert pw_overlap_viol({"roles_by_key": {"k|c": ["A", "B"]}}, {"k|c": {"roles": ["A", "B"]}}) == [] and \
        pw_overlap_viol({"roles_by_key": {"k|c": ["A", "B"]}}, {"k|c": {"roles": ["B"]}}), "overlap authorization by roles"
    assert rule37_gate_chk({"o1": {"S": True, "B": True, "A": False}}) and not rule37_gate_chk({"o1": {"S": True}}) and not rule37_gate_chk({}), "RULE 37 gate"
    ps0_, sm_ = {"path": "p", "sha256": "h"}, {"point": 0.5, "finite_draws": 3, "draws": 3}
    mt0_ = {"inputs_final": {"path": "i", "sha256": "a"}, "full_length": {"inputs_full": {"path": "f", "sha256": "b"}}, "cohort_list": {"ref": "c"},
            "merge": {"seed": [1, 2]}}
    blk_ = {"joint": {"0.01": {"catch": {k_: sm_ for k_ in ("sim", "run1", "run2", "real_mean")}, "gap": {"delta_full": sm_}, "x": nan_}}}
    cb0_, rw0_ = "paper_cells", {"draws_sha256": "d", "cohort_cell_keys": ["c1"]}
    six_g = lambda ro_: {"primary": {k_: sm_ for k_ in ("mean_G", "mean_Y", "mean_G_minus_mean_Y")} | {"rows": [f"{fa} {ro_}" for fa in FULL_GROUPS_[0][1]], "n": 6},  # noqa: E731
                         "unreadable_rows": []}
    psr_ = {"meta": mt0_, "pooled": {"L C": {cb0_: blk_}}, "rows": {"L C": {f"{o_} (all-origins draws)": dict(rw0_) for o_ in PW_COMBO_ORIGINS[cb0_]}},
            "coheadline_signed": {cb0_: {"joint": {"0.01": {g_: {"S": sm_, "Y": sm_} for g_ in ("headline_six_full|C", "headline_six_full|R")}}}},
            "panel_groups": {cb0_: {"pooled": {"joint": {"0.01": {f"headline_six_full|{r_}": six_g(r_) for r_ in "CR"}}}}}}
    rb_ = {"meta": mt0_ | {"pooled_world": {"combos": [cb0_], "per_set_record": ps0_}},
           "rows": {"L C": {f"{o_} (all-origins draws) [pooled: {cb0_}]": dict(rw0_) for o_ in PW_COMBO_ORIGINS[cb0_]}},
           "pooled": {"L C": {cb0_: {"members": [f"{o_} (all-origins draws) [pooled: {cb0_}]" for o_ in PW_COMBO_ORIGINS[cb0_]]}}},
           "per_set_beside": {"source": ps0_, "label": "registered: per-set calibration", "by_combo": {cb0_: {"L C": json.loads(json.dumps(blk_))}},
                              "coheadline_signed": psr_["coheadline_signed"], "panel_groups": {cb0_: psr_["panel_groups"][cb0_]}}}
    assert beside_check(rb_, psr_)[1] == [], ("an exact copy of a compatible companion passes", beside_check(rb_, psr_)[1])
    psr_["pooled"]["L C"][cb0_]["joint"]["0.01"]["x"] = 0.5
    assert beside_check(rb_, psr_)[1] == [f"per_set_beside|by_combo|{cb0_}|L C"], "a changed per-set block"
    psr_["pooled"]["L C"][cb0_]["joint"]["0.01"]["x"] = nan_
    psr_["meta"] = mt0_ | {"inputs_final": {"path": "i", "sha256": "z"}}
    assert beside_check(rb_, psr_)[1] == ["per_set_beside|companion|inputs_final", "per_set_beside|companion|pool-defining inputs"], \
        "a companion from other inputs (Astra g7 #2)"
    fold_ = selftest_fold() + "; " + selftest_value() + "; " + selftest_routec()
    return ("passed: NaN on both sides is equal and labelled ('empty block' when the row has no cohort cells); NaN against a number, 0, None or "
            "text is a mismatch; an all-empty D-weighted combo is NaN; numpy booleans and scalars compare as numbers; exclusion classes by longest "
            "prefix, an unclassified class stops the check; a non-TEST record with pending waitlist cells or an unpinned exclusions file stops; launched = "
            "final + excluded + pending by key, pending listed apart only in a TEST record; 09's per-key ruling relabels nm290_tiktok only; the default "
            "class lists follow 09's ruling; G6: S/Y means and unavailable text, the PANEL readable, saturation, R+C and per-origin gates, the record "
            "comparison flags a shifted S and a changed row list, a combo deleted from both trees and an extra PANEL level; mismatches reach the "
            "laptop as keys only; G7: the pooled union dedups by run key with its sizes and membership patterns; 09's identity gates flag a false "
            "merge (a key with other units, also across roles), a false split (identical content under another key unless a declared E12 replicate) "
            "and a relaunch key in either form; overlaps need authorization for their roles; the RULE 37 gate reads S and B; per_set_beside must "
            "equal the blocks of a compatible per-set companion; " + fold_)


@app.function(image=image, cpu=(4, 4), memory=(16384, 24576), timeout=900, retries=0, startup_timeout=120, scaledown_window=60,  # rc1b W2 AMEND1: timeout 900
              volumes={"/sol": modal.Volume.from_name(VOLS["solavg"], version=2).read_only(), "/p2": modal.Volume.from_name(VOLS["p2"], version=2).read_only(),
                       "/conv": modal.Volume.from_name(VOLS["conv"]).read_only(), "/res": res_vol})
def check(inp: dict, recb: bytes, started: str, run_dir: str = "") -> dict:  # noqa: C901
    import numpy as np

    inp_sha0 = hashlib.sha256(json.dumps(inp, sort_keys=True).encode()).hexdigest()  # at entry: check() adds ao_w to the variants below (Codex r29)
    if not recb:  # G5: the record shipped once to the results Volume; read it there and check it against the run manifest's sha256
        assert run_dir, "no record bytes and no run directory"
        res_vol.reload()
        recb = (Path("/res") / run_dir / "record.json").read_bytes()
        assert hashlib.sha256(recb).hexdigest() == json.loads((Path("/res") / run_dir / "manifest.json").read_bytes())["record_sha256"], \
            "the record on the results Volume differs from the run manifest's sha256"
    rec = json.loads(recb)
    levels = inp["levels"]
    levels_all = inp.get("levels_all") or levels  # B's level split (A22): the full set the record's level structure is checked against
    assert set(levels) <= set(levels_all) and levels, "this call's levels are not a non-empty part of the full set"
    lvk_all = sorted(f"{x:g}" for x in levels_all)

    FL5 = {bool((inp["by_label"].get(l_) or {}).get("five")) for l_ in inp["labels"]}  # G7b FOLD-MODEL: five per label (one label per call)
    assert len(FL5) <= 1, "a call mixes fold-model and single-model labels"
    FIVE = bool(inp.get("five")) or FL5 == {True}
    FAM = {(inp["by_label"].get(l_) or {}).get("family") for l_ in inp["labels"]}.pop() if inp["labels"] else None  # own_all_rs: rows_rs's guards

    def load(vol, path, sha, efold=None, simh=False):  # -> key, designated score, fold, five model scores (rows x 5; NaN where absent)
        b = (Path(MOUNT[vol]) / path.lstrip("/")).read_bytes()
        assert hashlib.sha256(b).hexdigest() == sha, (vol, path)
        return rows_from_npz(np.load(io.BytesIO(b), allow_pickle=False), FIVE, FAM, simh, efold, path)

    out = {"read_started_pt": started, "selftest": selftest(), "rows": [], "pooled": [], "rules": []}
    mine, mine_rd, mine_cells, mine_mainof = {}, {}, {}, {}
    G6_ = inp["mode"] == "alltasks" and bool(rec["meta"].get("full_length"))  # G6: the per-label PANEL block and the S/Y and excess inputs
    aggin = {}  # label -> combo -> level -> this check's draw-0 inputs to S/Y and the excess (written to vals/ on the results Volume)
    W2 = inp.get("w2")  # rc1b W2: {"kind": "pertask" | "kappa", "f", "d1_pins", "modified", "script_pair", ...} (main, per call), else None
    W2N, W2CTL = {}, {}  # rc1b W2: label -> the values compared (the count gate); the control call's plants (what each moved, keys only)
    PWM = inp.get("pw")  # G7: the pooled-world build {"combos", "stage", "files": {"<label> | <combo>": pooled pools file}}, else None
    if PWM:  # G7b / Astra g7 #1: the pinned manifest (keys, counts, pooled pools files) read on the Volume and checked: sha256; a catalogue test's code
        mb_ = (Path("/sol") / PWM["manifest"]["path"]).read_bytes()  # commit; a scored record's pool-defining inputs and the mapping its meta names
        assert hashlib.sha256(mb_).hexdigest() == PWM["manifest"]["sha256"], "pooled manifest sha256"
        MFc = json.loads(mb_)
        mfiles_ = {k_: v_["pooled_file"] for k_, v_ in MFc["per_label_combo"].items()}
        mpi_ = MFc.get("pool_inputs") or {}  # the code pins (modules_sha256, runner @ 8a15fb10f) are checked file by file against the code commit's tree
        w2m_ = {k_: tuple(v_) for k_, v_ in ((W2 or {}).get("modified") or {}).items()}  # rc1b W2 R1: the runner, (S1's = the manifest's sha, W2's)
        pa_ = {k_: v_ for k_, v_ in mpi_.items() if k_ != "modules_sha256"}
        pb_ = {k_: v_ for k_, v_ in PWM["pool_inputs"].items() if k_ != "modules_sha256"}
        if W2:  # the manifest (S1's catalogue, built by A's runner) and the W2 record differ in the runner script only, exactly as listed
            assert (pa_.pop("script_sha256", None), pb_.pop("script_sha256", None)) == tuple(W2["script_pair"]), "rc1b W2 R1: the runner script pair (STOP)"
        assert pa_ == pb_, "the pinned manifest was built from other pool-defining inputs or another runner script"
        mods_ = mpi_.get("modules_sha256")
        assert isinstance(mods_, dict) and set(PWM["required_modules"]) <= set(mods_) and all(
            k_ in PWM["tree_sha"] and (PWM["tree_sha"][k_] == v_ or w2m_.get(k_) == (v_, PWM["tree_sha"][k_])) for k_, v_ in mods_.items()), \
            "the pinned manifest's module pins are incomplete or not the code commit's files"  # Codex r45 #4: every module the runner pins by name
        mdiff_ = sorted(set(mods_) ^ set(PWM["captured_modules"] or {}))  # Codex r46 #4 / r47 #1: the client's sys.modules decides extra same-folder pins,
        out["rules"].append({"row": inp["labels"][0], "rule": "manifest module pins vs this capture's (info: the launch client may import more)",  # so a
                             "violations": [], "info": [f"{len(mdiff_)} module paths in one map only: {mdiff_[:4]}"] if mdiff_ else []})  # difference is noted
        if PWM["stage"] == "scored":  # a scored record names its own module pins: the manifest's
            rmo_ = PWM["pool_inputs"].get("modules_sha256") or {}  # rc1b W2 R1: equal but the listed runner (the manifest's sha, the record's)
            assert rmo_ == mods_ or (W2 and set(rmo_) == set(mods_) and all(rmo_[k_] == mods_[k_] or w2m_.get(k_) == (mods_[k_], rmo_[k_]) for k_ in mods_)), \
                "the record's module pins differ from the manifest's"
        if PWM["stage"] == "catalogue":
            assert (MFc.get("meta") or {}).get("git") == PWM["git"], "the pooled manifest was built at another code commit"
        else:
            assert mfiles_ == PWM["files"], "meta.pooled_world.manifest_per_label differs from the pinned manifest's pooled files"
        ctb_ = (Path("/sol") / PWM["manifest"]["path"].replace("_manifest.json", "_catalogue.json")).read_bytes()  # Codex r40 #7: the catalogue beside it,
        CTc = json.loads(ctb_)["runs"]  # the one the manifest was built with (its runs sha256); runs, roles and canonical mains compared after each union
        assert hashlib.sha256(json.dumps({"runs": CTc}, sort_keys=True, default=str).encode()).hexdigest() == MFc["catalogue_runs_sha256"], \
            "the catalogue beside the manifest is not the one it was built with"
        PWM = PWM | {"files": mfiles_, "mf": MFc["per_label_combo"], "cat": CTc}
        mmc_ = sorted(r_ for r_, e_ in CTc.items() if inp["canon"].get(r_.split("|", 1)[1]) != e_.get("main"))  # Codex r41 #2: EVERY catalogued run's
        out["rules"].append({"row": inp["labels"][0], "rule": "catalogue: every run's canonical main is the union stream's (all catalogue entries)",  # main
                             "violations": [f"{len(mmc_)} catalogued runs with another canonical main, e.g. {mmc_[:3]}"] if mmc_ else [], "runs": len(CTc)})
    PWX, PWU, PWH = {}, {}, {}  # (label, combo) -> the export pass's per-row admitted pools; -> the union rebuilt here; -> members the runner withholds
    PWZ = {}  # (label, combo) -> the union's admitted zero-unit runs and structural no-row cells, by role (G7e)
    PWF1 = {}  # G7b FOLD-MODEL: (label, combo) -> the union of its rows' attack fold -1 columns {B, A, S} (pw_prepare l.3895)
    NCMP, VCNT, VZERO = {}, {}, []  # G7b VALUE: (label, row, level, mode) -> value comparisons; per label x combo x level x mode counts; zero cells
    VNAN, BNAN = [], {}  # G7b VALUE: catch / gap / panel and seeded-draw keys NaN on both sides (violations, by key); per label, the seeded draws' count (info)
    VP = inp.get("value_plant")  # G7b VALUE PLANT (Astra r5 #3): this call re-runs one label with ONE source score perturbed (a separate call)
    VPI = {}  # the planted source unit (keys only: row, arm, cell)
    BOOT = inp.get("boot")  # G7b VALUE (09 via 2e 23:2x): the seeded sample of bootstrap draws, recomputed per draw and compared with the record's npz
    BW, BZF, BREF, BPOOL, BROW, BAGG = None, None, {}, {}, {}, {}  # BAGG: label -> combo -> mode -> level -> draw -> S / Y inputs (mine, the npz's)
    PLV = inp.get("plant_v3")  # PLANT V3: {"labels", "points", "deltas"} from the PLANT record's meta.plant / rows (main), else None
    P1 = inp.get("plant_p1")  # PLANT V3 P1 (AMEND4 D3, AMEND5 F2): {"label", "seed"} on the P1 call only (its shards under p1/)
    PFIT, BROWP, P1HIT = {}, {}, {}  # (label, origin, level) -> (solve record, {delta: c}); (label, origin, level, draw) -> {delta: planted catch}
    SIM2N = {"il_reports": 0, "sim2_solves": 0}  # AMEND7 G2: the IL reports and sim2 solves compared here (the launch gate: 60 and 120 over the labels)
    if BOOT:
        assert PWM and PWM["stage"] == "scored", "the bootstrap-draw sample runs on a pooled scored record only"
        sd_ = BOOT_SEED_BY_RECORD.get(hashlib.sha256(recb).hexdigest(), BOOT_SEED)  # PLANT V3: the record-specific seed (A: unchanged)
        assert BOOT["ids"] == [0] + boot_ids(sd_) and BOOT["seed"] == sd_ and BOOT["n"] == BOOT_N and BOOT["D"] == BOOT_D, "the sampled draw ids"
    assert not (PLV or P1) or (BOOT and hashlib.sha256(recb).hexdigest() == PLANT_RECORD_SHA), "PLANT V3 runs on the PLANT record's value run only"
    assert not P1 or (PLV and P1["label"] in inp["labels"] and not VP), "P1 runs on its own label's call, with the plant inputs, never with a value plant"
    if P1:  # AMEND4 D3 / AMEND5 F2: the P1 target, chosen HERE by its seed (keys only): an origin row of its label, a plant level, the delta order
        rg_ = np.random.default_rng(int(P1["seed"]))
        os_ = sorted(o_ for o_ in rec["rows"][P1["label"]] if "(registered cohort)" not in o_)
        lv_ = sorted(f"{x_:g}" for x_ in PLV["points"])
        P1 = dict(P1) | {"origin": os_[int(rg_.integers(len(os_)))], "level": lv_[int(rg_.integers(len(lv_)))],
                         "order": [int(x_) for x_ in rg_.permutation([d_ for d_ in PLV["deltas"] if d_ != 0])]}
    P1HIT = {"fired": False, "why": "the P1 row was not reached"} if P1 else {}
    if BOOT and W2 and W2["kind"] == "kappa":  # rc1b W2b: kappa carries no per-draw npz; its draws are regenerated here and compared with its row store
        assert BOOT["npz"] is None and not rec.get("npz"), "rc1b W2b: a per-draw npz pin on the kappa run (STOP)"
        BW = boot_draws(rec["meta"]["merge"], inp["canon"], inp["merge_mains"], BOOT["ids"])
    elif BOOT:
        npz_ = rec.get("npz") or {}
        pzt_ = [npz_.get("volume"), npz_.get("path"), npz_.get("sha256")]  # read from THIS record by key (2e (b)); the map and inp only assert it
        assert pzt_ == list(BOOT["npz"]) == list(PERDRAW_BY_RECORD.get(hashlib.sha256(recb).hexdigest()) or []) and pzt_[0] == VOLS["solavg"], \
            "per-draw npz pin: the record's own npz, the run's and the per-record map's (09 cond. 1)"
        assert not npz_.get("label_escapes"), "the per-draw npz has label escapes (not ported: STOP)"
        BW = boot_draws(rec["meta"]["merge"], inp["canon"], inp["merge_mains"], BOOT["ids"])  # the union stream regenerated from the record's seed (ported); the run's ids
        pz_ = Path("/sol") / pzt_[1]
        h_ = hashlib.sha256()
        with pz_.open("rb") as fh:
            for ch_ in iter(lambda: fh.read(1 << 24), b""):
                h_.update(ch_)
        assert h_.hexdigest() == pzt_[2], "the per-draw npz sha256 differs from the record's"
        BZF = np.load(pz_, allow_pickle=False)  # members decompressed one key at a time

        def bref(*parts):  # the record's per-draw array at the sampled draws (Volume detail only; keys and verdicts are released)
            k_ = npzkey(*parts)
            if k_ not in BREF:
                assert k_ in BZF.files, ("per-draw npz key missing", k_)
                a_ = BZF[k_]
                assert a_.shape == (BOOT_D + 1,), ("per-draw npz array shape", k_, a_.shape)
                BREF[k_] = [float(x_) for x_ in a_[BW["ids"]]]
            return BREF[k_]
    FOLDINFO = {}  # (label, origin) -> keys / counts of the fold-model rules applied in the export pass (for the reader)
    assert not (PWM and FIVE) or all((inp["by_label"].get(l_) or {}).get("fold_label") for l_ in inp["labels"]), \
        "pooled world: a fold-model probe only as a --fold-labels label (catalogue stage, or the scored stage's draw-0 value reproduction)"
    held_by = {}  # label -> {origin: withheld text}: rows the runner computes but withholds (pool_reduce_n pools the rest)
    CAP = inp.get("cap")

    def rank(runs):  # position of each row within its run, in file (concatenation) order
        seen_, rk_ = {}, np.empty(len(runs), np.int64)
        for i_, r_ in enumerate(runs.tolist()):
            rk_[i_] = seen_.get(r_, 0)
            seen_[r_] = rk_[i_] + 1
        return rk_
    for lab in inp["labels"]:
        L = inp["by_label"][lab]
        # ---- real rows: S_old file(s), S_new arm files (+ 25f honest-A roots for S_new), S_newmains folder rows owned by record
        base = {"run": [], "score": [], "fold": [], "sbm": [], "source": []}

        def add(k, s, f, sb, src):
            base["run"].append(k), base["score"].append(s), base["fold"].append(f), base["sbm"].append(sb), base["source"].append(np.array([src] * len(k)))

        LOOF = {"S_old": 0, "S_new": 1, "S_newmains": 2} if L.get("loo") else {}
        for e in L["old_files"]:
            add(*load(e["vol"], e["path"], e["sha256"], LOOF.get("S_old")), "S_old")
        for e in L["new_files"]:
            add(*load("conv", e["path"], e["sha256"], LOOF.get("S_new")), "S_new")
        U0 = ({x: np.concatenate(v) for x, v in base.items()} if base["run"] else
              {"run": np.array([], str), "score": np.array([]), "fold": np.array([], np.int64), "sbm": np.empty((0, 5)), "source": np.array([], str)})
        held_rows = held_by.setdefault(lab, {})  # origin -> withheld text (the runner's merge_counts and pool_reduce_n skip withheld rows)
        ao_done = False
        seq_ = ([o for o in inp["origins"] if not aob(o).endswith("(all-origins draws)")] + [o for o in inp["origins"] if aob(o).endswith("(all-origins draws)")])
        seq_ = ([(o, "export") for o in seq_] * bool(PWM)  # G7: the runner's two stages (export pass, then scored pass); a catalogue check exports only
                + [(o, "scored") for o in seq_] * (not PWM or PWM["stage"] != "catalogue"))
        for orig, pass_ in seq_:
            V = inp["variants"][orig]
            if PWM and pass_ == "scored" and (lab, V["pw_combo"]) not in PWU:  # G7: the label x combo's union, once every row of it is exported
                PWU[(lab, V["pw_combo"])] = pw_union_chk(PWX[(lab, V["pw_combo"])], f"{lab} | {V['pw_combo']}", PWZ.get((lab, V["pw_combo"]), ({}, {}))[0])
            if aob(orig).endswith("(all-origins draws)") and not ao_done:  # 09 @ e3a459bd0 item 1 + am.1 §5: each canonical main's cell count per group,
                ao_done = True  # over the group's all-origins rows whose origin row is not withheld (by key)
                aov_ = {o: v for o, v in inp["variants"].items() if "ao_groups" in v and aob(o).removesuffix(" (all-origins draws)") not in held_rows}
                cnt_ = {}
                for o, v in aov_.items():
                    for g in v["ao_groups"]:
                        for c in v["cohort"] or []:
                            cnt_.setdefault(g, {})[inp["canon"][c]] = cnt_.setdefault(g, {}).get(inp["canon"][c], 0) + 1
                for o, v in aov_.items():
                    n_ = {}
                    for g in v["ao_groups"]:
                        for c in v["cohort"] or []:
                            m = inp["canon"][c]
                            assert n_.setdefault(m, cnt_[g][m]) == cnt_[g][m], (lab, o, m, "a main counted differently by two all-origins groups")
                    v["ao_w"] = {c: 1.0 / n_[inp["canon"][c]] for c in v["cohort"] or []}
            if "ao_groups" in V and orig.removesuffix(" (all-origins draws)") in held_rows:  # its origin row is withheld, so is this one
                r_ = rec["rows"][lab].get(orig, {})
                out["rows"].append({"row": lab, "origin": orig, "withheld": r_.get("withheld"), "n_compared": 1,
                                    "mismatches": [] if "withheld" in r_ else [["all-origins row of a withheld origin row", True, r_.get("withheld")]]})
                held_rows[orig] = r_.get("withheld") or True
                continue
            if V.get("unavailable"):  # own_all_LOO's with-S4 S_newmains (09 via 507-prep 00:1x): not computed; the record carries the rule only
                held_rows[orig] = "unavailable"
                r_ = rec["rows"][lab][orig]
                ok_u = r_ == {"label": lab, "origin": orig, "unavailable_rule": "unavailable: the S4 cells were not in the deciding eligibility file"}  # exactly the runner's placeholder (Codex r16)
                out["rows"].append({"row": lab, "origin": orig, "unavailable": r_.get("unavailable_rule"), "n_compared": 1,
                                    "mismatches": [] if ok_u else [["unavailable row", "unavailable: the S4 cells were not in the deciding eligibility file", r_]]})
                continue
            if V.get("absent"):  # LOO has no S_paper493 or all-origins row: the record must not carry one
                out["rows"].append({"row": lab, "origin": orig, "absent": True, "n_compared": 1,
                                    "mismatches": [] if orig not in rec["rows"].get(lab, {}) else [["a row the runner does not compute", None, orig]]})
                continue
            SPO = bool(V.get("sp"))  # S_paper493 (09 @ e3a459bd0): its own origin line, pools, cohort and draws
            AO = "ao_groups" in V  # the all-origins row: the same pools and thresholds, cells weighted 1/n_main
            UN = orig.startswith("union")
            NM = orig.startswith("S_newmains") or UN
            bo = orig.split(" ")[0]  # base origin ("S_old (registered cohort)" -> S_old)
            REG = "(registered cohort)" in orig  # also "X (registered cohort) (all-origins draws)" (G4, r28 item 11)
            srcs = ["S_new", "S_newmains"] if UN else ["S_newmains"] if NM else [bo]
            U = {x: v.copy() for x, v in U0.items()}
            extra = {"run": [], "score": [], "fold": [], "sbm": [], "source": []}
            ha25, ha25_bad, seen_ha = [], [], set(U0["run"].tolist())
            if bo == "S_new" or UN or SPO:  # the 25f honest-A roots join S_new, the union and S_paper493 (as row())
                for e in L["ha25f"]:
                    k, s, f, sb = load("conv", e["path"], e["sha256"], LOOF.get("S_new"))
                    ha25.append(sorted(set(k.tolist())))
                    if "keys" in e and (ha25[-1] != e["keys"] or set(ha25[-1]) & seen_ha):  # G7b FOLD-MODEL (l.1417-1419): the root's keys, none present
                        ha25_bad.append(e["path"].rsplit("/", 2)[-2])  # already (the runner STOPs)
                    seen_ha |= set(ha25[-1])
                    extra["run"].append(k), extra["score"].append(s), extra["fold"].append(f), extra["sbm"].append(sb), extra["source"].append(np.array(["S_new"] * len(k)))
            orphan_hits, taken, raw_nm = [], {}, set()
            if SPO:
                seen_sp = set(U0["run"].tolist()) | {x for k_ in extra["run"] for x in k_.tolist()}
                for e in V["folders"]:
                    k, s, f, sb = load(e["vol"], e["path"], e["sha256"])
                    if e.get("world") is not None:  # G7b FOLD-MODEL (l.1446-1448): the file's world token aliased to the folder's routing world
                        k, ren_ = sp_world_chk(k, e["world"], f"{lab} {e['arm']}")
                        if ren_:
                            FOLDINFO.setdefault((lab, orig), {}).setdefault("sp_world_aliases", {})[e["arm"]] = {"alias": f"{ren_} -> {e['world']}",
                                                                                                                "renamed_rows": int(len(k)), "renamed_runs": len(set(k.tolist()))}
                    assert set(k.tolist()) <= set(e["keys"]) | set(e["refused_keys"]), (lab, e["arm"], "S_paper493 rows outside the folder's routed keys")
                    m = np.isin(k, e["keys"])
                    assert not set(k[m].tolist()) & seen_sp, (lab, e["arm"], "S_paper493 run keys already present")
                    seen_sp |= set(k[m].tolist())
                    extra["run"].append(k[m]), extra["score"].append(s[m]), extra["fold"].append(f[m]), extra["sbm"].append(sb[m])
                    extra["source"].append(np.array(["S_paper493"] * int(m.sum())))
            if NM:
                for e in L["nm_files"]:
                    k, s, f, sb = load("conv", e["path"], e["sha256"], LOOF.get("S_newmains"))
                    raw_nm |= set(k.tolist())  # every loaded key, before the ownership mask (the contradiction audit reads these)
                    m = np.array([inp["owner"].get(x) == e["arm"] for x in k], bool)
                    orphan_hits += [x for x in k[m].tolist() if [x, e["arm"].split("__")[0]] in inp["s4_orphans"]]
                    for x in set(k[m].tolist()):
                        assert taken.setdefault(x, e["arm"]) == e["arm"], (lab, x, "two S_newmains folders")
                    assert not set(k[m].tolist()) & seen_ha, (lab, e["arm"], "S_newmains run keys also in the 455 files (the runner STOPs, l.1434)")
                    extra["run"].append(k[m]), extra["score"].append(s[m]), extra["fold"].append(f[m]), extra["sbm"].append(sb[m])
                    extra["source"].append(np.array(["S_newmains"] * int(m.sum())))
            if extra["run"]:
                U = {x: np.concatenate([U[x]] + extra[x]) for x in U}
            E12 = V.get("e12")  # G2 (Codex r28 item 6): epoch 1 + 2 calibration (09 @ 64f00b09e), as row(): this origin's epoch-2 honest rows,
            if E12:  # renamed <world_e2>|<cell> (redraw / _e2a|b tags stripped from the cell), only the keys the pools admit
                want12 = set(E12["H_B"]) | set(E12["H_B2"]) | set(E12["H_B_coverage"]) | {k_ for g_ in E12["H_A"].values() for k_ in g_}
                seen12 = set(U["run"].tolist())
                for a_ in E12["folders"]:
                    k, s, f, sb = load(a_["vol"], a_["path"], a_["sha256"], LOOF.get(E12["origin"]))  # LOO: the origin's held-out model (Codex r35)
                    w_ = {x.split("|", 1)[0] for x in k.tolist()}
                    assert w_ <= {a_["world"]}, (lab, orig, a_["arm"], "e12: rows of another world", sorted(w_))
                    k2 = np.array([f"{a_['world_e2']}|{re.sub(r'(_e2[ab]|_redraw[0-9]+)+$', '', x.split('|', 1)[1])}" for x in k.tolist()])
                    assert len(set(k2.tolist())) == len(set(k.tolist())), (lab, orig, a_["arm"], "e12: two raw keys name one cell")
                    m = np.isin(k2, list(want12))
                    assert not set(k2[m].tolist()) & seen12, (lab, orig, a_["arm"], "e12: epoch-2 run keys already present")
                    seen12 |= set(k2[m].tolist())
                    U = {x: np.concatenate([U[x], v_]) for x, v_ in zip(("run", "score", "fold", "sbm", "source"),
                                                                       (k2[m], s[m], f[m], sb[m], np.array([E12["origin"]] * int(m.sum()))))}
            ZERO = (set(V["zero_runs"]) if SPO else set() if REG else set(inp.get("zero_runs", [])))  # 09 (2e 00:5x): a kept zero-tool arm is not caught
            # (S_paper493: the v2 file's structural_zero_keys at this readout, for every S_paper493 row; registered older-origin rows keep their own rule)
            src_mask_ = np.isin(U["source"], srcs)
            zero_with_rows = sorted(r_ for r_ in ZERO if (U["run"][src_mask_] == r_).any() or r_ in raw_nm)  # before the cap and the ownership mask (Codex r11/r12)
            present_pre = set(U["run"][src_mask_].tolist())  # attack keys with rows before the cap
            capinfo = {}
            if CAP in (30, "30", "taskmin"):  # run-length cap (runlength RULE): K = 30 per run; taskmin: attack runs cut to the per-task min over sim/run1/run2
                rkU = rank(U["run"])
                KU = np.full(len(rkU), 30, np.int64)
                if CAP == "taskmin":
                    w0 = np.array([r.split("|", 1)[0] for r in U["run"]])
                    c0 = np.array([r.split("|", 1)[1] for r in U["run"]])
                    isA_ = np.isin(U["source"], srcs) & np.isin(w0, A3) & np.isin(c0, V["cohort"])
                    nrun = {}
                    for r_ in U["run"][isA_].tolist():
                        nrun[r_] = nrun.get(r_, 0) + 1
                    kt = {c: min(nrun.get(f"{w}|{c}", 0) for w in A3) for c in V["cohort"]}
                    KU = np.where(isA_, np.array([kt.get(c, 0) for c in c0]), KU)
                    capinfo["taskmin_K"] = {"min": int(min(kt.values())), "median": float(np.median(list(kt.values()))), "max": int(max(kt.values())),
                                            "zero_K_cells": sorted(c for c, k_ in kt.items() if k_ == 0)}
                keepU = rkU < KU
                U = {x: v[keepU] for x, v in U.items()}
            world = np.array([r.split("|", 1)[0] for r in U["run"]])
            cell = np.array([r.split("|", 1)[1] for r in U["run"]])
            isN = (U["source"] == "S_paper493") if SPO else np.isin(U["source"], srcs)
            # ---- H_sim
            HK, HS, HF, HB, HR = [], [], [], [], []
            orph_h = []  # G2 (r28 item 10): the S_newmains sim-honest roots are S4-orphan guarded too (the runner STOPs on any orphan (key, root) row)
            fi_ = FOLDINFO.setdefault((lab, orig), {}) if FIVE else {}  # G7b FOLD-MODEL: the rules' keys and counts on this row
            for e in V["simh"]:
                k, s, f, sb = load("conv", e["path"], e["sha256"], LOOF.get("S_old" if bo == "S_old" else "S_new" if bo == "S_new" else "S_newmains"), simh=True)
                if f is None:  # l.1722-1732: an A25 Sol / c_asis file without fold: its cells' r36 fold from this row's S_old real rows (else STOP)
                    f, cfl_ = r36_fill(k, U["run"], U["fold"], U["source"])
                    fi_.setdefault("a25_r36_files", []).append(e.get("root", e["batch"]))
                    fi_["a25_r36_rows"] = fi_.get("a25_r36_rows", 0) + int(len(k))
                    fi_["r36_cells_with_two_real_folds"] = sorted(set(fi_.get("r36_cells_with_two_real_folds", [])) | set(cfl_))
                if e.get("root") in set(V.get("nm_simh_roots", inp.get("nm_simh_roots", []))):  # (the runner's orphan guard reads the file's own keys,
                    # before any alias; a fold label's own job's roots, Claude N4)
                    orph_h += [f"{x}@{e['root']}" for x in k.tolist() if [x, e["root"].split("__")[0]] in inp["s4_orphans"]]
                if SPO and e.get("sp_world"):  # l.1735-1738: S_paper493's own sim-honest roots, their world token aliased to the folder's
                    k, ren_ = sp_world_chk(k, e["sp_world"], f"{lab} {e.get('root')}")
                    if ren_:
                        FOLDINFO.setdefault((lab, orig), {}).setdefault("sp_world_aliases", {})[e.get("root")] = {"alias": f"{ren_} -> {e['sp_world']}",
                                                                                                                 "renamed_rows": int(len(k)), "renamed_runs": len(set(k.tolist()))}
                HK.append(k), HS.append(s), HF.append(f), HB.append(sb), HR.append(np.array([e.get("root", e["batch"])] * len(k)))
            hk, hs, hf, hsb, hr = np.concatenate(HK), np.concatenate(HS), np.concatenate(HF), np.concatenate(HB), np.concatenate(HR)
            assert len(set(hk.tolist())) == len(set(zip(hk.tolist(), hr.tolist()))), (lab, orig, "a sim-honest run key in two roots")
            assert not orph_h, (lab, orig, "S4 orphan sim-honest rows (the runner STOPs)", orph_h[:5])
            hrs_ = (hk.copy(), hf.copy())  # l.1763-1765: own_all_rs's fold check reads H_sim before the fill-in filter (S_paper493: after its keep)
            nocell_h = sorted({x for x in hk.tolist() if x.split("|", 1)[1] not in inp["canon"]}) if (AO and PWM) else []  # G7b (l.1741): every
            # sim-honest key's cell is a cell of the union stream's map, before any filter (the runner STOPs)
            if V.get("fill_admitted"):  # G2 (r28 item 7): S_old tau_sim = primary_v3 u A25 fill-in u g4, the fill-in roots' admitted keys only, each with rows
                fr_ = V["fill_admitted"]
                keep_ = np.array([(r_ in set(fr_[h_])) if h_ in fr_ else True for r_, h_ in zip(hk.tolist(), hr.tolist())], bool)
                hk, hs, hf, hsb, hr = hk[keep_], hs[keep_], hf[keep_], hsb[keep_], hr[keep_]
                nof_ = {rt_: sorted(set(ad_) - set(hk[hr == rt_].tolist())) for rt_, ad_ in fr_.items()}
                assert not any(nof_.values()), (lab, orig, "e12 fill-in admitted runs without rows (the runner STOPs)", nof_)
            if CAP in (30, "30", "taskmin"):  # paired leaves every pool (and H_sim) uncut
                kH = rank(hk) < 30
                capinfo["H_sim_rows_kept"] = {"kept": int(kH.sum()), "of": int(len(kH))}
                hk, hs, hf, hsb, hr = hk[kH], hs[kH], hf[kH], hsb[kH], hr[kH]
            sp_w = []  # S_paper493 reasons the runner must withhold this row
            miss_h = []
            if SPO:  # H_sim(P): g4/g5 rows on their admitted keys; the existing roots' rows on S_paper493's mains only (by key)
                spr_, adm_ = set(V["sp_roots"]), set(V["adm_sp"])
                keep_ = np.array([(r_ in adm_) if h_ in spr_ else (inp["sp_moa"][r_.split("|", 1)[1]] in set(V["spm"])) for r_, h_ in zip(hk.tolist(), hr.tolist())], bool)
                excl_kept = sorted(set(hk[keep_].tolist()) & set(V["pex"]))
                hk, hs, hf, hsb, hr = hk[keep_], hs[keep_], hf[keep_], hsb[keep_], hr[keep_]
                miss_h = sorted(adm_ - set(hk.tolist()))
                if set(miss_h) - set(V["simh_zero_unit"]):
                    sp_w.append("admitted sim-honest runs without rows, unclassified")
                if V["pending"]:
                    sp_w.append("files pending")
                if V["cohort"] is None:
                    sp_w.append("registered cohort not pinned")
                hrs_ = (hk.copy(), hf.copy())
            if FIVE and FAM == "own_all_rs":  # l.1763-1765 (09 3a (c)): every U and H_sim unit on its main's fold in the committed list, else STOP
                RSF_ = inp["by_label"][lab]["rs_folds"]
                cmr_ = RSF_["cmains"][V["rs_cmain"]]  # this job's own map (the runner's cmain_)
                off_rs = rs_fold_viol(U["run"].tolist(), U["fold"].tolist(), cmr_, RSF_["fold_of_main"]) + \
                    rs_fold_viol(hrs_[0].tolist(), hrs_[1].tolist(), cmr_, RSF_["fold_of_main"])
                assert not off_rs, (lab, orig, "own_all_rs units off the fold list (the runner STOPs)", len(off_rs), off_rs[:3])
                fi_["own_all_rs_fold_check"] = {"units_checked": int(len(U["run"]) + len(hrs_[0])), "off_list": 0, "fold_list": "@".join(RS_FOLDS[:2])}
            if FIVE and SPO and V.get("r36") is not None:  # l.1820-1832: Sol / c_asis S_paper493 rows and H_sim on their r36 / real-side fold, else WITHHELD
                off36_ = sp_r36_off(np.array([r.split("|", 1)[1] for r in U["run"]]), U["fold"], U["source"], hk, hf, V["r36"])
                fi_["sp493_r36_off_fold"] = len(off36_)
                if off36_:
                    sp_w.append(f"S_paper493 rows off their main's r36 fold ({len(off36_)}, e.g. {off36_[:3]})")
            hcells = sorted({x.split("|", 1)[1] for x in hk.tolist()} | {x.split("|", 1)[1] for x in miss_h})  # S_paper493: zero-unit cells too
            excl_pref = set(V["excl_prefixes"])
            excl_rows = sorted({x for x in hk.tolist() if x.split("|", 1)[1].split("_", 1)[0] in excl_pref})
            rep_root = V["replaced_root"]
            rep_bad = sorted({x for x, r_ in zip(hk.tolist(), hr.tolist()) if x.split("|", 1)[1].split("_", 1)[0] in rep_root
                              and r_.removesuffix("__sim_honest") != rep_root[x.split("|", 1)[1].split("_", 1)[0]]})
            rep_missing = sorted(set(rep_root) - {x.split("|", 1)[1].split("_", 1)[0] for x in hk.tolist()})
            rep_rows = int(sum(x.split("|", 1)[1].split("_", 1)[0] in rep_root for x in hk.tolist()))
            # ---- pools (spec): S_old H_B = FINAL pool_keys_old (B u C), A = its honest_c, B2 = its honest_b; S_new H_B = pool_keys_new,
            # A = S_new honest_a on H_sim's cells; S_newmains H_B = pool_honest_b (+ 4r lists when every S4r folder has the file), A = honest_a on H_sim's cells
            if SPO:  # H_B(P) by key from any source, minus pool exclusions; H_A(P) = honest_a on S_paper493's mains, on H_sim(P)'s cells
                cm_ = np.array([inp["sp_moa"].get(c_) for c_ in cell], dtype=object)
                hb = np.isin(U["run"], V["pool_b"]) & ~np.isin(U["run"], V["pex"])
                pools = {"B": hb, "A": (world == "honest_a") & np.isin(cm_, V["spm"]) & ~np.isin(U["run"], V["pex"]) & np.isin(cell, hcells)}
                coh = V["cohort"] or []
                main_of = V["main_of"]
                noa_ = sorted(set(hcells) - set(cell[pools["A"]].tolist()))
                if noa_:
                    sp_w.append("H_sim cells without an admitted honest_a run")
            elif bo == "S_old":
                hb = isN & np.isin(U["run"], inp["pool_old"])
                pools = {"B": hb, "A": hb & (world == "honest_c"), "B2": hb & (world == "honest_b")}
                coh = V["cohort"]
                main_of = {c: c for c in coh}
            else:
                hb = isN & np.isin(U["run"], V["pool_b"])
                pools = {"B": hb, "A": isN & (world == "honest_a") & np.isin(cell, hcells)}
                coh = V["cohort"]
                main_of = V["main_of"]
            if E12:  # G2 (r28 item 6): the real honest pools gain their admitted epoch-2 replicates, by key, replicate-only (09 @ 64f00b09e §2, §6)
                e1b = set(inp["pool_old"]) if bo == "S_old" else set(V["pool_b"])  # the epoch-1 members this row's H_B already holds
                e2b = E12["H_B"] if (bo != "S_newmains" or V.get("use4r", True)) else E12["H_B_coverage"]
                assert set(e2b.values()) <= e1b, (lab, orig, "e12: an epoch-2 H_B entry replicates a non-member")
                pools["B"] = pools["B"] | (isN & np.isin(U["run"], list(e2b)))
                if bo == "S_old":  # Q4: H_A (honest_c) gains no epoch 2; the B-alone pool gains the e2b replicates of its B members
                    e1b2 = {k_ for k_ in inp["pool_old"] if k_.startswith("honest_b|")}
                    assert set(E12["H_B2"].values()) <= e1b2 and not E12["H_A"], (lab, orig, "e12: S_old B-alone / honest_c pools differ from the artifact")
                    pools["B2"] = pools["B2"] | (isN & np.isin(U["run"], list(E12["H_B2"])))
                else:  # H_A gains the e2a replicates of its own epoch-1 members (honest_a on H_sim's cells), by key; S_new by the family's artifact group
                    e1a = {f"honest_a|{c_}" for c_ in hcells}
                    all_a = {k_: r_ for g_ in E12["H_A"].values() for k_, r_ in g_.items()}
                    src_a = E12["H_A"][E12["H_A_group"]] if E12["H_A_group"] else all_a
                    take = {k_ for k_, r_ in src_a.items() if r_ in e1a}
                    have_a = set(U["run"][pools["A"]].tolist())
                    hard = sorted(k_ for k_, r_ in all_a.items() if r_ in e1a and k_ not in take and r_ in have_a)
                    assert not hard, (lab, orig, "e12 H_A: replicates of this row's members missing from the family's artifact list (the runner STOPs)", hard[:5])
                    pools["A"] = pools["A"] | (isN & np.isin(U["run"], list(take)))
                used12 = {"H_B": set(e2b)} | ({"H_B2": set(E12["H_B2"])} if bo == "S_old" else {"H_A": take})
                e12_miss = {nm_: sorted(v_ - set(U["run"].tolist())) for nm_, v_ in used12.items() if v_ - set(U["run"].tolist())}  # the runner: no shrunken pool
                adm12 = {"H_B": e1b | set(e2b), "H_A": ({k_ for k_ in inp["pool_old"] if k_.startswith("honest_c|")} if bo == "S_old" else e1a | take)}
                if bo == "S_old":
                    adm12["H_B2"] = e1b2 | set(E12["H_B2"])
            else:
                e12_miss, adm12 = {}, None
            fold = U["fold"]
            att = {w: isN & (world == w) & np.isin(cell, coh) for w in A3}
            five_ = {}
            hk_adm = set(hk.tolist())  # G7b FOLD-MODEL: H_sim's admitted runs with rows, before the fold -1 rule (l.1772: H_rows_runs)
            coh_pre = list(coh)  # the cohort before the fold -1 rule (the runner's cohN; Astra r2: l.1866's draw-universe check reads it)
            if FIVE:  # 25h §1 fold rules (as run_peraction311.row): mapped-vs-fold -1 counts (nm360), then 21b where every pool can carry it, else fold -1 dropped
                if NM or (SPO and PWM):  # G7b FOLD-MODEL: S_paper493 too (l.1834; the per-set check never reads a fold-model S_paper493 row)
                    five_["mvf"] = {nm_: {"mapped": int(m_.sum()), "fold_minus1": int(f_.sum())} for nm_, m_, f_ in
                                    (("H_B", pools["B"] & (fold >= 0), pools["B"] & (fold == -1)), ("H_A", pools["A"] & (fold >= 0), pools["A"] & (fold == -1)),
                                     ("H_sim", hf >= 0, hf == -1))}
                    five_["withheld"] = mvf_withheld(five_["mvf"], V.get("allunits_dev"), V.get("fxp"), SPO)  # l.1838 (Astra r1 #1)
                use21b = V["use21b_listed"] and all(int((pools[x] & (fold == -1)).sum()) >= 2 for x in ("B", "A")) and int((hf == -1).sum()) >= 2
                if not use21b and V["per_pool_21b_listed"]:
                    f1n = {"H_B": int((pools["B"] & (fold == -1)).sum()), "H_A": int((pools["A"] & (fold == -1)).sum()), "H_sim": int((hf == -1).sum())}
                    if "B2" in pools:
                        f1n["H_B2"] = int((pools["B2"] & (fold == -1)).sum())
                    att_f1 = sorted({c for w in A3 for c in cell[att[w] & (fold == -1)].tolist()})
                    if all(n == 0 or n >= 2 for n in f1n.values()) and (not att_f1 or all(n >= 2 for n in f1n.values())):
                        use21b = True
                five_["use21b"] = use21b
                dropped = {}
                if not use21b:
                    f1_cells = sorted({c for w in A3 for c in cell[att[w] & (fold == -1)].tolist()})
                    dropped = {"cohort_cells": f1_cells, "H_B_runs": sorted(set(U["run"][pools["B"] & (fold == -1)].tolist())),
                               "H_A_runs": sorted(set(U["run"][pools["A"] & (fold == -1)].tolist())), "H_sim_runs": sorted(set(hk[hf == -1].tolist()))}
                    pools = {x: m_ & (fold >= 0) for x, m_ in pools.items()}
                    kf = hf >= 0
                    hk, hs, hf, hsb, hr = hk[kf], hs[kf], hf[kf], hsb[kf], hr[kf]
                    coh = [c for c in coh if c not in set(f1_cells)]
                    att = {w: isN & (world == w) & np.isin(cell, coh) for w in A3}
                five_["dropped"] = {k_: {"n": len(v_), "keys": v_} for k_, v_ in dropped.items()}
            Pz = {nm: (U["run"][m], U["score"][m], fold[m], U["sbm"][m]) for nm, m in pools.items()} | {"S": (hk, hs, hf, hsb)}
            r3_check, alias3 = None, None  # alias3: the relaunch key alias count, set whenever the uncut relaunch files are read (Astra g3b)
            if V.get("r3") and not FIVE:  # G3 (Codex r28 item 9): S_old's ALTERNATE real-honest pool R3 (09 item 15 @ 4c095dec9; Append 25b), as row():
                R3 = [load(e["vol"], e["path"], e["sha256"]) for e in V["r3"]["files"]]  # the probe's uncut relaunch files, in the config's order
                rr3, rs3, rf3, rb3 = (np.concatenate([x[i] for x in R3]) for i in range(4))
                rr3, alias3 = relaunch_alias(rr3, f"{lab} {orig} relaunch")  # G3b: the runner @ 20c6fbd20 aliases before its membership check (l.1677-1681)
                mem3 = set(V["r3"]["members"])
            if V.get("r3") and not FIVE and len(rr3):  # no relaunch rows at all: the runner records the line as unavailable and computes no R3 (Codex r37)
                off3 = sorted(set(rr3.tolist()) - mem3)
                assert set(off3) <= set(V["r3"]["staged_outside"]), (lab, orig, "R3: relaunch rows outside the members file and its listed extras (the runner STOPs)")
                k3 = np.isin(rr3, list(mem3))
                assert set(rr3[k3].tolist()) == mem3, (lab, orig, "R3: pool keys != the frozen 958 members (the runner STOPs)")
                Pz["R3"] = (rr3[k3], rs3[k3], rf3[k3], rb3[k3])
                r3_check = {"pool_runs": len(mem3), "members": len(mem3), "pool_keys_equal_members": True, "members_without_units": [],
                            "rows_outside_membership_listed_extras": off3}
            in_rep = np.array([x.split("|", 1)[1].split("_", 1)[0] in rep_root for x in hk.tolist()], bool)
            if in_rep.any():  # removal pools (09 17:1x) exist where replacement rows survive the fold rules: replaced keys out of H_sim, matching honest_a out of H_A
                Pz["Snr"] = (hk[~in_rep], hs[~in_rep], hf[~in_rep], hsb[~in_rep])
                mA = pools["A"] & ~np.array([c_.split("_", 1)[0] in rep_root for c_ in cell], bool)
                Pz["Anr"] = (U["run"][mA], U["score"][mA], fold[mA], U["sbm"][mA])
            mains = sorted({main_of[c] for c in coh})
            PzS = Pz  # the row's own per-set pools (its coverage fields describe these, also in the pooled build)
            if PWM:  # G7 (item 42): scored pass: the label x combo's union replaces H_B, H_A, H_sim (the export pass stashes them below, after the
                pk_ = (lab, V["pw_combo"])  # availability checks). No pooled B-alone pool (y1 = run1@B, item 45(1)); no R3 on an all-origins row; the per-set
                if pass_ == "scored":  # removal pools (Snr, Anr) stay, as in the runner, which still fits their thresholds but computes no removal catch
                    Pz = dict(PWU[pk_][0]) | {nm: P_ for nm, P_ in PzS.items() if nm in ("Snr", "Anr")}

            def zp(P_, oof=False):  # draw-0 unit z of a pool and its moments (25h §1; module-level pool_z0, G7b VALUE)
                return pool_z0(P_, FIVE, oof)

            def zw(P_, ZPB):  # G7b VALUE (runner zwith l.2276-2296, item 44(3)): pooled H_A with the pooled H_B's moments (module-level pool_z0_with)
                return pool_z0_with(P_, ZPB, f"{lab} {orig}", float((Pz["B"][2] == -1).sum()))  # t0r at draw 0: H_B's fold -1 unit count

            OOF_ = bool(L.get("oof"))
            if PWM and pass_ == "scored" and FIVE:  # G7b VALUE: the pooled pools (union): B and S fit their own moments, A takes B's
                for nm_ in ("B", "A", "S"):  # the runner's STOPs on the pooled pools (l.2443-2453): pool() on B and S; a fold count on B and S; 21b finiteness
                    _, s0_, f0_, b0_ = Pz[nm_]
                    col0_ = any(int((f0_ == j).sum()) < 2 for j in range(5))
                    assert nm_ == "A" or not (col0_ and not OOF_ and len(b0_) and not np.isfinite(b0_).all()), (lab, orig, nm_, "pooled pool(): non-finite fallback")
                    assert nm_ == "A" or five_["use21b"] or OOF_ or not col0_, (lab, orig, nm_, "pooled pool lacks a fold (the runner STOPs)")
                    assert not (f0_ == -1).any() or np.isfinite(b0_[f0_ == -1]).all(), (lab, orig, nm_, "pooled fold -1 units without five finite model scores")
                ZP = {nm: zp(P_, OOF_) for nm, P_ in Pz.items() if nm != "A"}
                ZP["A"] = zw(Pz["A"], ZP["B"])
            else:
                ZP = {nm: zp(P_, OOF_) for nm, P_ in Pz.items()}
            if CAP == "paired":  # paired arms: sim and the paired real run cut to the shorter one's unit count per task (file order); pools uncut
                rkU_ = rank(U["run"])
                for tag_, wx in (("p1", "run1"), ("p2", "run2")):
                    n_ = {}
                    for w in ("sim", wx):
                        for r0 in U["run"][att[w]].tolist():
                            n_[r0] = n_.get(r0, 0) + 1
                    mt = {c: min(n_.get(f"sim|{c}", 0), n_.get(f"{wx}|{c}", 0)) for c in coh}
                    mvec = np.array([mt.get(c, 0) for c in cell])
                    att[f"sim_{tag_}"] = att["sim"] & (rkU_ < mvec)
                    att[f"{wx}_{tag_}"] = att[wx] & (rkU_ < mvec)
                    capinfo[tag_] = {"pair": f"sim vs {wx}", "m": {"min": int(min(mt.values())), "median": float(np.median(list(mt.values()))), "max": int(max(mt.values()))},
                                     "units_removed": {"sim": int((att["sim"] & ~att[f"sim_{tag_}"]).sum()), wx: int((att[wx] & ~att[f"{wx}_{tag_}"]).sum())}}
            rmax, rfold, runits, missing = {}, {}, {}, []
            if CAP == "taskmin" and ZERO:  # a kept zero arm makes its task's K 0: the cell's other arms are cut to nothing by the cap, so they are structural zeros too
                ZERO = ZERO | {f"{w}|{c}" for c, k_ in kt.items() if k_ == 0 and any(f"{a}|{c}" in ZERO for a in A3) for w in A3
                               if f"{w}|{c}" in present_pre}  # only arms whose rows the cap removed; a genuinely absent arm stays missing (Codex r12)
            def arm_runs(w):  # an arm's cohort runs from the SOURCE unit rows (module-level arm_runs_of)
                return arm_runs_of(w, U, att[w], fold, coh, FIVE, ZERO)

            for w in att:
                keys_, rmax[w], rfold[w], runits[w], ub_ = arm_runs(w)
                if w in A3:
                    missing += [k_ for k_, v, fo in zip(keys_, rmax[w], rfold[w]) if not np.isfinite(v) and not (FIVE and fo == -1 and k_ in ub_) and k_ not in ZERO]

            def zrun_m(w, mom):  # attack-run z under one pool's moments (module-level zrun_of: zatt l.2261-2274)
                return zrun_of(rmax[w], rfold[w], runits[w], mom, FIVE, f"{lab} {orig} {w}")

            def zrun(w, nm):  # attack-run z under pool nm's draw-0 moments
                return zrun_m(w, ZP[nm][1] if FIVE else None)

            if VP and pass_ == "scored" and lab == VP["label"] and orig == VP["origin"]:  # G7b VALUE PLANT (Astra r5 #1 / #3, r6 4; 507-prep 23:11):
                w_ = VP["arm"]  # ONE source unit score of this row's arm is moved (module-level apply_plant, end-to-end self-tested), then the arm's runs
                U, built_, pinfo_ = apply_plant(w_, U, att[w_], fold, coh, FIVE, ZERO, ZP["B"])  # are rebuilt from the source and the normal path runs on
                _, rmax[w_], rfold[w_], runits[w_], _ = built_
                VPI.update({"row": lab, "origin": orig, "arm": w_, "run": pinfo_["run"], "unit_rank_in_run": pinfo_["unit_rank_in_run"], "pass": pass_})
            PT_, PRE_ = None, {}  # 09 (c), 10-20 ~04:5x: in the UNPLANTED value call, the run a value plant on this row would move (plant_target, as
            if BOOT is not None and not VP and pass_ == "scored":  # apply_plant) and whether it is caught at draw 0 at each level (keys / booleans only)
                for lb0_, o0_, a0_ in VALUE_PLANTS + VALUE_PLANTS_S2:
                    if lb0_ == lab and o0_ == orig and a0_ in att:
                        PT_ = (a0_, coh[plant_target(a0_, U, att[a0_], fold, coh, FIVE, ZERO, ZP["B"])])

            CH = {}

            def catch(w, nm, T):
                if np.isnan(T):  # unavailable threshold (k < 1 or undefined moments): the runner reports a NaN catch
                    CH[(w, nm)] = None
                    return float("nan")
                if not coh:  # an empty block (S_paper493 P_new with no cells): no cell to catch, NaN as the runner's mean over no main
                    CH[(w, nm)] = {}
                    return float("nan")
                zr = zrun(w, nm)
                hit = {c: bool(v > T) for c, v in zip(coh, zr)}  # the runner's za > T (l.2633): NaN never above, +inf always (G7b VALUE, Astra r5 #1)
                CH[(w, nm)] = hit
                if AO:  # all-origins row: sum_c hit_c / n_main(c) over sum_c 1 / n_main(c) (draw 0: every multiplicity 1); G7b VALUE: tpr l.2213-2215
                    Wc_ = np.array([[V["ao_w"][c] for c in coh]], dtype=np.float64)  # in the runner's form: Wc = W[0, ccol] * wcell (1 x cells), then
                    return float(((Wc_ * np.array([[hit[c] for c in coh]], dtype=np.float64)).sum(1) / Wc_.sum(1))[0])  # (Wc * hit).sum(1) / Wc.sum(1)
                return float(np.mean([np.mean([hit[c] for c in coh if main_of[c] == m_]) for m_ in mains]))

            nf_bad = []
            nf_all = []
            if inp.get("nf_audit") and not REG and not SPO:  # a launched cell outside the headline must lack a scored arm here unless it is in a structural class (09 @ 01012e87e);
                # S_paper493 has no no-filter list (its cohort is the v2 file's own no-filter cohort, compared by key below)
                STR_ = inp["nf_struct"]  # class -> structural, decided in main() by prefix (Codex r9: no substring match; 09's ruling via 2e 02:3x)
                valid_ = np.isfinite(U["score"]) & (fold >= 0) if FIVE else np.isfinite(U["score"])
                if FIVE:  # fold -1 rows are measured through their five model scores (21b), as the attack evaluator reads them
                    valid_ |= (fold == -1) & np.all(np.isfinite(U["sbm"]), 1)
                for o_ in (["S_new", "S_newmains"] if UN else ["S_newmains"] if NM else [bo]):
                    for c_, cl_ in inp["nf_audit"][o_].items():
                        has_all = all(bool(valid_[(U["source"] == o_) & (world == w) & (cell == c_)].any()) for w in A3)
                        if has_all:
                            nf_all.append(f"{o_}|{c_}|{cl_}")  # every excluded cell with all three arms scored here, whatever its class (listed for the reader)
                        if has_all and not STR_[cl_]:
                            nf_bad.append(f"{o_}|{c_}|{cl_}")
            if SPO and missing:  # every S_paper493 cohort cell needs attack rows in all three arms (a kept zero-unit run only by key)
                sp_w.append("cohort cells without attack rows")
            if SPO:  # and every cohort attack unit finite (fold >= 0 or a single-model probe: its score; a five-model fold -1 unit: all five model scores)
                am_ = isN & np.isin(world, list(A3)) & np.isin(cell, coh)
                fin_ = np.where((fold >= 0) | (not FIVE), np.isfinite(U["score"]), np.all(np.isfinite(U["sbm"]), 1))
                if (am_ & ~fin_).any():
                    sp_w.append(f"non-finite attack scores ({int((am_ & ~fin_).sum())} rows)")
            if PWM and pass_ == "export":  # G7 (item 42) export pass, after every admission and availability check: this row's admitted H_B, H_A, H_sim
                PWX.setdefault(pk_, []).append((orig, {nm: PzS[nm] for nm in ("B", "A", "S")}))
                adr_ = {"B": set(adm12["H_B"]) if adm12 else set(inp["pool_old"] if bo == "S_old" else V["pool_b"]),  # G7e: this row's admitted runs
                        "A": (set(adm12["H_A"]) if adm12 else {k_ for k_ in inp["pool_old"] if k_.startswith("honest_c|")} if bo == "S_old"
                              else {f"honest_a|{c_}" for c_ in hcells}),  # (the pool's key list, H_A's rule, H_sim's rows and S_paper493's zero-unit keys)
                        "S": hk_adm | set(miss_h)}  # G7b FOLD-MODEL: H_sim's runs with rows before the fold -1 rule (= hk without one)
                zr_ = {nm: adr_[nm] - set(PzS[nm][0].tolist()) for nm in ("B", "A", "S")}  # its zero-unit runs, and its structural no-row cells
                sx_ = {nm: {c_ for c_ in PWM["structural_cells"] if NM and c_ in inp["canon"]} for nm in ("B", "S")}  # Codex r42 #2: the all-origins map
                zu0_, su0_ = PWZ.setdefault(pk_, ({"B": {}, "A": {}, "S": {}}, {"B": set(), "S": set()}))  # zu0_: role -> {zero-unit run: its admitting
                # sets} (gate 1 and the manifest's patterns); every role present, as the runner's lists (Codex r45 #1)
                for nm, v_ in zr_.items():
                    for r0 in v_:
                        zu0_.setdefault(nm, {}).setdefault(r0, set()).add(aob(orig).split(" ", 1)[0])
                for nm, v_ in sx_.items():
                    su0_.setdefault(nm, set()).update(v_)
                why_ = ((["e12 admitted runs without rows"] if e12_miss else []) + sp_w + ([f"{len(missing)} attack runs missing or non-finite"] if missing else [])
                        + ([f"{len(nf_bad)} no-filter excluded cells scored outside a structural class"] if nf_bad else [])
                        + ([f"{len(zero_with_rows)} structural-zero runs with rows"] if zero_with_rows else [])
                        + [f"{len(x_)} {t_}" for t_, x_ in (("excluded cells with H_sim rows", excl_rows), ("replaced keys outside their root", rep_bad),  # Codex r42 #1
                                                            ("S4 orphan rows taken", orphan_hits),  # (r43: replaced keys without rows is the runner's diagnostic)
                                                            ("excluded S_paper493 pool keys with sim-honest rows", excl_kept if SPO else [])) if x_])
                if V.get("e12_reader_missing"):  # G7b FOLD-MODEL (l.1466-1468): an e12 build withholds a family without an epoch-2 / fill-in reader
                    why_.append("e12: no epoch-2 / fill-in reader for this probe family")
                xadm_ = {nm: len(set(PzS[nm][0].tolist()) - adr_[nm]) for nm in ("B", "A", "S")}  # G7b (l.1977-1979): every run with units is admitted
                why_ += ([f"ha25f roots whose keys differ from the config's or are already present: {ha25_bad}"] if ha25_bad else []) + (
                    [f"{len(nocell_h)} sim-honest keys whose cell is not in the union stream's map, e.g. {nocell_h[:3]}"] if nocell_h else []) + (
                    [f"pool runs with units outside the pool's admitted runs {xadm_}"] if any(xadm_.values()) else []) + (
                    [f"S_paper493 cohort cells whose main is outside the union-stream draws ({len(offu_)})"]
                    if SPO and "merge_mains" in inp and (offu_ := [c_ for c_ in coh_pre if inp["canon"].get(c_) not in set(inp["merge_mains"])]) else [])  # l.1866
                if FIVE:  # G7b FOLD-MODEL: the runner's fold-model withhold / STOP rules before its export pass returns (each one STOPs the pooled build)
                    LB_ = inp["by_label"][lab]
                    fw_ = ([f"A25 r36: cells whose S_old rows carry two folds {fi_['r36_cells_with_two_real_folds'][:3]} (Claude N2: a gate at 0)"]
                           if fi_.get("r36_cells_with_two_real_folds") else []) + (
                        ["Sol / c_asis on S_newmains without the held-out fold gate (l.1338)"] if V.get("nm_sol_withheld") else []) + (
                        ["a pool without mapped units (25h §1; l.1834-1841)"] if five_.get("withheld") else []) + (
                        [f"fold -1 cells or pool units dropped ({ {k_: v_['n'] for k_, v_ in five_['dropped'].items()} }; NO_FOLD_DROPS, l.1925)"]
                        if any(v_["n"] for v_ in five_["dropped"].values()) and lab.rsplit(" ", 1)[0] in inp["no_fold_drops"] else []) + (
                        [] if ((V.get("pw_model") or {}).get("fit_record") or {}).get("sha256") else ["a fold-model probe without a pinned fit record (l.2166)"])
                    hac_ = set(cell[pools["A"]].tolist())
                    hsc_ = {x.split("|", 1)[1] for x in hk.tolist()} | {x.split("|", 1)[1] for x in miss_h}
                    if LB_["own_all_family"] and bo != "S_old" and hac_ != hsc_:  # l.1994-1999
                        fw_.append(f"own_all: H_A cells != H_sim cells by key (only H_A {len(hac_ - hsc_)}, only H_sim {len(hsc_ - hac_)})")
                    fw_ += fold_pool_stops({nm: PzS[nm] for nm in ("B", "A", "B2", "S", "Snr", "Anr") if nm in PzS}, five_["use21b"], LB_["oof"])
                    why_ += fw_
                    f1_ = (f1cols_nos4(cell, world, fold, isN, V["cohort_full"], five_["use21b"], inp["canon"], inp["merge_mains"]) if V.get("no_s4")
                           else f1cols_of(rfold, coh, inp["canon"], inp["merge_mains"]))  # l.2146-2160 on this all-origins row (without S4: the with-S4
                    # cohort's attack fold -1 cells under 21b, Astra r1 #2); unioned per label x combo
                    pf1_ = PWF1.setdefault(pk_, {"B": set(), "A": set(), "S": set()})
                    for nm, v_ in f1_.items():
                        pf1_[nm] |= v_
                    fi_ |= {"use21b": bool(five_["use21b"]), "units_mapped_vs_fold_minus1": five_.get("mvf"),
                            "fold_minus1_dropped": {k_: v_["n"] for k_, v_ in five_["dropped"].items()},
                            "pool_folds": {nm: {str(j): int((PzS[nm][2] == j).sum()) for j in range(-1, 5)} for nm in ("B", "A", "S")},
                            "attack_fold_minus1_cols": {nm: len(v_) for nm, v_ in f1_.items()}, "oof_label": bool(LB_["oof"]),
                            "own_all_H_A_vs_H_sim_cells": {"H_A": len(hac_), "H_sim": len(hsc_)} if LB_["own_all_family"] else None, "withhold_or_stop": fw_}
                if why_:  # Codex r40 #5 / r41 #1: a member that fails an admission or availability check (the runner withholds or STOPs): the gate fails
                    PWH.setdefault(pk_, []).append(f"{orig}: " + "; ".join(why_))
                continue
            res = {"row": lab, "origin": orig, "cohort": len(coh),  # G7b VALUE: D_o in the runner's form (W[0, ccol] * wcell).sum(1), l.2246
                   "mains": (float(np.array([[V["ao_w"][c] for c in coh]], dtype=np.float64).sum(1)[0]) if AO else len(mains)), "attack_runs_missing": missing, "orphan_rows_taken": orphan_hits,
                   "excluded_cells_with_H_sim_rows": excl_rows, "replaced_keys_outside_their_root": rep_bad, "replaced_keys_without_rows": rep_missing,
                   "replacements": {"keys": len(rep_root), "roots": sorted(set(rep_root.values())), "H_sim_rows_of_replaced_keys": rep_rows}}
            hcells_rows = {x.split("|", 1)[1] for x in hk.tolist()}  # H_sim's cells with rows, after the fold rules
            hcells_now = hcells_rows | {x.split("|", 1)[1] for x in miss_h}  # S_paper493: its admitted zero-unit runs' cells too (as row())
            if bo != "S_old":
                hac = set(cell[pools["A"]].tolist())
                res["H_A_cells_vs_H_sim_cells"] = {"H_A": len(hac), "H_sim": len(hcells_now), "H_sim_cells_without_honest_a_rows": len(hcells_now - hac)}
                cmp_pre = [("H_A cells = H_sim cells (by key)", hac == hcells_now, True)]
            else:
                cmp_pre = []
            r_ = rec["rows"][lab][orig]
            if SPO and ("withheld" in r_ or sp_w):  # S_paper493: withheld exactly when one of the runner's fail-closed conditions holds
                res |= {"withheld": r_.get("withheld"), "sp_withheld_reasons": sp_w, "n_compared": 1,
                        "mismatches": [] if ("withheld" in r_ and sp_w and not zero_with_rows) else [["S_paper493 withheld", sp_w, r_.get("withheld")]]}
                if sp_w:
                    held_rows[orig] = "; ".join(sp_w)
                out["rows"].append(res)
                continue
            if e12_miss:  # G2 (Codex r35): an admitted epoch-2 run without rows: the runner withholds the row (or STOPs a headline family outside TEST)
                res |= {"withheld": r_.get("withheld"), "e12_admitted_without_rows": e12_miss, "n_compared": 1,
                        "mismatches": [] if ("withheld" in r_ and not zero_with_rows) else [["e12 admitted runs without rows", e12_miss, r_.get("withheld")]]}
                held_rows[orig] = "e12: admitted epoch-2 runs without rows"
                out["rows"].append(res)
                continue
            if "withheld" in r_ or five_.get("withheld"):  # a withheld row must be withheld here too (25h §1: a pool without mapped units)
                ok_w = ("withheld" in r_ and bool(five_.get("withheld")) and r_.get("units_mapped_vs_fold_minus1") == five_.get("mvf")
                        and not zero_with_rows)
                res |= {"withheld": r_.get("withheld"), "mismatches": [] if ok_w else [["withheld", bool(five_.get("withheld")), r_.get("withheld")]], "n_compared": 1}
                if five_.get("withheld"):
                    held_rows[orig] = "a pool without mapped units"
                out["rows"].append(res)
                continue
            cmp = list(cmp_pre)
            if W2 and W2["kind"] == "pertask":  # rc1b W2a (PRESTATED_W2 section 1 (a)): Stage D's rows hold the per-task entries only, no per-level
                assert PWM and pass_ == "scored", (lab, orig, "rc1b W2a: a per-task row outside the pooled scored pass (STOP)")  # values
                fpt_ = float(W2["f"])
                Tpt_ = {}
                for nm in ("B", "S"):  # the runner's draw-0 tau on each pipeline pool (its D1 read, L2671-2676, of A's file; recomputed here as wave 1 does)
                    ps_, _, okz_ = ZP[nm]
                    assert okz_, (lab, orig, nm, "STOP: PERTASK draw-0 moments invalid (the runner STOPs, L2681-2682)")
                    Tpt_[nm] = float(wthr0(ps_, fpt_))
                    assert np.isfinite(Tpt_[nm]), (lab, orig, nm, "STOP: PERTASK draw-0 threshold not finite (the runner STOPs, L2684)")
                zr_pt_ = {w: zrun(w, nm) for w, nm in W2_PIPE}
                if W2.get("plant_score") and W2["plant_score"]["origin"] == orig:  # AMEND1 section 4 (a): the control call's score plant, this row only
                    def zplus_(w, i, nm_of={w0_: n0_ for w0_, n0_ in W2_PIPE}):  # that run's max raw unit +1.0, through zatt (zrun_of)
                        rm_ = np.array(rmax[w], dtype=np.float64, copy=True)
                        rm_[i] += 1.0
                        return float(zrun_of(rm_, rfold[w], runits[w], ZP[nm_of[w]][1] if FIVE else None, FIVE, "w2 score plant")[i])
                    zr_pt_, W2CTL["score"] = w2_score_plant(zr_pt_, coh, ZERO, W2["plant_score"], zplus_, rfold, FIVE)
                mine_pt_ = w2_pertask_entries(orig, coh, w2_pt_main_map(coh, main_of, inp.get("canon"), AO), zr_pt_, Tpt_, {nm: ZP[nm][0] for nm in ("B", "S")}, ZERO)  # AMEND8
                rpt_ = list(r_.get("per_task") or [])
                if W2.get("plant_record") and W2["plant_record"]["origin"] == orig:  # AMEND1 section 4 / AMEND2-3: the record-side plants (a copy)
                    rpt0_ = rpt_
                    rpt_, W2CTL["record"] = w2_record_plants(rpt0_, W2["plant_record"])
                    W2CTL["record"]["added"] = w2_added_check(mine_pt_, rpt0_)  # AMEND6 (Codex W2 r1 #4): the added entry on its own copy
                pc_, nv_ = w2_pertask_cmp(mine_pt_, rpt_)
                cmp += pc_
                cmp += [("per_task|f", fpt_, r_.get("f")), ("per_task|cohort_cells", len(coh), r_.get("cohort_cells")),
                        ("per_task|tau keys", ["B", "S"], sorted(r_.get("tau") or {}))]
                cmp += [(f"per_task|tau|{nm}", Tpt_[nm], (r_.get("tau") or {}).get(nm)) for nm in ("B", "S")]
                g0_ = {"run1@B": catch("run1", "B", Tpt_["B"]), "run2@B": catch("run2", "B", Tpt_["B"]), "sim@S|primary": catch("sim", "S", Tpt_["S"])}
                cmp.append(("per_task|draw0_catch keys", sorted(g0_), sorted(r_.get("draw0_catch") or {})))
                cmp += [(f"per_task|draw0_catch|{k_}", v_, (r_.get("draw0_catch") or {}).get(k_)) for k_, v_ in g0_.items()]
                pin_w_ = (W2.get("d1_pins") or {}).get(f"{lab} | {V['pw_combo']}")  # A's D1 file for this label x combo, by path and sha256
                rpin_ = dict(r_.get("pw_calib") or {})
                if W2.get("plant_pin") and W2["plant_pin"]["origin"] == orig:  # AMEND1 section 4: the pin fixture (a copy)
                    rpin_, W2CTL["pin"] = rpin_ | {"sha256": "0" * 64}, {"origin": orig}
                cmp.append(("per_task|pw_calib pin = A's D1 file", pin_w_, rpin_))
                W2N[lab] = W2N.get(lab, 0) + nv_
                res |= compare(cmp, not coh) | {"empty_block": not coh, "w2_values_compared": nv_}
                res["n_compared"] = len(cmp)
                out["rows"].append(res)
                continue
            if W2 and W2["kind"] == "kappa":  # rc1b W2b (PRESTATED_W2 section 1 (b)): kappa's per-row per-draw 2x2 margins vs its row store
                assert PWM and pass_ == "scored" and BOOT and BW is not None, (lab, orig, "rc1b W2b: a kappa row outside the pooled scored pass (STOP)")
                cb_ = V["pw_combo"]
                ucol_ = BW["ucol"]
                if (lab, cb_) not in BPOOL:  # the same lines as the G7b value path's pooled per-draw calibration above (the runner's D1 wave 1)
                    f1p_ = PWM["files"][f"{lab} | {cb_}"]["f1cols"]
                    colp_ = {nm: np.array([ucol_[inp["canon"][r0.split("|", 1)[1]]] for r0 in Pz[nm][0].tolist()], dtype=int) for nm in ("B", "A", "S")}
                    bp_ = {}
                    for md_ in ("joint", "fixed"):
                        for i_, b_ in enumerate(BW["ids"]):
                            if md_ == "fixed" and i_ > 0:
                                bp_[(md_, b_)] = bp_[(md_, BW["ids"][0])]
                                continue
                            wr_ = BW["W"][i_] if md_ == "joint" else np.ones(BW["W"].shape[1])
                            zB_ = pool_zw(Pz["B"], FIVE, OOF_, wr_[colp_["B"]], float(wr_[np.asarray(f1p_["B"], dtype=int)].sum()))
                            zS_ = pool_zw(Pz["S"], FIVE, OOF_, wr_[colp_["S"]], float(wr_[np.asarray(f1p_["S"], dtype=int)].sum()))
                            zA_ = pool_zw_with(Pz["A"], zB_[:3], wr_[colp_["A"]], float(wr_[colp_["B"][Pz["B"][2] == -1]].sum()), f"{lab} {cb_}")
                            zz_ = {"B": zB_, "A": zA_, "S": zS_}
                            bp_[(md_, b_)] = {"mom": {nm: zz_[nm][1] for nm in zz_},
                                              "T": {f"{lv:g}": {nm: (wthr_w(zz_[nm][0], wr_[colp_[nm]], lv) if zz_[nm][2] else float("nan")) for nm in zz_}
                                                    for lv in levels}}
                    BPOOL[(lab, cb_)] = bp_
                bp_ = BPOOL[(lab, cb_)]
                ccol_ = np.array([ucol_[inp["canon"][c]] for c in coh], dtype=int)
                wcell_ = np.array([V["ao_w"][c] for c in coh], dtype=np.float64)
                cmp.append(("boot|draws_sha256 (the regenerated union stream = this row's attestation)", BW["draws_sha256"], r_.get("draws_sha256")))
                st_ = (W2.get("row_store") or {}).get(orig) or {}
                drop_ = bool(W2.get("plant_store") and W2["plant_store"].get("drop_origin") == orig)  # AMEND1 section 4 (b): one stored row dropped
                cmp.append(("kappa|stored rows (two-way)", True, bool(st_.get("paths")) and not drop_))
                if st_.get("paths") and not drop_:
                    obj_, sp_, sh_, nca_ = w2_load_stored_row(st_["paths"], lambda p_: (Path("/sol") / p_).read_bytes())
                    assert obj_.get("origin") == orig and obj_.get("label") in st_["labels"], (lab, orig, "rc1b W2b: a stored row of another row (STOP)")
                    w2_stored_tie(obj_, r_)  # AMEND5 section 1: the record's own attempt (resources and draws_sha256), else STOP
                    K_ = {md_: {k_: np.asarray(v_, dtype=np.float64) for k_, v_ in (obj_.get("_kappa") or {}).get(md_, {}).items()} for md_ in ("joint", "fixed")}
                    assert all(sorted(K_[md_]) == sorted(W2_KAPPA_KEYS) and all(v_.shape == (BOOT_D + 1,) for v_ in K_[md_].values()) for md_ in K_), \
                        (lab, orig, "rc1b W2b: the stored margins are not the 7 arrays per mode over D + 1 draws (STOP)")
                    res["w2_stored_row"] = {"path": sp_, "sha256": sh_, "n_files": len(st_["paths"]), "n_complete_attempts": nca_,
                                            "tied_to_record": ["resources", "draws_sha256"]}  # AMEND5 section 1: keys, the full sha256 and counts only
                    if W2.get("plant_store") and W2["plant_store"]["origin"] == orig:  # AMEND1 section 4 (b): 14 plants on an in-memory copy
                        K_, W2CTL["store"] = w2_store_plants(K_, BW["ids"], W2["plant_store"])
                        W2CTL["store"]["dropped"] = W2["plant_store"]["drop_origin"]
                    flip_ = None
                    if W2.get("plant_margin") and W2["plant_margin"]["origin"] == orig:  # AMEND1 section 4 (b): the flip, in this call's own hx only
                        flip_ = w2_flip_pick(bp_, BW, ccol_, wcell_, len(coh), W2["plant_margin"])
                        W2CTL["flip"] = {k_: flip_[k_] for k_ in ("mode", "draw", "eligible")} | {"cell": coh[flip_["cell"]] if flip_["eligible"] else None}
                    nv_ = 0
                    for md_ in ("joint", "fixed"):
                        for i_, b_ in enumerate(BW["ids"]):
                            wr_ = BW["W"][i_]
                            mom_ = bp_[(md_, b_)]["mom"]
                            T_ = bp_[(md_, b_)]["T"]["0.01"]
                            fl_ = flip_["cell"] if (flip_ and flip_["eligible"] and flip_["mode"] == md_ and flip_["draw"] == b_) else None
                            mg_ = w2_margins(zrun_m("run1", mom_["B"]), zrun_m("run2", mom_["B"]), zrun_m("sim", mom_["S"]), T_["B"], T_["S"], wr_, ccol_, wcell_, fl_)
                            cmp += [(f"kappa|{md_}|draw {b_}|{k_}", mg_[k_], float(K_[md_][k_][b_])) for k_ in W2_KAPPA_KEYS]
                            nv_ += len(W2_KAPPA_KEYS)
                    W2N[lab] = W2N.get(lab, 0) + nv_
                    res["w2_values_compared"] = nv_
                res |= compare(cmp, not coh) | {"empty_block": not coh}
                res["n_compared"] = len(cmp)
                out["rows"].append(res)
                continue
            T_pw = {}
            for md_ in ("joint", "fixed"):  # the resolved levels, not the record's, decide what is compared
                cmp.append((f"levels|{md_}", lvk_all, sorted(r_["points"].get(md_, {}))))  # B's split: the full set (unsplit: as before)
            for lv in levels:
                key = f"{lv:g}"
                pz = {}
                for nm, P_ in Pz.items():
                    pk = P_[0]
                    ps, _, okz = ZP[nm]
                    assert not okz or np.isfinite(ps).all(), (lab, orig, nm, "non-finite pool z")
                    srt = np.sort(ps)[::-1]
                    kk = int(np.floor(lv * len(srt) + 1e-9))
                    T = float(srt[min(kk, len(srt) - 1)]) if (kk >= 1 and okz) else float("nan")
                    if PWM and len(ps):  # G7b VALUE: wthr (l.1210-1229) ported line by line at the draw-0 unit weights, beside the order statistic above
                        T_w = wthr0(ps, lv) if okz else float("nan")
                        assert eq(T_w, T) or (math.isnan(T_w) and math.isnan(T)), (lab, orig, nm, key, "wthr port and order statistic differ")
                        T = T_w
                    rm = {}
                    for r0, v0 in zip(pk.tolist(), ps.tolist()):
                        rm[r0] = max(rm.get(r0, -np.inf), v0)
                    pz[nm] = {"T": T, "k": kk, "n_units": int(len(srt)), "units_above": int((ps > T).sum()), "runs": len(rm),
                              "per_run_alarm_share": float(np.mean([v0 > T for v0 in rm.values()]))}
                    if PWM and nm in ("B", "A", "S"):  # G7e (2e 13:5x; RUN-LENGTH): the registered honest share counts the UNION's admitted zero-unit runs
                        zu_, su_ = PWZ[pk_]  # and structural no-row cells (by key, deduped) as non-alarming runs, against the pooled pool's own runs
                        z0_ = set(zu_.get(nm, [])) - set(rm)
                        s0_ = {c_ for c_ in su_.get(nm, []) if not any(r0.split("|", 1)[1] == c_ for r0 in rm)}
                        pz[nm] |= {"per_run_alarm_share_with_no_unit_runs": float(sum(v0 > T for v0 in rm.values()) / max(len(rm) + len(z0_) + len(s0_), 1)),
                                   "no_unit_runs_counted_non_alarming": {"zero_unit_admitted": len(z0_), "structural_no_row_cells": sorted(s0_)}}
                T_pw[key] = {nm: pz[nm]["T"] for nm in ("B", "A", "S")}  # G7: the draw-0 cutoffs (the pooled rows' cutoffs_draw0)
                c = {"run1@B": catch("run1", "B", pz["B"]["T"]), "run2@B": catch("run2", "B", pz["B"]["T"]), "run2@A": catch("run2", "A", pz["A"]["T"]),
                     "run1@A": catch("run1", "A", pz["A"]["T"]), "sim@B": catch("sim", "B", pz["B"]["T"]), "sim@S|primary": catch("sim", "S", pz["S"]["T"])}
                if PT_ is not None:  # 09 (c): the would-be planted run's draw-0 hit at tau_B, from this (unplanted) call's own cell hits; None: no finite tau_B
                    h_ = CH.get((PT_[0], "B"))
                    PRE_[key] = None if h_ is None else bool(h_.get(PT_[1], False))
                c["real_mean@B"] = (c["run1@B"] + c["run2@B"]) / 2
                if "B2" in pz:
                    c["run1@B2"] = catch("run1", "B2", pz["B2"]["T"])
                y1 = c.get("run1@B2", c["run1@B"])
                g = {"delta_full": c["sim@S|primary"] - c["real_mean@B"], "delta_rr": c["run2@A"] - y1, "same_attack": c["run1@A"] - y1,
                     "real_line": c["sim@B"] - c["real_mean@B"], "sim_minus_run1": c["sim@S|primary"] - c["run1@B"],
                     "sim_minus_run2": c["sim@S|primary"] - c["run2@B"]}
                if CAP == "paired":  # same lines as the base arms; only the attack units differ
                    c |= {"sim@S|p1": catch("sim_p1", "S", pz["S"]["T"]), "run1@B|p1": catch("run1_p1", "B", pz["B"]["T"]),
                          "sim@S|p2": catch("sim_p2", "S", pz["S"]["T"]), "run2@B|p2": catch("run2_p2", "B", pz["B"]["T"])}
                    g |= {"sim_minus_run1_paired": c["sim@S|p1"] - c["run1@B|p1"], "sim_minus_run2_paired": c["sim@S|p2"] - c["run2@B|p2"]}
                    g["delta_full_paired"] = (g["sim_minus_run1_paired"] + g["sim_minus_run2_paired"]) / 2
                opt_c, opt_g = {}, {}  # replacement removal (scenario levels only, where the record has them)
                if "Snr" in pz:
                    opt_c = {"sim@S|removal": catch("sim", "Snr", pz["Snr"]["T"]), "run2@A|removal": catch("run2", "Anr", pz["Anr"]["T"])}
                    opt_g = {"delta_full_removal": opt_c["sim@S|removal"] - c["real_mean@B"], "delta_rr_removal": opt_c["run2@A|removal"] - y1}
                if "R3" in pz:  # G3: every arm at T_alt (Append 25b): threshold swap = real at T_alt - real at T_ref; full analogue = sim at T_alt - real at T_ref
                    c |= {"run1@R3": catch("run1", "R3", pz["R3"]["T"]), "run2@R3": catch("run2", "R3", pz["R3"]["T"]), "sim@R3": catch("sim", "R3", pz["R3"]["T"])}
                    c["real_mean@R3"] = (c["run1@R3"] + c["run2@R3"]) / 2
                    g |= {"threshold_swap_R3": c["real_mean@R3"] - c["real_mean@B"], "full_analogue_R3": c["sim@R3"] - c["real_mean@B"]}
                mine[(lab, orig, key)] = c
                mine_mainof[(lab, orig)] = {c_: (inp["canon"][c_] if AO else inp["main_t"][c_] if bo == "S_old" else main_of[c_]) for c_ in coh}  # canonical (Codex r7)
                cmap_ = {"sim@S|primary": ("sim", "S"), "run1@B": ("run1", "B"), "run2@B": ("run2", "B"), "run2@A": ("run2", "A"), "run1@B2": ("run1", "B2"),
                         "sim@S|p1": ("sim_p1", "S"), "run1@B|p1": ("run1_p1", "B"), "sim@S|p2": ("sim_p2", "S"), "run2@B|p2": ("run2_p2", "B"),
                         "sim@B": ("sim", "B")}  # sim@B: the pooled real_line (sim at the pooled tau_B)
                mine_cells[(lab, orig, key)] = {x: CH[wn] for x, wn in cmap_.items() if wn in CH}
                mine_rd[(lab, orig, key)] = {nm: v["k"] >= 1 and v["units_above"] >= 10 for nm, v in pz.items() if nm not in ("Snr", "Anr", "R3")}
                pj = r_["points"]["joint"][key]
                cmp += [(f"{key}|catch|{x}", v, pj["catch"].get(x, {}).get("point")) for x, v in c.items()]
                cmp += [(f"{key}|gap|{x}", v, pj["gap"].get(x, {}).get("point")) for x, v in g.items()]
                opt_ = [(f"{key}|catch|{x}", v, pj["catch"][x]["point"]) for x, v in opt_c.items() if x in pj["catch"]]
                opt_ += [(f"{key}|gap|{x}", v, pj["gap"][x]["point"]) for x, v in opt_g.items() if x in pj["gap"]]
                cmp += opt_
                res["removal_comparisons"] = res.get("removal_comparisons", 0) + len(opt_)
                pf = r_["points"].get("fixed", {}).get(key, {})  # fixed mode: the same draw-0 values; every primary/paired field required
                cmp += [(f"{key}|fixed|catch|{x}", v, pf.get("catch", {}).get(x, {}).get("point")) for x, v in c.items()]
                cmp += [(f"{key}|fixed|gap|{x}", v, pf.get("gap", {}).get(x, {}).get("point")) for x, v in g.items()]
                cmp += [(f"{key}|fixed|catch|{x}", v, pf["catch"][x]["point"]) for x, v in opt_c.items() if x in pf.get("catch", {})]
                cmp += [(f"{key}|fixed|gap|{x}", v, pf["gap"][x]["point"]) for x, v in opt_g.items() if x in pf.get("gap", {})]
                for nm, v in pz.items():  # the removal pools (Snr, Anr) and the alternate R3 carry T, k, units and units above only, and no readable flag
                    rm_ = nm in ("Snr", "Anr", "R3")
                    cmp += [(f"{key}|pool|{nm}|{x}", v[x], r_["pools_at_draw0"][key].get(nm, {}).get(x)) for x in v
                            if not rm_ or x in ("T", "k", "n_units", "units_above")]
                    if not rm_:
                        cmp.append((f"{key}|readable|{nm}", v["k"] >= 1 and v["units_above"] >= 10, r_["readable"][key].get(nm)))
                if set(r_["pools_at_draw0"][key]) != set(pz):
                    cmp.append((f"{key}|pool names", sorted(pz), sorted(r_["pools_at_draw0"][key])))
            hb3_ = r_.get("honest_b_relaunch") or {}
            cmp.append(("R3 pool computed (honest_b_relaunch)", "R3" in Pz, "pools_check" in hb3_ and "withheld" not in hb3_))
            if r3_check is not None:  # G3: the record's own pools check, by key counts (its mains-outside-universe list is not recomputed here)
                cmp += [(f"honest_b_relaunch|pools_check|{x}", v, (hb3_.get("pools_check") or {}).get(x)) for x, v in r3_check.items()]
            if alias3 is not None:  # whenever the uncut relaunch files were read, even with no rows (the runner records the count before that test)
                cmp.append(("honest_b_relaunch_key_alias|renamed_rows", alias3, (r_.get("honest_b_relaunch_key_alias") or {}).get("renamed_rows")))
            cv = r_["coverage"]
            cmp += [("coverage|cohort_cells", len(coh), cv["cohort_cells"]), ("coverage|H_sim_runs", len(set(hk.tolist())), cv["H_sim_runs"]),
                    ("coverage|H_sim_units", int(len(hk)), cv["H_sim_units"]), ("coverage|H_B_runs", len(set(PzS["B"][0].tolist())), cv["H_B_runs"]),
                    ("coverage|H_B_units", int(len(PzS["B"][0])), cv["H_B_units"]), ("coverage|H_A_runs", len(set(PzS["A"][0].tolist())), cv["H_A_runs"]),
                    ("coverage|H_A_units", int(len(PzS["A"][0])), cv["H_A_units"]),
                    ("coverage|H_sim_cells_in_cohort", len(hcells_rows & set(coh)), cv["H_sim_cells_in_cohort"])]
            if PWM:  # G7: the row's pooled_world block: combo, pools file, the union's sizes after dedup, the draw-0 cutoffs and calibration mass
                pwr_, sz_ = r_.get("pooled_world") or {}, PWU[pk_][1]  # (draw 0: every multiplicity 1, so the mass is the unit count per membership pattern)
                mupf_ = ({nmr: {str(j): int((Pz[nm][2] == j).sum()) for j in range(5)} | {"fold_minus1": int((Pz[nm][2] == -1).sum())}  # G7b VALUE: the UNION's
                          for nm, nmr in (("B", "H_B"), ("A", "H_A"), ("S", "H_sim"))} if FIVE else "single model")  # (pooled pools') counts, l.2457-2458
                cmp += [("pooled_world|combo", V["pw_combo"], pwr_.get("combo")), ("pooled_world|mapped_units_per_fold", mupf_, pwr_.get("mapped_units_per_fold")),
                        ("pooled_world|pools_file", PWM["files"].get(f"{lab} | {V['pw_combo']}"), pwr_.get("pools_file"))]
                cmp += [(f"pooled_world|pools|{nmr}|{x}", sz_[nm][x], ((pwr_.get("pools") or {}).get(nmr) or {}).get(x))
                        for nm, nmr in (("B", "H_B"), ("A", "H_A"), ("S", "H_sim")) for x in ("runs", "units")]
                cm_ = pwr_.get("calibration_mass_draw0") or {}
                fk_ = [str(j) for j in range(5)] + ["-1"] if FIVE else ["0"]  # G7b VALUE: l.2564-2566, folds + [-1] for a fold-model probe
                for nm in ("B", "A", "S"):
                    if sorted(cm_.get(nm) or {}) != sorted(sz_[nm]["pattern_units"]):
                        cmp.append((f"pooled_world|calibration_mass_draw0|{nm}|patterns", sorted(sz_[nm]["pattern_units"]), sorted(cm_.get(nm) or {})))
                    if FIVE:  # the union's units per membership pattern and fold (draw 0: every weight 1)
                        pat_u_ = np.array([PWU[pk_][2][nm][r0] for r0 in Pz[nm][0].tolist()], dtype=str)
                        cmp += [(f"pooled_world|calibration_mass_draw0|{nm}|{p_}|{fj_}", float(((pat_u_ == p_) & (Pz[nm][2] == int(fj_))).sum()),
                                 ((cm_.get(nm) or {}).get(p_) or {}).get(fj_)) for p_ in sz_[nm]["pattern_units"] for fj_ in fk_]
                    else:
                        cmp += [(f"pooled_world|calibration_mass_draw0|{nm}|{p_}", float(u_), ((cm_.get(nm) or {}).get(p_) or {}).get("0"))
                                for p_, u_ in sz_[nm]["pattern_units"].items()]
                    cmp += [(f"pooled_world|calibration_mass_draw0|{nm}|{p_}|folds", sorted(fk_), sorted((cm_.get(nm) or {}).get(p_) or {})) for p_ in sz_[nm]["pattern_units"]]
                cu_ = pwr_.get("cutoffs_draw0") or {}
                cmp.append(("pooled_world|cutoffs_draw0|levels", sorted(set(T_pw) | (set(lvk_all) - {f"{x:g}" for x in levels})), sorted(cu_)))  # B's split: this part's
                # computed levels plus the other parts' (from the full set) vs the record's cutoffs: equal iff this part is computed and the record holds
                # exactly the full set (unsplit: sorted(T_pw) vs sorted(cu_), as before)
                for key, t_ in T_pw.items():
                    for nm in ("B", "A", "S"):
                        cq_ = cu_.get(key, {}).get(nm) or {}
                        mom_ = ZP[nm][1] if FIVE else None
                        if mom_ is None:  # pw_cutoff l.926-934: a single-model probe's raw cutoff
                            cmp += [(f"pooled_world|cutoffs_draw0|{key}|{nm}", t_[nm], cq_.get("raw_cutoff")),
                                    (f"pooled_world|cutoffs_draw0|{key}|{nm}|fields", ["raw_cutoff"], sorted(cq_))]
                            continue
                        MU0_, SD0_, mu00_, sd00_ = mom_  # a fold-model probe: T_z, per-fold mu / sd / raw cutoff, the fold -1 transform (H_A: H_B's)
                        cmp += [(f"pooled_world|cutoffs_draw0|{key}|{nm}|fields", ["T_z", "fold_minus1_transform", "per_fold"], sorted(cq_)),
                                (f"pooled_world|cutoffs_draw0|{key}|{nm}|T_z", t_[nm], cq_.get("T_z")),
                                (f"pooled_world|cutoffs_draw0|{key}|{nm}|folds", [str(j) for j in range(5)], sorted((cq_.get("per_fold") or {})))]
                        for j in range(5):
                            pq_ = (cq_.get("per_fold") or {}).get(str(j)) or {}
                            cmp += [(f"pooled_world|cutoffs_draw0|{key}|{nm}|{j}|mu", float(MU0_[j]), pq_.get("mu")),
                                    (f"pooled_world|cutoffs_draw0|{key}|{nm}|{j}|sd", float(SD0_[j]), pq_.get("sd")),
                                    (f"pooled_world|cutoffs_draw0|{key}|{nm}|{j}|raw_cutoff", float(MU0_[j] + t_[nm] * SD0_[j]), pq_.get("raw_cutoff"))]
                        tq_ = cq_.get("fold_minus1_transform")
                        if mu00_ is None:
                            cmp.append((f"pooled_world|cutoffs_draw0|{key}|{nm}|fold_minus1_transform", None, tq_))
                        else:
                            cmp += [(f"pooled_world|cutoffs_draw0|{key}|{nm}|mu0", float(mu00_), (tq_ or {}).get("mu0")),
                                    (f"pooled_world|cutoffs_draw0|{key}|{nm}|sd0", float(sd00_), (tq_ or {}).get("sd0"))]
            if NM:
                cmp += [("s_newmains|cohort_cells", V["nm_cohort_n"], r_["s_newmains"]["cohort_cells"]), ("s_newmains|pool_honest_b", V["nm_pool_n"], r_["s_newmains"]["pool_honest_b"]),
                        ("s_newmains|s4r_lists_joined", V["use4r"], r_["s_newmains"]["s4r_lists_joined"])]
            if bo == "S_new" or UN:
                cmp.append(("ha25f roots' keys", ha25, [x["keys"] for x in r_.get("ha25f", [])]))
            if FIVE:  # fold bookkeeping the record states (the row's own per-set pools, l.2097: PzS, also on a pooled row; G7b VALUE)
                mpf = {nm_: {str(j): int((PzS[x][2] == j).sum()) for j in range(5)} | {"fold_minus1": int((PzS[x][2] == -1).sum())}
                       for nm_, x in (("H_B", "B"), ("H_A", "A"), ("H_sim", "S"))}
                cmp += [("mapped_units_per_fold", mpf, r_.get("mapped_units_per_fold")),
                        ("fold -1 rule is 21b", five_["use21b"], str(r_.get("fold_minus1_rule", "")).startswith("weighted 21b")),
                        ("fold -1 dropped (n and keys)", five_["dropped"], r_.get("fold_minus1_dropped"))]
                if NM or (SPO and PWM):  # l.1834-1844: S_paper493 too (G7b VALUE)
                    cmp.append(("units_mapped_vs_fold_minus1", five_["mvf"], r_.get("units_mapped_vs_fold_minus1")))
            for k_, v in capinfo.items():
                cmp.append((f"cap|{k_}", v, r_.get("cap", {}).get(k_)))
            if adm12 is not None:  # G2: the row's admitted calibration keys (epoch 1 + 2) by sha256, and its e1 / e2 counts per pool, as the record states them
                ks_ = r_.get("calibration_e12_key_sets") or {}
                hsh_ = lambda x: hashlib.sha256(json.dumps(sorted(x)).encode()).hexdigest()  # noqa: E731
                cmp += [(f"calibration_e12_key_sets|{nm_}", hsh_(v_), ks_.get(nm_)) for nm_, v_ in adm12.items()]
                cnt12 = {"H_B": (len(e1b), len(e2b))} | ({"H_B2": (len(e1b2), len(E12["H_B2"]))} if bo == "S_old" else {"H_A": (len(e1a), len(take))})
                pl_ = (r_.get("calibration_e12") or {}).get("pools") or {}
                cmp += [(f"calibration_e12|{nm_}|{x_}", n_, (pl_.get(nm_) or {}).get(x_)) for nm_, (a_, b_) in cnt12.items()
                        for x_, n_ in (("e1_admitted", a_), ("e2_admitted", b_))]
            if SPO:
                spr0_ = r_.get("s_paper493", {})
                cmp += [("s_paper493|cohort_cells", len(coh), spr0_.get("cohort_cells")),
                        ("s_paper493|structural_zero_runs", sorted(ZERO & {f"{w}|{c}" for w in A3 for c in coh}), spr0_.get("structural_zero_runs")),
                        ("s_paper493|H_sim_admitted_without_rows", {"n": len(miss_h), "keys": miss_h},
                         {k_: (spr0_.get("H_sim_admitted_without_rows") or {}).get(k_) for k_ in ("n", "keys")}),
                        ("s_paper493|H_sim_existing_keys", sorted(x for x, h_ in zip(hk.tolist(), hr.tolist()) if h_ not in set(V["sp_roots"])), spr0_.get("H_sim_existing_keys")),
                        ("s_paper493|excluded pool keys with sim-honest rows", excl_kept, [])]
            if AO:
                own_ = {}
                for c_ in coh:
                    own_[inp["canon"][c_]] = own_.get(inp["canon"][c_], 0) + 1
                ao0_ = r_.get("all_origins", {})
                cmp += [("all_origins|cells", len(coh), ao0_.get("cells")), ("all_origins|mains_in_this_origin", len(own_), ao0_.get("mains_in_this_origin")),
                        ("all_origins|mains_shared_with_other_origins", sum(1 for m_, n_ in own_.items() if round(1 / min(V["ao_w"][c_] for c_ in coh if inp["canon"][c_] == m_)) > n_),
                         ao0_.get("mains_shared_with_other_origins"))]
            if "Snr" in Pz and not PWM:  # the removal sensitivity must have been compared somewhere (not vacuous); pooled: no removal catch (scenarios off)
                cmp.append(("replacement removal compared at some level", res.get("removal_comparisons", 0) > 0, True))
            if CAP is not None and not capinfo:
                cmp.append(("cap applied", False, True))
            if inp.get("nf_audit") and not REG and not SPO:
                res["nofilter_excluded_but_scored"] = nf_bad
                res["nofilter_excluded_with_all_arms_scored_any_class"] = nf_all
                res["nofilter_cells_checked_not_in_a_structural_class"] = sum(1 for o_ in (["S_new", "S_newmains"] if UN else ["S_newmains"] if NM else [bo])
                                                                              for cl_ in inp["nf_audit"][o_].values() if not STR_[cl_])
                cmp.append(("no-filter: excluded cells not in a structural class with all three arms scored", len(nf_bad), 0))
            if "structural_zero_runs" in r_ and not SPO and CAP is None:  # 7ad9f0c32+: the record names its no-filter structural zeros per row (by key)
                cmp.append(("structural_zero_runs (row)", sorted(k_ for k_ in ZERO if k_.split("|", 1)[1] in set(coh)), r_["structural_zero_runs"]))
            if ZERO:
                res["structural_zero_runs_in_cohort"] = sorted(r_ for r_ in ZERO if r_.split("|", 1)[1] in set(coh))
                cmp.append(("structural-zero runs that have rows in the loaded files", zero_with_rows, []))
            cmp += [("attack runs missing or non-finite", len(missing), 0), ("S4 orphan rows taken", len(orphan_hits), 0),
                    ("excluded cells with H_sim rows", len(excl_rows), 0), ("replaced keys outside their root", len(rep_bad), 0),
                    ("replaced keys without rows", len(rep_missing), 0)]
            if V.get("expected_replacement_roots"):  # 2e 22:2x: the frozen inputs admit replacement roots; the check must find them non-empty and inside their roots
                cmp += [("replacement roots (captured) = INPUTS_FINAL pools.stage_a", sorted(set(rep_root.values())), V["expected_replacement_roots"]),
                        ("replacement keys exercised (non-empty, every key with rows)", bool(rep_root) and not rep_missing and rep_rows > 0, True)]
            if BOOT and PWM and pass_ == "scored":  # G7b VALUE (09 via 2e 23:2x): this member row at draw 0 and the seeded draws, every value per draw
                cb_ = V["pw_combo"]  # through the runner's per-draw arithmetic (ported), against the record's per-draw npz at the same draw index
                ucol_ = BW["ucol"]
                if (lab, cb_) not in BPOOL:  # the label x combo's pooled calibration per draw, once (the runner's D1 wave 1, l.2510-2589: zpool on the
                    f1p_ = PWM["files"][f"{lab} | {cb_}"]["f1cols"]  # pooled H_B and H_sim with the combo's union attack fold -1 columns, zwith on H_A,
                    colp_ = {nm: np.array([ucol_[inp["canon"][r0.split("|", 1)[1]]] for r0 in Pz[nm][0].tolist()], dtype=int) for nm in ("B", "A", "S")}
                    bp_ = {}  # validity masks with H_A's undefined transform, wthr where valid)
                    for md_ in ("joint", "fixed"):
                        for i_, b_ in enumerate(BW["ids"]):
                            if md_ == "fixed" and i_ > 0:  # mode 'fixed': Wb = ones (l.2514): the draw-0 calibration in every draw
                                bp_[(md_, b_)] = bp_[(md_, BW["ids"][0])]
                                continue
                            wr_ = BW["W"][i_] if md_ == "joint" else np.ones(BW["W"].shape[1])
                            zB_ = pool_zw(Pz["B"], FIVE, OOF_, wr_[colp_["B"]], float(wr_[np.asarray(f1p_["B"], dtype=int)].sum()))
                            zS_ = pool_zw(Pz["S"], FIVE, OOF_, wr_[colp_["S"]], float(wr_[np.asarray(f1p_["S"], dtype=int)].sum()))
                            zA_ = pool_zw_with(Pz["A"], zB_[:3], wr_[colp_["A"]], float(wr_[colp_["B"][Pz["B"][2] == -1]].sum()), f"{lab} {cb_}")
                            zz_ = {"B": zB_, "A": zA_, "S": zS_}
                            bp_[(md_, b_)] = {"mom": {nm: zz_[nm][1] for nm in zz_},
                                              "T": {f"{lv:g}": {nm: (wthr_w(zz_[nm][0], wr_[colp_[nm]], lv) if zz_[nm][2] else float("nan")) for nm in zz_}
                                                    for lv in levels}}
                    BPOOL[(lab, cb_)] = bp_
                bp_ = BPOOL[(lab, cb_)]
                ccol_ = np.array([ucol_[inp["canon"][c]] for c in coh], dtype=int)  # this row's cells -> union-stream columns, and wcell = 1 / n_cells(main)
                wcell_ = np.array([V["ao_w"][c] for c in coh], dtype=np.float64)
                cmp.append(("boot|draws_sha256 (the regenerated union stream = this row's attestation)", BW["draws_sha256"], r_.get("draws_sha256")))
                for md_ in ("joint", "fixed"):
                    for i_, b_ in enumerate(BW["ids"]):
                        wr_ = BW["W"][i_]  # the catches' main weights resample in both modes (tpr reads W, l.2205-2215)
                        mom_ = bp_[(md_, b_)]["mom"]
                        zr_ = {(w, nm): zrun_m(w, mom_[nm]) for w, nm in (("run1", "B"), ("run2", "B"), ("run2", "A"), ("run1", "A"), ("sim", "B"), ("sim", "S"))}
                        D_ = float((wr_[None, :][:, ccol_] * wcell_[None, :]).sum(1)[0])  # l.2675: D = (W[:, ccol] * wcell).sum(1)
                        for lv in levels:
                            key = f"{lv:g}"
                            T_ = bp_[(md_, b_)]["T"][key]
                            c_ = {x: boot_catch(zr_[(w, nm)], T_[nm], wr_, ccol_, wcell_) for x, (w, nm) in
                                  (("run1@B", ("run1", "B")), ("run2@B", ("run2", "B")), ("run2@A", ("run2", "A")), ("run1@A", ("run1", "A")),
                                   ("sim@B", ("sim", "B")), ("sim@S|primary", ("sim", "S")))}
                            c_["real_mean@B"] = (c_["run1@B"] + c_["run2@B"]) / 2
                            g_ = {"delta_full": c_["sim@S|primary"] - c_["real_mean@B"], "delta_rr": c_["run2@A"] - c_["run1@B"],  # y1 = run1@B (no B2 pooled)
                                  "sim_minus_run1": c_["sim@S|primary"] - c_["run1@B"], "sim_minus_run2": c_["sim@S|primary"] - c_["run2@B"],
                                  "real_line": c_["sim@B"] - c_["real_mean@B"]}
                            BROW[(lab, orig, md_, key, b_)] = (c_, D_)
                            pre_ = f"boot|{md_}|{key}|draw {b_}"
                            cmp += [(f"{pre_}|catch|{x}", c_[x], bref(lab, cb_, md_, key, aob(orig), "catch_" + x)[i_]) for x in BOOT_ROW_CATCH]
                            cmp += [(f"{pre_}|gap|{x}", g_[x], bref(lab, cb_, md_, key, aob(orig), "gap_" + x)[i_]) for x in BOOT_ROW_GAP]
                            cmp.append((f"{pre_}|D", D_, bref(lab, cb_, md_, key, aob(orig), "D")[i_]))
                            if b_ == 0:  # draw 0 through the per-draw path = this check's draw-0 path (a self-check, both values computed here)
                                cmp += [(f"{pre_}|self-check: per-draw path = draw-0 path|catch|{x}", c_[x], mine[(lab, orig, key)][x]) for x in BOOT_ROW_CATCH]
                            if PLV and md_ == "joint" and lab in PLV["labels"] and not REG and any(abs(lv - x_) <= 1e-12 * x_ for x_ in PLV["points"]):
                                # PLANT V3 blocks (ii) / (iii): the runner's plant ported (run_peraction311 @ 752413181: plant_fit L2465-2496 at draw 0;
                                # the planted arm per draw with the same c, L2862-2873; its in-row checks L2964-2966), sim arm only (PLANT_NOT_COMPARED)
                                zS_ = zr_[("sim", "S")]
                                if b_ == 0:  # the draw-0 solve on this row's own sim catch at tau_sim (ht = boot_catch at draw 0, the runner's ht_(.)[0])
                                    ht0_ = lambda zz_, T0_=T_["S"], w0_=wr_, cc_=ccol_, wc_=wcell_: boot_catch(zz_, T0_, w0_, cc_, wc_)  # noqa: E731
                                    sv_, cx_ = plant_fit_v3(zS_, T_["S"], ht0_, PLV["deltas"])
                                    if P1 and lab == P1["label"] and orig == P1["origin"] and key == P1["level"]:  # P1: c' with achieved' (AMEND5 F2)
                                        sv_, cx_, P1HIT = plant_p1_apply(sv_, cx_, zS_, T_["S"], ht0_, P1)
                                    PFIT[(lab, orig, key)] = (sv_, cx_)
                                    rsd_ = ((r_.get("plant") or {}).get("solve_draw0") or {}).get(key) or {}
                                    cmp += plant_solve_cmp(key, sv_, rsd_.get("sim"))
                                    il2_ = bo in IL_ORIGINS and orig in (PLV.get("il") or {})  # AMEND6 (Codex V3 #3): the second sim repeat (runner L2863-2869)
                                    cmp.append(plant_arms_cmp(key, il2_, rsd_))
                                    if il2_:  # sim2's z at draw 0: its run max over its admitted IL rows; the row's il_arms report (compared below, ported in
                                        # main) has no folder with this probe's file, hence no row, structural zero or fold -1 run: NaN on every cohort cell,
                                        # so plant_fit returns its 'unavailable' record at this tau_sim (an IL file with rows STOPs in main: not ported)
                                        assert PLV["il"][orig]["rows"] == 0 and not PLV["il"][orig]["structural_zero_keys"], "STOP: sim2 rows (not ported)"
                                        sv2_, cx2_ = plant_fit_v3(np.full(len(coh), np.nan), T_["S"], None, PLV["deltas"])
                                        assert not cx2_ and "arms" not in sv2_, (lab, orig, key, "STOP: a sim2 planted arm (not ported)")
                                        cmp += plant_solve_cmp(key, sv2_, rsd_.get("sim2"), "sim2")
                                        SIM2N["sim2_solves"] += int(isinstance(rsd_.get("sim2"), dict))
                                sv_, cx_ = PFIT[(lab, orig, key)]
                                pcz_ = {d_: boot_catch(zS_ + c0_, T_["S"], wr_, ccol_, wcell_) for d_, c0_ in sorted(cx_.items())}
                                BROWP[(lab, orig, key, b_)] = pcz_
                                cmp += [(f"{pre_}|plant|catch|sim_plant_{d_:+d}", v_, bref(lab, cb_, "joint", key, aob(orig), f"catch_sim_plant_{d_:+d}")[i_])
                                        for d_, v_ in pcz_.items()]
                                if 0 in pcz_:  # runner L2964-2965: the delta-0 arm equals the unplanted arm in every draw (exact; a difference STOPs)
                                    assert pcz_[0] == c_["sim@S|primary"] or (math.isnan(pcz_[0]) and math.isnan(c_["sim@S|primary"])), \
                                        (lab, orig, key, b_, "STOP: the delta-0 plant arm differs from sim@S|primary")
                                if b_ == 0:  # runner L2966: each arm's draw-0 change equals its achieved_points (exact; a difference STOPs); the row's points
                                    for d_, v_ in pcz_.items():
                                        assert 100 * (v_ - c_["sim@S|primary"]) == sv_["arms"][f"{d_:+d}"]["achieved_points"], \
                                            (lab, orig, key, d_, "STOP: the arm's draw-0 change is not the solved one")
                                    rcp_ = (((r_.get("points") or {}).get("joint") or {}).get(key) or {}).get("catch") or {}
                                    cmp += [(f"{key}|plant|catch|sim_plant_{d_:+d}", v_, (rcp_.get(f"sim_plant_{d_:+d}") or {}).get("point")) for d_, v_ in pcz_.items()]
                                    rcf_ = (((r_.get("points") or {}).get("fixed") or {}).get(key) or {}).get("catch") or {}
                                    cmp += plant_catch_keys_cmp(key, pcz_, rcp_, rcf_)  # Codex V3 r2 #3: every planted catch key (a sim2 one MISMATCHES)
            if BOOT and PWM and pass_ == "scored" and PLV and lab in PLV["labels"] and not REG:  # PLANT V3: the row's plant block fields and (AMEND6,
                pr_ = r_.get("plant") or {}  # Codex V3 #3) its il_arms report, ported in main from the pinned block (il_arms.py L341-379; runner L2205-2206)
                ilr_ = (PLV.get("il") or {}).get(orig) if bo in IL_ORIGINS else None
                cmp += [("plant|fields", ["arms", "deltas", "points", "rule", "solve_draw0"], sorted(pr_)),
                        ("plant|arms", ["sim", "sim2"] if ilr_ is not None else ["sim"], pr_.get("arms")), ("plant|deltas", PLV["deltas"], pr_.get("deltas")),
                        ("plant|points", PLV["points"], pr_.get("points")), ("plant|solve_draw0 levels", sorted(f"{x_:g}" for x_ in PLV["points"]),
                                                                               sorted(pr_.get("solve_draw0") or {})),
                        ("plant|cell_hits_draw0: no sim2 planted string (AMEND7 B)", True, not any(k_.startswith("sim2|plant") for lv_ in
                                                                                             ((r_.get("cell_hits_draw0") or {}).get("levels") or {}).values() for k_ in lv_)),
                        ] + il_cmp(ilr_, r_.get("il_arms"))
                SIM2N["il_reports"] += int(ilr_ is not None and isinstance(r_.get("il_arms"), dict))
            for lv in levels if PWM else ():  # G7b VALUE (2e 22:3x): the value comparisons per level and mode (n_compared > 0 gate below)
                key = f"{lv:g}"
                NCMP[(lab, orig, key, "joint")] = sum(1 for k0_ in cmp if k0_[0].startswith(f"{key}|") and not k0_[0].startswith(f"{key}|fixed|"))
                NCMP[(lab, orig, key, "fixed")] = sum(1 for k0_ in cmp if k0_[0].startswith(f"{key}|fixed|"))
                for md_ in ("joint", "fixed") if BOOT else ():  # the seeded draws' comparisons on this row (per mode and level)
                    NCMP[(lab, orig, key, f"boot|{md_}")] = sum(1 for k0_ in cmp if k0_[0].startswith(f"boot|{md_}|{key}|"))
            if PWM and PWM["stage"] == "scored":  # G7b VALUE (507-prep 23:11 NB; Astra r6 1): a draw-0 catch / gap / panel value or ANY seeded-draw value
                vq_ = vacuous(cmp)  # NaN on BOTH sides is vacuous: a violation by key (never a MATCH that counts)
                VNAN.extend(f"{lab} | {orig} | {k0_}" for k0_ in vq_)
                BNAN[lab] = BNAN.get(lab, 0) + sum(1 for k0_ in vq_ if k0_.startswith("boot|"))
            res |= compare(cmp, not coh) | {"empty_block": not coh}
            res["n_compared"] = len(cmp)
            if PT_ is not None:  # 09 (c): keys and booleans only (the run key, and caught at draw 0 per level)
                res["plant_precheck"] = {"arm": PT_[0], "run": f"{PT_[0]}|{PT_[1]}", "caught_at_draw0": PRE_}
            out["rows"].append(res)
        for pk_ in sorted(PWX):  # a catalogue check has no scored pass to build the unions
            if pk_[0] == lab and pk_ not in PWU:
                PWU[pk_] = pw_union_chk(PWX[pk_], f"{pk_[0]} | {pk_[1]}", PWZ.get(pk_, ({}, {}))[0])
        for (lb_, cb_), (pl_, _, pt_) in sorted(PWU.items()) if PWM else ():  # G7: the union rebuilt here against the runner's pooled pools file
            if lb_ != lab:
                continue
            fr_ = PWM["files"].get(f"{lab} | {cb_}")
            bad_ = []
            if not fr_:
                bad_.append("no pooled pools file named for this label x combo")
            else:
                b_ = (Path("/sol") / fr_["path"]).read_bytes()
                if hashlib.sha256(b_).hexdigest() != fr_["sha256"]:
                    bad_.append("pooled pools file sha256")
                else:
                    z_ = np.load(io.BytesIO(b_), allow_pickle=False)
                    for nm in ("B", "A", "S") if FIVE else ():  # G7b FOLD-MODEL: per run key, scores, folds AND model scores as canonical bytes (NaN-safe:
                        fo_, mo_, nd_ = pools_file_diff_five(tuple(z_[f"{nm}_{x}"] for x in ("run", "s", "f", "sbm", "org")), pl_[nm], pt_[nm])  # fold -1 units
                        if fo_ or mo_:  # carry NaN designated scores), and the membership pattern
                            bad_.append(f"{nm}: run keys differ ({fo_} only in the file, {mo_} only here)")
                        if nd_:
                            bad_.append(f"{nm}: {nd_} runs whose units (scores, folds, model scores) or membership pattern differ")
                    for nm in ("B", "A", "S") if not FIVE else ():  # by run key: the same runs, each with the same units in order (scores, folds) and membership pattern
                        th_, mi_ = {}, {}  # (model scores are not compared: a single-model probe carries none)
                        for d_, (k_, s_, f_, o_) in ((th_, (z_[f"{nm}_run"].astype(str), z_[f"{nm}_s"], z_[f"{nm}_f"], z_[f"{nm}_org"].astype(str))),
                                                     (mi_, (pl_[nm][0], pl_[nm][1], pl_[nm][2], np.array([pt_[nm][r] for r in pl_[nm][0].tolist()], dtype=str)))):
                            for i, r in enumerate(k_.tolist()):
                                d_.setdefault(r, ([], [], set()))
                                d_[r][0].append(float(s_[i]))
                                d_[r][1].append(int(f_[i]))
                                d_[r][2].add(str(o_[i]))
                        if set(th_) != set(mi_):
                            bad_.append(f"{nm}: run keys differ ({len(set(th_) - set(mi_))} only in the file, {len(set(mi_) - set(th_))} only here)")
                        nd_ = sum(1 for r in set(th_) & set(mi_) if th_[r][0] != mi_[r][0] or th_[r][1] != mi_[r][1] or th_[r][2] != mi_[r][2])
                        if nd_:
                            bad_.append(f"{nm}: {nd_} runs whose units or membership pattern differ")
            sz_ = PWU[(lab, cb_)][1]
            prs_ = set()  # gate 2's exemptions: the declared E12 pairs {epoch-2 key, the epoch-1 key it replicates}, every origin's pool maps (the E12 block)
            for v0_ in inp["variants"].values():
                e_ = v0_.get("e12") or {}
                for mp_ in (e_.get("H_B") or {}, e_.get("H_B2") or {}, e_.get("H_B_coverage") or {}, *(e_.get("H_A") or {}).values()):
                    prs_ |= {frozenset((k2_, r2_)) for k2_, r2_ in mp_.items()}
            dcl_ = {frozenset(g_) for g_ in inp["pw"].get("gate2_declared") or []}  # 09 (a') / 2e 20:1x: declared groups, cited by the token-prefix record
            pfs_ = None
            if dcl_:
                fl_ = set()  # this label's own (scorer, name) files: every pinned path of its row inputs (per-set and epoch-2 folders)

                def pths_(x):
                    if isinstance(x, dict):
                        for k__, v__ in x.items():
                            if k__ == "path" and isinstance(v__, str) and v__.endswith(".npz"):
                                fl_.add((scorer_of(v__), re.sub(r"__L\d+(?=\.npz$)", "", v__.rsplit("/", 1)[-1])))
                            else:
                                pths_(v__)
                    elif isinstance(x, list):
                        for v__ in x:
                            pths_(v__)
                pths_(inp["by_label"].get(lab))
                pths_(inp["variants"])  # the per-label call carries its label's variants as inp["variants"] (main pops variants_by_label)
                pfs_ = prefix_support(inp["pw"]["token_prefix"], fl_)
            if inp["pw"].get("auth_rule") is not None and not PWM.get("cat"):  # CUT (1): the overlap rule needs the catalogue; never a silent skip
                raise SystemExit(f"STOP: {lab} {cb_}: the overlap rule is captured but no catalogue runs are loaded")
            gid_ = pw_identity_gates(sz_, pt_, prs_, PWZ.get((lab, cb_), ({}, {}))[0], dcl_, pfs_)
            gv_ = (gid_
                   + (pw_overlap_viol(sz_, PWM["auth"]) if PWM.get("auth") is not None else [])
                   + (pw_auth_rule_viol(sz_["roles_by_key"], PWM["cat"], inp["pw"]["auth_rule"]) if inp["pw"].get("auth_rule") and PWM.get("cat") else [])
                   + ([f"{len(PWH[(lab, cb_)])} member rows the runner withholds (the pooled build STOPs), e.g. {PWH[(lab, cb_)][:2]}"] if PWH.get((lab, cb_)) else []))
            out["rules"].append({"row": lab, "rule": f"pooled identity gates, {PW_ID_RULE} ({cb_})", "violations": gv_,
                                 "content_identical_groups": len(sz_["content_identical_keys"]), "cross_pool_overlaps": len(sz_["roles_by_key"]),
                                 "declared_groups_here": sorted(sorted(g_) for g_ in sz_["content_identical_keys"] if frozenset(g_) in dcl_),
                                 "e12_pair_groups_here": sum(1 for g_ in sz_["content_identical_keys"] if len(g_) == 2 and frozenset(g_) in prs_
                                                             and frozenset(g_) not in dcl_)})
            if FIVE:  # G7b FOLD-MODEL PLANTED control (2e's spec (c)): a COPY of this union's exports with one H_B run repeated under a new key must
                pa_ = plant_alias(PWX[(lab, cb_)], "B", {k_ for g_ in sz_["content_identical_keys"] for k_ in g_})  # STOP gate 2 (no false split) with
                # exactly that pair (the copied run is in no identical-content group of the real union); the real verdict above is untouched
                pv_ = ["no H_B run to copy (the control could not run)"]
                if pa_ is not None:
                    u2_ = pw_union_chk(pa_[0], f"{lab} | {cb_} | PLANT", PWZ.get((lab, cb_), ({}, {}))[0])
                    gv2_ = pw_identity_gates(u2_[1], u2_[2], prs_, PWZ.get((lab, cb_), ({}, {}))[0], dcl_, pfs_)
                    grp2_ = [g_ for g_ in u2_[1]["content_identical_keys"] if pa_[1] in g_]
                    sp2_ = [v_ for v_ in gv2_ if v_.startswith("no false split")]
                    pv_ = [] if (len(grp2_) == 1 and sorted(grp2_[0]) == sorted((pa_[1], pa_[2])) and sp2_ and not u2_[1]["identity_conflicts"]
                                 and not any(v_.startswith("no false split") for v_ in gid_)) else ["the planted alias did not STOP gate 2 as one undeclared group"]
                out["rules"].append({"row": lab, "rule": f"PLANTED control: an alias inserted into the fold-model pooled H_B STOPs gate 2 ({cb_})",
                                     "violations": pv_, "planted_key": pa_[1] if pa_ else None, "copied_run": pa_[2] if pa_ else None,
                                     "planted_group_size": len(grp2_[0]) if pa_ and grp2_ else None})
            ctv_ = sorted({f"{r} {nm}" for nm in ("B", "A", "S") for r in pt_[nm] if r not in PWM["cat"] or nm not in PWM["cat"][r]["roles"]
                           or PWM["cat"][r]["main"] != inp["canon"].get(r.split("|", 1)[1])})
            if ctv_:  # Codex r40 #7: every pooled run and role catalogued, with the canonical main the rows weight by
                bad_.append(f"{len(ctv_)} pooled runs or roles outside the catalogue or with another canonical main, e.g. {ctv_[:3]}")
            zinfo_ = []  # the catalogue gate's zero-unit comparisons: reported, never gating (2e 13:5x, confirmed by 0d 16:5x); they gate a scored record
            zu1_ = PWZ.get((lab, cb_), ({}, {}))[0]
            zct_ = sorted({f"{r} {nm}" for nm in ("B", "A", "S") for r in (zu1_.get(nm) or {}) if r not in PWM["cat"] or nm not in PWM["cat"][r]["roles"]
                           or PWM["cat"][r]["main"] != inp["canon"].get(r.split("|", 1)[1])})
            if zct_:  # Codex r45 #2: the zero-unit appearances catalogued too (roles, canonical main)
                (bad_ if PWM["stage"] == "scored" else zinfo_).append(f"{len(zct_)} zero-unit runs or roles outside the catalogue, e.g. {zct_[:3]}")
            f1_ = (fr_ or {}).get("f1cols")  # Codex r45 #3: a single-model probe has no attack fold -1 columns: B, A and S present and empty
            mf1_ = ((PWM.get("mf") or {}).get(f"{lab} | {cb_}") or {}).get("f1cols")
            if not FIVE and (f1_ != {"B": [], "A": [], "S": []} or (PWM.get("mf") and mf1_ != {"B": [], "A": [], "S": []})):
                bad_.append("pooled file or manifest f1cols missing or not empty for a single-model probe")
            if FIVE:  # G7b FOLD-MODEL: the union of the member rows' attack fold -1 columns rebuilt here (f1cols_of) = the pooled file's and the manifest's
                xf1_ = {nm: sorted(int(c_) for c_ in v_) for nm, v_ in PWF1.get((lab, cb_), {"B": set(), "A": set(), "S": set()}).items()}
                if f1_ != xf1_ or (PWM.get("mf") and mf1_ != xf1_):
                    bad_.append(f"pooled file or manifest f1cols differ from the rows' attack fold -1 columns (here {({k_: len(v_) for k_, v_ in xf1_.items()})})")
            if PWM.get("mf"):  # G7b: the manifest's sizes after dedup, membership patterns, exported rows, identical-content groups and overlaps
                mf_ = PWM["mf"].get(f"{lab} | {cb_}") or {}
                cio_ = (mf_.get("sizes") or {}).get("content_identical_keys")  # Codex r40 #6: required fields, present and exactly equal
                if not isinstance(cio_, list) or sorted(map(sorted, cio_)) != sorted(map(sorted, sz_["content_identical_keys"])):
                    bad_.append("manifest identical-content key groups missing or different")
                if sorted(((mf_.get("cross_pool_overlaps") or {}).get("keys")) or []) != sorted(sz_["roles_by_key"]):
                    bad_.append("manifest cross-pool overlap keys differ")
                rbk_ = (mf_.get("cross_pool_overlaps") or {}).get("roles_by_key")
                if not isinstance(rbk_, dict) or rbk_ != sz_["roles_by_key"]:
                    bad_.append("manifest cross-pool overlap roles missing or different")
                eub_ = (mf_.get("sizes") or {}).get("eligible_units_by_run")  # Codex r41 #3: required, exactly the union's units per run and role; the
                pos_ = lambda d_: {(r_, nm_): n_ for r_, x_ in (d_ or {}).items() for nm_, n_ in x_.items() if n_}  # noqa: E731  admitted zero-unit entries (0)
                if not isinstance(eub_, dict) or pos_(eub_) != pos_(sz_["eligible_units_by_run"]):  # gate the scored record only (2e 13:5x on the
                    bad_.append("manifest eligible_units_by_run missing or different (runs with units)")  # zero-unit lists), as the zero-unit runs below
                zdiff_ = ([] if isinstance(eub_, dict) and eub_ == sz_["eligible_units_by_run"] else ["manifest eligible_units_by_run zero-unit entries differ"]) + [
                    f"{nm}: manifest zero-unit runs differ" for nm in ("B", "A", "S") if ((mf_.get("sizes") or {}).get(nm) or {}).get("zero_unit_runs") != sz_[nm]["zero_unit_runs"]]
                if pos_(eub_ if isinstance(eub_, dict) else {}) != pos_(sz_["eligible_units_by_run"]):
                    zdiff_ = [z_ for z_ in zdiff_ if not z_.startswith("manifest eligible")]  # already a violation above
                bad_ += zdiff_ if PWM["stage"] == "scored" else []
                zinfo_ += zdiff_ if PWM["stage"] != "scored" else []  # (extends the catalogue-membership note above; Codex r46 optional)
                pm_ = {json.dumps(inp["variants"][o_].get("pw_model"), sort_keys=True) for o_, _ in PWX[(lab, cb_)]}  # Codex r41 #4: the model identity
                if len(pm_) != 1 or json.dumps((mf_.get("model_identity") or {}).get("pinned"), sort_keys=True) not in pm_:  # the jobs carry, pinned
                    bad_.append("manifest model identity differs from the jobs' (pw_model)")
                LBm_ = inp["by_label"].get(lab) or {}
                if FIVE and [(mf_.get("model_identity") or {}).get(k_) for k_ in ("folds", "probe_file", "readout")] != [5, LBm_.get("probe_file"), LBm_.get("readout")]:
                    bad_.append("manifest model identity: not five models on this label's probe file and readout")  # G7b FOLD-MODEL (pw_prepare l.3897)
                for nm in ("B", "A", "S"):
                    ms_ = (mf_.get("sizes") or {}).get(nm) or {}
                    if ms_.get("after") != {"runs": sz_[nm]["runs"], "units": sz_[nm]["units"]}:
                        bad_.append(f"{nm}: manifest sizes after dedup differ")
                    if ms_.get("by_membership_pattern") != {p_: {"runs": sz_[nm]["pattern_runs"][p_], "units": u_} for p_, u_ in sz_[nm]["pattern_units"].items()}:
                        bad_.append(f"{nm}: manifest membership patterns differ")
                if sorted(mf_.get("origins") or []) != sorted(o_ for o_, _ in PWX[(lab, cb_)]):
                    bad_.append("manifest origins differ from the rows exported here")
            zu_, su_ = PWZ.get((lab, cb_), ({}, {}))  # G7e (2e 13:5x): the union's zero-unit runs and structural cells against the pooled file's lists
            zb_ = [] if (fr_ or {}).get("zero_unit_runs") == {nm: sorted(v_) for nm, v_ in zu_.items()} else ["the pooled file's zero_unit_runs differ"]
            zb_ += [] if (fr_ or {}).get("structural_cells") == {nm: sorted(v_) for nm, v_ in su_.items()} else ["the pooled file's structural_cells differ"]
            out["rules"].append({"row": lab, "rule": f"pooled pools file = the union rebuilt here ({cb_})", "runs": {nm: len(pt_[nm]) for nm in ("B", "A", "S")},
                                 "violations": bad_ + (zb_ if PWM["stage"] == "scored" else []),  # the catalogue gate reports the zero-unit lists, never gates on them
                                 "zero_unit_lists_info": zb_ + zinfo_ if PWM["stage"] != "scored" else None,
                                 "zero_unit_runs": {nm: len(v_) for nm, v_ in zu_.items()}, "structural_cells": {nm: len(v_) for nm, v_ in su_.items()},
                                 "not_covered": "the manifest's sizes.before, by_origin, by_key_world and attack_cells_by_origin (diagnostics)"})
        # ---- without S4 keeps the with-S4 calibration (09 @ 51a612b29): pools at draw 0 and pool counts equal; cohort = with-S4 minus S4 cells
        for wtag in [o for o in ("S_newmains", "union") if o in inp["origins"] and not inp["variants"][o].get("unavailable")]:
            w_, wo_ = rec["rows"][lab][wtag], rec["rows"][lab][f"{wtag} without S4"]
            bad = []
            def same_(a_, b_):  # the runner's own without-S4 tolerance (meta.without_S4.per_draw_calibration: 1e-12 abs): the two rows run in separate
                if isinstance(a_, dict) and isinstance(b_, dict):  # containers, so z-scored thresholds can differ in the last float bits
                    return set(a_) == set(b_) and all(same_(a_[k_], b_[k_]) for k_ in a_)
                if isinstance(a_, float) and isinstance(b_, float):
                    return (np.isnan(a_) and np.isnan(b_)) or a_ == b_ or abs(a_ - b_) <= 1e-12
                return a_ == b_
            if not same_(w_["pools_at_draw0"], wo_["pools_at_draw0"]):
                bad.append("pools_at_draw0 differ beyond 1e-12")
            for k_ in ("H_sim_runs", "H_sim_units", "H_A_runs", "H_A_units", "H_B_runs", "H_B_units"):
                if w_["coverage"][k_] != wo_["coverage"][k_]:
                    bad.append(k_)
            cw, cwo = set(inp["variants"][wtag]["cohort"]), set(inp["variants"][f"{wtag} without S4"]["cohort"])
            if cwo != cw - set(inp["s4_cells"]):
                bad.append("cohort difference is not S4 cells")
            out["rules"].append({"row": lab, "rule": f"{wtag}: without S4 keeps the with-S4 calibration", "S4_cells_dropped": len(cw - cwo), "violations": bad})
        nmo_ = [o for o in inp["origins"] if aob(o) == "S_newmains" or (PWM and aob(o).split(" ")[0] == "S_newmains")]  # G7: the pooled S_newmains rows too
        if any(not inp["variants"][o].get("use4r", True) for o in nmo_):
            out["rules"].append({"row": lab, "rule": "coverage pool used", "violations": ["this public probe uses the coverage pool"]})
    # ---- pooled combinations from this check's per-origin catches; D = each origin's main count at draw 0
    if inp["mode"] == "alltasks":
        combos = {"pooled_all": ("S_old", "S_new", "S_newmains"), "pooled_all_without_S4": ("S_old", "S_new", "S_newmains without S4"), "pooled_455": ("S_old", "S_new")}
        if any(o.endswith("(registered cohort)") for o in inp["origins"]):  # the no-filter job's appendix line (registered attempt-filtered cohort)
            combos["pooled_all_registered_cohort"] = ("S_old (registered cohort)", "S_new (registered cohort)", "S_newmains (registered cohort)")
        if "S_paper493" in inp["origins"]:  # the second refresh: S_paper493's own line and blocks; the per-origin-stream sensitivities
            combos |= {"S_paper493": ("S_paper493",), "S_paper493_P_new": ("S_paper493 [P_new]",), "S_paper493_P_rep": ("S_paper493 [P_rep]",),
                       "pooled_all_origins_per_origin_streams": ("S_old", "S_new", "S_newmains", "S_paper493"),
                       "paper_cells_per_origin_streams": ("S_old", "S_new", "S_paper493")}
        if any(o.endswith("(registered cohort) (all-origins draws)") for o in inp["origins"]):  # G4: the registered cohort on the all-origins draws
            combos["pooled_all_origins_registered_cohort"] = tuple(o + " (registered cohort) (all-origins draws)" for o in ("S_old", "S_new", "S_newmains", "S_paper493"))
        if any(o.endswith("(all-origins draws)") for o in inp["origins"]):  # one main = one main over every origin (09 @ e3a459bd0 item 1)
            AOS = " (all-origins draws)"
            combos |= {"pooled_all_origins": tuple(o + AOS for o in ("S_old", "S_new", "S_newmains", "S_paper493")),
                       "pooled_all_origins_without_S4": tuple(o + AOS for o in ("S_old", "S_new", "S_newmains without S4", "S_paper493")),
                       "paper_cells": tuple(o + AOS for o in ("S_old", "S_new", "S_paper493"))}
        if PWM:  # G7 (items 42 / 45(3)): only the pooled combos, each over its own suffixed union rows (a catalogue check has no record to compare)
            combos = {cb: tuple(PWM["combo_rows"][cb]) for cb in PWM["combos"]} if PWM["stage"] != "catalogue" else {}
        if W2:  # rc1b W2: Stage D and kappa carry no pooled block (Stage D's rows hold the per-task entries; kappa's margins are per row)
            combos = {}
    elif inp["mode"] in ("sweep", "union_sol"):
        combos = {}  # the sweep and union-Sol records carry no pooled block
    else:
        combos = {"pooled": ("S_old", "S_new")}
    nmains = {(r_["row"], r_["origin"]): r_["mains"] for r_ in out["rows"] if "mains" in r_}  # unavailable rows carry no mains
    ALLO_ = ("pooled_all_origins", "pooled_all_origins_without_S4", "paper_cells", "pooled_all_origins_registered_cohort")  # the runner's COMBOS_ALLO
    for lab in inp["labels"]:
        held_ = held_by.get(lab, {})
        bagg = BAGG.setdefault(lab, {})
        for cb, origs in combos.items():  # as the runner's pool_reduce_n: members = rows computed and not withheld (absent rows are no member)
            prs_ = [o for o in origs if not inp["variants"].get(o, {}).get("absent")]
            pb = rec["pooled"].get(lab, {}).get(cb)
            if cb in ALLO_ and not prs_:  # LOO: no all-origins rows, so the runner emits no such combo
                out["pooled"].append({"row": lab, "combo": cb, "mismatches": [] if pb is None else [["a combo over absent rows", cb, None]]})
                continue
            if pb is None:
                out["pooled"].append({"row": lab, "combo": cb, "mismatches": [["combo missing from the record", cb, None]]})
                continue
            un_ = [o for o in origs if inp["variants"].get(o, {}).get("unavailable")]
            if un_:  # own_all_LOO's with-S4 S_newmains: the combo is unavailable, never pooled over the rest (exact, Codex r16; role only where the checker pins it)
                ok_p = pb == {"members": [], "unavailable_members": {o: "unavailable: the S4 cells were not in the deciding eligibility file" for o in un_},
                              "role": COMBO_ROLE.get(cb, pb.get("role")), "pooled": "unavailable: the S4 cells were not in the deciding eligibility file"}
                out["pooled"].append({"row": lab, "combo": cb, "mismatches": [] if ok_p else [["unavailable combo", un_, {k_: pb.get(k_) for k_ in ("members", "unavailable_members", "pooled")}]]})
                continue
            hm_ = [o for o in prs_ if o in held_]
            origs = tuple(o for o in prs_ if o not in held_)  # the members pooled below
            wm_ = {o: rec["rows"][lab][o].get("withheld") for o in hm_}  # the runner copies each withheld row's text
            if not origs:  # every origin withheld or absent (LOO's S_paper493 lines): the runner's placeholder
                ok_p = pb == {"members": [], "withheld_members": wm_, "role": COMBO_ROLE.get(cb, pb.get("role")), "pooled": "unavailable (every origin withheld)"}
                out["pooled"].append({"row": lab, "combo": cb, "mismatches": [] if ok_p else [["every member withheld or absent", wm_, pb]]})
                continue
            mism, pc, pmv_ = [], [], {}  # pc: value comparisons (compare(): NaN on both sides equal and labelled)
            if inp["mode"] == "alltasks" and (pb["members"] != list(origs) or pb.get("withheld_members") != wm_):
                mism.append(["members", [list(origs), wm_], [pb["members"], pb.get("withheld_members")]])
            elif inp["mode"] != "alltasks" and hm_:
                mism.append(["a withheld member outside the all-tasks record (not modelled here)", hm_, None])
            for lv in levels:
                key = f"{lv:g}"
                if inp["mode"] == "alltasks":
                    Dw = pb["joint"][key]["D_draw0"]
                    if sorted(Dw) != sorted(origs):
                        mism.append([f"{key}|D_draw0 origins", sorted(origs), sorted(Dw)])
                    pc += [(f"{key}|D_draw0|{o}", float(nmains[(lab, o)]), float(Dw.get(o, float("nan")))) for o in origs]
                all_ = (rule37_gate_chk({o: mine_rd[(lab, o, key)] for o in origs}) if PWM  # G7 (item 45(2)): the pooled gate reads S and B only
                        else all(all(mine_rd[(lab, o, key)].values()) for o in origs))
                for md_ in ("joint", "fixed"):  # each origin's pool flags (a dict) and "all" over every flag; required in both modes
                    rdp = pb.get(md_, {}).get(key, {}).get("readable")
                    want_ = {o: mine_rd[(lab, o, key)] for o in origs} | {"all": all_}
                    if rdp != want_:
                        mism.append([f"{key}|{md_}|readable", want_, rdp])
                per = {o: mine[(lab, o, key)] for o in origs}
                dcomb = lambda x, alt=None: dweighted([per[o].get(x) if per[o].get(x) is not None else per[o][alt] for o in origs], [nmains[(lab, o)] for o in origs])  # noqa: E731
                excl_ = {tuple(x) for x in inp.get("pooled_exclude_cells", [])}  # (origin, cell): P_dup in the second refresh

                def comb(x, alt=None):  # main-union pooling over the member origins' cells (each at its own origin's threshold)
                    by_main = {}
                    for o in origs:
                        h = mine_cells[(lab, o, key)].get(x, mine_cells[(lab, o, key)].get(alt) if alt else None)
                        if h is None:
                            return float("nan")
                        mo = mine_mainof[(lab, o)]
                        for c_, v in h.items():
                            if (o, c_) not in excl_:
                                by_main.setdefault(mo[c_], []).append(v)
                    return float(np.mean([np.mean(v) for v in by_main.values()])) if by_main else float("nan")
                mains_ = [set(mine_mainof[(lab, o)].values()) for o in origs]
                disjoint_ = sum(len(m_) for m_ in mains_) == len(set().union(*mains_))
                if cb == "pooled_all_registered_cohort" or cb.endswith("_per_origin_streams"):  # D-weighted per-origin streams (Codex r14; COMBOS_SP)
                    comb = dcomb  # noqa: F811
                sim, r1, r2, y2 = (comb(x) for x in ("sim@S|primary", "run1@B", "run2@B", "run2@A"))
                y1 = comb("run1@B2", "run1@B")
                rm_ = (r1 + r2) / 2
                if PWM:  # G7b VALUE: pool_reduce_n l.3124-3151 ported: the D-weighted member combination in the runner's order (Python sum over members,
                    Dd_ = sum(np.float64(nmains[(lab, o)]) for o in origs)  # where(D > 0) on the all-origins combos), real_mean from the members' own
                    for o in origs:  # real_mean@B, y1 = run1@B (no B2 pooled); the main-union values above stay as a cross-check (pc below)
                        per[o].setdefault("y1", per[o].get("run1@B2", per[o]["run1@B"]))

                    def comb_rt(k_):
                        return sum(np.where(np.float64(nmains[(lab, o)]) > 0, per[o][k_] * np.float64(nmains[(lab, o)]), 0.0) for o in origs) / Dd_
                    mu_x_ = {"sim@S|primary": sim, "run1@B": r1, "run2@B": r2, "run2@A": y2, "y1": y1}
                    sim, r1, r2, rm_, y2, y1 = (float(comb_rt(k_)) for k_ in ("sim@S|primary", "run1@B", "run2@B", "real_mean@B", "run2@A", "y1"))
                    pc += [(f"{key}|main-union vs runner D-weighted|{x_}", mu_x_[x_], v_) for x_, v_ in
                           (("sim@S|primary", sim), ("run1@B", r1), ("run2@B", r2), ("run2@A", y2), ("y1", y1))]
                if G6_:  # G6 (full length): the PANEL excess's per-row inputs at draw 0 (pool_reduce_n): G = 0.5|sim - run1@B| + 0.5|sim - run2@B|,
                    # Y = |run2@A - y1|, pooled and per origin; pooled_abs_per_origin = the per-origin G and Y D-weighted; saturated = Python's max over
                    # the five arms >= 0.99 (the runner's own max, NaN handling included); exported for the aggregate step, and compared with the record
                    # G7 (pooled world, item 45(1) and Astra r1 #9 there): Y reads run2@B (not run2@A) against y1 at the pooled tau_B, saturation reads
                    # the four arms G and Y use, and each origin's readable flag is the RULE 37 gate
                    yo_ = {o: per[o].get("run1@B2", per[o]["run1@B"]) for o in origs}
                    y2o_ = {o: per[o]["run2@B" if PWM else "run2@A"] for o in origs}
                    Go = {o: 0.5 * abs(per[o]["sim@S|primary"] - per[o]["run1@B"]) + 0.5 * abs(per[o]["sim@S|primary"] - per[o]["run2@B"]) for o in origs}
                    Yo = {o: abs(y2o_[o] - yo_[o]) for o in origs}
                    Dl_ = [nmains[(lab, o)] for o in origs]
                    sato = {o: max(*(per[o][x_] for x_ in ("sim@S|primary", "run1@B", "run2@B") + (() if PWM else ("run2@A",))), yo_[o]) >= 0.99 for o in origs}
                    stv = ({"pooled": (0.5 * np.abs(sim - r1) + 0.5 * np.abs(sim - r2), np.abs((r2 if PWM else y2) - y1),  # l.3171-3172
                                       max(*((sim, r1, r2, y1) if PWM else (sim, r1, r2, y2, y1))) >= 0.99),
                            "pooled_abs_per_origin": (dweighted([Go[o] for o in origs], Dl_), dweighted([Yo[o] for o in origs], Dl_), any(sato.values()))}
                           | {o: (Go[o], Yo[o], sato[o]) for o in origs})
                    rdo_ = {o: (rule37_gate_chk({o: mine_rd[(lab, o, key)]}) if PWM else all(mine_rd[(lab, o, key)].values())) for o in origs}
                    aggin.setdefault(lab, {}).setdefault(cb, {})[key] = {  # native floats and booleans (the vals/ JSON has no conversion handler)
                        "df": float(sim - rm_), "yr": float(r2 - r1), "rd": bool(all_), "rdo": {o: bool(v_) for o, v_ in rdo_.items()},
                        "st": {s_: [float(g_), float(y_)] for s_, (g_, y_, _) in stv.items()}, "sat": {s_: bool(t_) for s_, (_, _, t_) in stv.items()}}
                    for md_ in ("joint", "fixed"):  # the record's per-label panel block, compared stream by stream
                        pn_ = pb[md_].get(key, {}).get("panel", {})
                        if sorted(pn_) != sorted(stv):
                            mism.append([f"{key}|{md_}|panel streams", sorted(stv), sorted(pn_)])
                        for s_, (g_, y_, t_) in stv.items():
                            q_ = pn_.get(s_, {})
                            pc += [(f"{key}|{md_}|panel|{s_}|G", g_, q_.get("G", {}).get("point")), (f"{key}|{md_}|panel|{s_}|Y", y_, q_.get("Y", {}).get("point")),
                                   (f"{key}|{md_}|panel|{s_}|G_minus_Y", g_ - y_, q_.get("G_minus_Y", {}).get("point")), (f"{key}|{md_}|panel|{s_}|saturated", t_, q_.get("saturated"))]
                            if s_ in rdo_:
                                pc.append((f"{key}|{md_}|panel|{s_}|readable", rdo_[s_], q_.get("readable")))
                if disjoint_ and not excl_:  # with disjoint mains the main-union value is the D-weighted mean of the per-origin catches
                    pc += [(f"{key}|main-union vs D-weighted|{x_}", v_, dcomb(x_)) for x_, v_ in (("sim@S|primary", sim), ("run1@B", r1), ("run2@B", r2), ("run2@A", y2))]
                mv = {"catch|sim": sim, "catch|run1": r1, "catch|run2": r2, "catch|real_mean": rm_, "gap|delta_full": sim - rm_,
                      "gap|delta_rr": y2 - y1, "gap|sim_minus_run1": sim - r1, "gap|sim_minus_run2": sim - r2}
                if PWM:  # G7 (item 45(1) / RENDER): the yardstick at the pooled tau_B, signed; 2e / 0d (b) @ 8a15fb10f: real_line = sim at the pooled
                    mv["gap|run2_minus_run1_at_tauB"] = r2 - r1  # tau_B minus the real mean (RULE 45's SENS shared-threshold line)
                    mv["gap|real_line"] = (float(comb_rt("sim@B")) if PWM else comb("sim@B")) - rm_
                pmv_[key] = dict(mv)  # G7b VALUE: this level's draw-0 pooled values (the per-draw path's draw-0 self-check below)
                if CAP == "paired":
                    s1, p1_, s2, p2_ = (comb(x) for x in ("sim@S|p1", "run1@B|p1", "sim@S|p2", "run2@B|p2"))
                    mv |= {"gap|sim_minus_run1_paired": s1 - p1_, "gap|sim_minus_run2_paired": s2 - p2_, "gap|delta_full_paired": ((s1 - p1_) + (s2 - p2_)) / 2}
                for md_ in ("joint", "fixed"):  # 7ad9f0c32+: the pooled catch block also carries the yardstick arms (run2@A, y1) and y1's definition
                    cr_ = pb[md_].get(key, {}).get("catch", {})
                    if "y1" in cr_ or "run2@A" in cr_ or PWM:  # a pooled block always carries them (Codex r45 #6); older records only where present
                        pc += [(f"{key}|{md_}|catch|run2@A", y2, cr_.get("run2@A", {}).get("point")), (f"{key}|{md_}|catch|y1", y1, cr_.get("y1", {}).get("point")),
                               (f"{key}|{md_}|catch|y1_def", Y1_DEF_PW if PWM else "run1@B2 on S_old, run1@B on other origins", cr_.get("y1_def"))]
                    for x, v in mv.items():
                        a_, b_ = x.split("|")
                        pc.append((f"{key}|{md_}|{x}", v, pb[md_].get(key, {}).get(a_, {}).get(b_, {}).get("point")))
            if BOOT:  # G7b VALUE (09 via 2e 23:2x): the pooled gaps at draw 0 and the seeded draws (pool_reduce_n l.3124-3151 per draw, ported: the
                for md_ in ("joint", "fixed"):  # members' D-weighted all-origins combination), against the npz's pooled keys; S / Y inputs for aggregate()
                    for lv in levels:
                        key = f"{lv:g}"
                        for i_, b_ in enumerate(BW["ids"]):
                            assert all((lab, o, md_, key, b_) in BROW for o in origs), (lab, cb, md_, key, b_, "a member row without per-draw values (STOP)")
                            per_ = {o: BROW[(lab, o, md_, key, b_)][0] | {"y1": BROW[(lab, o, md_, key, b_)][0]["run1@B"]} for o in origs}
                            Dm_ = {o: BROW[(lab, o, md_, key, b_)][1] for o in origs}
                            sim_, r1_, r2_, rmb_, y2_, y1_ = (boot_comb(per_, Dm_, origs, k_) for k_ in ("sim@S|primary", "run1@B", "run2@B", "real_mean@B", "run2@A", "y1"))
                            gp_ = {"delta_full": sim_ - rmb_, "delta_rr": y2_ - y1_, "sim_minus_run1": sim_ - r1_, "sim_minus_run2": sim_ - r2_,
                                   "run2_minus_run1_at_tauB": r2_ - r1_, "real_line": boot_comb(per_, Dm_, origs, "sim@B") - rmb_}
                            rf_ = {x: bref(lab, cb, md_, key, "pooled", "gap_" + x)[i_] for x in BOOT_POOLED_GAP}
                            pc += [(f"boot|{md_}|{key}|draw {b_}|pooled|gap|{x}", gp_[x], rf_[x]) for x in BOOT_POOLED_GAP]
                            if b_ == 0:  # self-check: the per-draw path's pooled gaps at draw 0 = this check's draw-0 pooled values (both computed here)
                                pc += [(f"boot|{md_}|{key}|draw 0|pooled|self-check: per-draw path = draw-0 path|gap|{x}", gp_[x], pmv_[key][f"gap|{x}"])
                                       for x in BOOT_POOLED_GAP]
                            bagg.setdefault(cb, {}).setdefault(md_, {}).setdefault(key, {})[str(b_)] = {
                                "df": gp_["delta_full"], "yr": gp_["run2_minus_run1_at_tauB"], "df_ref": rf_["delta_full"], "yr_ref": rf_["run2_minus_run1_at_tauB"]}
                            if PLV and md_ == "joint" and lab in PLV["labels"] and any(abs(lv - x_) <= 1e-12 * x_ for x_ in PLV["points"]):
                                # PLANT V3 block (iii), pooled: the runner's planted pooled catch and Delta_full (L3338-3353): the members' D-weighted
                                # combination of their planted catches per arm and delta, minus the members' combined real mean; an arm missing in
                                # any member (unattainable / unavailable) gives no pooled planted key (checked absent from the npz)
                                plg_ = {}
                                for d_ in PLV["deltas"]:
                                    kc_, kg_ = f"catch_sim_plant_{d_:+d}", f"gap_delta_full_plant_{d_:+d}"
                                    if all(d_ in (PFIT.get((lab, o, key)) or ({}, {}))[1] for o in origs):
                                        pp_ = {o: {kc_: BROWP[(lab, o, key, b_)][d_]} for o in origs}
                                        sp_ = boot_comb(pp_, Dm_, origs, kc_)
                                        rgp_ = bref(lab, cb, "joint", key, "pooled", kg_)[i_]
                                        pc += [(f"boot|joint|{key}|draw {b_}|pooled|plant|{kc_}", sp_, bref(lab, cb, "joint", key, "pooled", kc_)[i_]),
                                               (f"boot|joint|{key}|draw {b_}|pooled|plant|{kg_}", sp_ - rmb_, rgp_)]
                                        plg_[f"{d_:+d}"] = [sp_ - rmb_, rgp_]
                                    else:
                                        pc.append((f"boot|joint|{key}|draw {b_}|pooled|plant|absent|{d_:+d}", True,
                                                   npzkey(lab, cb, "joint", key, "pooled", kc_) not in BZF.files and npzkey(lab, cb, "joint", key, "pooled", kg_) not in BZF.files))
                                bagg[cb][md_][key][str(b_)]["pl"] = plg_
                                # block (iv)'s domain (Codex V3 #5): the deltas SOME member row reaches (the runner's pk_ keys, L3338): coh_pl's tags
                                bagg[cb][md_][key][str(b_)]["pl_any"] = [f"{d_:+d}" for d_ in PLV["deltas"] if any(d_ in (PFIT.get((lab, o, key)) or ({}, {}))[1] for o in origs)]
                                if b_ == 0:  # AMEND6 (sim2): no sim2 planted array in the npz (no sim2 arm reaches a delta; its solves are compared per row)
                                    pc.append((f"boot|joint|{key}|draw 0|pooled|plant|sim2 arrays absent", True, sim2_npz_absent(lab, cb, key, origs, PLV["deltas"], BZF.files)))
            for md_ in ("joint", "fixed"):
                if sorted(pb.get(md_, {})) != lvk_all:  # B's split: the record's pooled levels = the full set (unsplit: as before)
                    mism.append([f"pooled levels|{md_}", sorted(pb.get(md_, {})), lvk_all])
            cp_ = compare(pc, sum(nmains[(lab, o)] for o in origs) == 0)
            out["pooled"].append({"row": lab, "combo": cb, "mismatches": mism + cp_["mismatches"], "nan_on_both_sides": cp_["nan_on_both_sides"],
                                  "empty_block": sum(nmains[(lab, o)] for o in origs) == 0, "n_compared": len(pc),
                                  "ledger": cp_["ledger"] + [[m_[0], "MISMATCH"] for m_ in mism], "ledger_values": cp_["ledger_values"] + [[m_[0], *m_[1:], "MISMATCH"] for m_ in mism]})
            if PWM and PWM["stage"] == "scored":  # G7b VALUE (2e / 09 22:4x): n_compared > 0 per label x combo x mode x level, the pooled block AND
                vq_ = vacuous(pc)  # 507-prep 23:11 NB; Astra r6 1: NaN on both sides at a draw-0 catch / gap / panel key or any seeded-draw key fails
                VNAN.extend(f"{lab} | {cb} | {k0_}" for k0_ in vq_)
                BNAN[lab] = BNAN.get(lab, 0) + sum(1 for k0_ in vq_ if k0_.startswith("boot|"))
                for lv in levels:  # every member row, else the gate fails (the STOP); the seeded draws' counts beside them (aggregate() checks their keys)
                    key = f"{lv:g}"
                    for md_ in ("joint", "fixed"):
                        npb_ = sum(1 for k0_ in pc if k0_[0].startswith(f"{key}|{md_}|"))
                        zr_ = [o for o in origs if NCMP.get((lab, o, key, md_), 0) == 0]
                        VCNT[(lab, cb, key, md_)] = {"pooled": npb_, "rows": {o: NCMP.get((lab, o, key, md_), 0) for o in origs}} | (
                            {"boot_pooled": sum(1 for k0_ in pc if k0_[0].startswith(f"boot|{md_}|{key}|")),
                             "boot_rows": {o: NCMP.get((lab, o, key, f"boot|{md_}"), 0) for o in origs}} if BOOT else {})
                        if npb_ == 0 or zr_:
                            VZERO.append(f"{lab} | {cb} | {key} | {md_}: pooled {npb_}, rows without comparisons {zr_}")

    for l_ in inp["labels"]:  # G7b FOLD-MODEL (2e 20:3x): the cfg0 regression and what this check does not cover, in the record
        if (inp.get("cfg0_regression") or {}).get(l_) is not None:
            out["rules"].append({"row": l_, "rule": "cfg0 regression: this label's own probe config and S_new arm files vs cfg0's (info; a public label's "
                                                    "difference STOPs at capture; a retrain label's is expected)", "violations": [], "info": inp["cfg0_regression"][l_]})
        if (inp["by_label"].get(l_) or {}).get("fold_label"):  # G7b VALUE (Astra r5 #6): a value run states VALUE_SCOPE only (never the catalogue scope)
            out["rules"].append({"row": l_, "rule": "fold-model check: not covered (stated)", "violations": [], "info": inp.get("fold_not_covered"),
                                 "not_applicable": inp.get("fold_not_applicable"), "scope": value_scope((BOOT or {}).get("seed")) if (PWM and PWM["stage"] == "scored") else FOLD_SCOPE})
    if PWM and PWM["stage"] == "scored":  # G7b VALUE: the gate and the SCOPE line (09 via 2e 22:4x / 23:2x: exact text) in every label's record
        for l_ in inp["labels"]:
            out["rules"].append({"row": l_, "rule": ("VALUE: n_compared > 0 per label x combo x mode x level (pooled block and every member row); no draw-0 "
                                                     "catch / gap / panel value and no seeded-draw value NaN on both sides"),
                                 "violations": [v_ for v_ in VZERO + VNAN if v_.startswith(f"{l_} | ")],
                                 "counts": {" | ".join(k_[1:]): v_ for k_, v_ in sorted(VCNT.items()) if k_[0] == l_},
                                 "boot": ({"draw_ids": BW["ids"], "seed": BOOT["seed"], "draws_sha256": BW["draws_sha256"], "nan_on_both_sides": BNAN.get(l_, 0),
                                           "npz": list(BOOT["npz"][1:]) if BOOT.get("npz") else None} if BOOT else None),  # rc1b W2b: no npz
                                 "scope": value_scope((BOOT or {}).get("seed")), "tolerance": "|a - b| <= 1e-9 * max(1, |a|, |b|); booleans, integers and keys exact; NaN equals only NaN",
                                 "not_compared": VALUE_NOT_COMPARED, "value_plant": ({"plant": VP, "fired": VPI} if VP else None)})
    for (lb_, o_), fi_ in sorted(FOLDINFO.items()):  # G7b FOLD-MODEL: what each row's fold-model rules did (keys and counts; the withhold / STOP
        out["rules"].append({"row": lb_, "rule": f"fold-model rules on {o_} (info; a withhold or STOP fails the identity gate above)",  # reasons
                             "violations": [], "info": fi_})  # gate through 'member rows the runner withholds')
    if VP:  # G7b VALUE PLANT: the planted call must have reached its row's scored pass and moved one source score (else STOP: a no-op plant)
        assert VPI.get("row") == VP["label"] and VPI.get("origin") == VP["origin"] and VPI.get("arm") == VP["arm"], ("the plant did not fire", VP)
    if W2:  # rc1b W2: what this call compared (the count gate reads values_compared per label) and, in the control call, what each plant moved
        out["w2"] = {"kind": W2["kind"], "values_compared": dict(W2N), "control": dict(W2CTL) if W2CTL else None, "scope": W2_SCOPE[W2["kind"]]}
    if PLV:  # PLANT V3: the plant blocks' scope and counts (keys and counts only), and P1's target and whether it fired (a P1 call only)
        out["plant_v3"] = {"not_compared": PLANT_NOT_COMPARED, "rows_solved": len(PFIT), "sim2_counts": dict(SIM2N),
                           "n_plant_comparisons": {"rows": sum(1 for r_ in out["rows"] for x_ in r_.get("ledger") or [] if "|plant|" in x_[0]),
                                                   "pooled": sum(1 for p_ in out["pooled"] for x_ in p_.get("ledger") or [] if "|plant|" in x_[0])}}
        if P1:
            out["plant_v3"]["p1"] = {"target": {k_: P1[k_] for k_ in ("label", "origin", "level", "order", "seed")}, "hit": P1HIT}
    if BZF is not None:
        BZF.close()
    out["all_equal"] = (all(not r_["mismatches"] for r_ in out["rows"]) and all(not p_["mismatches"] for p_ in out["pooled"])
                        and all(not q_["violations"] for q_ in out["rules"]))
    out["n_compared"] = sum(r_["n_compared"] for r_ in out["rows"])
    ob_ = out | {k_: [{x_: y_ for x_, y_ in r_.items() if x_ != "ledger_values"} | {"mismatches": blind(r_["mismatches"])} for r_ in out[k_]]
                  for k_ in ("rows", "pooled")}  # every return path: keys only (G7b VALUE: the values ledger stays in the Volume's detail shard)
    if run_dir:  # detach: this label's output persists on the results Volume under an exclusive name (a re-execution cannot overwrite it)
        lab_ = inp["labels"][0]
        sh_ = {"run_dir": run_dir, "label": lab_, "input_sha256": inp_sha0,
               "record_sha256": hashlib.sha256(recb).hexdigest(), "read_started_pt": started, "out": out}
        w2c_on_ = bool(W2 and any(k_.startswith("plant_") for k_ in W2))  # rc1b W2: the control call's shards in w2ctl/
        d_ = Path("/res") / run_dir / ("plant" if VP else "p1" if P1 else "w2ctl" if w2c_on_ else "")  # G7b VALUE PLANT / PLANT V3 P1: own folders
        for sub_ in ("detail", "vals"):
            (d_ / sub_).mkdir(parents=True, exist_ok=True)
        with (d_ / "detail" / f"{shard_name(lab_)}.json").open("x") as fh:  # G6: the full output (mismatch values) stays on the Volume
            fh.write(json.dumps(sh_, default=float))
        if G6_:  # this label's S/Y and excess inputs, for aggregate()
            with (d_ / "vals" / f"{shard_name(lab_)}.json").open("x") as fh:
                fh.write(json.dumps({"label": lab_, "input_sha256": inp_sha0, "record_sha256": sh_["record_sha256"], "aggin": aggin.get(lab_, {}),
                                     "boot": BAGG.get(lab_, {})}))  # G7b VALUE: the per-draw S / Y inputs (values: Volume only)
        with (d_ / f"{shard_name(lab_)}.json").open("x") as fh:  # the laptop's shard: mismatch keys only
            fh.write(json.dumps(sh_ | {"out": ob_}, default=float))
        res_vol.commit()
    return ob_


class _Stop(Exception):
    pass


MAIN_DEFAULTS = (["arm not routed (no folder)", "no folder", "never launched", "arm never launched", "setup failure", "sim never launched", "staging exclusion",
                  "no clean donor", "approach-assert"],
                 ["errored run", "invalid measurement (malformed call)", "malformed call", "attempt timeout", "other (cause unknown)"])  # 09's ruling (2e 02:3x, 02:5x)


PLANT = ("FIBS|uncut C", "df", None, 1e-3), ("FIBS|uncut R", "G", "pooled", 1e-3)  # G6's positive controls: one external row's input shifted


@app.function(image=image, cpu=(2, 2), memory=(16384, 24576), timeout=1800, retries=0, startup_timeout=120, scaledown_window=60,  # 14:10 limits; AMEND11
              volumes={"/res": res_vol, "/sol": modal.Volume.from_name(VOLS["solavg"], version=2).read_only()})  # /sol: G7's per-set companion record
def aggregate(run_dir: str) -> dict:
    """G6 (0d's partial ruling via 2e 13:1x): the co-headline S/Y (six-, five- and four-probe groups) and the PANEL excess at draw 0, from this run's
    check() inputs for the external probes (vals/) and the stated gap's fold-model rows as recorded, against the record's coheadline_signed and
    panel_groups; then the positive controls (PLANT): each must flag exactly the aggregates that hold the shifted row. Writes the laptop's keys-only
    aggregate.json and the full detail (Modal only)."""
    res_vol.reload()
    d_ = Path("/res") / run_dir
    man = json.loads((d_ / "manifest.json").read_bytes())
    recb = (d_ / "record.json").read_bytes()
    assert hashlib.sha256(recb).hexdigest() == man["record_sha256"], "record sha256 (aggregate)"
    rec = json.loads(recb)
    if man.get("w2"):  # rc1b W2 (PRESTATED_W2 + AMEND1-3): no co-headline or PANEL tree on these records; the W2 verdict instead
        w2_ = man["w2"]
        det_ = {}
        for lb in man["labels"]:
            sh_ = json.loads((d_ / "detail" / f"{shard_name(lb)}.json").read_bytes())
            assert sh_["label"] == lb and sh_["input_sha256"] == man["per_label_input_sha256"][lb], (lb, "a detail shard from another input")
            det_[lb] = sh_["out"]
        cnt_ = {lb: ((det_[lb].get("w2") or {}).get("values_compared") or {}).get(lb) for lb in man["labels"]}
        cl_ = w2_["control"]["label"]
        pd_ = json.loads((d_ / "w2ctl" / "detail" / f"{shard_name(cl_)}.json").read_bytes())
        assert pd_["label"] == cl_ and pd_["input_sha256"] == w2_["control_input_sha256"] and pd_["record_sha256"] == man["record_sha256"], \
            "rc1b W2: a control shard from another run or input (STOP)"
        pc_cnt_ = ((pd_["out"].get("w2") or {}).get("values_compared") or {}).get(cl_)
        if w2_["kind"] == "pertask":
            cv_ = w2_control_verdict_pertask(ledger_of(det_[cl_]), ledger_of(pd_["out"]), (pd_["out"].get("w2") or {}).get("control") or {},
                                             w2_["control"], cnt_.get(cl_), pc_cnt_)
        else:
            cv_ = w2_control_verdict_kappa(ledger_of(det_[cl_]), ledger_of(pd_["out"]), (pd_["out"].get("w2") or {}).get("control") or {},
                                           w2_["control"], cnt_.get(cl_), pc_cnt_)
        gates_ = {"count_per_label_as_predicted": all(cnt_[lb] == w2_["count_per_label"] for lb in man["labels"]),
                  "every_label_all_equal": all(det_[lb]["all_equal"] for lb in man["labels"]), "control_pass": cv_["pass"]}
        out = {"label": f"rc1b W2 aggregate ({w2_['kind']}, stage {w2_['stage']})", "run_dir": run_dir, "record_sha256": man["record_sha256"],
               "w2": {"kind": w2_["kind"], "stage": w2_["stage"], "scope": w2_["scope"], "gates": gates_, "control": cv_,
                      "labels_count_ok": {lb: cnt_[lb] == w2_["count_per_label"] for lb in man["labels"]}},
               "n_compared": sum(int(cnt_[lb] or 0) for lb in man["labels"]), "mismatches": [], "positive_controls": [],
               "all_equal": all(gates_.values())}
        if w2_["kind"] == "kappa":  # AMEND5 section 1: each chosen stored file's full sha256 and the attempt counts (keys, shas and counts only)
            sr_ = {f"{r_['row']} | {r_['origin']}": r_["w2_stored_row"] for lb in man["labels"] for r_ in det_[lb]["rows"] if r_.get("w2_stored_row")}
            out["w2"]["stored_rows"] = {"n_tied": len(sr_), "chosen_sha256": {k_: v_["sha256"] for k_, v_ in sorted(sr_.items())},
                                        "jobs_with_more_than_one_complete_attempt": sum(int(v_["n_complete_attempts"] > 1) for v_ in sr_.values()),
                                        "jobs_with_more_than_one_file": sum(int(v_["n_files"] > 1) for v_ in sr_.values())}
        (d_ / "detail").mkdir(parents=True, exist_ok=True)
        with (d_ / "detail" / "aggregate.json").open("x") as fh:
            fh.write(json.dumps(out | {"values_compared": cnt_, "control_values_compared": pc_cnt_}, default=float))
        with (d_ / "aggregate.json").open("x") as fh:
            fh.write(json.dumps(out, default=float))
        res_vol.commit()
        return {"all_equal": out["all_equal"], "n_compared": out["n_compared"], "n_mismatches": 0}
    vals, bvals = {}, {}
    for lb in man["labels"]:
        v_ = json.loads((d_ / "vals" / f"{shard_name(lb)}.json").read_bytes())
        assert v_["label"] == lb and v_["record_sha256"] == man["record_sha256"] and v_["input_sha256"] == man["per_label_input_sha256"][lb], (lb, "vals from another run")
        vals[lb], bvals[lb] = v_["aggin"], v_.get("boot") or {}
    pwt_ = PLANT_PANEL_WITHHELD if man["record_sha256"] == PLANT_RECORD_SHA else None  # PLANT V3 (AMEND6): the plant job's withheld panel_groups
    a_ = agg_check(rec, vals, man["levels"], levels_all=man.get("levels_all"), panel_withheld=pwt_)
    if man.get("value") and sorted(a_["used"]) != sorted(STAGE2_AS_RECORDED if man.get("value_stage") == "2" else ()):  # 09 (10-20), pre-stated: a
        a_["mism"].append(["as_recorded_labels differ from the stage's pre-stated set"])  # Stage 2 value run takes exactly STAGE2_AS_RECORDED, Stage 1 none
    cp_ = compare(a_["cmp"], False)
    val_ = None
    if man.get("value"):  # G7b VALUE: the pre-built domain, the seeded draws' S / Y, and the two source-score plants (each a separate check() call)
        det_, bcmp_ = {}, []
        for lb in man["labels"]:
            p_ = d_ / "detail" / f"{shard_name(lb)}.json"
            sh_ = json.loads(p_.read_bytes()) if p_.exists() else None
            assert sh_ is None or (sh_["label"] == lb and sh_["input_sha256"] == man["per_label_input_sha256"][lb]), (lb, "a detail shard from another input")
            det_[lb] = sh_["out"] if sh_ else None
        dom_ = value_domain(man, det_)
        st2_ = man.get("value_stage") == "2"  # Stage 2 (09, 10-20): R1 / R2
        if st2_:  # the Stage 2 labels are in no aggregate group, by the group definitions (checked, not assumed)
            assert not stage2_in_groups(), "a Stage 2 family is in an aggregate group (STOP)"
        if man.get("boot") and not st2_:  # R2: Stage 2's seeded-draw co-headline aggregate is out of scope (reproduced at S3); Stage 1 computes it
            bcmp_ = agg_boot(bvals, list(man["combo_rows"]), man["levels"], man["boot"]["ids"])
        bc_ = compare(bcmp_, False)

        def akeys(ac_, bcm_):  # the aggregate-level comparisons by key -> (mine, record or npz, verdict); structural findings as MISMATCH
            d0_ = {("agg", k_): (m_, r_, "MATCH" if eq(m_, r_) else "MISMATCH") for k_, m_, r_ in ac_["cmp"] + bcm_}
            assert len(d0_) == len(ac_["cmp"]) + len(bcm_), "aggregate comparison keys are not unique"
            return d0_ | {("agg-structure", m_[0]): (None, None, "MISMATCH") for m_ in ac_["mism"]}
        pls_, pld_ = [], []  # pld_: each plant's aggregate comparisons with values (the Volume detail file only)
        for pl_ in man["plants"]:
            lb_ = pl_["label"]
            pd_ = json.loads((d_ / "plant" / "detail" / f"{shard_name(lb_)}.json").read_bytes())
            assert pd_["label"] == lb_ and pd_["input_sha256"] == man["plant_input_sha256"][lb_] and pd_["record_sha256"] == man["record_sha256"], \
                (lb_, "a plant shard from another run or input")
            pv_ = json.loads((d_ / "plant" / "vals" / f"{shard_name(lb_)}.json").read_bytes())
            vr_ = [q for q in pd_["out"]["rules"] if q["row"] == lb_ and q["rule"].startswith("VALUE:")]
            fired_ = bool(vr_) and (vr_[0].get("value_plant") or {}).get("fired", {}).get("origin") == pl_["origin"]
            o_, a0_ = pl_["origin"], pl_["arm"]
            need_ = {"member row draw-0 catch": lambda k_, o_=o_, a0_=a0_: k_[0] == "row" and k_[1] == o_ and re.fullmatch(rf"[0-9.e-]+\|catch\|{a0_}@B", k_[2]),
                     "pooled_all_origins pooled catch or gap": lambda k_: k_[:2] == ("pooled", "pooled_all_origins") and re.match(r"[0-9.e-]+\|joint\|(catch|gap)\|", k_[2]),
                     "PANEL G": lambda k_: k_[0] == "pooled" and k_[2].endswith("|panel|pooled|G"),
                     "PANEL Y": lambda k_: k_[0] == "pooled" and k_[2].endswith("|panel|pooled|Y"),
                     "PANEL G_minus_Y": lambda k_: k_[0] == "pooled" and k_[2].endswith("|panel|pooled|G_minus_Y"),
                     "seeded-draw member catch": lambda k_, o_=o_: k_[0] == "row" and k_[1] == o_ and k_[2].startswith("boot|") and "|draw 0|" not in k_[2],
                     "seeded-draw pooled gap": lambda k_: k_[0] == "pooled" and k_[2].startswith("boot|") and "|pooled|gap|delta_full" in k_[2] and "|draw 0|" not in k_[2],
                     "co-headline S (draw 0)": lambda k_: k_[0] == "agg" and k_[1].endswith("|S") and not k_[1].startswith("boot|"),
                     "co-headline S (seeded draw)": lambda k_: k_[0] == "agg" and k_[1].startswith("boot|") and "|draw 0|" not in k_[1] and k_[1].endswith("|S")}
            if st2_:  # 09 R1: Stage 2's categories (row and pooled, draw 0 and seeded draws), no co-headline category
                need_ = stage2_need(o_)
            n_ = ledger_of(det_[lb_]) | akeys(a_, bcmp_)
            vp_, bp_ = dict(vals) | {lb_: pv_["aggin"]}, dict(bvals) | {lb_: pv_.get("boot") or {}}
            p_ = ledger_of(pd_["out"]) | akeys(agg_check(rec, vp_, man["levels"], levels_all=man.get("levels_all"), panel_withheld=pwt_),
                                              agg_boot(bp_, list(man["combo_rows"]), man["levels"], man["boot"]["ids"]) if man.get("boot") and not st2_ else [])
            pv2_ = plant_verdict(n_, p_, need_)
            if st2_:  # review F-12: the planted origin is a member row of its combo in this run
                cbp_ = o_.rsplit("[pooled: ", 1)[1].removesuffix("]") if "[pooled: " in o_ else None
                assert cbp_ in man["combo_rows"] and o_ in man["combo_rows"][cbp_], ("STOP: the planted origin is not a member of its combo", o_)
            if st2_:  # 09 (c): the pre-plant record, read by key from the UNPLANTED call's own draw-0 output, names the run the planted call moved
                pre0_ = [r0_.get("plant_precheck") for r0_ in det_[lb_]["rows"] if r0_.get("origin") == o_ and r0_.get("plant_precheck")]
                assert len(pre0_) == 1 and pre0_[0]["arm"] == a0_ and pre0_[0]["run"] == (((vr_[0] if vr_ else {}).get("value_plant") or {}).get("fired") or {}).get("run"), \
                    ("STOP: the unplanted call's pre-plant record does not name the planted run", lb_)
            pv2_ = stage2_plant(n_, p_, pv2_, o_, a0_, man["levels"], pre0_[0]["caught_at_draw0"]) \
                if st2_ else pv2_  # 09 R1 option (A) + R-1 / F-3 / F-4 + 09 (a) / (c): scope, certain = not-yet-caught levels, plain moved == flagged
            pls_.append({"plant": pl_, "fired": fired_} | pv2_ | {"pass": pv2_["pass"] and fired_})
            pld_.append({"plant": pl_, "aggregate_ledger_values": [[k_[1], *v0_] for k_, v0_ in sorted(p_.items()) if k_[0] == "agg"]})  # Astra r6 2 (Volume)
        bnan_ = vacuous(bcmp_)  # Astra r6 1: a seeded-draw S / Y NaN on both sides is a violation, by key
        bst_ = BOOT_OOS_S2 if st2_ else "computed"  # 09 R2: the status written into the result; the gate reads it back from the result dict
        val_ = {"scope": man.get("value_scope"), "domain_missing": dom_, "plants": pls_,
                "boot_aggregate": {"n_compared": len(bcmp_), "mismatches": blind(bc_["mismatches"]), "nan_on_both_sides_keys": bnan_,
                                   "groups": [g_ for g_, _ in BOOT_GROUPS], "draw_ids": (man.get("boot") or {}).get("ids"), "status": bst_}}
        val_["pass"] = (not dom_ and not bc_["mismatches"] and not bnan_ and all(x_["pass"] for x_ in pls_)
                        and [x_["plant"]["label"] for x_ in pls_] == [x_[0] for x_ in (VALUE_PLANTS_S2 if st2_ else VALUE_PLANTS)]  # review O3
                        and boot_gate_ok(st2_, val_["boot_aggregate"]["n_compared"], val_["boot_aggregate"]["status"]))  # R2: Stage 1 > 0; Stage 2 the string
        if man.get("plant_v3"):  # PLANT V3 (PRESTATED_V3 fb34c414 + AMEND1-6): block (iv), block (v) and P1; each fails closed into val_["pass"]
            pv3_ = man["plant_v3"]
            s2c_ = {k_: sum(((((det_.get(lb_) or {}).get("plant_v3") or {}).get("sim2_counts") or {}).get(k_, 0)) for lb_ in man["labels"])
                    for k_ in ("il_reports", "sim2_solves")}  # AMEND7 G2: summed over the normal calls (the launch gate: 60 and 120)
            pcb_ = sorted(man["combo_rows"])  # block (iv)'s domain (Codex V3 #5): the run's combos and the plant points, built before the record is read
            pcm_ = plant_coh_cmp(rec, bvals, vals, pv3_["labels"], pv3_["points"], pcb_)  # block (iv), draw 0
            pcc_ = compare(pcm_, False)
            import plant_rows_v3 as PRV  # block (v): the container's /root copy (image add_local_file); the ported producer is plant_rows_v3_port
            hlp_ = {nm_: hashlib.sha256(Path(m0_.__file__).read_bytes()).hexdigest() for nm_, m0_ in (("plant_rows_v3.py", PRV), ("plant_rows_v3_port.py", PRV.PR))}
            assert hlp_ == pv3_["helpers_sha256"], ("STOP (Codex V3 #6): a deployed helper differs from the sha256 main recorded", sorted(hlp_))
            stv_ = PRV.selftest()
            fb_ = {}
            for nm_, h_ in sorted(pv3_["files"].items()):  # the block (v) inputs main shipped, each against its pin
                b0_ = (d_ / "plant_v3" / nm_).read_bytes()
                assert hashlib.sha256(b0_).hexdigest() == h_, (nm_, "a block (v) input differs from its pin (STOP)")
                fb_[nm_] = b0_
            bvs_, bvd_ = PRV.run_block_v(json.loads(fb_["plant_rows_config.json"]), fb_["plant_record.json.gz"], fb_["base_record.json.gz"],
                                         fb_["plant_rows_target.json"], eq)
            p1r_ = None
            if pv3_.get("p1"):  # P1 (AMEND4 D3, AMEND5 F2; 2e / reviewer: the allowed set from the compared keys)
                lb1_ = pv3_["p1"]["label"]
                pd1_ = json.loads((d_ / "p1" / "detail" / f"{shard_name(lb1_)}.json").read_bytes())
                assert pd1_["label"] == lb1_ and pd1_["input_sha256"] == pv3_["p1_input_sha256"] and pd1_["record_sha256"] == man["record_sha256"], \
                    "a P1 shard from another run or input (STOP)"
                pvl1_ = json.loads((d_ / "p1" / "vals" / f"{shard_name(lb1_)}.json").read_bytes())
                hit_ = ((pd1_["out"].get("plant_v3") or {}).get("p1") or {}).get("hit") or {}

                def cohk_(bv_, va_):
                    return {("agg", k_): (m_, r_, "MATCH" if eq(m_, r_) else "MISMATCH") for k_, m_, r_ in plant_coh_cmp(rec, bv_, va_, pv3_["labels"], pv3_["points"], pcb_)}
                if hit_.get("fired"):
                    o1_, lv1_, d1_ = hit_["origin"], hit_["level"], f"{hit_['delta']:+d}"
                    cb1_, ro1_ = o1_.rsplit("[pooled: ", 1)[1].removesuffix("]"), lb1_.rsplit(" ", 1)[1]
                    n1_ = ledger_of(det_[lb1_]) | cohk_(bvals, vals)
                    p1_ = ledger_of(pd1_["out"]) | cohk_(dict(bvals) | {lb1_: pvl1_.get("boot") or {}}, dict(vals) | {lb1_: pvl1_["aggin"]})
                    need1_, cond1_ = p1_need(n1_, o1_, lv1_, d1_, cb1_, ro1_)  # AMEND6 G1 (Codex V3 #2, r2 #2): the aggregate categories only where the
                    # NORMAL run compares a value there, at P1's combo, level and delta; the draw-0 row categories always
                    pc1ok_ = p1_cond_ok(int(hit_["delta"]), cond1_)  # AMEND7 G1: the pre-stated pair for the delta's sign; else STOP (no verdict, recorded)
                    pvv_ = plant_verdict(n1_, p1_, need1_)
                    flg1_ = {k_ for k_ in set(n1_) & set(p1_) if n1_[k_][2] == "MATCH" and p1_[k_][2] == "MISMATCH"}
                    oa_ = sorted(" | ".join(map(str, k_)) for k_ in flg1_ if not plant_p1_allowed(k_, hit_, cb1_, ro1_))
                    p1r_ = {"hit": hit_, "outside_allowed": oa_[:20], "n_outside_allowed": len(oa_), "aggregate_categories_compared_in_normal_run": cond1_,
                            "aggregate_categories_as_prestated": pc1ok_} | pvv_ | {"pass": bool(pvv_["pass"] and not oa_ and pc1ok_)} \
                        | ({} if pc1ok_ else {"stop": "AMEND7 G1: P1's aggregate categories are not the pre-stated pair for its delta's sign (STOP)"})
                else:
                    p1r_ = {"hit": hit_, "pass": False, "why": "P1 did not fire (STOP)"}
            pass3_ = (not pcc_["mismatches"] and bool(pcm_) and stv_.get("pass") is True and bvs_.get("pass") is True
                      and (p1r_ or {}).get("pass") is True and s2c_ == PLANT_SIM2_COUNTS)  # AMEND9: the 60 / 120 sim2 counts fail closed here too
            val_["plant_v3"] = {"coheadline_signed_plant": {"n_compared": len(pcm_), "mismatches": blind(pcc_["mismatches"])}, "block_v": bvs_,
                                "block_v_selftest": stv_, "p1": p1r_, "not_compared": PLANT_NOT_COMPARED, "sim2": PLANT_SIM2_SCOPE, "sim2_counts": s2c_, "sim2_counts_as_prestated": s2c_ == PLANT_SIM2_COUNTS,
                                "panel_groups": "the plant job's withheld marker, asserted exactly (runner L4514); no PANEL group compared (AMEND6)",
                                "helpers_sha256": hlp_, "step_spec": pv3_.get("step_spec"), "pass": bool(pass3_)}
            val_["pass"] = bool(val_["pass"] and pass3_)
            pld_.append({"plant_v3": {"coheadline_ledger_values": pcc_["ledger_values"],  # Volume only: every value (Codex V3 #7: block (v)'s recomputed
                                      "block_v_detail": bvd_}})  # output, rows / counts / aggregate / gates, and its mismatch paths, saved for audit)
    keys_ = lambda c_, m_: {x[0] for x in c_["mismatches"]} | {x[0] for x in m_}  # noqa: E731
    base_ = keys_(cp_, a_["mism"])
    ctl = []
    for pl_ in PLANT:
        b_ = agg_check(rec, vals, man["levels"], pl_, levels_all=man.get("levels_all"), panel_withheld=pwt_)
        got_ = keys_(compare(b_["cmp"], False), b_["mism"]) - base_
        ctl.append({"plant": list(pl_), "expected_keys": len(b_["expect"]), "flagged_keys": len(got_), "pass": got_ == set(b_["expect"]),
                    "applicable": bool(b_["expect"]),  # Codex r45 #5: a pooled record whose groups are all out of scope or withheld has no finite aggregate
                    # holding the row (the control is reported not applicable; it never passes vacuously unseen)
                    "unexpected_keys": sorted(got_ - set(b_["expect"]))[:20], "missed_keys": sorted(set(b_["expect"]) - got_)[:20]})
    assert set(a_["used"]) <= {f"{f_} {r_}" for f_ in AS_RECORDED for r_ in "CR"}, "an as-recorded label outside the stated gap"
    bes_ = None
    pw_ = rec["meta"].get("pooled_world")
    if pw_:  # Codex r46 #1 / r47 #2: EVERY pooled label (meta.pooled_world.labels, checked against the capture in main) has its rows and a pooled block
        cr_ = man.get("combo_rows") or {}  # per combo whose members are the runner's combo rows (the checked labels are also compared in check())
        a_["mism"] += [[f"members|{lb}|{cb}"] for lb in sorted(pw_["labels"]) for cb in pw_["combos"]
                       if not isinstance((rec["pooled"].get(lb) or {}).get(cb), dict) or rec["pooled"][lb][cb].get("members") != cr_.get(cb)]
        a_["mism"] += [] if sorted(rec["pooled"]) == sorted(pw_["labels"]) == sorted(rec["rows"]) else [["pooled labels: rows or pooled blocks differ from meta"]]
        a_["mism"] += [[f"rows|{lb}"] for lb in sorted(pw_["labels"]) if sorted(rec["rows"].get(lb, {})) != sorted(o_ for cb in pw_["combos"] for o_ in cr_.get(cb, []))]
    if pw_ and pw_.get("per_set_record"):  # G7 (RULE 43): the per-set companion, read here by its pinned path and sha256
        ps_ = pw_["per_set_record"]
        pb_ = (Path("/sol") / ps_["path"]).read_bytes()
        assert hashlib.sha256(pb_).hexdigest() == ps_["sha256"], "per-set companion record sha256"
        n_, b_ = beside_check(rec, json.loads(pb_), man.get("combo_rows"))
        bes_ = {"per_set_record": ps_, "n_compared": n_, "mismatches": b_}
    elif pw_:
        bes_ = {"per_set_record": None, "n_compared": 1, "mismatches": ["per_set_beside without a per-set record"] if "per_set_beside" in rec else []}
    out = {"label": "co-headline S/Y and PANEL excess at draw 0 (G6)", "as_recorded_labels": a_["used"],
           "stated_gap": GAP_TEXT if a_["used"] else "none: every group member's draw-0 inputs computed in this run (G7b VALUE; " + value_scope((man.get("boot") or {}).get("seed")) + ")",
           "reproduced_labels": sorted(vals), "not_compared": ("estimator (B): S|B and caveat_B (A selected); coheadline_signed_plant (optional); "
                                                               + (f"draws other than draw 0 and the {BOOT_N} seeded draws; interval quantiles" if man.get("boot") else "draws 1+")
                                                               + ("; PANEL groups (the plant job's record withholds them; its exact marker is asserted, AMEND6)" if pwt_ else "")),
           "n_compared": len(a_["cmp"]), "nan_on_both_sides": cp_["nan_on_both_sides"], "mismatches": blind(a_["mism"]) + blind(cp_["mismatches"]),
           "positive_controls": ctl, "run_dir": run_dir, "record_sha256": man["record_sha256"]} | ({"per_set_beside": bes_} if bes_ else {}) | (
        {"value": val_, "ledger": cp_["ledger"] + bc_["ledger"]} if val_ else {})  # G7b VALUE (Astra r5 #5): every aggregate comparison, key + verdict
    pcok_ = plant_ctl_ok(ctl) if pwt_ else None  # AMEND7 G3: on PLANT, [co-headline control applicable with expected keys, PANEL control not]
    out["plant_g6_controls_as_prestated"] = pcok_  # (else STOP: the launch RUN_ME's gate; all_equal fails closed)
    out["all_equal"] = not out["mismatches"] and all(c_["pass"] for c_ in ctl) and not (bes_ or {}).get("mismatches") and (val_ is None or val_["pass"]) \
        and pcok_ is not False
    (d_ / "detail").mkdir(parents=True, exist_ok=True)
    with (d_ / "detail" / "aggregate.json").open("x") as fh:
        fh.write(json.dumps(out | {"mismatches_detail": a_["mism"] + cp_["mismatches"], "aggregate_ledger_values": cp_["ledger_values"]}  # Astra r6 2: every
                            | ({"boot_aggregate_ledger_values": bc_["ledger_values"], "plant_ledger_values": pld_} if val_ else {}), default=float))  # value (Volume)
    with (d_ / "aggregate.json").open("x") as fh:
        fh.write(json.dumps(out, default=float))
    res_vol.commit()
    return {"all_equal": out["all_equal"], "n_compared": out["n_compared"], "n_mismatches": len(out["mismatches"])}


SHAPE_NEEDS = {"five": ("key", "fold", "score|score_by_model|score_by_fold"), "simh_sol": ("key", "score"),  # simh_sol: an A25 file without fold
               "own_all_rs": ("key", "score", "score_by_model", "fold")}  # (r36_fill); with fold it is read as five (Astra r1 #3 / Claude N1). rows_rs l.476
READERS = {"check.load (npz per-row files)": ["key", "score", "fold", "score_by_model | score_by_fold (five models)", "own_all_rs: score, score_by_model, fold"],
           "check, union gate (pooled pools npz)": [f"{nm}_{x}" for nm in ("B", "A", "S") for x in ("run", "s", "f", "sbm", "org")],
           "check, PWM (manifest per_label_combo[label | combo] and the sibling *_catalogue.json)": [
               "pooled_file.path", "pooled_file.sha256", "pooled_file.f1cols", "pooled_file.zero_unit_runs", "pooled_file.structural_cells", "f1cols",
               "sizes.{B,A,S}.after", "sizes.{B,A,S}.by_membership_pattern", "sizes.{B,A,S}.zero_unit_runs", "sizes.content_identical_keys",
               "sizes.eligible_units_by_run", "cross_pool_overlaps.keys", "cross_pool_overlaps.roles_by_key", "model_identity.{pinned,folds,probe_file,readout}",
               "origins", "manifest: per_label_combo, catalogue_runs_sha256, meta.git, pool_inputs", "catalogue: runs (digest = the manifest's)"]}
MF_NEED = (("pooled_file", "path"), ("pooled_file", "sha256"), *(("pooled_file", "f1cols", nm) for nm in "BAS"), ("pooled_file", "zero_unit_runs"),
           ("pooled_file", "structural_cells"), *(("f1cols", nm) for nm in "BAS"), ("sizes", "content_identical_keys"), ("sizes", "eligible_units_by_run"),
           *(("sizes", nm, x) for nm in "BAS" for x in ("after", "by_membership_pattern", "zero_unit_runs")), ("cross_pool_overlaps", "keys"),
           ("cross_pool_overlaps", "roles_by_key"), *(("model_identity", x) for x in ("pinned", "folds", "probe_file", "readout")), ("origins",))  # Astra r1 #6


def path_missing(d, path):  # a nested key path absent (each level a dict holding the next key)
    for k_ in path:
        if not isinstance(d, dict) or k_ not in d:
            return True
        d = d[k_]
    return False


def shape_of(names, arrs, kind):
    """G7b FOLD-MODEL --shapes: one per-row npz's arrays (name -> (shape, dtype str)) against its reader. The arrays judged are exactly the ones
    rows_from_npz consumes, by its own branch and precedence (Astra r2 #2): an A25 sim-honest file without fold reads key and score; an own_all_rs
    file reads key, score, score_by_model, fold; any other five-model file reads key, fold, score if present, and score_by_model if present else
    score_by_fold. Judged: key 1-D text; fold 1-D integer (stricter than the reader, which takes integral floats; fails closed); score 1-D; the
    model scores (n, 5). Required names: SHAPE_NEEDS by the same branch. -> (missing required names, faults)"""
    if kind == "simh_sol" and "fold" not in names:
        kind_, used = "simh_sol", ["key", "score"]
    elif kind == "own_all_rs":
        kind_, used = "own_all_rs", ["key", "score", "score_by_model", "fold"]
    else:
        kind_ = "five"
        used = ["key", "fold"] + (["score"] if "score" in names else []) + (
            ["score_by_model"] if "score_by_model" in names else ["score_by_fold"] if "score_by_fold" in names else [])
    miss = [n_ for n_ in SHAPE_NEEDS[kind_] if not any(x_ in names for x_ in n_.split("|"))]
    bad = []
    ks_ = arrs.get("key")
    n0 = ks_[0][0] if ks_ is not None and len(ks_[0]) == 1 else None
    if ks_ is not None and (n0 is None or ks_[1][1:2] not in ("U", "S")):
        bad.append(f"key shape {ks_[0]} dtype {ks_[1]} (want 1-D text)")
    for nm_ in [x_ for x_ in used if x_ in arrs and x_ != "key"]:
        want_ = [n0, 5] if nm_ in ("score_by_model", "score_by_fold") else [n0]
        if arrs[nm_][0] != want_:
            bad.append(f"{nm_} shape {arrs[nm_][0]} (want {want_})")
        if nm_ == "fold" and arrs[nm_][1][1:2] not in ("i", "u"):
            bad.append(f"fold dtype {arrs[nm_][1]} (want integer)")
    return miss, bad


def pooled_shape_faults(arrs):  # a pooled pools npz (pw_prepare l.3934): per role run / org text, s float, f integer, all 1-D of one length; sbm (n, 5)
    bad = []
    for nm in ("B", "A", "S"):
        r_ = arrs.get(f"{nm}_run")
        if r_ is not None and r_[0] == [0] and all((arrs.get(f"{nm}_{x}") or ([None], ""))[0] == [0] for x in ("s", "f", "sbm", "org")):
            continue  # an empty role: pw_prepare writes np.zeros(0) for every array (507-prep NB1)
        n0 = r_[0][0] if r_ is not None and len(r_[0]) == 1 else None
        if r_ is None or n0 is None or r_[1][1:2] not in ("U", "S"):
            bad.append(f"{nm}_run not 1-D text")
            continue
        for x, want, kinds in (("s", [n0], "f"), ("f", [n0], "iu"), ("org", [n0], "US"), ("sbm", [n0, 5], "f")):
            a_ = arrs.get(f"{nm}_{x}")
            if a_ is None or a_[0] != want or a_[1][1:2] not in tuple(kinds):
                bad.append(f"{nm}_{x} shape {None if a_ is None else a_[0]} dtype {None if a_ is None else a_[1]} (want {want}, kind {kinds})")
    return bad


@app.function(image=image, cpu=(2, 2), memory=(8192, 12288), timeout=1500, retries=0, startup_timeout=120, scaledown_window=60,  # 14:10 limits; AMEND11
              volumes={"/sol": modal.Volume.from_name(VOLS["solavg"], version=2).read_only(), "/p2": modal.Volume.from_name(VOLS["p2"], version=2).read_only(),
                       "/conv": modal.Volume.from_name(VOLS["conv"]).read_only(), "/res": res_vol})
def shapes_fn(files: list, pw: dict, labels: list, run_dir: str) -> dict:
    """G7b FOLD-MODEL --shapes (2e's spec (d), the item-39 lesson): every pinned file the fold labels read, opened as the check opens it: sha256, array
    names, shapes and dtypes (no value); the manifest (sha256), its sibling catalogue (its runs digest = the manifest's catalogue_runs_sha256, as
    check() authenticates it; Astra r1 #5), the per_label_combo entries for these labels (nested fields present; Astra r1 #6) and their pooled pools
    npz files. Writes the result to the results Volume (exclusive) and returns it."""
    import numpy as np

    out = {"files": [], "pooled": {}, "readers": READERS}
    for vol, path, sh, kind in files:
        b = (Path(MOUNT[vol]) / path.lstrip("/")).read_bytes()
        with np.load(io.BytesIO(b), allow_pickle=False) as z:
            arrs = {n_: (list(z[n_].shape), z[n_].dtype.str) for n_ in z.files}
        miss, bad = shape_of(set(arrs), arrs, kind)
        out["files"].append({"vol": vol, "path": path, "kind": kind, "read_as": "five" if kind == "simh_sol" and "fold" in arrs else kind,
                             "sha256_ok": hashlib.sha256(b).hexdigest() == sh, "arrays": arrs, "missing_required": miss, "shape_faults": bad})
    mb_ = (Path("/sol") / pw["manifest"]["path"]).read_bytes()
    out["manifest_sha256_ok"] = hashlib.sha256(mb_).hexdigest() == pw["manifest"]["sha256"]
    MF_ = json.loads(mb_)
    mpi_ = MF_.get("pool_inputs") or {}  # Astra r2 #3: the manifest's top-level pins, asserted as check() asserts them before any pooled read
    mods_ = mpi_.get("modules_sha256")
    out["manifest_findings"] = (["meta.git is not the catalogue's code commit"] if (MF_.get("meta") or {}).get("git") != pw.get("git") else []) + (
        ["pool_inputs differ from this capture's (modules aside)"] if not mpi_ or {k_: v_ for k_, v_ in mpi_.items() if k_ != "modules_sha256"}
        != {k_: v_ for k_, v_ in (pw.get("pool_inputs") or {}).items() if k_ != "modules_sha256"} else []) + (
        ["module pins missing, incomplete or not the code commit's files"] if not isinstance(mods_, dict) or not mods_  # Astra r3 #1: every pin a
        or not set(pw.get("required_modules") or []) <= set(mods_) or not all(  # 64-hex sha256 of a file present in the code commit's tree, equal
            isinstance(v_, str) and re.fullmatch(r"[0-9a-f]{64}", v_) and k_ in (pw.get("tree_sha") or {}) and pw["tree_sha"][k_] == v_
            for k_, v_ in mods_.items()) else []) + (
        ["catalogue_runs_sha256 missing"] if not MF_.get("catalogue_runs_sha256") else [])
    ct_ = json.loads((Path("/sol") / pw["manifest"]["path"].replace("_manifest.json", "_catalogue.json")).read_bytes())["runs"]
    out["catalogue"] = {"runs": len(ct_), "digest_equals_manifest": hashlib.sha256(json.dumps({"runs": ct_}, sort_keys=True, default=str).encode()).hexdigest()
                        == MF_.get("catalogue_runs_sha256")}
    plc_ = MF_["per_label_combo"]
    for lb in labels:
        for cb in pw["combos"]:
            e_ = plc_.get(f"{lb} | {cb}")
            r_ = {"in_manifest": e_ is not None}
            if e_ is not None:
                r_["missing_fields"] = [".".join(k_) for k_ in MF_NEED if path_missing(e_, k_)]
                pf_ = e_.get("pooled_file") or {}
                if pf_.get("path"):
                    b = (Path("/sol") / pf_["path"]).read_bytes()
                    with np.load(io.BytesIO(b), allow_pickle=False) as z:
                        arrs = {n_: (list(z[n_].shape), z[n_].dtype.str) for n_ in z.files}
                    r_ |= {"pooled_sha256_ok": hashlib.sha256(b).hexdigest() == pf_.get("sha256"), "arrays": arrs, "shape_faults": pooled_shape_faults(arrs),
                           "f1cols_lengths": {k_: len(v_) for k_, v_ in (pf_.get("f1cols") or {}).items()}}
            out["pooled"][f"{lb} | {cb}"] = r_
    res_vol.reload()
    d_ = Path("/res") / run_dir
    d_.mkdir(parents=True, exist_ok=True)
    with (d_ / "shapes.json").open("x") as fh:
        fh.write(json.dumps(out))
    res_vol.commit()
    return out


@app.function(image=image, cpu=(1, 1), memory=(4096, 8192), timeout=3600, retries=0, startup_timeout=120, scaledown_window=60,
              volumes={"/res": res_vol})  # COMMON_RULES 14:10 limits; AMEND11
def drive(args_: list, run_dir: str, agg: bool = False) -> dict:
    """The ONE spawned call of a launch (under modal run --detach only the last-triggered function outlives the client; 2e 09:5x): it fans the
    per-label checks out inside Modal and waits, so every shard lands on the results Volume even if the laptop drops; then (G6) the aggregate step."""
    outs = list(check.starmap([(*a_, run_dir) for a_ in args_]))
    ag_ = aggregate.remote(run_dir) if agg else None
    return {"run_dir": run_dir, "labels": [a_[0]["labels"][0] for a_ in args_], "all_equal": [o_["all_equal"] for o_ in outs], "aggregate": ag_}


def shard_name(label: str) -> str:  # one file per label, by the label's sha256 (labels hold spaces and "|")
    return hashlib.sha256(label.encode()).hexdigest()[:24]


def assemble(outs: dict, man: dict) -> dict:
    """The results file from the per-label outputs (the launch's returned values, or the Volume shards read back by the reducer)."""
    assert sorted(outs) == sorted(man["labels"]), ("labels differ from the run manifest", sorted(set(outs) ^ set(man["labels"])))
    out = {"rows": [r for lb in man["labels"] for r in outs[lb]["rows"]], "pooled": [r for lb in man["labels"] for r in outs[lb]["pooled"]],
           "rules": [r for lb in man["labels"] for r in outs[lb]["rules"]]}
    out["all_equal"] = all(outs[lb]["all_equal"] for lb in man["labels"])
    out["n_compared"] = sum(outs[lb]["n_compared"] for lb in man["labels"])
    out["selftest"] = outs[man["labels"][0]]["selftest"]  # the same self-test ran in every check() container
    out["empty_blocks"] = {"rows": sorted(f"{r['row']} | {r['origin']}" for r in out["rows"] if r.get("empty_block")),
                           "combos": sorted(f"{q['row']} | {q['combo']}" for q in out["pooled"] if q.get("empty_block"))}
    out |= {k_: man[k_] for k_ in ("record", "record_sha256", "inputs_sha256", "script_sha256", "git", "read_started_pt", "mode", "cap", "nofilter_source",
                                    "cohort_sha256", "app_id", "run_dir")}
    out |= {"level_part": man.get("level_part"), "levels": man.get("levels"), "levels_all": man.get("levels_all")}  # B's split (A22): this result's part,
    # its level subset and the full level set it checked against (unsplit: level_part None, levels = levels_all)
    out["label"] = man.get("value_scope") or ("independent draw-0 check of a Stage A record: public probes x readouts x origins, pooled combos, rule "
                                               "checks (arithmetic from scratch; input resolution reused)")  # G7b VALUE (Astra r5 #6): VALUE_SCOPE verbatim
    if man.get("value"):
        out |= {"value": True, "value_stage": man.get("value_stage"), "scope": man["value_scope"], "boot": man.get("boot"), "plants": man.get("plants")}
    return out


def exit_verdict(out, missing):
    """The run's exit code (2e via 507-prep 10-17 00:11): non-zero when (a) the MAIN comparison is not all equal (every label's rows, pooled blocks
    and rules; the aggregate's mismatches, its G6 controls, per_set_beside, the value domain and the seeded-draw S / Y), or (b) a value plant's
    VERDICT failed (it did not fire, or its flagged keys differ from the moved ones; a value run without both plants), or (c) a shard is missing.
    A plant's own expected mismatches live in its plant/ shards and never enter (a). -> (code, reasons); the caller writes the results file first"""
    ag_ = out.get("aggregates") or {}
    val_ = ag_.get("value")
    rs_ = [f"missing shards: {sorted(missing)}"] if missing else []
    main_ok = out.get("labels_all_equal") is True and (not ag_ or (not ag_.get("mismatches") and all(c_.get("pass") for c_ in ag_.get("positive_controls") or [])
                                                                  and not (ag_.get("per_set_beside") or {}).get("mismatches")))
    w2v_ = ag_.get("w2")  # rc1b W2: its own gates (count per label, every label all_equal, the control) replace the G7b value block
    if w2v_ is not None:
        main_ok = main_ok and bool(w2v_.get("gates")) and all(v_ is True for v_ in w2v_["gates"].values())
        if not (w2v_.get("control") or {}).get("pass"):
            rs_.append("rc1b W2 control verdict failed")
    elif val_ is not None:
        ba_ = val_.get("boot_aggregate") or {}
        main_ok = main_ok and not val_.get("domain_missing") and not ba_.get("mismatches") and not ba_.get("nan_on_both_sides_keys") and val_.get("pass") is True
        pl_ = val_.get("plants") or []
        bad_ = [x_["plant"]["label"] for x_ in pl_ if not (x_.get("pass") is True and x_.get("fired") is True)]
        if bad_ or len(pl_) != len(VALUE_PLANTS):
            rs_.append(f"plant verdict failed: {bad_ or 'plants missing'}")
    elif out.get("value"):
        rs_.append("plant verdict failed: a value run without its value block")
    if not main_ok:
        rs_.append("main comparison not all equal")
    if not rs_ and out.get("all_equal") is not True:  # fail closed: any other way all_equal is not True
        rs_.append("all_equal is not True")
    return (1 if rs_ else 0), rs_


def plant_v3_print(p_):
    """PLANT V3: the keys-only summary line (booleans, counts, key names); never a value."""
    bv_ = p_.get("block_v") or {}
    p1_ = p_.get("p1") or {}
    return {"pass": p_.get("pass"), "coheadline_signed_plant": {"n_compared": p_["coheadline_signed_plant"]["n_compared"],
                                                                 "n_mismatches": len(p_["coheadline_signed_plant"]["mismatches"])},
            "block_v": {"pass": bv_.get("pass"), "gate_fail": bv_.get("gate_fail", False), "all_equal": bv_.get("all_equal"),
                        "printed_rows_all_true": bv_.get("printed_rows_all_true"), "printed_rows_false": sorted(r_ for r_, v_ in (bv_.get("printed_rows") or {}).items() if v_ is not True),
                        "blocks": {b_: {k_: v_[k_] for k_ in ("n_leaf", "n_empty", "n_compared", "n_mismatch", "counts_as_predicted")} for b_, v_ in (bv_.get("blocks") or {}).items()},
                        "controls": {c_: (v_ or {}).get("pass") for c_, v_ in (bv_.get("controls") or {}).items()},
                        "v3_ctl_none_at_start": bv_.get("v3_ctl_none_at_start"), "v3_ctl_none_at_end": bv_.get("v3_ctl_none_at_end")},
            "block_v_selftest": (p_.get("block_v_selftest") or {}).get("pass"),
            "p1": {"pass": p1_.get("pass"), "fired": (p1_.get("hit") or {}).get("fired"), "expected_keys": p1_.get("expected_keys"),
                   "flagged_keys": p1_.get("flagged_keys"), "categories": p1_.get("categories"), "n_outside_allowed": p1_.get("n_outside_allowed"),
                   "aggregate_categories_compared_in_normal_run": p1_.get("aggregate_categories_compared_in_normal_run"),
                   "aggregate_categories_as_prestated": p1_.get("aggregate_categories_as_prestated"), "stop": p1_.get("stop"), "delta_sign": (
                       None if not (p1_.get("hit") or {}).get("fired") else "negative" if p1_["hit"]["delta"] < 0 else "positive")},
            "sim2_counts": p_.get("sim2_counts"), "sim2_counts_as_prestated": p_.get("sim2_counts_as_prestated"),
            "helpers_sha256": p_.get("helpers_sha256")}


def reduce_run(app_id: str, suffix: str = "_fromvolume") -> None:
    """Rebuild a launch's results file from its Volume shards (detach: the launching client may be gone). Every expected label must be there, from
    this run (run_dir), on this record and these per-label inputs. The results file (keys / verdicts) is written in EVERY case, then the exit is
    non-zero on a mismatch, a failed plant verdict or a missing shard (exit_verdict; 2e via 507-prep 10-17 00:11)."""
    run_dir = f"stagea_check/{app_id}"
    man = json.loads(b"".join(res_vol.read_file(f"{run_dir}/manifest.json")))
    have_ = {e_.path.removeprefix(f"{run_dir}/") for e_ in res_vol.listdir(run_dir, recursive=True)}  # every file of this run, relative paths
    missing = sorted([f"{lb} ({shard_name(lb)}.json)" for lb in man["labels"] if f"{shard_name(lb)}.json" not in have_]
                     + (["aggregate.json"] if man.get("aggregate") and "aggregate.json" not in have_ else [])
                     + [f"plant {pl_['label']}" for pl_ in man.get("plants") or [] if f"plant/{shard_name(pl_['label'])}.json" not in have_])
    outs = {}
    for lb in man["labels"]:
        if f"{lb} ({shard_name(lb)}.json)" in missing:
            continue
        sh_ = json.loads(b"".join(res_vol.read_file(f"{run_dir}/{shard_name(lb)}.json")))
        assert sh_["run_dir"] == run_dir and sh_["label"] == lb and sh_["record_sha256"] == man["record_sha256"] \
            and sh_["input_sha256"] == man["per_label_input_sha256"][lb], (lb, "a shard from another run, record or input")
        outs[lb] = sh_["out"]
    if len(outs) != len(man["labels"]):  # (c): a results file with the missing shards named, all_equal false, written before the non-zero exit
        out = {k_: man.get(k_) for k_ in ("record", "record_sha256", "inputs_sha256", "script_sha256", "git", "read_started_pt", "mode", "app_id", "run_dir",
                                           "value", "value_scope", "boot", "plants", "level_part", "levels", "levels_all")}
        out |= {"label": man.get("value_scope") or "independent draw-0 check of a Stage A record (incomplete run)", "all_equal": False,
                "labels_all_equal": False, "missing_shards": missing, "labels_present": sorted(outs),
                "labels_all_equal_present": {lb: o_["all_equal"] for lb, o_ in sorted(outs.items())}, "assembled_from": "results Volume shards (reducer)"}
        dest = fresh_dest(HERE / "results" / f"check_stagea_{man['record_stem']}_{man['read_started_pt']}{suffix}.json")  # Astra r6 3: never a name an
        dest.open("x").write(json.dumps(out, indent=1, default=float) + "\n")  # earlier attempt took (a later complete reduction gets its own)
        code_, rs_ = exit_verdict(out, missing)
        print(json.dumps({"all_equal": False, "missing_shards": missing, "file": str(dest), "exit": code_, "exit_reasons": rs_}, indent=1))
        raise SystemExit(f"STOP: {rs_}")
    out = assemble(outs, man) | {"assembled_from": "results Volume shards (reducer)"}
    out["labels_all_equal"] = out["all_equal"]  # the per-label checks alone (exit_verdict's main comparison reads it with the aggregate)
    if man.get("aggregate") and "aggregate.json" not in missing:  # G6: the S/Y and excess step's keys-only output; the run passes only if it does
        ag_ = json.loads(b"".join(res_vol.read_file(f"{run_dir}/aggregate.json")))
        assert ag_["run_dir"] == run_dir and ag_["record_sha256"] == man["record_sha256"], "an aggregate output from another run or record"
        out |= {"aggregates": ag_, "stated_gap": ag_.get("stated_gap", GAP_TEXT), "all_equal": out["all_equal"] and ag_["all_equal"]}
    if missing:  # every label present, but the aggregate or a plant shard is missing: all_equal false, the missing shards named in the file
        out |= {"all_equal": False, "missing_shards": missing}
    dest = fresh_dest(HERE / "results" / f"check_stagea_{man['record_stem']}_{man['read_started_pt']}{suffix}.json")  # Astra r6 3
    dest.open("x").write(json.dumps(out, indent=1, default=float) + "\n")
    ag_ = out.get("aggregates")
    print(json.dumps({"all_equal": out["all_equal"], "n_compared": out["n_compared"], "file": str(dest)}
                     | ({"aggregates": {"all_equal": ag_["all_equal"], "n_compared": ag_["n_compared"], "n_mismatches": len(ag_["mismatches"]),
                                        "positive_controls_pass": [c_["pass"] for c_ in ag_["positive_controls"]],
                                        "positive_controls_applicable": [c_.get("applicable") for c_ in ag_["positive_controls"]],
                                        "positive_controls_expected_keys": [c_.get("expected_keys") for c_ in ag_["positive_controls"]],
                                        "plant_g6_controls_as_prestated": ag_.get("plant_g6_controls_as_prestated"),
                                        "w2": ({"gates": ag_["w2"]["gates"], "control_pass": ag_["w2"]["control"]["pass"]}
                                               | ({"stored_rows": {k_: v_ for k_, v_ in ag_["w2"]["stored_rows"].items() if k_ != "chosen_sha256"}}
                                                  if ag_["w2"].get("stored_rows") else {}) if ag_.get("w2") else None)}
                                         | ({"per_set_beside": {k_: (len(v_) if k_ == "mismatches" else v_) for k_, v_ in ag_["per_set_beside"].items() if k_ != "per_set_record"}}
                                            if ag_.get("per_set_beside") else {})
                                         | ({"value": {"pass": ag_["value"]["pass"], "domain_missing": len(ag_["value"]["domain_missing"]),
                                                       "boot_aggregate": {k_: (len(v_) if k_ in ("mismatches", "nan_on_both_sides_keys") else v_) for k_, v_ in ag_["value"]["boot_aggregate"].items()
                                                                          if k_ != "draw_ids"},
                                                       "plants": [{k_: x_[k_] for k_ in ("pass", "fired", "expected_keys", "flagged_keys", "categories")}
                                                                  for x_ in ag_["value"]["plants"]]}
                                             | ({"plant_v3": plant_v3_print(ag_["value"]["plant_v3"])} if ag_["value"].get("plant_v3") else {})} if ag_.get("value") else {})}
                        if ag_ else {}), indent=1))
    code_, rs_ = exit_verdict(out, missing)  # (a) / (b) / (c): the results file is already written; then a non-zero exit on any failure
    print(json.dumps({"exit": code_, "exit_reasons": rs_}), flush=True)
    if code_:
        raise SystemExit(f"STOP: {rs_}")


@app.local_entrypoint()
def main(commit: str = "", record: str = "", launch: bool = False, inputs_sha: str = "", nofilter: str = "", dry: str = "",  # noqa: C901
         structural: str = ",".join(MAIN_DEFAULTS[0]), nonstructural: str = ",".join(MAIN_DEFAULTS[1]), reduce: str = "", pw_catalogue: str = "",
         schema: bool = False, fold_labels: str = "", shapes: bool = False, compare_g7b_new: str = "", record_sha: str = "", level_part: str = "",
         w2_rowstore_sha: str = ""):
    # G7b FOLD-MODEL, --fold-labels <one stage's families> (FOLD_STAGES: Stage 1 "sol_a_uncut,own_all_rs_uncut_L48"; Stage 2 "c_asis_uncut,
    # own_all_rs25_uncut_L48,own_all_rs10_uncut_L48"; 2e 10-16 ~20:3x / 22:2x): with --pw-catalogue, the pooled build's
    # fold-model labels (C and R) join the check: the runner's fold-model readers and rules ported here, their unions, gates and pooled files.
    # --shapes (with --launch): instead of the check, a keys / shapes read on Modal CPU of every pinned file the fold labels read and of their
    # pooled pools files and manifest entries (array names, shapes, dtypes; no value)
    # G7 --schema (2e 18:xx): a dry run on a stub pooled record: the capture and its record checks, then every key path the pooled checks read
    # (schema_walk); keys and counts only, nothing launched
    # G7b, --pw-catalogue "<pooled build's code commit>|<manifest path on the solavg Volume>@<sha256>|<combo,...>": a catalogue-stage dry test before any
    # pooled record exists, on the per-set companion record's arguments (--commit/--record) at the pooled code commit: export pass, unions, manifest
    # 09's ruling (2e 02:3x) on the canonical classes (config/canonical_classes.json, 09 RULE §7): structural = the cause precedes, or is independent
    # of, the arm's own policy behaviour; errored runs (EPCE, OutputContractError, RetryError and other wrappers by their error) are not
    git = lambda *a, cwd=HERE: subprocess.run(["git", *a], capture_output=True, check=True, cwd=cwd).stdout  # noqa: E731
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    if reduce:  # --reduce <app id>: assemble a launch's results file from its Volume shards; nothing else runs
        return reduce_run(reduce)
    if compare_g7b_new:  # G7b FOLD-MODEL (2e 21:1x), --compare-g7b-new <this run's results file>: the 8 labels vs the G7B_PRIOR record, by key; local
        ob_ = git("show", f"{G7B_PRIOR[0]}:{G7B_PRIOR[1]}")
        assert sha(ob_) == G7B_PRIOR[2], "the prior G7b record differs from its pinned sha256"
        nb_ = (HERE / "results" / Path(compare_g7b_new).name).read_bytes()
        df_, rn_ = compare_g7b(json.loads(ob_), json.loads(nb_), 8, [f"{p_} {r_}" for p_ in EXT_UNC for r_ in "CR"])  # the 8 by name (Astra r3 NB)
        dest = HERE / "results" / f"check_stagea_g7b_compare_{Path(compare_g7b_new).stem}.json"
        dest.open("x").write(json.dumps({"prior": f"{G7B_PRIOR[1]} @ {G7B_PRIOR[0]} sha256 {G7B_PRIOR[2]}", "new": Path(compare_g7b_new).name,
                                         "new_sha256": sha(nb_), "differences": df_, "verdict": "UNCHANGED" if not df_ else "DIFFERENT (STOP: report by key; 09 rules)"}
                                        | rn_, indent=1) + "\n")
        print(json.dumps({"verdict": "UNCHANGED" if not df_ else "DIFFERENT", "differences": len(df_), "first": df_[:5],
                          "renamed_rows_nonzero": {k_: v_ for k_, v_ in rn_["renamed_rows_by_label_combo"].items() if v_}, "file": str(dest)}, indent=1))
        if df_:
            raise SystemExit("--compare-g7b-new: the 8 labels' gates differ from the prior G7b record (STOP; report by key)")
        return
    NF = None
    if nofilter:  # the no-filter headline cohort by key (09 @ 01012e87e): <commit>:<path>:<sha256> of 507-prep's nofilter_exclusions record
        c_, p_, h_ = nofilter.split(":")
        b_ = git("show", f"{c_}:{p_}")
        assert sha(b_) == h_, "no-filter cohort file sha256"
        NF = json.loads(b_)["origins"]
        ZR, ZRO = [], {}
        for o_, d_ in NF.items():
            z_ = ([f"{a_}|{c_}" for c_, arms_ in (d_.get("kept_zero_tool_arms") or {}).items() for a_ in arms_] + list(d_.get("kept_zero_tool_runs") or [])
                  + list(d_.get("structural_zero_runs") or {}))  # v3 @ 48a90b1d2: structural_zero_runs {run key: evidence}
            for r_, ev_ in (d_.get("structural_zero_runs") or {}).items():  # kept only if completed without a sample error and with no scored action
                assert ev_.get("sample_error") is None and ev_.get("n_tool_nonsubmit") == 0, (o_, r_, "a structural-zero run with a sample error or tool calls")
            assert all(r_.split("|", 1)[1] in set(d_["headline_cells"]) for r_ in z_), (o_, "a kept zero-tool run outside its own origin's headline (Codex r12)")
            ZRO[o_] = sorted(set(z_))
            ZR += z_
        assert all(r_.split("|", 1)[0] in A3 for r_ in ZR), "a kept zero-tool run that is not an attack arm"
        head_ = set().union(*[set(d_["headline_cells"]) for d_ in NF.values()])
        assert all(r_.split("|", 1)[1] in head_ for r_ in ZR), "a kept zero-tool run whose cell is not in a headline cohort (Codex r11)"
    DRY = bool(dry)
    if DRY:  # capture-only dry run on a manifest with no record yet (the second refresh's TEST): "<code commit>|<inputs path@sha256>|<name tags>|<cohort list path@sha>"
        assert not launch, "a dry capture never launches"
        code_d, inp_d, tags_d, cl_d, *ex_d = dry.split("|")  # optional 5th: an exclusions file "<commit>:<path>:<sha256>" (TEST validation only);
        # full length (G1 dev checks on a TEST INPUTS_FULL): 6th "<INPUTS_FULL path>@<sha256>", 7th after_32k_log "<c>:<path>:<line>", 8th rec32_final "<c>:<path>@<sha>"
        rpath = f"{D507}/results/alltasks_CRD{tags_d}_DRY.json"
        m_ = {"git": git("rev-parse", code_d).decode().strip(), "inputs_final": {"path": inp_d.rsplit("@", 1)[0], "sha256": inp_d.rsplit("@", 1)[1]},
              "roots": "v2", "readouts": "CRD", "cap": None, "cohort_list": {"ref": cl_d} if cl_d else None, "test": "_TEST" in tags_d}
        if ex_d and ex_d[0]:
            m_["exclusions"] = dict(zip(("commit", "path", "sha256"), ex_d[0].split(":")))
        if len(ex_d) >= 4 and ex_d[1]:  # the full-length arguments as a record would carry them, from the TEST INPUTS_FULL's own pins
            fp_d, fh_d = ex_d[1].rsplit("@", 1)
            FIN_d = json.loads(git("show", f"{code_d}:{D507}/{fp_d}"))
            lc_d, lp_d, ln_d = ex_d[2].split(":", 2)
            rcp_d, rh_d = ex_d[3].rsplit("@", 1)
            m_ |= {"full_length": {"inputs_full": {"path": fp_d, "sha256": fh_d}, "estimator": {"used": "B"}},
                   "il_arms": {"block": {"path": FIN_d["ilarms"]["path"], "sha256": FIN_d["ilarms"]["sha256"]}},
                   "honest_b_relaunch": {"pin": {"path": FIN_d["relaunch_members"]["path"], "sha256": FIN_d["relaunch_members"]["sha256"]}},
                   "calibration": {"block": {"path": FIN_d["e12"]["path"].removeprefix(D507 + "/"), "sha256": FIN_d["e12"]["sha256"]}},
                   "after_32k_log": {"commit": lc_d, "path": lp_d, "line": int(ln_d), "rec32_final": dict(zip(("commit", "path"), rcp_d.split(":", 1))) | {"sha256": rh_d}}}
        rec, recb = {"meta": m_, "rows": {}, "pooled": {}}, b""
    else:
        rpath = record if "/" in record else f"{D507}/results/{record}"
        recb = git("show", f"{commit}:{rpath}")
        if rpath.endswith(".gz"):  # G7b VALUE: pooled A is committed as gzip -n -9 of the Volume file; the decompressed bytes are the record (2981a831...)
            recb = gzip.decompress(recb)
        assert not (record_sha or rpath.endswith(".gz")) or sha(recb) == record_sha, "the record's (decompressed) sha256 is not --record-sha"
        rec = json.loads(recb)
        m_ = rec["meta"]
    CAT = bool(pw_catalogue)
    FL_ = [x_.strip() for x_ in fold_labels.split(",") if x_.strip()]
    assert not FL_ or sorted(FL_) in [sorted(v_) for v_ in FOLD_STAGES.values()], \
        "--fold-labels: exactly one stage's families (Stage 1: the lead's two; Stage 2: the sensitivity rows' three)"
    assert not shapes or FL_, "--shapes reads the fold labels' files: pass --fold-labels"
    if CAT:  # G7b: the per-set record's arguments, the pooled build's code commit, the catalogue stage
        assert not DRY and record, "--pw-catalogue needs --commit/--record of the per-set companion record (and no --dry)"
        cc_, mf_c, cbs_ = pw_catalogue.split("|")
        assert not m_.get("pooled_world"), "--pw-catalogue takes the per-set record"
        m_ = dict(m_) | {"git": git("rev-parse", cc_).decode().strip(), "pooled_world": {"stage": "catalogue", "combos": cbs_.split(","),
                                                                                       "manifest": dict(zip(("path", "sha256"), mf_c.rsplit("@", 1)))}}
    nm_, mode = record_mode(Path(rpath).name)  # 507-prep's stub pooled record (2e 16:5x) carries the real record's name after STUB_; route C's A after
    # routec_A_ (G7b on route C, 09 cond. 4: the other names' modes are unchanged, selftest_routec)
    W2K_ = W2_RECORDS.get(sha(recb), (None,))[0] if not DRY else None  # rc1b W2: Stage D ("pertask") or kappa ("kappa"), by the record's sha256
    assert W2K_ == "kappa" or not w2_rowstore_sha, "rc1b W2: --w2-rowstore-sha is for a W2b (kappa) run only (STOP)"  # AMEND6: never ignored
    if W2K_:  # its launch flags are A's plus one (asserted against the pinned spec at the capture): A's name tokens select A's flags
        nm_, mode = record_mode(Path(W2_A_REF[1]).name)
    assert not Path(rpath).name.startswith("STUB_") or schema, "a STUB record runs with --schema only"
    FULLREC = "_FULL" in nm_  # a full-length (uncut) all-tasks record (G1)
    PWm = m_.get("pooled_world")  # G7: a pooled-world record (09 RULE item 42)
    VAL_ = bool(FL_) and bool(PWm) and PWm.get("stage") == "scored"  # G7b VALUE (2e 22:3x): the draw-0 value reproduction on a pooled scored record
    assert w2_value_path_ok(W2K_, VAL_), "rc1b W2 (AMEND7): a W2 record off the value path (no --fold-labels) would skip every W2 gate (STOP)"
    assert not FL_ or CAT or (VAL_ and value_stage(FL_) in ("1", "2")), \
        "--fold-labels: a --pw-catalogue check, or one stage's families on a pooled scored record (the value reproduction)"
    VPL_ = (VALUE_PLANTS_S2 if value_stage(FL_) == "2" else VALUE_PLANTS) if (VAL_ and not W2K_) else ()  # the value plants of this run's stage (Stage 2:
    # 09, 10-20); rc1b W2 runs its own control call instead (AMEND1 section 4)
    assert not PWm or (mode == "alltasks" and FULLREC and PWm.get("stage") in ("scored", "catalogue")), "a pooled-world record outside the full-length all-tasks build"
    STUB = Path(rpath).name.startswith("STUB_")  # a stub's pins may be placeholders ("<sha> <full 40-hex sha at launch>"): resolved and reported, not asserted
    stub_notes = []
    if STUB and not re.fullmatch(r"[0-9a-f]{40}", str(m_.get("git"))):
        stub_notes.append(f"meta.git placeholder {m_.get('git')!r}: resolved from its leading sha")
        m_ = dict(m_) | {"git": git("rev-parse", re.match(r"[0-9a-f]{7,40}", str(m_["git"])).group(0)).decode().strip()}
    code = m_["git"]
    tree = TREES / code[:12]
    if not tree.exists():
        TREES.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(tree), code)
    assert git("rev-parse", "HEAD", cwd=tree).decode().strip() == code, "tree is not at the record's commit"
    if STUB and sha((tree / D507 / "run_peraction311.py").read_bytes()) != m_.get("script_sha256"):
        stub_notes.append("meta.script_sha256 is not the code commit's runner (placeholder)")
    assert DRY or CAT or STUB or sha((tree / D507 / "run_peraction311.py").read_bytes()) == m_["script_sha256"], "runner sha256 differs from the record's"
    if STUB:  # 507-prep's stub (synthetic values, placeholder pins): no runner capture is possible; the schema walk, the runner's combo members at the
        assert schema and m_.get("pooled_world"), "a STUB record: --schema on a pooled stub only"  # code commit, and the aggregation run on the stub's own
        sys.path.insert(0, str(tree / D507))  # per-label blocks (every label as recorded), with both positive controls: keys and counts only
        import run_peraction311 as RPs

        PWs = m_["pooled_world"]
        labs_ = sorted(lb_ for lb_ in rec["rows"] if lb_.rsplit(" ", 1)[0] in EXT_UNC)
        cbr_ = {cb: [f"{o} [pooled: {cb}]" for o in getattr(RPs, "PW_MEMBERS", RPs.COMBOS_ALLO)[cb]] for cb in PWs["combos"]}
        orig_ = [o for cb in PWs["combos"] for o in cbr_[cb]]
        rows_bad_ = sorted(f"{lb_} | {o}" for lb_ in labs_ for o in orig_ if o not in rec["rows"].get(lb_, {})) + sorted(
            f"{lb_} | {o}" for lb_ in labs_ for o in rec["rows"].get(lb_, {}) if o not in orig_) + sorted(  # Codex r46 #2: every label's members too
            f"members | {lb_} | {cb}" for lb_ in PWs["labels"] for cb in PWs["combos"]  # Codex r47 #3: every expected label x combo block, required
            if not isinstance((rec["pooled"].get(lb_) or {}).get(cb), dict) or rec["pooled"][lb_][cb].get("members") != cbr_[cb]) + (
            [] if sorted(rec["pooled"]) == sorted(PWs["labels"]) == sorted(rec["rows"]) else ["labels: rows or pooled blocks differ from meta.pooled_world.labels"]) + sorted(
            f"rows | {lb_}" for lb_ in PWs["labels"] if sorted(rec["rows"].get(lb_, {})) != sorted(orig_))
        lv0_ = sorted(float(k_) for k_ in rec["pooled"][labs_[0]][PWs["combos"][0]]["joint"])
        mw_ = schema_walk(rec, labs_, orig_, list(PWs["combos"]), lv0_)
        vals_ = {}
        for lb_ in labs_:  # the external labels' S/Y and excess inputs taken from the stub's own per-label blocks (as recorded), mode joint
            for cb in PWs["combos"]:
                for k_ in (rec["pooled"].get(lb_, {}).get(cb, {}).get("joint") or {}):
                    q = rec["pooled"][lb_][cb]["joint"][k_]
                    pn = q["panel"]
                    vals_.setdefault(lb_, {}).setdefault(cb, {})[k_] = {
                        "df": q["gap"]["delta_full"]["point"], "yr": q["catch"]["run2"]["point"] - q["catch"]["run1"]["point"], "rd": q["readable"]["all"],
                        "st": {s_: [pn[s_]["G"]["point"], pn[s_]["Y"]["point"]] for s_ in pn}, "sat": {s_: pn[s_]["saturated"] for s_ in pn},
                        "rdo": {s_: pn[s_]["readable"] for s_ in pn if s_ not in ("pooled", "pooled_abs_per_origin")}}
        # the runner's draw 0 is mode-independent; the stub's synthetic 'fixed' blocks are not (counted below), so the aggregation is compared on mode joint
        fj_ = sum(1 for lb_ in rec["pooled"] for cb, b0_ in rec["pooled"][lb_].items() if isinstance(b0_, dict) and isinstance(b0_.get("fixed"), dict)
                  for k_, q in b0_["fixed"].items() if q.get("gap", {}).get("delta_full", {}).get("point") != b0_["joint"][k_].get("gap", {}).get("delta_full", {}).get("point"))
        jo_ = lambda ks: {x_ for x_ in ks if "|joint|" in f"|{x_}|"}  # noqa: E731
        a_ = agg_check(rec, vals_, lv0_)
        cp_ = compare(a_["cmp"], False)
        base_ = {x[0] for x in cp_["mismatches"]} | {x[0] for x in a_["mism"]}
        ctl_ = []
        for pl_ in PLANT:
            b_ = agg_check(rec, vals_, lv0_, pl_)
            got_ = jo_(({x[0] for x in compare(b_["cmp"], False)["mismatches"]} | {x[0] for x in b_["mism"]}) - base_)
            ctl_.append({"plant": list(pl_), "expected_keys": len(jo_(b_["expect"])), "flagged_keys": len(got_), "pass": got_ == jo_(b_["expect"]),
                         "applicable": bool(jo_(b_["expect"])), "mode": "joint"})
        a_["mism"] = [m_ for m_ in a_["mism"] if not m_[0].split("|")[1:2] == ["fixed"]]
        cp_["mismatches"] = [m_ for m_ in cp_["mismatches"] if "|joint|" in f"|{m_[0]}|"]
        a_["cmp"] = [c_ for c_ in a_["cmp"] if "|joint|" in f"|{c_[0]}|"]
        stub_notes.append(f"the stub's fixed-mode draw-0 Delta_full differs from joint in {fj_} pooled blocks (synthetic); aggregation compared on joint")
        stem_s = Path(rpath).stem
        dest = HERE / "results" / f"check_stagea_schema_{stem_s}_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
        res_s = {"record": f"{rpath} @ {commit}", "record_sha256": sha(recb), "code_commit": code, "labels": labs_, "combos": list(PWs["combos"]),
                 "combo_rows_from_runner": cbr_, "levels": [f"{x:g}" for x in lv0_], "capture": "not run (stub pins are placeholders)", "stub_notes": stub_notes,
                 "rows_vs_runner_combo_members": rows_bad_, "missing_paths": mw_,
                 "aggregation_on_stub": {"n_compared": len(a_["cmp"]), "mismatches": blind(a_["mism"]) + blind(cp_["mismatches"]), "positive_controls": ctl_,
                                         "as_recorded_labels": a_["used"], "rule": "every S/Y and PANEL aggregate rebuilt from the stub's own per-label blocks"}}
        dest.open("x").write(json.dumps(res_s, indent=1) + "\n")
        print(json.dumps({"schema": "PASS" if not mw_ and not rows_bad_ else "FINDINGS", "n_patterns_missing": len(mw_), "rows_vs_runner": len(rows_bad_),
                          "aggregation_mismatches": len(res_s["aggregation_on_stub"]["mismatches"]), "aggregation_compared": len(a_["cmp"]),
                          "controls": [(c_["pass"], c_["applicable"], c_["expected_keys"]) for c_ in ctl_], "stub_notes": stub_notes, "file": str(dest)}, indent=1))
        return
    ib_ = (tree / D507 / m_["inputs_final"]["path"]).read_bytes()
    assert sha(ib_) == m_["inputs_final"]["sha256"], "INPUTS_FINAL sha256"
    sa_ = json.loads(ib_).get("pools", {}).get("stage_a", {})
    exp_rep = {o: sorted(r_.removesuffix("__sim_honest") for r_ in sa_.get(o, {}).get("replacement_roots", [])) for o in ("S_new", "S_newmains")}
    sys.path.insert(0, str(tree / D507))
    import run_peraction311 as RP  # input resolution only

    jobs, uploads = [], {}

    class _BU:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def put_file(self, f, path):
            uploads[path.lstrip("/")] = f.read()

    class _Vol:
        def batch_upload(self, *a, **k):
            return _BU()

        def listdir(self, *a, **k):  # nothing on the "Volume": every input blob is put (and captured here)
            return []

    class _Row:
        def starmap(self, js):
            jobs.extend(js)
            raise _Stop()

    class _Drive:  # G1c: a remote-driver build (full length) uploads its job spec and spawns drive(); the capture stops at the spawn
        def spawn(self, *a, **k):
            raise _Stop()

        def with_options(self, *a, **k):  # Codex r40 #2: the pooled dispatcher spawns drive.with_options(timeout=...)
            return self

    RP.row, RP.vol_out = _Row(), _Vol()
    if hasattr(RP, "drive"):
        RP.drive = _Drive()
    if hasattr(RP, "run_rows"):  # the second refresh's runner runs all-origins cohort passes first: capture every job at run_rows

        def _cap_rows(js):
            jobs.extend(js)
            raise _Stop()
        RP.run_rows = _cap_rows
    ipath = m_["inputs_final"]["path"]
    try:
        if mode == "alltasks":
            cl_ = m_.get("cohort_list")
            kwa_ = {"cohort_list": cl_["ref"] if isinstance(cl_, dict) else cl_} if cl_ else {}  # the no-filter job (--cohort-list)
            if "ownallLOO" in nm_:  # the all-tasks own_all_LOO addendum (507-prep loo3 @ a97897cef)
                kwa_["own_all_loo"] = True
            if "_allorigins" in nm_:  # the second refresh (refresh2): the all-origins rows and combos
                kwa_["all_origins"] = True
            if "_sweep15" in nm_:
                kwa_["levels"] = "sweep15"
            if "_TEST" in nm_:
                kwa_["test"] = True
            if m_.get("fold_minus1_drops"):  # PROVISIONAL, capture argument added without Codex review (0d's GO via 2e 05:2x): the pinned drops file, passed through
                kwa_["fold_drops"] = f"{m_['fold_minus1_drops']['path'].removeprefix(D507 + '/')}@{m_['fold_minus1_drops']['sha256']}"
            if "_plant" in nm_ and sha(recb) == PLANT_RECORD_SHA and "_ADD_" not in nm_:  # PLANT V3: route C's PLANT record is rebuilt with the
                kwa_["plant"] = True  # runner's --plant (A's flags plus --plant: NOTE_routec_plant @ a2d637568 L20-27; rp L5464-5465, L5822-5826)
            elif "_plant" in nm_ or "_ADD_" in nm_:  # G1: plant and addendum records are other jobs with their own gates
                raise SystemExit("a plant or addendum record: not covered by this check")
            if "_e12" in nm_:  # G1 (Codex r28 item 2): the epoch 1 + 2 calibration switch, the record's own block pin (path relative to the runner folder)
                cb_ = m_["calibration"]["block"]
                kwa_ |= {"calib": "e12", "e12": f"{cb_['path']}@{cb_['sha256']}"}
            if FULLREC:  # G1 (r28 item 2): the full-length build's arguments, each from the record's own meta
                fl_ = m_["full_length"]
                kwa_ |= {"inputs_full": f"{fl_['inputs_full']['path']}@{fl_['inputs_full']['sha256']}",  # G7b VALUE: the record's computed estimator flag
                         "estimator": fl_["estimator"].get("computed_flag", fl_["estimator"]["used"]),  # (pooled A: used "A", computed with --estimator B)
                         "remote_driver": True}  # G1c: the runner builds a full-length record only through its remote driver
                if m_.get("il_arms"):
                    kwa_["il_arms"] = f"{m_['il_arms']['block']['path'].removeprefix(D507 + '/')}@{m_['il_arms']['block']['sha256']}"
                if m_.get("honest_b_relaunch"):
                    pin_ = m_["honest_b_relaunch"]["pin"]
                    kwa_["honest_b_relaunch"] = f"{pin_['path'].removeprefix(D507 + '/')}@{pin_['sha256']}"
                a32_ = m_["after_32k_log"]
                kwa_ |= {"after_32k_log": f"{a32_['commit']}:{a32_['path']}:{a32_['line']}",
                         "rec32_final": f"{a32_['rec32_final']['commit']}:{a32_['rec32_final']['path']}@{a32_['rec32_final']['sha256']}"}
                if (m_.get("adds") or {}).get("cell_hits_ext"):  # the record's declared switch (meta.adds); its extra fields are not compared here
                    kwa_["cell_hits_ext"] = True
                if routec_zero_unit_arg(fl_):  # G7b on route C: the record's pinned pilot zero-unit list, passed as its build passed it (rp l.6332)
                    kwa_["routec_zero_unit"] = routec_zero_unit_arg(fl_)
            import inspect

            if "copies_root" in inspect.signature(RP.main.info.raw_f).parameters:  # G7b on route C: runners from 6ef198426 read the fpr507 manifest
                kwa_["copies_root"] = COPIES_ROOT  # copies through copies.py only; its root() STOPs without the flag or the environment (input resolution)
            if PWm:  # G7: the pooled-world build's arguments (runner @ e9b78de6d main), each from the record's own meta.pooled_world
                pj_ = lambda d_: f"{d_['path']}@{d_['sha256']}" if d_ else ""  # noqa: E731
                scd_ = PWm["stage"] == "scored"  # Codex r40 #1: the manifest and authorized-overlap pins go to the scored stage only (the runner rejects them otherwise)
                pil_ = PWm.get("pilot")  # @ 8a15fb10f: the pilot's families (a list), its rows' resources from --pw-row-res (outside the code pin)
                rr_ = PWm.get("row_resources")
                kwa_ |= {"calib_pooled_world": True, "pw_stage": PWm["stage"], "pw_manifest": pj_(PWm.get("manifest")) if scd_ else "",
                         "pw_authorized": pj_(PWm.get("authorized_overlaps")) if scd_ else "", "per_set_record": pj_(PWm.get("per_set_record")) if scd_ else "",
                         "pw_without_s4": "pooled_all_origins_without_S4" in PWm["combos"],
                         "pw_pilot": ",".join(pil_) if isinstance(pil_, list) else (pil_ or ""), "pw_level_part": PWm.get("level_part") or ""}
                import inspect

                if isinstance(rr_, dict) and "cpu" in rr_ and "pw_row_res" in inspect.signature(RP.main.info.raw_f).parameters:  # runners from 8a15fb10f
                    kwa_["pw_row_res"] = f"{rr_['cpu'][0]},{rr_['cpu'][1]},{rr_['memory'][0]},{rr_['memory'][1]},{rr_['timeout']}"
                if "_rule37" in nm_:  # Codex r40 #10: a pooled RULE 37 record's levels
                    kwa_["levels"] = "rule37"
            if W2K_:  # rc1b W2: the record's pinned launch flags (spec at the record's code commit, by sha256; its script is this runner): A's flags
                _, w2sp_, w2sh_ = W2_RECORDS[sha(recb)]  # plus --pertask / --kappa; every other set flag must equal the spec's (w2_flags_viol)
                spb_ = git("show", f"{code}:{w2sp_}")
                assert sha(spb_) == w2sh_, "rc1b W2: the pinned step spec's sha256 (STOP)"
                spj_ = json.loads(spb_)
                assert spj_["script_blob"] == git("rev-parse", f"{code}:{D507}/run_peraction311.py").decode().strip() and code == W2_CODE, "rc1b W2: spec / runner"
                kwa_[W2K_] = spj_["entry"]["flags"][W2K_]
                if "pw_row_res" not in spj_["entry"]["flags"]:  # a resource flag, not a value input (cpu, memory, timeout; 09, AMEND4): A's meta carries it, this launch did not pass it
                    kwa_.pop("pw_row_res", None)
                assert kwa_[W2K_] == f"{W2_A_REF[0]}:{W2_A_REF[1]}@{W2_A_REF[2]}", \
                    "rc1b W2: the --pertask / --kappa reference is not route C A (STOP)"
                w2fv_ = w2_flags_viol({"readouts": "C", "roots": m_["roots"], "alltasks": True, "inputs": f"{ipath}@{m_['inputs_final']['sha256']}"} | kwa_,
                                      spj_["entry"]["flags"])
                assert not w2fv_, ("rc1b W2: the capture's flags differ from the record's pinned launch flags (STOP)", w2fv_)
            RP.main.info.raw_f(readouts="C", roots=m_["roots"], alltasks=True, inputs=f"{ipath}@{m_['inputs_final']['sha256']}", **kwa_)
        elif mode == "sweep":
            RP.main.info.raw_f(readouts=m_["readouts"], roots=m_["roots"], sweep=True, inputs=f"{ipath}@{m_['inputs_final']['sha256']}")
        elif mode == "union_sol":
            RP.main.info.raw_f(readouts=m_["readouts"], roots=m_["roots"], sweep=True, union_sol=True, inputs=f"{ipath}@{m_['inputs_final']['sha256']}")
        else:
            kw_ = {"own_all_loo": True} if nm_.startswith("pooled455v2_ownallLOO") else {}  # the LOO-3 joiner record (09 @ 8b9ca793f, part 1)
            RP.main.info.raw_f(readouts=m_["readouts"], roots=m_["roots"], pooled=True, cap=str(m_["cap"]) if m_.get("cap") is not None else "",
                               inputs=f"{ipath}@{m_['inputs_final']['sha256']}", **kw_)
    except _Stop:
        pass
    if not jobs:  # G1c: a remote-driver build left its job list in the one spec it uploaded (pickle) before drive.spawn
        specs_ = [k_ for k_ in uploads if "/drivers/" in "/" + k_ and k_.endswith(".pkl")]
        assert len(specs_) == 1, ("expected one driver spec upload", specs_)
        import pickle

        spec_ = pickle.loads(uploads[specs_[0]])
        jobs.extend(spec_["jobs"])
        cap_mods_ = ((spec_.get("meta") or {}).get("pooled_world") or {}).get("modules_sha256")  # Codex r46 #4: the captured code pins (exact)
    else:
        cap_mods_ = None
    assert jobs, "no jobs captured"
    xr0_ = sorted(((m_.get("full_length") or {}).get("s_old_simh_extra_roots") or {}))  # G7b on route C (09 cond. 3): fail closed on pilot zero-unit runs
    prw_ = rec.get("rows") or {}
    if W2K_ == "pertask":  # rc1b W2: Stage D's per-task rows carry no s_old_simh_extra; its jobs are A's (asserted by the flags above), so the
        arb_ = gzip.decompress(git("show", f"{W2_A_REF[0]}:{W2_A_REF[1]}"))  # fail-closed pilot zero-unit check reads route C A's rows (by sha256)
        assert sha(arb_) == W2_A_REF[2], "rc1b W2: route C A's record sha256 (STOP)"
        prw_ = json.loads(arb_)["rows"]
    pzv_ = pilot_zero_unit_viol(prw_, [cf_ for _, cf_, _ in jobs], xr0_, require=bool(PWm) and PWm.get("stage") == "scored")
    assert not pzv_, ("route C pilot zero-unit runs or unlisted counts: not ported here (STOP)", len(pzv_), pzv_[:3])
    pzi_ = {"extra_roots": xr0_, "rows_with_entry": sum(isinstance((r_ or {}).get("s_old_simh_extra"), dict) for od_ in prw_.values()
                                                         for r_ in (od_ or {}).values() if isinstance(r_, dict)),
            "jobs_with_s_old_only_entry": sum(any(isinstance(e_, dict) and e_.get("s_old_only") for es_ in (cf_.get("simh_old") or {}).values() for e_ in es_ or [])
                                              for _, cf_, _ in jobs), "violations": 0,
            "rule": "09 cond. 3: zero-unit pilot members are not ported; every s_old_simh_extra zero_unit / zero_unit_unlisted and every captured "
                    "S_old-only zero_unit_ok is an empty list, else STOP"}
    refs = jobs[0][2]["_ref"]
    if STUB and {k_: v["sha256"] for k_, v in refs.items()} != {k_: (v or {}).get("sha256") for k_, v in (m_.get("shipped_inputs") or {}).items()}:
        stub_notes.append("meta.shipped_inputs differ from the captured ones (placeholder)")
    assert DRY or STUB or {k_: v["sha256"] for k_, v in refs.items()} == {k_: v["sha256"] for k_, v in m_["shipped_inputs"].items()}, "shipped inputs differ from the record's"
    blobs = {k_: uploads[v["path"]] for k_, v in refs.items()}
    assert all(sha(blobs[k_]) == refs[k_]["sha256"] for k_ in refs)
    by_origin, job_of = {}, {}
    for lb, cf, _ in jobs:
        o = (cf.get("origin", "S_new") + (" without S4" if cf.get("no_s4") else "") + (" (all-units deviation)" if cf.get("allunits_dev") else "")
             + (" (registered cohort)" if cf.get("cohort") == "registered" else "")
             + (f" [{cf['sp_block']}]" if cf.get("origin") == "S_paper493" and cf.get("sp_block") else "") + (" (all-origins draws)" if cf.get("merge") else "")
             + (f" [pooled: {cf['pw_combo']}]" if cf.get("pw_combo") else ""))  # the runner's vtag (G7: the pooled combo suffix, item 45(3))
        ol_ = cf.get("full_as", lb)  # G1 (r28 item 4): a full-length job runs under its 32k label and leaves under full_as ("FIBS|uncut C")
        assert ol_ not in by_origin.get(o, {}), (o, ol_, "two jobs for one record row")
        by_origin.setdefault(o, {})[ol_] = cf
        job_of.setdefault(ol_, lb)
        assert job_of[ol_] == lb, (ol_, "one record label from two job labels")
    fin = json.loads(blobs["final"])["result"]
    t196 = json.loads(blobs["t196"])
    env_t = {c["key"] for c in t196["cells"]}
    coh455 = sorted(fin["cohort"]["cohort_cells"])
    jl_ = set(job_of)  # the record's labels
    UNC = "|uncut" if FULLREC else ""
    fold_lab_ = []
    if mode == "union_sol":  # Sol (a) and c_asis on the shared pool (09's ruling (b)): the runner's own jobs give the rows
        labels = sorted(jl_)
        origins = ["union", "union without S4"]
    else:
        labels = [f"{f}{UNC} {ro}" for f in PUBLIC for ro in ("C", "R") if f"{f}{UNC} {ro}" in jl_]  # the runner's own jobs say which public rows must exist
        labels += sorted(lb for lb in jl_ if lb.rsplit(" ", 1)[0] == "own_all_LOO_L48")  # LOO-3 rows: one held-out model per origin, raw scores
        fold_lab_ = [f"{fa} {ro}" for fa in FL_ for ro in ("C", "R")]  # G7b FOLD-MODEL: the pooled build's own jobs must carry every requested label
        assert all(lb in jl_ for lb in fold_lab_), ("--fold-labels: labels without captured jobs", [lb for lb in fold_lab_ if lb not in jl_])
        labels += fold_lab_
        assert mode != "alltasks" or "ownallLOO" not in nm_ or FULLREC or any(lb.startswith("own_all_LOO_L48") for lb in labels), "LOO addendum without LOO jobs"
        origins = ORIGINS if mode == "alltasks" else SWEEP_ORIGINS if mode == "sweep" else ["S_old", "S_new"]
        origins = origins + [o for o in ("S_old (registered cohort)", "S_new (registered cohort)", "S_newmains (registered cohort)") if o in by_origin]
        SPO_O = ["S_paper493", "S_paper493 [P_new]", "S_paper493 [P_rep]", "S_paper493 (registered cohort)"]  # the second refresh's own S_paper493 line
        AO_O = [f"{o} (all-origins draws)" for o in ("S_old", "S_new", "S_newmains", "S_newmains without S4", "S_paper493")]
        AO_O += [f"{o} (registered cohort) (all-origins draws)" for o in ("S_old", "S_new", "S_newmains", "S_paper493")]  # G4 (r28 item 11)
        origins = origins + [o for o in SPO_O + AO_O if o in by_origin]
    bl_ = by_origin  # the shared per-origin lookups below (FINAL's pool reference, membership, S_newmains arms, 25f roots, S_paper493's block)
    if PWm:  # G7: the pooled build runs only the suffixed union rows, every one of them a job; the shared lookups read the lead combo's union rows
        cbr_ = {cb: [f"{o} [pooled: {cb}]" for o in getattr(RP, "PW_MEMBERS", RP.COMBOS_ALLO)[cb]] for cb in PWm["combos"]}  # the runner's members (09's
        origins = [o for cb in PWm["combos"] for o in cbr_[cb] if o in by_origin]  # rulings may change them, e.g. ruling (A) on the registered cohort)
        assert sorted(origins) == sorted(by_origin), ("pooled world: job rows outside the pooled combos' union rows", sorted(set(by_origin) - set(origins))[:4])
        assert "pooled_all_origins" in PWm["combos"], "pooled world without the lead combo"
        bl_ = by_origin | {o: by_origin[f"{o} (all-origins draws) [pooled: pooled_all_origins]"] for o in ("S_new", "S_newmains", "S_paper493")}
    assert labels, "no public-probe job captured"
    missing_ = [] if DRY or CAT else [(lb, o) for lb in labels for o in origins if o not in rec["rows"].get(lb, {}) and lb in by_origin.get(o, {})]
    assert not missing_, f"record lacks public-probe rows the runner produced: {missing_[:5]}"
    cfg0 = bl_["union" if mode == "union_sol" else "S_new"][labels[0]]
    ref = fin["calibration_records"]["primary"][cfg0["pool_reference_key"]]
    sch = ref["schemes"][ref["primary"]]
    nmo_ = "union" if mode == "union_sol" else "S_newmains"
    mem = (bl_[nmo_][labels[0]]["membership"] if mode in ("alltasks", "sweep", "union_sol") else
           {"status_rows": {}, "batch_of_cell": {}, "folder_keys": {}, "main_of": {}})
    broken = {c for c, v in mem["status_rows"].items() if v.startswith("setup failure")}
    s4c = sorted(c for c, b_ in mem["batch_of_cell"].items() if b_ in ("S4", "S4r"))
    if NF:  # membership710's batch map covers its old cohort only; the no-filter record's map covers every launched nm360 cell (Codex r9)
        bnf_ = NF["S_newmains"]["batch_of_cell"]
        assert all(bnf_.get(c) == b_ for c, b_ in mem["batch_of_cell"].items() if b_ is not None), "batch maps disagree"
        s4c = sorted(c for c, b_ in bnf_.items() if b_ in ("S4", "S4r"))
    owner = {k: f for f, ks in mem["folder_keys"].items() for k in ks}
    moa_ = dict(cfg0.get("main_of_all", {}))
    assert all(moa_.get(c, m) == m for c, m in fin["maps"]["main_of"].items()), "FINAL's main_of and main_of_all disagree"
    main_new = moa_ | dict(fin["maps"]["main_of"])
    main_nm = main_new | dict(mem["main_of"])
    lv_ = cfg0.get("F_list") or list(RP.F)
    main_t = {c["key"]: c["main"] for c in t196["cells"]}
    u455 = set(main_t.values()) | set(fin["design"]["primary_universe_mains"])
    old_coh_ = [c for c in coh455 if c in env_t]
    assert len({main_t[c] for c in old_coh_}) == len(old_coh_), "two S_old cohort cells share a main (S_old's per-cell grouping would differ from canonical mains)"
    assert {main_new[c] for c in coh455 if c not in env_t} <= u455, "an S_new cohort main outside the draw universe (catch grouping would alias)"
    inp = {"levels": sorted(float(x) for x in lv_), "labels": labels, "pool_old": sorted(sch["pool_keys_old"]), "main_t": main_t,
           "owner": owner, "s4_orphans": [list(x) for x in (bl_[nmo_][labels[0]].get("s4_orphans", []) if mode in ("alltasks", "sweep", "union_sol") else [])],
           "nm_simh_roots": sorted({x_["root"] for v_ in (cfg0.get("simh_nm") or {}).values() for x_ in v_ if isinstance(x_, dict) and "root" in x_}),  # G2 (r28 item 10)
           "s4_cells": s4c, "by_label": {}, "variants_by_label": {}, "mode": mode, "origins": origins, "cap": cfg0.get("cap")}
    assert all(by_origin[o][lb].get("cap") == inp["cap"] for o in origins for lb in labels if lb in by_origin.get(o, {})), "cap differs between jobs"
    assert mode == "alltasks" or inp["cap"] is not None or not Path(rpath).name.startswith("runlen"), "a run-length record without a cap in its config"
    listing = []
    reads_ = set()  # (volume, path, sha256) as check() reads each file (G1: the full-length allowlist is by volume too)
    inp["five"] = mode == "union_sol"
    if PWm:  # G7b FOLD-MODEL: the S_paper493 world alias set written here is the code commit's (Claude N5: mandatory on a pooled check)
        assert set(getattr(RP, "SP_WORLD_ALIAS", None) or ()) == SP_WORLD_ALIAS, "the runner's SP_WORLD_ALIAS is missing or differs from the one written here"
    if FL_:  # G7b FOLD-MODEL: the code commit's rule lists equal the ones written here (a changed rule STOPs); one union-stream main list
        assert (tuple(RP.FO.NO_FOLD_DROPS) == NO_FOLD_DROPS and tuple(RP.OWN_ALL) + tuple(RP.OWN_ALL_LOO) + tuple(RP.OWN_ALL_RS) == OWN_ALL_FAMS
                and tuple(RP.OWN_ALL_RS) == OWN_ALL_RS_FAMS and set(RP.OOF_FALLBACK_LABELS) == {f"{fa} {ro}" for fa in OWN_ALL_RS_FAMS for ro in "CR"}
                and tuple(RP.E12_FILE_FAMILIES) == E12_FILE_FAMILIES and tuple(RP.OWN_ALL_RS_FOLDS) == RS_FOLDS), "a fold-model rule list differs from the runner's"
        mm_ = {json.dumps(sorted(cf_["merge"]["mains"])) for _, cf_, _ in jobs if cf_.get("merge")}
        assert len(mm_) == 1, "the all-origins jobs carry different union-stream main lists"
        inp["merge_mains"] = json.loads(mm_.pop())  # F1COLS: the attack fold -1 columns index the sorted union-stream mains (draws_main)
        inp["no_fold_drops"] = list(NO_FOLD_DROPS)
        rsfb_ = git("show", f"{RS_FOLDS[0]}:{RS_FOLDS[1]}")  # own_all_rs's committed fold list, read here (the runner's l.4713-4721 pin)
        assert sha(rsfb_) == RS_FOLDS[2], "own_all_rs fold list sha256"
        rsfl_ = json.loads(rsfb_)
        assert (rsfl_["mains"] == 901 == len(rsfl_["by_main"]) and rsfl_["asserts"]["mains_without_fold"] == 0
                and rsfl_["asserts"]["mains_not_in_exactly_one_fold"] == 0 and all(0 <= int(v_[0]) <= 4 for v_ in rsfl_["by_main"].values())), \
            "own_all_rs fold list shape or asserts (the runner's l.4718-4720)"
        RS_FOM = {m_: int(v_[0]) for m_, v_ in rsfl_["by_main"].items()}
        fitchk_ = {}
        fbx_ = (tree / D507 / m_["full_length"]["inputs_full"]["path"]).read_bytes()
        assert sha(fbx_) == m_["full_length"]["inputs_full"]["sha256"], "INPUTS_FULL sha256"
        for fa_, (fc_, fp_, fh_) in ((f_, FOLD_FITS[f_]) for f_ in FL_):  # the named fit record = INPUTS_FULL fits[fam].fit, its git bytes hashed
            fit_ = ((json.loads(fbx_).get("fits") or {}).get(fa_) or {}).get("fit") or {}
            assert (fit_.get("commit"), fit_.get("path"), fit_.get("sha256")) == (fc_, fp_, fh_), (fa_, "INPUTS_FULL's fit record is not the named one")
            assert sha(git("show", f"{fc_}:{fp_}")) == fh_, (fa_, "the fit record's bytes differ from its sha256")
            fitchk_[fa_] = {"commit": fc_, "path": fp_, "sha256": fh_, "inputs_full_fits_equal": True, "git_bytes_sha256_equal": True}
        inp["fold_fit_check"] = fitchk_
        inp["fold_not_covered"] = VALUE_FOLD_NOT_COVERED if VAL_ else FOLD_NOT_COVERED  # Astra r5 #6: a value run never says "catalogue stage only"
        na_ = {"allunits_dev": sorted(f"{cf_.get('full_as', lb_)} | {cf_.get('pw_combo')}" for lb_, cf_, _ in jobs if cf_.get("allunits_dev")),
               "no_s4": sorted(f"{cf_.get('full_as', lb_)} | {cf_.get('pw_combo')}" for lb_, cf_, _ in jobs if cf_.get("no_s4"))}
        assert not na_["allunits_dev"] and not na_["no_s4"], ("a pooled row with allunits_dev or without S4: outside what this check was reviewed for "
                                                              "(Astra r1 #1-#2 ported, never exercised; 507-prep / 2e: STOP)", na_)
        inp["fold_not_applicable"] = {"asserted_absent_on_every_pooled_job": ["allunits_dev", "no_s4 (without-S4 member rows)"], "jobs_checked": len(jobs),
                                      "note": "both runner branches are ported (mapped-unit veto, without-S4 f1cols) but cannot fire on these combos"}
    if PWm:  # G7: the pooled combos, the stage, and each label x combo's pooled pools file as the record names it (meta.pooled_world.manifest_per_label)
        inp["pw"] = {"combos": list(PWm["combos"]), "stage": PWm["stage"], "files": dict(PWm.get("manifest_per_label") or {}), "manifest": PWm["manifest"],
                     "structural_cells": sorted(cfg0.get("structural_cells") or []),  # G7e: INPUTS_FINAL's structural no-row cells (the runner's cfg)
                     "tree_sha": {f"{D507}/{f_.name}": sha(f_.read_bytes()) for f_ in sorted((tree / D507).glob("*.py"))},  # the code commit's module files
                     "required_modules": [f"{D507}/{n_}" for n_ in getattr(RP, "PW_CODE_MODULES", ("run_peraction311.py",))],  # the modules it pins by name
                     "combo_rows": cbr_,  # each pooled combo's member rows, the runner's COMBOS_ALLO at the code commit
                     "captured_modules": cap_mods_}  # the runner's own module pins, as this capture computed them (compared exactly with the manifest's)
        assert isinstance(cap_mods_, dict) and cap_mods_, "the captured pooled spec carries no module pins"
        dcl_ = getattr(RP, "PW_GATE2_DECLARED", None)  # 09 (a') at the code commit; 2e 20:1x: each declared group cites the token-prefix record
        inp["pw"] |= {"gate2_declared": sorted(sorted(g_) for g_ in (dcl_ or ())), "auth_rule": getattr(RP, "PW_AUTH_OVERLAP", None)}
        if hasattr(RP, "pw_auth_overlap_check") and not isinstance(inp["pw"]["auth_rule"], dict):  # CUT (1): no silent skip of the overlap rule
            raise SystemExit("STOP: the runner defines pw_auth_overlap_check but no PW_AUTH_OVERLAP was captured")
        if dcl_:
            tpb_ = subprocess.run(["git", "show", f"{TOKEN_PREFIX_REC[0]}:{TOKEN_PREFIX_REC[1]}"], capture_output=True, check=True).stdout
            assert sha(tpb_) == TOKEN_PREFIX_REC[2], "the token-prefix record differs from its pinned sha256"
            tpn_ = lambda p_: re.sub(r"__L\d+(?=\.npz$)", "", p_.rsplit("/", 1)[-1])  # noqa: E731
            inp["pw"]["token_prefix"] = {"record": f"{TOKEN_PREFIX_REC[1]} @ {TOKEN_PREFIX_REC[0]}", "sha256": TOKEN_PREFIX_REC[2], "pairs": [
                {"pair": sorted(p_["pair"]), "verdict": p_["verdict"],
                 "files": [[scorer_of(f_["file_x"]), tpn_(f_["file_x"]), scorer_of(f_["file_y"]), tpn_(f_["file_y"]), f_["outcome"]]
                           for f_ in (p_["lines"].get("uncut") or {}).get("files", [])]} for p_ in json.loads(tpb_)["pairs"]]}
        assert all(sorted(cf_.get("structural_cells") or []) == inp["pw"]["structural_cells"] for _, cf_, _ in jobs), "structural cells differ between jobs"
        if CAT:  # G7b: check() reads the catalogue's manifest on the Volume (its pooled files, sizes, patterns) and checks its code commit and, Codex
            inp["pw"] |= {"git": code, "pool_inputs": pw_pool_inputs_chk(m_ | {"script_sha256": sha((tree / D507 / "run_peraction311.py").read_bytes())})}  # r40 #8,
        else:  # its pool-defining inputs (the companion's pins at the pooled runner). Astra g7 #1: a scored record's own; the overlap gate's committed file
            inp["pw"]["pool_inputs"] = pw_pool_inputs_chk(m_)
            ao_ = PWm.get("authorized_overlaps")
            if ao_:
                ab_ = (tree / D507 / ao_["path"]).read_bytes()
                assert sha(ab_) == ao_["sha256"], "authorized-overlaps file sha256"
                inp["pw"]["auth"] = {e_["key"]: {"roles": sorted(e_["roles"])} for e_ in json.loads(ab_)["authorized_overlaps"]}
    inp["zero_runs"] = sorted(set(ZR)) if NF else []
    inp["structural_classes"] = [x.strip() for x in structural.split(",") if x.strip()]
    inp["nonstructural_classes"] = [x.strip() for x in nonstructural.split(",") if x.strip()]
    # ---- second refresh (09 RULE @ 07ad588be; 2e 02:4x): a non-TEST record waits for every in-flight cell (STOP otherwise); the exclusions file
    # (make_exclusions.py, canonical classes) gives launched = final + excluded + pending by key per set, and its classes decide the audit
    r2_ = json.loads(ib_).get("refresh2") or {}
    if FULLREC:  # G1 (r28 item 5): a full-length record's completion pins are INPUTS_FULL's own (the exclusions may be a class-only revision)
        fb_ = (tree / D507 / m_["full_length"]["inputs_full"]["path"]).read_bytes()
        assert sha(fb_) == m_["full_length"]["inputs_full"]["sha256"], "INPUTS_FULL sha256"
        FIN_ = json.loads(fb_)
        r2_ = {"waitlist_pending": FIN_["waitlist_pending"], "exclusions": FIN_["exclusions"]}
    TESTREC = bool(m_.get("test")) or "_TEST" in nm_
    completion_gate(m_, r2_, TESTREC)
    ex_ = m_.get("exclusions") or (r2_.get("exclusions") if DRY else None)
    EX, wl_ = None, {}
    if ex_:
        eb_ = git("show", f"{ex_['commit']}:{ex_['path'] if ex_['path'].startswith('experiments/') else D507 + '/' + ex_['path']}")
        assert sha(eb_) == ex_["sha256"], "exclusions file sha256"
        EX = json.loads(eb_)
        assert NF and EX["sources"]["nofilter"]["sha256"] == nofilter.rsplit(":", 1)[1], "the exclusions file was built from another no-filter list (pass --nofilter)"
        spc_ = next(iter(bl_["S_paper493"].values()))["sp493"]
        univ_ = {"S_old": set(env_t), "S_new": set(moa_), "S_newmains": set(mem["main_of"]), "S_paper493": set(spc_["main_of_cell"])}
        exc_by, wl_ = exclusion_identity(EX, univ_, {s_: (NF[s_]["headline_cells"] if s_ != "S_paper493" else spc_["cohort"] or []) for s_ in univ_},
                                         {s_: NF[s_]["measurement_excluded"] for s_ in ("S_old", "S_new", "S_newmains")}, TESTREC)
        relab_ = apply_rulings(exc_by, RULED)  # after the file's own per-class counts are checked, before classification
    if NF:  # each exclusion class decided once, here (listed below); the file's own counts must add up by key
        inp["nf_struct"] = classify(sorted({cl_ for d_ in (exc_by.values() if EX else [d_["measurement_excluded_by_class"] for d_ in NF.values()]) for cl_ in
                                            (d_.values() if EX else d_)}), inp["structural_classes"], inp["nonstructural_classes"] if EX else [])
        # (no exclusions file: an older record's local wording, structural prefixes only, unmatched = not structural, as before; Codex r25)
        for o_, d_ in NF.items():
            bc_ = d_["measurement_excluded_by_class"]
            n_ = len(set(d_["measurement_excluded"]))  # one class per cell; classes cover the excluded cells; the stated count agrees
            assert sum(len(cs_) for cs_ in bc_.values()) == len(set().union(*[set(cs_) for cs_ in bc_.values()])) == n_ == d_.get("measurement_excluded_n", n_), \
                (o_, "the exclusion file's per-class lists do not add up by key")
            bad_ = sorted(c_ for cl_, cs_ in bc_.items() for c_ in cs_ if d_["measurement_excluded"][c_]["class"] != cl_)  # each cell's own record agrees (Codex r23)
            assert not bad_, (o_, "cells whose recorded class differs from their per-class list", bad_[:5])
    if NF:  # provenance (0d's recheck, 2e 02:0x): the consumed no-filter file and each origin's headline cohort, by key and by sha256
        inp["nofilter_source"] = {"file": nofilter.rsplit(":", 1)[0], "sha256": nofilter.rsplit(":", 1)[1],
                                  "headline_sha256": {o_: sha(json.dumps(sorted(d_["headline_cells"])).encode()) for o_, d_ in NF.items()},
                                  "headline_cells": {o_: sorted(d_["headline_cells"]) for o_, d_ in NF.items()},
                                  "excluded_classes": sorted({cl_ for d_ in NF.values() for cl_ in d_["measurement_excluded_by_class"]}),
                                  "structural_class_prefixes": inp["structural_classes"], "nonstructural_class_prefixes": inp["nonstructural_classes"],
                                  "unclassified_classes_are": ("a stop (canonical classes; both lists)" if EX and inp["nonstructural_classes"] else
                                                               "not structural (older record's local wording: structural prefixes only)"),
                                  "excluded_by_class": {o_: {cl_: {"n": len(cs_), "cells_sha256": sha(json.dumps(sorted(cs_)).encode()),
                                                                   "structural": "decided on the exclusions file's canonical class" if EX else inp["nf_struct"][cl_]}
                                                             for cl_, cs_ in sorted(d_["measurement_excluded_by_class"].items())} for o_, d_ in NF.items()}}
        if EX:  # the canonical classes by key, per set; in-flight cells (TEST only) apart
            byc_ = {s_: {} for s_ in exc_by}
            for s_, d_ in exc_by.items():
                for c_, cl_ in d_.items():
                    byc_[s_].setdefault(cl_, []).append(c_)
            inp["nofilter_source"]["exclusions"] = {"file": f"{ex_['commit']}:{ex_['path']}", "sha256": ex_["sha256"], "per_set": EX["per_set"],
                                                    "excluded_by_class": {s_: {cl_: {"n": len(cs_), "cells_sha256": sha(json.dumps(sorted(cs_)).encode()),
                                                                                     "structural": inp["nf_struct"][cl_]} for cl_, cs_ in sorted(d_.items())}
                                                                          for s_, d_ in byc_.items()},
                                                    "waitlisted (in flight)": wl_, "test_record": TESTREC, "waitlist_pending": m_.get("waitlist_pending"),
                                                    "relabelled_by_ruling": relab_}
    if NF:  # audit: launched universe by key, and every launched cell outside the headline either lacks a scored arm here or has a structural class
        univ_ = {"S_old": set(env_t), "S_new": set(moa_), "S_newmains": set(mem["main_of"])}
        inp["nf_audit"] = {}
        for o_, u_ in univ_.items():
            lau_ = set(NF[o_]["headline_cells"]) | set(NF[o_]["measurement_excluded"])
            assert lau_ == u_, (o_, "no-filter launched cells != the launched universe", len(lau_), len(u_))
            cls_ = {c_: cl_ for cl_, cs_ in NF[o_]["measurement_excluded_by_class"].items() for c_ in cs_}
            assert set(cls_) == set(NF[o_]["measurement_excluded"]), (o_, "an excluded cell without a class")
            inp["nf_audit"][o_] = dict(exc_by[o_]) if EX else cls_  # the canonical class (09 RULE §7) where the exclusions file is pinned
    cfg_reg_ = {}  # G7b FOLD-MODEL (2e 20:3x): whether the probe config and S_new arm files the base checker took from cfg0 are this label's own
    for lab in labels:
        cfgL_ = bl_["union" if mode == "union_sol" else "S_new"][lab]  # the label's OWN job cfg (a retrain job carries its own substituted blocks:
        p = next(q for q in cfgL_["probes"] if q["label"] == job_of[lab])  # full_override.full_jobs l.319; cfg0's "Sol (a) C" is the registered file)
        p0_ = next((q for q in cfg0["probes"] if q["label"] == job_of[lab]), None)
        cfg_reg_[lab] = {"probe_config_equal": p0_ == p, "s_new_arm_files_equal": p0_ is not None and [a["files"].get(p["new"]) for a in cfg0["arms"]]
                         == [a["files"].get(p["new"]) for a in cfgL_["arms"]], "fold_label": lab in fold_lab_}
        loo_ = p.get("family") == "own_all_loo"
        FLD_ = lab in fold_lab_  # a --fold-labels label: five models (Sol: no family; own_all_rs: rows_rs), run under its counterpart's job label
        assert (p["folds"] == 5) == (inp["five"] or FLD_) and (not loo_ or p["folds"] == 1), (lab, p["folds"], "five-model rows only in union-Sol records or --fold-labels")
        assert not FLD_ or (job_of[lab].rsplit(" ", 1)[0], p.get("family")) in (("Sol (a)", None), ("Sol (a)", "Sol (a)"), ("own_all_rs_L48", "own_all_rs"),
                                                                                ("c_asis", None), ("c_asis", "c_asis"), ("own_all_rs25_L48", "own_all_rs"),
                                                                                ("own_all_rs10_L48", "own_all_rs")), \
            (lab, job_of[lab], p.get("family"), "a fold label under another counterpart (the ported rules cover Sol (a), c_asis and own_all_rs*)")
        so = p["old"]
        if not any(aob(o_).split(" ")[0] == "S_old" for o_ in origins) and "S_old" not in origins:  # Codex r40 #3: pooled rows by base origin
            old = []
        elif so["kind"] == "own_all_loo_t196":  # T196's six per-world own_all_loo folders (p2 volume), every row scored by o0
            old = [{"vol": "p2", "path": e["path"], "sha256": e["sha256"]} for _, e in sorted(so["files"].items())]
        elif so["kind"] == "npz":
            old = [{"vol": so["volume"], "path": so["path"], "sha256": so["sha256"]}]
        elif so["kind"] == "uncut_list":  # G1 (r28 item 4): the full-length S_old real folders, one file each, in the runner's order (sorted by folder)
            old = [{"vol": e.get("vol", "conv"), "path": e["path"], "sha256": e["sha256"]} for _, e in sorted(so["files"].items())]
        elif so["kind"] == "own_all_rs_t196" and FLD_:  # G7b FOLD-MODEL: load_own_all (l.538): the S_old folders sorted by name, p2 unless pinned
            old = [{"vol": e.get("vol", "p2"), "path": e["path"], "sha256": e["sha256"]} for _, e in sorted(so["files"].items())]
        else:
            raise SystemExit(f"{lab}: S_old kind {so['kind']} not handled by this check")
        new = [a["files"][p["new"]] for a in cfgL_["arms"]]
        nm_files, use4r = [], None
        if mode in ("alltasks", "sweep", "union_sol"):
            cfgN = bl_[nmo_].get(lab) or bl_[f"{nmo_} without S4"][lab]  # LOO computes only the without-S4 S_newmains row
            nm_files = [{"arm": a["arm"], **a["files"][p["new"]]} for a in cfgN["arms_nm"] if p["new"] in a["files"]]
            use4r = set(mem["s4r_landing"]["arms"]) <= {e["arm"] for e in nm_files}
        ha25 = [hr["files"][p["new"]] | ({"keys": list(hr["keys"])} if PWm else {})  # G7b: the root's keys (the runner's l.1417 check, pooled rows)
                for hr in bl_["union" if mode == "union_sol" else "S_new"][lab].get("ha25f", {}).get("roots", []) if p["new"] in hr["files"]]
        inp["by_label"][lab] = {"old_files": old, "new_files": new, "nm_files": nm_files, "ha25f": ha25, "loo": loo_}
        if FLD_:  # G7b FOLD-MODEL: five models; the family's readers; the job-label rules (oof fallback, own_all H_A = H_sim cells)
            fam_j_ = job_of[lab].rsplit(" ", 1)[0]
            inp["by_label"][lab] |= {"five": True, "fold_label": True, "family": p.get("family"), "oof": fam_j_ in OWN_ALL_RS_FAMS,
                                     "own_all_family": fam_j_ in OWN_ALL_FAMS, "probe_file": p["new"], "readout": p["readout"]}
            if p.get("family") == "own_all_rs":  # l.1762-1765: each job's own cell -> main map for the fold check (an S_old job carries no membership),
                cms_ = {}  # shipped once per distinct map (by sha256) and named by each variant (rs_cmain)
                for o_ in origins:
                    cf_ = by_origin[o_].get(lab)
                    if cf_ is not None:
                        cm_ = (dict(fin["maps"]["main_of"]) | dict(main_t) | dict(cf_["main_of_all"]) | (dict(cf_["membership"]["main_of"]) if "membership" in cf_
                               else {}) | (dict(cf_["sp493"]["main_of_cell"]) if "sp493" in cf_ else {}))
                        cms_[sha(json.dumps(cm_, sort_keys=True).encode())] = cm_
                fom_ = {json.dumps(by_origin[o_][lab]["own_all_rs_fold_of_main"], sort_keys=True) for o_ in origins if lab in by_origin.get(o_, {})}
                assert fom_ == {json.dumps(RS_FOM, sort_keys=True)}, (lab, "the jobs' own_all_rs_fold_of_main differs from the committed fold list read here")
                inp["by_label"][lab]["rs_folds"] = {"cmains": cms_, "fold_of_main": RS_FOM}
            fit_ = FOLD_FITS[lab.rsplit(" ", 1)[0]]
            pwm_ = {json.dumps((by_origin[o_][lab].get("pw_model") or {}).get("fit_record"), sort_keys=True) for o_ in origins if lab in by_origin.get(o_, {})}
            assert pwm_ == {json.dumps({"commit": fit_[0], "path": fit_[1], "sha256": fit_[2]}, sort_keys=True)}, \
                (lab, "the jobs' pw_model fit record is not INPUTS_FULL's named fit (2e 20:3x)")
        listing += old + new + nm_files + ha25
        reads_ |= {(e["vol"], e["path"], e["sha256"]) for e in old} | {("conv", e["path"], e["sha256"]) for e in new + nm_files + ha25}
        vs = {}
        for o_full in origins:
            if lab not in by_origin.get(o_full, {}):  # the runner computes no row here: own_all_LOO's with-S4 S_newmains (its record row must say so);
                assert lab.startswith("own_all_LOO_L48"), (lab, o_full, "a variant without a job")  # LOO has no S_paper493 or all-origins rows at all
                if o_full == "S_newmains":
                    vs[o_full] = {"unavailable": True}
                else:
                    assert o_full.startswith("S_paper493") or o_full.endswith("(all-origins draws)"), (lab, o_full)
                    vs[o_full] = {"absent": True}
                continue
            cf = by_origin[o_full][lab]
            o = aob(o_full).replace(" (all-origins draws)", "").replace(" (registered cohort)", "")  # either tag, in either order (G4); G7: no pooled suffix
            NFo = None if "(registered cohort)" in o_full else NF
            if NFo and cf.get("cohort") == "nofilter":  # the record's own list must be the audited one
                assert all(list(cf["cohort_list"][x]) == sorted(NF[x]["headline_cells"]) for x in ("S_old", "S_new", "S_newmains")), "cohort list differs"
                assert cf["cohort_list"]["batch_nm"] == NF["S_newmains"]["batch_of_cell"], "the record's S_newmains batches differ from the audited file's (Codex r14)"
            assert not (cf.get("cohort") == "nofilter" and not NF), "a no-filter record: pass --nofilter with its cohort file"
            coh_old = sorted(NFo["S_old"]["headline_cells"]) if NFo else [c for c in coh455 if c in env_t]
            coh_new = sorted(NFo["S_new"]["headline_cells"]) if NFo else [c for c in coh455 if c not in env_t]
            if o.startswith("S_paper493"):  # 09 @ e3a459bd0: S_paper493 is its own origin (Append 39 §§3-4), every pin from the v2 INPUTS_FINAL block
                fk_ = RP.fam_kind((job_of[lab] if FLD_ else lab).rsplit(" ", 1)[0])  # the runner's other S_paper493 gates (DT reader, Append 39 §5) are not
                assert p.get("kind") != "dt" and (fk_ in ("sol", "own_all") if FLD_ else (fk_ not in ("dt", "famexp", "own_a", "sol") and p["folds"] != 5)), \
                    (lab, fk_, "an S_paper493 gate this check does not model")  # modelled here; G7b FOLD-MODEL: the r36 fold and mapped units are, for the fold labels
                sp_ = cf["sp493"]
                blk_ = cf.get("sp_block")
                pex_ = sorted(sp_["pool_exclusions"])
                csrc_ = sp_["cohort_registered"] if "(registered cohort)" in o_full else sp_["cohort"]
                spm_ = set(sp_["env_of_main"])
                c2m_ = {c.split("_", 1)[0]: m for c, m in cfg0["main_of_all"].items()}  # S_new's exclusions and replacements: only on S_paper493's mains
                simh = list(cf["simh_old"][job_of[lab]]) + list(cf["simh_files"][p["new"]]) + list(sp_["simh"].get(p["new"], []))
                folders_ = [{"arm": a_["arm"], "keys": a_["keys"], "refused_keys": a_.get("refused_keys", []), "vol": a_.get("vol", "conv"), "world": a_["world"],
                             **a_["files"][p["new"]]} for a_ in sp_["arms"] if p["new"] in a_["files"]]  # G7b: the routing world (sp_world, l.1446)
                rr_ = {c0: v["root"].removesuffix("__sim_honest") for d in cf.get("replaced", {}).values() for c0, v in d.items() if c2m_.get(c0) in spm_}
                vs[o_full] = {"sp": True, "simh": simh, "sp_roots": sorted({e["root"] for e in sp_["simh"].get(p["new"], [])}),
                              "cohort": None if csrc_ is None else sorted(c for c in csrc_ if blk_ is None or sp_["kind_of_cell"][c] == blk_),
                              "adm_sp": sorted(set(sp_["pools"]["H_sim_new"]) - set(pex_)), "pool_b": sorted(set(sp_["pools"]["H_B"]) - set(pex_)), "pex": pex_,
                              "spm": sorted(spm_), "main_of": dict(sp_["main_of_cell"]), "folders": folders_,
                              "pending": sp_["families_pending"].get(p["new"]) or (None if len(folders_) == len(sp_["arms"]) else "a folder without this probe's file"),
                              "excl_prefixes": sorted({c0 for d in cf.get("excl", {}).values() for c0 in d if c2m_.get(c0) in spm_}),
                              "replaced_root": rr_, "zero_runs": sorted(sp_["structural_zero_keys"].get(p["readout"], [])),
                              "simh_zero_unit": sorted(sp_["simh_zero_unit_keys"].get(p["readout"], []))}
                if FLD_ and job_of[lab].startswith(("Sol (a)", "c_asis")):  # G7b FOLD-MODEL (l.1820-1832): S_paper493 cells' held-out r36 folds
                    vs[o_full]["r36"] = dict(sp_["r36_fold_of_cell"])
                inp["sp_moa"] = dict(main_t) | dict(cfg0["main_of_all"]) | dict(sp_["main_of_cell"])  # T196, +311 and S_paper493 cells -> mains (keys disjoint)
                listing += folders_
                reads_ |= {(e["vol"], e["path"], e["sha256"]) for e in folders_}
            elif o == "S_old":
                simh = cf["simh_old"][job_of[lab]]  # keyed by the job label (full_override keeps "FIBS C")
                vs[o_full] = {"simh": simh, "cohort": coh_old, "excl_prefixes": [], "replaced_root": {}}
                hbr_ = cf.get("honest_b_relaunch")  # G3 (r28 item 9): the R3 alternate pool, on the plain S_old row only (not all-origins, not registered)
                if hbr_ and o_full == "S_old" and not cf.get("merge") and cf.get("cohort") != "registered" and p.get("relaunch_uncut"):
                    r3f_ = [{"vol": e["vol"], "path": e["path"], "sha256": e["sha256"]} for e in cf["relaunch_files"][p["relaunch_uncut"]]]
                    vs[o_full]["r3"] = {"files": r3f_, "members": sorted(hbr_["members"]), "staged_outside": sorted(hbr_["staged_outside"])}
                    listing += r3f_
                    reads_ |= {(e["vol"], e["path"], e["sha256"]) for e in r3f_}
            elif o == "S_new":
                simh = cf["simh_files"][p["new"]]
                vs[o_full] = {"simh": simh, "cohort": coh_new, "pool_b": sorted(sch["pool_keys_new"]), "main_of": main_new,
                         "excl_prefixes": sorted({c0 for d in cf.get("excl", {}).values() for c0 in d}),
                         "replaced_root": {c0: v["root"].removesuffix("__sim_honest") for d in cf.get("replaced", {}).values() for c0, v in d.items()},
                         "expected_replacement_roots": exp_rep["S_new"]}
                assert vs[o_full]["replaced_root"] or not exp_rep["S_new"], "INPUTS_FINAL admits S_new replacement roots but the runner's S_new config has none"
            else:
                un_ = o.startswith("union")
                simh = (list(cf["simh_files"][p["new"]]) if un_ else []) + list(cf["simh_nm"][p["new"]])
                assert not NFo or use4r, (lab, "no-filter cohort with the coverage pool: not covered")
                coh_nm = sorted(c for c in (NFo["S_newmains"]["headline_cells"] if NFo else list(mem["cohort"]) + (list(mem["cohort_4r_sol_only"]) if use4r else []))
                                if c not in broken and not (cf.get("no_s4") and c in set(s4c)))
                pool_nm = set(mem["pool_honest_b"]) | (set(mem["pool_honest_b_4r_sol_only"]) if use4r else set())
                assert not (un_ and pool_nm & set(sch["pool_keys_new"])), "a key in both S_new's and S_newmains' H_B"
                rr_ = {c0: v["root"].removesuffix("__sim_honest") for d in cf.get("replaced_nm", {}).values() for c0, v in d.items()}
                if un_:
                    rr_ |= {c0: v["root"].removesuffix("__sim_honest") for d in cf.get("replaced", {}).values() for c0, v in d.items()}
                vs[o_full] = {"simh": simh, "cohort": (coh_new if un_ else []) + coh_nm, "use4r": use4r, "main_of": main_nm,
                         "pool_b": sorted(pool_nm | (set(sch["pool_keys_new"]) if un_ else set())), "nm_cohort_n": len(coh_nm), "nm_pool_n": len(pool_nm),
                         "excl_prefixes": sorted({c0 for d in cf.get("excl_nm", {}).values() for c0 in d} |
                                                 ({c0 for d in cf.get("excl", {}).values() for c0 in d} if un_ else set())),
                         "replaced_root": rr_,
                         "expected_replacement_roots": sorted(set(exp_rep["S_newmains"]) | (set(exp_rep["S_new"]) if un_ else set()))}
                if FLD_:  # G7b FOLD-MODEL (Astra r1 #2): the with-S4 cohort (cohN_full, l.1617-1626), for a without-S4 row's attack fold -1 columns
                    vs[o_full]["cohort_full"] = (coh_new if un_ else []) + sorted(c for c in (NFo["S_newmains"]["headline_cells"] if NFo else list(mem["cohort"])
                                                                                            + (list(mem["cohort_4r_sol_only"]) if use4r else [])) if c not in broken)
            listing += simh
            reads_ |= {("conv", e["path"], e["sha256"]) for e in simh}
            vs[o_full] |= {"use21b_listed": job_of[lab] in cf.get("use21b", []), "per_pool_21b_listed": job_of[lab] in cf.get("per_pool_21b", []),
                      "fxp": job_of[lab].rsplit(" ", 1)[0] in cf.get("fold_ext_passed", {})}
            if FLD_:  # G7b FOLD-MODEL (Claude N4): the S4-orphan-guarded sim-honest roots of THIS job (the runner's nmr_, l.1712), not cfg0's
                vs[o_full]["nm_simh_roots"] = sorted({x_["root"] for v_ in (cf.get("simh_nm") or {}).values() for x_ in v_ if isinstance(x_, dict) and "root" in x_})
            NMo_ = o.startswith(("S_newmains", "union"))  # G7b FOLD-MODEL: l.1338 (Sol / c_asis on S_newmains need the held-out gate) and l.1466-1468
            if FLD_ and p.get("family") == "own_all_rs":  # G7b FOLD-MODEL: this job's cell -> main map (rs_folds.cmains, by sha256)
                vs[o_full]["rs_cmain"] = sha(json.dumps(dict(fin["maps"]["main_of"]) | dict(main_t) | dict(cf["main_of_all"]) | (
                    dict(cf["membership"]["main_of"]) if "membership" in cf else {}) | (dict(cf["sp493"]["main_of_cell"]) if "sp493" in cf else {}),
                    sort_keys=True).encode())
            vs[o_full] |= {"nm_sol_withheld": bool(NMo_ and job_of[lab].startswith(("Sol (a)", "c_asis")) and not cf.get("allunits_dev")
                                                   and not (o.startswith("union") and cf.get("union_sol_b")) and not vs[o_full]["fxp"]),
                           "e12_reader_missing": bool(cf.get("e12")) and (o.startswith("union") or p.get("kind") == "dt"  # (e12 reader per family;
                                                                          or p["new"].split("/", 1)[0] not in E12_FILE_FAMILIES),  # l.1464: union, DT)
                           "allunits_dev": bool(cf.get("allunits_dev")), "no_s4": bool(cf.get("no_s4"))}  # Astra r1 #1 / #2
            E12c = cf.get("e12")  # G2 (r28 items 6-7): the runner's e12 view of this job: this origin's epoch-2 folders and pool maps, S_old's fill-in roots
            org12 = "S_old" if o == "S_old" else "S_new" if o == "S_new" else "S_newmains" if o.startswith("S_newmains") else None
            if E12c and org12 in E12c:  # not S_paper493 (no epoch 2) and not the union (withheld under e12)
                P12 = E12c[org12]
                miss12 = [a_["arm"] for a_ in P12["folders"] if p["new"] not in a_["files"]] + (
                    [f_["root"] for f_ in E12c["fill_S_old"] if p["new"] not in f_["files"]] if org12 == "S_old" else [])
                assert not miss12, (lab, o_full, "e12: epoch-2 / fill-in files missing (the runner withholds or STOPs this row)", miss12)
                b12_ = "H_B" if (org12 != "S_newmains" or use4r) else "H_B_coverage"  # the runner's epoch-1 member check, by sha256 of the sorted keys
                e1_ = set(inp["pool_old"]) if org12 == "S_old" else set(vs[o_full]["pool_b"])
                assert sha(json.dumps(sorted(e1_)).encode()) == P12["e1_sha"][b12_], (lab, o_full, "e12: epoch-1 H_B members differ from the pool-key artifact's")
                if org12 == "S_old":  # and the B-alone pool's (Codex r35)
                    assert sha(json.dumps(sorted(k_ for k_ in inp["pool_old"] if k_.startswith("honest_b|"))).encode()) == P12["e1_sha"]["H_B2"], \
                        (lab, o_full, "e12: epoch-1 H_B2 members differ from the pool-key artifact's")
                fol12 = [{"arm": a_["arm"], "world": a_["world"], "world_e2": a_["world_e2"], "vol": a_["files"][p["new"]].get("vol", "conv"),
                          "path": a_["files"][p["new"]]["path"], "sha256": a_["files"][p["new"]]["sha256"]} for a_ in P12["folders"]]
                ha12 = P12.get("H_A", {})  # S_new: {family group: {e2 key: replicated member}}; S_newmains / S_old: {e2 key: replicated member}
                vs[o_full]["e12"] = {"origin": org12, "folders": fol12, "H_B": dict(P12.get("H_B", {})), "H_B2": dict(P12.get("H_B2", {})),
                                     "H_B_coverage": dict(P12.get("H_B_coverage", {})),
                                     "H_A": {g_: dict(v_) for g_, v_ in ha12.items()} if org12 == "S_new" else ({"_all": dict(ha12)} if ha12 else {}),
                                     "H_A_group": P12.get("H_A_group_of_family", {}).get(p["new"].split("/", 1)[0]) if org12 == "S_new" else None}
                listing += fol12
                reads_ |= {(e["vol"], e["path"], e["sha256"]) for e in fol12}
                if org12 == "S_old":  # tau_sim pool = primary_v3 u A25 fill-in u g4 (the fill-in roots first, as row() reads them)
                    fill12 = [{"batch": f_["batch"], "root": f_["root"], **f_["files"][p["new"]]} for f_ in E12c["fill_S_old"]]
                    vs[o_full]["simh"] = fill12 + list(vs[o_full]["simh"])
                    vs[o_full]["fill_admitted"] = {f_["root"]: sorted(f_["admitted"]) for f_ in E12c["fill_S_old"]}
                    listing += fill12
                    reads_ |= {("conv", e["path"], e["sha256"]) for e in fill12}
            if cf.get("merge"):  # the all-origins row: the same pools and thresholds as its origin's row; cells weighted 1/n_main over the combo's origins
                vs[o_full] |= {"ao_groups": list(cf["merge"]["groups"])} | ({"pw_combo": cf["pw_combo"], "pw_model": cf.get("pw_model")} if cf.get("pw_combo") else {})  # G7
                inp["canon"] = dict(cf["merge"]["main_of_cell"])
        inp["variants_by_label"][lab] = vs
    inp["cfg0_regression"] = cfg_reg_  # G7b FOLD-MODEL (2e 20:3x): the public labels' own probe config and arm files equal cfg0's, else STOP
    bad0_ = sorted(lb for lb, v_ in cfg_reg_.items() if not v_["fold_label"] and not (v_["probe_config_equal"] and v_["s_new_arm_files_equal"]))
    assert not bad0_, ("cfg0 regression FAILED: a public label's own probe config or S_new arm files differ from cfg0's", bad0_)
    if NF:  # every kept run's cell survives into each variant cohort of its origin, except an explicit variant exclusion (S4 cells without S4)
        for lab_, vs_ in inp["variants_by_label"].items():
            for o, v_ in vs_.items():
                if "(registered cohort)" in o or v_.get("unavailable"):  # appendix rows keep the registered cohort; unavailable rows have none
                    continue
                parts_ = ["S_new", "S_newmains"] if o.startswith("union") else ["S_newmains"] if o.startswith("S_newmains") else [o]
                for o_ in parts_:
                    for r_ in ZRO.get(o_, []):
                        c_ = r_.split("|", 1)[1]
                        assert c_ not in broken, (o_, c_, "a kept zero-tool run in a setup-failure cell")
                        assert c_ in set(v_["cohort"]) or (o.endswith("without S4") and c_ in set(s4c)), (lab_, o, c_, "a kept zero-tool cell dropped from the cohort")
    if VAL_:  # G7b VALUE (09 via 2e 23:2x): the seeded bootstrap-draw sample (ids from the stated seed; check() asserts them again) and the record's
        pzt_ = [(rec.get("npz") or {}).get(k_) for k_ in ("volume", "path", "sha256")]  # read from the record by key (2e (b), 10-20)
    if VAL_ and W2K_:  # rc1b W2: no per-draw npz on Stage D or kappa; kappa's seeded draws (G7b's sample, seed BOOT_SEED) compare against its row
        assert not rec.get("npz"), "rc1b W2: a per-draw npz on a W2 record (not expected; STOP)"  # store (W2b); Stage D compares draw 0 only (W2a)
        inp["boot"] = {"ids": [0] + boot_ids(BOOT_SEED), "seed": BOOT_SEED, "n": BOOT_N, "D": BOOT_D, "npz": None} if W2K_ == "kappa" else None
    elif VAL_:
        assert sha(recb) in PERDRAW_BY_RECORD and pzt_ == list(PERDRAW_BY_RECORD[sha(recb)]), \
            "the record's own per-draw npz is not the one pinned for this record (09 cond. 1: PERDRAW_BY_RECORD)"
        sd_ = BOOT_SEED_BY_RECORD.get(sha(recb), BOOT_SEED)  # PLANT V3: the PLANT run's own seed (AMEND4 D6); A / B / R37 unchanged
        inp["boot"] = {"ids": [0] + boot_ids(sd_), "seed": sd_, "n": BOOT_N, "D": BOOT_D, "npz": pzt_}  # companion npz pin, the record's own
        for lb_, o_, a_ in VPL_:  # plant presence: each plant's row is a scored member row of a checked label
            assert lb_ in labels and o_ in inp["variants_by_label"][lb_] and a_ in A3, ("a value plant outside the checked rows", lb_, o_, a_)
    # one variants dict per label is shipped; check() reads inp["variants"] per label
    inb = json.dumps(inp, sort_keys=True).encode()
    files = sorted({(e["path"], e["sha256"]) for e in listing})
    if FULLREC:  # G1 (r28 item 3): every file this check reads is in INPUTS_FULL's uncut allowlist; with a record, the captured jobs are the record's
        al_ = {tuple(x) for lb_, cf_, _ in jobs for x in (cf_.get("full_allowed") or [])}
        assert al_ and all(cf_.get("full_uncut") for _, cf_, _ in jobs), "a full-length job without its uncut allowlist"
        out_al_ = sorted(reads_ - al_)
        assert not out_al_, ("reads outside INPUTS_FULL's uncut allowlist (volume, path, sha256)", out_al_[:3])
    if FULLREC and not DRY:
        fl_ = m_["full_length"]
        if CAT:  # G7b: no pooled record yet; the catalogue's own manifest is checked on Modal (origins per label x combo, sizes, patterns, files)
            pass
        elif PWm:  # G7: the pooled build's job list is filtered after meta.full_length is written; meta.pooled_world states the filtered rows and labels
            assert len(jobs) == PWm["rows"] and sorted(job_of) == sorted(PWm["labels"]), ("captured pooled jobs differ from the record's", len(jobs), PWm["rows"])
        else:
            assert len(jobs) == fl_["jobs"] and sorted(job_of) == sorted(fl_["labels"]), ("captured full-length jobs differ from the record's", len(jobs), fl_["jobs"])
        assert len(al_) == fl_["allowed_files"], "uncut allowlist differs from the record's"
        for lab_ in labels if not CAT else ():  # every row the record carries for a checked label is a row the captured jobs produce, and the reverse
            assert set(rec["rows"].get(lab_, {})) == {o_ for o_, d_ in by_origin.items() if lab_ in d_}, (lab_, "record rows differ from the captured jobs")
    lst = {"record": f"{rpath} @ {commit}", "mode": mode, "cap": inp["cap"], "record_sha256": sha(recb), "code_commit": code, "inputs_sha256": sha(inb),
           "identity_parts": sorted([k_, v_] for k_, v_ in identity_parts(inp).items()),  # G7b VALUE: every resolved input part by sha256 (pairs)
           "shipped_inputs_equal_record": not DRY, "dry": DRY, "selftest": selftest(), "files": [{"path": p_, "sha256": h_} for p_, h_ in files],
           "nofilter_source": inp.get("nofilter_source"),
           "cohort_sha256": {f"{lab_} | {o}": sha(json.dumps(sorted(v["cohort"])).encode()) for lab_, vs_ in inp["variants_by_label"].items()
                             for o, v in vs_.items() if isinstance(v.get("cohort"), list)},
           "resolved_counts": {lab_: {o: ({"absent": True} if v.get("absent") else {"unavailable": True} if v.get("unavailable") else
                                          {k_: (len(v[k_]) if isinstance(v.get(k_), (list, dict)) else v.get(k_)) for k_ in
                                           ("cohort", "simh", "pool_b", "folders", "adm_sp", "pex", "spm", "zero_runs", "simh_zero_unit", "excl_prefixes", "replaced_root", "pending")
                                           if k_ in v} | ({"ao_groups": v["ao_groups"]} if "ao_groups" in v else {}))  # all-origins weights: built in check() after each row's withhold
                                      for o, v in vs_.items()} for lab_, vs_ in inp["variants_by_label"].items()}}  # counts only, for the reader
    lst["routec_pilot_zero_unit"] = pzi_  # G7b on route C (09 cond. 3): the fail-closed check's scope, by count
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    stem_ = Path(Path(rpath).name.removesuffix(".gz")).stem + ("_pwcatalogue" if CAT else "") + ("_foldmodel" if FL_ else "")  # G7b: a catalogue dry test is named apart from the
    # per-set record's own check; G7b FOLD-MODEL: and the fold-model extension apart from the 8-label catalogue check
    if CAT:
        lst["pw_catalogue"] = {"code_commit": code, "runner_sha256": sha((tree / D507 / "run_peraction311.py").read_bytes()), "pooled_world": m_["pooled_world"]}
    lst["cfg0_regression"] = {"rule": ("2e 20:3x: each public label's probe config and S_new arm files, from its own job cfg, equal the ones the base checker took "
                                       "from cfg0 (the first label's job); a difference STOPs before this listing is written"), "held": True, "per_label": cfg_reg_}
    if FL_:  # G7b FOLD-MODEL: the labels added, their fit pins, and what this check does not cover (2e 20:3x: stated in the record)
        lst["fold_model"] = {"labels": fold_lab_, "fit_check": inp["fold_fit_check"], "own_all_rs_fold_list": "@".join(RS_FOLDS[:2]) + f" sha256 {RS_FOLDS[2]}",
                             "not_covered": FOLD_NOT_COVERED, "not_applicable": inp["fold_not_applicable"], "scope": FOLD_SCOPE,
                             "public_labels_rechecked": [lb for lb in labels if lb not in fold_lab_],
                             "prior_g7b_record": f"{G7B_PRIOR[1]} @ {G7B_PRIOR[0]} sha256 {G7B_PRIOR[2]} (compared after the run: --compare-g7b-new)"}
    if VAL_:  # G7b VALUE (09 via 2e 22:4x): the captured inputs must equal Stage 1's independently verified sets BY HASH, else STOP: the files read
        S1L_, S1N_, S1I_, _ = value_pins(value_stage(FL_))  # this stage's verified inputs (Stage 2: STAGE2_TIE, the route C Stage 2 S1 listing)
        s1b_ = git("show", f"{S1L_[0]}:{S1L_[1]}")  # (path, sha256), the cohorts by key (cohort_sha256), the resolved counts, and the
        assert sha(s1b_) == S1L_[2], "the Stage 1 listing differs from its pinned sha256"  # pooled manifest (its pools files = the unions
        s1_ = json.loads(s1b_)  # Stage 1 rebuilt by key)
        hj_ = lambda x: sha(json.dumps(x, sort_keys=True).encode())  # noqa: E731
        tie_ = {"files": (hj_(sorted([f_["path"], f_["sha256"]] for f_ in lst["files"])), hj_(sorted([f_["path"], f_["sha256"]] for f_ in s1_["files"]))),
                "cohort_sha256": (hj_(lst["cohort_sha256"]), hj_(s1_["cohort_sha256"])), "resolved_counts": (hj_(lst["resolved_counts"]), hj_(s1_["resolved_counts"])),
                "pooled_manifest": (hj_(PWm.get("manifest")), hj_(((s1_.get("pw_catalogue") or {}).get("pooled_world") or {}).get("manifest")))}
        bad_tie_ = sorted(k_ for k_, (a_, b_) in tie_.items() if a_ != b_)
        assert not bad_tie_, ("captured inputs differ from Stage 1's verified sets (STOP; 09's tie)", bad_tie_)
        # Astra r5 #1 (2e 23:1x item 1): the tie BY IDENTITY: every resolved input part (each label's probe entry, each label x origin variant: cohorts,
        # pool keys and maps; every top-level map and list) by sha256 against the Stage 1 identity record, a recapture of Stage 1's inputs (its
        # inputs_sha256 = the Stage 1 listing's), pinned by sha256; only the parts in VALUE_ID_ALLOWED may differ, each for its stated reason
        s1ib_ = (HERE / "results" / S1I_[0].rsplit("/", 1)[1]).read_bytes()
        assert S1I_[0].startswith("experiments/2026-10-15_e2e_sim_pipeline/results/") and sha(s1ib_) == S1I_[1], \
            "the Stage 1 identity record differs from its pinned sha256"
        s1i_ = json.loads(s1ib_)
        assert s1i_["inputs_sha256"] == s1_["inputs_sha256"] == S1N_, "the Stage 1 identity record is not a recapture of Stage 1's inputs"
        idv_, ids1_ = identity_parts(inp), dict(s1i_["identity_parts"])
        assert len(ids1_) == len(s1i_["identity_parts"]), "the Stage 1 identity record repeats a part"
        idd_ = sorted(k_ for k_ in set(idv_) | set(ids1_) if idv_.get(k_) != ids1_.get(k_))
        tsx_ = None  # option B (route C, POST HOC): pw|tree_sha may differ only as TREE_SHA_DIFF lists, checked map against map; never by allow-list
        if "pw|tree_sha|sha256" in idd_:
            s1c_ = s1_["pw_catalogue"]["code_commit"]  # S1's capture tree, its map rebuilt from git blobs and tied to S1's identity part by hash
            s1t_ = {ln_.split("\t", 1)[1]: sha(git("show", f"{s1c_}:{ln_.split(chr(9), 1)[1]}")) for ln_ in
                    git("ls-tree", "--full-tree", s1c_, f"{D507}/").decode().splitlines() if ln_.split()[1] == "blob" and ln_.endswith(".py")}
            assert hj_(s1t_) == ids1_["pw|tree_sha|sha256"], "S1's tree_sha map rebuilt from git is not S1's identity part (STOP)"
            pin_ = set(inp["pw"]["required_modules"]) | set(inp["pw"].get("captured_modules") or {})
            add_ = TREE_SHA_ADDED.get(sha(recb)) or (W2_TREE_ADDED if W2K_ else None)  # PLANT V3: 09's grant (GATES @ 4e4ab4142 L596-598); rc1b W2 R1 (c)
            w2r1_ = None
            if W2K_:  # rc1b W2 R1 (09, POST HOC): (a) the runner diff is exactly the 14 hunks; (b) none inside a ported span (AST, both blobs)
                rpp_ = f"{D507}/run_peraction311.py"
                ob_, nb_ = git("show", f"{TREE_SHA_DIFF_BASE}:{rpp_}"), git("show", f"{W2_CODE}:{rpp_}")
                assert (sha(ob_), sha(nb_)) == W2_TREE_MODIFIED[rpp_], "rc1b W2 R1: the runner blobs' sha256 (STOP)"
                hk_ = tuple(re.sub(r" @@.*", " @@", ln_) for ln_ in git("diff", "-U0", TREE_SHA_DIFF_BASE, W2_CODE, "--", f":(top){rpp_}").decode().splitlines()
                            if ln_.startswith("@@"))
                assert hk_ == W2_RUNNER_HUNKS, ("rc1b W2 R1 (a): the runner diff is not exactly the 14 listed hunks (STOP)", len(hk_))
                svx_ = w2_span_viol(ob_.decode(), nb_.decode(), hk_)
                assert not svx_, ("rc1b W2 R1 (b): a hunk inside a ported span (STOP to 09)", svx_[:5])
                w2r1_ = {"hunks": list(hk_), "span_violations": svx_, "ported": list(W2_PORTED), "per_task_block": W2_PT_BLOCK,
                         "per_task_block_hunk": W2_PT_BLOCK_HUNK, "modified": {k_: list(v_) for k_, v_ in W2_TREE_MODIFIED.items()}}
            tv_ = tree_sha_diff_viol(s1t_, inp["pw"]["tree_sha"], pin_, TREE_SHA_DIFF, add_, W2_TREE_MODIFIED if W2K_ else None)
            assert not tv_, ("pw|tree_sha differs from S1's other than exactly TREE_SHA_DIFF (and the PLANT grant) (STOP)", len(tv_), tv_[:5])
            assert not add_ or not (set(add_) & pin_), "PLANT V3 grant (b): an added file is in the captured / required module list (STOP)"
            tsx_ = {"s1_code_commit": s1c_, "files": len(s1t_), "pinned_modules_equal": len(pin_), "differing": {k_: list(v_) for k_, v_ in TREE_SHA_DIFF.items()},
                    "rule": "option B (2e / 09, 10-20; post hoc): the catalogue rules pin S1 to the manifest's meta.git (4feb6694b), A was scored at "
                            "f2d47f19f; the two runner-folder trees may differ only in these 2 files, never in a pinned module"} | (
                {"added": dict(add_), "added_in_captured_modules": sorted(set(add_) & pin_),
                 "added_rule": "PLANT V3, 09's grant (GATES_routec @ 4e4ab4142 L596-598; post hoc): the PLANT value run only; these 4 files added at "
                               "752413181 since f2d47f19f, each by its sha256; none in the captured / required module list"} if add_ else {})
            if w2r1_:  # rc1b W2 R1: the record's own grant text (the PLANT one above names PLANT; this replaces it for a W2 record)
                tsx_["added_rule"] = ("rc1b W2 R1 (09, POST HOC; PRESTATED_W2 section 4 + AMEND1 section 2): the W2 records only; the 9 files added at 7797f2cc2 "
                                      "since f2d47f19f, each by its sha256; none in the captured / required module list; the runner modified by exactly the 14 "
                                      "listed hunks, none inside a ported span")
                tsx_["w2_r1"] = w2r1_
            idd_ = [k_ for k_ in idd_ if k_ != "pw|tree_sha|sha256"]
        cmx_ = None  # rc1b W2 R1 (09, POST HOC): pw|captured_modules may differ from S1's ONLY in the runner, exactly as W2_TREE_MODIFIED lists; S1's map
        if "pw|captured_modules|sha256" in idd_ and W2K_:  # is rebuilt from S1's tree (the same module paths) and tied to S1's identity part by hash
            s1c2_ = s1_["pw_catalogue"]["code_commit"]
            cm_ = dict(inp["pw"]["captured_modules"])
            s1cm_ = {k_: sha(git("show", f"{s1c2_}:{k_}")) for k_ in cm_}
            assert hj_(s1cm_) == ids1_["pw|captured_modules|sha256"], "rc1b W2 R1: S1's captured-module map rebuilt from git is not S1's identity part (STOP)"
            dfm_ = sorted(k_ for k_ in cm_ if cm_[k_] != s1cm_[k_])
            cmv_ = w2_capmods_viol(s1cm_, cm_, W2_TREE_MODIFIED)
            assert not cmv_ and dfm_ == sorted(W2_TREE_MODIFIED), ("rc1b W2 AMEND4: captured modules differ from S1's other than the listed runner (STOP)", cmv_)
            cmx_ = {"modules": len(cm_), "differing": {k_: [s1cm_[k_], cm_[k_]] for k_ in dfm_}, "s1_code_commit": s1c2_,
                    "rule": "rc1b W2 R1 (09, POST HOC): the captured module map equals S1's (rebuilt from S1's tree, tied by hash) except the runner, "
                            "exactly (S1's sha, W2's sha)"}
            idd_ = [k_ for k_ in idd_ if k_ != "pw|captured_modules|sha256"]
        w2lv_ = None  # rc1b W2b: kappa's job computes level 0.01 only (its cfg F_list; the margins are at 0.01, runner L2886): top|levels may differ
        if "top|levels|sha256" in idd_ and W2K_ == "kappa":  # from S1's ONLY as that subset (S1's set rebuilt and tied to S1's identity part by hash)
            s1lv_ = list(W2_S1_LEVELS)  # S1's level set (the default F_list), checked against S1's identity part by hash just below
            assert hj_(s1lv_) == ids1_["top|levels|sha256"] and not w2_levels_viol(inp["levels"], s1lv_), \
                "rc1b W2b: top|levels differs from S1 other than kappa's level-0.01 subset (STOP)"
            w2lv_ = {"s1_levels": s1lv_, "kappa_levels": inp["levels"], "rule": "rc1b W2b: kappa's job builds level 0.01 only; S1's set is tied by hash"}
            idd_ = [k_ for k_ in idd_ if k_ != "top|levels|sha256"]
        lvx_ = None  # GATES A22 @ f5441bb2c (POST HOC): top|levels may differ from S1 only as levels_allowance_viol allows; never by allow-list
        if "top|levels|sha256" in idd_:
            assert sha(recb) in ROUTEC_LEVEL_SPECS, "top|levels differs from S1 on a record without a pinned level spec (STOP)"
            sc_, sp_, sh_ = ROUTEC_LEVEL_SPECS[sha(recb)]
            assert git("rev-parse", sc_).decode().strip() == code, "the pinned level spec's commit is not the record's code commit (STOP; S3 review O3)"
            sb_ = git("show", f"{sc_}:{sp_}")
            assert sha(sb_) == sh_, "the pinned level spec's sha256 (STOP)"
            snm_ = json.loads(sb_)["entry"]["flags"].get("levels")
            reg_ = registered_levels(RP, snm_)  # (a) the spec's registered set, by the runner's own rule at the record's code commit
            rl_ = (rec["meta"].get("alltasks") or {}).get("levels")  # (b) the record's own levels (and RULE 37's own list where the record has one)
            r37_ = (rec["meta"].get("levels_rule37") or {}).get("levels")
            lv_v_ = levels_allowance_viol(inp["levels"], reg_, rl_ or []) + ([] if r37_ is None else levels_allowance_viol(inp["levels"], reg_, r37_))
            assert isinstance(rl_, list) and rl_ and not lv_v_, ("top|levels differs from S1 other than GATES A22 allows (STOP)", len(lv_v_), lv_v_[:5])
            lvx_ = {"spec": f"{sp_} @ {sc_} sha256 {sh_}", "spec_levels_name": snm_, "levels": len(inp["levels"]),
                    "record_levels_keys": ["meta.alltasks.levels"] + ([] if r37_ is None else ["meta.levels_rule37.levels"]),
                    "rule": "GATES A22 @ f5441bb2c (post hoc): top|levels equals BOTH the pinned spec's registered set AND the record's own levels"}
            idd_ = [k_ for k_ in idd_ if k_ != "top|levels|sha256"]
        fnx_ = None  # PLANT V3 (POST HOC; for 09's ruling): top|fold_not_applicable may differ from S1 ONLY in its jobs_checked count (PLANT builds the
        if "top|fold_not_applicable|sha256" in idd_ and sha(recb) == PLANT_RECORD_SHA:  # plant labels' jobs only); every other field must equal S1's
            s1f_ = (s1_.get("fold_model") or {}).get("not_applicable")  # S1's own value, from its listing, tied to S1's identity part by hash
            assert isinstance(s1f_, dict) and hj_(s1f_) == ids1_["top|fold_not_applicable|sha256"], "S1's not_applicable (listing) is not S1's identity part (STOP)"
            mf_ = inp["fold_not_applicable"]
            assert sorted(mf_) == sorted(s1f_) and all(mf_[k_] == s1f_[k_] for k_ in mf_ if k_ != "jobs_checked") and mf_["jobs_checked"] == len(jobs) \
                and 0 < mf_["jobs_checked"] < s1f_["jobs_checked"], "PLANT V3: fold_not_applicable differs from S1 other than its job count (STOP)"
            # 09's fifth condition: the captured PLANT job set equals S1's job set restricted to PLANT's labels, BY JOB KEY (record label, origin vtag).
            # S1's listing holds no per-job list; its job keys for its checked labels are its resolved_counts keys (label -> origin vtag), which this
            # checker builds from S1's captured jobs (variants_by_label; L4861-4870's key). S1's checked labels are exactly PLANT's 12, so the
            # restriction is that whole map; S1's other jobs (jobs_checked minus these) are of labels outside both.
            pl12_ = sorted(rec["rows"])
            jk_ = {(ol_, o_) for o_, d_ in by_origin.items() for ol_ in d_}
            s1k_ = {(lb_, o_) for lb_, od_ in (s1_.get("resolved_counts") or {}).items() for o_ in od_}
            assert len(jk_) == len(jobs) == mf_["jobs_checked"] and all(ol_ in set(pl12_) for ol_, _ in jk_) and sorted(s1_["resolved_counts"]) == pl12_ \
                and jk_ == {k_ for k_ in s1k_ if k_[0] in set(pl12_)}, "PLANT V3 (09 condition 5): the captured job keys are not S1's restricted to PLANT's labels (STOP)"
            # S1's BUILT vs CHECKED labels (2e; AMEND6), by key: built = route C A's meta.pooled_world.labels (A's record @ a936a5d97, by sha256;
            # S1 checked A's inputs), checked = S1's resolved_counts labels = PLANT's 12; 7 captured job keys per PLANT label, so PLANT's jobs_checked
            # = 7 x checked and S1's = 7 x built; the built-not-checked labels carry the rest (S1 lists no per-job keys for them: by count only)
            ac_, ap_, ah_ = PLANT_V3_FILES["base_record.json.gz"]
            ab_ = git("show", f"{ac_}:{ap_}")
            assert sha(ab_) == ah_, "PLANT V3: A's record sha256 (STOP)"
            al_ = sorted(json.loads(gzip.decompress(ab_))["meta"]["pooled_world"]["labels"])
            slv_ = s1_labels_viol(al_, pl12_, jk_, s1f_["jobs_checked"], mf_["jobs_checked"])
            assert not slv_, ("PLANT V3: S1's built / checked labels and job counts (STOP)", slv_)
            fnx_ = {"jobs_checked": {"plant": mf_["jobs_checked"], "s1": s1f_["jobs_checked"]}, "other_fields_equal": True,
                    "job_keys_equal_s1_restricted": True, "n_job_keys": len(jk_), "s1_jobs_outside_plant_labels": s1f_["jobs_checked"] - len(jk_),
                    "s1_labels": {"built": len(al_), "checked": len(pl12_), "built_not_checked": len(set(al_) - set(pl12_)),
                                  "built_not_checked_labels": sorted(set(al_) - set(pl12_)), "jobs_per_label": 7,
                                  "jobs": {"built": s1f_["jobs_checked"], "checked": mf_["jobs_checked"], "built_not_checked": s1f_["jobs_checked"] - len(jk_)},
                                  "source": f"built = {ap_} @ {ac_[:9]} (sha256 {ah_[:12]}) meta.pooled_world.labels; checked = S1's resolved_counts labels "
                                            "(= PLANT's rows); the built-not-checked labels' jobs by count (S1 lists no per-job keys for them)"},
                    "rule": "POST HOC tie allowance 2 (09): fold_not_applicable.jobs_checked differs only because PLANT builds the plant labels' jobs "
                            f"({mf_['jobs_checked']} = S1's {s1f_['jobs_checked']} restricted to PLANT's labels); job keys (label, origin) equal S1's "
                            "resolved_counts keys for PLANT's 12 labels"}
            idd_ = [k_ for k_ in idd_ if k_ != "top|fold_not_applicable|sha256"]
        assert not tie_parts_viol(idd_, VALUE_ID_ALLOWED), ("captured input parts differ from Stage 1's by identity (STOP)", tie_parts_viol(idd_, VALUE_ID_ALLOWED))
        ideq_ = sorted(k_ for k_ in set(idv_) & set(ids1_) if idv_[k_] == ids1_[k_])
        assert sum(k_.startswith("variant|") for k_ in ideq_) == sum(len(v_) for v_ in inp["variants_by_label"].values()) and \
            sum(k_.startswith("by_label|") for k_ in ideq_) == len(labels), "not every label and label x origin part is tied by identity"
        lst["value_reproduction"] = {"scope": W2_SCOPE[W2K_] if W2K_ else value_scope(inp["boot"]["seed"]), "stage1_tie": {"listing": "@".join(S1L_[1::-1]) + f" sha256 {S1L_[2]}",
                                                                          "equal_by_sha256": {k_: a_ for k_, (a_, _) in tie_.items()},
                                                                          "identity_record": f"{S1I_[0]} sha256 {S1I_[1]}",
                                                                          "identity_parts_equal": len(ideq_), "identity_parts_differing": {k_: VALUE_ID_ALLOWED[k_] for k_ in idd_},
                                                                          "tree_sha_exception": tsx_, "levels_exception": lvx_, "fold_not_applicable_exception": fnx_,
                                                                          "captured_modules_exception": cmx_, "w2_levels_exception": w2lv_},
                                     "plants": [list(x_) for x_ in VPL_], "not_compared": VALUE_NOT_COMPARED, "notes": VALUE_NOTES,
                                     "boot": inp["boot"], "tolerance": "|a - b| <= 1e-9 * max(1, |a|, |b|); booleans, integers and keys exact; NaN equals only NaN"}
    W2C_, W2CTL_, W2PINS_ = None, None, {}  # rc1b W2 (PRESTATED_W2 + AMEND1-3): the per-call W2 block, the control call's plants, A's D1 pins
    if VAL_ and W2K_:
        arb2_ = gzip.decompress(git("show", f"{W2_A_REF[0]}:{W2_A_REF[1]}"))
        assert sha(arb2_) == W2_A_REF[2], "rc1b W2: route C A's record sha256 (STOP)"
        W2PINS_ = {k_: {"path": v_["path"], "sha256": v_["sha256"]} for k_, v_ in json.loads(arb2_)["meta"]["pooled_world"]["calibrations"]["by_label_combo"].items()}
        assert len(W2PINS_) == 36, "rc1b W2: A's D1 calibration pins are not 18 labels x 2 combos (STOP)"
        if W2K_ == "pertask":  # the record's own D1 reference is route C A (meta.pertask.reference_record names A's commit, path and sha256)
            assert m_["pertask"]["reference_record"] == f"{W2_A_REF[0]}:{W2_A_REF[1]} ({W2_A_REF[2]})", "rc1b W2a: Stage D's reference is not route C A (STOP)"
        else:
            assert str((m_.get("kappa") or {}).get("reference_record") or "").startswith(f"{W2_A_REF[0]}:{W2_A_REF[1]}") \
                and W2_A_REF[2] in str(m_["kappa"]["reference_record"]), "rc1b W2b: kappa's reference is not route C A (STOP)"
        rpp2_ = f"{D507}/run_peraction311.py"
        W2C_ = {"kind": W2K_, "f": float(m_["pertask"]["f"]) if W2K_ == "pertask" else 0.01, "stage": value_stage(FL_),
                "modified": {k_: list(v_) for k_, v_ in W2_TREE_MODIFIED.items()}, "script_pair": list(W2_TREE_MODIFIED[rpp2_])}
        rg2_ = __import__("numpy").random.default_rng(W2_CTL_SEED)  # the control call: a seeded label, then two distinct seeded rows of it
        cl2_ = sorted(labels)[int(rg2_.integers(len(labels)))]
        os2_ = sorted(inp["variants_by_label"][cl2_])
        o1_ = os2_[int(rg2_.integers(len(os2_)))]
        o2_ = [o_ for o_ in os2_ if o_ != o1_][int(rg2_.integers(len(os2_) - 1))]
        W2CTL_ = {"label": cl2_, "seed": W2_CTL_SEED,
                  **({"plant_score": {"origin": o1_, "seed": W2_CTL_SEED}, "plant_record": {"origin": o2_, "seed": W2_CTL_SEED + 1},
                      "plant_pin": {"origin": o2_}} if W2K_ == "pertask" else
                     {"plant_margin": {"origin": o1_, "seed": W2_CTL_SEED}, "plant_store": {"origin": o2_, "seed": W2_CTL_SEED + 1, "drop_origin": [
                         o_ for o_ in os2_ if o_ not in (o1_, o2_)][int(rg2_.integers(len(os2_) - 2))]}})}
        W2RS_ = {}  # rc1b W2b: label -> row -> its row-store files and names (job i -> "<dir>/<i:04d>_<sha16>_<attempt>.pkl", runner drive L5049 / L1337)
        if W2K_ == "kappa":
            rsd_ = m_["row_store"]["dir"]
            assert rsd_ == f"xfer/fpr507prep/rowstore/{Path(rpath).name.removesuffix('.gz').removesuffix('.json')}", "rc1b W2b: the row store is not the record's (STOP)"
            byi_ = {}
            for e_ in modal.Volume.from_name(VOLS["solavg"], version=2).listdir(rsd_):  # names only (no file content is read on the laptop)
                mi_ = re.fullmatch(r"(\d{4})_([0-9a-f]{16})_([0-9a-f]{8})\.pkl", e_.path.rsplit("/", 1)[-1])
                assert mi_, ("rc1b W2b: an unexpected file in the row store (STOP)", e_.path.rsplit("/", 1)[-1])
                byi_.setdefault(int(mi_[1]), []).append(e_.path.lstrip("/"))
            assert sorted(byi_) == list(range(len(jobs))), ("rc1b W2b: the row store does not hold one row per captured job (STOP)", len(byi_), len(jobs))
            for i_, (lb_, cf_, _) in enumerate(jobs):
                ol_ = cf_.get("full_as", lb_)
                assert by_origin[job_vtag(cf_)][ol_] is cf_, ("rc1b W2b: the job order does not map to the captured rows (STOP)", i_)
                W2RS_.setdefault(ol_, {})[job_vtag(cf_)] = {"paths": sorted(byi_[i_]), "labels": sorted({lb_, ol_})}
        lst["w2"] = {"kind": W2K_, "stage": W2C_["stage"], "scope": W2_SCOPE[W2K_], "control": W2CTL_, "d1_pins": len(W2PINS_),
                     "count_per_label": W2_COUNT[W2K_], "fanout": W2_FANOUT[W2C_["stage"]], "app": APP_NAME}
        if W2K_ == "kappa":  # AMEND5 section 1: the row store by names (the attempts' completeness is counted in the container); AMEND6 (Codex W2
            # r1 #3): the full name map and its digest are in the listing, and a launch STOPs unless its live map has the approved capture's digest
            rsh_ = sha(json.dumps(W2RS_, sort_keys=True).encode())
            lst["w2"]["row_store_names"] = {"dir": rsd_, "n_jobs": len(byi_), "n_files": sum(len(v_) for v_ in byi_.values()),
                                            "jobs_with_more_than_one_file": sum(int(len(v_) > 1) for v_ in byi_.values()), "sha256": rsh_, "by_row": W2RS_}
            assert not launch or w2_rowstore_sha == rsh_, "rc1b W2b: the row store's file names are not the approved capture's (--w2-rowstore-sha; STOP)"
        import sysconfig  # AMEND6 (Codex W2 r1 #2; PRESTATED_W2 section 4 (c)): the module-path gate as in PLANT V3 (AMEND9), for W2 captures

        envw_ = sorted({str(Path(p_).resolve()) for p_ in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sysconfig.get_paths().values())})
        LMW_ = {n_: str(Path(m_.__file__).resolve()) for n_, m_ in sorted(sys.modules.items()) if isinstance(getattr(m_, "__file__", None), str)}
        LMW_ = {n_: f_ for n_, f_ in LMW_.items() if not any(f_.startswith(p_.rstrip("/") + "/") for p_ in envw_)}
        lmw_ = [str(HERE.parent.parent.resolve()), str((TREES / code[:12]).resolve())]
        assert not local_modules_viol(LMW_, envw_, lmw_), ("STOP (rc1b W2): a local module outside the W2 worktree and the record's tree",
                                                           local_modules_viol(LMW_, envw_, lmw_))
        lst["w2"] |= {"local_modules": LMW_, "local_module_roots": lmw_}  # paths only
        assert APP_NAME == W2_APPS[("a1", "a2", "b1", "b2").index(("a" if W2K_ == "pertask" else "b") + W2C_["stage"])], \
            ("rc1b W2: PROBE_SHIFT_W2_APP is not this run's app (STOP)", APP_NAME)
    PLV_, P1_, PV3F_ = None, None, {}  # PLANT V3 (PRESTATED_V3 fb34c414 + AMEND1-5): the plant inputs, P1's label, block (v)'s files (keys and shas only)
    if VAL_ and sha(recb) == PLANT_RECORD_SHA:
        mp_ = rec["meta"]["plant"]
        assert not plant_level_part_viol(True, level_part), plant_level_part_viol(True, level_part)  # Codex V3 r2 #4 (AMEND7): unsplit only (STOP)
        PLV_ = {"labels": sorted(rec["rows"]), "points": [float(x_) for x_ in mp_["points"]], "deltas": [int(x_) for x_ in mp_["deltas"]]}
        assert sorted(labels) == PLV_["labels"] and value_stage(FL_) == "1", "PLANT V3: the checked labels are the PLANT record's 12 rows (Stage 1)"
        assert 0 in PLV_["deltas"] and all(any(abs(x_ - y_) <= 1e-12 * y_ for y_ in inp["levels"]) for x_ in PLV_["points"]), "PLANT V3: deltas / points"
        rd_ = [ln_.split("\t") for ln_ in git("diff", "--name-status", PLANT_REPO_DIFF[0], PLANT_REPO_DIFF[1]).decode().splitlines()]
        assert len(rd_) == PLANT_REPO_DIFF[2] and all(x_[0] == "A" for x_ in rd_), "PLANT V3 fixture (AMEND4 D6): the repo diff f2d47f19f -> 752413181 is 9 x A"
        assert set(TREE_SHA_ADDED[PLANT_RECORD_SHA]) <= {x_[1] for x_ in rd_}, "PLANT V3: the 4 granted files are among the 9 additions"
        for nm_, (c_, p_, h_) in sorted(PLANT_V3_FILES.items()):  # block (v) inputs, from git, by sha256 (the values are parsed only on Modal)
            b_ = git("show", f"{c_}:{p_}")
            assert sha(b_) == h_, (nm_, "a block (v) input's sha256 differs from its pin (STOP)")
            PV3F_[nm_] = b_
        P1_ = {"label": sorted(labels)[int(__import__("numpy").random.default_rng(PLANT_P1_SEED).integers(len(labels)))], "seed": PLANT_P1_SEED}
        # Codex V3 #8: the PLANT pinned step spec, read at the record's code commit by sha256: its flags say plant, name no level set (so the level
        # set is the runner's default, which top|levels ties to S1's by identity above) and the same IL block as the record; its script is the runner
        spb_ = git("show", f"{code}:{PLANT_STEP_SPEC[0]}")
        assert sha(spb_) == PLANT_STEP_SPEC[1], "PLANT V3: the pinned step spec's sha256 at the record's code commit (STOP)"
        spj_ = json.loads(spb_)
        ilb_ = (rec["meta"].get("il_arms") or {}).get("block") or {}
        assert spj_["entry"]["flags"].get("plant") is True and "levels" not in spj_["entry"]["flags"] and lvx_ is None \
            and spj_["script"] == f"{D507}/run_peraction311.py" and spj_["script_blob"] == git("rev-parse", f"{code}:{D507}/run_peraction311.py").decode().strip() \
            and spj_["entry"]["flags"].get("il_arms") == f"{ilb_.get('path')}@{ilb_.get('sha256')}", "PLANT V3: the pinned step spec's flags / script (STOP)"
        # AMEND6 (Codex V3 #3): the IL block (sim2's source), read at the record's code commit by the record's own pin; each label x IL origin's report
        # ported here (il_report_v3); the runner's constants this port copies are checked in the pinned sources
        ilbb_ = git("show", f"{code}:{D507}/{ilb_['path']}")
        assert sha(ilbb_) == ilb_["sha256"], "PLANT V3: the IL block's sha256 at the record's code commit (STOP)"
        ilj_ = json.loads(ilbb_)
        assert ilj_.get("schema") == "ilarms/v1" and set(ilj_["sets"]) <= set(IL_ORIGINS), "PLANT V3: the IL block's schema / origins"
        ils_ = git("show", f"{code}:{D507}/il_arms.py").decode()
        rps_ = git("show", f"{code}:{D507}/run_peraction311.py").decode()
        assert all(f'"{x_}"' in ils_ for x_ in (IL_COUNTS_RULE[:IL_COUNTS_RULE.index("; ") + 2], IL_COUNTS_RULE[IL_COUNTS_RULE.index("; ") + 2:])) \
            and f'{{"withheld": "{PLANT_PANEL_WITHHELD}"}}' in rps_ and f'{{"unavailable": "{PLANT_COH_UNAV["unavailable"]}"}}' in rps_, \
            "PLANT V3: a ported constant is not in the pinned source (STOP)"
        IL_, ilv_ = {}, []  # per label x row (vtag): the report from the row's OWN captured job cfg (the runner's ILX = cfg["il_arms"], L1526), which
        for o_, d_ in sorted(by_origin.items()):  # is the pinned block as full_override.apply_full left it (L310-325 @ 752413181: a 32k file without
            bo_ = o_.split(" ")[0]  # an uncut entry dropped, else swapped for the uncut one), checked against the pinned block but for files
            for ol_, cf_ in sorted(d_.items()):
                if ol_ not in labels or bo_ not in IL_ORIGINS:
                    continue
                assert isinstance(cf_.get("il_arms"), dict), (ol_, o_, "PLANT V3: an IL-origin job without its IL block (STOP)")
                ilv_ += [f"{ol_} | {o_} | {x_}" for x_ in il_block_viol(cf_["il_arms"], ilj_)]
                p_ = next(q_ for q_ in cf_["probes"] if q_["label"] == job_of[ol_])
                IL_.setdefault(ol_, {})[o_] = il_report_v3(cf_["il_arms"], bo_, p_["new"])
        assert not ilv_, ("PLANT V3: a job's IL block differs from the pinned block other than in its files (STOP)", len(ilv_), ilv_[:5])
        assert all(sorted(IL_.get(lab_, {})) == sorted(o_ for o_ in inp["variants_by_label"][lab_] if o_.split(" ")[0] in IL_ORIGINS) for lab_ in labels), \
            "PLANT V3: an IL-origin row without its job's IL report (STOP)"
        HLP_ = {nm_: sha((HERE / nm_).read_bytes()) for nm_ in ("plant_rows_v3.py", "plant_rows_v3_port.py")}  # Codex V3 #6: the helpers' fingerprints
        import sysconfig  # AMEND9 (2e): the local modules this capture imported, each under the V3 worktree or the record's code-commit tree

        envp_ = sorted({str(Path(p_).resolve()) for p_ in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sysconfig.get_paths().values())})
        LMOD_ = {n_: str(Path(m_.__file__).resolve()) for n_, m_ in sorted(sys.modules.items()) if isinstance(getattr(m_, "__file__", None), str)}
        LMOD_ = {n_: f_ for n_, f_ in LMOD_.items() if not any(f_.startswith(p_.rstrip("/") + "/") for p_ in envp_)}
        lmr_ = [str(HERE.parent.parent.resolve()), str((TREES / code[:12]).resolve())]
        assert not local_modules_viol(LMOD_, envp_, lmr_), ("STOP (AMEND9): a local module outside the V3 worktree and the record's tree", local_modules_viol(LMOD_, envp_, lmr_))
        print(json.dumps({"local_modules": LMOD_, "local_module_roots": lmr_}), flush=True)  # paths only
        lst["plant_v3"] = {"labels": PLV_["labels"], "points": PLV_["points"], "deltas": PLV_["deltas"], "boot_seed": inp["boot"]["seed"],
                           "step_spec": {"path": PLANT_STEP_SPEC[0], "commit": code, "sha256": PLANT_STEP_SPEC[1], "plant_flag": True, "levels_flag": None,
                                         "script_blob_is_runner": True, "il_arms_flag_is_record_block": True},
                           "il_block": {"path": f"{D507}/{ilb_['path']}", "commit": code, "sha256": ilb_["sha256"], "reports": sum(len(v_) for v_ in IL_.values()),
                                        "reports_with_rows": sum(1 for v_ in IL_.values() for r0_ in v_.values() if r0_["rows"] or r0_["folders_read"]), "scope": PLANT_SIM2_SCOPE},
                           "helpers_sha256": HLP_, "local_modules": LMOD_, "local_module_roots": lmr_,
                           "files": {nm_: sha(b_) for nm_, b_ in PV3F_.items()}, "p1": P1_, "not_compared": PLANT_NOT_COMPARED,
                           "p1_allowed_templates": ["row <origin> | <level>|plant|solve_draw0|sim|arms|<delta>|c", "row <origin> | <level>|plant|solve_draw0|sim|arms|<delta>|achieved_points",
                                                    "row <origin> | <level>|plant|catch|sim_plant_<delta>", "row <origin> | boot|joint|<level>|draw <b>|plant|catch|sim_plant_<delta>",
                                                    "pooled <combo> | boot|joint|<level>|draw <b>|pooled|plant|catch_sim_plant_<delta>",
                                                    "pooled <combo> | boot|joint|<level>|draw <b>|pooled|plant|gap_delta_full_plant_<delta>",
                                                    "agg | plant|<combo>|joint|<level>|headline_six|<readout>|<delta>|S"],
                           "repo_diff": {"from": PLANT_REPO_DIFF[0], "to": PLANT_REPO_DIFF[1], "n": len(rd_), "all_added": True}}
    if schema:  # G7 --schema: the capture above passed its record checks; now the key paths, keys and counts only
        assert PWm and not CAT and not launch, "--schema takes a pooled scored record and never launches"
        mw_ = schema_walk(rec, labels, origins, list(PWm["combos"]), inp["levels"])
        dest = HERE / "results" / f"check_stagea_schema_{stem_}_{stamp}.json"
        dest.open("x").write(json.dumps({"record": lst["record"], "record_sha256": lst["record_sha256"], "code_commit": code, "labels": labels,
                                         "origins": origins, "combos": list(PWm["combos"]), "levels": [f"{x:g}" for x in inp["levels"]], "inputs_sha256": sha(inb),
                                         "capture": "passed (jobs, labels, rows and allowlist against the record)", "missing_paths": mw_,
                                         "stub_notes": stub_notes}, indent=1) + "\n")
        print(json.dumps({"schema": "PASS" if not mw_ else "MISSING PATHS", "n_patterns_missing": len(mw_), "stub_notes": stub_notes, "file": str(dest)}, indent=1))
        return
    if not launch:
        dest = HERE / "results" / f"check_stagea_inputs_{stem_}_{stamp}.json"
        dest.open("x").write(json.dumps(lst, indent=1) + "\n")
        print(json.dumps({"inputs_sha256": sha(inb), "listing": str(dest), "listing_sha256": sha(dest.read_bytes()), "files": len(files)}
                         | ({"plant_v3": lst["plant_v3"], "tree_sha_exception": (lst.get("value_reproduction") or {}).get("stage1_tie", {}).get("tree_sha_exception"),
                               "fold_not_applicable_exception": (lst.get("value_reproduction") or {}).get("stage1_tie", {}).get("fold_not_applicable_exception"),
                               "identity_parts_differing": sorted((lst.get("value_reproduction") or {}).get("stage1_tie", {}).get("identity_parts_differing") or {})}
                            if PLV_ else {})
                         | ({"w2_row_store_names_sha256": lst["w2"]["row_store_names"]["sha256"]} if (lst.get("w2") or {}).get("row_store_names") else {}),
                         indent=1))  # PLANT V3: keys, shas and templates only (the P1 allowed set's key names, 2e); rc1b W2b (AMEND6): the names' digest
        return
    assert sha(inb) == inputs_sha, "the resolved inputs differ from the READ_LOG listing"
    if shapes:  # G7b FOLD-MODEL --shapes: the fold labels' pinned files and pooled entries, keys / shapes only, one spawned call
        fset_ = {}

        def walk_(x, kind):
            if isinstance(x, dict):
                if isinstance(x.get("path"), str) and x["path"].endswith(".npz") and isinstance(x.get("sha256"), str):
                    fset_.setdefault((x.get("vol", "conv"), x["path"], x["sha256"]), kind)
                for k_, v_ in x.items():
                    walk_(v_, "simh_sol" if (k_ == "simh" and kind == "five") else kind)
            elif isinstance(x, list):
                for v_ in x:
                    walk_(v_, kind)
        nfam_ = {}
        for lab in fold_lab_:  # per family: the files its two labels read (a family's C and R files differ; families share none)
            k0_ = "own_all_rs" if inp["by_label"][lab].get("family") == "own_all_rs" else "five"
            n0_ = set(fset_)
            walk_(inp["by_label"][lab], k0_)
            walk_(inp["variants_by_label"][lab], k0_)
            nfam_[lab.rsplit(" ", 1)[0]] = nfam_.get(lab.rsplit(" ", 1)[0], 0) + len(set(fset_) - n0_)
        fl5_ = sorted([*k_, v_] for k_, v_ in fset_.items())
        assert {tuple(f_[:3]) for f_ in fl5_} <= reads_, "a fold-label file outside this check's reads"
        assert nfam_ == {f_: FOLD_FILES[f_] for f_ in FL_} and len(fl5_) == sum(nfam_.values()), \
            ("--shapes: the walk's file count per family differs from the listing's (NB-a)", nfam_)
        run_dir = f"shapes/{app.app_id}"
        print(json.dumps({"app_id": app.app_id, "run_dir": run_dir, "files": len(fl5_)}), flush=True)
        fc_ = shapes_fn.spawn(fl5_, inp["pw"], fold_lab_, run_dir)
        print(json.dumps({"shapes_call": fc_.object_id}), flush=True)
        r_ = fc_.get()
        fbad_ = [f_["path"] for f_ in r_["files"] if not f_["sha256_ok"] or f_["missing_required"] or f_["shape_faults"]]
        pbad_ = sorted(k_ for k_, v_ in r_["pooled"].items() if not v_["in_manifest"] or v_.get("missing_fields") or not v_.get("pooled_sha256_ok")
                       or v_.get("shape_faults"))
        vd_ = ("PASS" if not fbad_ and not pbad_ and r_["manifest_sha256_ok"] and r_["catalogue"]["digest_equals_manifest"] and not r_["manifest_findings"]
               else "FINDINGS")
        dest = HERE / "results" / f"check_stagea_shapes_{stem_}_{stamp}.json"
        dest.open("x").write(json.dumps({"listing_inputs_sha256": sha(inb), "app_id": app.app_id, "run_dir": run_dir, "labels": fold_lab_, "verdict": vd_,
                                         "files_with_findings": fbad_, "pooled_with_findings": pbad_,
                                         "kinds": {k_: sum(1 for f_ in fl5_ if f_[3] == k_) for k_ in SHAPE_NEEDS}} | r_, indent=1) + "\n")
        print(json.dumps({"shapes": vd_, "files": len(fl5_), "files_with_findings": len(fbad_), "pooled_entries": len(r_["pooled"]),
                          "pooled_with_findings": pbad_, "catalogue": r_["catalogue"], "manifest_findings": r_["manifest_findings"], "file": str(dest)}, indent=1))
        if vd_ != "PASS":  # Claude N7: FINDINGS exit non-zero (the wrapper's exit and the guard's record carry it)
            raise SystemExit(f"--shapes: FINDINGS ({len(fbad_)} files, {len(pbad_)} pooled entries); see {dest}")
        return
    assert not level_part or VAL_, "--level-part is for a value run only"
    lvp_ = level_part_levels(inp["levels"], level_part) if level_part else None  # B's split (A22): this run's levels; the identity tie and the inputs
    args_ = []  # sha above stay on the FULL inputs; each per-label call gets the part (levels) and the full set (levels_all)
    for lab in labels:  # one call per label keeps each input small; the calls run in parallel
        one = dict(inp) | {"labels": [lab], "variants": inp["variants_by_label"][lab]} | ({"levels": lvp_, "levels_all": inp["levels"]} if lvp_ else {})
        one |= {"plant_v3": PLV_ | {"il": IL_.get(lab, {})}} if PLV_ else {}
        one |= {"w2": W2C_ | {"d1_pins": {k_: v_ for k_, v_ in W2PINS_.items() if k_.startswith(f"{lab} | ")}}
                       | ({"row_store": W2RS_.get(lab, {})} if W2C_["kind"] == "kappa" else {})} if W2C_ else {}  # rc1b W2, after the tie  # PLANT V3: added per call AFTER the identity tie (inputs_sha256 and identity parts
        # unchanged); AMEND6: this label's own IL reports (ported in main from the pinned block)
        one.pop("variants_by_label")
        args_.append((one, b"", stamp))  # G5: the record ships ONCE, to the results Volume (not inside each of the per-label inputs)
    pargs_ = []  # G7b VALUE PLANT: one extra call per plant, its label's input plus the plant (a separate execution; shards under plant/)
    for lb_, o_, a_ in VPL_:
        pargs_.append((dict(args_[labels.index(lb_)][0]) | {"value_plant": {"label": lb_, "origin": o_, "arm": a_}}, b"", stamp))
    p1args_ = [(dict(args_[labels.index(P1_["label"])][0]) | {"plant_p1": dict(P1_)}, b"", stamp)] if P1_ else []  # PLANT V3 P1: its own call (p1/)
    w2cargs_ = [(dict(args_[labels.index(W2CTL_["label"])][0]) | {"w2": dict(args_[labels.index(W2CTL_["label"])][0]["w2"]) | {
        k_: v_ for k_, v_ in W2CTL_.items() if k_.startswith("plant_")}}, b"", stamp)] if W2CTL_ else []  # rc1b W2: the control call (w2ctl/)
    run_dir = f"stagea_check/{app.app_id}"  # detach: the run manifest first, then one spawned call per label, each writing its own shard
    man = {"run_dir": run_dir, "app_id": app.app_id, "labels": list(labels), "record": lst["record"], "record_stem": stem_ + (f"_levelpart{level_part.replace('/', 'of')}" if lvp_ else ""), "record_file": f"{run_dir}/record.json",
           "record_sha256": sha(recb), "inputs_sha256": sha(inb), "script_sha256": sha(Path(__file__).read_bytes()), "git": git("rev-parse", "HEAD").decode().strip(),
           "read_started_pt": stamp, "mode": mode, "cap": inp["cap"], "nofilter_source": inp.get("nofilter_source"), "cohort_sha256": lst["cohort_sha256"],
           "per_label_input_sha256": {lab: sha(json.dumps(a_[0], sort_keys=True).encode()) for lab, a_ in zip(labels, args_)},
           "levels": lvp_ or inp["levels"], "levels_all": inp["levels"], "level_part": level_part or None, "aggregate": inp["mode"] == "alltasks" and bool(rec["meta"].get("full_length")) and not CAT,  # G6: S/Y and the excess
           "pw_catalogue": lst.get("pw_catalogue"), "combo_rows": (inp.get("pw") or {}).get("combo_rows"),  # Codex r46 #1: every label's members, in aggregate()
           "value": VAL_, "value_stage": value_stage(FL_) if VAL_ else None, "value_scope": (W2_SCOPE[W2K_] if W2K_ else value_scope((inp.get("boot") or {}).get("seed"))) if VAL_ else None, "boot": inp.get("boot"),  # G7b VALUE
           "plants": [a_[0]["value_plant"] for a_ in pargs_], "plant_input_sha256": {a_[0]["value_plant"]["label"]: sha(json.dumps(a_[0], sort_keys=True).encode()) for a_ in pargs_}}
    if W2C_:  # rc1b W2: what aggregate() reads by key: the kind, stage, count gate, the control call's label, plants and input sha256
        man["w2"] = dict(lst["w2"]) | {"control_input_sha256": sha(json.dumps(w2cargs_[0][0], sort_keys=True).encode())}
    if PLV_:  # PLANT V3: the plant inputs, block (v)'s file pins, and P1's call (aggregate() reads them by key)
        assert man["aggregate"] and man["value"], "PLANT V3 needs the aggregate step of a value run"
        man["plant_v3"] = dict(PLV_) | {"files": {nm_: sha(b_) for nm_, b_ in PV3F_.items()}, "p1": dict(P1_), "helpers_sha256": dict(HLP_),  # Codex V3 #6
                                        "step_spec": dict(lst["plant_v3"]["step_spec"]), "il_block": dict(lst["plant_v3"]["il_block"]),  # Codex V3 #8 / #3
                                        "local_modules": dict(lst["plant_v3"]["local_modules"]),  # AMEND9
                                        "p1_input_sha256": sha(json.dumps(p1args_[0][0], sort_keys=True).encode())}
    with res_vol.batch_upload() as bu:  # exclusive (an existing file fails): the record once, then the manifest that names its sha256
        bu.put_file(io.BytesIO(recb), f"/{run_dir}/record.json")
        for nm_, b_ in sorted(PV3F_.items()):  # PLANT V3 block (v)'s inputs (their sha256 in the manifest; checked again on Modal)
            bu.put_file(io.BytesIO(b_), f"/{run_dir}/plant_v3/{nm_}")
        bu.put_file(io.BytesIO(json.dumps(man, indent=1).encode()), f"/{run_dir}/manifest.json")
    print(json.dumps({"app_id": app.app_id, "run_dir": run_dir, "reduce_with": f"--reduce {app.app_id}"}), flush=True)
    assert not plant_fanout_viol(bool(PLV_), len(args_ + pargs_ + p1args_)), plant_fanout_viol(bool(PLV_), len(args_ + pargs_ + p1args_))  # AMEND11
    assert not W2C_ or len(args_ + pargs_ + p1args_ + w2cargs_) == W2_FANOUT[W2C_["stage"]], ("rc1b W2: the fan-out is not the pre-stated 13 / 15 (STOP)",
                                                                                         len(args_ + pargs_ + p1args_ + w2cargs_))
    fc_ = drive.spawn(args_ + pargs_ + p1args_ + w2cargs_, run_dir, man["aggregate"])  # the single last-triggered call; it runs the per-label checks and the plants (then G6's aggregate)
    print(json.dumps({"driver_call": fc_.object_id}), flush=True)
    fc_.get()
    reduce_run(app.app_id, "")  # the results file always comes from the Volume shards (the same path as a later --reduce)
