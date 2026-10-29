"""S_paper493 admission and routing rules for the step-3 recipes (RULE Addendum P; 09's rulings relayed by 4f 10-15/16). Pure functions
over pinned git files; no Modal, no activation, no score. Used by score_step3_sp493.py (launch checks) and test_sp493_rules.py.

- Grammar: 09's four regexes @ 9cb3d0291, verbatim (checked against the pinned file at load).
- Refused by name: d9's names record @ 32226c415 (superseded, burned_or_taken, superseded_v1_launches, non_routable); and the _try rule:
  real sp493_<box>_{main,run2}_v1 needs _try >= 3; g1/g2 snapshots (_s<j>) need _try >= 2; g1/g2 final remainders (no _s) are
  admitted at _v1 iff not on the lists (09 00:2x). Either rule trips -> refused (fail closed).
- Cells: the S_paper493 list in the facts file @ d800edc6a; t987 is refused by name with 09's reason; g4 (world sp493_t196) carries
  T196 task keys, admitted iff the task's (env, main) is an S_paper493 main with main_origin "t196"; any other T196 key is refused.
- Folds (step-3 recipes, trained on T196 rows only): a T196 main uses the fold model that held it out (runs36 by main); p311 and P_new
  mains are fold -1 (no step-3 model trained on them: 09's original section 5 gate, which stands for T196-only probes @ 15d68afc3).
- Held-out gate by key with positive and negative controls; census of mapped vs fold -1 cells per pool.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGEX_PIN = ("9cb3d0291", "experiments/2026-10-15_q3_sp493/results/sp493_regex_test_2026-10-15_171735.json", "f4ae9ca2677cce27")
FACTS_PIN = ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json", "07caa527db89ec51")
NAMES_PIN = ("69a1d211f", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/sp493_staged_set_names_2026-10-16__r3.json", "88313a80728988fef4c5")  # names r3 (4f r4)
T196_PIN = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
REGEX_V2_PIN = ("b9e0b6eda", "experiments/2026-10-14_priority1_spec_09/sp493_sim_regex_v2_test.pl", "dab0dec69172912f")  # 09 00:58: sim_attack v2
REGEX_V2_OUT = ("b9e0b6eda", "experiments/2026-10-14_priority1_spec_09/sp493_sim_regex_v2_test.out", "ad154a67fae33a16")
R36_PIN = ("c24106ca0", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json", "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")
SIM_V1 = r"^sp493_g[0-9]+_sim(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack$"  # replaced by v2; the pinned v1 file still carries it
# 09 00:58 tail assert inputs (relayed by 4f 01:1x, shas verified): the L1/L2 split (moved = g6 u g7), L3's 37 references, and the X / Y
# tail task files with their per-row group map
TAIL_SPLIT_PIN = ("59f333d03", "experiments/2026-10-15_q3_interleaved_run3_sim2/results/sp493_sim_split_2026-10-15.json", "5444cc11b4a6d6520fdb")
L3_PIN = ("627400b22", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/sp493sim/refindex_L3.json", "7f98a7de46d2")
TAIL_TASK_PINS = {"x": ("a400f9af0", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/sp493sim/tasks_sp493_tail_X.json", "26c73723b4cf10b070b5"),
                  "y": ("a400f9af0", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/sp493sim/tasks_sp493_tail_Y.json", "e0192c9d0622844ead59")}
GROUP = {"g1": "L1", "g6": "L1", "g2": "L2", "g7": "L2", "g3": "L3"}  # 09: recipe grouping per row (g1 u g6, g2 u g7, g3)
REGEX = {"real_main": r"^sp493_(?:a|b|c)_main_v1(?:_try[0-9]+)?__(?:attack|honest_a|honest_b)$",  # 09 @ 9cb3d0291, verbatim
         "real_run2": r"^sp493_(?:a|b|c)_run2_v1(?:_try[0-9]+)?__attack$",
         "sim_attack": r"^sp493_g[0-9]+_sim(?:_[xy])?(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack$",  # 09's v2 @ b9e0b6eda (x/y = the tail launch)
         "sim_honest": r"^sp493_g[0-9]+_simh(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$"}
T987_REASON = ("t987_model_registry: no_usable_reference (refindex_L3 @ 627400b22); out of the S_paper493 attack cohort for every row "
               "(09 READ_LOG @ 65e738c7a / 4e173ffb9)")
EXCLUDED_KEYS = {"run2|t615_media_processing_r2": "malformed text_editor call (09 00:2x): b_run2 excludes it by key"}
WORLD = {"real_main": {"attack": "run1", "honest_a": "honest_a", "honest_b": "honest_b"}, "real_run2": {"attack": "run2"}}
STRATUM = {"real_main": "S_paper493", "real_run2": "S_paper493", "sim_attack": "S_paper493_sim", "sim_honest": "S_paper493_simh"}
P_NEW_LABEL = "P_new mains are new to T196 / +311 / nm360 (not to Sol #36); step-3 recipes never trained on them: fold -1, all five models"


def _git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def pinned(pin, raw=False):
    c, p, h = pin
    b = _git("show", f"{c}:{p}")
    if not hashlib.sha256(b).hexdigest().startswith(h):
        raise SystemExit(f"{p} @ {c}: sha256 differs from its pin")
    return b.decode() if raw else json.loads(b)


def load():
    """Every pinned input, checked; the regexes must equal the pinned test file's, verbatim."""
    rx = pinned(REGEX_PIN)
    if rx["regex"] != REGEX | {"sim_attack": SIM_V1} or rx.get("pass") is not True:
        raise SystemExit("09's regex file differs from the regexes here, or its test did not pass")
    if f"qr/{REGEX['sim_attack']}/" not in pinned(REGEX_V2_PIN, raw=True) or not pinned(REGEX_V2_OUT, raw=True).rstrip().endswith("failures: 0"):
        raise SystemExit("09's sim_attack v2 regex is not verbatim in its pinned test, or that test failed")
    names = pinned(NAMES_PIN)
    refused = {x["staged_set"] for k in ("superseded", "burned_or_taken", "superseded_v1_launches") for x in names[k]} | set(names["non_routable"]["names"])
    facts = {r["key"]: r for r in pinned(FACTS_PIN)["rows"]}
    t196 = {c["key"]: c for c in pinned(T196_PIN)["cells"]}
    fom = {}
    for r in pinned(R36_PIN):
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    moved = {r["key"]: r["group"] for r in pinned(TAIL_SPLIT_PIN)["rows"] if r["group"] in ("g6", "g7")}
    tail_group = moved | {k: "g3" for k in pinned(L3_PIN)["references"]}
    tail_launch = {}
    for x, pin in TAIL_TASK_PINS.items():
        t = pinned(pin)
        if set(t["groups"]) != {c["key"] for c in t["cells"]} or any(tail_group.get(k) != g for k, g in t["groups"].items()):
            raise SystemExit(f"tail task file {x}: its group map disagrees with the split list / L3")
        for k in t["groups"]:
            if tail_launch.setdefault(k, x) != x:
                raise SystemExit(f"{k}: in both tail launches")
    if set(tail_launch) != set(tail_group):
        raise SystemExit("the tail task files do not cover exactly the split list's moved keys plus L3's 37")
    return {"refused_sets": refused, "facts": facts, "t196": t196, "r36": fom, "t196_mains": {c["main"] for c in t196.values()},
            "tail_group": tail_group, "tail_launch": tail_launch}


def kind_of(folder):
    ks = [k for k, rx in REGEX.items() if re.fullmatch(rx, folder)]
    return ks[0] if len(ks) == 1 else None


def refuse_folder(folder, P):
    """None if admitted, else the reason (09's _try rule AND d9's list, failing closed)."""
    k = kind_of(folder)
    if k is None:
        return "outside 09's S_paper493 grammar"
    ss = folder.split("__")[0]
    if ss in P["refused_sets"]:
        return f"{ss}: on d9's names record (superseded / burned / v1 launch / non-routable)"
    t = re.search(r"_try([0-9]+)$", ss)
    n = int(t.group(1)) if t else 1
    if k in ("real_main", "real_run2") and n < 3:
        return f"{ss}: real arms need _try >= 3 (09 @ 1243f1e27)"
    if re.match(r"^sp493_g[12]_sim_s[0-9]+_v1", ss) and n < 2:
        return f"{ss}: g1/g2 snapshots need _try >= 2"
    return None


def world_of(folder, arm_world):
    """The scorer's world: real arms from the folder's arm token; sp493_sim -> sim (09 Q1); sim-honest keeps sp493_t196 / sp493_new."""
    k = kind_of(folder)
    if k in WORLD:
        return WORLD[k][folder.split("__")[1]]
    if k == "sim_attack":
        if arm_world != "sp493_sim":
            raise SystemExit(f"{folder}: a sim-attack row outside world sp493_sim")
        return "sim"
    if arm_world not in ("sp493_t196", "sp493_new"):
        raise SystemExit(f"{folder}: a sim-honest world other than sp493_t196 (g4) or sp493_new (g5)")
    return arm_world


def admit_cell(world, cell, P):
    """(main, fold, None) if admitted, else (None, None, reason). Step-3 folds: T196 mains held out by runs36; others fold -1."""
    if cell.startswith("t987_"):
        return None, None, T987_REASON
    if f"{world}|{cell}" in EXCLUDED_KEYS:
        return None, None, EXCLUDED_KEYS[f"{world}|{cell}"]
    if world == "sp493_t196":  # g4: T196 task keys, admitted iff (env, main) is an S_paper493 main of origin t196 (09 00:0x)
        c = P["t196"].get(cell)
        ok = c is not None and any(r["main_origin"] == "t196" and (r["env"], r["main"]) == (c["env"], c["main"]) for r in P["facts"].values())
        if not ok:
            return None, None, "g4: not a T196 task whose (env, main) is an S_paper493 main of origin t196"
        main = c["main"]
    else:
        r = P["facts"].get(cell)
        if r is None:
            return None, None, "not in the S_paper493 cell list (facts @ d800edc6a)"
        main = r["main"]
        if (r["main_origin"] == "t196") != (main in P["t196_mains"]):
            raise SystemExit(f"{cell}: facts main_origin disagrees with the T196 pin")
        if r["main_origin"] == "t196" and r["r36_main_fold"] != P["r36"].get(main):
            raise SystemExit(f"{cell}: facts r36_main_fold disagrees with the runs36 pin")
    return main, (P["r36"][main] if main in P["t196_mains"] else -1), None


def heldout_violations(rows, P):
    """rows: [(main, fold)]. A step-3 fold-f model trained on the T196 tasks whose main's runs36 fold is not f. Violation: f >= 0 and the
    main is in f's training (f != its runs36 fold, or not a T196 main); f == -1 and the main is a T196 main (a model trained on it)."""
    bad = []
    for main, f in rows:
        t = main in P["t196_mains"]
        if (f >= 0 and (not t or P["r36"].get(main) != f)) or (f < 0 and t):
            bad.append((main, f))
    return bad


def heldout_gate(rows, P):
    """The by-key gate plus its controls (both must behave, or nothing is relayed)."""
    m = sorted(P["t196_mains"])[0]
    pnew = next(r["main"] for r in P["facts"].values() if r["main_origin"] == "new")
    controls = {"positive_t196_heldout": not heldout_violations([(m, P["r36"][m])], P), "negative_t196_wrong_fold": bool(heldout_violations([(m, (P["r36"][m] + 1) % 5)], P)),
                "positive_pnew_fold_minus1": not heldout_violations([(pnew, -1)], P), "negative_t196_fold_minus1": bool(heldout_violations([(m, -1)], P)),
                "negative_pnew_mapped": bool(heldout_violations([(pnew, 0)], P))}
    v = heldout_violations(rows, P)
    return {"controls": controls, "violations": v[:5], "n_violations": len(v), "pass": all(controls.values()) and not v}


def census(cells_by_pool):
    """{pool: [(main, fold)]} -> mapped vs fold -1 cells per pool (units are counted in the scorer's manifest per probe x readout)."""
    return {p: {"mapped": sum(f >= 0 for _, f in xs), "fold_minus1": sum(f < 0 for _, f in xs)} for p, xs in sorted(cells_by_pool.items())}


def one_set_per_cell(sets):
    """09 00:2x and 00:58: per recipe group (g1 u g6 = L1, g2 u g7 = L2, g3 = L3), each sim-attack cell is in exactly ONE admitted set,
    across the x and y tail launches; checked across all groups at once (grouping is per row, so a cell in two groups also STOPs).
    {staged set: {world|cell}} -> the cells found in two sets."""
    seen, dup = {}, []
    for ss, cells in sorted(sets.items()):
        g = re.match(r"^sp493_(g[0-9]+)_sim(?:_[xy])?(?:_s[0-9]+)?_v1", ss)
        if g and g.group(1) not in GROUP:
            raise SystemExit(f"{ss}: a sim-attack group outside 09's map (g1, g2, g3, g6, g7)")
        for c in cells if g else []:
            if c in seen:
                dup.append((c, seen[c], ss))
            seen.setdefault(c, ss)
    return dup


def tail_assert(sets, P):
    """09's per-set half, at EVERY launch (01:2x): over every admitted g3/g6/g7 tail set ({staged set: {world|cell}}) each cell's group
    (the folder's g<k>) and launch token (x/y) equal the pinned per-row map (no cell in two sets is one_set_per_cell's). A partial
    release is allowed. Returns the problems (STOP if any). The union half is tail_union_assert, run before any tail row is read."""
    bad = []
    for ss, cells in sorted(sets.items()):
        m = re.match(r"^sp493_(g3|g6|g7)_sim_([xy])(?:_s[0-9]+)?_v1", ss)
        if re.match(r"^sp493_g(?:3|6|7)_sim", ss) and not m:
            bad.append(f"{ss}: a tail set without its x/y launch token")
            continue
        for wc in cells if m else []:
            c = wc.split("|", 1)[1]
            if P["tail_group"].get(c) != m.group(1) or P["tail_launch"].get(c) != m.group(2):
                bad.append(f"{ss}: {c} is pinned to {P['tail_group'].get(c)}/{P['tail_launch'].get(c)}")
    return bad


def tail_union_assert(admitted, classed, P):
    """09's union half, the FINAL assert before any tail row is READ (01:2x): admitted tail cells (by cell key) u cells listed by key
    with an exclusion class (setup failure, errored run, staging exclusion, no clean donor, ...) == the 295 pinned cells. Every gap
    must be classed; a cell neither admitted nor classed, a classed cell that was also admitted, an empty class, or a cell outside the
    295 = STOP. Returns the problems."""
    pinned_cells, a, c = set(P["tail_group"]), set(admitted), dict(classed)
    bad = [f"{k}: an empty exclusion class" for k, v in sorted(c.items()) if not isinstance(v, str) or not v.strip()]
    bad += [f"{k}: both admitted and classed ({c[k]})" for k in sorted(a & set(c))]
    bad += [f"{k}: outside the 295 pinned tail cells" for k in sorted((a | set(c)) - pinned_cells)]
    bad += [f"{k}: neither admitted nor classed" for k in sorted(pinned_cells - a - set(c))]
    return bad


def sample_error_ok(manifest_files, admitted_keys):
    """09 / d9's revised manifests: EVERY row carries sample_error (a missing field refuses); every ADMITTED row's is null (identity
    test); non-null is allowed only on rows the staging excluded. Keys are "<launch>/<rel>#e0"."""
    keys = {f"{f['launch']}/{f['rel']}#e0" for f in manifest_files}
    return set(admitted_keys) <= keys and all("sample_error" in f for f in manifest_files) and \
        all(f["sample_error"] is None for f in manifest_files if f"{f['launch']}/{f['rel']}#e0" in set(admitted_keys))
