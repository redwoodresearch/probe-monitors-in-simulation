"""Amendment 45: 09 Append 39 amendment 1 section 2's held-out gate and census for DT on S_paper493 (keys, folds and counts only; no score).
DT's fold models (dt_attack_fits_00f4994d6c1e, 4 designs) were fit by run_score_onpolicy.py @ 79cefaf7: model k trains, selects its layer and C
(inner CV) and fits its standardization on rows of folds T_k = {0..4} - {k, k+1 mod 5} only (lines 238-276). Rows: rows.json (task = index into
the sorted fit cohort = T196 keys minus summary.excluded_tasks). Per sp493 cell, by (env, main): MAPPED to DT's fold of that main if the main is in
the fit's rows (it must equal the job's scaleup_folds value), else fold -1. Gate: a mapped main is in no row of its designated model's T_k; a fold -1
main is in no row at all. Controls in the same code: positive (every fit-cohort T196 main passes at its own fold), negative (the same mains at
fold k+1, and mapped sp493 mains treated as fold -1, must fail every time). Census: mapped vs fold -1 cells and mains per design x readout x pool.
Usage: python3 -I check_sp493_heldout.py <repo> <fitcheck dir> <out json> [<rev of this folder's files: HEAD; "" = the index>]"""
import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO, FC, OUT, REV = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), (sys.argv[4] if len(sys.argv) > 4 else "HEAD")
HERE = "experiments/2026-10-14_p2_dt_scaleup"
sys.path.insert(0, str(Path(REPO) / HERE))
import dt_q4b_lib  # noqa: E402  the job's own fold rule (scaleup_folds)

show = lambda s: subprocess.run(["git", "-C", REPO, "show", s], capture_output=True, check=True).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
DESIGNS = ("destructive_primary", "transfer_primary", "destructive_secondary", "transfer_secondary")
tasks_raw = show("HEAD:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json")
assert sha(tasks_raw) == "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb", "T196 tasks pin"
tasks = json.loads(tasks_raw)
t196 = {c["key"]: (c["env"], c["main"]) for c in tasks["cells"]}
xf = json.loads(show(f"{REV}:{HERE}/results/dt_training_crossfit_check_2026-10-15_194337.json"))["designs"]
cl_raw = show(f"{REV}:{HERE}/config/sp493_cells_v1.json")
cells = json.loads(cl_raw)["cells"]
facts_raw = show("d800edc6a:experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json")
assert sha(facts_raw) == "07caa527db89ec519314486d8c529c37430ca41d773a8bce26c041eb141b0a57", "facts pin"
facts = {r["key"]: r for r in json.loads(facts_raw)["rows"]}
assert len(cells) == 492 and all((facts[c["key"]]["env"], facts[c["key"]]["main"], facts[c["key"]]["main_origin"]) == (c["env"], c["main"], c["main_origin"]) for c in cells)
T = lambda k: {f for f in range(5) if f not in (k, (k + 1) % 5)}  # noqa: E731  run_score_onpolicy.py @ 79cefaf7 l.240-241


def fit_view(des: str) -> dict:
    raw = (FC / f"{des}.rows.json").read_bytes()
    assert sha(raw) == xf[des]["rows_sha256"], (des, "rows.json differs from the cross-fit record's pin")
    r = json.loads(raw)
    cohort = sorted(set(t196) - set(r["summary"]["excluded_tasks"]))
    assert len(cohort) == max(r["task"]) + 1 == len(set(r["task"])), des
    fold_of = collections.defaultdict(set)
    for t, f in zip(r["task"], r["fold"]):  # every row of every world: a superset of what any model trained, selected or standardized on
        fold_of[t196[cohort[t]]].add(f)
    assert all(len(v) == 1 for v in fold_of.values()), (des, "a main in two folds")
    return {"fold": {m: next(iter(v)) for m, v in fold_of.items()}, "excluded_tasks": sorted(r["summary"]["excluded_tasks"]), "rows_sha256": xf[des]["rows_sha256"]}


def trained_on(view: dict, main: tuple, k: int) -> bool:
    """True if model k (k = -1: any of the five models) saw a row of this main."""
    f = view["fold"].get(main)
    return f is not None and (k == -1 or f in T(k))


views = {des: fit_view(des) for des in DESIGNS}
assert all(v["fold"] == views[DESIGNS[0]]["fold"] for v in views.values()), "designs differ in their fold map"
scale = dt_q4b_lib.scaleup_folds(tasks, cells)  # the job's rule: the main's frozen T196 fold, -1 off T196
route, disagree = {}, []
for c in cells:
    m = (c["env"], c["main"])
    f = views[DESIGNS[0]]["fold"].get(m, -1)
    route[c["key"]] = f
    if f >= 0 and scale[c["key"]] != f:
        disagree.append(c["key"])
override = sorted(k for k in route if route[k] == -1 and scale[k] >= 0)  # a T196 main outside the fit cohort: the job must set -1
sp_t196 = {(c["env"], c["main"]) for c in cells if c["main_origin"] == "t196"}
g4t = {k: views[DESIGNS[0]]["fold"].get(m, -1) for k, m in t196.items() if m in sp_t196}  # 09's g4 rule (~23:5x): T196 task keys on sp493 T196-origin mains
scale4 = dt_q4b_lib.scaleup_folds(tasks, [{"key": k, "env": t196[k][0], "main": t196[k][1]} for k in g4t])
disagree += sorted(k for k, f in g4t.items() if f >= 0 and scale4[k] != f)
gate = {des: {"mapped_main_in_designated_training": sorted(c["key"] for c in cells if route[c["key"]] >= 0 and trained_on(v, (c["env"], c["main"]), route[c["key"]])),
              "fold_minus_1_main_in_any_row": sorted(c["key"] for c in cells if route[c["key"]] == -1 and trained_on(v, (c["env"], c["main"]), -1)),
              "g4_t196_keys_in_designated_training": sorted(k for k, f in g4t.items() if (f >= 0 and trained_on(v, t196[k], f)) or (f < 0 and trained_on(v, t196[k], -1)))}
        for des, v in views.items()}
fitm = sorted(views[DESIGNS[0]]["fold"])
ctrl = {des: {"positive_t196_mains_own_fold_held_out": sum(not trained_on(v, m, v["fold"][m]) for m in fitm), "of": len(fitm),
              "negative_t196_mains_at_fold_k_plus_1_detected": sum(trained_on(v, m, (v["fold"][m] + 1) % 5) for m in fitm),
              "negative_mapped_sp493_as_fold_minus_1_detected": sum(trained_on(v, (c["env"], c["main"]), -1) for c in cells if route[c["key"]] >= 0),
              "mapped_sp493": sum(route[c["key"]] >= 0 for c in cells),
              "models_trained_per_t196_main": dict(collections.Counter(sum(trained_on(v, m, k) for k in range(5)) for m in fitm))} for des, v in views.items()}
pools = {}
B, P = "origin/aswin/q3-sp493", "experiments/2026-10-15_q3_sp493"
for q in "ABC":
    ks = {c["key"] for c in json.loads(show(f"{B}:{P}/config/tasks_sp493_box{q}.json"))["cells"]}
    pools[f"real box {q.lower()} (sp493_{q.lower()}_main: run1/honest_a/honest_b; sp493_{q.lower()}_run2: run2)"] = ks
for g, f in (("g1", "L1"), ("g2", "L2")):
    pools[f"sim {g} (sp493_{g}_sim, task file {f} @ 3fc2cb00c)"] = {c["key"] for c in json.loads(show(f"3fc2cb00c:experiments/2026-10-15_q3_interleaved_run3_sim2/config/sp493sim/tasks_sp493_sim_{f}_in.json"))["cells"]}
pools["sim g3 (sp493_g3_sim: the cohort minus L1 and L2; its task file is not committed yet)"] = set(route) - set().union(*[v for k, v in pools.items() if k.startswith("sim g")])
cohort = set(route)
census = {f"{des} | DT call-level (q4b layout; one readout) | sim-honest g4 T196 task keys (09's g4 rule)": {
    "t196_keys_admissible": len(g4t), "mapped": sum(f >= 0 for f in g4t.values()), "fold_minus_1": sum(f < 0 for f in g4t.values()),
    "mains": len({t196[k] for k in g4t})} for des in DESIGNS}
for des in DESIGNS:
    for pname, ks in pools.items():
        ks_ = ks & cohort
        census[f"{des} | DT call-level (q4b layout; one readout) | {pname}"] = {
            "cells": len(ks_), "mapped_cells": sum(route[k] >= 0 for k in ks_), "fold_minus_1_cells": sum(route[k] == -1 for k in ks_),
            "mapped_mains": len({(facts[k]["env"], facts[k]["main"]) for k in ks_ if route[k] >= 0}), "fold_minus_1_mains": len({(facts[k]["env"], facts[k]["main"]) for k in ks_ if route[k] == -1}),
            "mapped_fold_counts": dict(sorted(collections.Counter(route[k] for k in ks_ if route[k] >= 0).items())), "cells_outside_cohort": sorted(ks - cohort)}
by_origin = {o: {"cells": sum(c["main_origin"] == o for c in cells), "mapped": sum(c["main_origin"] == o and route[c["key"]] >= 0 for c in cells),
                 "fold_minus_1": sum(c["main_origin"] == o and route[c["key"]] == -1 for c in cells)} for o in ("t196", "p311", "new")}
r36 = collections.Counter("agree" if route[c["key"]] == facts[c["key"]]["r36_main_fold"] else "differ" for c in cells if route[c["key"]] >= 0)
ok = not disagree and all(not g["mapped_main_in_designated_training"] and not g["fold_minus_1_main_in_any_row"] and not g["g4_t196_keys_in_designated_training"] for g in gate.values()) \
    and all(c["positive_t196_mains_own_fold_held_out"] == c["of"] == c["negative_t196_mains_at_fold_k_plus_1_detected"]
            and c["negative_mapped_sp493_as_fold_minus_1_detected"] == c["mapped_sp493"] > 0 for c in ctrl.values())
out = {"what": "DT on S_paper493: held-out gate and mapped / fold -1 census (09 Append 39 am.1 section 2; Amendment 45); keys, folds and counts only, no score read",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"), "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
       "inputs": {"cells": f"{HERE}/config/sp493_cells_v1.json (sha256 {sha(cl_raw)})", "facts": "d800edc6a (07caa527...)", "fit_rows": {d: v["rows_sha256"] for d, v in views.items()},
                  "fit_rule": "run_score_onpolicy.py @ 79cefaf7 l.238-276: model k trains, selects and standardizes on folds {0..4} - {k, k+1 mod 5}",
                  "fit_excluded_tasks": views[DESIGNS[0]]["excluded_tasks"]},
       "routing": {"rule": "DT's own fold of the main (the model that held it out) when the main is in the fit's rows; else fold -1 (no DT model saw it)",
                   "by_origin": by_origin, "job_rule_disagrees_with_fit_fold": disagree, "t196_mains_outside_fit_cohort_job_must_set_minus_1": override,
                   "r36_main_fold_vs_dt_fold_on_mapped_cells": dict(r36), "g4_t196_keys": {k: f for k, f in sorted(g4t.items())}},
       "gate": gate, "controls": ctrl, "census": census, "pass": ok}
OUT.open("x").write(json.dumps(out, indent=1))
print(json.dumps({"pass": ok, "by_origin": by_origin, "override": override, "disagree": disagree[:5], "r36": dict(r36), "controls": ctrl[DESIGNS[0]],
                  "census_d0": {k.split(" | ")[2][:22]: [v.get(x) for x in ("cells", "t196_keys_admissible", "mapped_cells", "mapped", "fold_minus_1_cells", "fold_minus_1") if x in v]
                                for k, v in census.items() if k.startswith(DESIGNS[0])}}, indent=0))
