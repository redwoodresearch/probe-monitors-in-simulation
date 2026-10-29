"""09's primary_comparator check for own_a (R = own_a_tight avg_top2, C = own_a_tight cmd_A), by key from committed records and code; no score
values are read. (a) training arms and counts per fold model; (b) folds by main task, in-fold violations of every scored row, +311 / nm360 mains
in any training set; in-sample rows. Usage (from fx): python3 -I <this> <runs36.json>"""
import collections
import glob
import hashlib
import json
import subprocess
import sys

git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True).stdout  # noqa: E731
X2E = "experiments/2026-10-14_q35_sol_avg_POSTHOC"
PINS = {"cb_record": ("cc53afc3a", f"{X2E}/results/correctedb_scores_2026-10-14_164934.json", "c002d5f822e49d7877f4e8b79a80ca9dd2e2f16c664b921d3ad12cb3da4c49ac"),
        "t196": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
        "plus311": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
        "new_mains": ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"),
        "owna507_record": ("54e18e68b", "experiments/2026-10-15_fpr_sweep_507_prep/results/owna507_2026-10-15_031024.json", "fa0c87be5d065842f6d4be58027ef8e5884b5ed5d53bc1a88ddf30d69366fe1a")}


def show(k):
    c, p, s = PINS[k]
    b = git("show", f"{c}:{p}")
    assert hashlib.sha256(b).hexdigest() == s, k
    return json.loads(b)


cb, t196, q2, nm, o507 = (show(k) for k in ("cb_record", "t196", "plus311", "new_mains", "owna507_record"))
r36b = open(sys.argv[1], "rb").read()
assert hashlib.sha256(r36b).hexdigest().startswith("76b11486cc353ab1eec3")
fom, conflicts = {}, 0
for r in json.loads(r36b):
    conflicts += fom.setdefault(r["main_task"], r["fold"]) != r["fold"]
main_of = {c["key"]: c["main"] for c in t196["cells"]}
code_git = cb["git"]
src = git("show", f"{code_git}:{X2E}/run_correctedb.py").decode().splitlines()
lines = lambda s: [i + 1 for i, x in enumerate(src) if s in x]  # noqa: E731
out = {"records": {k: f"{v[1]} @ {v[0]} (sha256 {v[2][:12]}…)" for k, v in PINS.items()} | {"r36": "runs36.json (sha256 76b11486…)", "code": f"{X2E}/run_correctedb.py @ {code_git[:9]} (the CB record's git)"},
       "code_lines": {"label rule y = run1 -> 1, honest_a -> 0, every other world NaN (never trained)": lines('y = np.where(rworld == "run1", 1.0, np.where(rworld == "honest_a", 0.0, np.nan))'),
                      "training rows: labelled runs with rfold != f": lines("np.flatnonzero(~np.isnan(y) & (rfold != f))"),
                      "scored rows: rfold == f only": lines("sc = np.flatnonzero(rfold[rid] == f)"),
                      "rfold from fold_of_task by the run's task": lines('rfold = np.array([fold_of_task[u.split("|", 1)[1]] for u in runs])'),
                      "R tight fits (cb_tight) / C fits (cb_cmd)": [lines("def cb_tight(")[0], lines("def cb_cmd(")[0]]},
       "fold_field": {"cb_record.folds": cb["folds"], "checked": "cb_record.per_run_folds (world|task -> fold) vs r36 main_task -> fold via T196 cell -> main"}}
prf = cb["per_run_folds"]
worlds = collections.Counter(k.split("|", 1)[0] for k in prf)
mism = sorted(k for k, f in prf.items() if fom.get(main_of.get(k.split("|", 1)[1])) != f)
out["(b) folds"] = {"r36 mains with two folds": conflicts, "runs in per_run_folds": len(prf), "by world": dict(sorted(worlds.items())),
                    "runs whose fold != r36 fold of their main": len(mism), "runs without a T196 main": sum(k.split("|", 1)[1] not in main_of for k in prf)}
train = {f: [k for k, g in prf.items() if g != f and k.split("|", 1)[0] in ("run1", "honest_a")] for f in range(5)}
out["(a) training runs per fold model (runs in the CB record; a run with 0 rows at a readout adds nothing)"] = {
    f"f{f}": dict(sorted(collections.Counter(k.split("|", 1)[0] for k in ks).items())) for f, ks in train.items()}
out["(a) arms never in training (scored only)"] = sorted(w for w in worlds if w not in ("run1", "honest_a"))
tm = {f: {main_of[k.split("|", 1)[1]] for k in ks} for f, ks in train.items()}
viol = [k for k, f in prf.items() if main_of[k.split("|", 1)[1]] in tm[f]]
out["(b) in-fold violations over every T196 run own_a scores (R and C use the same fold field and rule)"] = len(viol)
allm = set().union(*tm.values())
out["(b) +311 / nm360 mains in any own_a training set"] = {"plus311": len(allm & {c["main"] for c in q2["cells"]}), "nm360": len(allm & {c["main"] for c in nm["cells"]}),
                                                         "own_a training mains (union over folds)": len(allm)}
ov = o507["owna_training_overlap_all_scaleup_mains"]
out["(b) scale-up rows (2e's owna507 record)"] = {k: ov[k] for k in ("scale_up_mains", "owna_training_mains", "intersection")}
simh = []
for p in sorted(glob.glob("experiments/2026-10-14_p2_family_expansion/results/simh_scaleup_simh*_20*.json")):
    if "note" in p or "pins" in p:
        continue
    ev = json.load(open(p))["routing_evidence"]["own_a"]
    simh.append(ev["fold_minus1_excluded_by_all_five"])
out["(b) sim-honest scale-up rows (famexp records): own_a fold"] = {"records": len(simh), "cells all fold -1 (five models, none trained on the main)": sum(simh)}
out["IL"] = "IL S_new / S_newmains cells are +311 / nm360 mains (0 above). IL S_old (ilold_) cells are T196 mains: in own_a training for the folds != their r36 fold, so own_a on them is out of fold only if routed by that fold; famexp never scored own_a on IL rows."
out["in-sample own_a rows"] = {"T196 (2e's CB scoring)": len(viol), "scale-up (507 + sim-honest)": ov["intersection"]}
print(json.dumps(out, indent=1))
