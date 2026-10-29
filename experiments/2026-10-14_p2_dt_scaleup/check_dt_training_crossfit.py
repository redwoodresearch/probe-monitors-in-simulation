"""09's primary_comparator check (keys and counts only; no score values): for each DT design's fit (dt_attack_fits_00f4994d6c1e/<design>/
rows.json and main.json, read-only from q3onpol_score), (a) which arms (worlds) enter training and how many epochs each, (b) whether folds are
assigned by MAIN task, whether every DT-scored T196 row was scored by the fold model whose held-out fold holds its main (0 violations), and
whether any +311 / nm360 / IL-new main enters training or is scored on a fold >= 0.
Usage: python3 -I check_dt_training_crossfit.py <fitcheck dir> <wt> <out json> <manifest dir> [<manifest dir> ...]"""
import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

FC, WT, OUT, MDIRS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), [Path(x) for x in sys.argv[4:]]
DESIGNS = ("destructive_primary", "transfer_primary", "destructive_secondary", "transfer_secondary")
show = lambda s: subprocess.run(["git", "-C", str(WT), "show", s], capture_output=True, check=True).stdout  # noqa: E731
T196 = json.loads(show("HEAD:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"))["cells"]
C311 = json.loads(show("0caadbfb5:experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json"))["cells"]
NMV1 = json.loads(show("ef171c813:experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json"))["cells"]
t196_main = {c["key"]: c["main"] for c in T196}

other_mains = {c["main"] for c in C311} | {c["main"] for c in NMV1}
out = {"what": "DT fits: training arms and cross-fitting by main; DT-scored T196 rows vs their scoring fold; keys and counts only",
       "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"), "fits": "q3onpol_score:/dt_attack_fits_00f4994d6c1e/<design>/",
       "fit_manifest": "experiments/2026-10-12_dt_acting_monitors/results/manifest_dt_attack_fits_v1.json @ ad8b64a40", "designs": {}}
fold_of_main = {}
for des in DESIGNS:
    rraw, mraw = (FC / f"{des}.rows.json").read_bytes(), (FC / f"{des}.main.json").read_bytes()
    r, m = json.loads(rraw), json.loads(mraw)
    n = len(r["world"])
    cohort = sorted(set(t196_main) - set(r["summary"]["excluded_tasks"]))  # the fit's loader: tasks = sorted(cohort keys); "task" = index (run_score_onpolicy.py:479-480 @ 79cefaf7)
    assert len(cohort) == len(set(r["task"])) == max(r["task"]) + 1, (des, len(cohort), len(set(r["task"])))
    task_key = [cohort[i] for i in r["task"]]
    tr = [i for i in range(n) if r["in_design"][i] and r["world"][i] == "real"]  # the fit trains on real attack 1 rows in the design
    epochs = collections.defaultdict(set)
    for i in range(n):
        if r["in_design"][i]:
            epochs[r["world"][i]].add(task_key[i])
    mains_fold = collections.defaultdict(set)
    for i in range(n):
        mains_fold[t196_main[task_key[i]]].add(r["fold"][i])
    fm = {mn: next(iter(f)) for mn, f in mains_fold.items() if len(f) == 1}
    fold_of_main[des] = fm
    out["designs"][des] = {
        "rows_sha256": hashlib.sha256(rraw).hexdigest(), "main_json_sha256": hashlib.sha256(mraw).hexdigest(),
        "worlds_in_rows": dict(collections.Counter(r["world"])), "in_design_epochs_by_world": {w: len(v) for w, v in sorted(epochs.items())},
        "training_rows": len(tr), "training_worlds": sorted({r["world"][i] for i in tr}), "training_epochs_real": len({task_key[i] for i in tr}),
        "honest_worlds_present": sorted(w for w in set(r["world"]) if "honest" in w),
        "folds_by_main": {"mains": len(mains_fold), "mains_with_one_fold": len(fm), "mains_in_two_folds": sorted(mn for mn, f in mains_fold.items() if len(f) > 1)[:5],
                          "folds": sorted({f for v in mains_fold.values() for f in v})},
        "fold_models": sorted(int(k.split("|")[1]) for k in m["probes"]), "probe_training_worlds": sorted({k.split("|")[0] for k in m["probes"]}),
        "choice_keys": sorted(m["choices"]),
        "training_mains_in_311_or_nm360": len({t196_main[task_key[i]] for i in tr} & other_mains)}
# DT-scored rows: every manifest's runs (latest copy per folder name)
man = {}
for d in MDIRS:
    for f in d.glob("*.json"):
        if "__" in f.stem or f.stem.startswith("simh_full"):
            man[f.stem.replace(".manifest_dt", "")] = f
viol, t196_rows, non_t196_pos, non_t196_rows, missing_main = collections.Counter(), collections.Counter(), collections.Counter(), 0, collections.Counter()
no_fold = collections.Counter()
for fo, f in sorted(man.items()):
    mm = json.loads(f.read_text())
    for k, v in (mm.get("runs") or {}).items():
        cell = k.split("|", 1)[1]
        if not isinstance(v, dict) or "fold" not in v:
            no_fold[fo] += 1
            continue
        if cell in t196_main:
            for des in DESIGNS:
                fm = fold_of_main[des].get(t196_main[cell])
                if fm is None:
                    missing_main[des] += 1
                elif v["fold"] != fm:
                    viol[des] += 1
            t196_rows[fo] += v.get("calls", 0)
        else:
            non_t196_rows += v.get("calls", 0)
            if v["fold"] >= 0:
                non_t196_pos[fo] += 1
out["scored"] = {"manifests_read": len(man), "runs_without_a_fold_field": dict(no_fold) or 0, "t196_runs_folders": len(t196_rows), "t196_calls_scored": sum(t196_rows.values()),
                 "t196_runs_whose_manifest_fold_differs_from_the_fit_fold_of_their_main": dict(viol) or 0,
                 "t196_runs_whose_main_is_not_in_the_fit_rows": dict(missing_main) or 0,
                 "non_t196_calls_scored": non_t196_rows, "non_t196_runs_on_a_fold_ge_0": dict(non_t196_pos) or 0,
                 "scoring_rule": "run_dt_q4b.py _score_and_label: k = fold_of[task]; sc = byf[n, k] (only k >= 0); fold_of = dt_q4b_lib.scaleup_folds (a main's fold by frozen_folds' rule; outside T196: -1); per-row field: manifest_dt.json runs[key]['fold'] and the npz 'fold' (dt_q4b_lib.write_npz)"}
OUT.write_text(json.dumps(out, indent=1))
for des, v in out["designs"].items():
    print(des, "| probe keys' world", v["probe_training_worlds"], "| training worlds", v["training_worlds"], "| training epochs (real)", v["training_epochs_real"], "| in-design epochs by world", v["in_design_epochs_by_world"],
          "| mains", v["folds_by_main"]["mains"], "one fold each", v["folds_by_main"]["mains_with_one_fold"], "| fold models", v["fold_models"],
          "| training mains in +311/nm360:", v["training_mains_in_311_or_nm360"])
print("scored:", {k: v for k, v in out["scored"].items() if k != "scoring_rule"})
