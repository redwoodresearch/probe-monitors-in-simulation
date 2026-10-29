"""Amendment 45: by-key check of one sp493 DT root after its launch (keys, folds and counts only; no score is read). Inputs: the job's local result
record, each folder's manifest_dt.json (fetched read-only from the work Volume), the routing file pinned in 4f's release list, the held-out gate
record and the cell list. Checks: the scored runs are exactly the routing's staged joins (with 09's world map); every run's fold equals the gate's
routing for its cell (sp493 cells) or its main (g4 T196 keys); the census and world map agree with the runs; stratum, world and label; every
listed npz exists in the folder listing. Expected fold of a run: the DT fit's fold of its (env, main) from rows.json (pinned by the cross-fit
record's sha), else -1. Usage: python3 -I check_sp493_records.py <repo> <fitcheck dir> <result json> <routing json> <release json> <out json>
<folder>=<manifest_dt.json>=<listing txt> [...]"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO, FC, RES, RTF, REL, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6])
FOLDERS = [a.split("=") for a in sys.argv[7:]]
HERE = "experiments/2026-10-14_p2_dt_scaleup"
show = lambda s: subprocess.run(["git", "-C", REPO, "show", s], capture_output=True, check=True).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
rb = RTF.read_bytes()
rel = {x["routing_file"]: x for x in json.loads(REL.read_bytes())["released"]}
r = json.loads(rb)
gate = json.loads(show(f"HEAD:{HERE}/results/dt_sp493_heldout_gate_2026-10-15_235329.json"))
cells = {c["key"]: c for c in json.loads(show(f"HEAD:{HERE}/config/sp493_cells_v1.json"))["cells"]}
p_, c_ = (y.strip() for y in r["manifest"].split(" @ "))
files = json.loads(show(f"{c_}:{p_}"))["files"]
arms, excl = {a["meta_arm"] for a in r["arms"]}, r.get("staging_exclusions") or {}
wmap = {"sp493_sim": "sim"}
staged = {f"{wmap.get(f['world'], f['world'])}|{f['task']}" for f in files if f["arm_dir"] in arms and f"{f['launch']}/{f['rel']}#e0" not in excl
          and f.get("attempt_kind") != "reference"}  # Amendment 45e: donor reference rows are never staged or scored here
res = json.loads(RES.read_text())
t196 = {c["key"]: c for c in json.loads(show("HEAD:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"))["cells"]}
xf = json.loads(show(f"HEAD:{HERE}/results/dt_training_crossfit_check_2026-10-15_194337.json"))["designs"]["destructive_primary"]
rraw = (FC / "destructive_primary.rows.json").read_bytes()
assert sha(rraw) == xf["rows_sha256"], "fit rows.json differs from its pin"
rj = json.loads(rraw)
coh = sorted(set(t196) - set(rj["summary"]["excluded_tasks"]))
fitfold = {(t196[coh[t]]["env"], t196[coh[t]]["main"]): f for t, f in zip(rj["task"], rj["fold"])}
main_of = {k: (c["env"], c["main"]) for k, c in {**t196, **cells}.items()}
out, runs_all, bad = {"folders": {}}, {}, []
for name, mf, lst in FOLDERS:
    m = json.loads(Path(mf).read_text())
    listing = {x.strip("/").split("/")[-1] for x in Path(lst).read_text().split()}
    runs = m["runs"]
    runs_all |= runs
    sp = m.get("sp493", {})
    exp_fold = {j: fitfold.get(main_of.get(j.split("|")[1]), -1) for j in runs}
    folds_bad = [j for j, v in runs.items() if v["fold"] != exp_fold[j] or (v.get("main") is not None and (t196.get(j.split("|")[1]) or cells.get(j.split("|")[1]) or {}).get("main") != v["main"])]
    census = sp.get("census", {})
    fk = [v["fold"] for v in runs.values()]
    chk = {"world": m["d9_arm_to_world"], "label": sp.get("label"), "family": sp.get("family"), "pool": sp.get("pool"), "runs": len(runs), "rows": m["rows"],
           "strata": sorted({v["stratum"] for v in runs.values()}), "folds_vs_gate_bad": folds_bad,
           "census_matches_runs": census.get("runs_mapped") == sum(f >= 0 for f in fk) and census.get("runs_fold_minus_1") == sum(f < 0 for f in fk)
           and census.get("calls_mapped", 0) + census.get("calls_fold_minus_1", 0) == sum(v["calls"] for v in runs.values()) == m["rows"],
           "world_map": sp.get("world_map"), "npz_listed_present": all(Path(v["path"]).name in listing for v in m["files"].values()),
           "manifest_in_listing": "manifest_dt.json" in listing, "preflight_routing_sha256": (m.get("preflight") or {}).get("routing_pin", {}).get("routing_sha256"),
           "git": m.get("git"), "stamp": m.get("stamp (PT)")}
    out["folders"][name] = chk
ok = set(runs_all) == staged and all(not v["folds_vs_gate_bad"] and v["census_matches_runs"] and v["npz_listed_present"] and v["manifest_in_listing"]
                                     and v["preflight_routing_sha256"] == rel[RTF.name]["routing_sha256"] == sha(rb) for v in out["folders"].values())
out |= {"what": "by-key check of one sp493 DT root (keys, folds, counts; no score read)", "written_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S"),
        "routing": {"file": RTF.name, "sha256": sha(rb), "release_pin_matches": rel[RTF.name]["routing_sha256"] == sha(rb), "manifest": r["manifest"]},
        "result_record": RES.name, "staged_joins": len(staged), "scored_runs": len(runs_all), "scored_equals_staged": set(runs_all) == staged,
        "missing": sorted(staged - set(runs_all))[:5], "extra": sorted(set(runs_all) - staged)[:5], "pass": ok}
OUT.open("x").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
