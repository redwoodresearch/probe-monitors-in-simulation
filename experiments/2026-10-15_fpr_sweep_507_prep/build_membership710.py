"""S_newmains membership for the ~710 common comparison, BY RECORD (keys, labels and exclusion records only; no score is read).

Inputs (pinned): new_mains_cells_v1.json @ ef171c813; d9's nm360 per-batch label files and records @ f60fb6cff
(aswin/q3-overnight-chain), setup_failures v4 @ 396da80fd, stopped_runs v2 and unusable_honest @ 396da80fd, the S5r supersession
@ c3e9e3e95; the nm360 arm folders' scorer manifests (local copies, sha256 recorded). Rules: RELEASE_newmains.md @ 5ba67b471 + S4
release @ c35c2087a; S4 amendment 0b234d283 (nm109, nm126 dropped by key); nm290 excluded by key (09 @ 1bc718e98).
S4r (2e's Sol landing @ 1ac2de70d): the five 4r arm folders have Sol manifests only, so 4r cells and pool runs are listed
separately and enter Sol (a) and c_asis only; every key in the landing's NOTE status-row sidecars is a status row (never a pool
or cohort run), including honest_b|nm014 (0 scoreable rows, classification pending 09).

  python3 -I build_membership710.py --man-dir <dir of <folder>__<sol|p2|famexp|own_a>.json> --out config/membership710.json
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--man-dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
HERE = Path(__file__).resolve().parent
H = "experiments/2026-10-14_q3_overnight_chain/results/hygiene/"
PIN = {"cells": ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"),
       "setup_v4": ("396da80fd", H + "setup_failures_nm360_v4_2026-10-15.json", "5427019e4d60d63c1c1da0dc547256b0a9eafa8d6b2da5dddbeff24ee207230e"),
       "stopped_v2": ("396da80fd", H + "stopped_runs_nm360_v2_2026-10-15.json", None),
       "unusable": ("396da80fd", H + "unusable_honest_nm360_2026-10-15.json", None),
       "supersession": ("c3e9e3e95", H + "supersession_nm360_s5r_2026-10-15.json", None),
       "run_table": ("f60fb6cff", H + "run_table_nm360.json", None)}
LABEL_COMMIT = "f60fb6cff"
OWNA_GAP = {"nm360_s4d_sim_v1__sim_attack", "nm360_s5r_sim_v1__sim_attack", "nm360_s4r_main_v1__attack", "nm360_s4r_main_v1__honest_a",
            "nm360_s4r_main_v1__honest_b", "nm360_s4r_run2_v1__attack", "nm360_s4r_sim_v1__sim_attack"}  # own_a written by 4f (record @ 268e720d3)
S4R = ("1ac2de70d", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/scaleup507_",
       {"nm360_s4r_main_v1__attack": "2026-10-15_104121", "nm360_s4r_main_v1__honest_a": "2026-10-15_104125",
        "nm360_s4r_main_v1__honest_b": "2026-10-15_104129", "nm360_s4r_run2_v1__attack": "2026-10-15_104134",
        "nm360_s4r_sim_v1__sim_attack": "2026-10-15_104138"})
LABEL_FILES = ["s1", "s1_m1", "s1fu", "s2", "s2_m1", "s2fu", "s3", "s3fu", "s3fu_s1", "s4", "s4_m1", "s4dfu", "s4r", "s4rfu", "s5", "s5_m1", "s5_m2", "s5fu", "s5fu_s1", "s5rfu"]
EXCLUDE_FOLDERS = {"nm360_s1_main_v1__honest_a", "nm360_s1_main_v1__honest_b", "nm360_s2_main_v1__honest_a", "nm360_s2_main_v1__honest_b",
                   "nm360_s5_main_v1__honest_a", "nm360_s5_main_v1__honest_b"}  # staged, all no-policy-call, not scored (09)


def show(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=HERE).stdout


src = {}
recs = {}
for k, (c, p, s) in PIN.items():
    b = show(c, p)
    recs[k] = json.loads(b)
    src[k] = {"commit": c, "path": p, "sha256": hashlib.sha256(b).hexdigest()}
    if s and src[k]["sha256"] != s:
        raise SystemExit(f"{k}: sha256 differs from the pin")
cells = {c["key"]: c for c in recs["cells"]["cells"]}
labels_by_sha, lab_src = {}, {}
for lf in LABEL_FILES:
    p = H + f"labels_nm360_nm360-{lf}.json"
    b = show(LABEL_COMMIT, p)
    d = json.loads(b)
    lab_src[lf] = hashlib.sha256(b).hexdigest()
    for sha in d["staged_sets"]:
        if sha in labels_by_sha and labels_by_sha[sha][0] != lf:
            raise SystemExit(f"staged sha {sha[:12]} in two label files")
        labels_by_sha[sha] = (lf, d["labels"])
broken = {c for v in recs["setup_v4"]["cells"].values() for c in v}
excluded_runs, rec_launch = {}, {}  # rec_launch: the launch a stopped-run or unusable-honest record names (records are per launch)
for r in recs["setup_v4"]["records"]:
    excluded_runs[f"{r['world']}|{r['cell']}"] = "setup failure (v4 @ 396da80fd)"
for r in recs["stopped_v2"]["records"]:
    if r.get("excluded"):
        excluded_runs[f"{r['world']}|{r['cell']}"] = "stopped run (stopped_runs v2)"
        rec_launch[f"{r['world']}|{r['cell']}"] = r["launch"]
for r in recs["unusable"]["records"]:
    excluded_runs[f"{r['world']}|{r['cell']}"] = "unusable honest (submit-only; excluded from calibration pools)"
    rec_launch[f"{r['world']}|{r['cell']}"] = r["launch"]
excluded_runs["sim|nm290_web_scraping"] = "excluded by key: unknown class, not rerun-eligible (09 @ 1bc718e98)"  # cell key resolved below
nm290 = [c for c in cells if c.startswith("nm290_")]
if len(nm290) != 1:
    raise SystemExit("nm290 cell not unique")
excluded_runs.pop("sim|nm290_web_scraping")
excluded_runs[f"sim|{nm290[0]}"] = "excluded by key: unknown class, not rerun-eligible (09 @ 1bc718e98)"
s4r_src = {}
for arm, st in S4R[2].items():  # landing records (manifest sha) and NOTE status rows, by record
    rb, nb = show(S4R[0], f"{S4R[1]}{arm}_{st}.json"), show(S4R[0], f"{S4R[1]}{arm}_NOTE_status_rows.json")
    s4r_src[arm] = {"record_sha256": hashlib.sha256(rb).hexdigest(), "note_sha256": hashlib.sha256(nb).hexdigest(),
                    "manifest_sha256": json.loads(rb)["outputs"]["manifest_sha256"]}
    for k, v in json.loads(nb)["status_rows"].items():
        excluded_runs.setdefault(k, f"S4r status row (NOTE @ {S4R[0]}): {v['status']}")
superseded = {r["key"]: r for r in recs["supersession"]["rows"]}
folders, holder, status_rows = {}, defaultdict(list), {}
for f in sorted(Path(a.man_dir).glob("*__sol.json")):
    folder = f.name.removesuffix("__sol.json")
    if folder in EXCLUDE_FOLDERS:
        continue
    m = json.loads(f.read_bytes())
    (world,) = m["arm_to_world"].values()
    sha = m["dump"]["sha256"]
    if sha not in labels_by_sha:
        raise SystemExit(f"{folder}: staged sha not in any label file's staged_sets")
    lf, labs = labels_by_sha[sha]
    batch = "S4r" if "_s4r_" in folder else ("S4" if folder.startswith("nm360_s4") else folder.split("_")[1].upper())
    if batch == "S4r" and (folder not in S4R[2] or hashlib.sha256(f.read_bytes()).hexdigest() != s4r_src[folder]["manifest_sha256"]):
        raise SystemExit(f"{folder}: S4r manifest copy differs from the landing record")
    folders[folder] = {"world": world, "label_file": lf, "staged_sha256": sha, "batch": batch,
                       "manifests": {k: hashlib.sha256((Path(a.man_dir) / f"{folder}__{k}.json").read_bytes()).hexdigest()
                                     for k in ("sol", "p2", "famexp", "own_a") if (Path(a.man_dir) / f"{folder}__{k}.json").exists()}}
    wf = Path(a.man_dir) / f"{folder}__writer_famexp.json"  # own_a gap-fill (4f, code @ e3ba90fa0): its sidecar must name this own_a manifest
    if folder in OWNA_GAP and not wf.exists():
        raise SystemExit(f"{folder}: own_a gap-fill arm without its writer_famexp sidecar")
    if wf.exists():
        w = json.loads(wf.read_bytes())
        if w["manifest_owna_sha256"] != folders[folder]["manifests"].get("own_a"):
            raise SystemExit(f"{folder}: writer_famexp sidecar names a different own_a manifest")
        folders[folder]["own_a_writer_sidecar_sha256"] = hashlib.sha256(wf.read_bytes()).hexdigest()
    for key in m["folds"]:
        holder[key].append(folder)
# supersession: the replacement folder is authoritative for its keys; no fallback
for key, r in superseded.items():
    rep = [f for f in holder.get(key, []) if r["replacement"]["staged_set"] in f]
    if len(rep) != 1:
        raise SystemExit(f"{key}: replacement folder not unique ({holder.get(key)})")
    holder[key] = rep
s4r_rt = {r["cell"] for r in recs["run_table"]["cells"] if r.get("real_batch") == "nm360-s4r"}
s4d_rt = {r["cell"] for r in recs["run_table"]["cells"] if r.get("sim_launch") == "2026-10-14_newmains_s4d_fu"}
s4r_rt_cells = s4r_rt - s4d_rt  # 4r cells by the run table; each cell's runs come from its own main (S4 amendment item 1)
s4r_over = {}
for k, v in list(holder.items()):  # 4r cells: only the 4r folders hold their runs; original-S4 copies are never used (no fallback)
    if k.split("|", 1)[1] in s4r_rt_cells and any(folders[f]["batch"] != "S4r" for f in v):
        s4r_over[k] = v
        holder[k] = [f for f in v if folders[f]["batch"] == "S4r"]
        if not holder[k]:
            del holder[k]
dups = {k: v for k, v in holder.items() if len(v) > 1}
if dups:
    raise SystemExit(f"world|cell keys held by two folders: {list(dups.items())[:3]}")
key_folder = {k: v[0] for k, v in holder.items()}
batch_of_cell = {}
for k, f in key_folder.items():
    w, c = k.split("|", 1)
    if w in ("run1", "honest_a", "honest_b"):
        batch_of_cell.setdefault(c, folders[f]["batch"])
bad = sorted(c for c in s4r_rt_cells if batch_of_cell.get(c) not in (None, "S4r"))
if bad:
    raise SystemExit(f"4r cells whose real runs come from a non-4r folder: {bad[:5]}")
not_applied = {}
for k in list(excluded_runs):  # a stopped/unusable record for another launch never excludes the 4r rerun of the same key
    f = key_folder.get(k)
    if f and folders[f]["batch"] == "S4r" and k in rec_launch and "s4r" not in rec_launch[k]:
        not_applied[k] = f"{excluded_runs.pop(k)} for launch {rec_launch[k]}; not applied to the 4r run held by {f}"
label_of = {}
for k, f in key_folder.items():
    lf, labs = labels_by_sha[folders[f]["staged_sha256"]]
    label_of[k] = labs.get(k)
cohort, cohort_4r, pool_b, pool_b_4r, held_a, held_a_4r = [], [], [], [], [], []
for c in sorted(cells):
    if c in broken:
        status_rows[c] = "setup failure, no pair (setup_failures v4); nm109/nm126: no real reference run, dropped by key (S4 amendment 0b234d283)"
        continue
    is4r = c in s4r_rt_cells or batch_of_cell.get(c) == "S4r"
    ks = {w: f"{w}|{c}" for w in ("run1", "run2", "sim")}
    why = [f"{w}: {excluded_runs[k]}" for w, k in ks.items() if k in excluded_runs]
    why += [f"{w}: no scored row (no folder)" for w, k in ks.items() if k not in key_folder]
    why += [f"{w}: label {label_of.get(k)}" for w, k in ks.items() if k in key_folder and label_of.get(k) != "attempted"]
    if why:
        status_rows[c] = ("S4r: " if is4r else "") + "; ".join(why)
    else:
        (cohort_4r if is4r else cohort).append(c)
for k, f in key_folder.items():
    w, c = k.split("|", 1)
    if c in broken or k in excluded_runs:
        continue
    is4r = folders[f]["batch"] == "S4r"
    if w == "honest_b":
        (pool_b_4r if is4r else pool_b).append(k)
    elif w == "honest_a":
        (held_a_4r if is4r else held_a).append(k)
out = {"label": "S_newmains membership by record (keys only)", "sources": src, "label_files": {"commit": LABEL_COMMIT, "sha256": lab_src},
       "s4r_landing": {"commit": S4R[0], "arms": s4r_src, "keys_moved_to_4r_folder": s4r_over, "records_for_other_launches_not_applied": not_applied},
       "folders": folders, "folder_keys": {f: sorted(k for k, g in key_folder.items() if g == f) for f in sorted(set(key_folder.values()))}, "cohort": cohort, "cohort_without_S4": [c for c in cohort if batch_of_cell.get(c) != "S4"],
       "cohort_4r_sol_only": cohort_4r, "batch_of_cell": {c: batch_of_cell.get(c) for c in cohort + cohort_4r},
       "pool_honest_b": sorted(pool_b), "pool_honest_b_4r_sol_only": sorted(pool_b_4r),
       "held_out_honest_a": sorted(held_a), "held_out_honest_a_4r_sol_only": sorted(held_a_4r),
       "status_rows": status_rows, "excluded_runs": excluded_runs, "superseded_keys": sorted(superseded),
       "main_of": {c: cells[c]["main"] for c in cells}, "env_of": {c: cells[c]["env"] for c in cells},
       "strategy_of": {c: (cells[c].get("q2_approach_chars") or 0) > 0 for c in cells},
       "pool_rule": "reconstructed by key (flagged): honest_b of each cell's own main, after the exclusion records; the release names no S_newmains pool keys",
       "s4_rule": "S4 = S4d (original S4 main + s4d_fu) and S4r (4r main), union by cell; 'without S4' drops both from estimation only; the *_4r_sol_only lists (legacy name) enter every probe whose files cover all five 4r folders",
       "counts": {"cells": len(cells), "cohort": len(cohort), "cohort_without_S4": sum(batch_of_cell.get(c) != "S4" for c in cohort),
                  "cohort_4r_sol_only": len(cohort_4r), "by_batch": dict(Counter(batch_of_cell.get(c) for c in cohort)),
                  "pool_honest_b": len(pool_b), "pool_honest_b_4r_sol_only": len(pool_b_4r), "held_out_honest_a": len(held_a),
                  "held_out_honest_a_4r_sol_only": len(held_a_4r), "keys_moved_to_4r_folder": len(s4r_over), "records_not_applied_to_4r": len(not_applied),
                  "status_rows": len(status_rows), "status_by_reason": dict(Counter(v.split(":")[0].split(";")[0] for v in status_rows.values()))}}
Path(a.out).open("x").write(json.dumps(out, indent=1))
print(json.dumps(out["counts"], indent=1))
