"""nm360 rerun census (09 via d9, 10-15 ~10:35): every (launch, arm, task) with a kind == "rerun" attempt, from rerun_census.py's
per-box raw output (read-only; box3, box1, box4, sims-astra in that order; box2 powered off, not read), joined with the chain's staging
records (chain_stage_nm360_*.json c_gate.excluded on <chain ref>, read by `git show`): per run, the rerun attempt's key, eval_status and
clean flag; the counted attempt's (first or redraw; box_files.py rule: kind != "rerun" and no redrawn_because) outcome; and whether the
stager excluded the counted attempt (the staging record and its reason). Counts, keys and class names only.
  python3 make_rerun_census.py <raw dir> <chain ref> <out json>"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

raw_dir, ref, dest = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
CH = "experiments/2026-10-14_q3_overnight_chain/results"
names = subprocess.run(["git", "ls-tree", "--name-only", f"{ref}:{CH}"], capture_output=True, text=True, check=True, timeout=60).stdout.split()
excl = {}
for n in sorted(x for x in names if x.startswith("chain_stage_nm360_")):
    r = json.loads(subprocess.run(["git", "show", f"{ref}:{CH}/{n}"], capture_output=True, text=True, check=True, timeout=60).stdout)
    for part, v in r.items():
        if part in ("git", "stamp") or not isinstance(v, dict):
            continue
        for k, why in (v.get("c_gate") or {}).get("excluded", {}).items():
            excl.setdefault(k.split("#")[0], []).append({"staging_record": n, "reason": why})
ref_sha = subprocess.run(["git", "rev-parse", ref], capture_output=True, text=True, check=True, timeout=60).stdout.strip()
boxes, rows, launches = {}, [], {}
for box in ("box3", "box1", "box4", "simsastra"):
    lines = (raw_dir / f"{box}.txt").read_text().splitlines()
    d = json.loads(lines[-1])
    boxes[box] = {"read": lines[0], "host": d["host"], "raw_sha256": hashlib.sha256((raw_dir / f"{box}.txt").read_bytes()).hexdigest()}
    for ln, v in d["launches"].items():
        launches[ln] = v | {"box": box}
    for r in d["runs"]:
        counted = [a for a in r["attempts"] if a["role"] == "counted"]
        reruns = [a for a in r["attempts"] if a["role"] == "rerun"]
        rows.append({"box": box, "launch": r["launch"], "key": r["key"], "arm": r["arm"], "task": r["task"], "real": r["real"],
                     "run_state": r["run_state"],
                     "rerun": [{k: a[k] for k in ("i", "attempt_dir", "exit_code", "eval_status", "clean", "policy_calls", "policy_calls_usable",
                                                  "sample_error_classes", "eval")} for a in reruns],
                     "counted": [{k: a[k] for k in ("i", "kind", "attempt_dir", "exit_code", "eval_status", "clean", "policy_calls",
                                                    "policy_calls_usable", "sample_error_classes", "eval")}
                                 | {"stager_excluded": bool(a["eval"] and a["eval"] in excl), "staging": excl.get(a["eval"]) if a["eval"] else None}
                                 for a in counted],
                     "redrawn_attempts": sum(a["role"] == "redrawn" for a in r["attempts"])})
clean_rerun = [x for x in rows if any(a["clean"] is True for a in x["rerun"])]
doc = {"for": "nm360 runs with a kind == 'rerun' attempt (09 via d9, 10-15 ~10:35); manifests count only kind != 'rerun' without "
              "redrawn_because (box_files.py @ 468380762), so a clean rerun never enters them",
       "chain_ref": ref, "chain_ref_sha": ref_sha, "boxes_read": boxes,
       "not_read": {"2026-10-14_newmains_s2": "box2 powered off, not read", "2026-10-14_newmains_s2_fu": "box2 powered off, not read"},
       "launches": launches, "runs": rows,
       "counts": {"runs_with_rerun": len(rows), "with_a_clean_rerun": len(clean_rerun),
                  "with_a_clean_rerun_and_counted_attempt_excluded": sum(any(c["stager_excluded"] for c in x["counted"]) for x in clean_rerun),
                  "counted_attempt_excluded": sum(any(c["stager_excluded"] for c in x["counted"]) for x in rows)}}
dest.write_text(json.dumps(doc, indent=1) + "\n")
print(json.dumps(doc["counts"]))
for x in rows:
    print(x["box"], x["launch"], x["key"], "| rerun:", [(a["attempt_dir"], a["eval_status"], a["clean"]) for a in x["rerun"]],
          "| counted:", [(a["attempt_dir"], a["eval_status"], a["clean"], "EXCLUDED" if a["stager_excluded"] else "staged-or-not-excluded") for a in x["counted"]])
print("sha256", hashlib.sha256(dest.read_bytes()).hexdigest())
