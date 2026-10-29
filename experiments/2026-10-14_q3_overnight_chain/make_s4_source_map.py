"""S4 source map, s4d part (09 @ bffe22e3e via d9, 10-15 ~04:10), committed before s4d_fu is staged. Keys, hashes and states only.
Per cell of the s4d follow-up (2026-10-14_newmains_s4d_fu, box4): its main launch_id (newmains_s4) and sim launch_id, both launch.json
sha256s, and its run state when read (the read-only box probe's output). Overlap check: the s4d cells against 4r's registered
61-cell list (the new-mains fork's s4r_selection.json and its 4r task file, both pinned by commit) must be empty; any overlap, a cell
listed twice, or a missing state is an ERROR (exit 2). The full map (4r and 4r_fu cells, the by-launch breakdown) follows when 4r ends.
Run: python make_s4_source_map.py --probe <probe output txt> --out <json>"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NM = "experiments/2026-10-14_q3_new_mains_launch"
SEL = ("1bbab93a2", f"{NM}/results/s4r_selection.json")
TASKS_4R = ("1bbab93a2", f"{NM}/config/tasks_nm_s4r_launch.json")


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    errors = []
    txt = Path(a.probe).read_text()
    shas = dict((p.split("/")[-2], h) for h, p in re.findall(r"([0-9a-f]{64})\s+(\S+/launch\.json)", txt))
    (line,) = [ln.strip() for ln in txt.splitlines() if ln.strip().startswith("{")]
    fu = json.loads(line)
    sel_raw, tasks_raw = git_show(*SEL), git_show(*TASKS_4R)
    sel, tasks = json.loads(sel_raw), json.loads(tasks_raw)
    r4 = set(sel["keys"])
    r4_tasks = {c["key"][:-3] if c["key"].endswith("_r2") else c["key"] for c in tasks["cells"]}
    if r4 != r4_tasks or len(r4) != sel["cells"]:
        errors.append(f"4r lists disagree: selection {len(r4)} keys, task file {len(r4_tasks)} cells")
    rows, seen = [], set()
    for r in fu["runs"]:
        if not (r["key"] or "").startswith("sim_attack@september/"):
            continue
        c = r["task"]
        if c in seen:
            errors.append(f"{c}: listed twice")
        seen.add(c)
        if not r.get("state"):
            errors.append(f"{c}: no state")
        rows.append({"key": f"sim|{c}", "cell": c, "main_launch_id": "2026-10-14_newmains_s4", "sim_launch_id": fu["run"], "state_when_read": r["state"],
                     "attempts": r["attempts"], "in_4r": c in r4})
    overlap = sorted(c for c in seen if c in r4)
    if overlap:
        errors.append(f"s4d cells overlap 4r's registered list: {overlap}")
    probe_at = re.search(r"probe at (.*)", txt)
    doc = {"rule": "09 @ bffe22e3e (S4 sim arm: union by cell of s4d_fu and s4r_fu; this is the s4d part)", "read_at": probe_at.group(1) if probe_at else None,
           "launch_json_sha256": {"2026-10-14_newmains_s4d_fu": shas.get("2026-10-14_newmains_s4d_fu"), "2026-10-14_newmains_s4": shas.get("2026-10-14_newmains_s4")},
           "s4d_finished": fu.get("finished"), "s4d_references_from": fu.get("references_from"),
           "four_r": {"selection": {"commit": SEL[0], "path": SEL[1], "sha256": hashlib.sha256(sel_raw).hexdigest(), "cells": len(r4)},
                      "task_file": {"commit": TASKS_4R[0], "path": TASKS_4R[1], "sha256": hashlib.sha256(tasks_raw).hexdigest(), "cells": len(r4_tasks)},
                      "of_launch_json_sha256": tasks["rerun"]["of_launch_json_sha256"]},
           "cells": len(rows), "overlap_with_4r": overlap, "rows": sorted(rows, key=lambda x: x["cell"]), "errors": errors, "pass": not errors}
    if doc["four_r"]["of_launch_json_sha256"] != doc["launch_json_sha256"]["2026-10-14_newmains_s4"]:
        doc["notes"] = ["the S4 main launch.json read now differs from the one the 4r task file was built from"]
    Path(a.out).open("x").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("cells", "overlap_with_4r", "errors", "pass", "launch_json_sha256")} | {"4r_cells": len(r4), "notes": doc.get("notes")}, indent=1))
    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
