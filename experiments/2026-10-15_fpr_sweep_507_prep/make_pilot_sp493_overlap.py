"""Route C B1 (09 via 2e, 10-17): which of the 20 pilot tasks' mains are S_paper493 mains. Keys and names only; no score is read.
Inputs are read from git by <commit>:<path>@<sha256>:
  - the pilot counted keys (simh_counted_keys_pilot_v3 @ 344829af6): its 20 tasks;
  - the T196 cells (the runner's main_t source, inputs_b.json local.t196): task -> main;
  - the INPUTS_FINAL v2 that INPUTS_FULL 110447 pins as base: extensions.S_paper493.cells -> S_paper493's mains (the runner's env_of_main keys).
The runner keeps another root's H_sim row on an S_paper493 row when the row's cell's main is an S_paper493 main (rp:1751), so this
count decides 09's option: 0 -> (b) a permanent STOP on any overlap; > 0 -> (a) tag and drop the pilot entries on S_paper493 rows.
POSITIVE CONTROL: the same join over all 196 T196 cells must find T196 mains that are S_paper493 mains (same name space), else STOP.
ASSERTS: 20 distinct pilot tasks, every one a T196 cell, none a +311 or S_paper493 cell key. Written with exclusive create.

  uv run python make_pilot_sp493_overlap.py --pilot <c>:<p>@<sha> --t196 <c>:<p>@<sha> --final-v2 <c>:<p>@<sha> --cells311 <c>:<p>@<sha> --out results/relock_pilot/<name>.json
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def show(ref):
    cp, h = ref.rsplit("@", 1)
    b = subprocess.run(["git", "show", cp], capture_output=True, check=True, cwd=HERE).stdout
    if hashlib.sha256(b).hexdigest() != h:
        raise SystemExit(f"{cp}: sha256 differs")
    return json.loads(b)


def main():
    ap = argparse.ArgumentParser()
    for a_ in ("--pilot", "--t196", "--final-v2", "--cells311", "--out"):
        ap.add_argument(a_, required=True)
    a = ap.parse_args()
    pl, t196, fv2, c311 = show(a.pilot), show(a.t196), show(a.final_v2), show(a.cells311)
    main_t = {c["key"]: c["main"] for c in t196["cells"]}
    sp_cells = fv2["extensions"]["S_paper493"]["cells"]
    sp_mains = sorted({v["main"] for v in sp_cells.values()})
    tasks = [k["task"] for k in pl["keys"]]
    bad = []
    if len(tasks) != 20 or len(set(tasks)) != 20:
        bad.append(f"pilot: {len(tasks)} tasks, {len(set(tasks))} distinct")
    if set(tasks) - set(main_t):
        bad.append(f"pilot tasks not T196 cells: {sorted(set(tasks) - set(main_t))}")
    c311_keys = {c["key"] for c in c311["cells"]}
    if set(tasks) & (c311_keys | set(sp_cells)):
        bad.append(f"pilot tasks that are +311 or S_paper493 cell keys: {sorted(set(tasks) & (c311_keys | set(sp_cells)))}")
    pc_ = sorted({m for m in main_t.values() if m in set(sp_mains)})
    if not pc_:
        bad.append("positive control: no T196 main is an S_paper493 main (name spaces differ?)")
    if bad:
        raise SystemExit(f"asserts FAILED: {bad}")
    rows = [{"task": t, "main": main_t[t], "s_paper493_main": main_t[t] in set(sp_mains)} for t in tasks]
    ov = [r for r in rows if r["s_paper493_main"]]
    rec = {"what": "route C B1 (09 via 2e, 10-17): pilot tasks whose main is an S_paper493 main (keys and names only)",
           "inputs": {"pilot": a.pilot, "t196": a.t196, "final_v2": a.final_v2, "cells311": a.cells311},
           "rule": "overlap = pilot task's T196 main in {S_paper493 cells' main} (the runner's env_of_main keys; rp:1751 keeps other roots' rows there)",
           "overlap_n": len(ov), "overlap_tasks": [r["task"] for r in ov], "overlap_mains": sorted({r["main"] for r in ov}),
           "pilot_tasks": rows, "n_pilot_tasks": len(rows), "n_s_paper493_mains": len(sp_mains),
           "positive_control": {"t196_mains_that_are_s_paper493_mains": len(pc_), "of_t196_mains": len(set(main_t.values()))},
           "decision_by_09": "0 -> option (b) permanent STOP on overlap; > 0 -> option (a) tag + drop pilot entries on S_paper493 rows"}
    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    b_ = json.dumps(rec, indent=1).encode()
    with out.open("xb") as fh:
        fh.write(b_)
    print(json.dumps({"written": str(out.relative_to(HERE)), "sha256": hashlib.sha256(b_).hexdigest(), "overlap_n": len(ov),
                      "overlap_tasks": rec["overlap_tasks"], "overlap_mains": rec["overlap_mains"], "positive_control": rec["positive_control"]}))


if __name__ == "__main__":
    main()
