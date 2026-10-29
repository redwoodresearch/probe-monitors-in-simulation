"""resolve_simh_nm.py (nm360 sim-honest): per launch group, each cell of config_nm/tasks_simh_nm_<g>.json -> RUN (its main's
pinned honest_a reference equals the COUNTED attempt, the stager's rule) or a "not run" status row by key. Rules (d9 ~11:30,
0d; 09 Append 25g): counted != pinned (a rerun cell) -> "not run: no counted usable reference" (never grounded on the rerun);
no pinned reference -> "not run: no reference"; the 9 broken-setup cells were dropped from the task files already. Inputs:
counted_vs_pinned.py outputs Aswin ran (results/nm_counted_vs_pinned_<main>_2026-10-15.json; s4d's 19 cells from newmains_s4,
4r's from newmains_s4r per the S4 source maps); for S2 (box2 down) the xbox3 presence check
(results/s2_presence_xbox3_2026-10-15.json): pinned references whose copy and source hash to the pin; its counted check is
PENDING until Aswin's one-main run lands, and S2 does not launch before it. Keys and hashes only.
  python3 resolve_simh_nm.py   (v2: results/simh_nm_resolution_v2_2026-10-15.json; the 25g cells are no longer in the task
  files, so they are listed from the counted_vs_pinned outputs)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
VERSION = 2  # append-only: v1 (S2 counted check pending) kept; v2 adds S2 counted_vs_pinned (Aswin ran it on xbox3)
R = HERE / "results"
MAIN_OF = {"s1": "s1", "s2": "s2", "s3": "s3", "s4d": "s4", "s4r": "s4r", "s5": "s5"}  # v2: S2 from Aswin's xbox3 run
out = {"rules": __doc__.split("Rules")[1].split("Inputs")[0].strip(), "groups": {}}
for g in ("s1", "s2", "s3", "s4d", "s4r", "s5"):
    keys = [c["key"] for c in json.loads((HERE / "config_nm" / f"tasks_simh_nm_{g}.json").read_text())["cells"]]
    if VERSION == 2:  # the 25g cells left the task files; list them from the builder's record so they keep a status row
        keys += json.loads((R / "simh_nm_tasks_2026-10-15.json").read_text())["groups"][g]["dropped_25g"]
    run, notrun = {}, {}
    if g == "s2" and VERSION == 1:
        p = json.loads((R / "s2_presence_xbox3_2026-10-15.json").read_text())
        refs = p["references"]
        for k in keys:
            e = refs.get(k)
            if e is None:
                notrun[k] = "not run: no reference"
            elif e["copy_ok"] and e["source_ok"]:
                run[k] = e["pinned_sha256"]
            else:
                notrun[k] = "not run: reference copy or source does not hash to the pin"
        counted_check = "PENDING (Aswin's one-main counted_vs_pinned on xbox3)"
    else:
        cvp = json.loads((R / f"nm_counted_vs_pinned_{MAIN_OF[g]}_2026-10-15.json").read_text())
        rows = {r["task"]: r for r in cvp["rows"]}
        for k in keys:
            r = rows.get(k)
            if r is None:
                notrun[k] = "not run: not an honest_a run of this main"
            elif not r["pinned"]:
                notrun[k] = "not run: no reference"
            elif not r.get("equal"):
                notrun[k] = "not run: no counted usable reference (counted attempt differs from the pinned rerun; 09 Append 25g)"
            else:
                run[k] = r["pinned_sha256"]
        counted_check = f"done: {cvp['main']}"
    out["groups"][g] = {"cells": len(keys), "n_run": len(run), "not_run": len(notrun), "counted_check": counted_check,
                        "not_run_rows": dict(sorted(notrun.items())),
                        "run": [{"task": k, "pinned_sha256": h} for k, h in sorted(run.items())]}  # task/sha pairs as fields, not key: hash
out["total"] = {k: sum(v[k] for v in out["groups"].values()) for k in ("cells", "n_run", "not_run")}
(R / ("simh_nm_resolution_2026-10-15.json" if VERSION == 1 else "simh_nm_resolution_v2_2026-10-15.json")).write_text(json.dumps(out, indent=1) + "\n")
for g, v in out["groups"].items():
    print(g, v["cells"], "run", v["n_run"], "not run", v["not_run"], v["not_run_rows"])
print("total", out["total"])
