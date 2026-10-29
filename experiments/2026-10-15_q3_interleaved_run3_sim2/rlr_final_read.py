"""Read ON the host, after a paired rerun launch ended (structure only; no transcript, no score): its launch.json sha256
before and after, finished, and per cell the real and simulated run state, attempt count and (for "not run") the
launcher's why. The record make_rerun3_files.py selects rlr3 cells from.
  python3 rlr_final_read.py <rlr run dir>  > results/il_rlr_final_<launch>.json
"""
import hashlib, json, sys
run = sys.argv[1]
h0 = hashlib.sha256(open(run + "/launch.json", "rb").read()).hexdigest()
d = json.load(open(run + "/launch.json"))
rs = d["runs"]; rs = list(rs.values()) if isinstance(rs, dict) else rs
cells = {}
for r in rs:
    arm, key = r["key"].split("/", 1)
    cells.setdefault(key, {})["real" if r["real"] else "sim"] = {"state": r["state"], "attempts": len(r.get("attempts") or []),
                                                                  **({"why": str(r.get("why") or "")[:120]} if r["state"] == "not run" else {})}
h1 = hashlib.sha256(open(run + "/launch.json", "rb").read()).hexdigest()
print(json.dumps({"run": run, "sha_before": h0, "launch_json_sha256": h1, "finished": d.get("finished"),
                  "stopped_by_signal": d.get("stopped_by_signal"), "cells": cells}, indent=1))
