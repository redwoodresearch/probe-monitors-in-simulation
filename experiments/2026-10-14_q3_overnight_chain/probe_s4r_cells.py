"""ON BOX 4, read-only (d9 10-15 ~10:55, the S4 source map's 4r part): for 4r's main (2026-10-14_newmains_s4r) and follow-up
(2026-10-14_newmains_s4r_fu), each launch.json's SHA-256 and, per cell, every run's world (honest_a, run1 = attack, honest_b, run2 =
attack _r2, sim = sim_attack@september), key and state. Keys and states only. Prints one JSON object.
  python3 probe_s4r_cells.py"""
import hashlib
import json
from pathlib import Path

RUNS = Path("/home/aswin/phase2/nm/runs")
out = {"launch_json_sha256": {}, "cells": {}}
for ln in ("2026-10-14_newmains_s4r", "2026-10-14_newmains_s4r_fu"):
    raw = (RUNS / ln / "launch.json").read_bytes()
    out["launch_json_sha256"][ln] = hashlib.sha256(raw).hexdigest()
    for r in json.loads(raw)["runs"]:
        arm, task = r["key"].split("/", 1)
        r2 = task.endswith("_r2")
        cell = task[:-3] if r2 else task
        world = "run2" if r2 else {"attack": "run1", "sim_attack@september": "sim"}.get(arm, arm)
        out["cells"].setdefault(cell, {})[world] = {"launch": ln, "key": r["key"], "state": r["state"]}
print(json.dumps(out))
