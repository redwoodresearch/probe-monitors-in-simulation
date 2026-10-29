"""Per-cell run states of an IL launch, read ON the host (read-only; structure only): for each cell, the real and simulated
run state, attempt count and (simulated) cost; plus the launch's finished time and last actual_usd.
  python3 il_cells_read.py <IL run dir> ...  > cells_<host>.json
"""
import json, sys
out = []
for run in sys.argv[1:]:
    d = json.load(open(run + "/launch.json"))
    rs = d["runs"]; rs = list(rs.values()) if isinstance(rs, dict) else rs
    cells = {}
    for r in rs:
        arm, key = r["key"].split("/", 1)
        c = cells.setdefault(key, {})
        cost = None
        if not r["real"]:
            a = r.get("attempts") or []
            cost = round(sum((x.get("sim_usd") if x.get("sim_usd") is not None else (x.get("live_usd") or 0)) for x in a), 4) if a else 0.0
        c["real" if r["real"] else "sim"] = {"state": r["state"], "attempts": len(r.get("attempts") or []), **({"sim_usd": cost} if cost is not None else {})}
    sp = (d.get("spend") or [{}])[-1]
    out.append({"run": run, "finished": d.get("finished"), "actual_usd": sp.get("actual_usd"), "spend_time": sp.get("time"), "cells": cells})
print(json.dumps(out))
