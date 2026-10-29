import json, collections
from pathlib import Path
import sys
sys.path.insert(0, "src")
from probe_shift.viewer.pairs import read_launch, pairs, _log
RD = Path.home() / "phase2/runs/2026-10-09_000523_rerun_strict"
L = read_launch(RD)
print("top keys:", list(L.keys()))
runs = L["runs"]
print("n runs:", len(runs))
print("run keys sample:", list(runs[0].keys()))
c = collections.Counter((r["arm"], r.get("simulator"), r.get("state")) for r in runs)
for k, v in sorted(c.items(), key=str): print(k, v)
print("watcher_stop_reason:", (RD/"watcher_stop_reason.txt").read_text())
P = pairs(RD)
print("n pairs:", len(P))
print(collections.Counter((p["arm"], p["simulator"]) for p in P))
# runs without logs
for r in runs:
    lg = _log(RD, r)
    if lg is None:
        print("NO LOG:", r["key"], r.get("state"), len(r.get("attempts") or []))
    elif lg["attempt"] != lg["of"]:
        print("not last attempt:", r["key"], lg["attempt"], lg["of"], r.get("state"))
# tasks
tasks = sorted({r.get("task") or r["key"].split("/",1)[1] for r in runs})
print("n tasks:", len(tasks))
# missing pairs: per task per sim arm/simulator
have = {(p["task"], p["arm"], p["simulator"]) for p in P}
for t in tasks:
    for arm in ("sim_attack","sim_honest"):
        for s in ("september","today"):
            if (t,arm,s) not in have: print("MISSING PAIR:", t, arm, s)
