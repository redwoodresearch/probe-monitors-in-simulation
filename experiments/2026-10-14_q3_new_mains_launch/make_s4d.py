"""The 19-cell sim-only follow-up of the S4 main ("4d", RULE.md Amendment 8; d9 about 22:45 PT, approved on one Codex
check): the S4 cells NOT selected by make_s4r.py (all 4 real arms finished on servers 0-3), which had no simulated run
because the S4 main's follow-up was refused (ATTEMPT_STOP: 30 < 50 cells, r2 22/30). References: the S4 main's own.
Writes, entries byte-identical to config/tasks_nm_s4_launch.json: config/tasks_nm_s4d_launch.json (the 19 cells' r1 and
r2 entries: the "main launch file" fu_launch.sh's gate 3 compares with; no launch runs it), config/tasks_nm_s4d_fu.json
(their r1 entries) and config/pilot_nm_fu_s4d.yaml (pilot_nm_fu_s4.yaml with only the task path and the caps changed:
cells x $16.50 + $300 + $1,000). Keys and hashes only.
  python make_s4d.py <config dir> <results/s4r_selection.json>"""
import hashlib, json, sys
from pathlib import Path
cfg, sel_f = Path(sys.argv[1]), Path(sys.argv[2])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
src = cfg / "tasks_nm_s4_launch.json"
out_l, out_f, out_c = cfg / "tasks_nm_s4d_launch.json", cfg / "tasks_nm_s4d_fu.json", cfg / "pilot_nm_fu_s4d.yaml"
for p in (out_l, out_f, out_c):
    assert not p.exists(), f"{p} exists"
d, sel = json.loads(src.read_text()), json.loads(sel_f.read_text())
rerun = set(sel["keys"])
assert len(rerun) == sel["cells"] == 61
r1 = [c for c in d["cells"] if "real_arms" not in c]
r2 = {c["key"]: c for c in d["cells"] if "real_arms" in c}
keep = [c for c in r1 if c["key"] not in rerun]
assert len(keep) == 19 and len(r1) == 80 and rerun <= {c["key"] for c in r1}
cells = []
for c in keep:
    cells += [c, r2[f"{c['key']}_r2"]]
meta = {"of_launch": "2026-10-14_newmains_s4", "from": src.name, "from_sha256": sha(src), "selection_sha256": sha(sel_f),
        "rule": "the S4 r1 cells not in make_s4r.py's rerun selection; sim-only follow-up with the S4 main's references",
        "cells": len(keep), "keys": [c["key"] for c in keep]}
out_l.write_text(json.dumps(dict(d, cells=cells, s4d=meta), indent=1))
out_f.write_text(json.dumps(dict(d, cells=keep, s4d=meta, followup={"from": out_l.name, "from_sha256": sha(out_l)}), indent=1))
cap = round(len(keep) * 16.5 + 300) + 1000
t = (cfg / "pilot_nm_fu_s4.yaml").read_text()
assert t.count("tasks: config/tasks_nm_s4_fu.json") == 1
t = t.replace("tasks: config/tasks_nm_s4_fu.json", "tasks: config/tasks_nm_s4d_fu.json")
lines = [l for l in t.splitlines() if l.strip().startswith(("cap_usd:", "worst_case_limit_usd:"))]
assert len(lines) == 2, lines
for l in lines:
    t = t.replace(l, f"  {l.strip().split(':')[0]}: {cap}")
out_c.write_text("# S4 19-CELL FOLLOW-UP (RULE.md Amendment 8, make_s4d.py): pilot_nm_fu_s4.yaml with only the task file and the caps changed\n" + t)
print(json.dumps({"cells": len(keep), "cap": cap, "launch_sha256": sha(out_l), "fu_sha256": sha(out_f), "config_sha256": sha(out_c)}))
