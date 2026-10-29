"""S4 rerun (RULE.md Amendment 8; d9 about 21:30 PT: S4's not-run cells tonight, in the Q2-rerun pattern). Rule, fixed
before the S4 main's final states were read: a cell is RERUN iff any of its 4 real runs (honest_a/<k>, honest_b/<k>,
attack/<k>, attack/<k>_r2) is "not run", or "stopped" and not relabelled done (RELABEL_RULE_drainstop.txt), or was assigned
to server index 4 (ap-Ptp5NWVwK27Gt7ksEsbhuB, gone from Modal; 13 RetryError failures), whatever its state. Selection is
by queue position and server death, never by outcome. For a rerun cell, the rerun's runs (main and follow-up) supersede
S4's. Writes, from config/tasks_nm_s4_launch.json (entries byte-identical): config/tasks_nm_s4r_launch.json (the
selected cells' r1 and r2 entries), config/tasks_nm_s4r_fu.json (their r1 entries, simulated attacks only),
config/pilot_nm_s4r.yaml and config/pilot_nm_fu_s4r.yaml (S4's configs with only the task file and the caps changed:
main cells x $75 + $750; follow-up cells x $16.50 + $300 + $1,000), and results/s4r_selection.json. Keys and hashes only.
  python make_s4r.py <config dir> <s4_final_states.json>"""
import hashlib, json, sys
from pathlib import Path
cfg, states_f = Path(sys.argv[1]), Path(sys.argv[2])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
src = cfg / "tasks_nm_s4_launch.json"
out_l, out_f = cfg / "tasks_nm_s4r_launch.json", cfg / "tasks_nm_s4r_fu.json"
for p in (out_l, out_f, cfg / "pilot_nm_s4r.yaml", cfg / "pilot_nm_fu_s4r.yaml"):
    assert not p.exists(), f"{p} exists"
d, st = json.loads(src.read_text()), json.loads(states_f.read_text())
runs = st["runs"]
r1 = [c for c in d["cells"] if "real_arms" not in c]
r2 = {c["key"]: c for c in d["cells"] if "real_arms" in c}
assert len(r1) == len(r2) == 80 and all(f"{c['key']}_r2" in r2 for c in r1)
reasons = {}
for c in r1:
    k = c["key"]
    keys = [f"honest_a/{k}", f"honest_b/{k}", f"attack/{k}", f"attack/{k}_r2"]
    assert all(x in runs and runs[x]["real"] for x in keys), k
    why = sorted(({("not run" if runs[x]["state"] == "not run" else
                    "stopped" if runs[x]["state"] == "stopped" and runs[x]["relabel"] != "done" else None) for x in keys} |
                  {"server 4" for x in keys if runs[x]["server"] == 4}) - {None})
    if why:
        reasons[k] = why
sel = [c for c in r1 if c["key"] in reasons]
cells = []
for c in sel:
    cells += [c, r2[f"{c['key']}_r2"]]
meta = {"of_launch": st["run"], "of_launch_json_sha256": st["launch_json_sha256"], "from": src.name, "from_sha256": sha(src),
        "states_sha256": sha(states_f), "rule": __doc__.split("\n")[0:6], "cells": len(sel), "reasons": reasons}
out_l.write_text(json.dumps(dict(d, cells=cells, rerun=meta), indent=1))
fu_src = json.loads((cfg / "tasks_nm_s4_fu.json").read_text())
assert [c["key"] for c in fu_src["cells"]] == [c["key"] for c in r1]
out_f.write_text(json.dumps(dict(d, cells=sel, rerun=meta, followup={"from": out_l.name, "from_sha256": sha(out_l)}), indent=1))
n = len(sel)
main_cap, fu_cap = n * 75 + 750, round(n * 16.5 + 300) + 1000  # d9 21:5x: the +$1,000 rule, as every split
for name, new, tasks, cap in (("pilot_nm_s4.yaml", "pilot_nm_s4r.yaml", out_l.name, main_cap),
                              ("pilot_nm_fu_s4.yaml", "pilot_nm_fu_s4r.yaml", out_f.name, fu_cap)):
    t = (cfg / name).read_text()
    old_tasks = "config/tasks_nm_s4_launch.json" if "fu" not in name else "config/tasks_nm_s4_fu.json"
    assert t.count(f"tasks: {old_tasks}") == 1
    t = t.replace(f"tasks: {old_tasks}", f"tasks: config/{tasks}")
    lines = [l for l in t.splitlines() if l.strip().startswith(("cap_usd:", "worst_case_limit_usd:"))]
    assert len(lines) == 2, lines
    for l in lines:
        key = l.strip().split(":")[0]
        t = t.replace(l, f"  {key}: {cap}")
    (cfg / new).write_text(f"# S4 RERUN (RULE.md Amendment 8, make_s4r.py): {name} with only the task file and the caps changed\n" + t)
summary = {"cells": n, "of_80": 80, "reasons": {r: sum(r in v for v in reasons.values()) for r in ("not run", "stopped", "server 4")},
           "main_cap": main_cap, "fu_cap": fu_cap, "launch_sha256": sha(out_l), "fu_sha256": sha(out_f)}
(cfg.parent / "results" / "s4r_selection.json").write_text(json.dumps(dict(summary, keys=sorted(reasons)), indent=1))
print(json.dumps(summary))
