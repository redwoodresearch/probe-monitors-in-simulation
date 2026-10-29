"""S5 follow-up rerun "5r" (09's Append 35 @ d99ce7997; d9 about 01:10 PT): a sim-only follow-up of exactly the 6 S5 cells
whose simulated run failed with an infrastructure rate limit (results/s5_rl_classification_evidence_2026-10-15.json:
every attempt's sample error wraps RateLimitError on the simulator), references from the S5 main, everything else frozen
(the 4d pattern). Writes, entries byte-identical to config/tasks_nm_s5_launch.json: config/tasks_nm_s5r_launch.json (the
6 cells' r1 and r2 entries: the "main launch file" fu_launch.sh's gate 3 compares with; no launch runs it),
config/tasks_nm_s5r_fu.json (their r1 entries; meta rerun_of per cell: original launch, run key, .eval sha256; rerun_reason
"infra: RateLimitError"), config/pilot_nm_fu_s5r.yaml (pilot_nm_fu_s5.yaml with only the task path and the caps changed:
cells x $16.50 + $300 + $1,000) and results/s5r_supersession_map.json (one authoritative counted run per world x cell,
committed before scoring). Keys and hashes only.
  python make_s5r.py <config dir> <results/s5_rl_classification_evidence_2026-10-15.json>"""
import hashlib, json, sys
from pathlib import Path
cfg, ev_f = Path(sys.argv[1]), Path(sys.argv[2])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
src = cfg / "tasks_nm_s5_launch.json"
out_l, out_f, out_c = cfg / "tasks_nm_s5r_launch.json", cfg / "tasks_nm_s5r_fu.json", cfg / "pilot_nm_fu_s5r.yaml"
out_m = cfg.parent / "results" / "s5r_supersession_map.json"
for p in (out_l, out_f, out_c, out_m):
    assert not p.exists(), f"{p} exists"
d, ev = json.loads(src.read_text()), json.loads(ev_f.read_text())
runs = {r["cell"]: r for r in ev["runs"]}
assert len(runs) == 6 and all(r["infra_rate_limit"] for r in runs.values()), "every rerun cell must be classified infra: rate limit"
r1 = [c for c in d["cells"] if "real_arms" not in c]
r2 = {c["key"]: c for c in d["cells"] if "real_arms" in c}
keep = [c for c in r1 if c["key"] in runs]
assert len(keep) == 6
rerun_of = {c["key"]: {"launch": runs[c["key"]]["launch"], "run_key": runs[c["key"]]["run_key"],
                       "eval_sha256": runs[c["key"]]["attempts"][-1]["eval_sha256"]} for c in keep}
meta = {"of_launch": "2026-10-14_newmains_s5_fu", "rerun_reason": "infra: RateLimitError", "from": src.name, "from_sha256": sha(src),
        "evidence_sha256": sha(ev_f), "cells": [c["key"] for c in keep], "rerun_of": rerun_of,
        "replacement_launch": "2026-10-14_newmains_s5r_fu"}
cells = []
for c in keep:
    cells += [c, r2[f"{c['key']}_r2"]]
out_l.write_text(json.dumps(dict(d, cells=cells, s5r=meta), indent=1))
out_f.write_text(json.dumps(dict(d, cells=keep, s5r=meta, followup={"from": out_l.name, "from_sha256": sha(out_l)}), indent=1))
cap = round(len(keep) * 16.5 + 300) + 1000
t = (cfg / "pilot_nm_fu_s5.yaml").read_text()
assert t.count("tasks: config/tasks_nm_s5_fu.json") == 1
t = t.replace("tasks: config/tasks_nm_s5_fu.json", "tasks: config/tasks_nm_s5r_fu.json")
lines = [l for l in t.splitlines() if l.strip().startswith(("cap_usd:", "worst_case_limit_usd:"))]
assert len(lines) == 2, lines
for l in lines:
    t = t.replace(l, f"  {l.strip().split(':')[0]}: {cap}")
out_c.write_text("# S5 FOLLOW-UP RERUN 5r (RULE.md Amendment 10, make_s5r.py; Append 35): pilot_nm_fu_s5.yaml with only the task file and the caps changed\n" + t)
out_m.write_text(json.dumps({"rule": "Append 35 item 2: one authoritative counted run per world x cell; the replacement keeps the key; failed originals and failed replacements contribute neither scores nor denominator observations",
    "map": [{"world": "sim", "cell": c["key"], "key": f"sim_attack@september/{c['key']}", "original": rerun_of[c["key"]] | {"state": runs[c["key"]]["state"], "classification": "infra: RateLimitError"},
             "replacement": {"launch": "2026-10-14_newmains_s5r_fu", "run_key": f"sim_attack@september/{c['key']}", "eval_sha256": "(filled at completion)"},
             "authoritative": "replacement if it is admitted and usable; otherwise none"} for c in keep]}, indent=1))
print(json.dumps({"cells": len(keep), "cap": cap, "launch_sha256": sha(out_l), "fu_sha256": sha(out_f), "config_sha256": sha(out_c), "map_sha256": sha(out_m)}))
