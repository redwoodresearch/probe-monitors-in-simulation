"""make_t196fill.py (the A25 fill-in: 8 T196 tasks (09 addendum 6; 7 more admitted from the A25 pilot), 09 registration @ 286b2f518, NOT Append 39; Aswin via 0d ~17:38 "ok all 3";
d9 ~17:50): config_sp493/tasks_simh_t196fill.json (each cell byte-identical to its entry in tasks_il_old_all.json, top-level
fields kept) and config_sp493/pilot_simh_t196fill.yaml (the +311 sim-honest config with tasks pointed at it; admission cap =
expected + 7 x $106 so all 8 can run at once; sim_cap_stop pauses at $500, Aswin's cap), plus results/t196fill_simh_plan_2026-10-15.json.
  python3 make_t196fill.py <repo>
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

repo = sys.argv[1]
HERE = Path(__file__).parent
show = lambda ref, p: subprocess.run(["git", "-C", repo, "show", f"{ref}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
TASKS = ("origin/aswin/q3-interleaved-run3-sim2", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/tasks_il_old_all.json",
         "69c709d91da2f8952c0a3f4ceddf3e05cac5044f73a596eb253bb5763f70ee4e")
IDS = ["t04", "t07", "t08", "t11", "t12", "t13", "t17", "t48"]  # 0d/09 ~18:10 (addendum 6): 8 keys; the 15-key list (d9 ~17:38) is superseded
FROM_PILOT = ["t21", "t23", "t28", "t30", "t31", "t32", "t33"]  # admitted from A25's usable pilot sim-honest runs (world simh_pilot), not generated
EXCLUDED = {"t88": "failed the September preflight in il_old", "t139": "no pinned reference", "t151": "no pinned reference",
            "t169": "failed the September preflight in il_old"}  # d9 ~17:38 (evidence: the 46-task partition table)
raw = show(TASKS[0], TASKS[1]); assert hashlib.sha256(raw).hexdigest() == TASKS[2], "tasks_il_old_all.json is not the pinned one"
d = json.loads(raw)
by = {}
for c in d["cells"]:
    by.setdefault(c["key"].split("_")[0], []).append(c)
assert all(len(by.get(i, [])) == 1 for i in IDS), [i for i in IDS if len(by.get(i, [])) != 1]
cells = sorted((by[i][0] for i in IDS), key=lambda c: c["key"])
name = "t196fill"
tp = HERE / "config_sp493" / f"tasks_simh_{name}.json"
tp.write_text(json.dumps(dict(d, cells=cells), indent=1) + "\n")
n = len(cells); exp = round(n * 19.6); K = 7; cap = exp + K * 106; stop = 500  # admission cap // 106 = 8: all at once
cfg = (HERE / "config" / "pilot_simh_q1.yaml").read_text()
cfg, c1 = re.subn(r"^tasks: .*$", f"tasks: config/tasks_simh_{name}.json   # make_t196fill.py: the A25 fill-in, 8 T196 tasks (cells of tasks_il_old_all.json)", cfg, flags=re.M)
cfg, c2 = re.subn(r"^  cap_usd: .*$", f"  cap_usd: {cap}   # {n} cells: expected ${exp} + K {K} x $106 (admission at start: cap // 106 = {cap // 106}); spend pause at ${stop}", cfg, flags=re.M)
cfg, c3 = re.subn(r"^  worst_case_limit_usd: .*$", f"  worst_case_limit_usd: {cap}   # = cap: the box-side admission bound", cfg, flags=re.M)
assert c1 == c2 == c3 == 1
cfg = f"# A25 FILL-IN SIM-HONEST launch simh_{name} (09 registration @ 286b2f518; not Append 39), from config/pilot_simh_q1.yaml:\n" + cfg
cp = HERE / "config_sp493" / f"pilot_simh_{name}.yaml"
cp.write_text(cfg)
plan = {"launch": name, "registration": "A25 fill-in (09 @ 286b2f518; second refresh only; primary_v3 and A25 rows unchanged)",
        "keys": [c["key"] for c in cells], "excluded": EXCLUDED, "admitted_from_pilot_not_generated": FROM_PILOT,
        "supersedes": "the 15-key plan/manifest (not to be used; 09 addendum 6, d9 ~18:10)", "n": n, "expected_usd": exp, "K": K, "cap_usd": cap, "sim_cap_stop_pause_at": stop,
        "approval": "Aswin via 0d ~17:38, verbatim: \"ok all 3\" (d9 relay: ~$390 with +30%, cap ~$500)",
        "task_source": {"ref": TASKS[0], "path": TASKS[1], "sha256": TASKS[2]},
        "tasks_file": tp.name, "tasks_sha256": hashlib.sha256(tp.read_bytes()).hexdigest(),
        "config": cp.name, "config_sha256": hashlib.sha256(cp.read_bytes()).hexdigest()}
(HERE / "results" / "t196fill_simh_plan_2026-10-15.json").write_text(json.dumps(plan, indent=1) + "\n")
print(name, n, "cells; cap", cap, "(admission", cap // 106, ") pause", stop, "tasks", plan["tasks_sha256"][:12], "config", plan["config_sha256"][:12])
