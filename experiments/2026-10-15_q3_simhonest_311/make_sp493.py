"""make_sp493.py (S_paper493 sim-honest arm, d9 ~17:20; 09's rulings via d9 ~17:30): Phase A = 27 of the 28 T196 mains (t98 deferred, see DEFERRED) of coverage v2
whose sim_honest action is "generate". Writes config_sp493/tasks_simh_sp493_t196.json (each cell byte-identical, as a JSON
object, to its origin-key entry in tasks_il_old_all.json, the file il_old/sel97 used; top-level fields kept) and
config_sp493/pilot_simh_sp493_t196.yaml (the +311 sim-honest config with tasks pointed at it; cap = expected + K x $106,
K chosen so cap // $106 (the launcher's admission at start) is about 18, the room d9 set; sim_cap_stop pause at 2 x expected), plus results/sp493_simh_plan_2026-10-15.json.
  python3 make_sp493.py <repo> <K>
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

repo, K = sys.argv[1], int(sys.argv[2])
HERE = Path(__file__).parent
show = lambda ref, p: subprocess.run(["git", "-C", repo, "show", f"{ref}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
COV = ("5237ca017", "experiments/2026-10-15_q3_sp493/results/sp493_honest_coverage_2026-10-15_170958.json")
TASKS = ("origin/aswin/q3-interleaved-run3-sim2", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/tasks_il_old_all.json",
         "69c709d91da2f8952c0a3f4ceddf3e05cac5044f73a596eb253bb5763f70ee4e")
cov = json.loads(show(*COV))
keys = sorted(k for m in cov["mains"] if m["origin"] == "t196" and m["sim_honest"].get("sp493_action") == "generate" for k in m["origin_task_keys"])
assert len(keys) == 28, len(keys)
# design fork via d9 ~17:45: t98_model_registry has NO reference in il_refindex_old (09 Append 35 @ 221bb94ea: staging blocker,
# no usable donor); 09 routed t98's main to a NEW honest_a from the B fork, so t98 moves to the new-mains phase
DEFERRED = {"t98_model_registry"}
keys = [k for k in keys if k not in DEFERRED]
assert len(keys) == 27, len(keys)
# 09 addendum 6 via d9 ~18:10: A25's usable pilot sim-honest runs (world simh_pilot, t21-t40) cover 13 of the 27; g4 generates 14
FROM_PILOT = {"t22_clinical_trial", "t24_tiktok", "t25_user_tracking", "t26_file_indexing", "t27_slack_clone", "t29_web_scraping", "t34_tiktok",
              "t35_user_tracking", "t36_file_indexing", "t37_slack_clone", "t38_model_registry", "t39_web_scraping", "t40_shared_dotfiles"}
assert FROM_PILOT <= set(keys), sorted(FROM_PILOT - set(keys))
keys = [k for k in keys if k not in FROM_PILOT]
assert len(keys) == 14, len(keys)
raw = show(TASKS[0], TASKS[1]); assert hashlib.sha256(raw).hexdigest() == TASKS[2], "tasks_il_old_all.json is not the pinned one"
d = json.loads(raw)
by = {c["key"]: c for c in d["cells"]}
assert all(k in by for k in keys), [k for k in keys if k not in by]
cells = [by[k] for k in keys]
name = "sp493_t196"
tp = HERE / "config_sp493" / f"tasks_simh_{name}.json"
tp.write_text(json.dumps(dict(d, cells=cells), indent=1) + "\n")
n = len(cells); exp = round(n * 19.6); cap = exp + K * 106; stop = 2 * exp
cfg = (HERE / "config" / "pilot_simh_q1.yaml").read_text()
cfg, c1 = re.subn(r"^tasks: .*$", f"tasks: config/tasks_simh_{name}.json   # make_sp493.py: S_paper493 Phase A (g4), 14 T196 mains (origin-key cells of tasks_il_old_all.json)", cfg, flags=re.M)
cfg, c2 = re.subn(r"^  cap_usd: .*$", f"  cap_usd: {cap}   # {n} cells: expected ${exp} + K {K} x $106 (admission at start: cap // 106 = {cap // 106} running at once)", cfg, flags=re.M)
cfg, c3 = re.subn(r"^  worst_case_limit_usd: .*$", f"  worst_case_limit_usd: {cap}   # = cap: the box-side admission bound", cfg, flags=re.M)
assert c1 == c2 == c3 == 1
cfg = f"# S_PAPER493 SIM-HONEST launch simh_{name} (09 Append 39; rulings via d9 ~17:30), from config/pilot_simh_q1.yaml:\n" + cfg
cp = HERE / "config_sp493" / f"pilot_simh_{name}.yaml"
cp.write_text(cfg)
plan = {"launch": name, "keys": keys, "n": n, "expected_usd": exp, "K": K, "cap_usd": cap, "sim_cap_stop_pause_at": stop,
        "coverage": {"commit": COV[0], "path": COV[1]}, "task_source": {"ref": TASKS[0], "path": TASKS[1], "sha256": TASKS[2]},
        "tasks_file": tp.name, "tasks_sha256": hashlib.sha256(tp.read_bytes()).hexdigest(),
        "config": cp.name, "config_sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
        "admitted_from_pilot_not_generated": sorted(FROM_PILOT),
        "deferred": {"t98_model_registry": "no reference in il_refindex_old (09 Append 35 @ 221bb94ea); fork pin from the B fork's new honest_a; new-mains phase"},
        "t43": "t43_auto_workflow regenerated once (09); primary_v3 excluded this main (25e); a repeat 25e failure is excluded by class, no further attempt"}
(HERE / "results" / "sp493_simh_plan_2026-10-15.json").write_text(json.dumps(plan, indent=1) + "\n")
print(name, n, "cells; cap", cap, "stop", stop, "tasks", plan["tasks_sha256"][:12], "config", plan["config_sha256"][:12])
