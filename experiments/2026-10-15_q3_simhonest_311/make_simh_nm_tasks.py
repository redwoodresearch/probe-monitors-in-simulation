"""make_simh_nm_tasks.py (nm360 sim-honest, d9 ~11:05 relaying Aswin via 0d): per launch group, the simulated-honest task
file = the committed nm follow-up task file's cells (the split's r1 entries; origin/aswin/q3-new-mains-launch config/)
minus the 9 broken-setup cells (setup_failures v4 @ 396da80fd), unchanged otherwise; plus its config (sim_honest@september,
the +311 simulated-honest spend settings: $106 worst per run, prior $21.13, cap = cells x $21.13 x 1.3). Keys and hashes
only; reads committed files through git. Writes config_nm/{tasks_simh_nm_<g>.json, pilot_simh_nm_<g>.yaml} and
results/simh_nm_tasks_2026-10-15.json.
  python3 make_simh_nm_tasks.py <repo>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO, HERE = sys.argv[1], Path(__file__).parent
REF, D = "origin/aswin/q3-new-mains-launch", "experiments/2026-10-14_q3_new_mains_launch/config"
BROKEN = ["nm208", "nm213", "nm241", "nm222", "nm106", "nm219", "nm126", "nm109", "nm203"]  # v4 @ 396da80fd
# 09 Append 25g (0d, d9 ~11:30): the counted attempt differs from the pinned rerun -> "not run: no counted usable
# reference"; dropped here so the launcher can never ground them on the rerun (results/simh_nm_resolution_2026-10-15.json)
RERUN_25G = ["nm142", "nm209", "nm280", "nm145", "nm235"]  # nm235 (S2): d9 ~11:40 from Aswin's xbox3 run; first attempt failed
FLOOR_CAP = {"s4d": 1060}  # 0d ~11:35 via d9: s4d floor cap (10 concurrent at $106); the others cells x $21.13 x 1.3
GROUPS = {  # group: (follow-up task file, its sha256 (make_splits_nm.json / git), main run dir, main's launch task file)
    "s1": ("tasks_nm_s1_fu.json", "9c39726a2f1464f2c86fa8866691c6cecc47f7b907abd64590e94b970205fde8", "2026-10-14_newmains_s1", "tasks_nm_s1_launch.json"),
    "s2": ("tasks_nm_s2_fu.json", "3db7f864fa574feec01e0830830c567d256ecaeb4fb8f17a0fa20b448c80caa9", "2026-10-14_newmains_s2", "tasks_nm_s2_launch.json"),
    "s3": ("tasks_nm_s3_fu.json", "1dd8280de5fc511823da64a6ae716320d27b0bf6ee732343dcab91c8b006cdd5", "2026-10-14_newmains_s3", "tasks_nm_s3_launch.json"),
    "s4d": ("tasks_nm_s4d_fu.json", "a16c73db952dd77a415bea0ad686860b7e5ffd039eec9d80408d3e5ab9d112d9", "2026-10-14_newmains_s4", "tasks_nm_s4_launch.json"),
    "s4r": ("tasks_nm_s4r_fu.json", "b921721819f00b673389bc9e8a7568d8f861a5a8b052ef76c16ecbf08ded4e96", "2026-10-14_newmains_s4r", "tasks_nm_s4r_launch.json"),
    "s5": ("tasks_nm_s5_fu.json", "1591fcce8bea840fb24a58260c6d8e20a7c811ddd057958994ea3feccb6cbe3a", "2026-10-14_newmains_s5", "tasks_nm_s5_launch.json"),
}
PER_RUN, WORST = 21.13, 106


def git_bytes(path: str) -> bytes:
    return subprocess.run(["git", "-C", REPO, "show", f"{REF}:{D}/{path}"], check=True, capture_output=True).stdout


(HERE / "config_nm").mkdir(exist_ok=True)
record = {"rerun_25g_excluded": RERUN_25G, "floor_cap": FLOOR_CAP, "source": f"{REF} {D}", "source_commit": subprocess.run(["git", "-C", REPO, "rev-parse", REF], check=True,
          capture_output=True, text=True).stdout.strip(), "broken_excluded": BROKEN, "groups": {}}
seen = {}
for g, (fu, fu_sha, main, launch_file) in GROUPS.items():
    raw = git_bytes(fu)
    assert hashlib.sha256(raw).hexdigest() == fu_sha, (g, fu)
    d = json.loads(raw)
    cells = d["cells"]
    keys = [c["key"] for c in cells]
    assert len(keys) == len(set(keys)), g
    drop = [k for k in keys if k.split("_")[0] in BROKEN]
    drop_25g = [k for k in keys if k.split("_")[0] in RERUN_25G]
    keep = [c for c in cells if c["key"].split("_")[0] not in BROKEN + RERUN_25G]
    for c in keep:
        assert c["key"] not in seen, (c["key"], g, seen.get(c["key"]))
        seen[c["key"]] = g
    out = dict(d, cells=keep)  # Codex nm-delta review: keep the nm metadata as is; no new top-level keys (provenance goes to results/)
    tpath = HERE / "config_nm" / f"tasks_simh_nm_{g}.json"
    tpath.write_text(json.dumps(out, indent=1) + "\n")
    cap = max(round(len(keep) * PER_RUN * 1.3), FLOOR_CAP.get(g, 0))
    cfg = f"""# SIMULATED-HONEST launch simh_nm_{g} (experiments/2026-10-15_q3_simhonest_311/RULE.md, nm360 amendment): per cell one
# sim_honest@september forked at 0 from the nm main's honest_a (--references-from ~/phase2/nm/runs/{main}). Derived from
# {D}/pilot_nm_fu_{g if g not in ('s4d', 's4r') else g}.yaml (the nm follow-up, sim_attack) with the +311
# simulated-honest settings (experiments/2026-10-15_q3_simhonest_311/config/pilot_simh_q1.yaml).
tasks: config/tasks_simh_nm_{g}.json   # make_simh_nm_tasks.py: {fu} minus the broken-setup cells ({len(drop)}) and 09 Append 25g rerun cells ({len(drop_25g)})
real_simulator: september
real_arms: []   # follow-up: no real arm; references from the main launch (--references-from)
simulators: [september]
sim_arms: [sim_honest]
simulator_confirmed: true
max_runs_per_server: 10
max_real_runs: 60
reference_tries: 3
digest_sets: config/digest_sets_tagform_2026-10-08.json
spend:
  cap_usd: {cap}   # {len(keep)} cells x ${PER_RUN} x 1.3{" (s4d floor $1,060, 0d ~11:35)" if g in FLOOR_CAP else ""} (Aswin via 0d ~11:05: nm360 sim-honest; stop at the cap, never raise it)
  worst_case_limit_usd: {cap}   # = cap: the box-side admission bound
  worst_per_sim_run_usd: {WORST}   # the simulated-honest continuation's value (as +311)
  prior_per_sim_run_usd: {PER_RUN}
"""
    cpath = HERE / "config_nm" / f"pilot_simh_nm_{g}.yaml"
    cpath.write_text(cfg)
    record["groups"][g] = {"follow_up_task_file": fu, "follow_up_sha256": fu_sha, "main": f"/home/aswin/phase2/nm/runs/{main}",
                           "main_launch_task_file": launch_file, "cells_in": len(cells), "dropped_broken": drop, "dropped_25g": drop_25g, "cells": len(keep),
                           "cap_usd": cap, "tasks_sha256": hashlib.sha256(tpath.read_bytes()).hexdigest(),
                           "config_sha256": hashlib.sha256(cpath.read_bytes()).hexdigest()}
record["total_cells"] = sum(v["cells"] for v in record["groups"].values())
record["total_cap_usd"] = sum(v["cap_usd"] for v in record["groups"].values())
(HERE / "results" / "simh_nm_tasks_2026-10-15.json").write_text(json.dumps(record, indent=1) + "\n")
print({g: (v["cells_in"], len(v["dropped_broken"]), v["cells"], v["cap_usd"]) for g, v in record["groups"].items()},
      "total", record["total_cells"], record["total_cap_usd"])
