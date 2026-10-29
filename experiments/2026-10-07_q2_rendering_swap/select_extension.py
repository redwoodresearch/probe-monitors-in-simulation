"""Pick the extension runs for the both-fixes arm (plan A, 2026-10-08): about 206 main tasks in all, from 89.

Eligible, as in select_runs.py: a sim-attack run with at least one Q2 call after an insert output that can be re-rendered
(an insert in an earlier turn whose category is exact, approx_window or approx_numbering). Excluded: every main task
already in config/swap_runs.json. From the remaining tasks, N_TASKS are drawn at random (seed SEED), and in each one
eligible run is drawn at random (same seed). Writes config/swap_runs_ext.json in swap_runs.json's format (stratum
"extension"), with the insert categories computed by the current rerender.py.

  uv run --with pandas python experiments/2026-10-07_q2_rendering_swap/select_extension.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-06_q2_impactful_call_transfer" / "shift"))
import examples as ex  # noqa: E402
import rerender as rr  # noqa: E402

N_TASKS, SEED = 117, 1

q = ex.table("qwen3-32b")
sim = q[q.domain == "sim"]
task_of = dict(zip(sim.key, sim.main_task))
runs = ex.load_runs(set(sim.key))
q2_calls = {k: set(zip(g.step, g.action)) for k, g in sim.groupby("key")}
info = {}
for key, calls in runs.items():
    order = sorted(calls)
    records = []
    for sa in order:
        c = calls[sa]
        a = rr.args_of(c)
        if c["tool"] != "text_editor" or a.get("command") != "insert":
            continue
        history = [(rr.args_of(calls[sb]), calls[sb]["output"], calls[sb]["tool"]) for sb in order if sb < sa]  # every earlier call
        records.append({"step": sa[0], "action": sa[1], "category": rr.rerender(a, c["output"] or "", history)[1]})
    first = min((r["step"] for r in records if r["category"] in rr.RENDERED), default=None)
    cats = Counter(r["category"] for r in records)
    info[key] = {"inserts": records, "q2_calls_after_rendered_insert": sum(1 for s, _ in q2_calls[key] if first is not None and s > first),
                 "all_exact": cats.get("exact", 0) > 0 and set(cats) <= {"exact", "not_rendered"}, "any_failed": cats.get("failed", 0) > 0,
                 "stratum": "extension"}
used = json.loads((HERE / "config" / "swap_runs.json").read_text())["runs"]
used_tasks = {task_of[k] for k in used}
eligible = sorted(k for k, v in info.items() if v["q2_calls_after_rendered_insert"] > 0)
by_task = {}
for k in eligible:
    if task_of[k] not in used_tasks:
        by_task.setdefault(task_of[k], []).append(k)
rng = np.random.default_rng(SEED)
tasks = sorted(rng.choice(sorted(by_task), N_TASKS, replace=False).tolist())
pick = sorted(by_task[t][rng.integers(len(by_task[t]))] for t in tasks)
assert len(pick) == N_TASKS and len({task_of[k] for k in pick}) == N_TASKS and not set(pick) & set(used)
out = {"seed": SEED, "n_tasks": N_TASKS, "n_eligible": len(eligible), "n_new_tasks_available": len(by_task),
       "excluded_tasks_from": "config/swap_runs.json", "runs": {k: info[k] for k in pick}}
(HERE / "config" / "swap_runs_ext.json").write_text(json.dumps(out, indent=1))
cats = Counter(r["category"] for k in pick for r in info[k]["inserts"])
print(f"eligible {len(eligible)}; new tasks available {len(by_task)}; picked {len(pick)} runs in {N_TASKS} tasks; "
      f"insert categories {dict(cats)}; Q2 calls after a rendered insert {sum(info[k]['q2_calls_after_rendered_insert'] for k in pick)}; "
      f"main tasks with the original 100: {len(used_tasks | set(tasks))}")
