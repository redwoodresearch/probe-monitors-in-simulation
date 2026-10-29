"""Real-vs-real control for figure 1's "new Qwen vs Sol" gap (local, free; Aswin's rule, 2026-10-09): the same
pipeline's gap between two independent real Qwen3.5-27B honest runs per task (honest_a vs honest_b, strict rerun), on
figure 1's 18 matched tasks (t09, scorer timeouts, and t19, placeholder scorer, left out). With one run per arm per
task, the task-weighted rate is the share of tasks passed. Interval: 10,000 paired task resamples (seed 0), as
figure 1. Also: how often two real runs of one task disagree.

  uv run --no-project --with numpy python experiments/2026-10-09_q2_real_vs_real/analyze_fig1_real_vs_real.py <honest_success_qwen3.5-27b json>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DROP = ("t09_", "t19_")  # figure 1's left-out tasks
p = Path(sys.argv[1])
rows = json.loads(p.read_text())["Qwen3.5-27B"]["rows"]
by = {}
for r in rows:
    if not r["task_dir"].startswith(DROP):
        by.setdefault(r["task_dir"], {})[r["arm"]] = r["main_task_success"]
tasks = sorted(t for t, v in by.items() if {"honest_a", "honest_b"} <= set(v) and "N" not in v.values())
a = np.array([by[t]["honest_a"] == "C" for t in tasks], float)
b = np.array([by[t]["honest_b"] == "C" for t in tasks], float)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(tasks), (10_000, len(tasks)))
d = (a[idx] - b[idx]).mean(1)
print(json.dumps({"input": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "tasks": len(tasks),
                  "honest_a passed": f"{int(a.sum())} of {len(tasks)}", "honest_b passed": f"{int(b.sum())} of {len(tasks)}",
                  "honest_a minus honest_b (points)": [round(100 * (a.mean() - b.mean()), 1)] + [round(100 * x, 1) for x in np.percentile(d, [2.5, 97.5])],
                  "tasks where the two real runs disagree": f"{int((a != b).sum())} of {len(tasks)}",
                  "disagreeing tasks": [t for t, x, y in zip(tasks, a, b) if x != y]}, indent=1))
