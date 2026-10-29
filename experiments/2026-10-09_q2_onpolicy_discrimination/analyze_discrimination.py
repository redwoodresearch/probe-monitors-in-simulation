"""One table from the three discrimination results (local, free): per comparison, TF-IDF per output and per run
(run-level AUROC, 95% task-resample interval, label-swap p), and the activation probe's six cells (3 layers x 2
readouts: median and range of run-level AUROC, and how many cells reach the smallest p, 1/21).

  uv run --no-project python experiments/2026-10-09_q2_onpolicy_discrimination/analyze_discrimination.py \
      <tfidf run-level json> <tfidf per-output json> <probe json>
"""
import json
import statistics
import sys
from pathlib import Path

run, per, probe = (json.loads(Path(p).read_text())["comparisons"] for p in sys.argv[1:4])
f = lambda r, a="AUROC", ci="95% (task resamples)": f"{r[a]:.2f} [{r[ci][0]:.2f}, {r[ci][1]:.2f}] p={r['p']:.3f}"
print("| Comparison | TF-IDF per output | TF-IDF per run | Probe, 6 cells: median AUROC [range]; cells at p = 1/21 |")
print("|---|---|---|---|")
for name in [*run, *(k for k in probe if k not in run)]:  # the scripts' comparison order, then the planted check
    base = name.removesuffix(", planted 0.5 SD shift")
    cells = probe[name].values()
    a = [c["run-level AUROC"] for c in cells]
    pr = f"{statistics.median(a):.2f} [{min(a):.2f}, {max(a):.2f}]; {sum(c['p'] <= 1 / 21 + 1e-9 for c in cells)} of 6"
    if name != base:
        print(f"| {base}, planted (probe: 0.5-SD shift; TF-IDF: fixed line) | "
              f"{f(per[base]['TF-IDF per output, planted line in the second world'], 'run-level AUROC', 'run-level 95% (task resamples)')} | "
              f"{f(run[base]['TF-IDF, planted line in the second world'])} | {pr} |")
    else:
        print(f"| {name} ({run[name]['tasks']} tasks) | {f(per[name]['TF-IDF per output'], 'run-level AUROC', 'run-level 95% (task resamples)')} | "
              f"{f(run[name]['TF-IDF'])} | {pr} |")
