"""Ties in each run's most common date (local, no Modal): per arm, how many runs with a date have a tied most common
date (Counter keeps the first-seen one), and the README's modal-day table and global overlap with and without those
runs. Reads a date-histogram result that records "runs whose most common date is tied" (runs from 2026-10-09 on).

  uv run --no-project python experiments/2026-10-08_q2_date_histogram/analyze_ties.py experiments/2026-10-08_q2_date_histogram/results/dates_visible_2026-10-09_131207.json
"""

import collections
import json
import sys

with open(sys.argv[1]) as fh:
    r = json.load(fh)
modes, tied = (
    r["most common day per run"],
    {a: set(v) for a, v in r["runs whose most common date is tied"].items()},
)
print(sys.argv[1].rsplit("/", 1)[-1])
print(
    "arm | runs with a date | tied | top modal days, all runs | top modal days, tied runs left out"
)
kept = {}
for a, d in sorted(modes.items()):
    kept[a] = {k: v for k, v in d.items() if k not in tied.get(a, set())}
    top = [
        ", ".join(
            f"{k} {v / len(x):.0%}"
            for k, v in collections.Counter(x.values()).most_common(3)
        )
        for x in (d, kept[a])
    ]
    print(
        f"{a} | {len(d)} | {len(tied.get(a, ()))} ({len(tied.get(a, ())) / len(d):.1%}) | {top[0]} | {top[1]}"
    )
changes = []
for a in sorted(modes):
    for b in sorted(modes):
        if a != b:
            v = [
                sum(day in set(m[b].values()) for day in m[a].values()) / len(m[a])
                for m in (modes, kept)
            ]
            changes.append((abs(v[1] - v[0]), f"{a} vs {b}: {v[0]:.1%} -> {v[1]:.1%}"))
print(
    "global overlap (share of a's runs whose most common date is some b run's), largest change from leaving out tied runs:",
    max(changes)[1],
)
