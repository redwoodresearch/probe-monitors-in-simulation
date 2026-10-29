"""Print the README's tables from a date-histogram result (local, no Modal): year shares, each arm's most common
days (share of runs with a date) and hour-of-day shares.

  python3 experiments/2026-10-08_q2_date_histogram/summarize_dates.py experiments/2026-10-08_q2_date_histogram/results/dates_2026-10-08_162309.json
"""
import collections
import json
import sys

with open(sys.argv[1]) as fh:
    r = json.load(fh)
print("arm | runs | runs with a date | dates | share by year (≥0.5%)")
for a in sorted(r["by year"]):
    n = r["dates per arm"][a]
    print(f"{a} | {r['runs per arm'][a]} | {r['runs with a date'].get(a, 0)} | {n} | "
          + ", ".join(f"{y} {c / n:.1%}" for y, c in r["by year"][a].items() if c / n >= 0.005))
print("\narm | top modal days (share of runs with a date)")
for a, d in sorted(r["most common day per run"].items()):
    c = collections.Counter(d.values())
    print(f"{a} | " + ", ".join(f"{k} {v / len(d):.0%}" for k, v in c.most_common(6)))
print("\narm | hour of day (share of dates; 'none' = no time)")
for a, h in sorted(r["by hour"].items()):
    n = sum(h.values())
    top = sorted(((v, k) for k, v in h.items() if k not in ("none", "")), reverse=True)[:5]
    print(f"{a} | none {(h.get('none', 0) + h.get('', 0)) / n:.0%}; " + ", ".join(f"{k}h {v / n:.1%}" for v, k in top))
