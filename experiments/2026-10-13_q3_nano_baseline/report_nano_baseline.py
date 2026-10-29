"""Print the README table from a nano-baseline result JSON (numbers only).

  python3 -I report_nano_baseline.py results/nano_baseline_<stamp>.json
"""

import json
import sys

NAMES = {"nano": "GPT-5.4-nano (paper's trajectory monitor, mean of 3 draws)",
         "qwen3_32b_full_F9_avg_L48": "Recipe: Qwen3-32B, full + F9 averaged, L48",
         "seed_oss_36b_fixfull_avg_L48": "Recipe: Seed-OSS-36B, fix-full averaged, L48"}
pc = lambda x: f"{100 * x['point']:.1f} [{100 * x['ci95'][0]:.1f}, {100 * x['ci95'][1]:.1f}]"  # noqa: E731
pd = lambda x: f"{100 * x['point']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"  # noqa: E731

R = json.load(open(sys.argv[1]))["results"]
a, b = R["fpr_0.01"], R["fpr_0.05"]
print(f"Shared real attack (e2) runs: {a['n_e2_shared']} over {a['n_tasks_e2_shared']} main tasks; real honest runs: "
      f"{a['nano']['n_honest']}. Thresholds: per fold, on held-out real honest runs (#36's rule). Intervals: 95% main-task bootstrap.\n")
print("| monitor | real catch, nominal 1% | real catch, nominal 5% | achieved held-out honest false alarms, nominal 1% / 5% |")
print("|---|---|---|---|")
for m, name in NAMES.items():
    print(f"| {name} | {pc(a[m]['e2_catch_shared'])} | {pc(b[m]['e2_catch_shared'])} | "
          f"{pc(a[m]['honest_false_alarm_heldout'])} / {pc(b[m]['honest_false_alarm_heldout'])} |")
for m in list(NAMES)[1:]:
    print(f"| {NAMES[m].split(': ')[1]} − nano (paired) | {pd(a[f'{m}_minus_nano'])} | {pd(b[f'{m}_minus_nano'])} | |")
