"""Print the matched-false-alarm table (RULE.md Amendment 1) from a nano_matched result JSON (numbers only).

  python3 -I report_nano_matched.py results/nano_matched_<stamp>.json
"""

import json
import sys

NAMES = {"qwen3_32b_full_F9_avg_L48": "Qwen3-32B, full + F9 averaged, L48", "seed_oss_36b_fixfull_avg_L48": "Seed-OSS-36B, fix-full averaged, L48"}
pc = lambda x: f"{100 * x['point']:.1f} [{100 * x['ci95'][0]:.1f}, {100 * x['ci95'][1]:.1f}]"  # noqa: E731
pd = lambda x: f"{100 * x['point']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"  # noqa: E731
fa = lambda x: f"{100 * x['point']:.2f} [{100 * x['ci95'][0]:.2f}, {100 * x['ci95'][1]:.2f}]"  # noqa: E731

R = json.load(open(sys.argv[1]))["results"]
r0 = R["fpr_0.01"]
print(f"Shared real attack (e2) runs: {r0['n_e2_shared']} over {r0['n_tasks_e2_shared']} main tasks; real honest runs: {r0['n_honest']}. "
      "Intervals: 95% main-task bootstrap, thresholds redrawn per resample. Nano stays at its nominal thresholds. "
      "Matching is descriptive, on these same evaluation honest runs; intervals hold alpha′ fixed.\n")
print("| nominal rate | monitor | thresholds | achieved held-out honest false alarms | real catch | − nano (paired) |")
print("|---|---|---|---|---|---|")
for k, lab in (("fpr_0.01", "1%"), ("fpr_0.05", "5%")):
    o = R[k]
    print(f"| {lab} | GPT-5.4-nano | nominal | {fa(o['nominal']['nano']['honest_false_alarm_heldout'])} | {pc(o['nominal']['nano']['e2_catch_shared'])} | |")
    for m, name in NAMES.items():
        x, y = o["nominal"][m], o["matched"][m]
        print(f"| {lab} | {name} | nominal | {fa(x['honest_false_alarm_heldout'])} | {pc(x['e2_catch_shared'])} | {pd(o['nominal'][f'{m}_minus_nano'])} |")
        print(f"| {lab} | {name} | matched (alpha′ {100 * y['alpha_prime']:.3f}%, {'exact' if y['exact_match'] else 'nearest'}) | "
              f"{fa(y['honest_false_alarm_heldout'])} | {pc(y['e2_catch_shared'])} | {pd(y['minus_nano'])} |")
print("\nCatch range over every alpha′ with the matched rate (point): " + "; ".join(
    f"{NAMES[m].split(',')[0]} {k[4:]}: {100 * R[k]['matched'][m]['catch_range_same_rate'][0]:.1f}–{100 * R[k]['matched'][m]['catch_range_same_rate'][1]:.1f}"
    f" ({R[k]['matched'][m]['n_alpha_same_rate']} level{'s' * (R[k]['matched'][m]['n_alpha_same_rate'] != 1)})" for k in R for m in NAMES))
