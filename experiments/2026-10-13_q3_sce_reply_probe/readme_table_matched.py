"""README table for T3 (Amendment 6) and its full-calibration variant (Amendment 9), by key. Local, $0.

  python3 readme_table_matched.py results/sce_matched_metrics_POSTHOC_2026-10-13_132839.json \\
      results/sce_matched_fullcal_metrics_POSTHOC_2026-10-13_133110.json
"""
import json
import sys

T3, FC = (json.load(open(p)) for p in sys.argv[1:3])
pct = lambda x: f"{100 * x:.1f}"
rng = lambda s: f"{pct(s['mean'])} ({pct(s['min'])}–{pct(s['max'])})"
VERSIONS = (("subsample calibration (T3)", T3, {"reduced": "subsample (primary)", "full": "all runs (beside)"}),
            ("full calibration (Amendment 9)", FC, {"reduced": "reduced test set", "full": "full test set"}))
for k, label in (("fpr_0.01", "1% (headline)"), ("fpr_0.05", "5% (beside)")):
    print(f"\n**Trained on 183-task subsamples, targeting {label}**: draw mean (min–max of 5 draws), %; false alarms with counts (draw 0)\n")
    print("| cell | calibration | test set | real attacks (e2) | sim attacks | SCE-reported − real | real honest false alarms | sim honest false alarms | e2 − e1 | AUROC e2 / sim |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c, name in (("a", "(a) real-trained"), ("c", "(c) SCE-built")):
        for vname, M, views in VERSIONS:
            for ts, key in views.items():
                x = M["cells"][c][k][key]
                d0 = x["counts_per_draw"][0]
                fa = f"{rng(x['false_alarms_real_honest'])}; {d0['caught']['real-honest']:g}/{d0['n']['real-honest']}, calibration {d0['calibration_honest_per_threshold']['mean']:g} per threshold"
                print(f"| {name} | {vname} | {ts} | {rng(x['catch_e2'])} | {rng(x['catch_own_sim'])} | **{rng(x['sce_reported_minus_real'])}** | {fa} | "
                      f"{rng(x['false_alarms_sim_honest'])} | {rng(x['yardstick_e2_minus_e1'])} | {x['auroc_e2']['mean']:.3f} / {x['auroc_own_sim']['mean']:.3f} |")
p = T3["cells"]["c"]["fpr_0.01"]["subsample (primary)"]["plant"]
print(f"\nPlanted cue in (c), 1%, subsample calibration: shift {rng(p['shift'])} points, ≥ 10 in {p['draws_with_shift_ge_0.10']} of 5 draws. "
      f"T3 reading: {T3['h3_reading']}. Amendment 9 reading: {FC['readings_1pct']['reading']} (full size {pct(FC['readings_1pct']['gap_full_size_amendment3'])}, "
      f"subsample calibration {pct(FC['readings_1pct']['gap_subsample_calibration_t3'])}, full calibration {pct(FC['readings_1pct']['gap_full_calibration_this'])}).")
