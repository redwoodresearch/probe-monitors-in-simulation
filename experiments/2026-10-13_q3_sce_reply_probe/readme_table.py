"""README tables from the result files, by key (no hand-copied numbers). Local, $0.

  python3 readme_table.py results/sce_metrics_POSTHOC_2026-10-13_123007.json results/sce_equal_metrics_2026-10-13_123204.json
"""
import json
import sys

E1, E2, SA, SAF, H, SH, SHF = "real-attack-e1", "real-attack-e2", "sim-attack", "sim-attack-fixed", "real-honest", "sim-honest", "sim-honest-edited"
NAMES = {"a": "(a) real-trained", "b": "(b) sim-trained", "c": "(c) mixed, SCE-built", "b_fixed": "(b′) sim-trained, fixed",
         "c_fixed": "(c′) mixed, fixed"}
M, Q = (json.load(open(p)) for p in sys.argv[1:3])
pct = lambda x: f"{100 * x:.1f}"
ci = lambda s: f"{pct(s['value'])} [{pct(s['ci95'][0])}, {pct(s['ci95'][1])}]"
for k, label in (("fpr_0.01", "1% (headline)"), ("fpr_0.05", "5% (beside: 1% rests on the 2nd or 3rd highest of 175–213 honest runs)")):
    print(f"\n**Targeting {label}**, catch in %, 95% task-bootstrap intervals\n")
    print("| layer | cell | real attacks (e2) | real attacks (e1) | sim attacks | fixed sim attacks | false alarms: real honest / sim honest / edited sim honest | SCE-reported − real | e2 − e1 | null p (gap) | equal-count gap, mean (min–max of 5 draws) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for L in ("48", "63"):
        for c, name in NAMES.items():
            x, q = M["layers"][L][c][k], Q["layers"][L][c][k]["sce_reported_minus_real"]
            fa = " / ".join(pct(x["false_alarms"][a]["value"]) for a in (H, SH, SHF))
            print(f"| {L} | {name} | {ci(x['catch'][E2])} | {pct(x['catch'][E1]['value'])} | {pct(x['catch'][SA]['value'])} | "
                  f"{pct(x['catch'][SAF]['value'])} | {fa} | **{ci(x['sce_reported_minus_real'])}** | {ci(x['yardstick_e2_minus_e1'])} | "
                  f"{x['null']['gap']['p_two_sided']} | {pct(q['mean'])} ({pct(q['min'])}–{pct(q['max'])}) |")
print("\n**Planted cue, cell (b), targeting 1%** (δ = 0.5 SD on every sim attack output)\n")
for L in ("48", "63"):
    p = M["layers"][L]["b"]["fpr_0.01"]["plant"]
    print(f"- L{L}: sim attacks caught {ci(p['catch_sa_planted'])}, e2 {ci(p['catch_e2_planted'])}; gap {ci(p['gap_planted'])}, "
          f"shift {pct(p['shift'])} points; detected {p['detected']}.")
print("\n**Repairs, targeting 1%** (fixed − unfixed SCE-reported − real gap)\n")
for L in ("48", "63"):
    for c in ("b", "c"):
        r = M["layers"][L][f"{c}_fixed"]["fpr_0.01"][f"fixed_minus_{c}_gap"]
        print(f"- L{L} ({c}′) − ({c}): {ci(r)} points; repairs_rescue {r['repairs_rescue']}.")
