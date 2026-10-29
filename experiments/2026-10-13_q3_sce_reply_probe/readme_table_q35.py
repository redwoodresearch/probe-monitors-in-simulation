"""README table for the Qwen3.5 check (Amendment 3), by key from its metrics file. Local, $0.

  python3 readme_table_q35.py results/sce_q35_metrics_POSTHOC_2026-10-13_131109.json
"""
import json
import sys

E1, E2, SA, SAF, H, SH = "real-attack-e1", "real-attack-e2", "sim-attack", "sim-attack-fixed", "real-honest", "sim-honest"
NAMES = {"a": "(a) real-trained", "c": "(c) mixed, SCE-built", "c_fixed": "(c′) mixed, fixed"}
M = json.load(open(sys.argv[1]))["layers"]["63"]
pct = lambda x: f"{100 * x:.1f}"
ci = lambda s: f"{pct(s['value'])} [{pct(s['ci95'][0])}, {pct(s['ci95'][1])}]"
for k, label in (("fpr_0.01", "1% (headline)"), ("fpr_0.05", "5% (beside: 1% rests on the 2nd or 3rd highest of 175–213 honest runs)")):
    print(f"\n**Qwen3.5-27B reader, L63, targeting {label}**, catch in %, 95% task-bootstrap intervals\n")
    print("| cell | real attacks (e2) | real attacks (e1) | sim attacks | fixed sim attacks | false alarms: real honest / sim honest | SCE-reported − real | e2 − e1 | null p (gap) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, name in NAMES.items():
        x = M[c][k]
        fa = " / ".join(pct(x["false_alarms"][a]["value"]) for a in (H, SH))
        print(f"| {name} | {ci(x['catch'][E2])} | {pct(x['catch'][E1]['value'])} | {pct(x['catch'][SA]['value'])} | {pct(x['catch'][SAF]['value'])} | "
              f"{fa} | **{ci(x['sce_reported_minus_real'])}** | {ci(x['yardstick_e2_minus_e1'])} | {x['null']['gap']['p_two_sided']} |")
    r = M["c_fixed"][k]["repaired_only"]
    print(f"\n(c′) on the {r['n_repaired']} repaired runs only: fixed sim attacks {ci(r['catch_fixed_sim_repaired'])}, e2 on the same "
          f"cells {ci(r['catch_e2_same_cells'])}, SCE-reported − real {ci(r['sce_reported_minus_real'])}. (c′) − (c) gap "
          f"{ci(M['c_fixed'][k]['fixed_minus_c_gap'])}.")
p = M["c"]["fpr_0.01"]["plant"]
print(f"\nPlanted cue in (c), targeting 1% (δ = 0.5 SD): sim attacks caught {ci(p['catch_sa_planted'])}, e2 {ci(p['catch_e2_planted'])}; "
      f"gap {ci(p['gap_planted'])}, shift {pct(p['shift'])} points; detected {p['detected']}; ceiling {p['uninformative_ceiling']}.")
