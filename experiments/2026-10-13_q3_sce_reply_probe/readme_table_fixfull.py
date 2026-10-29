"""README table for fix-full, N and F9 (Amendments 14, 16 and 18), by key from the metrics file. Local, $0.

  python3 readme_table_fixfull.py results/sce_fixfull_metrics_POSTHOC_2026-10-13_153447.json
"""
import json
import sys

M = json.load(open(sys.argv[1]))
NAMES = {"a": "(a) real-trained", "c": "(c) as-is, mixed", "c_fixed (c′)": "(c′) fix939, mixed", "c_full (c″)": "(c″) fix-full, mixed",
         "c_N": "(c_N) newline only, mixed", "c_F9": "(c_F9) Full + F9, mixed", "b": "(b) as-is, sim-only", "b_fixed (b′)": "(b′) fix939, sim-only",
         "b_full (b″)": "(b″) fix-full, sim-only", "b_N": "(b_N) newline only, sim-only", "b_F9": "(b_F9) Full + F9, sim-only"}
pct = lambda s: f"{100 * s['value']:.1f}"
ci = lambda s: f"{100 * s['value']:+.1f} [{100 * s['ci95'][0]:+.1f}, {100 * s['ci95'][1]:+.1f}]"
cnt = lambda s: f"{pct(s)} ({s['value'] * s['n']:.1f}/{s['n']})"
for L in ("48", "63"):
    print(f"\n**Qwen3-32B last token, L{L}, targeting 1%** (5% gap beside); % (caught/runs); D = e2 catch − (a)'s, paired; rule 4b verdict\n")
    print("| cell | real attacks (e2) | fake attacks | real honest false alarms | fake honest false alarms | fake − real, points [95%] | D, points [95%] | 4b | gap at 5% |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, name in NAMES.items():
        x = M["layers"][L][key]
        a, b = x["fpr_0.01"], x["fpr_0.05"]
        print(f"| {name} | {cnt(a['real_attack_catch_e2'])} | {cnt(a['fake_attack_catch'])} | {cnt(a['real_honest_fa'])} | {cnt(a['fake_honest_fa'])} | "
              f"**{ci(a['fake_minus_real'])}** | {ci(a['D_e2_vs_a'])} | {a['rule_4b']['verdict']} | {100 * b['fake_minus_real']['value']:+.1f} |")
for L in ("48", "63"):
    n, f = M["newline_readings_1pct"][L], M["f9_readings_1pct"][L]
    print(f"\nL{L}: N reading \"{n['reading']}\" (r = {n['r_share_of_full_cut']}; c_N − c {ci(n['c_N_minus_c'])}, c_N − c″ {ci(n['c_N_minus_c_full'])}). "
          f"F9 reading \"{f['reading']}\" (s = {f['s_share_of_remaining_closed']}; c_F9 − c″ {ci(f['c_F9_minus_c_full'])}).")
