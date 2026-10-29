"""Print run_fx_realism.py's checkpoint JSONs (realism RULE.md Amendment 3) as a plain table. Reads the JSONs only.

  python print_fx_results.py <folder holding the L{layer}__fx_{row}.json files>
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
print("Realism rows for F10 and F11 (Amendment 3): summary (ii) [95% main-task bootstrap], label-shuffle null (100 within-task shuffles)")
for L in (48, 63):
    for i in range(3):
        p = d / f"L{L}__fx_{i}.json"
        if not p.exists():
            print(f"L{L} row {i}: not computed")
            continue
        r = json.loads(p.read_text())
        print(f"L{L} {r['row']:16s} {r['summary_ii']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | null mean {r['null_mean']:.4f}, 95th {r['null_p95']:.4f}, "
              f"max {r['null_max']:.4f} | p {r['p']:.4f} | runs (sim, real) {r['runs']} | runs swapped in {r['runs_swapped_in']}, new runs without rows {r['new_runs_without_rows']}")
        if "gate_passes" in r:
            c = r["loader_check"]
            print(f"    loader gate: as run {r['as_run']:.4f}, passes {r['gate_passes']} | runs bit-exact to d9's full + F9 rows {c['runs_bit_exact_to_d9']} of 437 "
                  f"(same row count {c['runs_with_same_row_count_as_d9']}; max |diff| where counts match {c['max_abs_diff_where_counts_match']})")
        else:
            lo, hi = r["minus_full_F9_q2_ci95"]
            print(f"    minus full + F9 (q2): {r['minus_full_F9_q2']:+.4f} [{lo:+.4f}, {hi:+.4f}] (paired, same bootstrap draws)")
