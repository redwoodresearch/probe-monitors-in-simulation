"""Print run_fx_followups.py's checkpoint JSONs (realism RULE.md Amendment 4, descriptive) as a plain table. Reads the JSONs only.

  python print_fx_followups.py <folder holding the L{layer}__{variant}_{row}.json files>
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
NAMES = {"A": "A, gate: last token, all rows", "B": "B: last token, text_editor rows dropped", "C": "C: averaged readout (output_end_mean)"}
print("Where F11's realism move lives (Amendment 4, descriptive): summary (ii) [95% main-task bootstrap], label-shuffle null (100 within-task shuffles)")
for L in (48, 63):
    for p in sorted(d.glob(f"L{L}__*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        r = json.loads(p.read_text())
        ci = f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]" if r.get("ci95") else "[not reported]"
        null = f"null mean {r['null_mean']:.4f}, 95th {r['null_p95']:.4f}, max {r['null_max']:.4f} | p {r['p']:.4f}" if r.get("null_mean") is not None else "null not reported"
        print(f"L{L} {NAMES[r['variant']]:42s} {r['row']:10s} {r['summary_ii']:.4f} {ci} | {null} | runs (sim, real) {r['runs']} | runs without rows {r['runs_without_rows']}")
        if "gate_passes" in r:
            c = r["loader_check"]
            print(f"    gate: as run {r['as_run']:.4f}, passes {r['gate_passes']} | real honest runs bit-exact to d9's pack {c['real_honest_runs_bit_exact_to_d9']} of {c['of']}")
        if "minus_full_F9" in r:
            lo, hi = r["minus_full_F9_ci95"]
            print(f"    + F11 minus full + F9: {r['minus_full_F9']:+.4f} [{lo:+.4f}, {hi:+.4f}] (paired, same bootstrap draws)")
