"""README grid tables for Amendment 11, by key from the grid metrics file. Local, $0.

  python3 readme_table_grid.py results/sce_grid_metrics_POSTHOC_2026-10-13_135610.json
"""
import json
import sys

G = json.load(open(sys.argv[1]))
NAMES = {"a": "(a) real attack vs real honest", "b": "(b) fake attack vs fake honest", "c": "(c) fake attack vs real honest",
         "d": "(d) real attack vs fake honest", "b_fixed": "(b′) fixed fake vs edited fake honest", "c_fixed": "(c′) fixed fake attack vs real honest"}
pct = lambda s: f"{100 * s['value']:.1f}"
cell = lambda s: f"{pct(s)} ({s['count']})"
ci = lambda s: f"{100 * s['value']:+.1f} [{100 * s['ci95'][0]:+.1f}, {100 * s['ci95'][1]:+.1f}]"
for reader, title in (("qwen35", "Qwen3.5-27B reading Sol, L63"), ("qwen3", "Qwen3-32B last token, L48 and L63")):
    print(f"\n**{title}**: targeting 1% (5% gap beside); catch and false alarms in % (caught/runs); new rows refit here, others as committed\n")
    print("| training cell | layer | real attack catch (e2) | fake attack catch | real honest false alarms | fake honest false alarms | fake − real, points [95%] | at 5% |")
    print("|---|---|---|---|---|---|---|---|")
    for r in G["grid"][reader]:
        a, b = r["fpr_0.01"], r["fpr_0.05"]
        tag = " (new)" if r["source"].startswith("new") else ""
        print(f"| {NAMES[r['cell']]}{tag} | {r['layer']} | {cell(a['real_attack_catch'])} | {cell(a['fake_attack_catch'])} | {cell(a['real_honest_fa'])} | "
              f"{cell(a['fake_honest_fa'])} | **{ci(a['fake_minus_real'])}** | {100 * b['fake_minus_real']['value']:+.1f} |")
