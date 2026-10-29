"""Amendment 2 and Amendment 3 side by side, every value read from the two records and formatted here (no hand rounding).

  python3 experiments/2026-10-14_cmd_readout_simh_control/report_a2_a3.py results/controls_a2_<stamp>.json results/controls_a3_<stamp>.json
"""

import hashlib
import sys
from pathlib import Path

import json

HERE = Path(__file__).resolve().parent
CELLS = ("reply|L48", "command|L63", "reply|L63", "command|L48")  # primary first
PROBS = (("c_simatk_vs_run1", "(c) sim vs run1"), ("d_run2_vs_run1", "(d) run2 vs run1"))


def f3(x):
    return f"{x:.3f}"


def ci(w):
    return f"[{w[0]:.3f}, {w[1]:.3f}]"


def sci(w):
    return f"[{w[0]:+.3f}, {w[1]:+.3f}]"


def main(a2p: str, a3p: str):
    recs = {}
    for tag, p in (("A2", a2p), ("A3", a3p)):
        b = (HERE / p).read_bytes()
        recs[tag] = json.loads(b)
        print(f"{tag}: {p} sha256 {hashlib.sha256(b).hexdigest()} git {recs[tag]['git'][:9]} gaps {recs[tag]['required_gaps']} failed {recs[tag]['failed']}"
              f" consistency {all(v['equal'] for v in recs[tag]['consistency_with_a1'].values())} epochs {recs[tag]['epochs_per_batch']}")
    print("\n| Cell | Contrast | Set | Runs (1/0) | TPR@1% k/n [Wilson] | null p95, p | TPR@5% k/n [Wilson] | null p95, p | AUROC (p) | call AUROC | call TPR@5% |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        for p, nm in PROBS:
            for tag in ("A2", "A3"):
                r = recs[tag]["cells"][cell]["problems"][p]
                rl, cl, n = r["run_level"], r["call_level"], r["null"]
                c1, c5 = rl["catch_0.01"], rl["catch_0.05"]
                n1, n5, na = n["run_level_catch_0.01_summary"], n["run_level_catch_0.05_summary"], n["run_level_auroc_summary"]
                print(f"| {cell} | {nm} | {tag} | {r['n_runs']['class1']}/{r['n_runs']['class0']} | {c1['k_caught']}/{c1['n']} = {f3(c1['tpr'])} {ci(c1['tpr_wilson95'])} | "
                      f"{f3(n1['p95'])}, {n1['p_add_one']:.4f} | {c5['k_caught']}/{c5['n']} = {f3(c5['tpr'])} {ci(c5['tpr_wilson95'])} | {f3(n5['p95'])}, {n5['p_add_one']:.4f} | "
                      f"{f3(rl['auroc'])} ({na['p_add_one']:.4f}) | {f3(cl['auroc'])} | {f3(cl['catch_0.05']['tpr'])} |")
    print("\n| Cell | Set | Bootstrap tasks | c−d run AUROC [95%] | c−d run TPR@5% [95%] | c−d call AUROC [95%] | Rule met |")
    print("|---|---|---|---|---|---|---|")
    for cell in CELLS:
        for tag in ("A2", "A3"):
            c = recs[tag]["cells"][cell]
            pc, rc = c["paired_c_minus_d"], c["problems"]["c_simatk_vs_run1"]
            o = pc["observed"]
            rule = (rc["null"]["run_level_catch_0.05_summary"]["p_add_one"] <= 0.01 and rc["null"]["run_level_auroc_summary"]["p_add_one"] <= 0.01
                    and o["run_auroc"]["ci95"][0] > 0 and o["run_catch_0.05"]["ci95"][0] > 0)
            print(f"| {cell} | {tag} | {pc['n_tasks']} (failed draws {pc['failed_draws']}) | {o['run_auroc']['c_minus_d']:+.3f} {sci(o['run_auroc']['ci95'])} | "
                  f"{o['run_catch_0.05']['c_minus_d']:+.3f} {sci(o['run_catch_0.05']['ci95'])} | {o['call_auroc']['c_minus_d']:+.3f} {sci(o['call_auroc']['ci95'])} | {'yes' if rule else 'no'} |")
    print("\nA3 fits:")
    for cell in CELLS:
        c = recs["A3"]["cells"][cell]
        fits = [x for x in c["certificates"] if "n_evals" in x]
        print(f"  {cell}: {len(fits)} fits, skipped {sum('skipped' in x for x in c['certificates'])}, uncertified columns {sum(len(x['uncertified_columns']) for x in fits)}, "
              f"max grad {max(x['max_grad'] for x in fits):.2e}, at the evaluation cap {sum(x['n_evals'] == 6251 for x in fits)}, seconds {c['seconds']}, runs without rows {len(c['runs_without_rows'])}")
    print("A3 realized FPR (run level, run1):", {cell: [recs["A3"]["cells"][cell]["problems"]["c_simatk_vs_run1"]["run_level"][f"catch_{a}"]["class0_above"] for a in ("0.01", "0.05")] for cell in CELLS},
          "of n_class0", recs["A3"]["cells"]["reply|L48"]["problems"]["c_simatk_vs_run1"]["run_level"]["catch_0.01"]["n_class0"])
    print("A3 watchdog log:", recs["A3"]["watchdog_log"], "coverage:", json.dumps(recs["A3"]["coverage"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
