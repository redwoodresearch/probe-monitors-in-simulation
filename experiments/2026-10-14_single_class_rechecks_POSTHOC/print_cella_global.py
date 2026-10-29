"""Format run_cella_global.py's result JSON as a printout, with RULE.md Amendment 2's wording on every global reading and
0d's borderline marker (p within 0.03 of 0.05). Reads the JSON by key; computes nothing beyond comparisons of stored p's.

  uv run --no-project --python 3.12 python experiments/2026-10-14_single_class_rechecks_POSTHOC/print_cella_global.py <result json>
"""
import json
import sys

d = json.load(open(sys.argv[1]))
print(f"gate: passes {d['gate']['passes']}; {d['gate']['compared']} summary-field and run-count comparisons (observed, p, null_p95, null_mean, runs) "
      f"matched the as-run JSON exactly | producer {d['git'][:8]}")
for v in d["global"]:
    for k in ("auroc_run2_vs_honest_b", "catch_run2_at_1pct", "catch_sim_at_1pct"):
        a, g = d["within_task_asrun"][v][k], d["global"][v][k]
        f = (lambda x: f"{x:.4f}") if k.startswith("auroc") else (lambda x: f"{100 * x:.2f}")
        rg = ("beyond a label-permutation null" if g["p"] <= 0.05 else "not beyond the null") + " (global; ignores task clustering)"
        flags = [m for m, c in (("PRIMARY", (v, k) == ("unswapped_L1", "catch_run2_at_1pct")), ("BORDERLINE", abs(g["p"] - 0.05) <= 0.03 or abs(a["p"] - 0.05) <= 0.03),
                                ("READING CHANGES", (a["p"] <= 0.05) != (g["p"] <= 0.05))) if c]
        print(f"{v} {k}: observed {f(a['observed'])} (runs {a['runs']}) | within-task (as run) p {a['p']:.4f}, 95th {f(a['null_p95'])} | "
              f"global p {g['p']:.4f}, 95th {f(g['null_p95'])} | {rg}" + (" | " + ", ".join(flags) if flags else ""))
