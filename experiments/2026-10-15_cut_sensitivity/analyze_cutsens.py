"""Tables from a cutsens455 record (read-only formatting of the record's values; no recomputation).

  python3 experiments/2026-10-15_cut_sensitivity/analyze_cutsens.py results/cutsens455_<stamp>.json
"""

import json
import sys
from pathlib import Path

rec_p = Path(sys.argv[1])
rec = json.loads(rec_p.read_text())


def pct(s):
    return f"{100 * s['point']:+.1f} [{100 * s['ci95'][0]:+.1f}, {100 * s['ci95'][1]:+.1f}]" if s["ci95"] else f"{100 * s['point']:+.1f}"


def cat(s):
    return f"{100 * s['point']:.1f}"


lines = [f"# Cut sensitivity tables ({rec_p.name}; PRELIMINARY, POST HOC; values copied from the record)", "",
         "gap = sim − real by the probe's registered comparator; rerun = run 2 − run 1; real = mean(run 1, run 2); points.", ""]
for unit_level in (False, True):
    lines += [f"## {'Per action (unit level)' if unit_level else 'Per run'}", "",
              "| probe | variant | block | FPR | n cells | real | sim | gap (registered) | rerun | sim − real mean |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in rec["probes"].items():
        reg = r["registered"]
        if unit_level:
            if "unit" not in r or "full" not in r["unit"]:
                continue
            variants = [("full", r["unit"]["full"]), ("untruncated", r["unit"]["untruncated"]), ("first K", r["unit"]["first_k"])]
            for vname, pts in variants:
                for q in pts:
                    for b in ("combined", "S_old", "S_new"):
                        if b not in q["blocks"]:
                            continue
                        x = q["blocks"][b]
                        lines.append(f"| {lab} | {vname} | {'all' if b == 'combined' else b} | {q['fpr_target']:.0e} | {x['n_cells']} | {cat(x['tpr']['real_mean'])} | "
                                     f"{cat(x['tpr']['sim'])} | {pct(x['diff'][reg])} | {pct(x['diff']['run2_minus_run1'])} | {pct(x['diff']['sim_minus_real_mean'])} |")
        else:
            for vname, key in (("full", "full"), ("untruncated", "untruncated"), ("first K", "first_k_run")):
                v = r.get(key, {})
                if "points" not in v:
                    lines.append(f"| {lab} | {vname} | – | – | – | – | – | {v.get('withheld', 'missing')} | – | – |")
                    continue
                for q in v["points"]:
                    for b in ("all", "S_old", "S_new"):
                        x = q["blocks"][b]
                        lines.append(f"| {lab} | {vname} | {b} | {q['alpha']:.0%} | {x['n_cells']} | {cat(x['tpr']['real_mean'])} | {cat(x['tpr']['sim'])} | "
                                     f"{pct(x['diff'][reg])} | {pct(x['diff']['run2_minus_run1'])} | {pct(x['diff']['sim_minus_real_mean'])} |")
    lines.append("")
lines += ["## Gates and first-K sizes", "", "| probe | gate | run values | unit values | first K: median K, rows dropped / attack rows, runs in file order |", "|---|---|---|---|---|"]
for lab, r in rec["probes"].items():
    g, fk = r["gate"], r.get("first_k", {})
    lines.append(f"| {lab} | {'pass' if g['pass'] else 'FAIL ' + str(g.get('mismatches', []) + g.get('unit_mismatches', []))[:120]} | {g.get('values_compared')} | "
                 f"{g.get('unit_values_compared')} | {fk.get('K_median')}, {fk.get('attack_rows_dropped')}/{fk.get('attack_rows')}, {fk.get('runs_in_file_order')} |")
out = rec_p.with_name(rec_p.stem + "_tables.md")
out.write_text("\n".join(lines) + "\n")
print(out)
