"""Readings of a run455 record, by script (tables only; no recomputation): coverage, the post hoc ceiling test, run-level gaps,
source ratios. Writes results/<record stem>_readings.json and prints markdown tables.

  python3 -I analyze_common455.py results/common_run455_<stamp>.json
"""

import hashlib
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
b = p.read_bytes()
rec = json.loads(b)
P = rec["probes"]
pct = lambda x: None if x is None else round(100 * x, 1)  # noqa: E731


def ci(s):
    return f"{pct(s['point'])} [{pct(s['ci95'][0])}, {pct(s['ci95'][1])}]" if s.get("ci95") else f"{pct(s['point'])} [withheld]"


out = {"record": p.name, "record_sha256": hashlib.sha256(b).hexdigest(), "design": rec.get("design"), "coverage": {}, "run_level": {}, "source_ratio": {}}
print("## Coverage (cells with run1, run2 and sim rows; pool units by source)\n")
print("| probe | S_old cells | S_new cells | pool units S_old (incl. relaunch) | pool units S_new | pool runs missing (S_old, S_new) | status |")
print("|---|---|---|---|---|---|---|")
for lb, v in P.items():
    c = v["coverage"]
    out["coverage"][lb] = c
    print(f"| {lb} | {c['cohort_cells_with_all_three']['S_old']} | {c['cohort_cells_with_all_three']['S_new']} | {c['pool_units']['S_old']} | "
          f"{c['pool_units']['S_new']} | {c['pool_runs_missing']['n_missing']['S_old']}, {c['pool_runs_missing']['n_missing']['S_new']} | {v.get('status', 'complete')} |")
print("\n## Run level, registered comparator, pooled [95%] (S_old / S_new points; run2 - run1 pooled)\n")
print("| probe | comparator | 1% | 2% | 5% | 10% |")
print("|---|---|---|---|---|---|")
for lb, v in P.items():
    r = v["run"]
    if "points" not in r:
        print(f"| {lb} | | withheld: {r.get('withheld')} | | | |")
        continue
    cells, rows = [], {}
    for q in r["points"]:
        d = q["blocks"]["all"]["diff"][v["registered"]]
        so, sn = q["blocks"]["S_old"]["diff"][v["registered"]]["point"], q["blocks"]["S_new"]["diff"][v["registered"]]["point"]
        y = q["blocks"]["all"]["diff"]["run2_minus_run1"]["point"]
        cells.append(f"{ci(d)} ({pct(so)} / {pct(sn)}; {pct(y)})")
        rows[str(q["alpha"])] = {"pooled": d, "S_old": so, "S_new": sn, "run2_minus_run1": y,
                                 "real_mean_catch": q["blocks"]["all"]["tpr"]["real_mean"]["point"], "sim_catch": q["blocks"]["all"]["tpr"]["sim"]["point"],
                                 "realized_fa_pool": q["realized_fa_pool"]}
    out["run_level"][lb] = rows
    print(f"| {lb} | {v['registered']} | " + " | ".join(cells) + " |")
print("\n## Source ratio (unit level, f >= 1e-3; tolerance a factor of 2)\n")
for lb, v in P.items():
    sr = v.get("source_ratio")
    if isinstance(sr, dict):
        out["source_ratio"][lb] = {"range": sr["range_f_ge_1e-3"], "within": sr["within_tolerance"]}
        print(f"- {lb}: range {sr['range_f_ge_1e-3']}, within tolerance: {sr['within_tolerance']}")
cel = P.get("Sol (a) R", {}).get("run", {}).get("ceiling_post_hoc")
if cel:
    out["ceiling"] = cel
    print("\n## post hoc: ceiling test (0d/Aswin ~10:0x): Sol (a) reply, run level, catch-matched per origin\n")
    print("| origin | target catch | achieved | pooled run FPR | sim − real (mean) [95%] | sim − run1 | sim − run2 | honest alarm (own-origin pool) |")
    print("|---|---|---|---|---|---|---|---|")
    for o in ("S_old", "S_new"):
        for q in cel[o]:
            fa = q["honest_alarm_rate_by_source_world"]
            own = {k: v for k, v in fa.items() if k.startswith(o) and k.endswith("honest_b")}
            print(f"| {o} | {pct(q['target_real_mean_catch'])} | {pct(q['achieved_real_mean_catch'])} | {pct(q['pooled_run_fpr_in_sample'])} | "
                  f"{ci(q['diff']['sim_minus_real_mean'])} | {ci(q['diff']['sim_minus_run1'])} | {ci(q['diff']['sim_minus_run2'])} | "
                  + ", ".join(f"{k}: {pct(v['rate'])}% of {v['runs']}" for k, v in own.items()) + " |")
    print("\n| target | S_old − S_new: sim − real (mean) | sim − run1 | sim − run2 |")
    print("|---|---|---|---|")
    for q in cel["S_old_minus_S_new_at_matched_catch"]:
        print(f"| {pct(q['target'])} | {ci(q['sim_minus_real_mean'])} | {ci(q['sim_minus_run1'])} | {ci(q['sim_minus_run2'])} |")
dest = p.with_name(p.stem + "_readings.json")
dest.open("x").write(json.dumps(out, indent=1, default=float))
print("\n", dest.name, hashlib.sha256(dest.read_bytes()).hexdigest())
