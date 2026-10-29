"""Readings of a ~710 record, by script (field reads only; no recomputation): coverage and pools per probe and source, source ratio,
run level by origin, 21j verdicts per probe x pair, the old-rule rows, the 455 reproduction, H1 versions. Writes
results/<record stem>_readings.json and prints markdown tables.

  python3 -I analyze_common710.py results/common_run710_<stamp>.json
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


out = {"record": p.name, "record_sha256": hashlib.sha256(b).hexdigest(), "design": {k: v for k, v in rec["design"].items() if k != "status_rows"},
       "repro_455": rec.get("repro_455_readings"), "coverage": {}, "pools": {}, "source_ratio": {}, "run_level": {}, "verdicts": {}, "old_rule": rec.get("old_rule"), "H1": {}}
print("## Coverage (cells with run1, run2 and sim) and pools (runs / units by source)\n")
print("| probe | S_old | S_new | S_newmains (without S4) | 4r | relaunch | pool by source | pool keys without rows | status |")
print("|---|---|---|---|---|---|---|---|---|")
for lb, v in P.items():
    c = v["coverage"]
    out["coverage"][lb] = {k: c[k] for k in ("cells_with_all_three", "cohort_cells", "n_missing_cells")} | {"S4r_included": v["S4r_included"]}
    out["pools"][lb] = v["pool_by_source"]
    pk = {s: c["pool_keys_without_rows"][s]["n"] for s in c["pool_keys_without_rows"]}
    print(f"| {lb} | {c['cells_with_all_three']['S_old']} | {c['cells_with_all_three']['S_new']} | {c['cells_with_all_three']['S_newmains']} "
          f"({c['cells_with_all_three']['S_newmains_without_S4']}) | {v['S4r_included']} | {v['relaunch']} | "
          + "; ".join(f"{s} {x['runs']}/{x['units']}" for s, x in v["pool_by_source"].items()) + f" | {pk} | {v.get('status', 'complete')} |")
print("\n## Source ratio (unit level, f >= 1e-3; factor of 2)\n")
for lb, v in P.items():
    sr = v.get("source_ratio")
    if isinstance(sr, dict):
        out["source_ratio"][lb] = {"range": sr["range_f_ge_1e-3"], "within": sr["within_tolerance"]}
        print(f"- {lb}: range {sr['range_f_ge_1e-3']}, within tolerance: {sr['within_tolerance']}")
print("\n## Run level, sim − real (mean) by block [95%] at 5% and 1%\n")
blocks = ["all", "S_old", "S_new", "S_newmains", "all_without_S4", "S_newmains_without_S4"]
print("| probe | α | " + " | ".join(blocks) + " | run2 − run1 (all) |")
print("|---|---|" + "---|" * (len(blocks) + 1))
for lb, v in P.items():
    r = v["run"]
    if "points" not in r:
        continue
    out["run_level"][lb] = {}
    for q in r["points"]:
        out["run_level"][lb][str(q["alpha"])] = {bk: {"sim_minus_real_mean": q["blocks"][bk]["diff"]["sim_minus_real_mean"], "registered": q["blocks"][bk]["diff"][v["registered"]],
                                                      "real_mean": q["blocks"][bk]["tpr"]["real_mean"]["point"], "sim": q["blocks"][bk]["tpr"]["sim"]["point"],
                                                      "n_cells": q["blocks"][bk]["n_cells"]} for bk in blocks} | {"run2_minus_run1": q["blocks"]["all"]["diff"]["run2_minus_run1"],
                                                                                                                 "realized_fa_pool": q["realized_fa_pool"]}
        out["verdicts"].setdefault(lb, {})[str(q["alpha"])] = q.get("verdicts_21j")
        if q["alpha"] in (0.05, 0.01):
            print(f"| {lb} | {pct(q['alpha'])}% | " + " | ".join(ci(q["blocks"][bk]["diff"]["sim_minus_real_mean"]) for bk in blocks)
                  + f" | {ci(q['blocks']['all']['diff']['run2_minus_run1'])} |")
    for q in v.get("run_7b", {}).get("points", []):
        out["verdicts"][lb][f"7b|{q['alpha']}"] = q["verdicts_21j"]
print("\n## 21j verdicts at 5% (D point [90%] [95%] verdict)\n")
pairs = ["S_old|S_new", "S_old|S_newmains", "S_new|S_newmains", "S_old|S_newmains_without_S4", "S_new|S_newmains_without_S4"]
print("| probe | 7b S_old|S_new | " + " | ".join(pairs) + " |")
print("|---|---|" + "---|" * len(pairs))
fmt = lambda t: (f"{pct(t['D_point'])} [{pct(t['ci90'][0])}, {pct(t['ci90'][1])}] [{pct(t['ci95'][0])}, {pct(t['ci95'][1])}] {t['verdict']}"  # noqa: E731
                 if t and t.get("ci90") else (t or {}).get("verdict"))
for lb in out["verdicts"]:
    v5 = out["verdicts"][lb].get("0.05") or {}
    b7 = (out["verdicts"][lb].get("7b|0.05") or {}).get("S_old|S_new")
    print(f"| {lb} | {fmt(b7)} | " + " | ".join(fmt(v5.get(pr)) for pr in pairs) + " |")
print("\n## Old rule (pool-if-match, TOST ±5), for comparison only\n")
for k, v in (rec.get("old_rule") or {}).items():
    for q in v["points"]:
        print(f"- {k} at {pct(q['alpha'])}%: gate {q['gate']}; {q['n_fail']} of {q['n_tests']} tests fail; old presentation: {q['old_presentation']}; differs from 21j: {q['differs_from_21j']}")
print("\n## H1 versions\n")
for k in [k for k in rec if k.startswith("H1_")]:
    h = rec[k]
    if "points" not in h:
        out["H1"][k] = h
        print(f"- {k}: {h}")
        continue
    rd = [q for q in h["points"] if "difference_sim_minus_rerun" in q]
    out["H1"][k] = {"probes": h["probes"], "n_cells": h["n_cells"], "n_readable": len(rd),
                    "points": [{"fpr_target": q["fpr_target"], "sim": q["mean_abs_sim_minus_real"], "rerun": q["mean_abs_run2_minus_run1"],
                                "difference": q["difference_sim_minus_rerun"]} for q in rd]}
    print(f"- {k}: {len(h['probes'])} probes, n_cells {h['n_cells']}, readable points {len(rd)}" + (f", identical_to_without_S4 {h['identical_to_without_S4']}" if "identical_to_without_S4" in h else ""))
print(f"\n455 reproduction: {rec.get('repro_455_readings')}")
dest = p.with_name(p.stem + "_readings.json")
dest.open("x").write(json.dumps(out, indent=1, default=float))
print("\n", dest.name, hashlib.sha256(dest.read_bytes()).hexdigest())
