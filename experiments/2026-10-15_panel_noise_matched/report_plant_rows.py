"""Values table of a run_plant_rows.py record (JSON reads only): per level, readout, probe and signed delta, the verdict, the unplanted and planted points,
the planted 95% interval, the move, whether it is in range, the margin and the mismatch flag (raw floats beside the printed strings, one shared formatter);
then the detected counts per signed delta (C, R, both). Refuses a GATE_FAIL record. Writes results/table_<record stem>.json (exclusive create).

  python3 experiments/2026-10-15_panel_noise_matched/report_plant_rows.py <record.json>
"""

import hashlib
import json
import sys
from pathlib import Path

from probe_shift.fmtpts import p2

P = Path(sys.argv[1])
b = P.read_bytes()
r = json.loads(b)
if r.get("GATE_FAIL") or r["gates"]["n_fail"]:
    raise SystemExit(f"{P.name} is a GATE_FAIL record: no values")
out = {"record": P.name, "sha256": hashlib.sha256(b).hexdigest(), "git": r["meta"]["git"], "label": r["label"], "rule": r["rule"], "comparator": r["comparator"],
       "test_expectations": r.get("test_expectations"), "rows": [], "counts": r["counts"]}
for f, by in r["rows"].items():
    for ro, labs in by.items():
        for lb, ds in labs.items():
            for tag, v in ds.items():
                ci = v.get("planted_ci95")
                out["rows"].append({"level": f, "readout": ro, "row": lb, "delta": tag, "verdict": v["verdict"], "reason": v.get("reason"), "raw": v,
                                    "printed": {"unplanted": p2(v.get("unplanted_point")), "planted": p2(v.get("planted_point")),
                                                "planted_ci95": f"[{p2(ci[0])}, {p2(ci[1])}]" if ci else "n/a", "move": p2(v.get("move_points")),
                                                "margin": p2(v.get("margin_points")), "mismatch": v.get("mechanism_mismatch") or ""}})
q_ = P.with_name(f"table_{P.stem}.json")
with q_.open("x") as fh:
    fh.write(json.dumps(out, indent=1))
print(f"{P.name} ({out['sha256'][:12]}): {r['label']}")
for x in out["rows"]:
    pr = x["printed"]
    print(f"{x['level']} {x['readout']} {x['row']:<28} {x['delta']:>3}  {x['verdict']:<13} move {pr['move']:>6}  unplanted {pr['unplanted']:>6}  planted {pr['planted']:>6} "
          f"{pr['planted_ci95']:<18} margin {pr['margin']:>6}  {pr['mismatch']}{('  (' + x['reason'] + ')') if x['reason'] else ''}")
for f, by in r["counts"].items():
    for tag, c_ in by.items():
        print(f"level {f} delta {tag}: " + "; ".join(f"{ro} detected {x['detected']} of {x['cases']} (unavailable {x['unavailable']}, mismatch {x['mechanism_mismatch']})"
                                                   for ro, x in c_.items()))
ag_ = r.get("aggregate")
if ag_:  # 09 RULE item 47 @ 1ce4a94c8: negative-shift verdicts only for the six-probe aggregate; the positive side names the ceiling members (never a subset mean)
    print(f"six-probe aggregate ({ag_['group']}; {ag_.get('rule47', {}).get('rule', 'RULE 47 not stated')}). Caption: {ag_.get('caption')}")
    for f, by in ag_["by_level"].items():
        for ro, by_d in by.items():
            for tag, e_ in by_d.items():
                for nm in ("S", "excess"):
                    v = e_[nm]
                    pl = v.get("planted") or {}
                    ci = pl.get("ci95")
                    print(f"{f} {ro} {tag:>3} {nm:<6} {v['verdict']}" + (f"  delta_exp {p2(v['delta_exp_points'])}  planted {p2(pl.get('point'))} "  # already points
                                                                       f"[{p2(ci[0]) if ci else 'n/a'}, {p2(ci[1]) if ci else 'n/a'}]" if "delta_exp_points" in v else "")
                          + (f"  unattainable members {v['unattainable_members']} (arm-rows {v['unattainable_arm_rows']})" if v.get("unattainable_members") else "")
                          + (f"  ({v['reason']})" if v.get("reason") else ""))
print(f"table: {q_.name}")
