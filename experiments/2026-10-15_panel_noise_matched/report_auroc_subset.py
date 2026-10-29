"""Values table of a PANEL record's AUROC-subset sentence (JSON reads only): per view (the headline combo and each optional combo) and readout, the subset's
S, Y, mean |Y| and excess with 95% joint intervals and N; and a DRAW-0 check that the record's six-probe and off-policy-5 S, Y and excess equal a reference
record's (same Stage A draws). Writes results/table_<record stem>_auroc.json (exclusive create).

  python3 experiments/2026-10-15_panel_noise_matched/report_auroc_subset.py <record.json> <reference record.json> [level]
"""

import hashlib
import json
import sys
from pathlib import Path

from probe_shift.fmtpts import raw_and_printed

P, Q = Path(sys.argv[1]), Path(sys.argv[2])
pt = sys.argv[3] if len(sys.argv) > 3 else "0.01"
b = P.read_bytes()
r, ref = json.loads(b), json.loads(Q.read_bytes())
co, cq = r["coheadline"], ref["coheadline"]
f = raw_and_printed  # raw floats + one shared 2-dp formatter (no pre-rounding)
views = [("pooled", co, cq)] + [(v["set"], v, cq["other_combos"][k]) for k, v in (co.get("other_combos") or {}).items()]
out = {"record": P.name, "sha256": hashlib.sha256(b).hexdigest(), "git": r["meta"]["git"], "label": r["pooled_label"], "auroc": co.get("auroc_subset"), "level": pt, "views": [], "draw0_check": []}
for set_, d, dq in views:
    for sel in ("C", "R"):
        a = d[pt][sel]["auroc_subset"]
        out["views"].append({"set": set_, "readout": sel, "rows": a["rows"], "N_tasks_per_row": sorted({r["results"][set_][pt]["rows"][lb].get("n_cells") for lb in a["rows"]}),
                             "S": f(a.get("S")), "Y": f(a.get("Y")), "mean_abs_Y": f(a.get("mean_abs_Y")), "excess": f(a.get("excess")), "unavailable": a.get("unavailable"), "label": a.get("label")})
        for key in ("six", "off5"):
            for st in ("S", "Y"):
                x_, y_ = d[pt][sel][key][st]["point"], dq[pt][sel][key][st]["point"]
                out["draw0_check"].append({"set": set_, "readout": sel, "group": key, "stat": st, "this": x_, "reference": y_, "equal": x_ == y_})
            g_ = r["results"][set_][pt]["groups"].get(f"off_policy|{sel}", {}).get("primary", {}).get("mean_G_minus_mean_Y", {}).get("point")
            gq = ref["results"][set_][pt]["groups"].get(f"off_policy|{sel}", {}).get("primary", {}).get("mean_G_minus_mean_Y", {}).get("point")
        out["draw0_check"].append({"set": set_, "readout": sel, "group": "off5", "stat": "excess", "this": g_, "reference": gq, "equal": g_ == gq})
out["draw0_all_equal"] = all(x["equal"] for x in out["draw0_check"])
q_ = P.with_name(f"table_{P.stem}_auroc.json")
q_.open("x").write(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("record", "sha256", "git", "draw0_all_equal")} | {"views": out["views"]}, indent=1))
