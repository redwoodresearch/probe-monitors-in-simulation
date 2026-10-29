"""Values table of a PANEL record's signed co-headline (JSON reads only): per view (the headline combo and each optional combo), probe set (off-policy 5;
off-policy 5 + the on-policy probe) and readout: S, Y, mean |Y|, the registered excess (from the record's groups, cross-checked where both exist) and N
tasks (the member rows' n_cells). Writes results/table_<record stem>.json (exclusive create) and prints it.

  python3 experiments/2026-10-15_panel_noise_matched/report_coheadline.py <record.json> [level] [readout]
"""

import hashlib
import json
import sys
from pathlib import Path

from probe_shift.fmtpts import raw_and_printed

P = Path(sys.argv[1])
pt, sel = (sys.argv[2] if len(sys.argv) > 2 else "0.01"), (sys.argv[3] if len(sys.argv) > 3 else "C")
b = P.read_bytes()
r = json.loads(b)
co, res = r["coheadline"], r["results"]
hg = r["headline_group"]
f = raw_and_printed  # raw floats + one shared 2-dp formatter (no pre-rounding)
views = [("pooled", co["set"], co[pt][sel])] + [(v["set"], f"{v['set']}: {v['label']}", v[pt][sel]) for v in (co.get("other_combos") or {}).values()]
out = {"record": P.name, "sha256": hashlib.sha256(b).hexdigest(), "git": r["meta"]["git"], "level": pt, "readout": sel, "label": r["pooled_label"], "views": []}
for set_, name, e in views:
    g = res[set_][pt]["groups"]
    for key, grp in (("off5", "off_policy"), ("six", hg)):
        blk = e[key]
        ex_rec = (g.get(f"{grp}|{sel}") or {}).get("primary", {}).get("mean_G_minus_mean_Y")
        ex_co = blk.get("excess")
        if ex_co and ex_rec and (abs(ex_co["point"] - ex_rec["point"]) > 1e-9 or ex_co.get("ci95") != ex_rec.get("ci95")):
            raise SystemExit(f"STOP: excess in the co-headline block != the group's ({set_} {key})")
        n = sorted({res[set_][pt]["rows"][lb].get("n_cells") for lb in blk["rows"]})
        out["views"].append({"view": name, "set": set_, "probes": key, "rows": blk["rows"], "S": f(blk.get("S")), "Y": f(blk.get("Y")), "mean_abs_Y": f(blk.get("mean_abs_Y")),
                             "excess": f(ex_rec or ex_co), "N_tasks_per_row": n, "unavailable": blk.get("unavailable")})
q = P.with_name(f"table_{P.stem}_{pt}_{sel}.json")
q.open("x").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
