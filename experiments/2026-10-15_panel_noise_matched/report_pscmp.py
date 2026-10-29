"""Values table of a pooled PANEL record's 'pooled vs registered per-set (item 43)' comparison (09 #8; JSON reads only): per combo, level and readout,
one line per quantity: pooled and per-set point [95%], the half-width ratio, sign flip, the 'includes 0' change, and the paired difference (or 'unpaired').
Writes results/table_pscmp_<record stem>.json (exclusive create) and prints it. Release order: 0d's recheck first; nothing goes back to the pooled builders.

  python3 experiments/2026-10-15_panel_noise_matched/report_pscmp.py <panel record.json>
"""

import hashlib
import json
import sys
from pathlib import Path

from probe_shift.fmtpts import raw_and_printed

P = Path(sys.argv[1])
b = P.read_bytes()
r = json.loads(b)
c = r.get("per_set_comparison")
if not c or c.get("label") != "pooled vs registered per-set (item 43)":
    raise SystemExit("STOP: the record has no 'pooled vs registered per-set (item 43)' block")
f = raw_and_printed  # raw floats + one shared 2-dp formatter (no pre-rounding); values are in points (x100)
pd_ = lambda d_: f(d_) if isinstance(d_, dict) else d_  # 'unpaired' stays a string
out = {"record": P.name, "sha256": hashlib.sha256(b).hexdigest(), "git": r["meta"]["git"], "label": c["label"], "per_set_record": c["per_set_record"],
       "paired": c["paired"], "paired_basis": c["paired_basis"], "y_definition": c["y_definition"], "lines": []}
for cb, by_pt in c["by_combo"].items():
    for pt, by_sel in by_pt.items():
        for sel, ent in by_sel.items():
            for e in ent:
                if "withheld" in e:  # incomplete readable membership (run_pooled per_set_compare): the reason, never a value
                    out["lines"].append({"combo": cb, "level": pt, "readout": sel, "quantity": e["quantity"], "withheld": e["withheld"]})
                    continue
                out["lines"].append({"combo": cb, "level": pt, "readout": sel, "quantity": e["quantity"], "pooled": f(e["pooled"]), "per_set": f(e["per_set"]),
                                     "half_width_ratio": e["half_width_ratio_pooled_over_per_set"], "sign_flip": e["sign_flip"], "includes_0": e["includes_0"],
                                     "includes_0_changes": e["includes_0_changes"], "paired_difference": pd_(e["paired_difference_pooled_minus_per_set"])})
q = P.with_name(f"table_pscmp_{P.stem}.json")
q.open("x").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
