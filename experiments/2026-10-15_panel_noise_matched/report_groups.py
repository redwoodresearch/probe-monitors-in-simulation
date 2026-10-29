"""Values table of a PANEL record's headline group per set (JSON reads only): for each set and readout, the headline group's mean G, mean Y and excess with
95% intervals, its member rows and each member's own n (n_cells), the off-policy-5 group beside, and the on-policy rows' separate statistics. Writes
results/table_<record stem>_<level>.json (exclusive create) and prints it.

  python3 experiments/2026-10-15_panel_noise_matched/report_groups.py <record.json> [level] [set ...]
"""

import hashlib
import json
import sys
from pathlib import Path

from probe_shift.fmtpts import raw_and_printed

P = Path(sys.argv[1])
pt = sys.argv[2] if len(sys.argv) > 2 else "0.01"
b = P.read_bytes()
r = json.loads(b)
sets = sys.argv[3:] or list(r["results"])
hg = r["headline_group"]
f = raw_and_printed  # raw floats + one shared 2-dp formatter (no pre-rounding)
out = {"record": P.name, "sha256": hashlib.sha256(b).hexdigest(), "git": r["meta"]["git"], "level": pt, "headline_group": hg, "label": r["pooled_label"], "sets": {}}
for s in sets:
    res = r["results"][s][pt]
    for sel in ("C", "R", "R+C"):
        for gname in (hg, "off_policy"):
            g = (res["groups"].get(f"{gname}|{sel}") or {}).get("primary")
            if g is None:
                continue
            out["sets"].setdefault(s, {}).setdefault(sel, {})[gname] = {"rows": g.get("rows"), "n_cells_per_row": {lb: res["rows"][lb].get("n_cells") for lb in g.get("rows") or []},
                                                                        "mean_G": f(g.get("mean_G")), "mean_Y": f(g.get("mean_Y")), "excess": f(g.get("mean_G_minus_mean_Y")),
                                                                        "unavailable": g.get("unavailable")}
    out["sets"].setdefault(s, {})["onpolicy_separate"] = {lb: {"G_registered": f(q["G_registered"]), "sim_minus_run1": f(q["signed_sim_minus_run1"]),
                                                               "sim_minus_run2": f(q["signed_sim_minus_run2"]), "delta_full": f(q.get("signed_delta_full")), "n_cells": q["n_cells"], "caption": q["caption"]}
                                                          for lb, q in res.get("onpolicy_separate", {}).items() if "LOO" in lb}
q_ = P.with_name(f"table_{P.stem}_{pt}.json")
q_.open("x").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
