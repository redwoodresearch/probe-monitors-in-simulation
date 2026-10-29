"""Collect the subset readings (all / out217 / in217) into one table, every number read by key from the result files."""
import glob, hashlib, json, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
HERE = Path(__file__).parent; R = HERE / "results"
def one(pat):
    f = sorted(glob.glob(str(R / pat)))
    assert len(f) == 1, (pat, f)
    return Path(f[0])
f2 = lambda v: f"{v[0]:+.2f} [{v[1][0]:+.2f}, {v[1][1]:+.2f}]"
rows, files = [], {}
for sub in ("all", "out217", "in217"):
    pr = one(f"subset_per_repair_{sub}_*.json"); p4 = one(f"subset_reply_delta_P4_{sub}_*.json"); p5 = one(f"subset_reply_delta_P5_{sub}_*.json")
    ll = one(f"subset_llama_plant_delta_{sub}_*.json"); yy = one(f"subset_y_{sub}_*.json")
    for p in (pr, p4, p5, ll, yy):
        files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    a = json.loads(pr.read_text()); d4 = json.loads(p4.read_text()); d5 = json.loads(p5.read_text()); dl = json.loads(ll.read_text()); y = json.loads(yy.read_text())
    for rate in ("0.05", "0.01"):
        for rd in ("Qwen3-32B", "Olmo-3.1-32B"):
            R_ = a[rd]["rates"][rate]["rows (points; 95% paired task bootstrap)"]; sh = a[rd]["rates"][rate]["share of the as-is gap closed"]
            cells = a[rd].get("cells in subset", a[rd]["cells"])
            rows.append({"subset": sub, "rate": rate, "stat": f"{rd}: real e2 - as simulated", "value": R_["real e2 - as simulated"], "cells": cells})
            rows.append({"subset": sub, "rate": rate, "stat": f"{rd}: real e2 - both repairs", "value": R_["real e2 - both repairs"], "cells": cells})
            rows.append({"subset": sub, "rate": rate, "stat": f"{rd}: edit effect, both repairs - as simulated", "value": R_["edit effect: both repairs - as simulated"], "cells": cells})
            for arm in ("insert repair alone", "str_replace repair alone", "both repairs"):
                rows.append({"subset": sub, "rate": rate, "stat": f"{rd}: share closed, {arm}", "value": sh[arm]["share of the as-is gap closed (point)"], "cells": cells})
        for name, dd in (("P4 (#36) delta e2", d4), ("P5 (Olmo) delta e2", d5), ("Llama row 3 delta e2", dl)):
            e = dd["rates"][rate]["real-attack-e2"]
            rows.append({"subset": sub, "rate": rate, "stat": name, "value": [e["delta (points)"], e["95%"]], "cells": e["units"]})
    pr_ = y["primary"]; mae = y["secondary"]["MAE_reader"]
    rows.append({"subset": sub, "rate": "0.01", "stat": "row 2: D = MAE_reader(B3) - MAE_reader(x + 1.432)", "value": [pr_["D = MAE_reader(B3) - MAE_reader(x + 1.432)"], pr_["95% (percentile, joint task bootstrap)"]],
                 "verdict": pr_["verdict"], "cells": {k: v["retained cells"] for k, v in y["configurations"].items() if v}})
    for m in ("B3", "planting (calibrated)", "planting (identity)", "B0"):
        rows.append({"subset": sub, "rate": "0.01", "stat": f"row 2: MAE_reader {m}", "value": mae[m]})
out = {"plan": "PLAN.md (this folder)", "files sha256": files, "rows": rows}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
with open(dest := R / f"summary_{stamp}.json", "x") as fh:
    fh.write(json.dumps(out, indent=1) + "\n")
# printed table: stat x subset at each rate
from collections import defaultdict
tab = defaultdict(dict)
for r in rows:
    v = r["value"]; s = f2(v) if isinstance(v, list) else (f"{v:+.3f}" if isinstance(v, float) else str(v))
    if "verdict" in r: s += f" ({r['verdict']})"
    tab[(r["stat"], r["rate"])][r["subset"]] = s
for (stat, rate), d in tab.items():
    print(f"{rate} | {stat} | 939: {d.get('all')} | 722: {d.get('out217')} | 217: {d.get('in217')}")
print(dest)
