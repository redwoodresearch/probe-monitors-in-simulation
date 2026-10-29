"""09's (d) exclusion label, recomputed by script from a run_peraction311 record with unavailable scenarios reported separately (Codex r2
on the 13:37/13:41 records; the records stay unchanged, append-only). Writes results/<record stem>_dlabels.json.

  python3 -I relabel_peraction311.py results/peraction311_all_C_<stamp>.json
"""

import hashlib
import json
import sys
from pathlib import Path


def rd(s):
    return "unavailable" if not s.get("ci95") else "resolved" if (s["ci95"][0] > 0 or s["ci95"][1] < 0) else "not resolved"


def label_d(lab_):
    av = {sc: v for sc, v in lab_.items() if sc != "primary" and v != "unavailable"}
    un = sorted(sc for sc in ("high", "low") if sc not in av)
    if lab_["primary"] == "unavailable" or not av:
        return "comparison unavailable"
    if any(v != lab_["primary"] for v in av.values()):
        return "differs under an exclusion scenario"
    return "same under both exclusion scenarios" if not un else f"same under the available scenario ({', '.join(un)} unavailable)"


p = Path(sys.argv[1])
b = p.read_bytes()
rec = json.loads(b)
out = {"record": p.name, "record_sha256": hashlib.sha256(b).hexdigest(), "rule": "09 (d) @ ed7932987; unavailable scenarios separate (Codex r2)", "labels": {}}
for lb, v in rec["rows"].items():
    for f, q in v["points"]["joint"].items():
        g = q["gap"]
        lab_ = {"primary": rd(g["delta_full"]), "high": rd(g["delta_full_high"]), "low": rd(g["delta_full_low"])}
        out["labels"].setdefault(lb, {})[f] = {"readings": lab_, "label": label_d(lab_),
                                               "keys": [f"rows.{lb}.points.joint.{f}.gap.{k}.ci95" for k in ("delta_full", "delta_full_high", "delta_full_low")]}
        print(lb, f, out["labels"][lb][f]["label"], lab_)
dest = p.with_name(p.stem + "_dlabels.json")
dest.open("x").write(json.dumps(out, indent=1))
print(dest.name, hashlib.sha256(dest.read_bytes()).hexdigest())
