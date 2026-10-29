"""t325 rebuild: re-read each new record's leaf diff with NaN == NaN and |a - b| <= 1e-12 as equal; classify every remaining path."""
import json
import math
import re
import sys
from pathlib import Path

R = Path(sys.argv[1])


def get(o, p):
    for k in p.strip("/").split("/"):
        m = re.fullmatch(r"(.*?)((\[\d+\])*)", k)
        o = o[m.group(1)] if m.group(1) else o
        for i in re.findall(r"\[(\d+)\]", m.group(2)):
            o = o[int(i)]
    return o


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12
    return a == b


HA = re.compile(r"@A|/A/|/A$|delta_rr|same_attack|H_A|ha25f")
HD = re.compile(r"hotdeck|exclusion_label|mapped_units_per_fold")
out = {}
for rd in ("C", "R", "X"):
    (fn,) = sorted(R.glob(f"peraction311_all_t325_{rd}_*.json"))
    new = json.loads(fn.read_text())
    for tag, d in new["repro_ha25f"]["diffs"].items():
        old = json.loads((R / d["record"].split(" @ ")[0]).read_text())["rows"]
        for lb, v in d["rows"].items():
            real, cats = [], {"H_A": 0, "hotdeck_or_label (plain only)": 0, "new field origin": 0, "float <= 1e-12 or NaN==NaN": 0, "OTHER": []}
            for p in v["paths"]:
                try:
                    a = get(new["rows"][lb], p)
                except (KeyError, IndexError, TypeError):
                    a = "<absent>"
                try:
                    b = get(old[lb], p)
                except (KeyError, IndexError, TypeError):
                    b = "<absent>"
                if p == "/origin":
                    cats["new field origin"] += 1
                elif same(a, b):
                    cats["float <= 1e-12 or NaN==NaN"] += 1
                elif HA.search(p):
                    cats["H_A"] += 1
                elif tag == "plain" and (HD.search(p) or p in ("/fold_minus1_rule", "/moments_source", "/moments_source/H_B", "/moments_source/H_sim", "/moments_source/H_A")):
                    cats["hotdeck_or_label (plain only)"] += 1
                else:
                    cats["OTHER"].append([p, a, b])
            out[f"{rd}|{tag}|{lb}"] = cats
print(json.dumps(out, indent=0, default=str))
