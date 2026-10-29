"""Gate check for the 7310476ae batch: leaf-by-leaf diff (NaN == NaN, |a - b| <= 1e-12 equal) of each new record's rows against the record it
supersedes; per row, the number of differing leaves and their path classes. Usage: analyze_batch_gates.py <results dir> <new>=<old> ..."""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


def flat(o, pre=""):
    if isinstance(o, dict):
        return {k2: v2 for k_, v_ in o.items() for k2, v2 in flat(v_, f"{pre}/{k_}").items()}
    if isinstance(o, list):
        return {k2: v2 for i_, v_ in enumerate(o) for k2, v2 in flat(v_, f"{pre}[{i_}]").items()}
    return {pre: o}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12
    return a == b


def rows(rec):
    out = {}
    for lb, v in rec["rows"].items():
        if isinstance(v, dict) and set(v) <= {"S_old", "S_new"} and v:
            out |= {f"{lb}|{o}": r for o, r in v.items()}
        else:
            out[lb] = v
    return out


def cls(p):
    for name, rx in (("fold_minus1_draws (new field)", r"^/fold_minus1_draws"), ("fold -1 rule text", r"fold_minus1_rule|fold_minus1_per_pool|moments_source"),
                     ("n_per_batch label", r"^/n_per_batch"), ("draw-0 per-run alarm share", r"per_run_alarm_share"),
                     ("finite_draws", r"finite_draws$"), ("interval", r"ci95"), ("point", r"/point$")):
        if re.search(rx, p):
            return name
    return "other"


R = Path(sys.argv[1])
res = {}
for pair in sys.argv[2:]:
    n_, o_ = pair.split("=")
    a, b = rows(json.loads((R / n_).read_text())), rows(json.loads((R / o_).read_text()))
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            res[f"{n_}::{k}"] = {"missing_in": "new" if k not in a else "old"}
            continue
        fa, fb = flat(a[k]), flat(b[k])
        d = [p for p in set(fa) | set(fb) if not same(fa.get(p, "<absent>"), fb.get(p, "<absent>"))]
        res[f"{n_}::{k}"] = {"n_differ": len(d), "classes": dict(Counter(cls(p) for p in d)), "other_paths": sorted(p for p in d if cls(p) == "other")[:12]}
print(json.dumps(res, indent=1))
