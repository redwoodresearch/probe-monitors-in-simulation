"""Compare two IL resolution index versions (keys only; no outcome field read): designation fields per row (authoritative
launch_id, run_key, replacement_kind, superseded originals without their cited record), status transitions, and every
other changed field. 09's rule (via d9): no designation may change between versions; a change is an error (exit 1).
  python compare_il_index_versions.py <old.json> <new.json> <out.json>
"""

import json
import sys
from collections import Counter

old_p, new_p, out_p = sys.argv[1:]
a, b = (json.load(open(p)) for p in (old_p, new_p))
key = lambda r: (r["launch"], r["world"], r["cell"])  # noqa: E731
dups = {p: [k for k, n in Counter(map(key, d["rows"])).items() if n > 1] for p, d in ((old_p, a), (new_p, b))}  # Codex r40: fail closed
if any(dups.values()):
    sys.exit(f"duplicate row keys: { {p: v[:5] for p, v in dups.items() if v} }")
A, B = {key(r): r for r in a["rows"]}, {key(r): r for r in b["rows"]}
strip = lambda so: [{f: v for f, v in s.items() if f != "record"} for s in so]  # noqa: E731
DES = ("launch_id", "run_key", "replacement_kind")
desig = [x for x in A.keys() & B.keys() if any(A[x][f] != B[x][f] for f in DES) or strip(A[x]["superseded_originals"]) != strip(B[x]["superseded_originals"])]
trans, fields = Counter(), Counter()
for x in A.keys() & B.keys():
    diff = sorted(f for f in A[x].keys() | B[x].keys() if A[x].get(f) != B[x].get(f))
    if diff:
        fields[f"{B[x]['launch_id']}: {', '.join(diff)}"] += 1
        if A[x]["status"] != B[x]["status"]:
            trans[f"{B[x]['launch_id']}: {A[x]['status']} -> {B[x]['status']}"] += 1
res = {"old": old_p.split("results/")[-1], "new": new_p.split("results/")[-1], "rows_old": len(A), "rows_new": len(B),
       "same_row_keys": A.keys() == B.keys(), "designation_changes": len(desig), "designation_changed_rows": sorted("|".join(x) for x in desig),
       "status_transitions": dict(trans), "changed_fields_by_launch": dict(fields),
       "new_pass": b["pass"], "new_errors": b["errors"], "new_pending": b["pending"]}
res["ok"] = res["same_row_keys"] and not desig and b["pass"] is True and not b["errors"]
json.dump(res, open(out_p, "w"), indent=1)
open(out_p, "a").write("\n")
print(json.dumps({k: res[k] for k in ("ok", "rows_old", "rows_new", "designation_changes", "status_transitions")}))
sys.exit(0 if res["ok"] else 1)
