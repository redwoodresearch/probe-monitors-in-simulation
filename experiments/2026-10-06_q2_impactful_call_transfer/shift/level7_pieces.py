"""How often each piece that only level 7 normalizes appears in editor outputs, real vs sim (all 1,072 attack runs)."""
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import examples as ex  # noqa: E402
from paths import RESULTS, stamp as now  # noqa: E402

q = ex.table("qwen3-32b")
runs = ex.load_runs(set(q.key))
dom = dict(zip(q.key, q.domain))
REVIEW = "Review the changes and make sure they are as expected"
c = collections.defaultdict(collections.Counter)
for key, calls in runs.items():
    d = dom[key]
    for cl in calls.values():
        if cl["tool"] != "text_editor" or not cl["output"]: continue
        cmd, o = ex.editor_args(cl)[0], cl["output"]
        if cmd not in ("str_replace", "insert", "view"): continue
        k = (d, cmd); c[k]["n"] += 1
        c[k]["review note"] += REVIEW in o
        c[k]["diff preamble"] += "Here's a diff showing the changes made" in o[:300]
        c[k]["cat -n snippet preamble"] += "result of running `cat -n` on a snippet" in o[:300]
        lines = o.splitlines()
        c[k]["removed lines (sum)"] += sum(1 for l in lines if re.match(r"^\s*\d*\|?-(?!--)", l))
        c[k]["bare + lines (sum)"] += sum(1 for l in lines if re.match(r"^\s*\d*\|?\+\s*$", l))
t = pd.DataFrame(c).T.fillna(0).astype(int)
for col in t.columns:
    if col != "n": t[col + " /output"] = (t[col] / t["n"]).round(3)
print(t[[x for x in t.columns if x == "n" or x.endswith("/output")]].sort_index().to_string())
(RESULTS / f"shift_level7_pieces_{now()}.json").write_text(json.dumps({"|".join(k): v for k, v in t.to_dict("index").items()}, indent=1))
