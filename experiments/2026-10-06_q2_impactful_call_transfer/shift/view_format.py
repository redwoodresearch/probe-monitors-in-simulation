"""How do text_editor view outputs look, real vs sim? Share with a tab gutter, with a '|' gutter, and first lines."""
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
stats = collections.defaultdict(collections.Counter); first = collections.defaultdict(list)
for key, calls in runs.items():
    for (s, a), c in calls.items():
        if c["tool"] != "text_editor" or not c["output"]: continue
        cmd = ex.editor_args(c)[0]
        o = c["output"]; d = dom[key]
        lines = o.splitlines()
        stats[(d, cmd)]["n"] += 1
        stats[(d, cmd)]["tab gutter"] += bool(re.search(r"^\s*\d+\t", o, re.M))
        stats[(d, cmd)]["pipe gutter"] += bool(re.search(r"^\s*\d+\|", o, re.M))
        stats[(d, cmd)]["starts 'Here's the result of running `cat -n`'"] += o.startswith("Here's the result of running `cat -n`")
        stats[(d, cmd)]["ends with newline"] += o.endswith("\n")
        if cmd == "view" and len(first[(d, cmd)]) < 4: first[(d, cmd)].append(repr(o[:260]))
tab = pd.DataFrame(stats).T.fillna(0).astype(int).sort_index()
print(tab.to_string())
(RESULTS / f"shift_editor_output_format_{now()}.json").write_text(json.dumps({"|".join(k): v for k, v in tab.to_dict("index").items()}, indent=1))
for k, v in first.items():
    print("\n", k); print("\n".join(v))
