"""First lines of text_editor edit outputs (str_replace, insert, create), real vs sim: the editor-format tell."""
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
heads = collections.defaultdict(collections.Counter); ex_ = collections.defaultdict(list)
for key, calls in runs.items():
    for c in calls.values():
        if c["tool"] != "text_editor" or not c["output"]: continue
        cmd = ex.editor_args(c)[0]
        if cmd not in ("str_replace", "create", "insert"): continue
        o = c["output"]
        head = re.sub(r"/\S+", "<path>", o.split("\n")[0])[:90]
        heads[(dom[key], cmd)][head] += 1
        if len(ex_[(dom[key], cmd)]) < 2: ex_[(dom[key], cmd)].append(repr(o[:500]))
template = lambda h: re.split(r"[`\"]|: \S", h)[0][:90]  # the tool's fixed wording only, no file content
tmpl = collections.defaultdict(collections.Counter)
for k, c in heads.items():
    for h, n in c.items():
        tmpl[k][template(h)] += n
(RESULTS / f"shift_edit_output_templates_{now()}.json").write_text(json.dumps({"|".join(k): v.most_common(10) for k, v in tmpl.items()}, indent=1))
for k in sorted(heads):
    print(k, sum(heads[k].values())); [print("   ", n, h) for h, n in heads[k].most_common(4)]
for k in sorted(ex_):
    print("\n", k); print("\n".join(ex_[k]))
