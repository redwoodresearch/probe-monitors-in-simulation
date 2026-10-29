"""Insert natural experiment: sim runs whose inserts so far were all real-style vs those that saw a diff-style insert."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
import examples as ex
from boot import report, fmt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402
REVIEW = "Review the changes and make sure they are as expected"
q = ex.table("qwen3-32b")
runs = ex.load_runs(set(q.key))
rows, census = [], {"real": {"real-style": 0, "diff": 0, "other": 0}, "sim": {"real-style": 0, "diff": 0, "other": 0}}
dom = dict(zip(q.key, q.domain))
for key, calls in runs.items():
    n_rs = n_df = n_edit = 0
    for s in sorted({o[0] for o in calls}):
        step_calls = sorted(o for o in calls if o[0] == s)
        for sa in step_calls:  # a turn's calls are written before any of their outputs
            rows.append({"key": key, "step": sa[0], "action": sa[1], "n_ins_real_style": n_rs, "n_ins_diff": n_df, "n_edit": n_edit})
        for sa in step_calls:
            c = calls[sa]
            if c["tool"] == "text_editor":
                cmd, o = ex.editor_args(c)[0], c["output"] or ""
                n_edit += cmd in ("create", "str_replace", "insert")
                if cmd == "insert":
                    rs = "result of running `cat -n` on a snippet" in o[:300]
                    df = "Here's a diff showing the changes made" in o[:300]
                    census[dom[key]]["real-style" if rs else "diff" if df else "other"] += 1
                    n_rs += rs; n_df += df
print("insert output census:", census)
ins = pd.DataFrame(rows)
K = ["key", "step", "action"]
out = {}
for reader, pos in [(r, p) for r in ("qwen3-32b", "qwen3.5") for p in ("pre_call", "prev_content")]:  # prev_content: the content-token control
    t = pd.read_json(DATA / "tables" / f"{reader}__{pos}__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
    df = t.merge(ins, on=K, how="left", validate="one_to_one").reset_index(drop=True)
    assert df.n_edit.notna().all()
    real, sim = (df.domain == "real").to_numpy(), (df.domain == "sim").to_numpy()
    any_ins = ((df.n_ins_real_style + df.n_ins_diff) > 0).to_numpy()
    rs_only = ((df.n_ins_real_style > 0) & (df.n_ins_diff == 0)).to_numpy()
    saw_diff = (df.n_ins_diff > 0).to_numpy()
    ne = df.n_edit.to_numpy()
    subs = {}
    for lo, hi, lab in ((1, 10**6, "any"), (1, 3, "1-3"), (4, 10**6, "4+")):
        m = (ne >= lo) & (ne <= hi)
        subs[f"{lab} earlier edits, sim inserts all real-style"] = (real & m & any_ins, sim & m & rs_only)
        subs[f"{lab} earlier edits, sim saw a diff insert"] = (real & m & any_ins, sim & m & saw_diff)
    s = df[sim & rs_only]
    print(reader, "sim calls with real-style-only insert history:", len(s), "runs", s.key.nunique(), "tasks", s.main_task.nunique())
    rep = report(df, subs); out[f"{reader} {pos}"] = rep
    print(fmt(rep))
stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
(RESULTS / f"shift_insert_experiment_{stamp}.json").write_text(json.dumps({"census": census, **out}, indent=1))
