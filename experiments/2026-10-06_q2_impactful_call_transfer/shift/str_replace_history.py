"""Is the gap after a str_replace output like the gap after an insert output? (pre_call and its control prev_content;
saved scores only; descriptive.) Real str_replace returns a numbered diff with no review note; the September simulator
returns a diff without the numbered gutter, also with no note. Real insert returns a `cat -n` snippet plus the note;
the simulator a diff.
A. By the output the reader saw last (the previous turn's last call): str_replace, insert, create, view, not an editor.
B. By what the run has shown so far, matched on earlier edits (1-3, 4+): >= 1 str_replace and no insert yet,
   >= 1 insert, only creates.
C. After a str_replace: the margin (score minus the frozen threshold) by the call's own kind and label, real vs sim,
   to tell an offset from a different mix of next calls.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import examples as ex  # noqa: E402
from boot import fmt, report  # noqa: E402
from paths import DATA, RESULTS, stamp as now  # noqa: E402

q = ex.table("qwen3-32b")
runs = ex.load_runs(set(q.key))
rows = []
for key, calls in runs.items():
    n = {"insert": 0, "str_replace": 0}
    for s in sorted({o[0] for o in calls}):
        step_calls = sorted(o for o in calls if o[0] == s)
        for sa in step_calls:  # a turn's calls are written before any of their outputs
            rows.append({"key": key, "step": sa[0], "action": sa[1], "n_insert_before": n["insert"], "n_str_replace_before": n["str_replace"]})
        for sa in step_calls:
            cmd = ex.editor_args(calls[sa])[0] if calls[sa]["tool"] == "text_editor" else None
            if cmd in n:
                n[cmd] += 1
hist = pd.DataFrame(rows)
ctx = pd.read_parquet(DATA / "context_features.parquet")
K = ["arm", "key", "step", "action"]
out = {}
for reader, pos in [(r, p) for r in ("qwen3-32b", "qwen3.5") for p in ("pre_call", "prev_content")]:
    t = pd.read_json(DATA / "tables" / f"{reader}__{pos}__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
    df = t.merge(ctx, on=K, how="left", validate="one_to_one").merge(hist, on=["key", "step", "action"], how="left", validate="one_to_one").reset_index(drop=True)
    assert df.n_insert_before.notna().all()
    real, sim = (df.domain == "real").to_numpy(), (df.domain == "sim").to_numpy()
    pc, ne = df.prev_cmd.to_numpy(), df.n_edit_before.to_numpy()
    ni, ns = df.n_insert_before.to_numpy(), df.n_str_replace_before.to_numpy()
    subs = {}
    for c in ("str_replace", "insert", "create", "view"):
        subs[f"A. last output: {c}"] = (real & (pc == c), sim & (pc == c))
    subs["A. last output: str_replace, no insert yet"] = (real & (pc == "str_replace") & (ni == 0), sim & (pc == "str_replace") & (ni == 0))
    subs["A. last output: not an editor call"] = (real & pd.isna(pc), sim & pd.isna(pc))
    for lo, hi, lab in ((1, 3, "1-3"), (4, 10**6, "4+")):
        m = (ne >= lo) & (ne <= hi)
        subs[f"B. {lab} edits, >= 1 str_replace, no insert yet"] = (real & m & (ns > 0) & (ni == 0), sim & m & (ns > 0) & (ni == 0))
        subs[f"B. {lab} edits, >= 1 insert"] = (real & m & (ni > 0), sim & m & (ni > 0))
        subs[f"B. {lab} edits, only creates"] = (real & m & (ns == 0) & (ni == 0), sim & m & (ns == 0) & (ni == 0))
    rep = report(df, subs)
    s = df[df.prev_cmd == "str_replace"].assign(margin=lambda d: d.score - d.threshold)
    mix = s.groupby(["next_kind", "y", "domain"]).agg(median_margin=("margin", "median"), n=("margin", "size"), flagged=("flag", "mean"))
    rep["C. after a str_replace, by next-call kind and label"] = [{**dict(zip(mix.index.names, i)), **{k: round(float(v), 4) for k, v in r.items()}}
                                                                   for i, r in mix.iterrows()]
    out[f"{reader} {pos}"] = rep
    print(f"\n### {reader} {pos}\n" + fmt({k: v for k, v in rep.items() if not k.startswith("C.")}))
    print(mix.unstack("domain").round(2).to_string())
(RESULTS / f"shift_str_replace_history_{now()}.json").write_text(json.dumps(out, indent=1))
