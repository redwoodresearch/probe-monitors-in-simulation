"""Does the gap (pre_call, and its control prev_content) depend on whether the run has shown an insert output yet? Matched on earlier edits."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
import examples as ex
from boot import report, fmt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402
q = ex.table("qwen3-32b")
runs = ex.load_runs(set(q.key))
rows = []
for key, calls in runs.items():
    n_ins = 0
    for s in sorted({o[0] for o in calls}):
        step_calls = sorted(o for o in calls if o[0] == s)
        for sa in step_calls:  # a turn's calls are written before any of their outputs
            rows.append({"key": key, "step": sa[0], "action": sa[1], "n_insert_before": n_ins})
        n_ins += sum(calls[sa]["tool"] == "text_editor" and ex.editor_args(calls[sa])[0] == "insert" for sa in step_calls)
ins = pd.DataFrame(rows)
ctx = pd.read_parquet(DATA / "context_features.parquet")
K = ["arm", "key", "step", "action"]
out = {}
for reader, pos in [(r, p) for r in ("qwen3-32b", "qwen3.5") for p in ("pre_call", "prev_content")]:  # prev_content: the content-token control
    t = pd.read_json(DATA / "tables" / f"{reader}__{pos}__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
    df = t.merge(ctx, on=K, how="left", validate="one_to_one").merge(ins, on=["key", "step", "action"], how="left", validate="one_to_one").reset_index(drop=True)
    assert df.n_insert_before.notna().all()
    real, sim = (df.domain == "real").to_numpy(), (df.domain == "sim").to_numpy()
    ne, ni = df.n_edit_before.to_numpy(), df.n_insert_before.to_numpy()
    subs = {}
    for lo, hi, lab in ((1, 3, "1-3"), (4, 10**6, "4+")):
        m = (ne >= lo) & (ne <= hi)
        subs[f"{lab} earlier edits, no insert yet"] = (real & m & (ni == 0), sim & m & (ni == 0))
        subs[f"{lab} earlier edits, >= 1 insert"] = (real & m & (ni > 0), sim & m & (ni > 0))
    rep = report(df, subs); out[f"{reader} {pos}"] = rep
    print(f"\n### {reader} {pos}\n" + fmt(rep))
stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
(RESULTS / f"shift_insert_history_{stamp}.json").write_text(json.dumps(out, indent=1))
