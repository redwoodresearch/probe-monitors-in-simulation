"""Matched check: at the same number of earlier edit outputs, does sim's gap depend on whether any was in sim format?"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
from boot import report, fmt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402
ctx = pd.read_parquet(DATA / "context_features.parquet")
K = ["arm", "key", "step", "action"]
out = {}
for reader, pos in [(r, p) for r in ("qwen3-32b", "qwen3.5") for p in ("pre_call", "prev_content")]:  # prev_content: the content-token control
    t = pd.read_json(DATA / "tables" / f"{reader}__{pos}__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
    df = t.merge(ctx, on=K, how="left", validate="one_to_one").reset_index(drop=True)
    real, sim = (df.domain == "real").to_numpy(), (df.domain == "sim").to_numpy()
    ne, seen = df.n_edit_before.to_numpy(), (df.n_simfmt_before > 0).to_numpy()
    subs = {}
    for lo, hi, lab in ((1, 1, "1"), (2, 3, "2-3"), (4, 10**6, "4+")):
        m = (ne >= lo) & (ne <= hi)
        subs[f"{lab} earlier edit outputs: sim, none in sim format"] = (real & m, sim & m & ~seen)
        subs[f"{lab} earlier edit outputs: sim, >= 1 in sim format"] = (real & m, sim & m & seen)
    # Among sim calls only, how many earlier edit outputs, and the share of runs that ever show sim format
    s = df[sim & (ne >= 1)]
    print(reader, "sim calls after >=1 edit output:", len(s), "| none in sim format:", int((s.n_simfmt_before == 0).sum()),
          "| their n_edit_before:", s[s.n_simfmt_before == 0].n_edit_before.value_counts().sort_index().head(6).to_dict(),
          "| runs:", s[s.n_simfmt_before == 0].key.nunique(), "| tasks:", s[s.n_simfmt_before == 0].main_task.nunique())
    rep = report(df, subs); out[f"{reader} {pos}"] = rep
    print(fmt(rep))
stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
(RESULTS / f"shift_format_matched_{stamp}.json").write_text(json.dumps(out, indent=1))
