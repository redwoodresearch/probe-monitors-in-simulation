"""1(a) history and the edit-format natural experiment: attack view, trained on real, both readers; pre_call and its content-token control prev_content."""
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
bins = lambda n: pd.cut(n, [-1, 0, 1, 3, 1e9], labels=["0", "1", "2-3", "4+"]).astype(str)
out = {}
for reader, pos in [(r, p) for r in ("qwen3-32b", "qwen3.5") for p in ("pre_call", "prev_content")]:  # prev_content: the content-token control
    t = pd.read_json(DATA / "tables" / f"{reader}__{pos}__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
    n = len(t)
    df = t.merge(ctx, on=K, how="left", validate="one_to_one").reset_index(drop=True)
    assert len(df) == n and df.next_kind.notna().all()
    real, sim = (df.domain == "real").to_numpy(), (df.domain == "sim").to_numpy()
    subs = {"all": (real, sim)}
    for col, label in (("n_editor_before", "earlier editor outputs"), ("n_editor_invented_before", "earlier invented editor outputs"),
                       ("n_edit_before", "earlier edit outputs")):
        b = bins(df[col]).to_numpy()
        for v in ("0", "1", "2-3", "4+"):
            subs[f"{label} {v}"] = (real & (b == v), sim & (b == v))
            for pf in ("none", "exact"):
                pfm = (df.prev_flag == pf).to_numpy()
                subs[f"{label} {v}, previous output {pf}"] = (real & (b == v) & pfm, sim & (b == v) & pfm)
    # Natural experiment: after an edit output, real-format vs sim-format previous output in sim.
    edit_prev = df.prev_cmd.isin(["str_replace", "insert", "create"]).to_numpy()
    sf = df.prev_simfmt.astype("boolean").fillna(False).to_numpy(bool)
    subs["previous output an edit: sim in sim format"] = (real & edit_prev, sim & edit_prev & sf)
    subs["previous output an edit: sim in real format"] = (real & edit_prev, sim & edit_prev & ~sf)
    for cmd in ("str_replace", "create"):
        m = (df.prev_cmd == cmd).to_numpy()
        subs[f"previous output {cmd}: sim in sim format"] = (real & m, sim & m & sf)
        subs[f"previous output {cmd}: sim in real format"] = (real & m, sim & m & ~sf)
    # History: has the run shown any sim-format edit output yet? (real runs: never)
    seen = (df.n_simfmt_before > 0).to_numpy()
    had_edit = (df.n_edit_before > 0).to_numpy()
    subs["after >= 1 edit output: sim with no sim-format output yet"] = (real & had_edit, sim & had_edit & ~seen)
    subs["after >= 1 edit output: sim with >= 1 sim-format output"] = (real & had_edit, sim & had_edit & seen)
    rep = report(df, subs)
    out[f"{reader} {pos}"] = rep
    print(f"\n### {reader} {pos}\n" + fmt(rep))
stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
(RESULTS / f"shift_history_{stamp}.json").write_text(json.dumps(out, indent=1))
