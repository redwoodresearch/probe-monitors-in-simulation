"""q2's str_replace split on the Qwen3.5 reader (2026-10-08): the frozen-threshold gaps by the output the reader saw last.

q2 found (Qwen3-32B, shift/str_replace_levels.py, dc3d6e8) that at text level 0 the calls right after a str_replace
output carry the whole frozen recall gap (+0.070 of +0.068) while the sim ranking there holds, so sim scores sit lower
after a str_replace (an offset), and that level 4 removes it. Here, the same split for Qwen3.5's matched attack-only
probes trained on real (pre_call, and its content-token control prev_content), at text levels 0, 4 and 7, on the calls
every level kept, with the cleaned-text test's layers and per-fold picks (`analyze_q2_cleaned_text_qwen35.py`).
Last output: the call whose output the reader saw last (the previous step's last call; q2 took the immediately previous
call, which for a second call in one turn is its sibling) grouped as str_replace, insert, other.
Per level, each group's share of the overall gap at the frozen threshold: real hits in the group over all real
impactful calls minus the same for sim (recall), and the same over read-only calls (FPR); shares add up to the overall
gap. Plus each group's recall, FPR and AUROC gaps. 1000 main-task draws, seed 0. Saved scores only, no compute.

  uv run python experiments/2026-10-06_qwen35_replication/analyze_q2_str_replace_split_qwen35.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_q2_cleaned_text_qwen35 as ct  # noqa: E402
from analyze_q2_level7_breakdown_qwen35 import context  # noqa: E402

POSITIONS, TRAIN, GROUPS = ["pre_call", "prev_content"], "real", ("str_replace", "insert", "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ct.ROOT / ".run_cache" / "q2_call_labels" / "2026-10-06_212047.jsonl"))
    a = ap.parse_args()
    vol = modal.Volume.from_name(ct.FEATS, version=2)
    ctx = context(Path(a.labels))
    ctx["last"] = ctx.out_kind.map({"edit": "str_replace", "insert": "insert"}).fillna("other")
    S = lambda x: {"value": round(float(x[0]), 4), "ci95": [round(float(v), 4) for v in np.nanpercentile(x[1:], [2.5, 97.5])]}
    out = {}
    for position in POSITIONS:
        tabs = {lv: ct.level_table(vol, lv, position) for lv in ct.LEVELS}
        common = set.intersection(*[set(map(tuple, t[ct.K].drop_duplicates().to_numpy())) for t in tabs.values()])
        for lv, t in tabs.items():
            t = t[(t.train_set == TRAIN) & pd.Series(list(map(tuple, t[ct.K].to_numpy())), index=t.index).isin(common)]
            t = t.sort_values(ct.K).reset_index(drop=True).merge(ctx, on=ct.K, how="left", validate="one_to_one")
            if t["last"].isna().any():
                raise ValueError(f"{int(t['last'].isna().sum())} calls missing from the label table")
            counts, inv = ct.task_counts(t.main_task.to_numpy())  # the same draws at every level: one set of tasks
            W = np.vstack([np.ones((1, counts.shape[1])), counts])[:, inv].astype(float)  # row 0: the point estimate
            y, s, flag = t.y.to_numpy(), t.score.to_numpy(np.float64), t.flag.to_numpy(float)
            dom = {d: (t.domain == d).to_numpy() for d in ("real", "sim")}
            g = {k: (t["last"] == k).to_numpy() for k in GROUPS}
            rate = lambda m, base: (W[:, m] * flag[m]).sum(1) / W[:, base].sum(1)
            res = {"calls": len(t)}
            for metric, lab in (("recall", 1), ("fpr", 0)):
                b = {d: dom[d] & (y == lab) for d in dom}
                res[f"overall {metric} gap"] = S(rate(b["real"], b["real"]) - rate(b["sim"], b["sim"]))
                res[f"{metric} gap share by last output"] = {k: S(rate(b["real"] & g[k], b["real"]) - rate(b["sim"] & g[k], b["sim"]))
                                                            for k in GROUPS}
            for k in GROUPS:
                m = {d: _m(y, s, flag, W, dom[d] & g[k]) for d in dom}
                res[f"last output: {k}"] = {"n": {f"{d} {lab}": int((dom[d] & g[k] & (y == v)).sum()) for d in dom
                                                  for lab, v in (("impactful", 1), ("read-only", 0))},
                                           **{f"{d} {x}": S(m[d][x]) for d in dom for x in ("recall", "fpr", "auroc")},
                                           **{f"{x} gap": S(m["real"][x] - m["sim"][x]) for x in ("recall", "fpr", "auroc")}}
            out[f"{position} L{lv}"] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q2_str_replace_split_qwen35_{stamp}.json"
    path.write_text(json.dumps({"train": TRAIN, "labels": a.labels, "volume": ct.FEATS, "features_run": ct.FR, "layers": ct.LAYERS,
                                "layer_tolerance": ct.TOL, "n_boot": ct.N_BOOT, "seed": ct.SEED, "last_output": "previous step's last call",
                                **out}, indent=1))
    f = lambda v: f"{v['value']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
    for name, r in out.items():
        print(f"== {name}: {r['calls']} calls")
        for metric in ("recall", "fpr"):
            print(f"   {metric:6} gap {f(r[f'overall {metric} gap'])}  shares: " +
                  "  ".join(f"{k} {f(v)}" for k, v in r[f"{metric} gap share by last output"].items()))
        for k in GROUPS:
            x = r[f"last output: {k}"]
            print(f"   after {k:11} n {x['n']}  recall {x['real recall']['value']:.3f}/{x['sim recall']['value']:.3f}  "
                  f"fpr {x['real fpr']['value']:.4f}/{x['sim fpr']['value']:.4f}  auroc gap {f(x['auroc gap'])}")
    print(f"-> {path}")


def _m(y, s, flag, W, m):
    return ct._weighted_metrics(y[m], s[m], flag[m], W[:, m])


if __name__ == "__main__":
    main()
