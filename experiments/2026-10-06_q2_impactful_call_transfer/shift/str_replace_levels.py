"""The frozen gap by the output the reader saw last (str_replace, insert, other), at text levels 0 and 4-7, for pre_call
(the `<tool_call>` template token) and its content-token control prev_content (Qwen3-32B, layers 9/16/17, probes
trained on real; the same calls at every level, as in compare_levels.py).
Per level: each last-output group's share of the overall recall gap at the frozen threshold (real hits / all real
impactful minus sim hits / all sim impactful; the shares sum to the overall gap), and the group's AUROC, recall and
FPR gaps with task-resampled intervals. Saved scores only; descriptive.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from boot import fmt, report  # noqa: E402
from compare_levels import K, LEVELS, level_table  # noqa: E402
from paths import DATA, RESULTS, stamp as now  # noqa: E402

ctx = pd.read_parquet(DATA / "context_features.parquet")[K + ["prev_cmd"]]
tabs = {(pos, lv): level_table(lv, pos).query("train_set == 'real'") for pos in ("pre_call", "prev_content") for lv in LEVELS}
common = set.intersection(*[set(map(tuple, t[K].to_numpy())) for t in tabs.values()])
out = {}
for (pos, lv), t in tabs.items():
    t = t[pd.Series(list(map(tuple, t[K].to_numpy())), index=t.index).isin(common)]
    df = t.merge(ctx, on=K, how="left", validate="one_to_one").reset_index(drop=True)
    df["last"] = df.prev_cmd.where(df.prev_cmd.isin(["str_replace", "insert"]), "other")
    imp = df[df.y == 1]
    n = {d: int((imp.domain == d).sum()) for d in ("real", "sim")}
    share = {g: round(float(s[s.domain == "real"].flag.sum() / n["real"] - s[s.domain == "sim"].flag.sum() / n["sim"]), 4)
             for g, s in imp.groupby("last")}
    overall = round(float(imp[imp.domain == "real"].flag.mean() - imp[imp.domain == "sim"].flag.mean()), 4)
    real, sim = (df.domain == "real").to_numpy(), (df.domain == "sim").to_numpy()
    rep = report(df, {f"last output: {g}": (real & (df["last"] == g).to_numpy(), sim & (df["last"] == g).to_numpy())
                      for g in ("str_replace", "insert", "other")})
    out[f"{pos} L{lv}"] = {"calls": len(df), "overall_recall_gap": overall, "recall_gap_share_by_last_output": share, "by_last_output": rep}
    print(f"\n### {pos} level {lv}: {len(df)} calls, frozen recall gap {overall:+.3f}, by last output {share}\n" + fmt(rep))
(RESULTS / f"shift_str_replace_levels_{now()}.json").write_text(json.dumps(out, indent=1))
