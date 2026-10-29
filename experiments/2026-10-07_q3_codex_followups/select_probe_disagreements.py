"""K7, step 1: LinuxArena runs where the two eval-awareness probes disagree most, for a blinded read.

Templated probe (Q3 off-policy, call_end, layer 40, the grouped folds' choice) and Needham probe (asst_mean, layer 27,
the pre-chosen primary). Run score = mean over the run's calls; each is turned into a percentile rank over all runs of
the four LinuxArena arms. Picks the 6 runs with the largest (templated - Needham) rank gap and the 6 with the largest
(Needham - templated) gap, seed-free, and writes them shuffled (seed 0) without arm or direction (the key goes to a
separate file).

  python select_probe_disagreements.py --templated <scores_eval_awareness_segment_end_*.parquet> --needham <scores_needham_alllayers_*.parquet>
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templated", required=True)
    ap.add_argument("--needham", required=True)
    args = ap.parse_args()
    t = pd.read_parquet(args.templated)
    t = t[t.kind == "call_end"].groupby(["arm", "key"]).L40.mean().rename("templated")
    n = pd.read_parquet(args.needham, columns=["arm", "key", "cell", "main_task", "asst_mean_L27"])
    meta = n.drop_duplicates("key").set_index(["arm", "key"])[["cell", "main_task"]]
    n = n.groupby(["arm", "key"]).asst_mean_L27.mean().rename("needham")
    r = pd.concat([t, n], axis=1, join="inner").join(meta)
    r["rank_t"], r["rank_n"] = r.templated.rank(pct=True), r.needham.rank(pct=True)
    r["gap"] = r.rank_t - r.rank_n
    print("runs", len(r), "| spearman", round(r.templated.corr(r.needham, method="spearman"), 3),
          "| by arm", r.groupby(level=0).apply(lambda g: round(g.templated.corr(g.needham, method="spearman"), 3)).to_dict())
    hi = r.sort_values("gap").tail(6).assign(direction="templated_higher")
    lo = r.sort_values("gap").head(6).assign(direction="needham_higher")
    pick = pd.concat([hi, lo]).reset_index()
    order = list(range(len(pick)))
    random.Random(0).shuffle(order)
    items = [{"item": i, "key": pick.loc[j, "key"], "arm_cohort_hidden": True} for i, j in enumerate(order)]
    key = [{"item": i, "arm": pick.loc[j, "arm"], "key": pick.loc[j, "key"], "direction": pick.loc[j, "direction"],
            "rank_templated": round(float(pick.loc[j, "rank_t"]), 3), "rank_needham": round(float(pick.loc[j, "rank_n"]), 3)}
           for i, j in enumerate(order)]
    for name, obj in (("probe_disagreement_items_2026-10-07.json", items), ("probe_disagreement_key_2026-10-07.json", key)):
        with (HERE / "results" / name).open("x") as f:  # never overwrite a blinded selection an audit may have used
            f.write(json.dumps(obj, indent=1))
    print(pick[["arm", "direction", "rank_t", "rank_n"]].groupby(["direction", "arm"]).size())


if __name__ == "__main__":
    main()
