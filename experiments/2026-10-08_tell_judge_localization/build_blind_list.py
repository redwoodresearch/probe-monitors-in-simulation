"""A blind list of every flag's reason for labelling: shuffled (seed 0), no side, run or action, numbered F001.. .
The key (number -> side, run, action) goes to .run_cache, not to the list.

  uv run --with pandas python experiments/2026-10-08_tell_judge_localization/build_blind_list.py --flags .run_cache/flags_swap_<stamp>.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("--flags", required=True)
args = ap.parse_args()
f = pd.read_json(HERE / args.flags, lines=True).sample(frac=1, random_state=0).reset_index(drop=True)
f["fid"] = [f"F{i + 1:03d}" for i in range(len(f))]
f[["fid", "side", "key", "action_n"]].to_json(HERE / ".run_cache/blind_key.jsonl", orient="records", lines=True)
lines = [f"{r.fid}: {json.dumps(r.reason)}" for r in f.itertuples()]
(HERE / ".run_cache/blind_list.txt").write_text("\n".join(lines) + "\n")
print(len(lines), "flags")
