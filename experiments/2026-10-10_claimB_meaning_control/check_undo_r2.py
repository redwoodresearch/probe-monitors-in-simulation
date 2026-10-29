"""Validates undo_r2 locally (no model, $0) on the fixed-sim build's own inputs (level 0, before staging):
1. simulator side: a round trip only (these inputs already have R2 applied); the byte check against the simulator's
   own pre-R2 outputs is run_meaning_plant_modal.py --mode validate on planted check 2's pairs (673 of 814 exact);
2. real side: for every real e1 str_replace output, how many undo_r2 accepts (the round-trip gate passes).
  uv run --no-project --python 3.12 python experiments/2026-10-10_claimB_meaning_control/check_undo_r2.py <fixed-sim run dir>
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-09_fixed_sim"))
import fixes  # noqa: E402
from undo_r2 import undo_r2  # noqa: E402

c = collections.Counter()
for f in sorted(Path(sys.argv[1], "inputs").glob("shard_*.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["arm"] == "sim-attack":
            fixed, recs = fixes.r2_str_replace(r["messages"])
            for x in recs:
                if not x["changed"]:
                    continue
                orig, new = r["messages"][x["msg"]]["content"], fixed[x["msg"]]["content"]
                u = undo_r2(new)
                c[f"sim {x['category']}: " + ("exact" if u == orig else "refused" if u is None else "differs")] += 1
        elif r["arm"] == "real-attack-e1":
            for x in fixes.calls_of(r["messages"]):
                if x["tool"] == "text_editor" and x["args"].get("command") == "str_replace":
                    c["real e1 str_replace: " + ("accepted" if undo_r2(r["messages"][x["msg"]]["content"]) else "refused")] += 1
for k, v in sorted(c.items()):
    print(f"{k}: {v}")
