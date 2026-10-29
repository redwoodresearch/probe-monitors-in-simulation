"""After a run with a restart alert (launch_simh.py, 0d's refined duplicate-writer rule): confirm the results come from exactly one
completed call per input. Takes a glob and checks the newest matching record. Exit 1 if not. Numbers only. $0, local.
- a fit record (run_sce_seed_grid.py / run_sce_seed_variants.py: "pack", "fit", "settings"): one pack per layer, and each
  cell's fit exactly once per layer;
- an extraction summary (run_extract_g3_simh.py: "shards"): one completed entry per shard index 0..n-1, none crashed.

  uv run --no-project --python 3.12 python experiments/2026-10-13_q3_sce_reply_probe/check_one_result_per_input.py '<glob>'
"""
import glob
import json
import os
import sys
from collections import Counter


def check(rec: dict) -> list:
    bad = []
    if "fit" in rec and "pack" in rec:
        layers = list(rec["settings"]["layers"])
        if sorted(p["layer"] for p in rec["pack"]) != sorted(layers):
            bad.append("pack: not exactly one per layer")
        cells = set(rec["settings"]["cells"])
        for l in layers:
            c = Counter(cell for f in rec["fit"] if f["layer"] == l for cell in f["cells"])
            if set(c) != cells or any(v != 1 for v in c.values()):
                bad.append(f"fit L{l}: cells fitted {dict(c)}, want each of {sorted(cells)} once")
    elif "shards" in rec:
        idx = Counter(x.get("shard") for x in rec["shards"])
        n = len(rec["shards"])
        if set(idx) != set(range(n)) or any(v != 1 for v in idx.values()) or rec.get("n_crashed") or any("chunks" not in x for x in rec["shards"]):
            bad.append(f"extraction: shards {dict(idx)}, crashed {rec.get('n_crashed')}")
    else:
        bad.append("not a fit record or an extraction summary")
    return bad


if __name__ == "__main__":
    files = sorted(glob.glob(sys.argv[1]), key=os.path.getmtime)
    if not files:
        raise SystemExit("no record matches")
    bad = check(json.loads(open(files[-1]).read()))
    print(json.dumps({"record": os.path.basename(files[-1]), "one completed call per input": not bad, "problems": bad}))
    sys.exit(1 if bad else 0)
