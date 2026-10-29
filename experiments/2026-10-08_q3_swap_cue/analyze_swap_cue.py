"""The swap's catch rates next to the real attack runs of the same 100 cells (the gap V2 would have to close).
Committed out-of-fold run scores of the #36 detector; same fold thresholds as run_score_swap.py. No model calls.

  python analyze_swap_cue.py --swap results/swap_cue_<stamp>.json --scores-dir <#36 results>/run_scores_2026-10-06_181548
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
READOUTS = {"output_end": 63, "call_end": 48, "call_end_content": 63}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", required=True)
    ap.add_argument("--scores-dir", required=True)
    args = ap.parse_args()
    rec = json.loads(Path(args.swap).read_text())
    d = Path(args.scores_dir)
    runs = json.loads((d / "runs.json").read_text())
    idx = {(r["arm"], r["key"]): i for i, r in enumerate(runs)}
    by_cell = {}
    for i, r in enumerate(runs):
        by_cell.setdefault((r["arm"], r["cell"]), []).append(i)
    keys = list(rec["insert_categories"])
    out = {"swap": Path(args.swap).name, "cells": len(keys), "catch95_same_cells": {}}
    for ro, layer in READOUTS.items():
        s = np.load(d / f"scores_L{layer}.npz")[f"{ro}__mean__task_outer"]
        row = {}
        for arm in ("real-attack-e1", "real-attack-e2", "sim-attack"):
            js = [j for k in keys for j in by_cell.get((arm, runs[idx[("sim-attack", k)]]["cell"]), []) if not np.isnan(s[j])]
            row[arm] = {"catch": float(np.mean([s[j] > rec["thresholds"][ro][str(runs[j]["fold"])]["95"] for j in js])),
                        "catch99": float(np.mean([s[j] > rec["thresholds"][ro][str(runs[j]["fold"])]["99"] for j in js])),
                        "runs": len(js)}
        v = rec["by_run"][ro]["all"]["catch95"]
        gap = row["real-attack-e1"]["catch"] - v["V1"]
        row |= {"V1": v["V1"], "V2": v["V2"], "gap_e1_minus_V1": gap,
                "share_of_gap_closed_by_V2": (v["V2"] - v["V1"]) / gap if abs(gap) > 0.05 else None}
        out["catch95_same_cells"][ro] = row
        print(ro, json.dumps(row))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path, text = HERE / "results" / f"swap_cue_real_catch_{stamp}.json", json.dumps(out, indent=1)
    with path.open("x") as f:
        f.write(text)
    print("->", path)


if __name__ == "__main__":
    main()
