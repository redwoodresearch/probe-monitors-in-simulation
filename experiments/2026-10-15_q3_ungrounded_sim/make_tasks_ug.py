"""The ungrounded arm's task file (RULE.md): the interleaved launch's seeded 150-cell PILOT slice (09's estimation set,
stratified environment x stratum), minus the cells with no real honest reference anywhere, in one seeded order.

Every cell entry is copied byte-for-byte (as parsed JSON) from the committed pilot task files at --source-commit; the
script asserts the count, the exclusions and that no key repeats. Order: random.Random(SEED).shuffle over the merged
list (spreads environments, so the launcher's in-order start rule doesn't stall on one environment's stagger).

  python make_tasks_ug.py --source-commit <sha> --out config/tasks_ug_pilot.json   (run in a repo checkout)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess

SRC_DIR = "experiments/2026-10-15_q3_interleaved_run3_sim2/config"
SLICES = ["q1", "q2", "q3", "q4", "old"]
NO_REFERENCE = {"t98_model_registry", "t139_model_registry", "t151_model_registry"}  # design fork: no reference in any source launch
SEED = "q3-ungrounded-sim-2026-10-15"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cells, sources = [], {}
    for s in SLICES:
        path = f"{SRC_DIR}/tasks_il_{s}_pilot.json"
        raw = subprocess.run(["git", "show", f"{args.source_commit}:{path}"], check=True, capture_output=True).stdout
        sources[path] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        assert data["stage"] == "pilot", (path, data["stage"])
        for c in data["cells"]:
            cells.append(c)
    keys = [c["key"] for c in cells]
    assert len(keys) == len(set(keys)) == 150, (len(keys), len(set(keys)))
    dropped = sorted(k for k in keys if k in NO_REFERENCE)
    cells = [c for c in cells if c["key"] not in NO_REFERENCE]
    random.Random(SEED).shuffle(cells)
    out = {"rule": "the interleaved launch's 150-cell pilot slice (09 estimation set, stratified environment x stratum) "
                   "minus cells with no real honest reference; one seeded order",
           "seed": SEED, "launch": "ug", "stage": "pilot", "source_commit": args.source_commit,
           "sources_sha256": sources, "dropped_no_reference": dropped, "cells": cells}
    with open(args.out, "x") as f:
        json.dump(out, f, indent=1)
    print(f"{len(cells)} cells ({150 - len(cells)} dropped: {dropped}) -> {args.out}")


if __name__ == "__main__":
    main()
