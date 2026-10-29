"""Merge stagings into one (e.g. the attack and honest stagings, for a model trained on both cell types together):
their output, call and run rows concatenated, unchanged. Fails if two parts share a run (arm, key) or a cell.
Writes .run_cache/examples_<stamp>/.

  uv run python experiments/2026-10-06_q1_text_baselines/run_stage_merge.py config/stage_merge_pooled.yaml
"""

import gzip
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

HERE = Path(__file__).parent


def main():
    cfg = yaml.safe_load((HERE / sys.argv[1]).read_text())
    parts = [HERE / ".run_cache" / p for p in cfg["parts"]]
    runs = [[json.loads(l) for l in (p / "runs.jsonl").read_text().splitlines()] for p in parts]
    keys = [{(r["arm"], r["key"]) for r in rs} for rs in runs]
    cells = [{r["cell"] for r in rs} for rs in runs]
    for i in range(len(parts)):
        for j in range(i):
            assert not keys[i] & keys[j], f"{cfg['parts'][i]} and {cfg['parts'][j]} share runs"
            assert not cells[i] & cells[j], f"{cfg['parts'][i]} and {cfg['parts'][j]} share cells"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / ".run_cache" / f"examples_{stamp}"
    out.mkdir(parents=True)
    for name in ("outputs.jsonl.gz", "calls.jsonl.gz"):
        with gzip.open(out / name, "wt") as fh:
            for p in parts:
                with gzip.open(p / name, "rt") as part:
                    shutil.copyfileobj(part, fh)
    (out / "runs.jsonl").write_text("".join((p / "runs.jsonl").read_text() for p in parts))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    (out / "stage.json").write_text(json.dumps({"config": cfg, "git": git, "runs_per_part": [len(r) for r in runs],
                                                "cells_per_part": [len(c) for c in cells]}, indent=1))
    print(f"{sum(len(r) for r in runs)} runs, {sum(len(c) for c in cells)} cells -> {out}")


if __name__ == "__main__":
    main()
