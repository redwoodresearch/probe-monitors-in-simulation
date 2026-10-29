"""Build the viewer's local cache: download the four LinuxArena cohorts and write what the reader sees per sample.

  uv run --group viewer python experiments/2026-10-06_transcript_viewer/run_build_cache.py

Writes the cache named in config/viewer.yaml and a summary to results/build_<timestamp>.json.
"""

import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from probe_shift.viewer import data

HERE = Path(__file__).parent


def main():
    cfg = yaml.safe_load((HERE / "config" / "viewer.yaml").read_text())
    assert cfg["tokenizer"] == data.TOKENIZER["name"], "the cache build uses data.TOKENIZER (probe_shift.models)"
    t0 = time.time()
    rows = data.build(Path(cfg["cache"]), cfg["max_tokens"], cfg["workers"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    summary = {
        "config": cfg, "tokenizer": data.TOKENIZER, "arms": data.ARMS, "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
        "built_at": stamp, "wall_seconds": round(time.time() - t0),
        "samples": dict(Counter(r["arm"] for r in rows)),
        "invalid": {arm: dict(Counter(r["invalid"] for r in rows if r["arm"] == arm and r["invalid"])) for arm in data.ARMS},
        "truncated": dict(Counter(r["arm"] for r in rows if r["truncated"])),
    }
    (HERE / "results" / f"build_{stamp}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":  # the build spawns worker processes, which re-import this file
    main()
