"""How often the probe's output_end readout token ends in a newline, per arm, on level-0 text (Figure 1's caption).

  uv run python experiments/2026-10-06_james_figures/analyze_readout_newlines.py

Reads the transcript viewer's cache (.run_cache/viewer, from experiments/2026-10-06_transcript_viewer/
run_build_cache.py): every output_end position from probe_shift.reader.segment_end_positions, on the paired tasks'
valid runs, inside the 32k window. Writes results/readout_newlines_<stamp>.json, which plot_james_figures.py reads.
"""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.viewer import data

HERE = Path(__file__).parent
CACHE = Path(".run_cache/viewer")
SPLITS = Path("splits/exclusions_v3.json")


def main():
    version = json.loads((CACHE / "version.json").read_text())["version"]
    assert version == data.CACHE_VERSION, f"cache version {version}, code expects {data.CACHE_VERSION}: rebuild the cache"
    paired = set(json.loads(SPLITS.read_text())["paired_attack_cells"]["cells"])
    counts = {}
    for r in map(json.loads, (CACHE / "index.jsonl").read_text().splitlines()):
        if r["arm"] == "real-honest" or r["cell"] not in paired or r["invalid"]:
            continue
        c = counts.setdefault(r["arm"], Counter())
        for ro in json.loads(data.sample_path(CACHE, r["arm"], r["key"]).read_text())["readouts"]:
            if ro["kind"] == "output_end" and ro["kept"]:
                c["output_end_tokens"] += 1
                c["ending_in_newline"] += ro["token_str"].endswith("\n")
                c["newline_only"] += ro["token_str"].strip("\n") == ""
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {
        "what": "output_end readout tokens whose decoded text ends in a newline, or is only newlines",
        "text_level": 0, "cells": f"{SPLITS} paired_attack_cells (valid runs only)", "window": "positions inside 32k",
        "cache": {"path": str(CACHE), "version": version,
                  "index_sha256": hashlib.sha256((CACHE / "index.jsonl").read_bytes()).hexdigest()},
        "tokenizer": data.TOKENIZER,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),  # the exact code that made this file
        "counts": {arm: dict(c) for arm, c in sorted(counts.items())},
    }
    path = HERE / "results" / f"readout_newlines_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path)
    for arm, c in out["counts"].items():
        print(f"{arm}: {c['ending_in_newline'] / c['output_end_tokens']:.1%} of {c['output_end_tokens']:,}")


if __name__ == "__main__":
    main()
