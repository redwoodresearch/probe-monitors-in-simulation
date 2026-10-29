"""Share of editor-diff tool outputs whose diff lines carry the `N|` line-number gutter, per arm.

A side finding of the copied-vs-invented split: real `text_editor` edit outputs ("The file ... has been edited.
Here's a diff ...") number every diff line (`  53|  ...`); the simulator's mostly don't. Reads baselines-q3's
staged per-output table (kept outputs on the 939 paired cells) and writes results/editor_gutter_<stamp>.json.

  uv run python experiments/2026-10-06_q1_copied_vs_invented/analyze_editor_gutter.py
"""

import gzip
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

HERE = Path(__file__).parent
EDIT_MARK = "has been edited. Here's a diff"
GUTTER = re.compile(r"\n\s*\d+\|")


def main():
    cfg = yaml.safe_load((HERE / "config" / "copied_vs_invented.yaml").read_text())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    n, g = Counter(), Counter()
    with gzip.open(Path(cfg["staged_outputs"]).expanduser(), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] in cfg["arms"] and r["kept"] and r["tool"] == "text_editor" and EDIT_MARK in r["text"]:
                n[r["arm"]] += 1
                g[r["arm"]] += bool(GUTTER.search(r["text"]))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"stamp": stamp, "git": git, "staged_outputs": cfg["staged_outputs"], "edit_marker": EDIT_MARK,
           "gutter_regex": GUTTER.pattern,
           "per_arm": {a: {"edit_outputs": n[a], "with_gutter": g[a], "share": round(g[a] / n[a], 4) if n[a] else None}
                       for a in cfg["arms"]}}
    (HERE / "results" / f"editor_gutter_{stamp}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["per_arm"], indent=1))


if __name__ == "__main__":
    main()
