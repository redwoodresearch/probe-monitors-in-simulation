"""Does the patched Q2 runner, without a patch, rebuild the committed Q2 inputs byte for byte (2026-10-07)?

`run_q2_alllayers_qwen35.py --inputs-only` with the `*_recheck` overlays (the committed overlays under a new
features_run) writes the call set and row index of each base and view. This compares every such file with the
committed features run's on this track's Q2 features Volume, by sha256. Identical files mean identical packs (pack
gathers exactly those rows), so the fits and summaries see the same inputs as the committed results. Reads only.

  uv run python experiments/2026-10-06_qwen35_replication/analyze_q2_recheck_inputs.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
FEATS = "probe-shift_aswin-surya_sub-qwen35_q2feats_kill-2026-11-06"
SUFFIX = "_eb014307fd1aab31_as_labeled"
FILES = ["manifest.json", "base_report.json", "alllayers/index.npz", "alllayers/chunks.json",
         "threeway/index.npz", "threeway/chunks.json", "threeway/manifest.json"]


def main():
    vol = modal.Volume.from_name(FEATS, version=2)
    sha = lambda path: hashlib.sha256(b"".join(vol.read_file(path))).hexdigest()
    rows = []
    for base in ("matched", "whole"):
        old, new = f"2026-10-06_q2_qwen35_{base}{SUFFIX}", f"2026-10-07_q2_qwen35_{base}_recheck{SUFFIX}"
        for f in FILES:
            a, b = sha(f"{old}/{f}"), sha(f"{new}/{f}")
            rows.append({"base": base, "file": f, "committed": a, "recheck": b, "identical": a == b})
            print(f"{base:7} {f:24} {'identical' if a == b else 'DIFFERENT'}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q2_recheck_inputs_{stamp}.json"
    path.write_text(json.dumps({"volume": FEATS, "files": rows}, indent=1))
    print(f"{sum(r['identical'] for r in rows)}/{len(rows)} identical -> {path}")


if __name__ == "__main__":
    main()
