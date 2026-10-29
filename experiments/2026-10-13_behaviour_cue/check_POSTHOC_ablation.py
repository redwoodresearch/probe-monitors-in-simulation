"""Post hoc (RULE.md): A plus B without the error-field share, on the saved rows of the 17:06:35 run. $0, local.
`uv run python experiments/2026-10-13_behaviour_cue/check_POSTHOC_ablation.py`"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_behaviour_cue import HERE, test

RESULT = HERE / "results" / "behaviour_cue_2026-10-13_170635.json"


def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    res = json.loads(RESULT.read_text())
    rows_path = Path(res["rows"]["path"])
    assert hashlib.sha256(rows_path.read_bytes()).hexdigest() == res["rows"]["sha256"], "rows changed"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    names = res["features_a"] + [k for k in res["features_b"] if k != "error_field_share"]
    X = np.array([[r[k] for k in names] for r in rows], dtype=float)
    y = np.array([r["world"] == "fake" for r in rows], dtype=int)
    groups = np.array([r["task"] for r in rows])
    out = {"git": git, "rows_sha256": res["rows"]["sha256"], "features": names,
           "a_plus_b_minus_error_field": test(X, y, groups),
           "for_reference": {k: res[k]["auroc"] for k in ("a_only", "a_plus_b")}}
    ts = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"ablation_POSTHOC_{ts}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path, file=sys.stderr)


if __name__ == "__main__":
    main()
