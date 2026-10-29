"""Scored outputs per run, by comparison, scope and arm, from step A's stored per-output scores (level 1; S3 via
experiments/2026-10-09_fixed_sim/results_manifest_v1.json, fetched and sha256-checked by that folder's fetch_results.py).
Checks the review's "about 21 outputs". Counts only; $0.

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_maxwin_rescore/analyze_outputs_per_run.py
"""

import gzip
import hashlib
import json
import statistics as st
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
DATA = HERE.parents[0] / "2026-10-09_fixed_sim" / "data"
FILES = {"fixed_vs_e1": "claimB_output_scores_A_2026-10-10_022646.json.gz", "e2_vs_e1": "claimB_output_scores_A_e2_vs_e1_2026-10-10_024204.json.gz",
         "plant1": "claimB_output_scores_A_plant1_2026-10-10_050334.json.gz", "plant2": "claimB_output_scores_A_plant2_2026-10-10_050400.json.gz"}
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}


def main():
    out = {"label": "scored outputs per run (level 1, step A's stored per-output scores; counts only)", "inputs": {}, "rows": {}}
    for comp, f in FILES.items():
        raw = (DATA / f).read_bytes()
        out["inputs"][f] = hashlib.sha256(raw).hexdigest()
        rows = json.loads(gzip.decompress(raw))["levels"]["level1"]["rows"]
        n = Counter((r["arm"], r["key"]) for r in rows)
        env = {(r["arm"], r["key"]): r["environment"] for r in rows}
        for scope in ("test", "all"):
            keep = {k: v for k, v in n.items() if scope == "all" or env[k] in TEST}
            for arm in sorted({a for a, _ in keep}) + ["pooled"]:
                c = [v for (a, _), v in keep.items() if arm in ("pooled", a)]
                out["rows"][f"{comp}|{scope}|{arm}"] = {"runs": len(c), "median": st.median(c), "mean": round(st.mean(c), 3)}
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"outputs_per_run_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for k, v in out["rows"].items():
        print(k, v)
    print("->", dst)


if __name__ == "__main__":
    main()
