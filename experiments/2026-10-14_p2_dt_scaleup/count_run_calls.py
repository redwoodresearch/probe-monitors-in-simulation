"""Per-run call counts for DT folders written before Amendment 9 added `calls` to manifest_dt.json (4f, 10-15 about 00:15: zero-call
runs in nm360 s3 main). Reads each folder's manifest_dt.json runs and ONE npz's `key` array (rows are world|task); no scores.
  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-14_p2_dt_scaleup/count_run_calls.py --folders <folder>[,...]
"""

import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_dtq4b_kill-2026-11-13"
WVOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.13").uv_pip_install("numpy==2.3.3"), cpu=(2, 2), memory=(4096, 4096), timeout=600,
              retries=0, scaledown_window=10, startup_timeout=120, volumes={"/work": modal.Volume.from_name(WVOL, version=2).read_only()})
def count(folder: str) -> dict:
    import numpy as np

    d = Path("/work/scaleup") / folder
    m = json.loads((d / "manifest_dt.json").read_text())
    with np.load(d / "dt" / "DT_destructive_primary.npz") as z:
        keys = Counter(str(k) for k in z["key"])
    runs = sorted(m["runs"])
    if set(keys) - set(runs):
        raise ValueError(f"{folder}: npz keys outside the manifest's runs")
    return {"folder": folder, "runs": len(runs), "rows": sum(keys.values()), "manifest_rows": m["rows"],
            "zero_call_runs": [r for r in runs if keys.get(r, 0) == 0], "calls_by_run": {r: keys.get(r, 0) for r in runs}}


@app.local_entrypoint()
def main(folders: str):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    me = "experiments/2026-10-14_p2_dt_scaleup/count_run_calls.py"
    if git("status", "--porcelain", "--", me) or not git("branch", "-r", "--contains", git("rev-parse", "HEAD")):
        raise SystemExit("commit and push this script first")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "keys and counts only", "git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "folders": list(count.map(folders.split(",")))}
    for r in rec["folders"]:
        if r["rows"] != r["manifest_rows"]:
            raise SystemExit(f"{r['folder']}: npz rows {r['rows']} differ from manifest_dt rows {r['manifest_rows']}")
        print(r["folder"], "runs", r["runs"], "rows", r["rows"], "zero-call runs", len(r["zero_call_runs"]), r["zero_call_runs"])
    dest = HERE / "results" / f"run_calls_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    print(dest.name)
