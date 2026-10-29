"""Runs sce_chart_boot.py in one Modal CPU container (nothing computes on the laptop). The caller stages
sce_scores_2026-10-13_121538.npz and sce_metrics_POSTHOC_2026-10-13_123007.json (sha256 checked inside) into SCE_IN.

  SCE_IN=<staged dir> uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-14_sce_chart_intervals_09/run_sce_chart_modal.py
"""
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("SCE_IN", "/nonexistent"))
app = modal.App("probe-shift_aswin-surya_sub-convaudit_scechart_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_file(HERE / "sce_chart_boot.py", "/root/sce_chart_boot.py")
if IN.exists():
    image = image.add_local_file(IN / "sce_scores_2026-10-13_121538.npz", "/root/in/scores.npz").add_local_file(
        IN / "sce_metrics_POSTHOC_2026-10-13_123007.json", "/root/in/metrics.json")


@app.function(image=image, cpu=8, memory=16384, timeout=1800, retries=0)
def run() -> dict:
    p = subprocess.run(["python", "/root/sce_chart_boot.py", "/root/in/scores.npz", "/root/in/metrics.json", "/tmp/out.json"], capture_output=True, text=True)
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:], "json": Path("/tmp/out.json").read_text() if Path("/tmp/out.json").exists() else None}


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    r = run.remote()
    print("git", git, "| rc", r["rc"])
    print(r["stdout"], r["stderr"][-2000:] if r["rc"] else "")
    if r["json"]:
        import json

        d = json.loads(r["json"]) | {"git": git, "stamp": stamp}
        (HERE / "results" / f"sce_chart_{stamp}.json").open("x").write(json.dumps(d, indent=1))
        (HERE / "results" / f"sce_chart_{stamp}.printout.txt").open("x").write(r["stdout"])
