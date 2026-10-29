"""Runs fa_check.py in one Modal CPU container. Input: recheck.json (staged; sha checked inside).  FA_IN=<dir with recheck.json> uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-14_priority1_spec_09/run_launch_check_modal.py"""
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("FA_IN", "/nonexistent"))
app = modal.App("probe-shift_aswin-surya_sub-convaudit_facheck_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2").add_local_file(HERE / "fa_check.py", "/root/fa_check.py")
if IN.exists():
    image = image.add_local_file(IN / "recheck.json", "/root/in/recheck.json")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2).read_only()


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0, volumes={"/w": vol})
def run() -> dict:
    p = subprocess.run(["python", "/root/fa_check.py", "/root/in/recheck.json", "/tmp/out.json"],
                       capture_output=True, text=True)
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-3000:], "json": Path("/tmp/out.json").read_text() if Path("/tmp/out.json").exists() else None}


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    r = run.remote()
    print("git", git, "| rc", r["rc"])
    print(r["stdout"], r["stderr"][-2000:] if r["rc"] else "")
    if r["rc"] != 0:
        raise SystemExit("fa_check.py failed; nothing saved")
    if r["json"]:
        import json

        (HERE / "results" / f"fa_check_{stamp}.json").open("x").write(json.dumps(json.loads(r["json"]) | {"git": git, "stamp": stamp}, indent=1))
        (HERE / "results" / f"fa_check_{stamp}.printout.txt").open("x").write(r["stdout"])
