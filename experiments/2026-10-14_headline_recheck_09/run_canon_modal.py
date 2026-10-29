"""Runs canon_boot.py in one Modal CPU container (Amendment 5; nothing computes on the laptop). The caller stages
labels.json, runs36.json, runs_sh.json, t196.json (sha256 checked inside), recheck.json and uncut.json (the saved per-run scores,
git show e6081f34) into CANON_IN.

  CANON_IN=<staged dir> uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_headline_recheck_09/run_boot_modal.py --out <results json>
"""
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("CANON_IN", "/nonexistent"))
app = modal.App("probe-shift_aswin-surya_sub-canonstrata_cpu_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_file(HERE / "canon_boot.py", "/root/canon_boot.py")
if IN.exists():
    image = image.add_local_dir(IN, "/root/in")


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0)
def run(git: str) -> dict:
    p = subprocess.run(["python", "/root/canon_boot.py", "/root/in", "/tmp/out.json", git], capture_output=True, text=True, cwd="/root")
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:], "json": Path("/tmp/out.json").read_text() if p.returncode == 0 else None}


@app.local_entrypoint()
def main(out: str = ""):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()  # read at launch
    r = run.remote(git)
    print(r["stdout"], r["stderr"][-2000:] if r["rc"] else "")
    if r["rc"] == 0 and out:
        Path(out).open("x").write(r["json"])
        Path(out[:-5] + ".printout.txt").open("x").write(r["stdout"])
