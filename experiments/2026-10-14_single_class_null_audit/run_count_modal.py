"""Runs count_single_class.py in one Modal CPU container (RULE.md; nothing computes on the laptop). The caller stages the
git-pinned inputs listed in jobs.json (`git show <ref>:<path>`, sha256 recorded in inputs.json) into NULLAUDIT_IN;
they ship in the image. Volume inputs are read where they live (read-only).

  NULLAUDIT_IN=<staged dir> PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_single_class_null_audit/run_count_modal.py --out <results json>
"""
import json
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("NULLAUDIT_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-nullaudit_count_kill-2026-11-14"
VOLS = {"/t1": "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"}  # mount point -> volume name, read-only (the T1 sweep records)
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3") \
    .add_local_file(HERE / "count_single_class.py", "/root/count_single_class.py").add_local_file(HERE / "jobs.json", "/root/jobs.json")
if IN.exists():
    image = image.add_local_dir(IN, "/root/in")


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0,
              volumes={m: modal.Volume.from_name(v, version=2).read_only() for m, v in VOLS.items()})
def run(git: str) -> dict:
    p = subprocess.run(["python", "/root/count_single_class.py", "/root/jobs.json", "/root/in", "/tmp/out.json", git],
                       capture_output=True, text=True, cwd="/root")
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:], "json": Path("/tmp/out.json").read_text() if p.returncode == 0 else None}


@app.local_entrypoint()
def main(out: str = ""):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    r = run.remote(git)
    print(r["stdout"], r["stderr"][-2000:] if r["rc"] else "")
    if r["rc"] == 0 and out:
        Path(out).open("x").write(r["json"])
        Path(out[:-5] + ".printout.txt").open("x").write(r["stdout"])
