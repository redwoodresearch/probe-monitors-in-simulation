"""Runs recheck_fresh.py in one Modal CPU container (nothing computes on the laptop). The caller stages, from git at 37b5bbc9,
the verdict record (score.json), the seven stages' inputs, run lists and Qwen3-32B checks, the task list and #36's runs.json
(runs36.json) into FRESH_IN; their sha256 is checked inside against the verdict record. Volumes are read only.

  FRESH_IN=<staged dir> uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_fresh_verdict_recheck_09/run_recheck_modal.py --out <results json>
"""
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("FRESH_IN", "/nonexistent"))
app = modal.App("probe-shift_aswin-surya_sub-freshrecheck_cpu_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_file(HERE / "recheck_fresh.py", "/root/recheck_fresh.py")
if IN.exists():
    image = image.add_local_dir(IN, "/root/in")
RO = lambda n: modal.Volume.from_name(n, version=2).read_only()  # noqa: E731
VOLS = {"/acts": RO("probe-shift_aswin-surya_sub-la-activations"), "/seed": RO("probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"),
        "/q3w": RO("probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"), "/df": RO("probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13")}


@app.function(image=image, cpu=16, memory=65536, timeout=1800, retries=0, volumes=VOLS)
def run(git: str) -> dict:
    p = subprocess.run(["python", "/root/recheck_fresh.py", "/root/in", "/tmp/out.json"], capture_output=True, text=True, cwd="/root")
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:], "json": Path("/tmp/out.json").read_text() if p.returncode == 0 else None, "git": git}


@app.local_entrypoint()
def main(out: str = ""):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    r = run.remote(git)
    print(r["stdout"], r["stderr"][-3000:] if r["rc"] else "")
    if r["rc"] == 0 and out:
        Path(out).open("x").write(r["json"][:-2] + f',\n "git": "{git}"\n}}\n')
        Path(out[:-5] + ".printout.txt").open("x").write(r["stdout"])
