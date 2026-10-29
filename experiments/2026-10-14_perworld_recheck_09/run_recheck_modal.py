"""Runs recheck_perworld.py in one Modal CPU container (nothing computes on the laptop). The trap scores file ships from
09's local copy (sha256 checked inside); the title-cell scores are read from df's Volume, read only.

  TRAP_NPZ=<local sce_scores_2026-10-13_121538.npz> uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_perworld_recheck_09/run_recheck_modal.py --out <results json>
"""
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
TRAP = Path(os.environ.get("TRAP_NPZ", "/nonexistent"))
app = modal.App("probe-shift_aswin-surya_sub-perworldrecheck_cpu_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_file(HERE / "recheck_perworld.py", "/root/recheck_perworld.py")
if TRAP.exists():
    image = image.add_local_file(TRAP, "/root/in/trap.npz")
DFV = modal.Volume.from_name("probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13", version=2).read_only()


@app.function(image=image, cpu=4, memory=16384, timeout=1200, retries=0, volumes={"/df": DFV})
def run(git: str) -> dict:
    p = subprocess.run(["python", "/root/recheck_perworld.py", "/root/in/trap.npz",
                        "/df/results/2026-10-13_185308/why_avg_scores_2026-10-13_185308.npz", "/tmp/out.json"],
                       capture_output=True, text=True, cwd="/root")
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:], "json": Path("/tmp/out.json").read_text() if p.returncode == 0 else None, "git": git}


@app.local_entrypoint()
def main(out: str = ""):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    r = run.remote(git)
    print(r["stdout"], r["stderr"][-2000:] if r["rc"] else "")
    if r["rc"] == 0 and out:
        Path(out).open("x").write(r["json"][:-2] + f',\n "git": "{git}"\n}}\n')
        Path(out[:-5] + ".printout.txt").open("x").write(r["stdout"])
