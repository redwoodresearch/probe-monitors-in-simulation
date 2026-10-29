"""Runs analyze_robust_matched_POSTHOC.py in one Modal CPU container (RULE_AMENDMENT4_matched.md). The caller checks the tree is
clean and passes the commit; the pinned inputs (sha256-checked again inside) ship in the image.

  PYTHONPATH=src MATCHED_IN=<dir with the pinned inputs> uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-13_robust_1pct_POSTHOC/run_robust_g3_modal.py --n 300 --out <results json>
"""
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the module sits at /root
IN = Path(os.environ.get("MATCHED_IN", "/nonexistent"))
FILES = ["sce_meanpool_b_scores_2026-10-13_140912.npz"]
APP = "probe-shift_aswin-surya_sub-robust1pct_matched_kill-2026-11-13"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.5.4", "scipy==1.18.1").add_local_python_source("probe_shift") \
    .add_local_file(HERE / "analyze_robust_matched_POSTHOC.py", "/root/analyze_robust_matched_POSTHOC.py")
for f in FILES:
    if IN.exists():
        image = image.add_local_file(IN / f, f"/root/in/{f}")


@app.function(image=image, cpu=32, memory=32768, timeout=3600, retries=0)
def run(n: int, git: str) -> dict:
    p = subprocess.run(["python", "/root/analyze_robust_matched_POSTHOC.py", f"/root/in/{FILES[0]}", "/tmp/out.json", str(n), git], capture_output=True, text=True, cwd="/root")
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:], "json": Path("/tmp/out.json").read_text() if p.returncode == 0 else None}


@app.local_entrypoint()
def main(n: int = 300, out: str = ""):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    r = run.remote(n, git)
    print(r["stdout"], r["stderr"][-2000:] if r["rc"] else "")
    if r["rc"] == 0 and out:
        Path(out).open("x").write(r["json"])
        Path(out[:-5] + ".printout.txt").open("x").write(r["stdout"])
