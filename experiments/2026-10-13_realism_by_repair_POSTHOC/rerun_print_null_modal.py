"""Re-run print_null_results.py as a Modal CPU function on the 16 checkpoint JSONs, for 09's audit (10-14). It prints the
script's stdout to --out, for a byte comparison with the committed results/null_gpu_full_20261013_222056.printout.txt.

  uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run experiments/2026-10-13_realism_by_repair_POSTHOC/rerun_print_null_modal.py --out <file>
"""
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-realism-null_printcheck_kill-2026-11-13")
ckpt = modal.Volume.from_name("probe-shift_aswin-surya_sub-realism-null_ckpt_kill-2026-11-13", version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").add_local_file(HERE / "print_null_results.py", "/root/print_null_results.py")


@app.function(image=image, cpu=1, memory=1024, timeout=300, retries=0, volumes={"/ckpt": ckpt})
def run() -> str:
    return subprocess.run(["python", "-I", "/root/print_null_results.py", "/ckpt/full_20261013_222056"], capture_output=True, text=True, check=True).stdout


@app.local_entrypoint()
def main(out: str):
    Path(out).write_text(run.remote())
