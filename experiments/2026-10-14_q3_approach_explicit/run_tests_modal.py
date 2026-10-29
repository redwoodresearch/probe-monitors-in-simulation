"""Run test_approach_explicit.py on Modal CPU (pure Python; no model, no approach text): the laptop runs no compute.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-14_q3_approach_explicit/run_tests_modal.py
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
FILES = ["2026-10-07_phase2_attack_check/approaches.py", "2026-10-06_astra_sim_honest/approach_guard.py",
         "2026-10-14_q3_approach_explicit/test_approach_explicit.py"]
image = modal.Image.debian_slim(python_version="3.12").pip_install("pytest==8.3.3")
for f in FILES:
    image = image.add_local_file(ROOT / f, f"/root/experiments/{f}")
app = modal.App("probe-shift_aswin-surya_sub-q3approach_tests_kill-2026-11-14", image=image)


@app.function(timeout=300, cpu=1)
def run() -> str:
    import subprocess

    out = subprocess.run(["python", "-m", "pytest", "-q", "/root/experiments/2026-10-14_q3_approach_explicit"],
                         capture_output=True, text=True)
    print(out.stdout[-3000:], out.stderr[-2000:])
    out.check_returncode()  # a failing test fails the Modal call and `modal run`
    return f"exit {out.returncode}\n{out.stdout[-3000:]}"


@app.local_entrypoint()
def main():
    print(run.remote())
