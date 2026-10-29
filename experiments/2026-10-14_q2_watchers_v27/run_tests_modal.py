"""Run test_watchers_v27.py on a small Modal CPU container (no laptop compute); prints the test output.
  uv run --no-project --with modal==1.6.1 modal run experiments/2026-10-14_q2_watchers_v27/run_tests_modal.py"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-q2-watchers_v27tests_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").add_local_dir(str(HERE), "/w27", ignore=["results", "__pycache__"])


@app.function(image=image, cpu=1, memory=512, timeout=120, retries=0)
def run() -> str:
    import subprocess
    p = subprocess.run(["python3", "/w27/test_watchers_v27.py"], capture_output=True, text=True)
    return f"exit {p.returncode}\n{p.stdout}{p.stderr}"


@app.local_entrypoint()
def main():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    out = run.remote()
    print(out)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    for k in range(100):  # one file per run, never overwritten: exclusive create, with a suffix on a same-second clash
        try:
            with open(HERE / "results" / f"test_output_v27_{stamp}{'' if k == 0 else f'_{k}'}.txt", "x") as f:
                f.write(out)
            break
        except FileExistsError:
            continue
    else:
        raise SystemExit("could not create a new test output file")
    if not out.startswith("exit 0\n"):
        raise SystemExit(f"tests failed on Modal: {out.splitlines()[0]}")
