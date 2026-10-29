"""Run eval_error_order_check.py on Modal CPU against ONE .eval on a Modal Volume (d9 ~02:5x: Y's t899 after box3's planned
poweroff; the chain relayed every manifest .eval to its Volume). Strictly read-only (Codex r64): the Volume is NOT mounted (a mount
is committed at container shutdown); the container reads exactly that file with Volume.read_file, asserts its sha256, writes it to
a fresh container-local temp folder, and runs the committed check there. The .eval never reaches the laptop. Output: the check's
ORDER line only (counts, timestamps, class names), or ORDER_FAILED <class>. Cost: one small CPU container for about a minute.
  uv run --no-project --python 3.12 --with modal==1.2.6 python run_eval_order_modal.py <volume> <path on volume> <sha256> <run key>
"""

import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-sp493_t899-order_kill-2026-10-23"
image = modal.Image.debian_slim(python_version="3.12").pip_install("zstandard==0.23.0").add_local_file(
    HERE / "eval_error_order_check.py", "/check/eval_error_order_check.py")
app = modal.App(APP, image=image)


@app.function(timeout=600, cpu=1.0)
def run(vol_name: str, path: str, sha: str, key: str) -> str:
    try:
        import hashlib, os, subprocess, tempfile
        data = b"".join(modal.Volume.from_name(vol_name).read_file(path))
        if hashlib.sha256(data).hexdigest() != sha:
            return "ORDER_FAILED ShaMismatch"
        d = tempfile.mkdtemp()
        with open(os.path.join(d, os.path.basename(path)), "wb") as f:
            f.write(data)
        r = subprocess.run([sys.executable, "-I", "/check/eval_error_order_check.py", d, key], capture_output=True, text=True, timeout=300)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("ORDER ", "ORDER_FAILED "))]
        return lines[-1] if lines else "ORDER_FAILED NoOutput"
    except BaseException as x:  # noqa: BLE001 - only the class name leaves (no message, no traceback)
        return "ORDER_FAILED " + (type(x).__name__ if type(x).__name__.isidentifier() else "Exception")


if __name__ == "__main__":
    try:  # Codex r64b: parse inside the handler; print only after app.run() has exited cleanly; one failure format
        vol_name, path, sha, key = sys.argv[1:5]
        with app.run():
            res = run.remote(vol_name, path, sha, key)
    except BaseException as x:  # noqa: BLE001
        print("ORDER_FAILED " + (type(x).__name__ if type(x).__name__.isidentifier() else "Exception"))
        sys.exit(1)
    print(res)
