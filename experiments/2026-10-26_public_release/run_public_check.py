"""Internal: run the public release folder the way a new user would, in a clean Modal container that holds ONLY release/ (no other
repository files, no Modal volumes, no secrets). uv fetches Python 3.12 and matplotlib, then the two documented commands run:
`uv run make_numbers.py` and `uv run make_figures.py`. Returns both logs, their exit codes, output/ as a tar and uv.lock.

  M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
  RELEASE_DIR=<clone>/release "$M" run experiments/2026-10-26_public_release/run_public_check.py --out-dir <RESULTS>
SPEND, worst-case bound: 1 call x (startup_timeout 300 + timeout 900 + scaledown_window 10) s at 2 cores and 4 GiB =
1,210 x ($0.0000262 + $0.00000888) = $0.0424468. Typical: about $0.01.
"""

import hashlib
import io
import os
import tarfile
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-public-release_check_kill-2026-11-25"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("uv==0.12.19")
if modal.is_local():
    image = image.add_local_dir(os.environ["RELEASE_DIR"], "/work/release", copy=True)  # ONLY release/


@app.function(image=image, cpu=(2.0, 2.0), memory=(4096, 4096), timeout=900, startup_timeout=300, scaledown_window=10, retries=0)
def check() -> dict:
    import subprocess

    env = {k: v for k, v in os.environ.items() if not k.startswith("MODAL_")}  # no Modal credentials inside the run
    env["UV_PYTHON_PREFERENCE"] = "only-managed"  # uv fetches its own Python 3.12, as on a fresh machine
    out = {}
    for name in ("make_numbers.py", "make_figures.py"):
        try:
            r = subprocess.run(["uv", "run", name], cwd="/work/release", env=env, capture_output=True, text=True, timeout=600)
            out[name] = {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        except subprocess.TimeoutExpired:
            out[name] = {"returncode": "timeout (600 s)", "stdout": "", "stderr": ""}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        if Path("/work/release/output").exists():
            tf.add("/work/release/output", arcname="output")
    lock = Path("/work/release/uv.lock")
    return {"runs": out, "out_tar": buf.getvalue(), "uv_lock": lock.read_bytes() if lock.exists() else None}


@app.local_entrypoint()
def main(out_dir: str):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    res = check.remote()
    d = Path(out_dir) / datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=False)
    for name, r in res["runs"].items():
        (d / f"{name}.log").write_text(f"returncode {r['returncode']}\n--- stdout\n{r['stdout']}\n--- stderr\n{r['stderr']}")
    (d / "output.tar.gz").write_bytes(res["out_tar"])
    if res["uv_lock"] is not None:
        (d / "uv.lock").write_bytes(res["uv_lock"])
    print("PUBLIC_CHECK", {n: r["returncode"] for n, r in res["runs"].items()}, "dir", d,
          "uv_lock", hashlib.sha256(res["uv_lock"]).hexdigest() if res["uv_lock"] else None)
    if any(r["returncode"] != 0 for r in res["runs"].values()):
        raise SystemExit(1)  # logs are saved above; a failed or timed-out script fails the check
