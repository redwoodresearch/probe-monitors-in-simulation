"""Fresh-clone check of the post's reproducibility folder: run the documented one command in a clean Modal container that has ONLY
the post/ folder of a fresh clone (no sibling repos, no Modal volumes, no secrets), with uv fetching Python 3.12 and the pinned
packages over the network once. Returns the command's exit code and full log, uv.lock (created on first run), and the rebuild's
output folder as a tar, so 0d can commit uv.lock and the Figure 3 reference after the first run and re-run to a clean pass.

  git clone --depth 1 --branch aswin/post-repro file:///Users/aswinsurya/probe-distribution-shift-from-sims <DIR>/fresh
  M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
  POST_DIR=<DIR>/fresh/post "$M" run experiments/2026-10-26_post_repro_check/run_fresh_clone_check.py --out-dir <RESULTS>
"""

import hashlib
import io
import os
import tarfile
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-post-repro_freshclone_kill-2026-11-25"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("uv==0.12.19")
if modal.is_local():
    image = image.add_local_dir(os.environ["POST_DIR"], "/work/post", copy=True)  # ONLY post/ of the fresh clone


@app.function(image=image, cpu=(2.0, 2.0), memory=(8192, 8192), timeout=1800, startup_timeout=300, scaledown_window=10, retries=0)
def check() -> dict:
    import subprocess

    env = {k: v for k, v in os.environ.items() if not k.startswith("MODAL_")}  # no Modal credentials inside the run
    env["UV_PYTHON_PREFERENCE"] = "only-managed"  # Codex: uv must fetch its own Python 3.12, as a fresh machine would
    try:
        r = subprocess.run(["uv", "run", "--project", "post", "post/rebuild_post.py", "--out", "/work/out"], cwd="/work", env=env,
                           capture_output=True, text=True, timeout=1500)  # below Modal's 1800 s, so logs survive
        code, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, so, se = "timeout (1500 s)", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        if Path("/work/out").exists():
            tf.add("/work/out", arcname="out")
    lock = Path("/work/post/uv.lock")
    return {"returncode": code, "stdout": so, "stderr": se, "out_tar": buf.getvalue(),
            "uv_lock": lock.read_bytes() if lock.exists() else None}


@app.local_entrypoint()
def main(out_dir: str):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    res = check.remote()
    d = Path(out_dir) / datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=False)
    (d / "run.log").write_text(f"returncode {res['returncode']}\n--- stdout\n{res['stdout']}\n--- stderr\n{res['stderr']}")
    (d / "out.tar.gz").write_bytes(res["out_tar"])
    if res["uv_lock"] is not None:
        (d / "uv.lock").write_bytes(res["uv_lock"])
    print("FRESH_CLONE_CHECK", "returncode", res["returncode"], "dir", d,
          "uv_lock", hashlib.sha256(res["uv_lock"]).hexdigest() if res["uv_lock"] else None)
