"""Follow-up (c): upload staged runs from data/ to the la-activations Volume (laptop -> Modal; new paths only), then check
every file's sha256 against the run's manifest.json inside a tiny Modal CPU container (pennies). Refuses a run that
already exists on the Volume. Staged inputs hold transcript text: Modal only, never S3 or git.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \
      experiments/2026-10-10_claimB_onpolicy/upload_staged.py --runs <run>[,<run>...]
"""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
app = modal.App("probe-shift_aswin-surya_sub-claimb_onpol-upload_kill-2026-11-10")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=2, memory=4096, volumes={"/acts": vol}, timeout=600, retries=0)
def verify(run: str) -> dict:
    vol.reload()
    man = json.loads((Path("/acts") / run / "manifest.json").read_text())
    bad = [n for n, h in man["files"].items() if hashlib.sha256((Path("/acts") / run / n).read_bytes()).hexdigest() != h]
    return {"run": run, "files": len(man["files"]), "sha256_mismatch": bad}


@app.local_entrypoint()
def main(runs: str):
    for run in runs.split(","):
        local = HERE / "data" / run
        man = json.loads((local / "manifest.json").read_text())
        for n, h in man["files"].items():  # the local copy first
            if hashlib.sha256((local / n).read_bytes()).hexdigest() != h:
                raise SystemExit(f"{run}/{n}: local file differs from its manifest")
        try:
            next(iter(vol.listdir(run)))
            raise SystemExit(f"{run} already exists on the Volume; staging runs are written once")
        except (FileNotFoundError, modal.exception.NotFoundError, StopIteration):
            pass
        with vol.batch_upload() as batch:  # no force: an existing file fails the upload
            for f in sorted(p for p in local.rglob("*") if p.is_file()):
                batch.put_file(f, f"{run}/{f.relative_to(local)}")
        res = verify.remote(run)
        print(json.dumps(res), flush=True)
        if res["sha256_mismatch"]:
            raise SystemExit(f"{run}: {len(res['sha256_mismatch'])} files differ after upload")
