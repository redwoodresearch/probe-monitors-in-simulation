"""Check one phase's uploaded .eval files on the Volume against its manifest (RULE.md amendment 6): count, bytes and
sha256 of every file under simh_evals_v1/ of that phase's launch; no file is opened as an archive. A copy of 2e's
verify_q4b_upload.py @ 0bf6ec95 with this job's app, Volume and folder, and the manifest passed in. One Modal CPU
container.
  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run <this file> --manifest <manifest.json>
"""

import hashlib
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3simh_verify_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=(4.0, 4.0), memory=(8192, 8192), timeout=900, retries=0, startup_timeout=300, scaledown_window=60,
              volumes={"/v": modal.Volume.from_name(VOL, version=2).read_only()})
def verify(files: list[dict], launch: str) -> dict:
    root = Path("/v/simh_evals_v1")
    on_vol = sorted(str(p.relative_to(root)) for p in (root / launch).rglob("*") if p.is_file())
    want = {f["path"]: f for f in files}
    bad = [p for p, f in want.items() if not (root / p).exists() or (root / p).stat().st_size != f["bytes"]
           or hashlib.sha256((root / p).read_bytes()).hexdigest() != f["sha256"]]
    return {"files_on_volume": len(on_vol), "files_in_manifest": len(want),
            "extra_on_volume": len(set(on_vol) - set(want)), "missing_or_mismatched": len(bad),
            "passes": not bad and set(on_vol) == set(want)}


@app.local_entrypoint()
def main(manifest: str):
    m = json.loads(Path(manifest).read_text())
    (launch,) = {f["launch"] for f in m["files"]}
    print(json.dumps(verify.remote(m["files"], launch)))
