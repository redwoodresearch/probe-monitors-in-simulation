"""Q4 (B), RULE.md Amendment 5: check the uploaded .eval files on the Volume against the committed upload manifest (count,
bytes and sha256 of every file; no file is opened as an archive). One Modal CPU container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach experiments/2026-10-13_q4_why_weaker_POSTHOC/verify_q4b_upload.py
"""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q4own_verify_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=4, memory=8192, timeout=900, retries=0,
              volumes={"/v": modal.Volume.from_name(VOL).read_only()})
def verify(files: list[dict]) -> dict:
    root = Path("/v/evals_v1")
    on_vol = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    want = {f["path"]: f for f in files}
    bad = [p for p, f in want.items() if not (root / p).exists() or (root / p).stat().st_size != f["bytes"]
           or hashlib.sha256((root / p).read_bytes()).hexdigest() != f["sha256"]]
    return {"files_on_volume": len(on_vol), "files_in_manifest": len(want), "extra_on_volume": len(set(on_vol) - set(want)),
            "missing_or_mismatched": len(bad), "passes": not bad and set(on_vol) == set(want)}


@app.local_entrypoint()
def main():
    m = json.loads((HERE / "results" / "manifest_q4b_eval_upload_v1.json").read_text())
    print(json.dumps(verify.remote(m["files"])))
