"""Read-only: sha256 and size of pooled per-draw npz files on the solavg Volume (diagnoses an input sha mismatch; nothing is downloaded).

  modal run experiments/2026-10-15_panel_noise_matched/check_npz_shas.py --names pooled455_2026-10-15_151846_perdraw.npz,...
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("probe-shift_aswin-surya_sub-panelnm_kill-2026-11-14")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=2, memory=4096, timeout=600, retries=0,
              volumes={"/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)})
def shas(names: list) -> dict:
    o = {}
    for n in names:
        p = Path("/sol/xfer/fpr507prep") / n
        b = p.read_bytes()
        o[n] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "mtime": p.stat().st_mtime}
    return o


@app.local_entrypoint()
def main(names: str):
    print(json.dumps(shas.remote(names.split(",")), indent=1))
