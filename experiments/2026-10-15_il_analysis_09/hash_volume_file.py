"""Hash files on the convaudit work Volume inside a Modal container (sha256 and size of the Volume copy itself).

Pointer JSONs pin full records kept off git (09, 10-15 ~16:16): every pointer write re-hashes the Volume copy here.

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_il_analysis_09/hash_volume_file.py --paths xerr/il/records/<file>.json
"""

import hashlib
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-convaudit_ilhash_kill-2026-11-14"
app = modal.App(APP)
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes={"/work": vol})
def hash_files(paths: list[str]) -> dict:
    out = {}
    for p in paths:
        if p.startswith("/") or ".." in p:
            raise ValueError(f"relative Volume path required: {p}")
        root = Path("/work").resolve(strict=True)
        f = (root / p).resolve(strict=True)
        if root not in f.parents or not f.is_file():
            raise ValueError(f"not a file under the Volume root: {p}")
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[p] = {"sha256": h.hexdigest(), "size_bytes": f.stat().st_size}
    return out


@app.local_entrypoint()
def main(paths: str):
    print(json.dumps(hash_files.remote([x for x in paths.split(",") if x]), indent=1))
