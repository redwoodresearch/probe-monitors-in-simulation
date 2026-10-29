"""Addendum 20d, step 1: a NAMES-ONLY listing of A's per-draw npz keys (no array is read), to find the exact per-origin catch and D
array names that addendum 20c's pre-stated names missed. Output: the distinct (origin part, name part) pairs under FIBS|uncut C,
joint, 0.01, and per-pattern key counts over the whole file. Strings and counts only.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run npz_keys20d.py
"""

import json
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
NPZ_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
NPZ = ("xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz",
       "5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1")
PREFIX = "FIBS_uncut_C__pooled_all_origins__joint__0.01__"

app = modal.App(OUT)
SVOL = modal.Volume.from_name(NPZ_VOL)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes={"/sv": SVOL.read_only()})
def names() -> dict:
    import hashlib
    import zipfile

    p = Path("/sv", NPZ[0])
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == NPZ[1], "npz sha"
    with zipfile.ZipFile(p) as z:  # member names only; no array is opened
        keys = [n[:-4] if n.endswith(".npy") else n for n in z.namelist()]
    under = sorted(k[len(PREFIX):] for k in keys if k.startswith(PREFIX))
    pairs = sorted({tuple(u.split("__", 1)) if "__" in u else (u, "") for u in under})
    origins = sorted({a for a, _ in pairs})
    return {"n_keys": len(keys), "prefix": PREFIX, "n_under_prefix": len(under), "origin_parts": origins,
            "names_by_origin": {o: sorted(b for a, b in pairs if a == o) for o in origins},
            "n_keys_with_catch_": sum("__catch_" in k for k in keys), "n_keys_ending_D": sum(k.endswith("__D") for k in keys)}


@app.local_entrypoint()
def main():
    out = names.remote()
    path = Path(__file__).parent / "results" / f"npz_keys20d_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("n_keys", "n_under_prefix", "origin_parts", "n_keys_with_catch_", "n_keys_ending_D")}))
    print("wrote", path)
