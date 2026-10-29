"""Read-only: sha256 of every per-draw npz that a record in results/ (and results/superseded/) points to on the solavg Volume, vs the sha the
record stores. Collision check after pooled R/D 151846 shared one npz path (2e, 15:5x).

  PYTHONPATH=src <modal> run experiments/2026-10-15_fpr_sweep_507_prep/check_npz_shas.py
"""
import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_kill-2026-11-14")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=(2, 2), memory=(4096, 4096), timeout=600, retries=0, volumes={"/sol": vol})
def shas(paths: list) -> dict:
    out = {}
    for p in paths:
        f = Path("/sol") / p
        out[p] = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "MISSING"
    return out


@app.local_entrypoint()
def main():
    want = {}
    for fn in sorted(list((HERE / "results").glob("*.json")) + list((HERE / "results" / "superseded").glob("*.json"))):
        try:
            d = json.loads(fn.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        npz = d.get("npz") if isinstance(d, dict) else None
        if isinstance(npz, dict) and npz.get("path") and npz.get("sha256"):
            want.setdefault(npz["path"], []).append({"record": str(fn.relative_to(HERE)), "expected": npz["sha256"]})
    got = shas.remote(sorted(want))
    rows = [{"path": p, "found": got[p], **r, "match": got[p] == r["expected"]} for p, rs in want.items() for r in rs]
    print(json.dumps({"checked": len(rows), "mismatch": [r for r in rows if not r["match"]], "rows": rows}, indent=1))
