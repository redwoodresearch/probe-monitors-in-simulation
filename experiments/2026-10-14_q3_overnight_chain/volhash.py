"""Overnight chain (RULE.md here): the relay's Volume-side check, deployed once (`modal deploy`): sha256 of each named file on the
chain's Volume (a missing file maps to None). upload_replicate.py's hash_on_volume, as a deployed function so the chain calls it
without starting an app each time. Hashes only; it never returns contents."""
import modal

APP = "probe-shift_aswin-surya_sub-q3chain_volhash_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True, version=2)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/vol": vol}, timeout=1800, cpu=2, retries=0, max_containers=4)
def hash_on_volume(paths: list[str]) -> dict:
    import hashlib
    from pathlib import Path

    vol.reload()
    out = {}
    for p in paths:
        f = Path("/vol") / p
        if not f.is_file():
            out[p] = None
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[p] = h.hexdigest()
    return out
