"""Move one replicate's files from the laptop's copy of the run folder to a Modal Volume (RULE.md, Amendments 2-3; 0d's
September path via d9). The laptop only moves files: it never opens, prints or parses their contents, it only hashes them.

For each line of <run>/upload_manifest_<part>.tsv (path relative to the run, bytes, sha256; written on the devbox by
gate1.py final from every attempt directory the launcher recorded):
  1. the laptop copy must exist with that size and sha256, and the laptop must have >= 50 GB free; else nothing uploads;
  2. one batch upload of exactly those files to VOLUME:/evals_v1/<run name>/<path> (2e's layout; force=True since Amendment 13, so a
     retried part is idempotent: the same bytes again, every hash checked in step 3);
  3. a Modal CPU function hashes each uploaded file on the Volume; every hash must equal the manifest's;
  4. only then each laptop copy is deleted (Amendment 13: a resumed part first verifies on the Volume the files an earlier run
     already deleted, then uploads the rest) (part `common` also uploads the six upload_manifest_*.tsv files, metadata only,
     and keeps them on the laptop for the later parts).
Every step is logged per file, tab-separated, to <log> (outside git): path, bytes, sha256 (manifest), sha256 (Volume),
uploaded, verified, deleted (PT times). A failed check stops before any deletion. One part at a time, in order:
common, r1, ..., r5.

  uv run --no-project --python 3.12 --with modal==1.6.1 python upload_replicate.py <local run dir> <part> <log>
"""
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3realhonest_volhash_kill-2026-11-14"
MIN_FREE = 50 * 1024**3
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")
VOLUME = "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14"  # this job's Volume (version 2, as 2e's): .eval files under evals_v1/<run>/, staged sets, activations
vol = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)


def stamp() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@app.function(image=image, volumes={"/vol": vol}, timeout=1800, cpu=2, retries=0)
def hash_on_volume(paths: list[str]) -> dict:
    """sha256 of each file on the Volume (a missing file maps to None)."""
    import hashlib as hl
    from pathlib import Path as P

    out = {}
    for p in paths:
        f = P("/vol") / p
        if not f.is_file():
            out[p] = None
            continue
        h = hl.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[p] = h.hexdigest()
    return out


def main(run: Path, part: str, log_path: Path) -> None:
    assert part in ("common", "r1", "r2", "r3", "r4", "r5"), part
    manifest = run / f"upload_manifest_{part}.tsv"
    rows = [line.rstrip("\n").split("\t") for line in manifest.read_text().splitlines() if line.strip()]
    assert rows and all(len(r) == 3 for r in rows), f"{manifest}: malformed"
    keep = set()
    if part == "common":  # the six manifests themselves (names, bytes, hashes only) go too, for the Volume-side upload list;
        for name in sorted(p.name for p in run.glob("upload_manifest_*.tsv")):  # they stay on the laptop for r1..r5
            rows.append([name, str((run / name).stat().st_size), sha256(run / name)])
            keep.add(name)
        expect = {f"upload_manifest_{x}.tsv" for x in ("common", "r1", "r2", "r3", "r4", "r5")}
        assert keep == expect, f"upload manifests: found {sorted(keep)}, expected {sorted(expect)}"
    free = shutil.disk_usage(run).free
    assert free >= MIN_FREE, f"laptop free disk {free / 1024**3:.0f} GB < 50 GB"
    remote = {path: f"evals_v1/{run.name}/{path}" for path, _, _ in rows}
    local = [r for r in rows if (run / r[0]).exists()]
    gone = [r for r in rows if not (run / r[0]).exists()]  # a resumed part: deleted only after an earlier verified upload
    if gone:
        with modal.enable_output(), app.run():
            seen = hash_on_volume.remote([remote[p] for p, _, _ in gone])
        assert all(seen.get(remote[p]) == d for p, _, d in gone), f"{part}: files missing locally and not verified on the Volume"
    for path, size, digest in local:  # every check before anything uploads
        f = run / path
        assert f.is_file() and f.stat().st_size == int(size), f"{path}: missing or size differs"
        assert sha256(f) == digest, f"{path}: sha256 differs from the manifest"
    if local:
        with vol.batch_upload(force=True) as batch:  # Amendment 13: a retried part re-puts the same bytes; step 3 checks every hash
            for path, _, _ in local:
                batch.put_file(str(run / path), "/" + remote[path])
    uploaded = stamp()
    with modal.enable_output(), app.run():
        on_vol = hash_on_volume.remote([remote[p] for p, _, _ in rows])
    verified = stamp()
    bad = [p for p, _, d in rows if on_vol.get(remote[p]) != d]
    with open(log_path, "a") as log:
        for path, size, digest in rows:
            ok = path not in bad
            log.write("\t".join([part, path, size, digest, str(on_vol.get(remote[path])), uploaded,
                                 verified if ok else "MISMATCH", ""]) + "\n")
    assert not bad, f"{len(bad)} of {len(rows)} files differ on the Volume; nothing deleted"
    with open(log_path, "a") as log:
        for path, size, digest in rows:
            if path in keep or not (run / path).exists():
                continue
            (run / path).unlink()
            log.write("\t".join([part, path, size, digest, digest, uploaded, verified, f"deleted {stamp()}"]) + "\n")
    print(f"{part}: {len(rows)} files uploaded to {VOLUME}:/evals_v1/{run.name}/, all sha256 verified on the Volume, "
          f"laptop copies deleted; log {log_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
