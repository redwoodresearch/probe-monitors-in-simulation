"""Verify the rr-2x2 record archive against the Volume, with no re-download: hash every file of the two record directories
on the Volume (sha256 and md5) in one Modal CPU container, and compare each archived object's S3 size, ETag (single-part, so
the md5 of the content) and sha256 metadata with the Volume copy. Writes the manifest; counts only.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-13_rr_2x2_POSTHOC/verify_rr_records_archive.py --s3-list <list-objects-v2 json> --upload-log <log>
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-rr2x2_archive_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
DIRS = ("claimcna_c_5b8e8a3a98ca", "claimcnad3c_5527e0d39a08")
BUCKET, PREFIX = "individual-aswin-surya-probe-shift-kill-2027-04-06", "experiments/2026-10-13_rr_2x2_POSTHOC/records/"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=2, memory=4096, timeout=1200, retries=0, volumes={"/v": modal.Volume.from_name(VOL).read_only()})
def verify(expected: dict) -> dict:
    import hashlib

    files = []
    for d in DIRS:
        for p in sorted(Path("/v", d).glob("*")):
            h, m = hashlib.sha256(), hashlib.md5()
            with p.open("rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
                    m.update(b)
            rel, e = f"{d}/{p.name}", expected.get(f"{d}/{p.name}")
            row = {"file": rel, "bytes": p.stat().st_size, "sha256": h.hexdigest(), "md5": m.hexdigest()}
            if e is not None:
                row["s3"] = f"s3://{BUCKET}/{PREFIX}{rel}"
                row["verified"] = e["bytes"] == row["bytes"] and e["etag"] == row["md5"] and e["sha256"] == row["sha256"]
            files.append(row)
    return {"files": files, "volume_files": len(files), "on_s3": sum("s3" in r for r in files),
            "verified": sum(bool(r.get("verified")) for r in files), "s3_objects_not_on_volume": sorted(set(expected) - {r["file"] for r in files})}


@app.local_entrypoint()
def main(s3_list: str, upload_log: str):
    meta = {}
    for line in Path(upload_log).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            meta[parts[0]] = parts[2]
    expected = {}
    for key, size, etag in json.loads(Path(s3_list).read_text()):
        rel = key.removeprefix(PREFIX)
        expected[rel] = {"bytes": size, "etag": etag.strip('"'), "sha256": meta.get(rel)}
    res = verify.remote(expected)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
    man = {"note": (f"rr-2x2 grid and early-position records (RULE.md, Amendment 1): archived {res['verified']}/{res['volume_files']} on S3, "
                    f"verified against the Volume copies by size, sha256 and the single-part ETag (md5), with no re-download ({stamp}). "
                    f"The rest stay only on Volume {VOL}; rows without an s3 key are not archived. No weights; scores, folds, masks, "
                    "labels and fit diagnostics only."),
           "retrieve": "aws s3 cp --profile redwood-sandbox <s3> .", "volume": VOL} | res
    dest = HERE / "results" / "manifest_rr_2x2_records_v1.json"
    dest.open("x").write(json.dumps(man, indent=1) + "\n")
    print({k: v for k, v in res.items() if k != "files"})
