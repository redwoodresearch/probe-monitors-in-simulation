"""Freeze the returned adjudication sheet (2026-10-13). $0, local; reads no text field.

- Checks the returned sheet's sha256 against the value 0d relayed.
- Writes a labels-only copy (audit_id, label, confidence) to results/. The CSV parser necessarily splits the note
  field in memory to find row boundaries (notes may hold quoted newlines); each row is cut to its first three fields at
  once, and no note value is printed, written or kept.
- Every file is read once; its hash, size, line count and upload all come from that one byte snapshot.
- Uploads the full sheet and its three side files create-only (If-None-Match '*', x-amz-meta-sha256), checks a fresh
  download byte for byte, and writes the manifest.
- Writes the freeze record: both sha256s, counts, and each side file's bytes and line count.

  uv run --no-project --python 3.12 python freeze_llm_return.py <returned results dir>
"""

import collections
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
STAMP = "2026-10-13_004742"
SHEET = f"adjudication_sheet_llm_{STAMP}.csv"
SIDE = [f"adjudication_freeze_llm_{STAMP}.json", f"adjudication_judgments_llm_{STAMP}.jsonl", f"adjudication_summary_llm_{STAMP}.json"]
SHEET_SHA = "d24ae5434301e2bff45ec198f9a6328a35eb2ee2f920e4039cb3742ef334be59"  # relayed by 0d, 2026-10-13
BUCKET = "individual-aswin-surya-probe-shift-kill-2027-04-06"
PREFIX = f"experiments/2026-10-12_q3_sabotage_adjudication/results/llm_return_{STAMP}"
AWS = ["aws", "--profile", "redwood-sandbox", "--region", "us-west-2"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def need(ok: bool, msg: str):
    if not ok:
        raise SystemExit(msg)


def put_once(name: str, body: bytes) -> dict:
    key = f"{PREFIX}/{name}"
    head = subprocess.run(AWS + ["s3api", "head-object", "--bucket", BUCKET, "--key", key], capture_output=True, text=True)
    need(head.returncode != 0, f"{key} already exists; not overwritten")
    with tempfile.TemporaryDirectory() as d:
        snap, dst = Path(d) / "snapshot", Path(d) / "download"
        snap.write_bytes(body)
        put = subprocess.run(AWS + ["s3api", "put-object", "--bucket", BUCKET, "--key", key, "--body", str(snap),
                                    "--if-none-match", "*", "--metadata", f"sha256={sha(body)}"], capture_output=True, text=True, check=True)
        subprocess.run(AWS + ["s3", "cp", "--quiet", f"s3://{BUCKET}/{key}", str(dst)], check=True)
        need(dst.read_bytes() == body, f"{key}: fresh download differs")
    return {"file": name, "s3": f"s3://{BUCKET}/{key}", "bytes": len(body), "sha256": sha(body),
            "etag": json.loads(put.stdout)["ETag"]}


def main(src: str):
    src = Path(src)
    snap = {n: (src / n).read_bytes() for n in [SHEET] + SIDE}  # one read per file
    raw = snap[SHEET]
    need(sha(raw) == SHEET_SHA, "the returned sheet's sha256 differs from the relayed value")
    reader = csv.reader(io.StringIO(raw.decode()))
    need(next(reader) == ["audit_id", "label", "confidence", "note"], "unexpected header")
    rows = [dict(zip(("audit_id", "label", "confidence"), r[:3])) for r in reader]
    need(len(rows) == 100 and len({r["audit_id"] for r in rows}) == 100, "expected 100 unique audit_ids")
    need({r["label"] for r in rows} <= {"sabotage", "not_sabotage", "unclear"}, "unexpected label value")
    need({r["confidence"] for r in rows} <= {"sure", "unsure"}, "unexpected confidence value")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["audit_id", "label", "confidence"], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    labels = buf.getvalue().encode()
    out = HERE / "results" / f"adjudication_labels_llm_{STAMP}.csv"
    with out.open("xb") as fh:
        fh.write(labels)
    side = {n: {"bytes": len(snap[n]), "lines": snap[n].count(b"\n"), "sha256": sha(snap[n])} for n in SIDE}
    manifest = {"note": "the returned LLM-adjudicated sheet and its three side files; create-only, sha256 verified on a fresh download. "
                        "The sheet's note column and the judgments may quote run text: S3 only, never git.",
                "retrieve": "aws s3 cp --profile redwood-sandbox <s3> .", "files": [put_once(n, snap[n]) for n in [SHEET] + SIDE]}
    (HERE / "results" / f"manifest_adjudication_llm_return_{STAMP}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    record = {"frozen_at_pt": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
              "original_sheet": {"file": SHEET, "sha256": SHEET_SHA, "bytes": len(raw), "columns": ["audit_id", "label", "confidence", "note"]},
              "labels_only_copy": {"file": out.name, "sha256": sha(labels), "bytes": len(labels), "columns": ["audit_id", "label", "confidence"]},
              "rows": len(rows), "label_counts": dict(collections.Counter(r["label"] for r in rows)),
              "confidence_counts": dict(collections.Counter(r["confidence"] for r in rows)), "side_files": side,
              "manifest": f"results/manifest_adjudication_llm_return_{STAMP}.json"}
    (HERE / "results" / f"adjudication_freeze_record_{STAMP}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
