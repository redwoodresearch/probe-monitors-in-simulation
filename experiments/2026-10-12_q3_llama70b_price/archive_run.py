"""Archive one Phase 2 launch to S3 as the Qwen3-32B native check's archive was made (manifest_v1): every file put
once (If-None-Match '*', never overwritten) with its sha256 as x-amz-meta-sha256, then the manifest beside the run.

  uv run --no-project --python 3.12 --with boto3 python archive_run.py --root <dir holding the run folder> --run <run> \\
      [--extra <local file>=<key suffix> ...] --note <text> [--note <text> ...]

Keys: phase2/runs/<run>/<path in the run folder>, phase2/runs/<run>.request_logs/<file> (the sibling folder, if any),
and phase2/runs/<run><suffix> for each --extra. Laptop, redwood-sandbox SSO (AWS_PROFILE).
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3

BUCKET = "individual-aswin-surya-probe-shift-kill-2027-04-06"


def main(root: str, run: str, extra: list[str], note: list[str]):
    base = Path(root)
    files = [(p, f"phase2/runs/{run}/{p.relative_to(base / run)}") for p in sorted((base / run).rglob("*")) if p.is_file()]
    logs = base / f"{run}.request_logs"
    files += [(p, f"phase2/runs/{run}.request_logs/{p.name}") for p in sorted(logs.glob("*")) if p.is_file()]
    for e in extra:
        local, suffix = e.split("=", 1)
        files.append((Path(local), f"phase2/runs/{run}{suffix}"))
    s3 = boto3.Session().client("s3")
    rows = []
    for p, key in files:
        body = p.read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        s3.put_object(Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*", Metadata={"sha256": sha})
        rows.append({"key": key, "sha256": sha, "bytes": len(body)})
        print(f"put {key} {len(body)}")
    manifest = {"run": run, "bucket": BUCKET,
                "made": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT"),
                "how": "laptop, redwood-sandbox SSO, put-object If-None-Match '*' with x-amz-meta-sha256 (archive_run.py)",
                "notes": note, "n_files": len(rows), "total_bytes": sum(r["bytes"] for r in rows), "files": rows}
    body = json.dumps(manifest, indent=1).encode()
    key = f"phase2/runs/{run}.manifest_v1.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*",
                  Metadata={"sha256": hashlib.sha256(body).hexdigest()})
    print(f"manifest s3://{BUCKET}/{key} sha256 {hashlib.sha256(body).hexdigest()} ({len(rows)} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--note", action="append", required=True)
    main(**vars(ap.parse_args()))
