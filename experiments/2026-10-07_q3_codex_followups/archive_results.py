"""Archive this folder's git-ignored results to S3 from the laptop (user's SSO profile; never from a remote job).

Each file goes to s3://individual-aswin-surya-probe-shift-kill-2027-04-06/experiments/2026-10-07_q3_codex_followups/<path>
with If-None-Match (an existing object is never overwritten: identical content is reported as already there, different
content fails). The receipt (every file's key, size and SHA-256) goes to a new results/archive_<stamp>.json.

  uv run --no-project --python 3.12 --with boto3 python archive_results.py results/<file> [...]
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).parent
BUCKET, PREFIX = "individual-aswin-surya-probe-shift-kill-2027-04-06", "experiments/2026-10-07_q3_codex_followups"


def main():
    s3 = boto3.Session(profile_name="redwood-sandbox").client("s3")
    out = []
    for p in map(Path, sys.argv[1:]):
        rel = p.resolve().relative_to(HERE.resolve()).as_posix()
        body = p.read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        key = f"{PREFIX}/{rel}"
        try:
            s3.put_object(Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*", Metadata={"sha256": sha})
            status = "uploaded"
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
                raise
            there = s3.head_object(Bucket=BUCKET, Key=key)["Metadata"].get("sha256")
            if there != sha:
                raise ValueError(f"{key} exists with different content ({there} != {sha})")
            status = "already there"
        out.append({"file": rel, "key": f"s3://{BUCKET}/{key}", "bytes": len(body), "sha256": sha, "status": status})
        print(status, key, len(body))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"archive_{stamp}.json").open("x") as f:
        f.write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
