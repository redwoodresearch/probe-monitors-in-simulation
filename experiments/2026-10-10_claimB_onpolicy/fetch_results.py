"""Fetch this experiment's result files from S3 (results_manifest_v*.json: path, S3 key, sha256), checking each file's
sha256. Git keeps the manifest; the files live on S3 (new keys only). Uses the AWS profile redwood-sandbox.

  uv run python experiments/2026-10-10_claimB_onpolicy/fetch_results.py [path-substring ...]
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]


def main(patterns: list[str]):
    files = {}
    for m in sorted(HERE.glob("results_manifest_v*.json")):  # append-only: each version lists the files it added
        files |= json.loads(m.read_text())["files"]
    for path, f in sorted(files.items()):
        if patterns and not any(p in path for p in patterns):
            continue
        dst = REPO / path
        if not (dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == f["sha256"]):
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["aws", "s3", "cp", f["s3"], str(dst), "--profile", "redwood-sandbox", "--only-show-errors"], check=True)
        if hashlib.sha256(dst.read_bytes()).hexdigest() != f["sha256"]:  # checked always, not as an assert (python -O)
            raise ValueError(f"{path}: sha256 differs from the manifest")
        print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
