"""Put the adjudication key on S3 (create-only), check a fresh download, write the manifest, delete the local key.

Never prints the key's rows. The manifest holds URIs, sha256, bytes, ETags, row count and column names only; commit it
after this runs. A rerun accepts objects already on S3 only if they match byte for byte.

  uv run --no-project python publish_key.py <key-dir>/adjudication_key_<stamp>.csv [--profile redwood-sandbox]
"""

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUCKET = "individual-aswin-surya-probe-shift-kill-2027-04-06"
PREFIX = "experiments/2026-10-12_q3_sabotage_adjudication/key"


def main(key, profile):
    summary = key.with_name(f"{key.stem}.summary.json")
    k_sum = json.loads(summary.read_text())
    assert k_sum["key"] == key.name and hashlib.sha256(key.read_bytes()).hexdigest() == k_sum["key_sha256"]
    with key.open() as f:
        reader = csv.DictReader(f)
        columns, rows = reader.fieldnames, sum(1 for _ in reader)
    aws, files = ["aws", "s3api", "--profile", profile, "--output", "json"], {}
    for path in (key, summary):
        s3key = f"{PREFIX}/{path.name}"
        put = subprocess.run([*aws, "put-object", "--bucket", BUCKET, "--key", s3key, "--body", str(path),
                              "--if-none-match", "*"], capture_output=True, text=True, check=False)  # fmt: skip
        if put.returncode and "PreconditionFailed" not in put.stderr:  # an existing object must match byte for byte
            raise SystemExit(put.stderr)
        with tempfile.TemporaryDirectory() as d:
            got = Path(d) / path.name
            get = subprocess.run([*aws, "get-object", "--bucket", BUCKET, "--key", s3key, str(got)],
                                 capture_output=True, text=True, check=True)  # fmt: skip
            if got.read_bytes() != path.read_bytes():
                raise SystemExit(f"S3 copy differs: {s3key}")
        files[path.name] = {"s3_uri": f"s3://{BUCKET}/{s3key}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bytes": path.stat().st_size, "etag": json.loads(get.stdout)["ETag"]}  # fmt: skip
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"git": git, "draw_git": k_sum["git"], "seed": k_sum["seed"], "key_rows": rows, "key_columns": columns,
                "sheet": k_sum["sheet"], "sheet_sha256": k_sum["sheet_sha256"], "files": files}  # fmt: skip
    out = HERE / "results" / f"manifest_{key.stem}.json"
    out.parent.mkdir(exist_ok=True)
    if out.exists() and json.loads(out.read_text()) != manifest:
        raise SystemExit(f"{out.name} exists with different contents")
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    key.unlink()
    summary.unlink()
    print(json.dumps({"manifest": str(out), **{n: {k: v[k] for k in ("s3_uri", "sha256")} for n, v in files.items()}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", type=Path)
    ap.add_argument("--profile", default="redwood-sandbox")
    a = ap.parse_args()
    main(a.key, a.profile)
