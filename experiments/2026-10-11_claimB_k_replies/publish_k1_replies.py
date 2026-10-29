"""k = 1 stage: put the simulated replies where the Modal reads find them. From <cache>/sim_results.jsonl (one row per
site, the latest), writes <cache>/replies.jsonl ({site, status, reader_output, tool, record_sha256}) and uploads it to the
Volume (claimB/k1_v1/replies.jsonl, never overwritten) and to S3 (the experiment's prefix, --if-none-match) with a
sha256 manifest in results/. Text never enters the repo. Every site's status is counted; only "ok" rows carry text.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' python experiments/2026-10-11_claimB_k_replies/publish_k1_replies.py \\
      --cache <dir> --records results/k1_records_<stamp>.json
"""

import argparse
import hashlib
import io
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
OUT, BUCKET = "claimB/k1_v1", "individual-aswin-surya-probe-shift-kill-2027-04-06"
KEY = "experiments/2026-10-11_claimB_k_replies/data/replies.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    a = p.parse_args()
    man = {m["site"]: m for m in json.loads((HERE / a.records).read_text())["manifest"]}
    rows = {}
    for line in (a.cache / "sim_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["site"]] = r  # the latest row per site
    if set(rows) - set(man):
        raise SystemExit("results hold sites the records manifest doesn't")
    out = a.cache / "replies.jsonl"
    with open(out, "x") as fh:
        for site in sorted(man):
            r = rows.get(site)
            status = r["status"] if r else "not_run"
            if r and r["record_sha256"] != man[site]["record_sha256"]:
                raise SystemExit(f"site {site}: result made from another record")
            fh.write(json.dumps({"site": site, "status": status, "tool": man[site]["tool"], "record_sha256": man[site]["record_sha256"],
                                 "reader_output": r["result"]["reader_output"] if status == "ok" else None}) + "\n")
    raw = out.read_bytes()
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    try:
        have = {e.path for e in vol.listdir(OUT)}
    except Exception:  # noqa: BLE001  (no folder yet)
        have = set()
    if f"{OUT}/replies.jsonl" in have:
        raise SystemExit(f"{OUT}/replies.jsonl exists; never overwritten")
    with vol.batch_upload() as b:
        b.put_file(io.BytesIO(raw), f"{OUT}/replies.jsonl")
    subprocess.run(["aws", "--profile", "redwood-sandbox", "s3api", "put-object", "--bucket", BUCKET, "--key", KEY, "--body", str(out),
                    "--if-none-match", "*"], check=True, capture_output=True)
    back = subprocess.run(["aws", "--profile", "redwood-sandbox", "s3", "cp", f"s3://{BUCKET}/{KEY}", "-"], check=True, capture_output=True).stdout
    sha = hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(back).hexdigest() != sha:
        raise SystemExit("S3 copy differs")
    statuses = Counter(json.loads(l)["status"] for l in raw.decode().splitlines())
    res = {"label": "k = 1 simulated replies (text on the Volume and S3 only)", "volume_path": f"{OUT}/replies.jsonl", "s3": f"s3://{BUCKET}/{KEY}",
           "sha256": sha, "bytes": len(raw), "statuses": dict(statuses),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"k1_replies_manifest_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("sha256", "bytes", "statuses")}))


if __name__ == "__main__":
    main()
