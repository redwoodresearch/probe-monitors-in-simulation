"""Positive check for count_attachments.py ($0, counts only): where unresolved `attachment://` strings sit in the cached
samples the k = 1 and Oct 7 inputs come from, per top-level sample field (messages, metadata, store, events, ...). If the
counter finds references in some fields but none in messages, metadata and store, the zero counts above are the counter
working on fields that hold none, not a counter that can't see them.

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_k_replies/count_attachments_where.py --cache <dir>
"""

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REF = "attachment://"


def refs(x) -> int:
    if isinstance(x, str):
        return x.count(REF)
    if isinstance(x, dict):
        return sum(refs(v) for v in x.values())
    if isinstance(x, list):
        return sum(refs(v) for v in x)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    out = {"label": "attachment:// references per sample field in the cached samples (counts only)", "cohorts": {}}
    for d in sorted((a.cache / "samples").rglob("*.json.gz")):
        alias = str(d.parent.relative_to(a.cache / "samples"))
        c = out["cohorts"].setdefault(alias, Counter())
        for s in json.loads(gzip.decompress(d.read_bytes())):
            c["samples"] += 1
            c["samples_with_attachment_entries"] += bool(s.get("attachments"))
            for k, v in s.items():
                if k != "attachments":
                    c[f"refs_in_{k}"] += refs(v)
    out["cohorts"] = {k: dict(v) for k, v in out["cohorts"].items()}
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attachment_where_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for k, v in out["cohorts"].items():
        print(k[-60:], {kk: vv for kk, vv in v.items() if vv})
    print("->", dst)


if __name__ == "__main__":
    main()
