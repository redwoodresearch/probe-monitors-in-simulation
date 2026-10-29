"""I3 stage 1, on the laptop ($0; no credentials ever reach Modal): for the 97 test-environment cells the reroll shares
with claim B (count_reroll_overlap.py), download the reroll's log (S2) and claim B's first-sim log (S1, the file named in
splits/exclusions_v3.json) from S3, take each cell's sample, resolve its attachments (probe_shift.transcripts
.resolve_attachments; any placeholder left raises), check S2's reroll_provenance points to the S1 sample, and upload
{cell, environment, main_task, arm, file, sample_id, epoch, id, messages} per sample as one gzip JSONL to the Volume
(`claimB/reroll_yardstick_v1/samples.jsonl.gz`, never overwriting). Downloads live in a temporary folder deleted at
exit; the local gzip is deleted after the upload. Writes results/reroll_upload_<stamp>.json: counts, sha256s and sizes
only; nothing is printed from the logs.

  AWS_PROFILE=redwood-sandbox PYTHONPATH=src uv run --no-project --python 3.14 --with boto3 --with pyyaml \\
      --with 'modal>=1.4.3' python -I experiments/2026-10-11_claimB_yardsticks/stage_reroll_upload.py <repo root> <sce root>
"""

import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO, SCE = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(REPO / "src"))
from probe_shift.transcripts import refs_left, resolve_attachments  # noqa: E402

HERE = REPO / "experiments/2026-10-11_claimB_yardsticks"
BUCKET = "rr-sce-sim-transcripts"
S1_PREFIX = "transcripts/2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000/logs/"
VOLUME, DEST = "probe-shift_aswin-surya_sub-la-activations", "claimB/reroll_yardstick_v1/samples.jsonl.gz"


def fetch(key: str, dst: Path) -> Path:
    subprocess.run(["aws", "s3", "cp", "--quiet", f"s3://{BUCKET}/{key}", str(dst)], check=True)
    return dst


def samples_of(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        return [json.loads(z.read(n)) for n in z.namelist() if n.startswith("samples/")]


def main():
    staged = json.loads((SCE / "experiments/2026-09-29_monitor_score_variance/staged_rerolls.json").read_text())
    excl = json.loads((REPO / "splits/exclusions_v3.json").read_text())
    test = set(json.loads((REPO / "experiments/2026-10-09_fixed_sim/split.json").read_text())["test"])
    paired = set(excl["paired_attack_cells"]["cells"])
    sims = [s for s in excl["samples"] if s["arm"] == "sim-attack"]
    s1 = {s["cell"]: s for s in sims}
    if len(s1) != len(sims) or len({r["cell"] for r in staged}) != len(staged):
        raise ValueError("duplicate cells")
    rows = []
    for r in staged:
        env = re.search(r"_la-([a-z-]+?)-sim-reroll-\d+_", Path(r["source"]).name).group(1).replace("-", "_")
        if r["cell"] in paired and env in test:
            rows.append(r | {"environment": env})
    if len(rows) != 97:
        raise ValueError(f"expected 97 test overlap cells, got {len(rows)}")
    out, counts = [], {"S1": 0, "S2": 0, "s1_files": 0}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s1_cache = {}
        for r in sorted(rows, key=lambda x: x["cell"]):
            a = s1[r["cell"]]
            if a["file"] not in s1_cache:
                s1_cache[a["file"]] = {s["id"]: s for s in samples_of(fetch(S1_PREFIX + a["file"], tmp / "s1.eval"))}
                counts["s1_files"] += 1
            one = s1_cache[a["file"]].get(a["sample_id"])
            two = samples_of(fetch(r["destination"], tmp / "s2.eval"))
            if one is None or len(two) != 1:
                raise ValueError(f"{r['cell']}: S1 sample missing or S2 log has {len(two)} samples")
            two = two[0]
            prov = (two.get("metadata") or {}).get("reroll_provenance") or {}
            if not (prov.get("original_sample_id") == a["sample_id"] and str(prov.get("original_file")).endswith(a["file"])):
                raise ValueError(f"{r['cell']}: S2 provenance does not point to the S1 sample")
            for arm, s, f in (("sim-attack", one, a["file"]), ("sim-reroll", two, Path(r["destination"]).name)):
                s = resolve_attachments(s)
                if refs_left(s["messages"]):
                    raise ValueError(f"{r['cell']} {arm}: attachment reference left")
                out.append({"cell": r["cell"], "environment": r["environment"], "main_task": a["main_task"], "arm": arm,
                            "file": f, "sample_id": s["id"], "epoch": s.get("epoch", 1), "id": s["id"], "messages": s["messages"]})
                counts["S1" if arm == "sim-attack" else "S2"] += 1
        data = gzip.compress("".join(json.dumps(x) + "\n" for x in out).encode(), mtime=0)
        local = tmp / "samples.jsonl.gz"
        local.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "--with", "modal>=1.4.3", "modal", "volume", "put",
                        VOLUME, str(local), DEST], check=True, capture_output=True)
    res = {"label": "I3 stage 1: reroll and first-sim samples uploaded (counts and hashes only)", "dest": f"{VOLUME}:{DEST}",
           "sha256": digest, "bytes": len(data), "counts": counts, "cells": len(rows),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"reroll_upload_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
