"""Attachment census, the raw sources: do the stored Inspect logs' sample MESSAGES (what probe_shift.transcripts reads,
without resolving attachments) hold `attachment://` placeholders? Covers the real attack epochs Q1's extraction read
(whose real e1/e2 rows step A's packs reuse), the repaired-v4 epochs, and the September Astra simulated attack and
honest batches. 10 seeded logs per prefix (random.Random(20261011)); per prefix: samples, samples with any placeholder in
messages, placeholders in messages, placeholders in events (for contrast), samples with an attachments map. Counts only;
$0 (S3 reads from the laptop; downloads in a temporary folder, deleted).

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_attachment_census/analyze_source_logs.py
"""

import hashlib
import json
import random
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
BUCKET, T, NEEDLE = "rr-sce-sim-transcripts", "transcripts/", "attachment://"
PREFIXES = ["2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch1-n1000",
            "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch2-n1000",
            "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-repaired-v4-epoch1-n1000",
            "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-repaired-v4-epoch2-n1000",
            "2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000",
            "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0"]


def aws(*a):
    return subprocess.run(["aws", "--profile", "redwood-sandbox", *a], capture_output=True, text=True, check=True).stdout


def main():
    out = {"label": "attachment census of source Inspect logs (sample messages vs events; counts only)", "prefixes": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for p in PREFIXES:
            try:  # a prefix with no logs/ folder lists nothing (aws exits 1): try the prefix root
                listing = aws("s3", "ls", "--recursive", f"s3://{BUCKET}/{T}{p}/logs/")
            except subprocess.CalledProcessError:
                listing = aws("s3", "ls", "--recursive", f"s3://{BUCKET}/{T}{p}/")
            keys = sorted(l.split()[-1] for l in listing.splitlines() if l.endswith(".eval"))
            if not keys:
                out["prefixes"][p] = {"logs": 0}
                continue
            pick = random.Random(20261011).sample(keys, min(10, len(keys)))
            c = {"logs_in_prefix": len(keys), "logs_sampled": len(pick), "samples": 0, "samples_with_msg_placeholder": 0, "msg_placeholders": 0,
                 "event_placeholders": 0, "samples_with_attachments_map": 0}
            for i, k in enumerate(pick):
                f = Path(tmp) / f"{i}.eval"
                aws("s3", "cp", "--quiet", f"s3://{BUCKET}/{k}", str(f))
                with zipfile.ZipFile(f) as z:
                    for n in z.namelist():
                        if n.startswith("samples/") and n.endswith(".json"):
                            s = json.loads(z.read(n))
                            m = json.dumps(s.get("messages", [])).count(NEEDLE)
                            c["samples"] += 1
                            c["msg_placeholders"] += m
                            c["samples_with_msg_placeholder"] += m > 0
                            c["event_placeholders"] += json.dumps(s.get("events", [])).count(NEEDLE)
                            c["samples_with_attachments_map"] += bool(s.get("attachments"))
                f.unlink()
            c["log_list_sha256"] = hashlib.sha256("\n".join(pick).encode()).hexdigest()
            out["prefixes"][p] = c
            print(p[:90], {k: v for k, v in c.items() if k != "log_list_sha256"})
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"source_log_census_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
