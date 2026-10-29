"""Attachment census (Aswin's top priority, relayed by 0d about 16:55 PT 10-11): do the staged run histories that claim
B's pipelines read contain unresolved Inspect `attachment://` placeholders? Inspect stores long strings as attachment
references; a builder that reads sample JSON without swapping the text back in leaves the placeholder in place.

One Modal CPU container reads the four claim-B staged input sets (the inputs every claim-B extraction, pack, step A,
step B, H2, B-dev, H3's evened reads, H1, the halves test and the AI judge's rendered transcripts were built from) and
counts, per set and arm: runs, runs with any placeholder, and placeholders by message role and field (content strings,
tool-call arguments, other fields). Counts only; no text leaves the container. Under the standing under-$5 rule.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \\
      experiments/2026-10-11_claimB_attachment_census/run_census_modal.py
"""

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SETS = ["2026-10-10_la_fixed-sim-v2_probe-inputs-l0_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1",
        "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l0_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"]
NEEDLE = "attachment://"
app = modal.App("probe-shift_aswin-surya_sub-claimb-census_kill-2026-11-11")
image = modal.Image.debian_slim(python_version="3.12")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": vol}, timeout=30 * 60, retries=0)
def count() -> dict:
    out = {}
    for s in SETS:
        files = sorted(Path(f"/acts/{s}/inputs").glob("shard_*.jsonl"))
        if not files:
            raise ValueError(f"{s}: no input shards")
        per = defaultdict(lambda: defaultdict(int))
        for f in files:
            raw = f.read_text()
            per["_set"]["raw_occurrences_in_files"] += raw.count(NEEDLE)
            for line in raw.splitlines():
                r = json.loads(line)
                a = per[r["arm"]]
                a["runs"] += 1
                n_run = 0
                for m in r["messages"]:
                    role = m.get("role", "?")
                    for k, v in m.items():
                        if k == "tool_calls":
                            n = json.dumps(v).count(NEEDLE)
                            a["in_tool_call_args"] += n
                        elif isinstance(v, str):
                            n = v.count(NEEDLE)
                            a[f"in_{role}_{k}"] += n
                        else:
                            n = json.dumps(v).count(NEEDLE)
                            a[f"in_{role}_{k}"] += n
                        n_run += n
                other = json.dumps({k: v for k, v in r.items() if k != "messages"}).count(NEEDLE)
                a["in_non_message_fields"] += other
                a["runs_with_any"] += (n_run + other) > 0
                a["placeholders"] += n_run + other
        out[s] = {arm: {k: v for k, v in d.items() if v} | {"runs": d.get("runs", 0)} for arm, d in per.items()}
        out[s]["_files"] = len(files)
    return out


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    res = {"label": "attachment census of claim B's staged inputs (counts only)", "needle": NEEDLE, "sets": count.remote(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"attachment_census_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    for s, d in res["sets"].items():
        print(s, {arm: (v.get("runs"), v.get("runs_with_any", 0), v.get("placeholders", 0)) for arm, v in d.items() if arm != "_files"})
    print("->", dst)
