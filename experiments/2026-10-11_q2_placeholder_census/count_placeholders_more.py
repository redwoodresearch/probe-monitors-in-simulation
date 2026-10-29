"""Placeholder census, part 2 (Codex on CENSUS.md): the fields part 1 skipped. Tool error text (`error.message` on tool
messages, which `probe_shift.transcripts.tool_result_text` can show the reader) in every cohort q2 reads, and the full
Sol attack cohorts, which q2's call labeler and off-policy extraction read. Counts only; no text is printed or written.
Counting attack logs this way was OK'd by Aswin just before 16:57 PT ("Yes, counts only is fine").

  nice -n 19 uv run --no-project --python 3.14 python experiments/2026-10-11_q2_placeholder_census/count_placeholders_more.py
"""
import glob
import json
import os
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REF = re.compile(r"attachment://[0-9a-f]+")
T = Path(os.path.expanduser("~/.cache/probe-shift/s3/rr-sce-sim-transcripts/transcripts"))
Q = Path(os.path.expanduser("~/.cache/probe-shift/s3/individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"))
SOURCES = {"Sol real honest n1000": sorted((T / "2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000/logs").glob("*.eval")),
           "Sol simulated honest n437": sorted((T / "2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0/logs").glob("*.eval")),
           "Sol real attack epoch 1 n1000": sorted((T / "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch1-n1000/logs").glob("*.eval")),
           "Sol real attack epoch 2 n1000": sorted((T / "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch2-n1000/logs").glob("*.eval")),
           "Sol simulated attack n1000 (Astra)": sorted((T / "2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000/logs").glob("*.eval")),
           "Qwen scale-up honest A": sorted(map(Path, glob.glob(str(Q / "*/honest_a/**/*.eval"), recursive=True))),
           "Qwen scale-up honest B": sorted(map(Path, glob.glob(str(Q / "*/honest_b/**/*.eval"), recursive=True)))}


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    out = {"git": head, "sources": {}}
    for name, files in SOURCES.items():
        c = Counter({"logs": len(files)})
        for f in files:
            with zipfile.ZipFile(f) as z:
                for n in (n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")):
                    s = json.loads(z.read(n)); c["samples"] += 1
                    for m in s.get("messages") or []:
                        content = m.get("content")
                        blob = content if isinstance(content, str) else json.dumps([x for x in (content or []) if x.get("type") != "reasoning"])
                        c["sample-message contents"] += 1; c["refs in sample-message contents"] += len(REF.findall(blob))
                        for tc in m.get("tool_calls") or []:
                            c["tool-call arguments"] += 1; c["refs in tool-call arguments"] += len(REF.findall(json.dumps(tc.get("arguments"))))
                        err = m.get("error") if m.get("role") == "tool" else None
                        if err:
                            c["tool errors"] += 1; c["refs in tool error text"] += len(REF.findall(json.dumps(err)))
                    for e in (e for e in s.get("events") or [] if e.get("event") == "model"):
                        c["refs in model-event inputs"] += len(REF.findall(json.dumps(e.get("input") or [])))
                    c["refs in sample metadata"] += len(REF.findall(json.dumps(s.get("metadata") or {})))
        out["sources"][name] = dict(c)
        print(name, dict(c), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"placeholders_q2_more_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
