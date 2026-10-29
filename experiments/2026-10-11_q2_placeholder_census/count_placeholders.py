"""Placeholder census, q2's pipelines (Aswin's top priority before Monday, relayed by the orchestrator): unresolved
"attachment://" references in each history source and index q2's pipelines read. Counts only; no text is printed or
written. Honest data only: attack-run logs are not opened (Aswin's rule against new passes over attack-run text).

Counted:
- every log of Sol's Phase 1 real honest (n1000) and simulated honest (n437) cohorts: sample-message contents and tool-call
  arguments (what `probe_shift.transcripts.reader_messages` and the label builders read), and model-event inputs;
- the Qwen 196-task scale-up's honest A and honest B logs, same fields, plus the last model event's call records
  (`events_data.calls` via `call_refs`, what E1's build_e1_index.py read);
- the replay indexes q2 built: E1 stage 1, the real-vs-simulated pilot, the fresh confirmation.

  uv run --no-project --python 3.14 python experiments/2026-10-11_q2_placeholder_census/count_placeholders.py
"""
import glob
import hashlib
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
C = Path(os.path.expanduser("~/.cache/probe-shift/s3"))
LOGS = {"Sol Phase 1 real honest n1000": C / "rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000/logs",
        "Sol Phase 1 simulated honest n437 (Astra)": C / "rr-sce-sim-transcripts/transcripts/2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0/logs"}
QWEN = C / "individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"
INDEXES = {"E1 stage 1 index (build_e1_index.py)": Path.home() / "probe-shift-q2-margins/data/e1_honest/index_2026-10-11_120853.jsonl",
           "E1 real-vs-sim pilot index (build_realsim_index.py)": Path("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad/wt_rs/data/e1_realsim/index_2026-10-11_162843.jsonl"),
           "E1 fresh confirmation index (build_confirm_index.py)": Path("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad/wt_conf/data/e1_confirm/index_2026-10-11_164154.jsonl")}


def count_sample(s: dict, c: Counter, call_records: bool):
    c["samples"] += 1
    for m in s.get("messages") or []:
        content = m.get("content")
        blob = content if isinstance(content, str) else json.dumps([x for x in (content or []) if x.get("type") != "reasoning"])
        c["sample-message contents"] += 1; c["refs in sample-message contents"] += len(REF.findall(blob))
        for tc in m.get("tool_calls") or []:
            c["tool-call arguments"] += 1; c["refs in tool-call arguments"] += len(REF.findall(json.dumps(tc.get("arguments"))))
    for e in (e for e in s.get("events") or [] if e.get("event") == "model"):
        c["refs in model-event inputs"] += len(REF.findall(json.dumps(e.get("input") or [])))
    if call_records:
        ev = [e for e in s.get("events") or [] if e.get("event") == "model"]
        call = (ev[-1].get("call") or {}) if ev else {}
        if call.get("call_refs"):
            pool = (s.get("events_data") or {}).get("calls") or []
            rec = [pool[i] for lo, hi in call["call_refs"] for i in range(lo, hi)]
            c["call-record messages (last model call)"] += len(rec); c["refs in call records (last model call)"] += len(REF.findall(json.dumps(rec)))


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    out = {"git": head, "sources": {}, "indexes": {}}
    for name, d in LOGS.items():
        c = Counter()
        for f in sorted(d.glob("*.eval")):
            with zipfile.ZipFile(f) as z:
                for n in (n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")):
                    count_sample(json.loads(z.read(n)), c, call_records=False)
        out["sources"][name] = dict(c) | {"logs": len(list(d.glob("*.eval")))}
    for arm in ("honest_a", "honest_b"):
        c = Counter(); files = sorted(glob.glob(str(QWEN / f"*/{arm}/**/*.eval"), recursive=True))
        for f in files:
            with zipfile.ZipFile(f) as z:
                for n in (n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")):
                    count_sample(json.loads(z.read(n)), c, call_records=True)
        out["sources"][f"Qwen 196-task scale-up {arm} (all batches)"] = dict(c) | {"logs": len(files)}
    for name, p in INDEXES.items():
        c = Counter()
        for line in p.open():
            r = json.loads(line); c["runs"] += 1
            for m in r["messages"]:
                content = m.get("content")
                c["message contents"] += 1; c["refs in message contents"] += len(REF.findall(content if isinstance(content, str) else json.dumps(content)))
                for tc in m.get("tool_calls") or []:
                    c["tool-call arguments"] += 1; c["refs in tool-call arguments"] += len(REF.findall(json.dumps(tc["function"]["arguments"])))
            c["refs in tools"] += len(REF.findall(json.dumps(r.get("tools"))))
        out["indexes"][name] = dict(c) | {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"placeholders_q2_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main()
