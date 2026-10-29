"""E1 index check (Codex thread on #701; local, $0): is any selected target edit a text-encoded failure, a text_editor
reply with no error field whose text reports a failure? Applies the failure-hiding study's own predicate
(experiments/2026-10-07_sim_failure_hiding/analyze_failure_hiding.py, `failure()`: the error field or TE_FAIL) to each
indexed run's selected edit and to every edit the eligibility rule accepted. Keys and counts only.

  PYTHONPATH=src:experiments/2026-10-07_sim_failure_hiding uv run --no-project --python 3.14 --with numpy --with pandas \
      --with boto3 --with pyyaml python experiments/2026-10-11_q2_e1_honest/check_editor_failures.py data/e1_honest/index_2026-10-11_120853.jsonl
"""
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import analyze_failure_hiding as fh

HERE = Path(__file__).parent
CACHE = Path.home() / ".cache/probe-shift/s3/individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"


def main(index: Path):
    rows, bad, accepted_failures = [json.loads(line) for line in index.open()], [], 0
    for r in rows:
        with zipfile.ZipFile(next(CACHE.glob(f"*/{r['run_id']}"))) as z:
            s = json.loads(z.read([n for n in z.namelist() if n.startswith("samples/")][0]))
        in_req = {m.get("tool_call_id") for m in r["messages"] if m["role"] == "tool"}
        replies = {m.get("tool_call_id"): m for m in s["messages"] if m["role"] == "tool"}
        ok = [tc for m in s["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []
              if tc["function"] == "text_editor" and tc["arguments"].get("command") in ("create", "str_replace", "insert")
              and tc["arguments"].get("path") and tc["id"] in in_req and tc["id"] in replies and replies[tc["id"]].get("error") is None]
        if ok[-1]["arguments"]["path"] != r["path"]:
            raise ValueError(f"{r['key']}: the rebuilt selection differs from the index")
        accepted_failures += sum(fh.failure("text_editor", replies[tc["id"]]) is not None for tc in ok)
        if why := fh.failure("text_editor", replies[ok[-1]["id"]]):
            bad.append({"part": r["part"], "key": r["key"], "why": why})
    out = {"index": str(index), "index sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "selected targets checked": len(rows),
           "text-encoded failures among selected targets": len(bad), "runs": bad,
           "text-encoded failures among all accepted edits": accepted_failures}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"editor_failures_{stamp}.json", "x") as fh_out:
        fh_out.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps(out)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
