"""POST HOC, exploratory (C): what did honest A and honest B do on t72 and t92 clinical_trial? Honest runs only. $0, local.

Reads the four honest transcripts (launch 2: honest_a takes the first counted attempt, honest_b the last, as the judge
and scorer do) and reports numbers plus short normalized signatures, never long quotes:
- the main-task score and whether the run stopped on a limit (message or token limit, per the sample);
- per bash call: the command's shape (first five tokens, digits as N) and whether its output is an error (the tool
  message's error field, or a nonzero exit code / an error marker in the output);
- over the run's tenths: calls, error share, distinct commands, and the share repeating the previous command's shape;
- for the Rscript one-liners: how many distinct commands, the longest run of consecutive erroring calls among the Rscript
  calls only (other calls in between do not break it), and the five
  most common error signatures (the first error line, digits and quoted text normalized, at most 60 characters).

Prediction, written before running: A retried near-identical failing `Rscript -e` one-liners (high error share, low
command diversity), while B wrote or reused a script and finished quickly; both may still pass the main task.

  uv run --no-project --python 3.14 python analyze_honest_runs_POSTHOC_t72_t92.py <launch dir 2026-10-09_182505_scaleup>
"""

import argparse
import collections
import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
TASKS = ("t72_clinical_trial", "t92_clinical_trial")
ERR = re.compile(r"(^|\n)\s*(Error|error:|Traceback|Execution halted|command not found|No such file|cannot open|exit code [1-9]|Exit code: [1-9])", re.I)


def shape(cmd: str) -> str:
    return " ".join(re.sub(r"\d+", "N", t)[:24] for t in cmd.split()[:5])


def signature(text: str) -> str:
    line = next((l for l in text.splitlines() if ERR.search("\n" + l)), text.splitlines()[0] if text.splitlines() else "")
    line = re.sub(r"\d+", "N", re.sub(r"(['\"]).*?\1", "'…'", line.strip()))
    return line[:60]


def content(m) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))


def run_summary(sample: dict) -> dict:
    msgs = sample["messages"]
    out_of = {m.get("tool_call_id"): m for m in msgs if m["role"] == "tool"}
    calls = []
    for m in msgs:
        for tc in ((m.get("tool_calls") or []) if m["role"] == "assistant" else []):
            args = tc.get("arguments") or {}
            args = json.loads(args) if isinstance(args, str) else args
            res = out_of.get(tc.get("id"), {})
            text = content(res)
            err = bool(res.get("error")) or bool(ERR.search(text))
            cmd = (args.get("cmd") or args.get("command") or "") if isinstance(args, dict) else ""
            calls.append({"fn": tc.get("function"), "shape": shape(cmd) if tc.get("function") == "bash" else str(tc.get("function")),
                          "cmd_sha": hashlib.sha256(cmd.encode()).hexdigest()[:12], "err": err, "sig": signature(text) if err else None,
                          "rscript": tc.get("function") == "bash" and "Rscript -e" in cmd})
    n = len(calls)
    same = [False] + [calls[i]["shape"] == calls[i - 1]["shape"] for i in range(1, n)]  # each call against its predecessor
    tenths = []
    for d in range(10):
        lo, hi = d * n // 10, (d + 1) * n // 10
        part = calls[lo:hi]
        comparable = range(max(lo, 1), hi)  # the run's first call has no predecessor
        tenths.append({"calls": len(part), "error_share": round(sum(c["err"] for c in part) / len(part), 3) if part else None,
                       "distinct_commands": len({c["cmd_sha"] for c in part}),
                       "same_shape_as_previous": round(sum(same[i] for i in comparable) / len(comparable), 3) if len(comparable) else None})
    r = [c for c in calls if c["rscript"]]
    longest = cur = 0
    for c in r:
        cur = cur + 1 if c["err"] else 0
        longest = max(longest, cur)
    scores = sample.get("scores") or {}
    return {"tool_calls": n, "assistant_turns": sum(m["role"] == "assistant" for m in msgs),
            "main_task": {k: (v.get("value") if isinstance(v, dict) else v) for k, v in scores.items()},
            "limit": sample.get("limit"), "error": bool(sample.get("error")),
            "tools": dict(collections.Counter(c["fn"] for c in calls)),
            "error_share_all_calls": round(sum(c["err"] for c in calls) / n, 3) if n else None,
            "rscript_inline": {"calls": len(r), "distinct_commands": len({c["cmd_sha"] for c in r}), "erroring": sum(c["err"] for c in r),
                               "longest_erroring_streak": longest, "top_error_signatures": collections.Counter(c["sig"] for c in r if c["err"]).most_common(5)},
            "by_tenth": tenths}


def main(launch: str):
    run_dir = Path(launch)
    L = json.loads((run_dir / "launch.json").read_text())
    out = {"label": "POST HOC, exploratory (C); honest runs only; counts and short normalized signatures",
           "prediction (written before running)": "A retried near-identical failing Rscript -e one-liners (high error share, low command diversity); B wrote or reused a script and finished quickly; both may pass",
           "runs": {}}
    for r in L["runs"]:
        if r["arm"] not in ("honest_a", "honest_b") or r["task"] not in TASKS:
            continue
        counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
        a = counted[0] if r["arm"] == "honest_a" else counted[-1]
        (ev,) = sorted((run_dir / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])).glob("*.eval"))
        with zipfile.ZipFile(ev) as z:
            (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
            s = json.loads(z.read(n))
        out["runs"][f"{r['arm']}|{r['task']}"] = {"process_seconds": a.get("process_seconds"), **run_summary(s)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"honest_runs_POSTHOC_t72_t92_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:6000])
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("launch")
    main(ap.parse_args().launch)
