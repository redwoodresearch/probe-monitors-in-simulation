"""POST HOC corrections (RULE_POSTHOC_feedback_turns.md, committed at 43a529c8 before counting): the made-up-files
feedback counts by model turn instead of by call, the 17 HTTP 400s' cause from vLLM's error bodies, and rates with one
denominator each. $0, honest runs only, counts only.

  uv run --no-project --python 3.14 --with tokenizers --with huggingface_hub python -I analyze_POSTHOC_feedback_turns.py \\
      --data <10-09 dir> --pilot <10-06 dir> --result results/tool_outputs_2026-10-12_100453.json
"""

import argparse
import glob
import gzip
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
import check_tool_outputs as C  # noqa: E402
from analyze_POSTHOC_relative_paths import messages_for  # noqa: E402

HERE = Path(__file__).parent
GATE = {"1009": {"editor_reads_or_edits": 824, "editor_missing_path_error": 577, "editor_on_missing_path_first_time": 141,
                 "editor_on_path_told_missing_before": 489, "editor_on_path_told_missing_before_and_still_missing": 436,
                 "editor_on_path_told_missing_by_previous_output": 342},
        "1006": {"editor_reads_or_edits": 372, "editor_missing_path_error": 13, "editor_on_missing_path_first_time": 10,
                 "editor_on_path_told_missing_before": 3, "editor_on_path_told_missing_by_previous_output": 3}}
EXPECTED_RUNS = {"1009": 51, "1006": 10}
CONTEXT = re.compile(r"maximum context length is (\d+) tokens.*?requested (\d+) output tokens.*?contains at least (\d+) input tokens", re.S)


def by_turn(msgs: list[dict], c: Counter, att: dict):
    """As behaviour(), but feedback counts only from earlier assistant messages (model turns)."""
    told: set[str] = set()
    prev_turn: set[str] = set()
    results = {m.get("tool_call_id"): m for m in msgs if m["role"] == "tool"}

    def same(mp: str, p: str) -> bool:
        return mp == p or (not p.startswith("/") and mp.endswith("/" + p.removeprefix("./")))

    for m in msgs:
        if m["role"] != "assistant":
            continue
        calls = m.get("tool_calls") or []
        c["assistant_messages_with_tool_calls"] += bool(calls)
        c["assistant_messages_with_more_than_one_tool_call"] += len(calls) > 1
        this_turn: set[str] = set()
        for tc in calls:
            a = tc.get("arguments") or {}
            r = results.get(tc.get("id"))
            out = C.as_sent(r, att) if r else ""
            if tc.get("function") == "text_editor":
                cmd, p = a.get("command"), a.get("path")
                if cmd in ("view", "str_replace", "insert") and p:
                    c["editor_reads_or_edits"] += 1
                    hit = any(same(mp, p) for mp in C.missing_paths(out))
                    c["editor_missing_path_error"] += hit
                    c["editor_on_path_told_missing_by_previous_turn"] += any(same(mp, p) for mp in prev_turn)
                    if any(same(mp, p) for mp in told):
                        c["editor_on_path_told_missing_in_an_earlier_turn"] += 1
                        c["editor_on_path_told_missing_in_an_earlier_turn_and_still_missing"] += hit
                    elif hit:
                        c["editor_on_missing_path_first_time_by_turn"] += 1
            this_turn |= C.missing_paths(out)
        told |= this_turn
        prev_turn = this_turn


def main(data: str, pilot: str, result: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = {"git": git, "rule": "RULE_POSTHOC_feedback_turns.md @ 43a529c8"}
    for label, root, launches in (("1009", Path(data), C.RUNS_1009), ("1006", Path(pilot), C.PILOT_1006)):
        old, new, n = Counter(), Counter(), 0
        for launch in launches:
            for ev in sorted(p for p in (root / launch).rglob("*.eval") if "references" not in p.parts):
                s = C.load_sample(ev)
                att = s.get("attachments") or {}
                msgs = s["messages"] if label == "1009" else messages_for(s)  # as in section 4
                C.behaviour(msgs, old, att)
                by_turn(msgs, new, att)
                n += 1
        if n != EXPECTED_RUNS[label]:
            raise SystemExit(f"{label}: {n} runs, expected {EXPECTED_RUNS[label]}")
        bad = {k: (old[k], v) for k, v in GATE[label].items() if old[k] != v}
        if bad:
            raise SystemExit(f"{label}: the as-run counts are not reproduced: {bad}")
        out[label] = {"runs": n, "as_run_by_call": {k: old[k] for k in GATE[label]}, "by_turn": dict(new)}
        print(label, "runs", n, "| by call (as run):", out[label]["as_run_by_call"])
        print(label, "by turn:", dict(sorted(new.items())))
    rej = []
    for f in sorted(glob.glob(str(Path(data) / "*.request_logs" / "requests_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("status") == 400:
                    m = CONTEXT.search(r.get("error_body") or "")
                    rej.append(tuple(int(x) for x in m.groups()) if m else None)
    out["http_400"] = {"records": len(rej), "with_context_message": sum(x is not None for x in rej),
                       "window_output_prompt_at_least": dict(Counter(f"{w}/{o}/{p}" for w, o, p in (x for x in rej if x)))}
    print("HTTP 400:", out["http_400"])
    T = Counter(json.loads(Path(result).read_text())["totals_1009"])
    carried = T["outputs_found_byte_exact"] + T["outputs_rejected_400"] + T["outputs_rejected_cancelled"] + T["outputs_no_record"]
    out["rates"] = {"outputs_carried_in_a_request": carried,
                    "empty": [T["outputs_delivered_empty"], carried, T["outputs_delivered_empty"] / carried],
                    "truncation_notice": [T["outputs_delivered_with_inspect_truncation_notice"], carried,
                                          T["outputs_delivered_with_inspect_truncation_notice"] / carried],
                    "tool_errors": [T["tool_errors"], T["tool_outputs"], T["tool_errors"] / T["tool_outputs"]]}
    print("rates:", json.dumps(out["rates"]))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"feedback_turns_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--result", required=True)
    main(**vars(ap.parse_args()))
