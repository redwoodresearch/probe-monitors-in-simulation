"""Print the simulator's final completion behind the empty replies of the 4 simulated long-streak runs (POST HOC). $0.

Aswin, 2026-10-13, relayed by the orchestrator: "Yes, print them". Prints only the simulator's final ('stop')
completion behind each empty reply in t118_clinical_trial, t66/t86/t96_file_indexing: one line per distinct string per
run (only the first 18 characters if longer, and says so), whether it is identical across calls and runs, and whether it
parses as JSON. Never prints the agent's commands or any other reply text. For the non-empty replies it only counts, by
script, how often the reply equals a string field of the parsed completion (field names only).

PREDICTION (committed before running): one identical string in all 4 runs; it parses as JSON with an empty output
field; non-empty replies equal that same field in >= 95% of calls, so it is an empty-output convention, not a parsing loss.
AMENDED after the first run printed '<OUTPUT>\n</OUTPUT>' (not JSON): also counts non-empty replies that equal the text
between their completion's <OUTPUT> tags. Prediction for that: >= 95% equal.

  uv run --no-project --python 3.14 python -I print_sim_empty_completion_POSTHOC.py <id sheet csv> <bundle_src dir>
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_empty_streak_runs_POSTHOC import REF, WIDTH_DROPPED, load  # noqa: E402

TAG = __import__("re").compile(r"<OUTPUT>\n?(.*?)\n?</OUTPUT>", __import__("re").S)


def text(c, att):
    if isinstance(c, str):
        m = REF.fullmatch(c)
        return text(att[m.group(1)], att) if m else c
    if isinstance(c, list):
        return "".join(text(p.get("text", ""), att) if isinstance(p, dict) else text(p, att) for p in c)
    return ""


def main(id_sheet: str, evals: str):
    files = {r["task"]: r["run_key"].rsplit("#e", 1)[0] for r in csv.DictReader(open(id_sheet, newline="")) if r["arm"] == "sim_attack"}
    every = Counter()
    for task in sorted(WIDTH_DROPPED):
        s = load(Path(evals), files[task])["sample"]
        att, ev = s.get("attachments") or {}, s["events"]
        span_of, claimed = {}, set()
        for i, e in enumerate(ev):
            if e.get("event") == "tool":
                for b in reversed(ev[max(0, i - 50):i]):
                    if (b.get("event") == "span_begin" and b.get("type") == "tool" and b["id"] not in claimed
                            and b.get("parent_id") == e.get("span_id") and b.get("name") == e.get("function")):
                        span_of[e["id"]] = b["id"]
                        claimed.add(b["id"])
                        break
        final = {}  # tool span -> text of its last 'stop' model event
        for e in ev:
            o = e.get("output") or {}
            if e.get("event") == "model" and (o.get("choices") or [{}])[0].get("stop_reason") == "stop":
                final[e.get("span_id")] = text(o.get("completion"), att)
        empty, fields, n_nonempty, n_json, tag = Counter(), Counter(), 0, 0, Counter()
        for m in (m for m in s["messages"] if m.get("role") == "tool"):
            comp = final.get(span_of.get(m.get("tool_call_id")))
            if comp is None:
                continue
            reply = text(m.get("content"), att)
            if not reply.strip():
                empty[comp] += 1
                continue
            n_nonempty += 1
            # counts only: does the reply equal the text between the completion's <OUTPUT> tags (outer newlines dropped)?
            inner = TAG.fullmatch(comp.strip())
            tag["one <OUTPUT> block" if inner else "no single <OUTPUT> block"] += 1
            if inner:
                tag["reply == inner text"] += reply == inner.group(1)
                tag["reply == inner text, ends stripped"] += reply.strip() == inner.group(1).strip()
            try:
                d = json.loads(comp)
            except ValueError:
                continue
            n_json += 1
            for k, v in (d.items() if isinstance(d, dict) else []):
                fields[k] += isinstance(v, str) and v == reply
        every.update(empty)
        print(f"== {task}: {sum(empty.values())} empty replies with a final simulator completion; {len(empty)} distinct")
        for c, n in empty.most_common():
            print(f"   {n:4d} x {c[:18]!r}" + (f"  (first 18 of {len(c)} characters)" if len(c) > 18 else f"  ({len(c)} characters)"))
        print(f"   non-empty replies: {n_nonempty}; completion parses as JSON: {n_json}; reply equals the parsed string field: {dict(fields)}; <OUTPUT> tag check: {dict(tag)}")
    print(f"\nidentical across all calls and runs: {len(every) == 1} ({len(every)} distinct over {sum(every.values())} calls)")
    for c in every:
        try:
            d = json.loads(c)
            print(f"parses as JSON: {type(d).__name__}; " + (", ".join(f"{k}: {type(v).__name__} of length {len(v) if hasattr(v, '__len__') else '-'}" for k, v in d.items()) if isinstance(d, dict) else ""))
        except ValueError as err:
            print(f"does not parse as JSON ({type(err).__name__})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
