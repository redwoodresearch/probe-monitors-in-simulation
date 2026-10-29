"""Bugbot on #1313: does the 50-event look-back that ties a tool reply to its own tool span ever miss or mis-pair?
(POST HOC, $0, local, metadata only: event types, ids, positions and counts; no text.)

Over every run file the streak analysis read (all six arms), for every tool event: is there a span_begin of type
tool with the same parent and name anywhere earlier (no window)? How many events back is the one the 50-event rule
picks, and is it also the nearest such span with no window? Is any span claimed twice, or are two same-name spans open
at once (the case where pairing could swap)? Also: do all simulator (non-agent) model events inside tool spans sit in a
claimed span?

PREDICTION (committed before running): every tool event's span is found, always at distance 1 (the span_begin
immediately before it); no window misses, no overlaps, every simulator model event sits in a claimed span.

  uv run --no-project --python 3.14 python -I analyze_span_window_check_POSTHOC.py <id sheet csv> <bundle_src dir>
"""

import csv
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_empty_streak_runs_POSTHOC import ARMS, load  # noqa: E402


def check(args):
    evals, key = args
    ev = load(Path(evals), key)["sample"]["events"]
    c, claimed, open_spans, max_open = Counter(), set(), defaultdict(int), 0
    ended = {e.get("id") for e in ev if e.get("event") == "span_end"}
    for i, e in enumerate(ev):
        if e.get("event") == "span_begin" and e.get("type") == "tool":
            open_spans[(e.get("parent_id"), e.get("name"))] += 1
            max_open = max(max_open, open_spans[(e.get("parent_id"), e.get("name"))])
        if e.get("event") == "span_end":
            pass
        if e.get("event") != "tool":
            continue
        c["tool_events"] += 1
        cands = [j for j in range(i - 1, -1, -1) if ev[j].get("event") == "span_begin" and ev[j].get("type") == "tool"
                 and ev[j].get("parent_id") == e.get("span_id") and ev[j].get("name") == e.get("function") and ev[j]["id"] not in claimed]
        win = [j for j in cands if j >= i - 50]
        if not cands:
            c["no_span_anywhere"] += 1
            continue
        if not win:
            c["missed_by_50_window"] += 1
        c[f"distance_{i - cands[0]}"] += 1
        if len(cands) > 1 and cands[1] >= i - 50:
            c["another_unclaimed_same_name_span_in_window"] += 1
        claimed.add(ev[cands[0]]["id"])
        open_spans[(e.get("span_id"), e.get("function"))] -= 1
    sims = [e for e in ev if e.get("event") == "model" and not str(e.get("model", "")).startswith("vllm/")]
    tool_span_ids = {e["id"] for e in ev if e.get("event") == "span_begin" and e.get("type") == "tool"}
    c["simulator_events_in_tool_spans"] = sum(e.get("span_id") in tool_span_ids for e in sims)
    c["simulator_events_in_claimed_spans"] = sum(e.get("span_id") in claimed for e in sims)
    c["tool_spans_unended"] = len(tool_span_ids - ended)
    c["max_same_name_tool_spans_open_before_their_tool_event"] = max_open
    return key, c


def main(id_sheet: str, evals: str):
    files = defaultdict(set)
    for r in csv.DictReader(open(id_sheet, newline="")):
        files[r["arm"]].add(r["run_key"].rsplit("#e", 1)[0])
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(check, [(evals, k) for a in ARMS for k in sorted(files[a])], chunksize=4))
    for arm in ARMS:
        tot = Counter()
        mx = 0
        for k in files[arm]:
            c = res[k]
            mx = max(mx, c.pop("max_same_name_tool_spans_open_before_their_tool_event"))
            tot.update(c)
        print(f"{arm:14s} runs {len(files[arm]):4d} | " + ", ".join(f"{k} {v}" for k, v in sorted(tot.items())) + f" | max same-name tool spans open at once {mx}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
