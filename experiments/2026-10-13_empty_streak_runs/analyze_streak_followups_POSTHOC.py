"""Two review follow-ups for the empty-streak result (POST HOC, $0, local, metadata only).

Codex on #1313 / #1314, 2026-10-13. (1) Provenance: record the --evals root and the sha256 of every run file the
streak analysis read (all six arms of the ID sheet), as a per-arm digest plus the full per-file list. (2) Empty replies
without a final 'stop' completion: in the 4 simulated long-streak runs, the completion printout counted 365 empty
replies but the streak printout counts 367 (t118 35 vs 34, t96 46 vs 45). For every empty reply whose own tool span
has no model event with stop reason 'stop', print its run, reply index, whether it lies in a streak of >= 20, and the
(model, stop reason) of each model event in its span, plus the event count. Counts and enum values only; no text.

PREDICTION (committed before running): exactly 2 such replies (1 in t118, 1 in t96), both outside the >= 20 streaks,
each with a 'content_filter' stop from the simulator and no 'stop' after it.

  uv run --no-project --python 3.14 python -I analyze_streak_followups_POSTHOC.py --id-sheet <csv> --evals <dir> --out <json>
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_empty_streak_runs_POSTHOC import ARMS, LAUNCHES, LONG, SHA, WIDTH_DROPPED, load, streaks, textlen  # noqa: E402


def main(id_sheet: str, evals: str, out: str):
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    if hashlib.sha256(Path(id_sheet).read_bytes()).hexdigest() != SHA["id_sheet"]:
        raise SystemExit("id_sheet sha256 mismatch")
    files = defaultdict(dict)
    for r in csv.DictReader(open(id_sheet, newline="")):
        files[r["arm"]][r["run_key"].rsplit("#e", 1)[0]] = r["task"]
    ev = Path(evals)
    prov = {}
    for arm in ARMS:
        rows = []
        for key, task in sorted(files[arm].items()):
            hits = [ev / L / key for L in LAUNCHES if (ev / L / key).exists()]
            if len(hits) != 1:
                raise SystemExit("run file not found exactly once")
            rows.append({"task": task, "launch": hits[0].relative_to(ev).parts[0], "run_key": key,
                         "sha256": hashlib.sha256(hits[0].read_bytes()).hexdigest()})
        prov[arm] = {"files": len(rows), "digest_of_sorted_sha256": hashlib.sha256("\n".join(sorted(x["sha256"] for x in rows)).encode()).hexdigest(),
                     "per_file": rows}
    missing = []
    for task in sorted(WIDTH_DROPPED):
        key = next(k for k, t in files["sim_attack"].items() if t == task)
        s = load(ev, key)["sample"]
        att, evs = s.get("attachments") or {}, s["events"]
        span_of, claimed = {}, set()
        for i, e in enumerate(evs):
            if e.get("event") == "tool":
                for b in reversed(evs[max(0, i - 50):i]):
                    if (b.get("event") == "span_begin" and b.get("type") == "tool" and b["id"] not in claimed
                            and b.get("parent_id") == e.get("span_id") and b.get("name") == e.get("function")):
                        span_of[e["id"]] = b["id"]
                        claimed.add(b["id"])
                        break
        in_span = defaultdict(list)
        for e in evs:
            if e.get("event") == "model":
                in_span[e.get("span_id")].append((e.get("model"), ((e.get("output") or {}).get("choices") or [{}])[0].get("stop_reason")))
        long_idx = {i for st, n in streaks(s) if n >= LONG for i in range(st, st + n)}
        tools = [m for m in s["messages"] if m.get("role") == "tool"]
        n_empty = 0
        for i, m in enumerate(tools):
            if textlen(m.get("content"), att)[1]:
                continue
            n_empty += 1
            sp = span_of.get(m.get("tool_call_id"))
            evts = in_span.get(sp, [])
            if not any(r == "stop" for _, r in evts):
                missing.append({"task": task, "reply_index": i, "function": m.get("function"), "in_streak_ge_20": i in long_idx,
                                "tool_span_found": sp is not None, "model_events": len(evts), "events": evts})
        print(f"{task}: empty replies {n_empty}; without a final 'stop' completion {sum(x['task'] == task for x in missing)}")
    for x in missing:
        print("  ", json.dumps({k: v for k, v in x.items()}))
    for arm, p in prov.items():
        print(f"provenance {arm:14s} files {p['files']:4d} digest {p['digest_of_sorted_sha256'][:16]}")
    Path(out).open("x").write(json.dumps({"label": "POST HOC follow-ups for the empty-streak result; metadata only",
                                          "git": g("rev-parse", "HEAD").strip(), "evals_root": str(ev.resolve()), "provenance": prov,
                                          "empty_replies_without_stop": missing,
                                          "written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--id-sheet", "--evals", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    main(a.id_sheet, a.evals, a.out)
