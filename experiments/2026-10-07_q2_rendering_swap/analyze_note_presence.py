"""Which text_editor outputs carry the real tool's review note ("Review the changes ... Edit the file again if
necessary."), by command, in real e1, the September simulator, and today's simulator (the 20-task pilot, sims
e561f936). Saved transcripts only; no GPU.

Sources:
- real e1 and Sept sim: every Q2 run (939 each), as the reader saw them (examples.load_runs);
- the same two restricted to the pilot's 20 attack cells (matched to today's attack pilot);
- today's attack and honest pilots: the completed attempt per task (the loader of
  ../2026-10-06_astra_sim_honest/analyze_tool_outputs.py), valid samples only.
Counts per command: outputs, outputs with the note, every distinct note line (the line holding "Review the" or
"Edit the file again") with its count, and each output's format (`cat -n` snippet, diff, error, other) with and
without the note. For the str_replace offset (shift/str_replace_levels.py), two format tells per command: the share of
outputs ending in a newline, and the share of diffs whose hunk lines all carry the real tool's `N|` gutter (the rule of
../2026-10-07_sim_failure_hiding/analyze_editor_templates.py). Writes results/note_presence_<stamp>.json.

  uv run --with pandas python experiments/2026-10-07_q2_rendering_swap/analyze_note_presence.py
"""
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.transcripts import invalidity, tool_result_text

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "2026-10-06_q2_impactful_call_transfer" / "shift"))
import examples as ex  # noqa: E402

PHASE2 = HERE.parent / "2026-10-06_astra_sim_honest"  # the pilot's loader and task list, from this checkout
# Local mirror of the pilot's two S3 run folders (src/probe_shift/cohorts.py: la_public10_sim_astra_*_simcode_e561f936_n20)
PILOT = Path.home() / "probe-shift-sims/experiments/2026-10-06_astra_sim_honest/data/today_pilot"
TASKS = PHASE2 / "results/today_pilot_tasks_20261006-175715.json"
NOTE_LINE = re.compile(r"[^\n]*(?:Review the|Edit the file again)[^\n]*")
ERROR = re.compile(r"Invalid `insert_line`|Invalid `view_range`|does not exist|No replacement was performed|Ran into|[Ee]rror"
                   r"|not found|Permission denied|Traceback|Invalid parameters|File already exists")
_spec = importlib.util.spec_from_file_location("tool_outputs", PHASE2 / "analyze_tool_outputs.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def count(pairs):
    """pairs: (command, output) for every text_editor call -> per command n, n with the note, note lines."""
    n, hit, lines, kinds, tells = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for cmd, out in pairs:
        cmd, out = cmd or "none", out or ""
        n[cmd] += 1
        found = NOTE_LINE.findall(out)
        hit[cmd] += bool(found)
        for line in found:
            lines[cmd][line.strip()] += 1
        kind = ("diff" if "Here's a diff" in out else "cat -n snippet" if "Here's the result of running `cat -n`" in out
                else "error" if ERROR.search(out[:400]) else "other")
        kinds[cmd][f"{kind}, {'note' if found else 'no note'}"] += 1
        tells[cmd]["ends_with_newline"] += out.endswith("\n")
        if kind == "diff":
            hunk = [l for l in (out.split("@@", 1)[1].splitlines()[1:] if "@@" in out else []) if l.strip() and not l.startswith("@@")]
            tells[cmd]["diffs"] += 1
            tells[cmd]["diffs_all_numbered"] += bool(hunk) and all(T.NUMBERED.match(l) for l in hunk)
    return {c: {"n": n[c], "with_note": hit[c], "share": round(hit[c] / n[c], 4), "note_lines": dict(lines[c].most_common()),
                "format": dict(kinds[c].most_common()), "ends_with_newline": round(tells[c]["ends_with_newline"] / n[c], 4),
                "diffs": tells[c]["diffs"], "diffs_all_numbered": round(tells[c]["diffs_all_numbered"] / tells[c]["diffs"], 4) if tells[c]["diffs"] else None}
            for c in sorted(n, key=lambda c: -n[c])}


def from_runs(runs):
    return [(ex.editor_args(c)[0], c["output"]) for calls in runs.values() for c in calls.values() if c["tool"] == "text_editor"]


def from_pilot(folder, mode, wanted):
    pairs, kept = [], []
    for s in T.samples(folder):
        if isinstance(s, tuple) or invalidity(s) is not None:
            continue
        main, side = T.main_and_side(s)
        if (main if mode == "honest" else f"{main}:{side}") not in wanted:
            continue
        kept.append([s["_log"], s["id"], s["epoch"]])
        calls = {tc["id"]: tc for m in s["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
        for m in s["messages"]:
            if m["role"] == "tool" and m.get("function") == "text_editor":
                args = (calls.get(m.get("tool_call_id")) or {}).get("arguments") or {}
                pairs.append((args.get("command"), tool_result_text(m)))
    return pairs, kept


q = ex.table("qwen3-32b")
tasks = json.loads(TASKS.read_text())["tasks"]
cells, mains = {t["cell"] for t in tasks}, {t["main"] for t in tasks}
runs = ex.load_runs(set(q.key))
dom = dict(zip(q.key, q.domain))
cell_of = dict(zip(q.key, q.cell))
out = {"sources": {}}
for d, label in (("real", "real e1 (all 939 runs)"), ("sim", "Sept sim attack (all 939 runs)")):
    sel = {k: v for k, v in runs.items() if dom[k] == d}
    out["sources"][label] = {"runs": len(sel), **count(from_runs(sel))}
    sel20 = {k: v for k, v in sel.items() if cell_of[k] in cells}
    out["sources"][label.split(" (")[0] + " (pilot's 20 cells)"] = {"runs": len(sel20), "cells": len({cell_of[k] for k in sel20}),
                                                                   **count(from_runs(sel20))}
for mode, wanted in (("attack", cells), ("honest", mains)):
    pairs, kept = from_pilot(PILOT / mode, mode, wanted)
    out["sources"][f"today's sim {mode} pilot (e561f936)"] = {"runs": len(kept), "measured": sorted(kept), **count(pairs)}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
(HERE / "results" / f"note_presence_{stamp}.json").write_text(json.dumps(out, indent=1))
for label, r in out["sources"].items():
    print(f"\n## {label}: {r['runs']} runs")
    for c, v in r.items():
        if isinstance(v, dict):
            print(f"  {c:12} {v['with_note']:5}/{v['n']:5} = {v['share']:.1%}  " + " | ".join(f"{k!r} x{m}" for k, m in list(v["note_lines"].items())[:4]))
            print(f"  {'':12} format: {v['format']} | ends with newline {v['ends_with_newline']:.1%} | diffs {v['diffs']}, all numbered {v['diffs_all_numbered']}")
print(f"wrote results/note_presence_{stamp}.json")
