"""text_editor result templates by command (insert, str_replace, create): real vs September simulator vs today's.

On the today's-simulator pilot's 20 tasks (honest: its main tasks; attack: its cells), one valid sample per task per
source: from a launcher run folder, the attempt the cohort uses (the loader's latest successful log, as the manifest
picks it); from a flat folder of logs, the first by log, sample id and epoch. For each source, the share of
each command's results that carry: the real tool's `cat -n` snippet preamble, its "Review the changes and make sure
they are as expected" note (and both together), a diff ("Here's a diff"), a numbered `N|` gutter on every diff hunk
line (every non-blank line after the first `@@` header), the create confirmation, or an editor error. Sources are
`label=mode:folder` as in
experiments/2026-10-06_astra_sim_honest/analyze_tool_outputs.py, whose loader (completed attempts, valid samples)
this reuses. Writes results/editor_templates_<stamp>.json.

  uv run python experiments/2026-10-07_sim_failure_hiding/analyze_editor_templates.py --tasks <today_pilot_tasks json> \\
    today_honest=honest:<dir> sept_honest=honest:<dir> real_honest=honest:<dir> today_attack=attack:<dir> ...
"""

import argparse
import importlib.util
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.transcripts import invalidity, tool_result_text

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "tool_outputs", HERE.parent / "2026-10-06_astra_sim_honest" / "analyze_tool_outputs.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
PATTERNS = {
    "cat_n_preamble": re.compile(r"Here's the result of running `cat -n`"),
    "review_note": re.compile(r"Review the changes and make sure they are as expected"),
    "diff": re.compile(r"Here's a diff"),
    "created": re.compile(r"File created successfully at"),
    "error": re.compile(r"No replacement was performed|does not exist\. Please provide a valid path|File already exists"
                        r"|Invalid `insert_line`|Invalid `view_range`|Ran into \[Errno"),
}


def measure(samples: list[dict]) -> dict:
    c = Counter()
    for s in samples:
        calls = {tc["id"]: tc for m in s["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
        for m in s["messages"]:
            if m["role"] != "tool" or m.get("function") != "text_editor":
                continue
            command = ((calls.get(m.get("tool_call_id")) or {}).get("arguments") or {}).get("command")
            if command not in ("insert", "str_replace", "create"):
                continue
            text = tool_result_text(m)
            c[(command, "n")] += 1
            for name, pat in PATTERNS.items():
                c[(command, name)] += bool(pat.search(text))
            c[(command, "cat_n_and_review_note")] += bool(PATTERNS["cat_n_preamble"].search(text)
                                                          and PATTERNS["review_note"].search(text))
            if PATTERNS["diff"].search(text):
                # every non-blank line after the first hunk header, other than further hunk headers, is a hunk line
                body = text.split("@@", 1)[1].splitlines()[1:] if "@@" in text else []
                hunk = [l for l in body if l.strip() and not l.startswith("@@")]
                c[(command, "diff_all_numbered")] += bool(hunk) and all(T.NUMBERED.match(l) for l in hunk)
    out = {}
    for command in ("insert", "str_replace", "create"):
        n = c[(command, "n")]
        out[command] = {"n": n, **{f"share_{k}": round(c[(command, k)] / n, 3) if n else None
                                   for k in (*PATTERNS, "cat_n_and_review_note", "diff_all_numbered")}}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("sources", nargs="+", help="label=mode:folder")
    p.add_argument("--note", default="")
    args = p.parse_args()
    tasks = json.loads(args.tasks.read_text())["tasks"]
    mains, cells = {t["main"] for t in tasks}, {t["cell"] for t in tasks}
    key = lambda s, mode: (lambda ms: ms[0] if mode == "honest" else f"{ms[0]}:{ms[1]}")(T.main_and_side(s))
    out = {}
    for source in args.sources:
        label, rest = source.split("=", 1)
        mode, folder = rest.split(":", 1)
        wanted = mains if mode == "honest" else cells
        valid = sorted((s for s in T.samples(Path(folder)) if not isinstance(s, tuple) and key(s, mode) in wanted
                        and invalidity(s) is None), key=lambda s: (s["_log"], s["id"], s["epoch"]))
        # one sample per task, so a task run several times weighs the same as the others (a launcher folder already
        # yields one attempt per task; a flat folder of logs keeps the first by log, id, epoch)
        kept = list({key(s, mode): s for s in reversed(valid)}.values())
        out[label] = {"mode": mode, "folder": str(Path(folder).resolve()), "samples": len(kept),
                      "tasks": len({key(s, mode) for s in kept}),
                      # exactly what was measured: a local folder mirrors an S3 cohort, so these are its object names
                      "measured": sorted([s["_log"], s["id"], s["epoch"]] for s in kept), **measure(kept)}
        print(label, out[label]["samples"], {c: (v["n"], v["share_cat_n_and_review_note"], v["share_diff"],
                                                 v["share_diff_all_numbered"], v["share_created"])
                                             for c, v in out[label].items() if isinstance(v, dict)})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    path = HERE / "results" / f"editor_templates_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps({"tasks": str(args.tasks), "note": args.note, "sources": out}, indent=1))
    print(path)


if __name__ == "__main__":
    main()
