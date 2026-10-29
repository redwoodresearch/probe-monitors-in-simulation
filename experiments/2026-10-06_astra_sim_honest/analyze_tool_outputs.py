"""Per-tool output format, today's simulator vs September's, on the pilot's 20 tasks.

For each source given as `label=mode:folder` (a run folder of `runs/<main>/*.eval`, or a folder of .eval files;
mode `honest` keeps samples of the pilot's main tasks, `attack` keeps samples of its cells), over every tool result:
  - edit diffs (text_editor results "has been edited. Here's a diff"): share whose hunk lines all carry the real
    tool's numbered column (`  12|- old`), all plain unified-diff lines, or mixed;
  - insert results (text_editor `insert` calls): a `cat -n` snippet, a diff, or something else;
  - stderr placement in bash results: the real bash tool returns `stderr + "\\n" + stdout` (inspect_ai
    tool/_tools/_execute.py), so stderr always comes first. Among results with both stderr-looking lines (the
    patterns below) and other lines: stderr block first, stderr last, or interleaved. A heuristic: it sees only lines
    that look like errors or warnings;
  - tool-error rate: share of tool results that are tool errors (the `error` field: timeouts, parsing, ...), by type.
Only completed runs count: in a run folder, the attempt whose log succeeded with an error-free sample; everywhere, samples
that are valid by `invalidity` (errored, stalled and no_actions samples are excluded whole). All sources of one mode are
then measured on their common task set (main task for honest, cell for attack), and the exclusions are reported.
Writes results/tool_outputs_<stamp>.json and prints a table.

  uv run python experiments/2026-10-06_astra_sim_honest/analyze_tool_outputs.py --tasks results/today_pilot_tasks_*.json \\
    today_honest=honest:<dir> sept_code_honest=honest:<dir> today_attack=attack:<dir> sept_attack=attack:<dir> ...
"""

import argparse
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.transcripts import invalidity, tool_result_text

HERE = Path(__file__).parent
NUMBERED = re.compile(r"^\s*\d+\|[ +\-] ")
PLAIN = re.compile(r"^[ +\-](?![\-+]{2} )")
STDERR = re.compile(r"^(bash: |sh: |/bin/sh: |.*: command not found|.*: No such file or directory|[Ww]arning: |WARNING|"
                    r"[Ee]rror: |ERROR|E: |Traceback \(most recent call last\)|npm (WARN|ERR!)|debconf: |find: |fatal: )")


def read(log: Path) -> tuple[str | None, list[dict]]:
    with zipfile.ZipFile(log) as z:
        names = z.namelist()
        status = json.loads(z.read("header.json")).get("status") if "header.json" in names else None
        samples = [json.loads(z.read(n)) for n in names if n.startswith("samples/") and n.endswith(".json")]
    for s in samples:
        s["_log"] = log.name  # which file it came from, for the provenance record
    return status, samples


FAILED = object()  # a launcher task folder with no completed attempt


def samples(folder: Path):
    """Completed samples: per task folder of a launcher run, the attempt that succeeded (or FAILED with the folder name
    when none did); else every log's samples."""
    assert folder.is_dir(), f"no such source folder: {folder}"
    task_dirs = sorted(d for d in (folder / "runs").glob("*") if d.is_dir()) if (folder / "runs").exists() else []
    assert task_dirs or any(folder.glob("*.eval")), f"{folder} holds no runs/ task folders and no .eval files"
    if task_dirs:
        for d in task_dirs:
            done = [s for log in sorted(d.glob("*.eval")) for status, ss in [read(log)] if status == "success"
                    for s in ss if not s.get("error")]
            yield done[-1] if done else (FAILED, d.name)
        return
    for log in sorted(folder.glob("*.eval")):
        yield from read(log)[1]


def main_and_side(s: dict) -> tuple[str, str | None]:
    md = s["metadata"]
    main = md.get("main_id") or (md.get("main_task") or {}).get("id")
    side = md.get("real_side_task_id") or ((md.get("side_task") or {}).get("id") if isinstance(md.get("side_task"), dict)
                                            else md.get("side_task"))
    return main, side


def measure(selected: list[dict]) -> dict:
    c = Counter()
    for s in selected:
        calls = {tc["id"]: tc for m in s["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
        for m in s["messages"]:
            if m["role"] != "tool" or m.get("function") == "submit":
                continue
            c["tool results"] += 1
            if m.get("error"):
                c["tool errors"] += 1
                c[f"tool error: {m['error'].get('type')}"] += 1
            text = tool_result_text(m)
            call = calls.get(m.get("tool_call_id")) or {}
            if m.get("function") == "text_editor":
                if (call.get("arguments") or {}).get("command") == "insert":
                    c["insert results"] += 1
                    c["insert: cat -n snippet" if "Here's the result of running `cat -n`" in text else
                      "insert: diff" if "has been edited. Here's a diff" in text else "insert: other"] += 1
                if "has been edited. Here's a diff" in text:
                    body = text.split("@@", 1)[1].splitlines()[1:] if "@@" in text else []
                    numbered = sum(bool(NUMBERED.match(l)) for l in body)
                    plain = sum(bool(PLAIN.match(l)) and not NUMBERED.match(l) for l in body)
                    c["edit diffs"] += 1
                    c["diff: all numbered" if numbered and not plain else "diff: all plain" if plain and not numbered
                      else "diff: mixed" if numbered and plain else "diff: no hunk lines"] += 1
            if m.get("function") == "bash" and not m.get("error"):
                lines = [l for l in text.splitlines() if l.strip()]
                err = [i for i, l in enumerate(lines) if STDERR.match(l)]
                if err and len(err) < len(lines):
                    c["bash with stderr and stdout"] += 1
                    first_out = min(i for i in range(len(lines)) if i not in err)
                    c["stderr first" if max(err) < first_out else
                      "stderr last" if min(err) > max(i for i in range(len(lines)) if i not in err) else
                      "stderr interleaved"] += 1
    share = lambda k, of: round(c[k] / c[of], 3) if c[of] else None
    return {"samples": len(selected), **dict(c),
            "share diffs all numbered": share("diff: all numbered", "edit diffs"),
            "share inserts as cat -n snippet": share("insert: cat -n snippet", "insert results"),
            "share stderr first": share("stderr first", "bash with stderr and stdout"),
            "tool error rate": share("tool errors", "tool results")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("sources", nargs="+", help="label=mode:folder (a local folder; name its S3 cohort in --note)")
    p.add_argument("--note", default="", help="where each source folder came from (cohort alias or S3 prefix)")
    args = p.parse_args()
    tasks = json.loads(args.tasks.read_text())["tasks"]
    mains, cells = {t["main"] for t in tasks}, {t["cell"] for t in tasks}
    key = lambda s, mode: (lambda ms: ms[0] if mode == "honest" else f"{ms[0]}:{ms[1]}")(main_and_side(s))
    loaded = {}
    for source in args.sources:
        label, rest = source.split("=", 1)
        mode, folder = rest.split(":", 1)
        wanted = mains if mode == "honest" else cells
        all_samples = list(samples(Path(folder)))
        failed = sorted(name for x in all_samples if isinstance(x, tuple) for name in [x[1]] if name in mains)
        selected = [s for s in all_samples if not isinstance(s, tuple) and key(s, mode) in wanted]
        bad = {}  # task key -> reason, for any key with an invalid run (every run of a kept task must be valid)
        for s in selected:
            if invalidity(s) is not None:
                bad.setdefault(key(s, mode), invalidity(s))
        loaded[label] = (mode, folder, [s for s in selected if key(s, mode) not in bad], bad, failed)
    # Each mode's sources are compared on the tasks every one of them has only valid runs for.
    common = {mode: set.intersection(*[{key(s, mode) for s in ss} for m, _, ss, _, _ in loaded.values() if m == mode])
              for mode in {v[0] for v in loaded.values()}}
    out = {}
    for label, (mode, folder, valid, bad, failed) in loaded.items():
        keep = [s for s in valid if key(s, mode) in common[mode]]
        out[label] = {"mode": mode, "folder": folder, "tasks_compared": len(common[mode]),
                      "excluded_invalid": bad, "excluded_no_completed_attempt": failed,
                      "excluded_not_in_common": sorted({key(s, mode) for s in valid} - common[mode]),
                      # Exactly what was measured. A local folder mirrors an S3 cohort, so these are its object names.
                      "measured": sorted([s["_log"], s["id"], s["epoch"]] for s in keep),
                      **measure(keep)}
    keys = ["tasks_compared", "samples", "tool results", "edit diffs", "share diffs all numbered", "insert results",
            "share inserts as cat -n snippet", "bash with stderr and stdout", "share stderr first", "tool error rate"]
    print(f"{'':22s}" + "".join(f"{k[:18]:>20s}" for k in keys))
    for label, r in out.items():
        print(f"{label:22s}" + "".join(f"{str(r.get(k, 0)):>20s}" for k in keys))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    with open(HERE / "results" / f"tool_outputs_{stamp}.json", "x") as f:
        f.write(json.dumps({"tasks": str(args.tasks), "note": args.note, "sources": out}, indent=1))


if __name__ == "__main__":
    main()
