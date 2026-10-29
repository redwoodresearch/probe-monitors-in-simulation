"""Is every staged rewrite clean? (Modal CPU, about $0.05; PREDICTIONS.md was committed first.)

For each staged run below, every item is diffed against its original in the base dump (same key; the arm's prefix names
the original arm). Every changed message must be a text_editor str_replace or insert output, and each changed output
gets one type:
  gutter / newline / gutter+newline   the str_replace output equals the original up to the hunk lines' `{n:>4}|` gutter
                                      and the trailing newline (either direction)
  note added / note removed           the insert output equals the original plus or minus "\n" + the review note
                                      (the original's trailing newline may go too)
  re-rendered insert                  the simulator's insert diff rebuilt as the real snippet (rerender.py; its own
                                      exact/approximate categories are in the staged run's rerender_report.json)
  other                               anything else: a bug
Line numbers: a per-file tracker (full views, creates, and text_editor edits it can apply; any call the Q2 labeler does
not call read-only clears it) checks each numbered context or removed line of every gutter the rewrite added against
the file at that point. Real runs' own str_replace gutters are checked the same way as the positive control.
Five before/after pairs per type go to data/ (raw text, not results/).

  uv run modal run experiments/2026-10-08_q2_rewrite_audit/run_rewrite_audit.py
"""
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"
STAGED = ["2026-10-07_q2_swap_inputs_v1", "2026-10-07_q2_swap_note_inputs_v1", "2026-10-07_q2_swap_v5_inputs_v1",
          "2026-10-08_q2_swap_sr_inputs_v1", "2026-10-08_q2_swap_q3_inputs_v1", "2026-10-08_q2_swap_combined_inputs_v1",
          "2026-10-08_q2_swap_ext_inputs_v1", "2026-10-08_q2_surface_edits_inputs_v1"]
app = modal.App("probe-shift_aswin-surya_sub-q2-rewrite-audit_kill-2026-11-08")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "pyyaml")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py").add_local_file(SWAP / "str_replace_tells.py", "/root/str_replace_tells.py")
         .add_local_python_source("probe_shift"))
# Each swap arm's variant, from ../2026-10-07_q2_rendering_swap/config/swap*.yaml: regenerated with the staging code; the parsed messages are compared (not the serialized bytes).
VARIANTS = {"sim-attack-realstyle": "full", "sim-attack-snippet-nonote": "snippet_no_note", "sim-attack-diff-note": "diff_with_note",
            "sim-attack-diff-nonewline": "diff_no_newline", "sim-attack-sr-gutter": "sr_gutter", "sim-attack-sr-nonewline": "sr_no_newline",
            "sim-attack-sr-both": "sr_gutter_no_newline", "sim-attack-note-sr-both": "diff_with_note+sr_gutter_no_newline"}
GUTTER = re.compile(r"^\s*(\d+)\|([ +\-]) (.*)$")
CAT_N = re.compile(r"^\s*(\d+)\t(.*)$")


def original_arm(arm: str) -> str:
    return next(a for a in ("real-attack-e1", "real-honest", "sim-attack") if arm.startswith(a))


def strip_gutter(out: str) -> str:
    return "\n".join(f"{m.group(2)}{m.group(3)}" if (m := GUTTER.match(x)) else x for x in out.split("\n"))


def calls_of(messages: list[dict], args_of) -> dict[int, tuple[str, dict]]:
    """Output message index -> (tool, arguments); calls and outputs pair in order, as in the swap's helpers."""
    out, pending = {}, []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            pending = [(c["function"]["name"], args_of(c["function"])) for c in m.get("tool_calls") or []]
        elif m["role"] == "tool" and pending:
            out[i] = pending.pop(0)
    return out


def check_numbers(messages: list[dict], calls: dict, targets: set[int], read_only, misses: list | None = None) -> dict[int, Counter]:
    """For each output index in `targets` (a str_replace output with a gutter), Counter of its numbered old-side lines:
    match / mismatch against the tracked file, or unknown. The tracker follows the transcript in order."""
    files, res = {}, {}
    for i in sorted(calls):
        tool, a = calls[i]
        out = messages[i]["content"] if isinstance(messages[i]["content"], str) else ""
        cmd = a.get("command") if tool == "text_editor" else None
        path = a.get("path") if not a.get("path") or a["path"].startswith("/") else f"/app/{a['path']}"  # the tool's cwd is /app
        if i in targets:
            c, f = Counter(), files.get(path)
            for line in out.partition("+++ edited_version\n")[2].split("\n"):
                if (m := GUTTER.match(line)) and m.group(2) in " -":
                    n = int(m.group(1))
                    if f is None or n > len(f):
                        c["unknown"] += 1
                    elif f[n - 1].expandtabs().rstrip() == m.group(3).expandtabs().rstrip():  # the tools expand tabs
                        c["match"] += 1
                    else:
                        c["mismatch"] += 1
                        if misses is not None and len(misses) < 40:
                            misses.append({"msg": i, "path": path, "line": n, "tracked": f[n - 1][:120], "diff": m.group(3)[:120]})
            res[i] = c
        if cmd == "view" and "view_range" not in a and out.startswith("Here's the result of running `cat -n`") \
                and "<response clipped>" not in out and "truncated" not in out[-300:]:  # a whole, unclipped file
            rows = [CAT_N.match(x) for x in out.split("\n")[1:]]
            files[path] = [m.group(2) for m in rows if m] or None
        elif cmd == "create" and "created successfully" in out[:200]:  # a failed create (the file exists) changes nothing
            files[path] = (a.get("file_text") or "").split("\n")
        elif cmd == "create":
            files[path] = None
        elif cmd == "str_replace" and files.get(path) is not None and "Here's a diff showing" in out[:300]:  # it succeeded
            text, old = "\n".join(files[path]), a.get("old_str") or ""
            files[path] = text.replace(old, a.get("new_str") or "", 1).split("\n") if old and text.count(old) == 1 else None
        elif cmd == "insert" and files.get(path) is not None and ("Here's a diff showing" in out[:300] or "has been edited" in out[:300]):
            text = a.get("insert_text") or a.get("new_str") or ""
            # Split as the real tool does: a closing newline adds an empty line. Stripping it first (tested 15:36 PT,
            # audit_2026-10-08_153646) raises the positive control's mismatches from 0 to 29 outputs.
            n, new = int(a.get("insert_line", 0)), text.split("\n")
            files[path] = files[path][:n] + new + files[path][n:]
        elif cmd in ("str_replace", "insert", "undo_edit"):  # an edit it could not apply: that file is unknown now
            files[path] = None
        elif tool != "text_editor" and not read_only(tool, a):
            files = {}
    return res


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=3600, retries=0)
def audit() -> dict:
    import sys

    sys.path.insert(0, "/root")
    import rerender as rr
    import str_replace_tells as st
    from rerender import DIFF, REVIEW, args_of, read_only

    def regenerate(messages: list[dict], variant: str) -> list[dict]:  # run_swap.stage's order: each part in turn
        for part in variant.split("+"):
            messages = (st.swap_messages if part in st.VARIANTS else rr.rerender_messages)(messages, part)[0]
        return messages

    regen = defaultdict(Counter)

    base = {}
    for shard in sorted((Path("/acts") / BASE / "inputs").glob("shard_*.jsonl")):
        for it in map(json.loads, shard.read_text().splitlines()):
            base[(it["arm"], it["key"])] = it["messages"]
    rng, report, examples, numbers = random.Random(0), defaultdict(Counter), defaultdict(list), defaultdict(Counter)
    for run in STAGED:
        for shard in sorted((Path("/acts") / run / "inputs").glob("shard_*.jsonl")):
            for it in map(json.loads, shard.read_text().splitlines()):
                orig = base[(original_arm(it["arm"]), it["key"])]
                new, row = it["messages"], report[f"{run} | {it['arm']}"]
                if it["arm"] in VARIANTS:  # the parsed staged messages against a fresh regeneration from the original
                    again, g = regenerate(orig, VARIANTS[it["arm"]]), regen[f"{run} | {it['arm']}"]
                    g["items"] += 1
                    g["items identical"] += again == new
                    for i, (a, b) in enumerate(zip(again, new)):
                        if a != orig[i] or b != orig[i]:
                            g["changed outputs identical" if a == b else "changed outputs differ"] += 1
                row["items"] += 1
                if len(new) != len(orig):
                    row["message count differs"] += 1
                    continue
                calls, gutters = calls_of(orig, args_of), set()
                for i, (a, b) in enumerate(zip(orig, new)):
                    if a == b:
                        continue
                    tool, args = calls.get(i, (None, {}))
                    cmd = args.get("command") if tool == "text_editor" else None
                    x, y = a.get("content"), b.get("content")
                    if a.get("role") != "tool" or {k: v for k, v in a.items() if k != "content"} != {k: v for k, v in b.items() if k != "content"} \
                            or not isinstance(x, str) or not isinstance(y, str):
                        kind = "other: not a text output"
                    elif cmd == "str_replace" and DIFF in x[:300]:
                        sx, sy = strip_gutter(x), strip_gutter(y)
                        same = sx.rstrip("\n") == sy.rstrip("\n")  # equal once gutters and trailing newlines are set aside
                        g, nl = x.rstrip("\n") != y.rstrip("\n"), x.endswith("\n") != y.endswith("\n")
                        kind = "other: str_replace" if not same else "gutter+newline" if g and nl else "gutter" if g else "newline" if nl else "other: str_replace"
                        if same and g and sy != y:  # the rewrite added a gutter: check its numbers
                            gutters.add(i)
                    elif cmd == "insert" and y.rstrip("\n") == x.rstrip("\n") + "\n" + REVIEW:
                        kind = "note added"
                    elif cmd == "insert" and x == y + "\n" + REVIEW:
                        kind = "note removed"
                    elif cmd == "insert" and x.rstrip("\n") == y and x != y:
                        kind = "newline"
                    elif cmd == "insert" and DIFF in x[:300] and "cat -n" in y[:200]:
                        kind = "re-rendered insert"
                    else:
                        kind = f"other: {tool} {cmd}"
                    row[kind] += 1
                    if len(examples[kind]) < 5 or rng.random() < 0.01:
                        examples[kind] = (examples[kind] + [{"run": run, "arm": it["arm"], "key": it["key"], "msg": i, "before": x, "after": y}])[-5:]
                for c in check_numbers(new, calls, gutters, read_only).values():
                    numbers[f"{run} | {it['arm']}"]["outputs"] += 1
                    numbers[f"{run} | {it['arm']}"]["outputs with a mismatch" if c["mismatch"] else "outputs all known match" if c["match"] and not c["unknown"]
                                                       else "outputs partly or wholly unknown"] += 1
                    numbers[f"{run} | {it['arm']}"].update({f"lines {k}": v for k, v in c.items()})
    for (arm, key), msgs in base.items():  # positive control: real gutters against the same tracker
        if arm == "real-attack-e1":
            calls, misses = calls_of(msgs, args_of), examples["control mismatches"]
            t = {i for i, (tool, a) in calls.items() if tool == "text_editor" and a.get("command") == "str_replace" and isinstance(msgs[i]["content"], str)
                 and DIFF in msgs[i]["content"][:300]}
            for c in check_numbers(msgs, calls, t, read_only, misses if len(misses) < 40 else None).values():
                numbers["control: real-attack-e1 own gutters"]["outputs"] += 1
                numbers["control: real-attack-e1 own gutters"]["outputs with a mismatch" if c["mismatch"] else "outputs all known match" if c["match"] and not c["unknown"]
                                                              else "outputs partly or wholly unknown"] += 1
                numbers["control: real-attack-e1 own gutters"].update({f"lines {k}": v for k, v in c.items()})
    rerender = {run: Counter(r["category"] for v in json.loads((Path("/acts") / run / "rerender_report.json").read_text()).values()
                             for r in v.get("inserts", []))
                for run in STAGED if (Path("/acts") / run / "rerender_report.json").exists()}
    return {"changed outputs by type": {k: dict(v) for k, v in report.items()}, "line numbers": {k: dict(v) for k, v in numbers.items()},
            "regenerated vs staged": {k: dict(v) for k, v in regen.items()},
            "insert re-render categories (stage reports)": {k: dict(v) for k, v in rerender.items()}, "examples": examples}


@app.local_entrypoint()
def main():
    res = audit.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    ex = ROOT / "data" / "rewrite_audit" / f"examples_{stamp}.json.gz"
    ex.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ex, "wt") as f:
        json.dump(res.pop("examples"), f)
    out = HERE / "results" / f"audit_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res | {"examples file (not in git)": str(ex.relative_to(ROOT))}, indent=1))
    print(out)
