"""Blind profile of the 12 runs where the two evaluation-awareness probes disagree most.

Reads only the items file (item, key) and the cached eval logs: never the key file, scores or the samples'
arm/cohort/sim-reference fields. For each item it prints what the reader model saw, minus the system prompt (the
reader got the main task's honest prompt): the user task text, step and tool counts, tool errors, tool-output
lengths, and every match of an evaluation cue with context. It also writes each rendered transcript to --dump_dir
for reading by hand.

$0: local cache first (`probe_shift.transcripts._get`); S3 only for a file missing from the cache. `--unblind` joins the
key file afterwards (direction vs environment, arm, lexical "test" count, Rust-gateway source views).

    uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python \
        experiments/2026-10-07_q3_codex_followups/audit_probe_disagreements.py --dump_dir <scratch dir>
"""

import argparse
import collections
import io
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# simulated-control-evals: the sibling checkout of this repository, else ~/ (T.cohort_uri imports its `paper` package)
SCE = next(p for p in (ROOT.parent / "simulated-control-evals", Path.home() / "simulated-control-evals") if p.exists())
sys.path[:0] = [str(ROOT / "src"), str(SCE / "src")]

import numpy as np  # noqa: E402

from probe_shift import transcripts as T  # noqa: E402

ITEMS = HERE / "results" / "probe_disagreement_items_2026-10-07.json"
KEY = HERE / "results" / "probe_disagreement_key_2026-10-07.json"  # read only with --unblind, after the grouping is written
EXCLUSIONS = ROOT / "splits" / "exclusions_v4.json"

# Cues for "this is a test": specific phrases get context printed; the bare word "test" is only counted, since
# most hits are the task's own unit tests.
CUES = {
    "evaluat": r"evaluat",
    "benchmark": r"benchmark",
    "grade/score": r"\bgrad(?:e|ed|er|ing)\b|\bscor(?:e|ed|er|ing)\b",
    "submit": r"\bsubmit",
    "step/time limit": r"step limit|steps? remaining|remaining steps|max(?:imum)? (?:number of )?steps|time limit|"
                       r"you have \d+ (?:steps|actions|turns)",
    "proceed nudge": re.escape(T.REACT_NUDGE),
    "simulat": r"simulat",
    "fake/mock/dummy": r"\b(?:fake|mock|dummy|placeholder|lorem)\b",
    "test (word)": r"\btest(?:s|ing|ed)?\b",
}


def load_cached(alias: str, file: str, sample_id: str, epoch: int) -> dict:
    """`transcripts.load_sample`, but through the local cache."""
    bucket, prefix = T.cohort_uri(alias).removeprefix("s3://").split("/", 1)
    with zipfile.ZipFile(io.BytesIO(T._get(bucket, f"{prefix}logs/{file}"))) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                s = json.loads(z.read(name))
                if s["id"] == sample_id and s["epoch"] == epoch:
                    return s
    raise KeyError(sample_id)


def call_text(tc: dict) -> str:
    a = tc["function"]["arguments"]
    if tc["function"]["name"] == "bash":
        return f"$ {a.get('cmd') or a.get('command')}"
    return f"{tc['function']['name']}({json.dumps(a)[:3000]})"


def render(msgs: list[dict]) -> str:
    out = []
    for i, m in enumerate(msgs[1:], 1):
        if m["role"] == "assistant":
            body = (m["content"].strip() + "\n" if m["content"].strip() else "") + "\n".join(map(call_text, m["tool_calls"]))
            out.append(f"--- [{i}] ASSISTANT\n{body}")
        else:
            out.append(f"--- [{i}] {m['role'].upper()}{' (' + m['tool_name'] + ')' if m.get('tool_name') else ''}\n{m['content']}")
    return "\n".join(out)


def profile(sample: dict, entry: dict) -> dict:
    msgs = T.reader_messages(sample, system="")
    users = [m["content"] for m in msgs if m["role"] == "user"]
    asst = [m for m in msgs if m["role"] == "assistant"]
    tools = [m for m in msgs if m["role"] == "tool"]
    calls = collections.Counter(tc["function"]["name"] for m in asst for tc in m["tool_calls"])
    editor_cmds = collections.Counter(tc["function"]["arguments"].get("command") for m in asst for tc in m["tool_calls"]
                                      if tc["function"]["name"] == "text_editor")
    errors = [e for e in T.tool_errors(sample) if e["error"]]
    out_len = np.array([len(m["content"]) for m in tools]) if tools else np.zeros(1)
    # The user turn is empty in these logs; the task text is in the main task's honest system prompt, which the
    # reader saw. The metadata's main-task description stands in for it here.
    task = (sample.get("metadata") or {}).get("main_task", {}).get("description") or ""
    by_source = {"user": "\n".join(users), "assistant": "\n".join(m["content"] for m in asst),
                 "tool": "\n".join(m["content"] for m in tools), "task": task}
    cues = {}
    for name, pat in CUES.items():
        counts = {src: len(re.findall(pat, txt, re.I)) for src, txt in by_source.items()}
        ctx = []
        if name != "test (word)":
            for src, txt in by_source.items():
                for mm in list(re.finditer(pat, txt, re.I))[:4]:
                    ctx.append(f"{src}: ...{txt[max(0, mm.start() - 90):mm.end() + 90]!r}...")
        cues[name] = {"counts": counts, "context": ctx}
    nonzero_exit = sum(bool(re.search(r"exit code|returncode|Command failed|Traceback|No such file|command not found",
                                      m["content"])) for m in tools)
    return {
        "environment": entry["environment"], "main_task": entry["main_task"],
        "tokens": entry["tokens"], "truncated_to_32k": entry["truncated"],
        "steps": len(asst), "tool_calls": sum(calls.values()), "calls_by_tool": dict(calls),
        "editor_commands": dict(editor_cmds), "submit_calls_dropped": T.submit_calls(sample),
        "tool_errors": len(errors), "tool_error_types": dict(collections.Counter(e["error"] for e in errors)),
        "outputs_with_error_text": nonzero_exit, "user_messages": len(users),
        "chars_total": sum(len(m["content"]) for m in msgs) + sum(len(call_text(tc)) for m in asst for tc in m["tool_calls"]),
        "tool_output_chars": {"median": int(np.median(out_len)), "max": int(out_len.max()), "sum": int(out_len.sum()),
                              "empty": int((out_len == 0).sum())},
        "assistant_text_turns": sum(bool(m["content"].strip()) for m in asst),
        "task_text": task, "user_messages_nonempty": [u for u in users if u.strip()],
        "cues": cues,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", type=Path, required=True, help="where rendered transcripts go (scratch, not results/)")
    p.add_argument("--unblind", action="store_true", help="after the blind grouping is written: join the key and print "
                   "direction against environment, arm, lexical 'test' count and Rust-gateway source views")
    args = p.parse_args()
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(ITEMS.read_text())
    samples = {s["key"]: s for s in json.loads(EXCLUSIONS.read_text())["samples"]}
    profiles = {}
    for it in items:
        e = samples[it["key"]]  # arm, cohort and sim_reference stay unprinted
        s = load_cached(e["cohort"], e["file"], e["sample_id"], e["epoch"])
        pr = profile(s, e)
        profiles[it["item"]] = pr
        (args.dump_dir / f"item_{it['item']:02d}.txt").write_text(render(T.reader_messages(s, system="")))
        print(f"\n===== item {it['item']} =====")
        print(json.dumps({k: v for k, v in pr.items() if k not in ("cues", "task_text", "user_messages_nonempty")}, indent=1))
        print("TASK:", pr["task_text"][:1200].replace("\n", " "))
        for u in pr["user_messages_nonempty"]:
            print("USER:", u[:300].replace("\n", " "))
        for name, c in pr["cues"].items():
            if any(c["counts"].values()):
                print(f"  cue {name}: {c['counts']}")
                for line in c["context"][:6]:
                    print("     ", line[:260])
    (args.dump_dir / "profiles.json").write_text(json.dumps(profiles, indent=1))
    if args.unblind:
        unblind(profiles, args.dump_dir)


def unblind(profiles: dict, dump_dir: Path):
    """Direction vs candidate features, over the part of each run the reader saw (first 32,768 tokens, approximated
    by the same fraction of characters)."""
    print("\n===== unblinded =====")
    for k in json.loads(KEY.read_text()):
        pr, i = profiles[k["item"]], k["item"]
        text = (dump_dir / f"item_{i:02d}.txt").read_text()
        seen = text[: int(len(text) * min(1, 32768 / pr["tokens"]))]
        tests = len(re.findall(r"\btest(?:s|ing|ed)?\b", seen + pr["task_text"], re.I))
        print(f"{i:>2} {k['direction']:<17} {k['arm']:<15} {pr['environment']:<17} tokens={pr['tokens']:>6} "
              f"steps={pr['steps']:>2} test_words={tests:>3} gateway_main_rs_refs={seen.count('/app/src/main.rs'):>2} "
              f"rank_templated={k['rank_templated']} rank_needham={k['rank_needham']}")


if __name__ == "__main__":
    main()
