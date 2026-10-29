"""Per-task metrics for a pilot run: main-task success, steps, tool-call format errors, stalls and refusals, tokens,
wall-clock and generation time. Reads the .eval logs run_pilot.py wrote (copied back from the devbox) and writes one
JSONL row per sample to results/, plus a plain-text transcript per sample for reading.

  uv run --project ../simulated-control-evals python experiments/2026-10-06_phase2_feasibility/analyze_pilot.py \
    .run_cache/phase2_pilot/2026-10-06_la_real_honest_phase2-qwen3_32b
"""

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

HERE = Path(__file__).parent
REFUSAL = re.compile(r"\b(I can(?:no|')t (?:help|assist|do)|I(?:'m| am) (?:sorry|unable)|I won't|I will not)\b", re.I)
# Tool-call syntax left in the visible text means the server's parser did not turn it into a call.
UNPARSED_CALL = re.compile(r"<tool_call>|</tool_call>|<function=|\"arguments\"\s*:", re.I)


def completed_model_events(sample) -> list[ModelEvent]:
    """Model calls that returned. A failed or interrupted call keeps a placeholder output (an empty completion with
    stop reason `stop`) that never happened, so it is left out of turn counts, stop reasons and usage."""
    return [e for e in sample.events if isinstance(e, ModelEvent) and e.output is not None and e.output.choices
            and e.error is None and not e.pending]


def conversation(sample) -> tuple[list, str]:
    """The sample's messages, or, when the policy was cut off (time limit, error), the longest conversation a model
    call saw plus that call's answer: sce's arena_policy copies messages onto the state only after the agent returns,
    so an interrupted sample's `messages` can hold just the task."""
    completed = completed_model_events(sample)
    # A failed final call's input still carries the last tool outputs; only completed calls contribute an answer.
    candidates = [[*e.input, e.output.message] if e in completed else list(e.input)
                  for e in sample.events if isinstance(e, ModelEvent)]
    from_events = max(candidates, key=len, default=[])
    if len(from_events) > len(sample.messages):
        return from_events, "model_events"
    return list(sample.messages), "messages"


def sample_row(path: Path, sample) -> dict:
    messages, source = conversation(sample)
    assistants = [m for m in messages if isinstance(m, ChatMessageAssistant)]
    tools = [m for m in messages if isinstance(m, ChatMessageTool)]
    calls = [c for m in assistants for c in m.tool_calls or []]
    models = completed_model_events(sample)
    failed_model_calls = sum(isinstance(e, ModelEvent) and (e.error is not None or bool(e.pending)) for e in sample.events)
    usages = [e.output.usage for e in models if e.output.usage]
    reasoning = [c.reasoning for m in assistants if isinstance(m.content, list) for c in m.content if c.type == "reasoning"]
    score = sample.scores.get("linux_scorer") if sample.scores else None
    md = sample.metadata or {}
    return {
        "file": path.name, "sample_id": sample.id, "epoch": sample.epoch,
        "env": (md.get("environment") or {}).get("id"), "main_task": (md.get("main_task") or {}).get("id"),
        "main_task_success": (score.value or {}).get("main_task_success") if score else None,
        "scorer_explanation": score.explanation if score else None,
        "error": sample.error.message if sample.error else None,
        "limit": sample.limit.model_dump() if sample.limit else None,
        "messages_source": source, "assistant_turns": len(assistants), "tool_calls": len(calls),
        "tool_calls_by_name": {n: sum(c.function == n for c in calls) for n in sorted({c.function for c in calls})},
        "submitted": any(c.function == "submit" for c in calls),
        "turns_without_tool_call": sum(not m.tool_calls for m in assistants),
        "empty_turns": sum(not m.tool_calls and not m.text.strip() for m in assistants),
        "call_parse_errors": sum(c.parse_error is not None for c in calls),
        "tool_errors_by_type": {t: sum(1 for m in tools if m.error and m.error.type == t)
                                for t in sorted({m.error.type for m in tools if m.error})},
        "unparsed_call_text_turns": sum(bool(UNPARSED_CALL.search(m.text)) for m in assistants),
        "refusal_like_turns": sum(bool(REFUSAL.search(m.text)) for m in assistants),
        "model_calls": len(models), "failed_model_calls": failed_model_calls,
        "stop_reasons": {r: sum(e.output.stop_reason == r for e in models) for r in sorted({e.output.stop_reason for e in models})},
        "input_tokens": sum(u.input_tokens for u in usages), "output_tokens": sum(u.output_tokens for u in usages),
        "reasoning_tokens": sum(u.reasoning_tokens or 0 for u in usages),
        "max_context_tokens": max((u.input_tokens + u.output_tokens for u in usages), default=0),
        "reasoning_chars": sum(len(r) for r in reasoning),
        "generation_seconds": round(sum(e.working_time or 0 for e in models), 1),
        "total_seconds": sample.total_time, "working_seconds": sample.working_time,
    }


def transcript_text(sample) -> str:
    messages, source = conversation(sample)
    lines = [f"(messages from {source})"]
    for m in messages:
        if isinstance(m, ChatMessageAssistant):
            for c in m.content if isinstance(m.content, list) else []:
                if c.type == "reasoning":
                    lines.append(f"--- [reasoning]\n{c.reasoning}")
            lines.append(f"--- [assistant]\n{m.text}")
            lines += [f"--- [call {c.function}] {json.dumps(c.arguments)[:4000]}"
                      + (f"  PARSE ERROR: {c.parse_error}" if c.parse_error else "") for c in m.tool_calls or []]
        else:
            text = m.text if m.role != "tool" else m.text[:3000]
            err = f"  ERROR {m.error.type}: {m.error.message}" if getattr(m, "error", None) else ""
            lines.append(f"--- [{m.role}]{err}\n{text}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--transcripts", type=Path, help="where to write readable transcripts (default: <run_dir>/text)")
    args = ap.parse_args()
    # The commit recorded with the rows must contain this analyzer and its config, so refuse uncommitted changes.
    dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "analyze_pilot.py", "config"],
                           capture_output=True, text=True, check=True).stdout
    assert not dirty, f"commit the analyzer and config before running; uncommitted:\n{dirty}"
    text_dir = args.transcripts or args.run_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = sorted(args.run_dir.rglob("*.eval"))
    print(f"{len(paths)} .eval files under {args.run_dir}")
    for path in paths:
        log = read_eval_log(str(path), resolve_attachments=True)  # event inputs are attachment:// refs otherwise
        if not log.samples:  # a startup failure; reported below as a launched task without a sample
            print(f"WARNING: {path.name} has no samples (status {log.status}: {log.error.message if log.error else None})")
        for sample in log.samples or []:
            row = sample_row(path, sample) | {"eval_status": log.status, "policy": log.plan.steps[-1].params.get("model_name")}
            rows.append(row)
            (text_dir / f"{row['env']}__{row['main_task']}.txt").write_text(transcript_text(sample))
    launch_path = args.run_dir / "launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
    # Every launched task must show up as a sample; one that crashed before writing a log is reported, not dropped.
    seen = {(r["env"], r["main_task"]) for r in rows}
    missing = [t for t in (launch or {}).get("tasks", []) if (t["task"]["env"], t["task"]["main"]) not in seen]
    for t in missing:
        rows.append({"env": t["task"]["env"], "main_task": t["task"]["main"], "main_task_success": None,
                     "error": f"no sample in any .eval (process exit code {t['exit_code']}; see {t['task']['env']}.log)"})
    meta = {"run_dir": str(args.run_dir), "launch": launch, "launch_json_found": launch is not None,
            "git_commit": subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip()}
    (HERE / "results").mkdir(exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"pilot_{args.run_dir.name}_{stamp}.jsonl"
    with open(out, "x") as f:  # results are append-only: a same-second rerun fails instead of overwriting
        f.write("\n".join(json.dumps(r | {"meta": meta}, default=str) for r in rows) + "\n")
    cols = ["env", "main_task_success", "error", "assistant_turns", "tool_calls", "submitted", "call_parse_errors",
            "unparsed_call_text_turns", "empty_turns", "refusal_like_turns", "input_tokens", "output_tokens",
            "max_context_tokens", "generation_seconds", "total_seconds"]
    for r in rows:
        print({c: (r.get(c)[:80] if isinstance(r.get(c), str) else r.get(c)) for c in cols}
              | {"limit": r.get("limit"), "stop_reasons": r.get("stop_reasons"), "tool_errors": r.get("tool_errors_by_type")})
    if launch is None:
        print("WARNING: no launch.json, so tasks that never wrote a log cannot be counted")
    print(f"{sum(r['main_task_success'] == 'C' for r in rows)}/{len(rows)} main tasks passed "
          f"({len(missing)} launched without a sample) -> {out}")


if __name__ == "__main__":
    main()
