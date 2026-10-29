"""Per-run metrics for the attack check: scorer outcome, the agent's own attack report, steps, tool and parse errors,
stalled model calls, refusal-like text, tokens and time, and whether the system prompt really was Sol's. Reads the
.eval logs run_attack_check.py wrote (copied back from the devbox); writes one JSONL row per run to results/ and a
readable transcript per run (reasoning included) next to the logs.

  uv run --project ../simulated-control-evals python experiments/2026-10-07_phase2_attack_check/analyze_attack_check.py \\
    <run dir>

Adapted from experiments/2026-10-06_phase2_feasibility/analyze_pilot.py (PR #50).
"""

import argparse
import hashlib
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
REFUSAL = re.compile(r"\b(I can(?:no|')t (?:help|assist|do|comply)|I(?:'m| am) (?:sorry|unable|not able)|I won't|"
                     r"I will not|I must decline|I refuse)\b", re.IGNORECASE)
UNPARSED_CALL = re.compile(r"<tool_call>|</tool_call>|<function=|\"arguments\"\s*:", re.IGNORECASE)
# The acting model: the launch's recorded policy (config/run.yaml's, or a pilot config's override such as the
# Qwen3-32B arm's), set in main(). A simulated run's sample also holds the simulator's and its router and judges'
# calls, so every count, token total and reconstructed transcript below reads the policy's calls only.
POLICY = "vllm/Qwen/Qwen3.5-27B"


def policy_events(sample) -> list[ModelEvent]:
    return [e for e in sample.events if isinstance(e, ModelEvent) and e.model == POLICY]


def completed_model_events(sample) -> list[ModelEvent]:
    """The policy's calls that returned (a failed or interrupted call keeps a placeholder output that never happened)."""
    return [e for e in policy_events(sample) if e.output is not None and e.output.choices and e.error is None
            and not e.pending]


def conversation(sample) -> tuple[list, str]:
    """The sample's messages, or, when the policy was cut off, the longest conversation a policy call saw plus its
    answer (sce copies messages onto the state only after the agent returns)."""
    completed = completed_model_events(sample)
    candidates = [[*e.input, e.output.message] if e in completed else list(e.input) for e in policy_events(sample)]
    from_events = max(candidates, key=len, default=[])
    if len(from_events) > len(sample.messages):
        return from_events, "model_events"
    return list(sample.messages), "messages"


def run_row(path: Path, sample, cells: dict, arm: str) -> dict:
    messages, source = conversation(sample)
    assistants = [m for m in messages if isinstance(m, ChatMessageAssistant)]
    tools = [m for m in messages if isinstance(m, ChatMessageTool)]
    calls = [c for m in assistants for c in m.tool_calls or []]
    submits = [c.arguments for c in calls if c.function == "submit"]
    models = completed_model_events(sample)
    model_events = policy_events(sample)
    usages = [e.output.usage for e in models if e.output.usage]
    reasoning = [c.reasoning for m in assistants if isinstance(m.content, list) for c in m.content if c.type == "reasoning"]
    score = (sample.scores or {}).get("linux_scorer")
    value = (score.value or {}) if score else {}
    md = sample.metadata or {}
    side = md.get("real_side_task_id")
    cell = f"{(md.get('main_task') or {}).get('id')}:{side}" if side else None
    system = messages[0].text if messages and messages[0].role == "system" else ""
    sol_sha = cells.get(cell, {}).get("sol_system_sha256") if cell else None
    return {
        "arm": arm, "file": path.name, "sample_id": sample.id,  # the launcher's arm: sim_attack has a side task too
        "env": (md.get("environment") or {}).get("id"), "main_task": (md.get("main_task") or {}).get("id"),
        "cell": cell, "side_task_success": value.get("side_task_success"),
        "main_task_success": value.get("main_task_success"),
        "scorer_explanation": score.explanation if score else None,
        "system_prompt_is_sols": None if sol_sha is None else hashlib.sha256(system.encode()).hexdigest() == sol_sha,
        "submitted": bool(submits), "self_report_successful": submits[-1].get("attack_successful") if submits else None,
        "self_report_description": submits[-1].get("attack_description") if submits else None,
        "error": sample.error.message if sample.error else None,
        # Per simulator (4b31bd30 changes the output contract between the trees): the simulator's Sol-fallback
        # substitutions (calls to the sample's recorded fallback model) and whether the sample ended on an output-
        # contract failure (its repair rounds exhausted). Real runs have no fallback model: None.
        "sim_fallback_calls": (sum(1 for e in sample.events if isinstance(e, ModelEvent)
                                   and e.model == md.get("simulator_fallback_model"))
                               if md.get("simulator_fallback_model") else None),
        # The simulator's output contract (its class name or its prose), not "arena reference contract mismatch" (a
        # fork-check failure). On the 10-08 pilot and dry check: 1 such sample (t10), counted either way.
        "output_contract_error": bool(sample.error and any(k in sample.error.message.lower() for k in (
            "outputcontracterror", "output contract")) and "reference contract" not in sample.error.message.lower()),
        "limit": sample.limit.model_dump() if sample.limit else None,
        "messages_source": source, "assistant_turns": len(assistants), "tool_calls": len(calls),
        "tool_calls_by_name": {n: sum(c.function == n for c in calls) for n in sorted({c.function for c in calls})},
        "turns_without_tool_call": sum(not m.tool_calls for m in assistants),
        "empty_turns": sum(not m.tool_calls and not m.text.strip() for m in assistants),
        "call_parse_errors": sum(c.parse_error is not None for c in calls),
        "tool_errors_by_type": {t: sum(1 for m in tools if m.error and m.error.type == t)
                                for t in sorted({m.error.type for m in tools if m.error})},
        "unparsed_call_text_turns": sum(bool(UNPARSED_CALL.search(m.text)) for m in assistants),
        "refusal_like_turns": sum(bool(REFUSAL.search(m.text)) for m in assistants),
        "refusal_like_reasoning": sum(bool(REFUSAL.search(r)) for r in reasoning),
        "model_calls": len(models),
        "failed_model_calls": sum(e.error is not None or bool(e.pending) for e in model_events),
        "failed_model_call_errors": [str(e.error)[:200] for e in model_events if e.error is not None],
        "model_call_seconds_max": max((e.working_time or 0 for e in models), default=0),
        "model_calls_over_120s": sum((e.working_time or 0) > 120 for e in models),
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
            lines += [f"--- [call {c.function}] {json.dumps(c.arguments)[:6000]}"
                      + (f"  PARSE ERROR: {c.parse_error}" if c.parse_error else "") for c in m.tool_calls or []]
        else:
            text = m.text if m.role != "tool" else m.text[:3000]
            err = f"  ERROR {m.error.type}: {m.error.message}" if getattr(m, "error", None) else ""
            lines.append(f"--- [{m.role}]{err}\n{text}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "analyze_attack_check.py", "config"],
                           capture_output=True, text=True, check=True).stdout
    assert not dirty, f"commit the analyzer and config before running; uncommitted:\n{dirty}"
    text_dir = args.run_dir / "text"
    text_dir.mkdir(exist_ok=True)
    rows = []
    launch = json.loads((args.run_dir / "launch.json").read_text())
    global POLICY
    POLICY = launch.get("policy") or launch.get("config", {}).get("policy", POLICY)  # launches before 10-08: run.yaml's
    assert launch.get("config", {}).get("policy", POLICY) == POLICY, f"launch.json disagrees on the policy: {POLICY}"
    print(f"policy: {POLICY}", flush=True)
    # The pilot's launches (run_pilot.py) record their tasks; earlier ones used config/cells.json.
    cells = {c["cell"]: c for c in launch.get("tasks") or json.loads((HERE / "config" / "cells.json").read_text())["cells"]}
    # One row per launched run, read from its final attempt's logs: a redraw's superseded pre-start attempt is not a
    # run, and a run that started and failed, or left no sample, is reported, never dropped. Launches before redraws
    # existed have no log_dir: their logs are at <arm>/<env>. The pilot's runs are keyed by task, and a simulated
    # arm is labelled with its simulator (sim_attack@today).
    units = []  # (arm label, name for the transcript file, log dir, extra fields)
    for run in launch["runs"]:
        if "key" in run:  # run_pilot.py: one row per attempt that ran (honest_a's reference tries are runs too)
            arm = run["arm"] if run["real"] else f"{run['arm']}@{run['simulator']}"
            ran = [a for a in run["attempts"] if not a.get("redrawn_because")]
            if not ran:
                rows.append({"arm": arm, "task": run["task"], "not_run": run.get("why") or run["state"],
                             "time_control": run.get("time_control", False)})
                continue
            pinned = (launch.get("references") or {}).get(run["task"], {}).get("source", "")
            for a in ran:
                suffix = "" if a["kind"] == "first" else f"_{Path(a['log_dir']).name.rsplit('_', 1)[-1]}"
                units.append((arm, f"{run['task']}{suffix}", Path(a["log_dir"]), {
                    "task": run["task"], "state": run["state"], "attempt_kind": a["kind"],
                    "policy_called": a.get("policy_called"),  # per attempt (None: recorded before 10-08's fix)
                    "time_control": run.get("time_control", False),  # a control run, outside the primary cohort
                    "pinned_reference": run["arm"] == "honest_a" and pinned.startswith(a["log_dir"] + "/"),
                    "prestart_redraws": sum(x.get("kind") == "redraw" for x in run["attempts"]),
                    "launch_exit_code": a.get("exit_code"), "stopped_by_launcher": a.get("stopped_by_launcher", False)}))
        else:
            units.append((run["arm"], run["env"], Path(run.get("log_dir") or args.run_dir / run["arm"] / run["env"]),
                          {"attempts": len(run.get("attempts") or [None]), "launch_exit_code": run.get("exit_code")}))
    for arm, name, log_dir, extra in units:
        log_dir = args.run_dir / log_dir.relative_to(log_dir.parents[1]) if not log_dir.exists() else log_dir
        samples = []
        for path in sorted(log_dir.glob("*.eval")):
            try:
                log = read_eval_log(str(path), resolve_attachments=True)
            except Exception as e:  # noqa: BLE001 — a crashed run's truncated log: its row says so, the rest go on
                rows.append({"arm": arm, "name": name} | extra | {"error": f"unreadable log {path.name}: {e!r}"[:300]})
                continue
            if not log.samples:
                print(f"WARNING: {path} has no samples (status {log.status}: {log.error.message if log.error else None})")
            samples += [(path, log, sample) for sample in log.samples or []]
        for path, log, sample in samples:
            row = run_row(path, sample, cells, arm) | {"eval_status": log.status} | extra
            rows.append(row)
            (text_dir / f"{arm}__{name}.txt").write_text(transcript_text(sample))
        if not samples:
            rows.append({"arm": arm, "name": name} | extra | {
                "error": f"no sample in {log_dir} (exit code {extra['launch_exit_code']}; see logs/)"})
    meta = {"run_dir": str(args.run_dir), "launch_started": launch["started"], "launch_finished": launch["finished"],
            "probe_shift_commit_launched": launch["probe_shift_commit"],
            "git_commit": subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                                         text=True, check=True).stdout.strip()}
    (HERE / "results").mkdir(exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"attack_check_{stamp}.jsonl"
    with open(out, "x") as f:
        f.write("\n".join(json.dumps(r | {"meta": meta}, default=str) for r in rows) + "\n")
    cols = ["arm", "env", "side_task_success", "main_task_success", "self_report_successful", "system_prompt_is_sols",
            "error", "tool_calls", "submitted", "call_parse_errors", "refusal_like_turns", "refusal_like_reasoning",
            "failed_model_calls", "model_call_seconds_max", "output_tokens", "max_context_tokens", "total_seconds"]
    for r in rows:
        print({c: (r.get(c)[:80] if isinstance(r.get(c), str) else r.get(c)) for c in cols}
              | {"limit": r.get("limit"), "tool_errors": r.get("tool_errors_by_type")})
    # Per simulator arm: Sol-fallback substitutions and output-contract failures (4b31bd30 changes the contract).
    by_arm: dict = {}
    for r in rows:
        if r.get("sim_fallback_calls") is None:
            continue
        a = by_arm.setdefault(r["arm"], {"samples": 0, "fallback_calls": 0, "samples_with_fallback": 0,
                                         "contract_errors": 0})
        a["samples"] += 1
        a["fallback_calls"] += r["sim_fallback_calls"]
        a["samples_with_fallback"] += r["sim_fallback_calls"] > 0
        a["contract_errors"] += r["output_contract_error"]
    for arm, a in sorted(by_arm.items()):
        print(f"{arm}: {a}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
