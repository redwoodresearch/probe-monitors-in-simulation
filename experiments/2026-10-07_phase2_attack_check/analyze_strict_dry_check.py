"""The end-to-end half of the strict-tools dry check: every run of a dry-check launch (config/dry_check_strict.yaml),
read from its .eval logs on the devbox. $0, read-only. Writes one JSON line per run and prints a verdict:

  analyze_strict_dry_check.py <run dir> <out.jsonl>

Per run (each attempt's log): the policy's requests (strict flag on every tool, from the logged raw request, with
Inspect's attachment:// references resolved), its str_replace calls (with or without new_str, from the raw reply),
whether each str_replace's tool output shows its edit (new_str's first non-blank line not in old_str), insert calls,
sample errors, and the policy's model-call count. Passes when every request in every run has every tool strict false,
every str_replace has a non-empty new_str (an empty one fails: read its raw reply), every str_replace that didn't
error shows its edit and a run's str_replace calls don't all error (errors are reported, never a success), every
call's arguments parse, every completed policy call's raw request and reply were logged, every editor call came from
an unforced request (a forced tool_choice constrains decoding whatever strict says; every request's tool_choice is
reported), every unsuperseded attempt left a log, every run made policy calls, no run called an Anthropic model, and
every simulated run made at least one simulator call with token usage (today's simulator included). Errors (a tool
error, a sample error) are counted separately, not as passes or failures.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log


def resolved(value, attachments: dict):  # noqa: ANN001, ANN201
    if isinstance(value, str) and value.startswith("attachment://"):
        return attachments[value.removeprefix("attachment://")]
    if isinstance(value, dict):
        return {k: resolved(v, attachments) for k, v in value.items()}
    if isinstance(value, list):
        return [resolved(v, attachments) for v in value]
    return value


def edit_line(args: dict) -> str:
    """The line an output showing this edit must contain: new_str's first non-blank line not in old_str (stripped),
    else its first line not in old_str as written (an indentation change), else its first non-blank line (a reorder)."""
    new_lines = [line for line in str(args.get("new_str") or "").splitlines() if line.strip()]
    old_raw = str(args.get("old_str") or "").splitlines()
    old = {line.strip() for line in old_raw}
    return (next((x.strip() for x in new_lines if x.strip() not in old), None)
            or next((x.rstrip() for x in new_lines if x not in old_raw), None) or new_lines[0].strip())


def one(path: Path, policy: str, simulators: set[str]) -> dict:
    sample = read_eval_log(path, resolve_attachments=True).samples[0]  # tool outputs too, not attachment://
    att = sample.attachments or {}
    requests, strict_flags, calls, policy_calls, missing_raw, forced_editor = 0, set(), {}, 0, 0, 0
    tool_choices, other_models = Counter(), Counter()
    for event in sample.events:
        if getattr(event, "event", "") != "model":
            continue
        if event.model != policy:  # the simulator (and anything else): calls with usage, by model
            usage = event.output.usage if event.output else None
            other_models[(event.model, bool(usage and (usage.input_tokens or usage.output_tokens)))] += 1
            continue
        policy_calls += 1
        call = event.call
        completed = event.error is None and not event.pending
        if not call or not isinstance(call.request, dict) or (completed and not isinstance(call.response, dict)):
            missing_raw += completed  # a completed call whose raw request or reply wasn't logged: unchecked
            continue
        requests += 1
        choice = json.dumps(call.request.get("tool_choice"))
        tool_choices[choice] += 1
        for tool in call.request.get("tools") or []:
            strict_flags.add(tool.get("function", {}).get("strict"))
        reply = resolved(call.response, att) if isinstance(call.response, dict) else {}
        reply_calls = ((reply.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
        # A forced tool_choice constrains decoding whatever strict says: no editor call may come from one.
        forced_editor += choice not in ('"auto"', "null") and any(
            tc.get("function", {}).get("name") == "text_editor" for tc in reply_calls)
        for tc in reply_calls:
            try:
                calls[tc["id"]] = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError, KeyError):
                calls[tc.get("id")] = None
    # Tool results from the final messages, then from later policy calls' inputs (an interrupted run's messages can
    # be incomplete). A result found nowhere is "unresolved": a failure, never an error.
    outputs = {m.tool_call_id: (m.text, m.error) for m in sample.messages if getattr(m, "role", "") == "tool"}
    for event in sample.events:
        if getattr(event, "event", "") == "model" and event.model == policy:
            for m in event.input or []:
                if getattr(m, "role", "") == "tool" and m.tool_call_id not in outputs:
                    outputs[m.tool_call_id] = (m.text, m.error)
    replaces = [(cid, a) for cid, a in calls.items() if a and a.get("command") == "str_replace"]
    shown, empty, tool_errors, unresolved = 0, 0, 0, 0
    for cid, a in replaces:
        if not isinstance(a.get("new_str"), str):
            continue  # missing new_str: the bug itself, failed by str_replace_with_new_str and never shown
        if not a["new_str"].strip():
            empty += 1  # an empty replacement, errored or not: the deletion this check exists to catch (read the raw)
            continue
        if cid not in outputs:
            unresolved += 1
            continue
        text, error = outputs[cid]
        if error:
            tool_errors += 1
            continue
        # The edit's first CHANGED line: new_str's first non-blank line that isn't in old_str. An output that shows
        # the edit (a snippet or a diff of the changed region) contains it; new_str's first line may be unchanged
        # context outside a diff's window (3 such simulated outputs in the 10-08 dry check's first read).
        shown += edit_line(a) in text
    return {"log": str(path), "policy_calls": policy_calls, "requests_logged": requests,
            "strict_flags": sorted(map(str, strict_flags)),
            "str_replace": len(replaces), "str_replace_with_new_str": sum(isinstance(a.get("new_str"), str)
                                                                          for _, a in replaces),
            "str_replace_text_shown": shown, "str_replace_empty_new_str": empty, "str_replace_tool_errors": tool_errors,
            "str_replace_unresolved": unresolved,
            "insert": sum(1 for a in calls.values() if a and a.get("command") == "insert"),
            "calls_missing_raw": missing_raw, "tool_choice": dict(tool_choices), "editor_calls_forced": forced_editor,
            "other_model_calls": {f"{m}|usage={u}": n for (m, u), n in other_models.items()},
            "anthropic_calls": sum(n for (m, _), n in other_models.items() if str(m).startswith("anthropic/")),
            # The simulator itself (PT_SIM_MODEL or its fallback), not its router or refusal judges.
            "simulator_calls_with_usage": sum(n for (m, u), n in other_models.items() if u and m in simulators),
            "unparsed_calls": sum(a is None for a in calls.values()),
            "sample_error": (sample.error.message[:300] if sample.error else None)}


def main(run_dir: str, out: str) -> None:
    launch = json.loads((Path(run_dir) / "launch.json").read_text())
    policy = launch["policy"] if isinstance(launch["policy"], str) else launch["config"]["policy"]
    sim_env = (launch.get("sim_config") or {}).get("env") or {}
    simulators = {sim_env[k] for k in ("PT_SIM_MODEL", "PT_SIM_FALLBACK_MODEL") if sim_env.get(k)}
    assert simulators, "launch.json's sim_config names no simulator model"
    rows, missing = [], []
    for run in launch["runs"]:
        if not run["attempts"]:  # a run the launch planned but never started: the check can't cover it
            missing.append(f"{run['arm']}@{run.get('simulator')} ({run['state']}): no attempt")
        for attempt in run["attempts"]:  # every attempt that wasn't superseded by a redraw must have left a log
            if not attempt.get("redrawn_because") and not list(Path(attempt["log_dir"]).glob("*.eval")):
                missing.append(f"{run['arm']}@{run.get('simulator')} {attempt['kind']}: no log")
        for attempt in run["attempts"]:
            if attempt.get("redrawn_because"):
                continue  # superseded by a redraw: a pre-start failure, not part of the verdict
            for path in sorted(Path(attempt["log_dir"]).glob("*.eval")):
                row = {"arm": run["arm"], "simulator": run.get("simulator"), "kind": attempt["kind"]}
                row |= one(path, policy, simulators)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "log"}), flush=True)
    with open(out, "x") as f:  # exclusive: a check's output is never overwritten
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    # The live check must have exercised the bug's own call: at least one str_replace across the launch (the rerun's
    # canary gate also requires a real one with new_str before anything else starts).
    ok = rows and not missing and sum(r["str_replace"] for r in rows) > 0 and all(
        r["strict_flags"] == ["False"] and r["str_replace"] == r["str_replace_with_new_str"]
        # every str_replace that didn't error shows its edit; none is empty; errors are reported, never a success,
        # and a run whose every str_replace errored fails
        and r["str_replace_empty_new_str"] == 0 and r["str_replace_unresolved"] == 0
        and r["str_replace_text_shown"]
        == r["str_replace"] - r["str_replace_tool_errors"] - r["str_replace_empty_new_str"]
        and (r["str_replace"] == 0 or r["str_replace_text_shown"] > 0)
        and r["policy_calls"] > 0 and r["unparsed_calls"] == 0 and r["calls_missing_raw"] == 0
        and r["editor_calls_forced"] == 0 and r["anthropic_calls"] == 0
        and (not str(r["arm"]).startswith("sim_") or r["simulator_calls_with_usage"] > 0) for r in rows)
    print(f"logs: {len(rows)}; runs without a log: {missing or 'none'}; verdict: {'PASS' if ok else 'FAIL'} -> {out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
