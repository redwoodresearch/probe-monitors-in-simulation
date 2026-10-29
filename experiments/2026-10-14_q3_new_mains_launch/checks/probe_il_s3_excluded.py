"""ON BOX 3, read-only (09 via d9, 10-15 ~10:38; IL embargo: counts, class names, booleans only): terminal-error evidence for il-s3's 3 run3
staging exclusions (nm106, nm141, nm219; launch 2026-10-14_il_s3) and their sim2 runs, in probe_s4r_excluded.py's format (itself the chain fork's t434
format (probe_t434.sh @ 3f1061844). Per run: launcher key, task, arm, state; per attempt: kind, redraw key (the attempt's log_dir name),
redrawn_because CLASS, exit code, launcher error CLASS, policy_called, start/end, the launch.json outcome's eval_status, clean, policy-call
count and usable count, sample-error CLASSES; and from the attempt's .eval (path relative to the runs dir, SHA-256): message count, counts
by role, system prompt present, assistant steps, assistant messages with tool calls, tool calls, tool outputs, the sample error CLASS, the
limit type and the event count. Counts, class names and booleans only; never message, command or tool-call text (not even function names).
  <September venv python> probe_il_s3_excluded.py"""
import hashlib
import json
import re
from pathlib import Path

RUNS = Path("/home/aswin/phase2/nm/runs")
TARGETS = [("2026-10-14_il_s3", f"{arm}/{c}") for c in ("nm106_media_processing", "nm141_model_registry", "nm219_slack_clone")
           for arm in ("attack", "sim_attack@september")]
def cls(e):
    if not e:
        return None
    m = re.search(r"raised (\w+)", str(e)) or re.match(r"\s*([A-Za-z_][A-Za-z0-9_.]*)", str(e))
    return m.group(1) if m else "unparsed"



out = {"launch_json_sha256": {}, "runs": []}
for launch, key in TARGETS:
    raw = (RUNS / launch / "launch.json").read_bytes()
    out["launch_json_sha256"][launch] = hashlib.sha256(raw).hexdigest()
    d = json.loads(raw)
    hits = [r for r in d["runs"] if r["key"] == key]
    assert len(hits) == 1, (launch, key, len(hits))
    r = hits[0]
    row = {"launch": launch, "key": key, "task": r["task"] if isinstance(r["task"], str) else r["task"].get("key"), "arm": r.get("arm"),
           "real": r["real"], "state": r["state"], "why": r.get("why"), "attempts": []}
    for i, a in enumerate(r.get("attempts") or []):
        o = a.get("outcome") or {}
        calls = o.get("policy_calls") or []
        ad = {"i": i, "kind": a.get("kind"), "redraw_key": Path(a["log_dir"]).name if a.get("log_dir") else None,
              "redrawn_because": cls(a.get("redrawn_because")), "exit_code": a.get("exit_code"), "launcher_error_class": cls(a.get("error")),
              "policy_called": a.get("policy_called"), "start": a.get("start"), "end": a.get("end"), "eval_status": o.get("eval_status"),
              "clean": o.get("clean"), "policy_calls": len(calls), "policy_calls_usable": sum(1 for c in calls if c.get("usable")),
              "sample_error_classes": [cls(e) for e in (o.get("sample_errors") or [])]}
        evs = sorted(Path(a["log_dir"]).glob("*.eval")) if a.get("log_dir") else []
        ad["evals"] = len(evs)
        if len(evs) == 1:
            ad["eval"] = str(evs[0].relative_to(RUNS))
            ad["eval_sha256"] = hashlib.sha256(evs[0].read_bytes()).hexdigest()
            from inspect_ai.log import read_eval_log
            for smp in (read_eval_log(str(evs[0])).samples or [])[:1]:
                s = json.loads(smp.model_dump_json())
                msgs = s.get("messages") or []
                roles = {}
                for m in msgs:
                    roles[m.get("role")] = roles.get(m.get("role"), 0) + 1
                err = s.get("error")
                ad |= {"messages": len(msgs), "messages_by_role": roles,
                       "system_prompt_present": any(m.get("role") == "system" and m.get("content") for m in msgs),
                       "assistant_steps": sum(1 for m in msgs if m.get("role") == "assistant"),
                       "assistant_with_tool_calls": sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")),
                       "tool_calls": sum(len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant"),
                       "tool_outputs": sum(1 for m in msgs if m.get("role") == "tool"),
                       "terminal_sample_error_class": cls(err.get("message") if isinstance(err, dict) else err),
                       "limit": (s.get("limit") or {}).get("type") if isinstance(s.get("limit"), dict) else s.get("limit"),
                       "events": len(s.get("events") or [])}
        row["attempts"].append(ad)
    out["runs"].append(row)
print(json.dumps(out))
