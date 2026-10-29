# Read-only evidence probe (09 via d9 10-15 ~08:05): il_q4c sim2|t411_media_processing and sim2|t501_user_tracking (staged out; d9 via 09 10-15 ~12:10). Prints launcher states, attempt fields,
# error CLASS names and per-attempt message counts by role and booleans only; never prints message or command text.
date '+probe at %Y-%m-%d %H:%M:%S %Z'; hostname
PY=$(ls /home/aswin/sims/*/.venv/bin/python 2>/dev/null | head -1); echo "python: $PY"; $PY - <<'PY'
import json, os, glob, zipfile, re, hashlib
R = os.path.expanduser("~/phase2/runs/2026-10-15_il_q4c"); CELLS = ("t411_media_processing", "t501_user_tracking")
d = json.load(open(f"{R}/launch.json"))
cls = lambda e: (re.match(r"\s*([A-Za-z_][A-Za-z0-9_.]*)", str(e)).group(1) if e else None)
out = []
for r in d["runs"]:
    if r["task"] not in CELLS and not any(str(r.get("key", "")).endswith(c_) for c_ in CELLS):
        continue
    row = {"key": r.get("key"), "task": r["task"], "arm": r.get("arm"), "state": r.get("state"), "attempts": []}
    for i, a in enumerate(r.get("attempts") or []):
        ad = {"i": i, "kind": a.get("kind"), "redrawn_because": cls(a.get("redrawn_because")), "exit_code": a.get("exit_code"),
              "error_class": cls(a.get("error")), "policy_called": a.get("policy_called"), "start": a.get("start"), "end": a.get("end")}
        rel = (a.get("log_dir") or "").split("2026-10-15_il_q4c/", 1)[-1]
        evs = sorted(glob.glob(f"{R}/{rel}/*.eval")) if rel else []
        ad["evals"] = len(evs)
        for ev in evs[:1]:
            ad["eval_sha256"] = hashlib.sha256(open(ev, "rb").read()).hexdigest()
            from inspect_ai.log import read_eval_log
            log = read_eval_log(ev)
            for smp in (log.samples or [])[:1]:
                    s = json.loads(smp.model_dump_json())
                    msgs = s.get("messages") or []
                    roles = {}
                    for m in msgs:
                        roles[m.get("role")] = roles.get(m.get("role"), 0) + 1
                    ad["messages"] = len(msgs); ad["by_role"] = roles
                    ad["system_prompt_present"] = any(m.get("role") == "system" and m.get("content") for m in msgs)
                    ad["assistant_with_tool_calls"] = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
                    ad["assistant_messages"] = sum(1 for m in msgs if m.get("role") == "assistant")
                    ad["tool_calls"] = sum(len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant")
                    ad["tool_outputs"] = sum(1 for m in msgs if m.get("role") == "tool")
                    ad["tool_call_function_names"] = [tc.get("function") for m in msgs if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]  # names only
                    ad["tool_output_error_types"] = [(m.get("error") or {}).get("type") if isinstance(m.get("error"), dict) else None for m in msgs if m.get("role") == "tool"]
                    err = s.get("error")
                    ad["sample_error_class"] = cls(err.get("message") if isinstance(err, dict) else err)
                    ad["limit"] = (s.get("limit") or {}).get("type") if isinstance(s.get("limit"), dict) else s.get("limit")
                    ad["events"] = len(s.get("events") or [])
        row["attempts"].append(ad)
    out.append(row)
print(json.dumps(out))
PY
