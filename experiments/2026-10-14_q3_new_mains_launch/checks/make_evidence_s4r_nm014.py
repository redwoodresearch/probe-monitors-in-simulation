"""Addendum evidence row for honest_b|nm014_auto_workflow in 4r (09 via d9, 10-15 ~10:50): staged and counted, but 2e's Sol scoring
(@ 1ac2de70d) found 0 scoreable rows at both readouts. From probe_s4r_nm014.py's raw output (read-only on box 4), in the
evidence_s4r_excluded format, plus 09's additions: the final attempt's tool-call counts side by side with 2e's 0 rows, and whether
the transcript ended cleanly (no terminal error, system prompt present, assistant messages > 0). Counts and class names only.
  python3 make_evidence_s4r_nm014.py <raw probe txt>"""
import hashlib
import json
import sys
from pathlib import Path

raw_p = Path(sys.argv[1])
lines = raw_p.read_text().splitlines()
probe = json.loads(next(l for l in reversed(lines) if l.startswith("{")))
(r,) = probe["runs"]
assert r["key"] == "honest_b/nm014_auto_workflow" and r["launch"] == "2026-10-14_newmains_s4r"
fin = r["attempts"][-1]
clean_end = fin.get("terminal_sample_error_class") is None and bool(fin.get("system_prompt_present")) and (fin.get("assistant_steps") or 0) > 0
keep = ("kind", "redraw_key", "redrawn_because", "exit_code", "launcher_error_class", "policy_called", "eval_status", "clean", "policy_calls",
        "policy_calls_usable", "sample_error_classes", "terminal_sample_error_class", "messages", "messages_by_role", "system_prompt_present",
        "assistant_steps", "assistant_with_tool_calls", "tool_calls", "submit_tool_calls", "tool_outputs", "limit", "events", "eval", "eval_sha256",
        "start", "end")
doc = {"for": "honest_b|nm014_auto_workflow (2026-10-14_newmains_s4r): staged and counted; 2e's Sol scoring @ 1ac2de70d found 0 scoreable rows "
              "(no output_end, no call_end); 09's question: a legitimate honest run with zero units, or an errored or empty run?",
       "launch_json_sha256": probe["launch_json_sha256"],
       "record": {"key": "honest_b|nm014_auto_workflow", "launch": r["launch"], "launcher_key": r["key"], "state": r["state"],
                  "final_attempt": fin["i"], "record": fin.get("eval"), "sha256": fin.get("eval_sha256"),
                  "attempts": [{"attempt": a["i"]} | {k: a.get(k) for k in keep} for a in r["attempts"]]},
       "side_by_side": {"tool_calls_in_eval": fin.get("tool_calls"), "assistant_messages_with_tool_calls": fin.get("assistant_with_tool_calls"),
                        "tool_messages": fin.get("tool_outputs"), "submit_tool_calls": fin.get("submit_tool_calls"),
                        "scoreable_rows_per_2e_1ac2de70d": 0, "launcher_policy_calls_usable": fin.get("policy_calls_usable")},
       "ended_cleanly": {"value": clean_end, "no_terminal_error": fin.get("terminal_sample_error_class") is None,
                         "system_prompt_present": fin.get("system_prompt_present"), "assistant_messages": fin.get("assistant_steps")},
       "reading": "for 09; facts only: the run's only tool call is the submit call (submit_tool_calls == tool_calls), and the launcher "
                  "marked its one policy call not usable; whether 0 scoreable rows is expected for a submit-only run is 09's call",
       "probe": {"script": "experiments/2026-10-14_q3_new_mains_launch/checks/probe_s4r_nm014.py", "raw": "results/" + raw_p.name,
                 "raw_sha256": hashlib.sha256(raw_p.read_bytes()).hexdigest(), "box": "i-07bce857a818b6f69 (box4)", "read": lines[0]}}
dest = raw_p.parent / "evidence_s4r_nm014_2026-10-15.json"
dest.write_text(json.dumps(doc, indent=1) + "\n")
print(json.dumps(doc["side_by_side"]), json.dumps(doc["ended_cleanly"]))
print("sha256", hashlib.sha256(dest.read_bytes()).hexdigest())
