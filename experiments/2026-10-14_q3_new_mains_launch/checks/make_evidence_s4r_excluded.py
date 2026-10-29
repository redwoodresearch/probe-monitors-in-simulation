"""Evidence rows (t434 format) for the 5 s4r runs the stager excluded (09 via d9, 10-15 ~10:25), from probe_s4r_excluded.py's raw output
(results/evidence_s4r_excluded_probe_2026-10-15.txt, read-only on box 4). One record per run with every attempt's fields; the class is left
to 09 (the row states only whether a RateLimitError appears). Counts, class names and booleans only.
  python3 make_evidence_s4r_excluded.py <raw probe txt>"""
import hashlib
import json
import sys
from pathlib import Path

raw_p = Path(sys.argv[1])
lines = raw_p.read_text().splitlines()
probe = json.loads(next(l for l in reversed(lines) if l.startswith("{")))
WORLD = {"attack": "run1", "honest_a": "honest_a", "honest_b": "honest_b", "sim_attack@september": "sim"}
recs = []
for r in probe["runs"]:
    arm = r["key"].split("/")[0]
    fin = r["attempts"][-1]
    cl = sorted({c for a in r["attempts"] for c in (a.get("sample_error_classes") or []) + [a.get("terminal_sample_error_class")] if c})
    recs.append({
        "record": {"batch": "nm360-s4r" if r["launch"].endswith("_s4r") else "nm360-s4r_fu", "world": WORLD[arm], "arm": arm,
                   "cell": r["task"], "launch_task": r["task"], "launch": r["launch"], "launcher_key": r["key"],
                   "launcher_redraw_key": fin.get("redraw_key"), "state": r["state"], "why": r.get("why"),
                   "policy_called": fin.get("policy_called"), "record": fin.get("eval"), "sha256": fin.get("eval_sha256"),
                   "final_attempt": fin["i"], "staging_exclusion": True,
                   "attempts": [{"attempt": a["i"], "kind": a.get("kind"), "launcher_redraw_key": a.get("redraw_key"),
                                 "redrawn_because": a.get("redrawn_because"), "exit_code": a.get("exit_code"),
                                 "launcher_error_class": a.get("launcher_error_class"), "policy_called": a.get("policy_called"),
                                 "eval_status": a.get("eval_status"), "clean": a.get("clean"), "policy_calls": a.get("policy_calls"),
                                 "policy_calls_usable": a.get("policy_calls_usable"), "sample_error_classes": a.get("sample_error_classes"),
                                 "terminal_sample_error_class": a.get("terminal_sample_error_class"), "messages": a.get("messages"),
                                 "messages_by_role": a.get("messages_by_role"), "assistant_steps": a.get("assistant_steps"),
                                 "assistant_with_tool_calls": a.get("assistant_with_tool_calls"), "tool_calls": a.get("tool_calls"),
                                 "tool_outputs": a.get("tool_outputs"), "system_prompt_present": a.get("system_prompt_present"),
                                 "limit": a.get("limit"), "events": a.get("events"), "eval": a.get("eval"), "eval_sha256": a.get("eval_sha256"),
                                 "start": a.get("start"), "end": a.get("end")} for a in r["attempts"]],
                   "error_classes_seen": cl,
                   "class": ("RateLimitError present" if "RateLimitError" in cl else f"not RateLimitError ({', '.join(cl) or 'no error class'})")
                            + "; class for 09"}})
doc = {"for": "the 5 s4r runs the stager excluded (09 via d9, 10-15 ~10:25): one terminal-error evidence row per run, t434 format",
       "launch_json_sha256": probe["launch_json_sha256"], "records": recs,
       "probe": {"script": "experiments/2026-10-14_q3_new_mains_launch/checks/probe_s4r_excluded.py", "raw": "results/" + raw_p.name,
                 "raw_sha256": hashlib.sha256(raw_p.read_bytes()).hexdigest(), "box": "i-07bce857a818b6f69 (box4)", "read": lines[0]},
       "counts": {"runs": len(recs), "attempts": sum(len(r["record"]["attempts"]) for r in recs)}}
assert doc["counts"]["runs"] == 5
dest = raw_p.parent / "evidence_s4r_excluded_2026-10-15.json"
dest.write_text(json.dumps(doc, indent=1) + "\n")
for r in recs:
    x = r["record"]
    print(x["launch"], x["launcher_key"], "final attempt", x["final_attempt"], x["launcher_redraw_key"], "|", x["class"])
print("sha256", hashlib.sha256(dest.read_bytes()).hexdigest())
