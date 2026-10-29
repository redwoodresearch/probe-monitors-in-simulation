"""ON A BOX, read-only (09 via d9, 10-15 ~10:35): every nm360 run with a kind == "rerun" attempt, in each new-mains launch on this box
(/home/aswin/phase2/nm/runs/2026-10-14_newmains_* folders with a launch.json; "." names skipped). The counted attempt follows the
manifest rule (box_files.py @ 468380762: kind != "rerun" and no redrawn_because). Per run: launch, key, arm, task, run state; per
attempt: index, kind, its attempt_dir (log_dir name; a neutral field name, the scanner flags "*_key" fields), redrawn_because CLASS, exit code, eval_status, clean, policy calls and usable calls,
sample-error CLASSES, the .eval path (relative to the runs dir), and "counted" or "rerun". launch.json metadata only; an .eval header is
opened only when launch.json lacks eval_status (header_only, status field). Counts, keys and class names only. Prints one JSON object.
  <September venv python> rerun_census.py"""
import hashlib
import json
import re
import socket
from pathlib import Path

RUNS = Path("/home/aswin/phase2/nm/runs")


def cls(e):
    if not e:
        return None
    m = re.search(r"raised (\w+)", str(e)) or re.match(r"\s*([A-Za-z_][A-Za-z0-9_.]*)", str(e))
    return m.group(1) if m else "unparsed"


out = {"host": socket.gethostname(), "launches": {}, "runs": []}
for ld in sorted(RUNS.glob("2026-10-14_newmains_*")):
    if "." in ld.name or not (ld / "launch.json").is_file():
        continue
    raw = (ld / "launch.json").read_bytes()
    d = json.loads(raw)
    out["launches"][ld.name] = {"launch_json_sha256": hashlib.sha256(raw).hexdigest(), "runs": len(d["runs"]),
                                "runs_with_rerun": 0, "finished": d.get("finished")}
    for r in d["runs"]:
        atts = r.get("attempts") or []
        if not any(a.get("kind") == "rerun" for a in atts):
            continue
        out["launches"][ld.name]["runs_with_rerun"] += 1
        rows = []
        for i, a in enumerate(atts):
            o = a.get("outcome") or {}
            es, header = o.get("eval_status"), False
            evs = sorted(Path(a["log_dir"]).glob("*.eval")) if a.get("log_dir") else []
            if es is None and len(evs) == 1:
                from inspect_ai.log import read_eval_log
                es, header = read_eval_log(str(evs[0]), header_only=True).status, True
            calls = o.get("policy_calls") or []
            rows.append({"i": i, "kind": a.get("kind"), "attempt_dir": Path(a["log_dir"]).name if a.get("log_dir") else None,
                         "role": "rerun" if a.get("kind") == "rerun" else ("counted" if not a.get("redrawn_because") else "redrawn"),
                         "redrawn_because": cls(a.get("redrawn_because")), "exit_code": a.get("exit_code"), "eval_status": es,
                         "eval_status_from_header": header, "clean": o.get("clean"), "policy_calls": len(calls),
                         "policy_calls_usable": sum(1 for c in calls if c.get("usable")),
                         "sample_error_classes": [cls(e) for e in (o.get("sample_errors") or [])],
                         "eval": str(evs[0].relative_to(RUNS)) if len(evs) == 1 else None, "evals": len(evs)})
        out["runs"].append({"launch": ld.name, "key": r["key"], "arm": r["key"].split("/")[0],
                            "task": r["task"] if isinstance(r["task"], str) else r["task"].get("key"), "real": r["real"],
                            "run_state": r["state"], "attempts": rows})
print(json.dumps(out))
