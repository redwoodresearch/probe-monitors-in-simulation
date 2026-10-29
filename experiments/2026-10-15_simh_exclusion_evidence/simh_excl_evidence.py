"""Evidence rows for sim-honest cells the stager excluded (09's rulings §6 @ 0f9927f3e; d9 10-15 ~12:40): for each cell,
from the ORIGINAL run across all its attempts, the metadata 09's 25e / Append 35 classification reads. Counts, booleans,
enumerated labels and error CLASS names only: no transcript text, no error message, no free-form field value.

  box:    run ON the box over devbox.sh, read-only, with the sims venv python (inspect_ai reads the zstd .eval logs);
          stdout is captured on the LAPTOP (nothing is written on the box):
          PYTHONDONTWRITEBYTECODE=1 ~/sims/simulated-control-evals-sept/.venv/bin/python - box <run dir> <cell>... < this file
  census: on the laptop, add each run's rows from one committed census file (per_box[<box>][<launch>].rows):
          python3 simh_excl_evidence.py census <box.json> <census.json> <census path @ sha> <box name>

Per run (exactly one `sim_honest@...` run per cell, else fail): key, state; per attempt: kind, the redraw fields present
(key names containing "redraw" or "because", with only their value's type), start, end, exit_code, policy_called, outcome
(eval_status and sample_limits as enumerated labels, clean, samples, sample_errors and errors as class names, unreadable
count, usable_reference, and per sample the policy-call summary: steps = completed policy calls, i.e. assistant steps,
step_limit_hits, cap_hits, failed_calls, timeouts, retried_calls, usable, failed_after_start, ended as a label); the gate12
basics; the run's usable-attempt count; every .eval under the attempts' log dirs (path, sha256, size; the newest marked
final) with per sample messages by role, assistant messages with tool calls, assistant_tool_calls and how many are submit (submit_tool_calls; counts only, no
tool name is emitted), events by type, tool_events, tool_events_submit and model_events (named: tool_events is 09's m_r),
error class, limit type; and per cell tool_events_final (the final .eval's tool events).
Any failure prints one line {"failed": "<exception class>"} and exits 2: no exception text, no traceback.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections
import contextlib
import hashlib
import io
import json
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
CLS = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Timeout|Exceeded|Exit))\b")
WRAPPED = re.compile(r"\braised ([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Timeout|Exceeded))\b")  # tenacity's RetryError text
EVAL_STATUS = {"success", "error", "cancelled", "started"}
LIMIT_TYPES = {"message", "token", "time", "working", "custom", "context", "operator", "cost"}
RUN_KEY = re.compile(r"sim_honest@[a-z0-9_]+/(?:t|nm)[0-9]+_[a-z0-9_]+")  # +311/T196 cells t<n>_, nm360 cells nm<n>_
SAFE_PATH = re.compile(r"/home/aswin/phase2/runs/[A-Za-z0-9_./@:+-]+\.eval")
REDRAW_KEYS = {"redrawn_because"}  # run_pilot.py:444, the only redraw field the launcher writes
ENDED = {"policy_timeout", "out_of_context", "time limit", "error", "submit", "step limit or no submit"} | {
    f"{t} limit" for t in LIMIT_TYPES}
CALL_INTS = ("steps", "step_limit_hits", "cap_hits", "failed_calls", "timeouts", "retried_calls", "usable",
             "failed_after_start")


class EvidenceError(Exception):
    """A required piece of evidence is missing or malformed (the name says which; no values are carried)."""


def cls(x) -> str:
    """The leading error class name of a message (never the message itself)."""
    s = x if isinstance(x, str) else x.get("message") if isinstance(x, dict) else getattr(x, "message", None)
    m = CLS.match(s) if isinstance(s, str) else None
    if m and m.group(1).split(".")[-1] == "RetryError":  # name what it wraps (09: RetryError wrapping RateLimitError)
        w = WRAPPED.search(s)
        return f"{m.group(1)}[{w.group(1) if w else '<unwrapped>'}]"
    return m.group(1) if m else ("<none>" if x is None else "<unclassified>")


def label(x, allowed: set) -> str:
    return x if isinstance(x, str) and x in allowed else "<other>"


def nonneg_int(x):
    return x if isinstance(x, int) and not isinstance(x, bool) and x >= 0 else None


def ts(x):
    """An offset-aware ISO time, normalized; anything else is None."""
    try:
        t = datetime.fromisoformat(x) if isinstance(x, str) else None
    except ValueError:
        return None
    return t.isoformat() if t is not None and t.tzinfo is not None else None


def gate12(o: dict) -> bool:
    return (bool(o.get("clean")) and o.get("samples") == 1 and not o.get("sample_errors") and not o.get("sample_limits")
            and not o.get("unreadable"))


def call_summary(c) -> dict:
    if not isinstance(c, dict):
        raise EvidenceError("policy_calls entry not a dict")
    return {k: nonneg_int(c.get(k)) for k in CALL_INTS} | {"ended": label(c.get("ended"), ENDED)}


def attempt_evals(d) -> tuple[str, list[str]]:
    """One attempt's log dir: ("missing" | "empty" | "ok", its .eval files by mtime). An unreadable dir or a failed walk
    raises (a failed discovery is never read as an absent artifact)."""
    if not isinstance(d, str) or not d:
        return "missing", []
    try:
        os.stat(d)  # only a confirmed absence is "missing"; any other OS error propagates (Codex r21)
    except FileNotFoundError:
        return "missing", []
    if not (os.path.isdir(d) and os.access(d, os.R_OK | os.X_OK)):
        raise EvidenceError("attempt log dir unreadable")
    found = []

    def boom(err):
        raise EvidenceError("log dir walk failed")
    for root, _, files in os.walk(d, onerror=boom):
        found += [os.path.join(root, f) for f in files if f.endswith(".eval")]
    if any(not SAFE_PATH.fullmatch(f) for f in found):
        raise EvidenceError("unexpected .eval path")
    return ("ok" if found else "empty"), sorted(found, key=os.path.getmtime)


def eval_counts(path: str) -> dict:
    from inspect_ai.log import read_eval_log  # the sims venv; box mode only

    log = read_eval_log(path)
    samples = [{
        "messages_by_role": dict(collections.Counter(str(m.role) for m in s.messages)),
        "assistant_messages": sum(1 for m in s.messages if m.role == "assistant"),
        "assistant_with_tool_calls": sum(1 for m in s.messages if m.role == "assistant" and getattr(m, "tool_calls", None)),
        "events_by_type": dict(collections.Counter(str(e.event) for e in s.events)),
        "tool_events": sum(1 for e in s.events if e.event == "tool"),  # 09's m_r (note @ ed7932987), named explicitly
        # counts only, never names (09, nm014): all assistant tool calls, and how many of them / of the tool events are submit
        "assistant_tool_calls": sum(len(getattr(m, "tool_calls", None) or []) for m in s.messages if m.role == "assistant"),
        "submit_tool_calls": sum(1 for m in s.messages if m.role == "assistant" for tc in getattr(m, "tool_calls", None) or []
                                 if getattr(tc, "function", None) == "submit"),
        "tool_events_submit": sum(1 for e in s.events if e.event == "tool" and getattr(e, "function", None) == "submit"),
        "model_events": sum(1 for e in s.events if e.event == "model"),
        "error_class": cls(s.error) if s.error is not None else None,
        "error_events": sum(1 for e in s.events if e.event == "error"),
        "limit_type": label(getattr(s.limit, "type", None), LIMIT_TYPES) if s.limit else None,
    } for s in log.samples or []]
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"path": path, "sha256": sha, "size": os.path.getsize(path), "status": label(log.status, EVAL_STATUS),
            "n_samples": len(samples), "samples": samples}


def box(run_dir: str, cells: list[str]) -> dict:
    launch = json.load(open(os.path.join(run_dir, "launch.json")))
    out = {"run_dir": run_dir, "launch_finished": bool(launch.get("finished")), "cells": []}
    for p in cells:
        runs = [r for r in launch["runs"] if r.get("arm") == "sim_honest" and str(r.get("key", "")).startswith("sim_honest@")
                and r["key"].split("/", 1)[-1].startswith(p + "_")]
        if len(runs) != 1:
            raise EvidenceError("cell does not match exactly one sim_honest run")
        r = runs[0]
        if not RUN_KEY.fullmatch(r["key"]):
            raise EvidenceError("unexpected run key form")
        atts, all_evals = [], []
        for a in r.get("attempts") or []:
            o = a.get("outcome") or {}
            pcs = o.get("policy_calls") or []
            if not isinstance(pcs, list):
                raise EvidenceError("policy_calls not a list")
            atts.append({
                "kind": label(a.get("kind"), {"first", "redraw", "rerun"}),  # run_pilot.py:445 / 443 / 860
                "prestart_class": cls(a.get("prestart")) if a.get("prestart") is not None else None,
                "redraw_fields_present": {k: type(a[k]).__name__ for k in a if k in REDRAW_KEYS},
                "other_redraw_like_keys": sum(1 for k in a if ("redraw" in k or "because" in k) and k not in REDRAW_KEYS),
                "start": ts(a.get("start")), "end": ts(a.get("end")), "exit_code": a.get("exit_code") if isinstance(a.get("exit_code"), int) else None,
                "policy_called": a.get("policy_called") if isinstance(a.get("policy_called"), bool) else None,
                "eval_status": [label(x, EVAL_STATUS) for x in o.get("eval_status") or []],
                "clean": o.get("clean") if isinstance(o.get("clean"), bool) else None, "samples": nonneg_int(o.get("samples")),
                "sample_errors": [cls(e) for e in o.get("sample_errors") or []],
                "errors": [cls(e) for e in o.get("errors") or []],
                "sample_limits": [label(x, LIMIT_TYPES) for x in o.get("sample_limits") or []],
                "unreadable": len(o.get("unreadable") or []),
                "usable_reference": o.get("usable_reference") if isinstance(o.get("usable_reference"), bool) else None,
                "policy_call_summaries": [call_summary(c) for c in pcs],
                "gate12_basics": gate12(o)})
            state, files = attempt_evals(a.get("log_dir"))
            atts[-1]["log_dir_state"], atts[-1]["eval_files"] = state, len(files)
            atts[-1]["final_eval_of_attempt"] = files[-1] if files else None
            all_evals += [f for f in files if f not in all_evals]
        final = atts[-1]["final_eval_of_attempt"] if atts else None
        out["cells"].append({"cell": p, "run": r["key"], "state": label(r.get("state"), {"done", "not run", "no policy call",
                             "running", "ready", "waiting"}), "n_attempts": len(atts),
                             "usable_attempts": sum(x["gate12_basics"] for x in atts),
                             "run_usable_gate12": r.get("state") == "done" and bool(atts) and atts[-1]["gate12_basics"],
                             "final_eval": final,  # the newest .eval of the LAST attempt (None: that attempt left none)
                             # 09's m_r: tool events in the final .eval's sample(s) (None if no final .eval)
                             "tool_events_final": None,
                             "attempts": atts, "evals": [eval_counts(f) | {"final": f == final} for f in all_evals]})
        fin = [e for e in out["cells"][-1]["evals"] if e["final"]]
        out["cells"][-1]["tool_events_final"] = sum(x["tool_events"] for x in fin[0]["samples"]) if fin else None
    return out


def pt_bin(end) -> str | None:
    """The census's own bin: the attempt end's PT HH:M0 (sim_error_census.py), from an offset-aware time."""
    if not end:
        return None
    t = datetime.fromisoformat(end)
    if t.tzinfo is None:
        raise EvidenceError("census row end without an offset")
    t = t.astimezone(PT)
    return f"{t:%H}:{t.minute // 10}0"


def census(box_json: str, census_json: str, census_ref: str, box_name: str) -> dict:
    """Census fields only, keyed by the box file's cells and bound to that file's sha256 (the box JSON is not re-emitted)."""
    raw = open(box_json, "rb").read()
    b = json.loads(raw)
    c = json.load(open(census_json))
    try:
        ctime = datetime.strptime(c["census_time_pt"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except (KeyError, TypeError, ValueError):
        raise EvidenceError("census time missing or malformed") from None
    launch = os.path.basename(str(b.get("run_dir", "")).rstrip("/"))
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[a-z0-9_]+", launch):
        raise EvidenceError("unexpected launch name")
    rec = (c["per_box"].get(box_name) or {}).get(launch)
    if not isinstance(rec, dict) or "error" in rec or not isinstance(rec.get("rows"), list):
        raise EvidenceError("launch missing from the census, or a census read error")
    out = {"box_json_sha256": hashlib.sha256(raw).hexdigest(), "cells": []}
    for cell in b["cells"]:
        if not RUN_KEY.fullmatch(str(cell.get("run"))):
            raise EvidenceError("unexpected run key form")
        hit = [x for x in rec["rows"] if isinstance(x, dict) and x.get("key") == cell["run"]]
        out["cells"].append({"cell": cell["run"], "census": {"ref": census_ref, "census_time_pt": ctime, "box": box_name, "launch": launch,
                          "launch_sim_ended": nonneg_int(rec.get("sim_ended")),
                          # the census lists only runs whose last attempt errored: no row means none was recorded
                          "error_row_present": bool(hit),
                          "rows": [{"class": cls(x.get("class")), "attempts": nonneg_int(x.get("attempts")),
                                    "end": ts(x.get("end")), "bin": pt_bin(ts(x.get("end")))} for x in hit]}})
    return out


def main() -> None:
    mode = sys.argv[1]
    if mode == "box":
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):  # dependency output is dropped
            res = box(sys.argv[2], sys.argv[3:])
    elif mode == "census":
        res = census(*sys.argv[2:6])
    else:
        raise EvidenceError("mode")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa: BLE001 - fail loud by class only: no message, no traceback (they can carry values)
        print(json.dumps({"failed": type(e).__name__, "detail": e.args[0] if isinstance(e, EvidenceError) and e.args else None}))
        sys.exit(2)
