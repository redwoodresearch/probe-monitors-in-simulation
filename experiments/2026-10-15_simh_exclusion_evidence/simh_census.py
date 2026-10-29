"""Sim-honest base-rate listing and EmptyPolicyCompletionError event pattern (d9 ~15:40, for 09 / 0d). ON a box, read-only,
stdout captured on the laptop. Identifiers, booleans and integers only: no transcript, tool-output or message text.

  list (any python3; reads launch.json only):
      python3 - list <run dir> < simh_census.py
    per sim_honest run: run key, cell, env and main (from launch.json `tasks`, the entry whose key is the cell; identifier-
    validated), state, ended (state "done" or "no policy call"), the last attempt's sample_errors and errors as CLASS names
    (RetryError named with the class it wraps), clean, completed_ok (done, clean, one sample, no sample error or limit).
  epce (sims venv; reads the final .eval):
      PYTHONDONTWRITEBYTECODE=1 ~/sims/simulated-control-evals-sept/.venv/bin/python - epce <run dir> <cell>... < simh_census.py
    per cell: the last 10 events as {type, tool (tool events), empty (model events: no text and no tool calls)}; the
    trailing run of empty model completions; all empty model completions; the last model event's input tokens if
    recorded; the sample store's EmptyCompletionCounter streak if recorded.
Failures print {"failed": <class>, "detail": <fixed string or null>} and exit 2 (fds 1/2 point at /dev/null during the work).
"""

import json
import os
import re
import sys

sys.dont_write_bytecode = True

CLS = re.compile(r"((?:[A-Za-z_][A-Za-z0-9_.]*)?(?:Error|Exception|Timeout|Exceeded|Exit))(?=[:(\[]|$)")
# tenacity's repr, matched whole (Codex r34): no class-shaped word is ever searched out of free text
RETRY = re.compile(r"(?:tenacity\.)?RetryError\[<Future at 0x[0-9a-f]+ state=finished raised ([A-Za-z_][A-Za-z0-9_.]*)>\]")
COUNTING = re.compile(r"counting [^:]*: (.*)", re.S)  # the launcher's errors[] entries: counting <file>: <exception repr>
RUN_KEY = re.compile(r"sim_honest@[a-z0-9_]+/(?:t|nm)[0-9]+_[a-z0-9_]+")
RUN_DIR = re.compile(r"/home/aswin/phase2/runs/[0-9]{4}-[0-9]{2}-[0-9]{2}_[a-z0-9_]+")
STATES = {"done", "no policy call", "not run", "running", "ready", "waiting", "stopped"}
IDENT = re.compile(r"[a-z0-9_]{1,80}")
TOOL = re.compile(r"[a-z_][a-z0-9_]{0,40}")
KIND = re.compile(r"[a-z_]{1,30}")
CELL = re.compile(r"(?:t|nm)[0-9]+_[a-z0-9_]+")
MODEL = re.compile(r"[A-Za-z0-9_./:-]{1,80}")  # a model NAME (configuration), never its output


class CensusError(Exception):
    pass


def cls(x) -> str:
    s = x if isinstance(x, str) else None
    if s is None:
        return "<unclassified>"
    c = COUNTING.fullmatch(s)
    if c:
        s = c.group(1)
    rt = RETRY.fullmatch(s.strip())
    if rt:
        return f"RetryError[{rt.group(1)}]"
    m = CLS.match(s)
    if not m:
        return "<unclassified>"
    if m.group(1).split(".")[-1] == "RetryError":
        return "RetryError[<unwrapped>]"
    return m.group(1)


def ident(x):
    return x if isinstance(x, str) and IDENT.fullmatch(x) else None


def runs_of(run_dir: str):
    launch = json.load(open(os.path.join(run_dir, "launch.json")))
    tasks = {t.get("key"): t for t in (launch.get("tasks") or []) if isinstance(t, dict)}
    out = []
    for r in launch["runs"]:
        if r.get("arm") != "sim_honest" or not str(r.get("key", "")).startswith("sim_honest@"):
            continue
        if not RUN_KEY.fullmatch(r["key"]):
            raise CensusError("unexpected run key")
        cell = r["key"].split("/", 1)[1]
        out.append((r, cell, tasks.get(cell) or {}))
    return launch, out


def list_mode(run_dir: str) -> dict:
    launch, runs = runs_of(run_dir)
    rows = []
    for r, cell, t in runs:
        att = (r.get("attempts") or [{}])[-1]
        o = att.get("outcome") or {}
        state = r.get("state") if r.get("state") in STATES else "<other>"
        rows.append({"run": r["key"], "cell": cell, "env": ident(t.get("env")), "main": ident(t.get("main")), "state": state,
                     "ended": state in ("done", "no policy call"), "n_attempts": len(r.get("attempts") or []),
                     "sample_errors": [cls(e) for e in o.get("sample_errors") or []], "errors": [cls(e) for e in o.get("errors") or []],
                     "clean": o.get("clean") if isinstance(o.get("clean"), bool) else None,
                     "completed_ok": state == "done" and o.get("clean") is True and o.get("samples") == 1
                     and not o.get("sample_errors") and not o.get("sample_limits")})
    return {"run_dir": run_dir, "finished": bool(launch.get("finished")), "n_runs": len(rows), "runs": rows}


def checked_dir(d: str) -> str:
    if not RUN_DIR.fullmatch(d):
        raise CensusError("unexpected run dir")
    return d


def epce_mode(run_dir: str, cells: list) -> dict:
    from inspect_ai.log import read_eval_log

    _, runs = runs_of(run_dir)
    out = []
    for c in cells:
        hit = [(r, cell) for r, cell, _ in runs if cell.startswith(c + "_")]
        if len(hit) != 1:
            raise CensusError("cell does not match exactly one sim_honest run")
        r, cell = hit[0]
        att = (r.get("attempts") or [None])[-1]
        if not att or not att.get("log_dir"):
            raise CensusError("no last attempt with a log dir")
        evals = sorted((os.path.join(dp, f) for dp, _, fs in os.walk(att["log_dir"]) for f in fs if f.endswith(".eval")),
                       key=os.path.getmtime)
        if not evals:
            raise CensusError("no .eval for the last attempt")
        log = read_eval_log(evals[-1])
        if len(log.samples or []) != 1:
            raise CensusError("not exactly one sample")
        s = log.samples[0]

        def status(e):
            """'failed' (error set), 'pending', 'no_choice', 'empty' (completed, no text, no tool calls) or 'ok'
            (Codex r34: inspect's blank placeholder on failed / pending events is not an empty completion)."""
            if getattr(e, "error", None) is not None:
                return "failed"
            if getattr(e, "pending", None):
                return "pending"
            ch = getattr(getattr(e, "output", None), "choices", None) or []
            if not ch:
                return "no_choice"
            msg = getattr(ch[0], "message", None)
            text = getattr(msg, "text", "") or ""
            return "empty" if not text.strip() and not (getattr(msg, "tool_calls", None) or []) else "ok"

        def kind(e):
            k = getattr(e, "event", None)
            return k if isinstance(k, str) and KIND.fullmatch(k) else "<other>"

        def tool(e):
            f = getattr(e, "function", None)
            return f if isinstance(f, str) and TOOL.fullmatch(f) else "<other>"

        def mname(e):
            m = getattr(e, "model", None)
            return m if isinstance(m, str) and MODEL.fullmatch(m) else "<other>"

        ev = list(s.events)
        last10 = [{"type": kind(e), **({"tool": tool(e)} if kind(e) == "tool" else {}),
                   **({"model": mname(e), "status": status(e)} if kind(e) == "model" else {})} for e in ev[-10:]]
        models = [e for e in ev if kind(e) == "model"]
        per_model = {}
        for name in sorted({mname(e) for e in models}):
            ms = [e for e in models if mname(e) == name]
            st = [status(e) for e in ms]
            done = [e for e, x in zip(ms, st) if x in ("empty", "ok")]
            trailing = 0
            for e in reversed(done):  # over completed events only
                if status(e) != "empty":
                    break
                trailing += 1
            usage = getattr(getattr(done[-1], "output", None), "usage", None) if done else None
            itok = getattr(usage, "input_tokens", None)
            per_model[name] = {"model_events": len(ms), "completed": len(done), "empty_total": st.count("empty"),
                               "empty_trailing": trailing, "failed": st.count("failed"), "pending": st.count("pending"),
                               "no_choice": st.count("no_choice"), "last_completed_input_tokens": itok if isinstance(itok, int) else None}
        store = s.store or {}
        v = store.get("EmptyCompletionCounter:streak")
        streak = v if isinstance(v, int) and not isinstance(v, bool) else None
        out.append({"run": r["key"], "cell": cell, "last_10_events": last10, "per_model": per_model, "store_empty_streak": streak})
    return {"run_dir": run_dir, "cells": out}


def main():
    out_fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        if len(sys.argv) < 3 or sys.argv[1] not in ("list", "epce"):
            raise CensusError("usage: list <run dir> | epce <run dir> <cell>...")
        res = list_mode(checked_dir(sys.argv[2])) if sys.argv[1] == "list" else epce_mode(checked_dir(sys.argv[2]), sys.argv[3:] or [])
        if sys.argv[1] == "epce" and not sys.argv[3:]:
            raise CensusError("epce needs at least one cell")
        os.write(out_fd, (json.dumps(res, indent=1) + "\n").encode())
        os._exit(0)
    except BaseException as e:  # noqa: BLE001
        os.write(out_fd, (json.dumps({"failed": type(e).__name__, "detail": e.args[0] if isinstance(e, CensusError) and e.args else None}) + "\n").encode())
        os._exit(2)


if __name__ == "__main__":
    main()
