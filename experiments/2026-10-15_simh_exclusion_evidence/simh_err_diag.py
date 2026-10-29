"""Diagnosis of sim-honest terminal errors (d9 10-15 ~15:20, for 0d; Aswin's diagnose-at-once rule): for each cell's final
.eval sample, the error class, the terminal traceback's FRAMES (code locations), the deepest frame and which side raised it,
the raising code line (only from trusted installed code), the message text ONLY when the raising line is `raise <Class>(
"<constant>")` for that class and the message equals that constant (else withheld: it may embed model output; its length and
sha256 are given), and the tool name and tool-step index of the last tool call before the terminal error event. No
transcript or tool-output text is emitted.

  ON the box, read-only, sims venv, stdout captured on the laptop:
  PYTHONDONTWRITEBYTECODE=1 ~/sims/simulated-control-evals-sept/.venv/bin/python - <run dir> <cell>... < simh_err_diag.py

Safety (Codex r31): frames are parsed only from the LAST "Traceback (most recent call last):" block, up to its exception
line, so text inside a message can't pose as a frame; source lines come only from absolute .py paths under TRUSTED roots;
the literal check uses ast on the raising line; env, main and tool names must match identifier patterns (else "<other>");
all output of the work runs with file descriptors 1 and 2 pointed at /dev/null, and any dependency exit is a failure.
STOP rule (d9): a credential-looking string (CRED_VALUE anywhere in the message or traceback; CRED_FIELD, a bare key or
token field name, in the message and every traceback line except verified installed-code lines) prints only {"STOP": ..., "cell": <run key>} and exits 3. Any other failure prints
{"failed": <class>, "detail": <fixed string or null>} and exits 2.
"""

import ast
import collections
import functools
import hashlib
import json
import linecache
import os
import re
import sys

sys.dont_write_bytecode = True

HOME = "/home/aswin"
TRUSTED = tuple(os.path.realpath(p) + "/" for p in (f"{HOME}/sims", f"{HOME}/phase2/attack_check_ug",
                                                    f"{HOME}/phase2/attack_check_311", f"{HOME}/phase2/sims-sept-strict-off",
                                                    f"{HOME}/phase2/sims-sept-ungrounded"))
FRAME = re.compile(r'^  File "(/[^"]+\.py)", line (\d+), in ([A-Za-z_<>][\w<>]*)$')
CLS = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Timeout|Exceeded|Exit))(?=[:(\[]|$)")
# value-shaped secrets: scanned over the whole message and traceback
CRED_VALUE = re.compile(r"(?i)(sk-[a-z0-9_-]{12,}|AKIA[0-9A-Z]{12,}|ASIA[0-9A-Z]{12,}|bearer\s+\S{8,}|"
                        r"-----BEGIN [A-Z ]*PRIVATE KEY-----)")
# field names (conservative: a bare name triggers): scanned over the message and every traceback line except VERIFIED
# installed-code lines (the code line after a verified frame header), where identifiers like api_key are ordinary
CRED_FIELD = re.compile(r"(?i)(api[\s_-]?key|access[\s_-]?token|session[\s_-]?token|secret[\s_-]?(access[\s_-]?)?key|"
                        r"authorization|password)")
IDENT = re.compile(r"[a-z0-9_]{1,80}")  # used with fullmatch
TOOL = re.compile(r"[a-z_][a-z0-9_]{0,40}")


class CredentialStop(Exception):
    pass


class DiagError(Exception):
    """A required piece is missing or inconsistent; the argument is a fixed string from this file."""


def side(path: str) -> str:
    if "/site-packages/inspect_ai/" in path:
        return "inspect_ai (library)"
    if any(f"/site-packages/{p}/" in path for p in ("openai", "anthropic", "httpx", "httpcore", "tenacity")):
        return "provider client library"
    if "/phase2/attack_check" in path:
        return "harness (phase2 attack_check)"
    if "/site-packages/" in path:
        return "other library"
    if path.startswith(f"{HOME}/sims/") or path.startswith(f"{HOME}/phase2/sims-sept"):
        return "simulator (simulated-control-evals tree)"
    return "other"


def rel(path: str) -> str:
    if "/site-packages/" in path:
        return "site-packages/" + path.split("/site-packages/", 1)[1]
    return path.removeprefix(HOME + "/")


def trusted(path: str):
    """The resolved real path if it is a .py file under a trusted root, else None (symlinks and .. resolved first)."""
    rp = os.path.realpath(path)
    return rp if rp.endswith(".py") and rp.startswith(TRUSTED) and os.path.isfile(rp) else None


@functools.lru_cache(maxsize=None)
def func_spans(rp: str) -> dict:
    tree = ast.parse(open(rp, encoding="utf-8").read())
    spans = collections.defaultdict(list)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans[n.name].append((n.lineno, n.end_lineno))
    return dict(spans)


def verify(path: str, line: int, func: str):
    """A frame is VERIFIED only if its real path is trusted code and `line` lies inside a function named `func` there
    (or func is <module> and the line exists): a spoofed frame then echoes only real code coordinates."""
    rp = trusted(path)
    if rp is None:
        return None
    try:
        if func == "<module>":
            return rp if linecache.getline(rp, line) else None
        return rp if any(a <= line <= b for a, b in func_spans(rp).get(func, [])) else None
    except (SyntaxError, OSError, UnicodeDecodeError):
        return None


def terminal_frames(tb: str) -> list:
    lines = tb.splitlines()
    heads = [i for i, l in enumerate(lines) if l == "Traceback (most recent call last):"]
    if not heads:
        raise DiagError("traceback without a header")
    out = []
    for l in lines[heads[-1] + 1:]:
        if not l.startswith(" "):
            break  # the exception line: the message starts here and is never parsed as frames
        m = FRAME.match(l)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3)))
    if not out:
        raise DiagError("no frames in the terminal traceback block")
    return out


def literal_message(src: str, cls_name: str):
    """The constant of exactly `raise <Name>("<constant>")` (one statement, a plain Name equal to the class, one str
    constant argument, no keywords, no `from`), else None."""
    try:
        body = ast.parse(src.strip()).body
    except SyntaxError:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].cause is not None:
        return None
    exc = body[0].exc
    if not isinstance(exc, ast.Call) or not isinstance(exc.func, ast.Name) or exc.keywords or len(exc.args) != 1:
        return None
    a = exc.args[0]
    if exc.func.id != cls_name.split(".")[-1] or not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
        return None
    return a.value


def message_body(msg: str, cls_name: str):
    """Exactly two formats: "<Class>: <body>" or "<Class>(<one str literal>)"; else None."""
    if msg.startswith(cls_name + ": "):
        return msg[len(cls_name) + 2:]
    if msg.startswith(cls_name + "(") and msg.endswith(")"):
        try:
            v = ast.literal_eval(msg[len(cls_name) + 1:-1])
        except (ValueError, SyntaxError):
            return None
        return v if isinstance(v, str) else None
    return None


def diag(run_dir: str, cell: str) -> dict:
    from inspect_ai.log import read_eval_log

    launch = json.load(open(os.path.join(run_dir, "launch.json")))
    runs = [r for r in launch["runs"] if r.get("arm") == "sim_honest" and r["key"].split("/", 1)[-1].startswith(cell + "_")]
    if len(runs) != 1:
        raise DiagError("cell does not match exactly one sim_honest run")
    r = runs[0]
    tasks = [t for t in (launch.get("tasks") or []) if isinstance(t, dict) and str(t.get("key", "")) == r["key"].split("/", 1)[1]]
    task = tasks[0] if len(tasks) == 1 else {}
    att = (r.get("attempts") or [None])[-1]
    if not att or not att.get("log_dir"):
        raise DiagError("no last attempt with a log dir")
    evals = sorted((os.path.join(dp, f) for dp, _, fs in os.walk(att["log_dir"]) for f in fs if f.endswith(".eval")),
                   key=os.path.getmtime)
    if not evals:
        raise DiagError("no .eval for the last attempt")
    log = read_eval_log(evals[-1])
    if len(log.samples or []) != 1:
        raise DiagError("not exactly one sample")
    s = log.samples[0]
    if s.error is None:
        raise DiagError("the sample has no error")
    msg, tb = s.error.message or "", s.error.traceback or ""
    # the field-name scan exempts ONLY lines that are verified source lines (the code line right after a verified frame
    # header, equal to the installed file's line); every other traceback line, indented or not, is scanned
    tlines = tb.splitlines()
    exempt = set()
    for i, l in enumerate(tlines):
        fm = FRAME.match(l)
        if fm and i + 1 < len(tlines):
            rp = verify(fm.group(1), int(fm.group(2)), fm.group(3))
            if rp and tlines[i + 1].strip() == linecache.getline(rp, int(fm.group(2))).strip():
                exempt.add(i + 1)
    scanned = "\n".join(l for i, l in enumerate(tlines) if i not in exempt)
    if CRED_VALUE.search(msg) or CRED_VALUE.search(tb) or CRED_FIELD.search(msg) or CRED_FIELD.search(scanned):
        raise CredentialStop(r["key"])
    m = CLS.match(msg)
    cls_name = m.group(1) if m else "<unclassified>"
    parsed = terminal_frames(tb)
    fr, unverified = [], 0
    for p, n, fn in parsed:
        rp = verify(p, n, fn)
        if rp is None:
            unverified += 1
            continue
        fr.append({"file": rel(rp), "line": n, "func": fn, "side": side(rp), "_rp": rp})
    if not fr:
        raise DiagError("no verified frame in the terminal traceback")
    deepest_parsed_verified = verify(*parsed[-1]) is not None
    d = fr[-1]
    src = linecache.getline(d["_rp"], d["line"]).rstrip("\n") or None
    for f in fr:
        f.pop("_rp")
    lit = literal_message(src, cls_name) if src and m else None
    body = message_body(msg, cls_name) if m else None
    fixed = lit is not None and body is not None and body == lit
    errs = [i for i, e in enumerate(s.events) if e.event == "error"]
    if not errs:
        raise DiagError("no error event")
    last_err = s.events[errs[-1]]
    ev_msg = getattr(getattr(last_err, "error", None), "message", None)
    tools = [(i, e) for i, e in enumerate(s.events) if e.event == "tool"]
    before = [(k, e) for k, (i, e) in enumerate(tools, 1) if i < errs[-1]]
    tname = lambda e: (lambda f: f if isinstance(f, str) and TOOL.fullmatch(f) else "<other>")(getattr(e, "function", None))
    return {"cell": cell, "run": r["key"],
            "env": task.get("env") if isinstance(task.get("env"), str) and IDENT.fullmatch(task["env"]) else None,
            "main": task.get("main") if isinstance(task.get("main"), str) and IDENT.fullmatch(task["main"]) else None,
            "eval_sha256": hashlib.sha256(open(evals[-1], "rb").read()).hexdigest(),
            "error_class": cls_name, "message_text": body if fixed else None,
            "message_withheld": None if fixed else "the raising line is not raise <Class>(<constant>) with this message",
            "message_len": len(msg), "message_sha256": hashlib.sha256(msg.encode()).hexdigest(),
            "terminal_error_event_matches_sample": ev_msg == msg,
            "frames": fr, "deepest_frame": fr[-1], "unverified_frames": unverified,
            "deepest_parsed_frame_verified": deepest_parsed_verified, "raise_source_line": src,
            "last_tool_before_error": {"tool": tname(before[-1][1]), "tool_step": before[-1][0]} if before else None,
            "tool_steps_before_error": len(before),
            "tool_name_counts": dict(collections.Counter(tname(e) for _, e in tools))}


def main():
    out_fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        if len(sys.argv) < 3:
            raise DiagError("usage: <run dir> <cell>...")
        run_dir, cells = sys.argv[1], sys.argv[2:]
        res = {"run_dir": run_dir, "cells": [diag(run_dir, c) for c in cells]}
        payload = {"ok": res}
    except CredentialStop as e:
        payload = {"stop": e.args[0]}
    except BaseException as e:  # noqa: BLE001 - includes a dependency's SystemExit: class only
        payload = {"failed": type(e).__name__, "detail": e.args[0] if isinstance(e, DiagError) and e.args else None}
    if "ok" in payload:
        os.write(out_fd, (json.dumps(payload["ok"], indent=1) + "\n").encode())
        code = 0
    elif "stop" in payload:
        os.write(out_fd, (json.dumps({"STOP": "credential-like string in an error record", "cell": payload["stop"]}) + "\n").encode())
        code = 3
    else:
        os.write(out_fd, (json.dumps(payload) + "\n").encode())
        code = 2
    os._exit(code)


if __name__ == "__main__":
    main()
