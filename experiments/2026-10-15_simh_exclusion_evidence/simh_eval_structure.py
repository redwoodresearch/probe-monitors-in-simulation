""".eval completeness check and error-field hashes, keys and counts only. No transcript or error text is emitted.
  struct <run dir> <cell prefix> <control run dir> <control cell prefix>: member kinds (allowlisted), log status,
    results/reductions, sample error (+ traceback present), messages by role, last event type, event counts, for the
    target and a clean control; plus the launch's fork-source reference sha256 and a re-hash of its copy.
  hash <run dir> <cell prefix>...: for each cell, the length and sha256 (of the UTF-8 bytes, no normalization) of
    (i) launch.json runs[key].attempts[i].outcome.sample_errors[j] as stored, and (ii) the final .eval's sample
    error.message (inspect_ai read_eval_log, the field simh_err_diag.py hashes); and whether they are equal.
  ON the box: PYTHONDONTWRITEBYTECODE=1 ~/sims/simulated-control-evals-sept/.venv/bin/python - <mode> <args> < simh_eval_structure.py
All work runs with file descriptors 1 and 2 on /dev/null; any failure prints only {"failed": <class>, "detail": <fixed
string or null>} and exits 2.
"""
import collections, glob, hashlib, json, os, re, sys, zipfile

# STRUCT_ARM selects the arm directory (allowlisted); default sim-honest
ARM = os.environ.get("STRUCT_ARM", "sim_honest@september")
if ARM not in ("sim_honest@september", "sim_attack@september"):
    raise SystemExit(json.dumps({"failed": "arm not allowlisted", "detail": None}))
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,80}")
EVENT_TYPES = {"span_begin", "span_end", "sample_init", "model", "tool", "store", "info", "logger", "error", "state", "input",
               "score", "sandbox", "approval", "subtask", "step", "sample_limit"}
KNOWN_FRAMES = {"sim/contract.py:313 validate_action_output", "sim/contract.py:344 validate_action_output",
                "runner/policy/step_limit.py:72 on_continue", "tenacity/__init__.py:414 exc_check"}
ev_name = lambda e: e if e in EVENT_TYPES else "<other>"
KNOWN_ERRORS = {"OutputContractError", "EmptyPolicyCompletionError", "RetryError", "RateLimitError", "AttemptTimeoutError",
                "TimeoutError", "LimitExceededError", "RuntimeError", "ValueError", "KeyError", "CancelledError", "APIError",
                "APIStatusError", "APITimeoutError", "APIConnectionError", "InternalServerError", "PrerequisiteError",
                "ToolError", "ToolParsingError", "SandboxError", "PolicyError", "SimulatorError"}
KINDS = {"header.json", "summaries.json", "reductions.json", "samples/<file>.json", "_journal/<file>.json",
         "_journal/summaries/<file>.json", "_journal/start.json"}
PATH_OK = re.compile(r"^[A-Za-z0-9_./@:-]{1,300}$")


class StructError(Exception):
    pass


def one_cell(run_dir, prefix):
    hits = [c for c in os.listdir(os.path.join(run_dir, ARM)) if c.startswith(prefix + "_")]
    if len(hits) != 1:
        raise StructError("cell prefix does not match exactly one cell dir")
    return hits[0]


def final_eval(run_dir, cell):
    ps = sorted(glob.glob(os.path.join(run_dir, ARM, cell, "*.eval")))
    if not ps:
        raise StructError("no .eval in the cell dir")
    return ps[-1], len(ps)


def kind(n):
    k = re.sub(r"[^/]+\.json$", "<file>.json", n) if n.startswith(("samples/", "_journal/")) and n != "_journal/start.json" else n
    return k if k in KINDS else "<other>"


def read_log(p):
    from inspect_ai.log import read_eval_log
    log = read_eval_log(p)
    if len(log.samples or []) != 1:
        raise StructError("eval does not hold exactly one sample")
    return log, log.samples[0]


def struct(run_dir, prefix):
    cell = one_cell(run_dir, prefix)
    p, n_evals = final_eval(run_dir, cell)
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
    log, s = read_log(p)
    roles = collections.Counter(str(m.role) for m in s.messages)
    ev_types = collections.Counter(str(e.event) for e in s.events)
    # the error CLASS only: the first column-0 line after the traceback's last frame, reduced to its dotted-name head and
    # reported only if it is in KNOWN_ERRORS (a message line can never be emitted, even an identifier-shaped word)
    ecls, deepest_frame = None, None
    if s.error is not None:
        lines = (s.error.traceback or "").splitlines()
        # the LAST "Traceback (most recent call last):" block, up to its exception line (the first column-0 line after it):
        # frames come only from inside that block, so text in the message can never pose as a frame or a class
        starts = [i for i, l in enumerate(lines) if l == "Traceback (most recent call last):"]
        b0 = starts[-1] + 1 if starts else len(lines)  # no header: nothing is parsed
        exc = next((i for i in range(b0, len(lines)) if lines[i] and not lines[i].startswith(" ")), None)
        frames = [i for i in range(b0, exc if exc is not None else len(lines)) if lines[i].startswith("  File ")]
        head = re.match(r"([A-Za-z_][A-Za-z0-9_.]*)(:|$)", lines[exc]) if exc is not None else None
        name = head.group(1).rsplit(".", 1)[-1] if head else None
        ecls = name if name in KNOWN_ERRORS else ("<other>" if s.error else None)
        # the deepest frame's code location, parsed only from frame lines BEFORE the exception line (message text can't
        # pose as a frame), and only for paths under the simulator src tree or site-packages
        fr = re.match(r'  File "([^"]+)", line ([0-9]+), in ([A-Za-z_<>][A-Za-z0-9_<>]*)$', lines[frames[-1]]) if frames else None
        deep = None
        if fr:
            path = fr.group(1)
            for root in ("/sims-sept-strict-off/src/", "/site-packages/"):
                if root in path:
                    deep = f"{path.split(root, 1)[1]}:{fr.group(2)} {fr.group(3)}"
        # only fixed, allowlisted locations are ever emitted
        deepest_frame = deep if deep in KNOWN_FRAMES else ("<other>" if frames else None)
    return {"run_dir": run_dir, "cell": cell, "n_eval_files": n_evals, "member_kinds": sorted({kind(n) for n in names}),
            "has_header": "header.json" in names, "n_sample_files": sum(n.startswith("samples/") for n in names),
            "has_reductions": "reductions.json" in names, "has_summaries": "summaries.json" in names,
            "log_status": str(log.status) if str(log.status) in ("success", "error", "cancelled", "started") else "<other>",
            "has_results_block": log.results is not None, "n_samples": 1, "sample_has_error": s.error is not None,
            "sample_error_has_traceback": bool(s.error and s.error.traceback),
            "messages_by_role": dict(sorted((r if r in ("system", "user", "assistant", "tool") else "<other>", c) for r, c in roles.items())),
            "last_event_type": ev_name(str(s.events[-1].event)) if s.events else None,
            "n_events": len(s.events), "n_error_events": sum(e.event == "error" for e in s.events),
            "events_by_type": {k: sum(v for kk, v in ev_types.items() if ev_name(kk) == k) for k in sorted({ev_name(kk) for kk in ev_types})},
            "sample_error_class": ecls, "deepest_frame": deepest_frame, "arm": ARM}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reference(run_dir, prefix):
    cell = one_cell(run_dir, prefix)
    refs = json.load(open(os.path.join(run_dir, "launch.json"))).get("references") or {}
    if cell not in refs:
        raise StructError("no launch reference for the cell")
    r = refs[cell]
    copy = r["copy"] if os.path.isabs(r["copy"]) else os.path.join(run_dir, r["copy"])
    rehash = sha_file(copy) if os.path.isfile(copy) else None
    safe = lambda v: v if isinstance(v, str) and PATH_OK.match(v) else ("<other>" if v is not None else None)
    launch = json.load(open(os.path.join(run_dir, "launch.json")))
    return {"cell": cell, "reference_sha256": r["sha256"] if re.fullmatch(r"[0-9a-f]{64}", str(r["sha256"])) else "<other>",
            "copy_rehash": rehash, "copy_rehash_equal": rehash == r["sha256"],
            "references_from": safe(launch.get("references_from")), "reused_from": safe(r.get("reused_from")),
            "try": r.get("try") if isinstance(r.get("try"), int) else None}


def h(v):
    if not isinstance(v, str):
        return {"type": type(v).__name__}
    return {"type": "str", "len": len(v), "sha256": hashlib.sha256(v.encode("utf-8")).hexdigest()}


def hashes(run_dir, prefix):
    cell = one_cell(run_dir, prefix)
    launch = json.load(open(os.path.join(run_dir, "launch.json")))
    runs = [r for r in launch["runs"] if r.get("key") == f"{ARM}/{cell}"]
    if len(runs) != 1:
        raise StructError("cell does not match exactly one launch.json run")
    lj = [{"attempt": i, "index": j, **h(v)} for i, a in enumerate(runs[0].get("attempts") or [])
          for j, v in enumerate(((a.get("outcome") or {}).get("sample_errors")) or [])]
    p, _ = final_eval(run_dir, cell)
    _, s = read_log(p)
    ev = h(s.error.message) if s.error is not None else {"type": "NoneType"}
    last = [x for x in lj if x["attempt"] == len(runs[0]["attempts"]) - 1 and x["index"] == 0]
    return {"run_dir": run_dir, "cell": cell, "eval_sha256": sha_file(p),
            "launch_json_outcome_sample_errors": lj,
            "eval_sample_error_message": ev,
            "last_attempt_sample_errors0_equals_eval_message": bool(last) and last[0].get("sha256") is not None
            and last[0].get("sha256") == ev.get("sha256")}


def main():
    out_fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        mode, args = sys.argv[1], sys.argv[2:]
        if mode == "struct" and len(args) == 4:
            res = {"target": struct(args[0], args[1]), "control": struct(args[2], args[3]), "reference": reference(args[0], args[1])}
        elif mode == "struct1" and len(args) >= 2:  # targets only: no control, no reference
            res = {"targets": [struct(args[0], c) for c in args[1:]]}
        elif mode == "hash" and len(args) >= 2:
            res = {"hash_method": "sha256 of the UTF-8 bytes of the stored string, no normalization",
                   "cells": [hashes(args[0], c) for c in args[1:]]}
        else:
            raise StructError("usage: struct <run dir> <cell> <control run dir> <control cell> | hash <run dir> <cell>...")
        payload, code = res, 0
    except BaseException as e:  # noqa: BLE001 - class only, never the message
        payload, code = {"failed": type(e).__name__, "detail": e.args[0] if isinstance(e, StructError) and e.args else None}, 2
    os.write(out_fd, (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode())
    os._exit(code)


if __name__ == "__main__":
    main()
