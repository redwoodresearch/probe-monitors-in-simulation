"""Scripted check of ONE sim-attack .eval (09's rule via d9, 10-16 ~02:1x): counts and timestamps only, NO text.
Prints: the .eval's sha256 and entry names; per sample: message count, event counts by a FIXED list of Inspect event types,
first/last event timestamps, the last model / tool / error event timestamps (and the last model event's completed time),
error CLASS names only (regex), the sample limit's type (fixed list: which limit ended it, if any), numeric total/working time; plus the launcher's attempt start/end/exit/seconds from launch.json.
Every printed value is re-emitted as a parsed timestamp, a number, a boolean, a count, a sha256, a validated name, a class name, or null
(JSON null = the value is absent or not of the allowed type);
any failure prints only its exception class (Codex r60). Exactly one .eval must be in the folder.
  python3 -I eval_timing_check.py <run folder of the cell> <launch.json> <run key>"""
import datetime as dt, glob, hashlib, json, os, re, sys, zipfile

CLS = re.compile(r"\b([A-Z][A-Za-z]+(?:Error|Exception|Exceeded))\b")
NAME = re.compile(r"[A-Za-z0-9_@/.:-]{1,200}")
LIMITS = ("context", "time", "working", "message", "token", "operator", "custom", "cost")  # Inspect's sample limit types
TYPES = ("sample_init", "sample_limit", "sandbox", "state", "store", "model", "tool", "approval", "input", "score", "error",
         "logger", "info", "span_begin", "span_end", "step", "subtask")


def ts(s):
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat() if isinstance(s, str) else None
    except ValueError:
        return None


def num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class BadName(Exception):
    pass


def name(v):
    if not (isinstance(v, str) and NAME.fullmatch(v)):
        raise BadName  # Codex r60b: a non-matching name stops the check (class-only failure), never printed
    return v


def read_entry(z, f, n):
    """An entry's bytes. Inspect's .eval uses zstd (zip method 93), which stdlib zipfile cannot decompress: read the raw
    compressed bytes after the local file header and decompress them with zstandard (run with the sims venv's python)."""
    i = z.getinfo(n)
    if i.compress_type != 93:
        return z.read(n)
    import struct
    import zstandard
    with open(f, "rb") as fh:
        fh.seek(i.header_offset)
        h = fh.read(30)
        if h[:4] != b"PK\x03\x04":
            raise ValueError
        nlen, xlen = struct.unpack("<HH", h[26:30])
        fh.seek(i.header_offset + 30 + nlen + xlen)
        raw = fh.read(i.compress_size)
    data = zstandard.ZstdDecompressor().decompress(raw, max_output_size=i.file_size)
    if len(data) != i.file_size:
        raise ValueError
    return data


def main():
    folder, lj, run_key = sys.argv[1:]
    out = {"run_key": name(run_key), "folder": name(folder), "evals": []}
    d = json.load(open(lj))
    runs = d["runs"]; runs = list(runs.values()) if isinstance(runs, dict) else runs
    r = [x for x in runs if x["key"] == run_key]
    if len(r) != 1:
        raise LookupError  # Codex r60b: explicit, not assert (-O safe)
    out["launcher_attempts"] = [{"start": ts(a.get("start")), "end": ts(a.get("end")), "exit_code": a.get("exit_code") if isinstance(a.get("exit_code"), int) else None,
                                 "process_seconds": num(a.get("process_seconds"))} for a in r[0]["attempts"]]
    files = sorted(glob.glob(os.path.join(folder, "**", "*.eval"), recursive=True))
    if len(files) != 1:
        raise FileNotFoundError  # exactly one .eval (Codex r60b: explicit, not assert)
    f = files[0]
    e = {"file": name(os.path.relpath(f, folder)), "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(), "samples": []}
    with zipfile.ZipFile(f) as z:
        names = z.namelist()
        e["n_zip_entries"] = len(names)
        for n in sorted(x for x in names if x.startswith("samples/") and x.endswith(".json")):
            s = json.loads(read_entry(z, f, n))
            ev = [x for x in (s.get("events") or []) if isinstance(x, dict)]
            types = {t: 0 for t in TYPES}; types["other"] = 0
            for x in ev:
                t = x.get("event")
                types[t if t in TYPES else "other"] += 1
            stamped = [(ts(x.get("timestamp")), x) for x in ev]
            stamped = [(t, x) for t, x in stamped if t]
            last_of = lambda typ: max(((t, x) for t, x in stamped if x.get("event") == typ), key=lambda p: p[0], default=(None, None))  # noqa: E731
            (tm, lm), (tt, _), (te, le) = last_of("model"), last_of("tool"), last_of("error")
            classes = set(CLS.findall(json.dumps(s.get("error") or ""))) | set(CLS.findall(json.dumps(le or "")))
            e["samples"].append({"entry": name(n), "n_messages": len(s.get("messages") or []), "n_events": len(ev), "events_by_type": types,
                                 "first_event": min(t for t, _ in stamped) if stamped else None, "last_event": max(t for t, _ in stamped) if stamped else None,
                                 "last_model_event": tm, "last_model_event_completed": ts(lm.get("completed")) if lm else None,
                                 "last_model_event_has_error_field": bool(lm and lm.get("error")), "last_tool_event": tt, "error_event": te,
                                 "error_classes": sorted(classes), "sample_has_error": bool(s.get("error")),
                                 "limit_type": (lambda t: t if t in LIMITS else ("other" if t is not None else None))((s.get("limit") or {}).get("type") if isinstance(s.get("limit"), dict) else None),
                                 "n_model_events": types["model"], "n_tool_events": types["tool"],
                                 "total_time": num(s.get("total_time")), "working_time": num(s.get("working_time"))})
    out["evals"].append(e)
    print("CHECK " + json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except BaseException as x:  # noqa: BLE001 - only the class name leaves this script (no message, no traceback)
        print("CHECK_FAILED " + (type(x).__name__ if CLS.fullmatch(type(x).__name__) or type(x).__name__.isidentifier() else "Exception"))
        sys.exit(1)
