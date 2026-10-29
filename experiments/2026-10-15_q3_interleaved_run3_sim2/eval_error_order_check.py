"""Error-order check of ONE sim-attack .eval (09's ruling via d9, 10-16 ~02:4x, for Y's t899): counts, timestamps and error CLASS
names only, NO text. For each error class name found (regex), prints its first and last event timestamps, the event type and index
of its first event, and how many events mention it; for the sample's error and the error event, the classes in order of first
appearance in their traceback (Python prints a chained exception's original cause first), and the chain markers' counts
("direct cause" / "during handling"). Built on eval_timing_check.py's reader and output contract (Codex r60-r60e).
  python3 -I eval_error_order_check.py <run folder of the cell> <run key>    # with the sims venv python (zstd)"""
import datetime as dt, glob, hashlib, json, os, re, sys, zipfile

CLS = re.compile(r"\b([A-Z][A-Za-z]+(?:Error|Exception|Exceeded))\b")
NAME = re.compile(r"[A-Za-z0-9_@/.:-]{1,200}")
TYPES = ("sample_init", "sample_limit", "sandbox", "state", "store", "model", "tool", "approval", "input", "score", "error",
         "logger", "info", "span_begin", "span_end", "step", "subtask")
CAUSE = "The above exception was the direct cause of the following exception"
DURING = "During handling of the above exception, another exception occurred"


class BadName(Exception):
    pass


def name(v):
    if not (isinstance(v, str) and NAME.fullmatch(v)):
        raise BadName
    return v


def ts(s):
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat() if isinstance(s, str) else None
    except ValueError:
        return None


def read_entry(z, f, n):
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


SUMMARY = re.compile(r"^((?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z_][A-Za-z0-9_]*))(?::|$)")
CLASSLIKE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Exceeded|Group|Interrupt)")


def order(text):
    """Exception-summary classes in traceback-block order (Codex r63/r63b). A block starts at a "Traceback (most recent call
    last):" line, at the start of the text, or after a chain marker (where an unraised cause is printed without a header); in it,
    indented frame/source lines are skipped, the FIRST unindented line is the block's summary, and every line after it (message
    continuations, notes) is skipped until the next chain-marker or Traceback line. Exception-group sub-tracebacks ("| ...") are not parsed. Python prints
    a chained exception's original cause first. Only class names and fixed marker counts leave."""
    lines = text.splitlines()
    seen, all_, state = [], [], "frames"  # Codex r63c: a headerless summary (an unraised cause) can open the text or follow a marker

    def take(line):
        m = SUMMARY.match(line)
        if m and CLASSLIKE.fullmatch(m.group(2)):
            all_.append(m.group(2))
            if m.group(2) not in seen:
                seen.append(m.group(2))

    for line in lines:
        if line.startswith("Traceback (most recent call last)"):
            state = "frames"
            continue
        if line.startswith(("During handling of the above exception", "The above exception was the direct cause")):
            state = "frames"  # the next unindented line is a Traceback header or a headerless summary
            continue
        if state != "frames" or not line or line[0] in " \t":
            continue
        take(line)
        state = "after_summary"
    return {"classes_in_order": seen, "summary_lines_in_order": all_, "n_direct_cause_markers": text.count(CAUSE), "n_during_handling_markers": text.count(DURING)}


def main():
    folder, run_key = sys.argv[1:]
    files = sorted(glob.glob(os.path.join(folder, "**", "*.eval"), recursive=True))
    if len(files) != 1:
        raise FileNotFoundError
    f = files[0]
    out = {"run_key": name(run_key), "eval": name(os.path.relpath(f, folder)), "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(), "samples": []}
    with zipfile.ZipFile(f) as z:
        for n in sorted(x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")):
            s = json.loads(read_entry(z, f, n))
            ev = [x for x in (s.get("events") or []) if isinstance(x, dict)]
            per = {}
            for i, x in enumerate(ev):
                t = ts(x.get("timestamp"))
                typ = x.get("event") if x.get("event") in TYPES else "other"
                for c in set(CLS.findall(json.dumps(x))):
                    p = per.setdefault(c, {"first_event": t, "first_event_index": i, "first_event_type": typ, "last_event": t, "n_events": 0})
                    p["n_events"] += 1
                    if t and (p["last_event"] is None or t > p["last_event"]):
                        p["last_event"] = t
            err = s.get("error") if isinstance(s.get("error"), dict) else {}
            err_ev = [x for x in ev if x.get("event") == "error"]
            e_err = err_ev[-1].get("error") if err_ev and isinstance(err_ev[-1].get("error"), dict) else {}
            out["samples"].append({"entry": name(n), "n_events": len(ev), "classes_by_first_event": dict(sorted(per.items(), key=lambda kv: kv[1]["first_event_index"])),
                                   "sample_error_traceback": order(str(err.get("traceback") or "")), "sample_error_message_classes": sorted(set(CLS.findall(str(err.get("message") or "")))),
                                   "error_event_traceback": order(str(e_err.get("traceback") or "")), "error_event_timestamp": ts(err_ev[-1].get("timestamp")) if err_ev else None})
    print("ORDER " + json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except BaseException as x:  # noqa: BLE001 - only the class name leaves this script
        print("ORDER_FAILED " + (type(x).__name__ if type(x).__name__.isidentifier() else "Exception"))
        sys.exit(1)
