"""Shared I/O for the 32k vs full-length appendix record (09 RULE @ 129b7c4f1, lead §3 (ii)). No recomputation: every value is read by key.

Inputs are named with their sha256 and checked on load; every output is written with exclusive create, so nothing is ever overwritten.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

LABEL = "32k (registered first read) vs full length (lead)"  # 09's label, verbatim
WINDOWS = {"cut32k": "first 32k tokens (registered first read)", "full": "full length (no 32k cut) (lead)"}
NO_TEST = "intervals side by side; no difference test (no paired or unpaired contrast is computed)"
SENS_ROLE = "registered probe scored on full transcripts (OOD past 32k)"  # 09's label for the registered weights read uncut (a sensitivity bar)
# 09 §3 (ii) and RULE item 33 @ ddadd5f15: the lead combo is pooled_all_origins in BOTH builds (the four-set union, a main counted once), pinned
# BY NAME; no role tag is read (record A's "lead" tag on pooled_all is superseded). FOUR_SETS: its members, by name, as Stage A tags them.
CONTRACT = {"f": "0.01", "mode": "joint", "combo": "pooled_all_origins", "readouts": ["C", "R"], "lead_readout": "C", "gap": "delta_full"}
FOUR_SETS = tuple(f"{o} (all-origins draws)" for o in ("S_old", "S_new", "S_newmains", "S_paper493"))


def stop(msg):
    sys.exit(f"STOP: {msg}")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, sha):
    raw = Path(path).read_bytes()
    if sha_bytes(raw) != sha:
        stop(f"sha256 mismatch for {path}: {sha_bytes(raw)} != {sha}")
    return json.loads(raw)


def fill(path, **kw):
    """A key-path template (list of str) with {label}, {combo}, {mode}, {f}, {ro}, {member} filled."""
    return [p.format(**kw) for p in path]


def keystr(path):
    return "".join(f'["{p}"]' for p in path)


def get(rec, path, where="", unavailable_ok=False):
    """The value at a key path; a missing key stops (never a silent default). With unavailable_ok, the first dict on the path (the root
    excluded) that carries the record's own 'unavailable' text wins over anything below it, retained numbers included."""
    x = rec
    for i, p in enumerate(path):
        if unavailable_ok and i > 0 and isinstance(x, dict) and "unavailable" in x:
            return {"unavailable": str(x["unavailable"]), "at": keystr(path[:i])}
        if not isinstance(x, dict) or p not in x:
            stop(f"{where}: no key {keystr(path[: i + 1])}")
        x = x[p]
    return x


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def node(rec, path, where=""):
    """A summary node by key: {point, finite_draws, draws, ci95 | None + withheld}. An 'unavailable' node (or ancestor) keeps the record's
    text. Every other shape stops: missing draws counts, non-finite or boolean numbers, reversed bounds, a withheld reason beside a numeric
    interval, or no interval without a reason."""
    x = get(rec, path, where, unavailable_ok=True)
    key = keystr(path)
    if isinstance(x, dict) and "unavailable" in x:
        return {"key": key, "unavailable": str(x["unavailable"]), "unavailable_at": x.get("at", key)}
    if not isinstance(x, dict):
        stop(f"{where}: {key} is not a summary node")
    bad = []
    if not num(x.get("point")):
        bad.append("point")
    d, fd = x.get("draws"), x.get("finite_draws")
    if not (isinstance(d, int) and not isinstance(d, bool) and d > 0 and isinstance(fd, int) and not isinstance(fd, bool) and 0 <= fd <= d):
        bad.append("draws/finite_draws")
    ci, wh = x.get("ci95", "absent"), x.get("withheld")
    if ci == "absent":
        bad.append("ci95 key")
    elif ci is None:
        if not (isinstance(wh, str) and wh.strip()):
            bad.append("ci95 null without a withheld reason")
    elif not (isinstance(ci, list) and len(ci) == 2 and all(num(v) for v in ci) and ci[0] <= ci[1]):
        bad.append("ci95 malformed")
    elif wh:
        bad.append("withheld reason beside a numeric ci95")
    if bad:
        stop(f"{where}: {key} fails the summary-node schema: {bad}")
    return {"key": key, "point": x["point"], "ci95": ci, "withheld": wh, "finite_draws": fd, "draws": d,
            **({"degenerate": x["degenerate"]} if "degenerate" in x else {})}


def write_new(path, text_or_bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if isinstance(text_or_bytes, bytes) else "x"
    with open(path, mode) as f:  # exclusive create: a collision fails loudly
        f.write(text_or_bytes)
    print(path)
    return path


def placeholders(obj, where="plan"):
    """Every '<...>' placeholder left in a filled plan (a template value never reaches a record)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += placeholders(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += placeholders(v, f"{where}[{i}]")
    elif isinstance(obj, str) and "<" in obj and ">" in obj:
        out.append(f"{where} = {obj}")
    return out
