"""F12 (RULE.md Addition 2): HTTP header-block line types, their derivation from real honest runs, and the transform.
No Modal and no I/O here; stage_f12.py, reach_f12.py and model_split.py import it.

Header blocks are found line by line in a bash output (a trailing \\r is ignored for matching): a status line
(`HTTP/x NNN`, plain or curl -v's `< `) opens a received block, a request line (`> GET /x HTTP/1.1`) a sent block; header
lines (`name: value`, same prefix) continue it; a blank line (same prefix) ends it. Types: status, header, end (plain),
status<, header<, end<, req>, header>, end>.
"""
import random
import re
from collections import Counter

FAMILIES = ("curl -i/-I/-v", "curl other", "other bash")
TYPES = ("status", "header", "end", "status<", "header<", "end<", "req>", "header>", "end>")
MIN_LINES, MIN_SHARE, SEED = 20, 0.95, 20261014
TOKEN = r"[!#$%&'*+.^_`|~0-9A-Za-z-]+"
STATUS = re.compile(r"^(< )?HTTP/\d(?:\.\d)? \d{3}(?: .*)?$")
REQ = re.compile(r"^> [A-Z]+ \S+ HTTP/\d(?:\.\d)?$")
HEAD = {"": re.compile(rf"^{TOKEN}:(?: .*)?$"), "< ": re.compile(rf"^< {TOKEN}:(?: .*)?$"), "> ": re.compile(rf"^> {TOKEN}:(?: .*)?$")}
BLANK = {"": ("",), "< ": ("<", "< "), "> ": (">", "> ")}
SUFFIX = {"": "", "< ": "<", "> ": ">"}
CURL_IV = re.compile(r"\bcurl\b[^|;&\n]*\s(?:-[a-zA-Z]*[iIv][a-zA-Z]*|--include|--verbose|--head)\b")
DPKG = "Preparing to unpack"


def call_family(args: dict) -> str:
    c = str(args.get("cmd") or args.get("command") or "")
    if re.search(r"\bcurl\b", c):
        return FAMILIES[0] if CURL_IV.search(c) else FAMILIES[1]
    return FAMILIES[2]


def typed_lines(text: str):
    """(line index, type) for every line in an HTTP header block. The empty piece after a final newline is not a line
    (Addition 4: typing it as a blank terminator gave a cut-off block a spurious "\r" at the very end)."""
    lines, out, pre = text.split("\n"), [], None
    if len(lines) > 1 and lines[-1] == "":
        lines = lines[:-1]
    for i, raw in enumerate(lines):
        s = raw[:-1] if raw.endswith("\r") else raw
        if pre is not None:
            if HEAD[pre].match(s):
                out.append((i, "header" + SUFFIX[pre]))
                continue
            if s in BLANK[pre]:
                out.append((i, "end" + SUFFIX[pre]))
                pre = None
                continue
            pre = None
        m = STATUS.match(s)
        if m:
            pre = m.group(1) or ""
            out.append((i, "status" + SUFFIX[pre]))
        elif REQ.match(s):
            pre = "> "
            out.append((i, "req>"))
    return out


def outputs(msgs, fixes):
    """(message index, call family) of every bash output with text content (calls paired as fixes.calls_of pairs them)."""
    return [(x["msg"], call_family(x["args"])) for x in fixes.calls_of(msgs)
            if x["tool"].endswith("bash") and isinstance(msgs[x["msg"]].get("content"), str)]


def tally(msgs, fixes, c: Counter, group: str):
    for i, fam in outputs(msgs, fixes):
        lines = msgs[i]["content"].split("\n")
        for j, t in typed_lines(msgs[i]["content"]):
            c[f"{group}|{fam}|{t}|lines"] += 1
            c[f"{group}|{fam}|{t}|ending in \\r"] += lines[j].endswith("\r")


def derive(real2, real1, fixes, c: Counter):
    """IN cells (family -> types) and the cell table: REAL2 derives (>= MIN_LINES lines, >= MIN_SHARE ending in \\r),
    REAL1 confirms (the same bars)."""
    for arm, ms in (("REAL2", real2), ("REAL1", real1)):
        for m in ms:
            tally(m, fixes, c, arm)
    cells, table = {}, {}
    for fam in FAMILIES:
        for t in TYPES:
            row = {}
            for arm in ("REAL2", "REAL1"):
                n = c[f"{arm}|{fam}|{t}|lines"]
                row[f"{arm} lines"], row[f"{arm} share ending in \\r"] = n, (c[f"{arm}|{fam}|{t}|ending in \\r"] / n if n else None)
            row["IN F12"] = all(row[f"{a} lines"] >= MIN_LINES and row[f"{a} share ending in \\r"] >= MIN_SHARE for a in ("REAL2", "REAL1"))
            table[f"{fam}|{t}"] = row
            if row["IN F12"]:
                cells.setdefault(fam, set()).add(t)
    return cells, table


def f12_text(text: str, types: set) -> str:
    lines = text.split("\n")
    for j, t in typed_lines(text):
        if t in types and not lines[j].endswith("\r"):
            lines[j] += "\r"
    return "\n".join(lines)


def f12(msgs, cells: dict, fixes):
    """New messages and the changed message indices; cells: family -> IN types."""
    out = [dict(m) for m in msgs]
    changed = []
    for i, fam in outputs(msgs, fixes):
        if cells.get(fam):
            new = f12_text(msgs[i]["content"], cells[fam])
            if new != msgs[i]["content"]:
                out[i] = msgs[i] | {"content": new}
                changed.append(i)
    return out, changed


def fixtures(cells: dict, c: Counter) -> bool:
    """Seeded synthetic curl outputs (plain -i blocks, -v blocks with `>`/`<`/`*` lines, header-like body lines after the
    blank line, cut-off outputs, outputs with no trailing newline): F12 must give exactly the CRLF form of every IN type and
    nothing else, and be idempotent. No transcript text."""
    rng = random.Random(SEED)
    names = ["Content-Type", "content-length", "Server", "X-Request-Id", "set-cookie", "Date", "ETag", "vary"]
    types = set().union(*cells.values()) if cells else set()
    for n in range(300):
        v = rng.random() < 0.5
        parts = []  # (text, type or None)
        if v:
            parts += [("*   Trying 127.0.0.1:8080...", None), ("* Connected to localhost", None)]
            parts += [(f"> {rng.choice(['GET', 'POST', 'PUT'])} /api/x{n} HTTP/1.1", "req>")]
            parts += [(f"> {rng.choice(names)}: v{k}", "header>") for k in range(rng.randint(1, 4))] + [(">", "end>")]
        p = "< " if v else ""
        parts += [(f"{p}HTTP/{rng.choice(['1.1', '2'])} {rng.choice([200, 201, 404, 500])}{rng.choice([' OK', ''])}", "status" + SUFFIX[p])]
        parts += [(f"{p}{rng.choice(names)}: {rng.choice(['a', 'application/json', '42', 'x; y=1'])}", "header" + SUFFIX[p]) for _ in range(rng.randint(1, 6))]
        parts += [("<" if v else "", "end" + SUFFIX[p])]
        parts += [(f"body: line {k}", None) for k in range(rng.randint(0, 3))] + [('{"ok": true}', None)]
        if rng.random() < 0.3:
            parts = parts[: rng.randint(2, len(parts))]
        tail = "\n" if rng.random() < 0.7 or parts[-1][0] == "" else ""  # an empty last line needs its newline to exist
        text = "\n".join(t for t, _ in parts) + tail
        want = "\n".join(t + ("\r" if ty in types else "") for t, ty in parts) + tail
        got = f12_text(text, types)
        c["fixture|cases"] += 1
        c["fixture|F12 equals the CRLF form"] += got == want
        c["fixture|idempotent"] += f12_text(got, types) == got
    return c["fixture|cases"] == c["fixture|F12 equals the CRLF form"] == c["fixture|idempotent"]
