"""Variable flow between launch blocks that run in ONE shell (route C; 2e's collision check after a03f5d93's NA/NC finding, 10-18).

A static scan of block TEXT: it runs nothing (rule 15). For each block it finds:
- the names the block assigns at its top level. These persist into later blocks. Assignments inside $( ) or ( ) do not persist, nor
  do command-prefix ones (`TZ=… date`);
- its free reads: the names it reads before assigning them itself, so their values come from an earlier block or the environment.

For a chain (blocks run in order in one shell), each free read's value comes from the LAST earlier block that assigns it. The read is OK
when that block is a VARS block. It is a COLLISION when that block is a working block (a later block overwrote the VARS value, as
PATHGATE3's NA and NC did), and UNPROVIDED when no block assigns it and it is not in ENV. Exit 1 on any COLLISION or UNPROVIDED.
`shadows` lists a working block's top-level names that a VARS block of the chain also assigns (harmless unless a later block reads them).

  uv run --offline --no-project --python 3.12 python -I block_vars.py [--md] CHAIN [CHAIN ...]
  CHAIN = LABEL=SRC#BLOCK[,SRC#BLOCK...]; SRC is a file path, or REV:PATH (read with git show); BLOCK is an id (`    # BEGIN <id>`) or
  `VARS <id>` (between the `<!-- BEGIN VARS <id> -->` markers, fence lines dropped)
"""
import re
import subprocess
import sys
from pathlib import Path

ENV = {"PYTHONPATH", "PYTHONHOME", "PYTHONCASEOK", "SECONDS", "PWD"}  # read from the shell or environment by design
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=")
REDIR = re.compile(r"[0-9]*(<<<|>>|>&|<&|>|<)")
OPEN_KW = {"if", "then", "else", "elif", "do", "while", "until", "!", "{", "}"}
TERM = set(" \t\n;&|()<>")


class Scan:
    def __init__(self, s):
        self.s, self.i, self.ev, self.sid = s, 0, [], 0

    def fail(self, m):
        raise SystemExit(f"block_vars: {m} at offset {self.i}: {self.s[self.i:self.i + 50]!r}")

    def at(self, t):
        return self.s.startswith(t, self.i)

    def c(self, k=0):
        return self.s[self.i + k] if self.i + k < len(self.s) else ""

    def emit(self, kind, name, scope):
        self.ev.append((kind, name, scope))

    def squote(self):
        j = self.s.find("'", self.i + 1)
        if j < 0:
            self.fail("unclosed single quote")
        self.i = j + 1

    def ansi(self):  # $'...'
        self.i += 2
        while self.c() != "'":
            if self.c() == "":
                self.fail("unclosed $'")
            self.i += 2 if self.c() == "\\" else 1
        self.i += 1

    def dquote(self, scope):
        self.i += 1
        while True:
            ch = self.c()
            if ch == "":
                self.fail("unclosed double quote")
            if ch == "\\":
                self.i += 2
            elif ch == '"':
                self.i += 1
                return
            elif ch == "$":
                self.dollar(scope)
            elif ch == "`":
                self.fail("backtick")
            else:
                self.i += 1

    def dollar(self, scope):
        if self.at("$(("):
            self.i += 3
            depth = 0
            while True:
                ch = self.c()
                if ch == "":
                    self.fail("unclosed $((")
                if ch == ")" and depth == 0 and self.c(1) == ")":
                    self.i += 2
                    return
                depth += (ch == "(") - (ch == ")")
                m = NAME.match(self.s, self.i)
                if ch == "$":
                    self.dollar(scope)
                elif m and not self.s[self.i - 1].isalnum():
                    self.emit("read", m.group(), scope)
                    self.i = m.end()
                else:
                    self.i += 1
        elif self.at("$("):
            self.i += 2
            self.sid += 1
            self.cmds(scope + (self.sid,), ")")
        elif self.at("${"):
            self.i += 2
            if self.c() == "#" and NAME.match(self.s, self.i + 1):
                self.i += 1
            m = NAME.match(self.s, self.i)
            if m:
                self.emit("read", m.group(), scope)
                self.i = m.end()
            while True:
                ch = self.c()
                if ch == "":
                    self.fail("unclosed ${")
                if ch == "}":
                    self.i += 1
                    return
                if ch == "\\":
                    self.i += 2
                elif ch == '"':
                    self.dquote(scope)
                elif ch == "$" and self.c(1) == "'":
                    self.ansi()
                elif ch == "$":
                    self.dollar(scope)
                elif ch in "'`":
                    self.fail("a quote inside ${ }")
                else:
                    self.i += 1
        elif self.at("$'"):
            self.ansi()
        elif NAME.match(self.s, self.i + 1):
            m = NAME.match(self.s, self.i + 1)
            self.emit("read", m.group(), scope)
            self.i = m.end()
        elif self.c(1) and self.c(1) in "#?$!@*-0123456789":
            self.i += 2
        else:
            self.i += 1

    def word(self, scope):
        """One shell word; emits its reads; returns its text with quotes removed (for keywords and `read` names)."""
        out = []
        while True:
            ch = self.c()
            if ch == "" or ch in TERM:
                return "".join(out)
            if ch == "'":
                j = self.i
                self.squote()
                out.append(self.s[j + 1:self.i - 1])
            elif ch == '"':
                self.dquote(scope)
                out.append("\x00")
            elif ch == "\\":
                out.append(self.c(1))
                self.i += 2
            elif ch == "$":
                self.dollar(scope)
                out.append("\x00")
            elif ch == "`":
                self.fail("backtick")
            else:
                out.append(ch)
                self.i += 1

    def cmds(self, scope, closer):
        start, pending, case, expect, cmd = True, [], [], None, None

        def commit():
            for n in pending:
                self.emit("assign", n, scope)
            pending.clear()

        while True:
            ch = self.c()
            if ch == "":
                if closer:
                    self.fail("unclosed $( or (")
                commit()
                return
            if ch in " \t":
                self.i += 1
                continue
            if self.at("\\\n"):
                self.i += 2
                continue
            if ch == "#" and (self.i == 0 or self.s[self.i - 1] in " \t\n"):
                while self.c() not in ("\n", ""):
                    self.i += 1
                continue
            if ch == "\n" or self.at(";;") or self.at("&&") or self.at("||") or ch in ";|&":
                if case and case[-1] == "pat" and ch == "|":
                    self.i += 1
                    continue
                commit()
                if self.at(";;") and case:
                    case[-1] = "pat"
                self.i += 2 if (self.at(";;") or self.at("&&") or self.at("||")) else 1
                start, cmd = True, None
                continue
            if ch == ")":
                if case and case[-1] == "pat":
                    case[-1] = "body"
                    self.i += 1
                    start, cmd = True, None
                    continue
                if closer != ")":
                    self.fail("unbalanced )")
                commit()
                self.i += 1
                return
            if ch == "(":
                if not start:
                    self.fail("( not at a command start")
                self.i += 1
                self.sid += 1
                self.cmds(scope + (self.sid,), ")")
                start = False
                continue
            m = REDIR.match(self.s, self.i)
            if m and (ch in "<>" or m.group(0) != m.group(1)):
                if self.s.startswith("<<", self.i) and not self.s.startswith("<<<", self.i):
                    self.fail("heredoc")
                self.i = m.end()
                while self.c() in " \t":
                    self.i += 1
                self.word(scope)
                continue
            if case and case[-1] == "pat":
                w = self.word(scope)
                if w == "esac":
                    case.pop()
                    start = False
                continue
            a = ASSIGN.match(self.s, self.i)
            if start and a and expect is None:
                self.i = a.end()
                self.word(scope)
                pending.append(a.group(1))
                continue
            w = self.word(scope)
            if expect == "forvar":
                self.emit("assign", w, scope)
                expect = "forin"
                continue
            if expect == "forin":
                expect = None if w == "in" else self.fail("for without in")
                continue
            if expect == "casesubj":
                expect = "casein"
                continue
            if expect == "casein":
                if w != "in":
                    self.fail("case without in")
                case.append("pat")
                expect = None
                continue
            if start:
                pending.clear()  # assignments before a command word are that command's environment only
                if w in OPEN_KW:
                    continue
                if w in ("fi", "done"):
                    start = False
                    continue
                if w == "esac" and case:
                    case.pop()
                    start = False
                    continue
                expect = {"case": "casesubj", "for": "forvar"}.get(w)
                start, cmd = False, w
                continue
            if cmd == "read" and not w.startswith("-") and NAME.fullmatch(w):
                self.emit("assign", w, scope)


def analyze(text):
    sc = Scan(text)
    sc.cmds((), None)
    top, free, seen = [], [], []
    for kind, name, scope in sc.ev:
        if kind == "assign":
            seen.append((name, scope))
            if scope == () and name not in top:
                top.append(name)
        elif not any(n == name and scope[:len(s)] == s for n, s in seen) and name not in free:
            free.append(name)
    return top, free


def extract(src, block):
    p = Path(src)
    if p.exists():
        text = p.read_text()
    else:
        r = subprocess.run(["git", "show", src], capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"block_vars: cannot read {src}")
        text = r.stdout.decode()
    lines = text.split("\n")
    a, b = ((f"<!-- BEGIN {block} -->", f"<!-- END {block} -->") if block.startswith("VARS ")
            else (f"    # BEGIN {block}", f"    # END {block}"))
    if lines.count(a) != 1 or lines.count(b) != 1:
        raise SystemExit(f"block_vars: {block} is not delimited exactly once in {src}")
    body = lines[lines.index(a) + 1:lines.index(b)]
    if block.startswith("VARS "):
        body = [x for x in body if not x.startswith("```")]
    return "\n".join(body) + "\n"


def chain(label, spec, md):
    blocks = []
    for part in spec.split(","):
        src, _, blk = part.rpartition("#")
        top, free = analyze(extract(src, blk))
        blocks.append((blk, blk.startswith("VARS "), top, free))
    vars_names = {n for b in blocks if b[1] for n in b[2]}
    bad, rows = [], []
    for k, (blk, isv, top, free) in enumerate(blocks):
        notes = []
        for v in free:
            prov = [blocks[j][0] for j in range(k) if v in blocks[j][2]]
            if not prov:
                if v not in ENV:
                    bad.append(f"UNPROVIDED {blk}: {v}")
                    notes.append(f"{v}: UNPROVIDED")
            elif not prov[-1].startswith("VARS "):
                bad.append(f"COLLISION {blk}: {v} last set by {prov[-1]} (all: {', '.join(prov)})")
                notes.append(f"{v}: COLLISION ({', '.join(prov)})")
            elif len(prov) > 1:
                notes.append(f"{v}: from {prov[-1]} (also set earlier by {', '.join(prov[:-1])})")
        shadows = [] if isv else [n for n in top if n in vars_names]
        rows.append((blk, top, free, shadows, notes))
    names = " → ".join(b[0] for b in blocks)
    if md:
        print(f"\n**Shell `{label}`:** {names}\n")
        print("| Block | Assigns at top level (persist) | Free reads (value from an earlier block or the environment) | Shadows a VARS name | Flags |")
        print("|---|---|---|---|---|")
        for blk, top, free, shadows, notes in rows:
            print(f"| {blk} | {' '.join(top) or '—'} | {' '.join(free) or '—'} | {' '.join(shadows) or '—'} | {'; '.join(notes) or 'none'} |")
        print(f"\nResult for `{label}`: {'; '.join(bad) if bad else 'no COLLISION and no UNPROVIDED read'}.")
    else:
        print(f"chain {label}: {names}")
        for blk, top, free, shadows, notes in rows:
            print(f"  {blk}: assigns {len(top)}, free {len(free)} [{' '.join(free)}], shadows [{' '.join(shadows)}]"
                  + (f", flags: {'; '.join(notes)}" if notes else ""))
        print(f"  RESULT {label}: " + ("; ".join(bad) if bad else "OK"))
    return bad


def main():
    args = sys.argv[1:]
    md = "--md" in args
    bad = []
    for a in (x for x in args if x != "--md"):
        label, _, spec = a.partition("=")
        bad += chain(label, spec, md)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
