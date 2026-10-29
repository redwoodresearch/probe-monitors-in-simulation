"""N2, post hoc extractor check (Codex threads on #682/#683; not in the plan; free). Two read-back extractor gaps found
after the run of record, measured here without touching the as-run code:
- change 1: a simple command led by a shell keyword (if, then, elif, else, while, until, do, {, !) now dispatches the
  command after the keyword, so reads and writes inside conditionals and loops count;
- change 2: text_editor undo_edit counts as a write.
`shell_events_v2` is a copy of probe_shift.shell_paths.shell_events (83c30b7e) with change 1 marked; everything else,
validation and statistics included, is the as-run analyze_n2_honest (fb25be88 logic, imported). Its `events` is swapped
for `events_v2` and its output stamp gets the suffix _extractor-v2_POSTHOC, so outputs never take the run of record's
names. The third thread (unbounded rsyslog/logrotate) matched 0 log-path hits in either arm and changes nothing.

  uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python experiments/2026-10-11_q2_n2_honest/analyze_n2_extractor_v2_POSTHOC.py <transcript cache>
"""
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import analyze_n2_honest as n2  # noqa: E402
from probe_shift.shell_paths import HEREDOC, PATTERN_FIRST, PREFIXES, READERS, REDIR_R, REDIR_SKIP, REDIR_W, SUDO_VALUE, cmd_of, strip_comments  # noqa: E402

KEYWORDS = {"if", "then", "elif", "else", "while", "until", "do", "{", "!"}


def shell_events_v2(cmd: str, depth: int = 0) -> list[tuple[str, str]]:
    """Ordered (kind, path) from one shell command. Line continuations are joined, heredoc bodies and comments dropped,
    and the rest tokenized (quotes removed) and split into simple commands at ; && || | & ( ) and newlines. `cd` sets the
    directory for later relative paths in the same command (subshell scope is not tracked). Raises ValueError on
    unbalanced quotes or nesting deeper than two shells."""
    if depth > 2:
        raise ValueError("bash -c nested too deep")
    lines, pending = [], []
    for line in cmd.replace("\\\n", " ").split("\n"):
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        lines.append(line)
        pending = [m.group(2) for m in HEREDOC.finditer(line)]
    lex = shlex.shlex(strip_comments("\n;\n".join(lines)), posix=True, punctuation_chars=";&|()<>")
    lex.whitespace_split, lex.commenters = True, ""
    toks, ev, words, i, cwd = list(lex), [], [], 0, [None]
    at = lambda p: os.path.join(cwd[0], p) if cwd[0] and p and not p.startswith(("/", "~")) else p

    def flush(w):
        while w:
            if w[0] == "sudo":
                w = w[1:]
                while w and w[0].startswith("-"):
                    w = w[2:] if w[0] in SUDO_VALUE else w[1:]
            elif w[0] in PREFIXES or w[0] in KEYWORDS or ("=" in w[0] and not w[0].startswith("-")):  # change 1: KEYWORDS
                w = w[1:]
            else:
                break
        if not w:
            return
        name, args = os.path.basename(w[0]), w[1:]
        files = [a for a in args if not a.startswith("-")]
        shell_c = [j for j, a in enumerate(args[:-1]) if re.fullmatch(r"-[a-z]*c[a-z]*", a)]
        if name in ("bash", "sh") and shell_c:
            ev.extend((k, at(p)) for k, p in shell_events_v2(args[shell_c[0] + 1], depth + 1))
        elif name == "cd":
            cwd[0] = at(files[0]) if files else None
        elif name == "sed" and any(a.startswith(("-i", "--in-place")) for a in args):
            ev.extend(("write", at(a)) for a in files[1:])
        elif name in READERS:
            ev.extend(("read", at(a)) for a in (files[1:] if name in PATTERN_FIRST else files))
        elif name in ("tee", "touch"):
            ev.extend(("write", at(a)) for a in files)
        elif name in ("cp", "mv") and len(files) >= 2:
            ev.extend(("read", at(a)) for a in files[:-1] if name == "cp")
            dest = files[-1]
            ev.extend(("write", at(dest + os.path.basename(a.rstrip("/")))) for a in files[:-1]) if dest.endswith("/") else ev.append(("write", at(dest)))

    while i < len(toks):
        t = toks[i]
        if t in REDIR_W | REDIR_R | REDIR_SKIP:
            if words and words[-1].isdigit():  # the fd in 2>file or 2>&1
                words.pop()
            if t not in REDIR_SKIP and i + 1 < len(toks):
                ev.append(("write" if t in REDIR_W else "read", at(toks[i + 1])))
            i += 2
        elif t and set(t) <= set(";&|()"):
            flush(words); words = []; i += 1
        else:
            words.append(t); i += 1
    flush(words)
    return ev


def events_v2(tool: str, args: dict) -> list[tuple[str, str]]:
    if tool == "text_editor":
        p, c = args.get("path"), args.get("command")
        ev = [] if not p else [("read", p)] if c == "view" else [("write", p)] if c in ("create", "str_replace", "insert", "undo_edit") else []  # change 2
    else:
        ev = shell_events_v2(cmd_of(tool, args))
    ev = [(k, os.path.normpath(re.sub(r"^/{2,}", "/", p))) for k, p in ev if p]
    return [(k, p) for k, p in ev if not (p.startswith("/proc/") or (p.startswith("/dev/") and not p.startswith("/dev/shm/")))]


class _Stamp(datetime):
    @classmethod
    def now(cls, tz=None):
        t = datetime.now(tz)
        return type("S", (), {"strftime": lambda self, f: t.strftime(f) + "_extractor-v2_POSTHOC"})()


def checks():
    """The two changes on synthetic runs (exact read-back counts), before any count."""
    b, te = (lambda c: ("bash", {"cmd": c})), (lambda c, p: ("text_editor", {"command": c, "path": p}))
    cases = [([b("touch x; if cat x; then cat x; fi")], 2), ([b("touch x"), b("while read l; do cat x; done < x")], 2),
             ([b("touch x; { cat x; }")], 1), ([b("touch x; if ! grep -q a x; then echo; fi")], 1),
             ([b("for f in a b; do touch x; done; cat x")], 1), ([te("undo_edit", "/a.py"), te("view", "/a.py")], 1),
             ([b("echo fi > then.txt; cat then.txt")], 1)]
    ok = sum(n2.count_run(c)[2] == want for c, want in cases)
    if ok != len(cases):
        raise SystemExit(f"v2 checks failed: {ok} of {len(cases)}")
    return {"v2 synthetic runs exact": f"{ok} of {len(cases)}"}


if __name__ == "__main__":
    n2.events, n2.datetime = events_v2, _Stamp
    print(checks())
    n2.main(Path(sys.argv[1]).expanduser(), False)
