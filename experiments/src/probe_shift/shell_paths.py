"""Shell-command file events for scripted behaviour measures (first used by experiments/2026-10-11_q2_n2_honest).
`events(tool, args)` turns one tool call into ordered (kind, path) pairs, kind in {"read", "write"}: text_editor view is a
read and create/str_replace/insert a write; a shell command is tokenized (shlex, after joining line continuations and
dropping heredoc bodies and # comments) and split into simple commands. Reads are the file operands of READERS (listings
via ls included), cp sources and input redirections; writes are output redirections, tee, touch, sed -i and the cp/mv destination.
`cd` within one command is followed. Paths are normalized lexically; paths under /proc/ and /dev/ are dropped, except under /dev/shm/.
Raises ValueError on unbalanced quotes. Not handled: quoted operator tokens, numeric operands before a redirection, >&
with a file name, per-command option values, --, cp -t, directory destinations without a trailing slash, subshell scope.
"""
import os
import re
import shlex

READERS = {"cat", "head", "tail", "less", "more", "grep", "wc", "diff", "stat", "md5sum", "sha256sum", "jq", "ls", "nl", "awk", "sed"}  # ls: "listed"
PATTERN_FIRST = {"grep", "jq", "awk", "sed"}  # the first non-option argument is a pattern, filter or script, not a file
SUDO_VALUE = {"-u", "-g", "-C", "-D", "-h", "-p", "-U", "-r", "-t"}
PREFIXES = {"sudo", "time", "nohup", "env", "command", "exec"}
REDIR_W, REDIR_R, REDIR_SKIP = {">", ">>", ">|", "&>", "&>>"}, {"<"}, {">&", "<&", "<<", "<<-", "<<<"}
HEREDOC = re.compile(r"(?<!<)<<-?(?!<)\s*(['\"]?)([^\s'\"<>;&|()]+)\1")  # not <<< (a here-string)


def cmd_of(tool: str, args: dict) -> str:
    if tool == "text_editor":
        return str(args.get("path") or "")
    c = args.get("cmd") or args.get("command") or ""
    return c if isinstance(c, str) else ""


def strip_comments(cmd: str) -> str:
    """Drop # comments (a # that starts a word, outside quotes) up to the end of the line."""
    out, q, i = [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if c == "\\" and q != "'" and i + 1 < len(cmd):
            out.append(cmd[i:i + 2]); i += 2; continue
        if q:
            q = None if c == q else q
        elif c in "'\"":
            q = c
        elif c == "#" and (i == 0 or cmd[i - 1] in " \t\n;&|()"):
            j = cmd.find("\n", i); i = len(cmd) if j < 0 else j; continue
        out.append(c); i += 1
    return "".join(out)


def shell_events(cmd: str, depth: int = 0) -> list[tuple[str, str]]:
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
            elif w[0] in PREFIXES or ("=" in w[0] and not w[0].startswith("-")):
                w = w[1:]
            else:
                break
        if not w:
            return
        name, args = os.path.basename(w[0]), w[1:]
        files = [a for a in args if not a.startswith("-")]
        shell_c = [j for j, a in enumerate(args[:-1]) if re.fullmatch(r"-[a-z]*c[a-z]*", a)]
        if name in ("bash", "sh") and shell_c:
            ev.extend((k, at(p)) for k, p in shell_events(args[shell_c[0] + 1], depth + 1))
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


def events(tool: str, args: dict) -> list[tuple[str, str]]:
    if tool == "text_editor":
        p, c = args.get("path"), args.get("command")
        ev = [] if not p else [("read", p)] if c == "view" else [("write", p)] if c in ("create", "str_replace", "insert") else []
    else:
        ev = shell_events(cmd_of(tool, args))
    ev = [(k, os.path.normpath(re.sub(r"^/{2,}", "/", p))) for k, p in ev if p]
    return [(k, p) for k, p in ev if not (p.startswith("/proc/") or (p.startswith("/dev/") and not p.startswith("/dev/shm/")))]
