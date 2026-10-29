"""A file's contents from a run's earlier tool calls, for the blinded context supplement (2026-10-12; fixed before any run).

Only calls recorded before the command's turn count, oldest first, and only those in the file's container (bash and
text_editor share the default one; db_bash, client_bash, ... each have their own). Nothing is executed.
- text_editor create / str_replace (old_str exactly once) / insert (new_str or insert_text) on a complete base, with a
  result that is neither an error nor error-like: applied; the editor expands tabs when it edits, so any tab is
  evidence_ambiguous. A missing result or an error other than the editor's pre-write refusals: evidence_ambiguous. A
  full view (no view_range, untruncated, headed with this path, a sequential `cat -n` gutter): read as displayed (tabs
  may show as spaces: display_normalized); a failed or unreadable one, an edit through a relative or ".." path, or any
  other command: evidence_ambiguous.
- bash `cat F` alone: read if error-free, untruncated and without a "cat:"/"bash:" line, else evidence_ambiguous.
  `cat >F <<D` or `cat <<D >F` (quoted D, or a body without $ ` \\; not <<-) starting a command: written (>> appends) if
  it printed nothing, or if what follows the body neither changes F nor printed a shell or write error.
- Any bash-type call, in any container, that could have changed F: possibly_modified_after_evidence (no contents).
  That is one that names F's basename as a redirect, -o/-O/--output target (wrappers' too), or other than as a read (cat, head, grep, ... ; sed without -i/--in-place naming F
  only as a plain file argument; cp's source with an absolute destination elsewhere; chmod; a safe git subcommand) or
  a run of it as a script; a glob (* ? [) with rm/mv/cp/perl/find or sed -i (any -…i… cluster); a parent of F given to
  anything but a read; git with any other subcommand (after its global options); xargs, eval or parallel; a quoted
  or backquoted command substitution; patch, rsync, unzip, tar x or find -delete/-exec; or such a command inside `sh -c`.
- More than one call touching F in one assistant message (they may run concurrently), or any other call in the
  command's own turn that could change F: evidence_ambiguous.
- other_programs_between (Aswin, about 18:00 PT 2026-10-12: "2e's rule + a flag"): true if any tool call other than a
  read, create or edit of F ran between F's last full read or write and the command (an edit of F keeps the flag, a
  later full read or write clears it), or if any call other than a read of F shares the command's turn. It is only a
  flag: programs run in between are assumed not to change F unless the rules above say they could have.
"""

import posixpath
import re
import shlex
from collections import Counter

from context_parse import INTERP, UNSAFE, bash_cmd, script_of, segments, words

TRUNC = ("was too long to be displayed", "<response clipped>")
ERRLIKE = re.compile(r"\s*(error|failed|invalid|no replacement|traceback|exception)\b", re.I)
PREWRITE = re.compile(r"\s*(No replacement was performed|The path \S+ (does not exist|is a directory|is not an absolute path)"
                      r"|Invalid `insert_line`|File already exists|Path is required|file_text is required|insert_line is required"
                      r"|str_replace: The `old_str` parameter cannot be empty|No edit history)")
SHELL_ERR = re.compile(r"(^|\n)(cat|bash|sh|zsh|tee)\b[^\n]*:|No such file or directory|Permission denied|Is a directory|Read-only"
                       r"|No space left|write error|cannot create")
READ = {"cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "wc", "ls", "diff", "stat", "file", "md5sum",
        "sha256sum", "sha1sum", "nl", "od", "hexdump", "strings", "cmp", "test", "[", "chmod", "realpath", "readlink",
        "dirname", "basename", "du", "sed", "cd", "pushd", "popd", "git"}
VIEWERS = {"cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "wc", "nl", "od", "hexdump", "strings", "md5sum",
           "sha256sum", "sha1sum", "diff", "cmp", "sed", "stat", "file", "ls", "cd"}
GIT_SAFE = {"diff", "log", "show", "status", "add", "blame", "ls-files", "grep", "commit", "init", "config", "branch", "fetch",
            "remote", "rev-parse", "tag", "push", "describe", "shortlog", "ls-remote", "clone"}
GIT_VALUED = {"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--config-env"}
BULK = re.compile(r"\bgit(\s+(-[cC]\s+\S+|--[\w-]+(=\S+)?|-\w+))*\s+(checkout|reset|stash|restore|apply|pull|merge|rebase|clean|am"
                  r"|cherry-pick|revert|switch|mv|rm)\b"
                  r"|\b(patch|rsync|unzip)\b|\btar\s+-?\w*x|-(delete|exec|execdir|ok)\b")
HD1 = re.compile(r"cat\s*(>>?)\s*(\S+)\s*<<(-?)\s*(['\"]?)(\w+)\4")
HD2 = re.compile(r"cat\s*<<(-?)\s*(['\"]?)(\w+)\2\s*(>>?)\s*(\S+)")
container = lambda tool: "default" if tool in ("bash", "text_editor") else tool  # noqa: E731


def gutter(out: str, shown: tuple):
    """The file text from a full `cat -n` view of one of the `shown` paths, or None if anything doesn't fit."""
    head, _, body = out.partition("\n")
    if head not in {f"Here's the result of running `cat -n` on {p}:" for p in shown} or not body.endswith("\n"):
        return None
    lines = []
    for k, ln in enumerate(body[:-1].split("\n"), 1):
        m = re.fullmatch(r" *(\d+)\t(.*)", ln, re.S)
        if not m or int(m.group(1)) != k:
            return None
        lines.append(m.group(2))
    return "\n".join(lines)


def heredoc(cmd: str):
    """(op, absolute target, body, literal, rest of the command) for `cat >F <<D ... D ...`, else None."""
    head, _, rest = cmd.strip().partition("\n")
    m1, m2 = HD1.fullmatch(head.strip()), HD2.fullmatch(head.strip())
    if not (m1 or m2):
        return None
    op, tgt, dash, q, d = m1.groups() if m1 else (m2[4], m2[5], m2[1], m2[2], m2[3])
    tgt, lines = tgt.strip("'\""), rest.split("\n")
    if d not in lines or not tgt.startswith("/") or ".." in tgt.split("/") or UNSAFE & set(tgt):
        return None
    body = "".join(ln + "\n" for ln in lines[:lines.index(d)])
    literal = bool(q or not re.search(r"[$`\\]", body)) and not dash
    return op, posixpath.normpath(tgt), body, literal, "\n".join(lines[lines.index(d) + 1:])


def git_sub(toks: list) -> str:
    i = 1
    while i < len(toks) and toks[i].startswith("-"):
        i += 2 if toks[i] in GIT_VALUED else 1
    return toks[i] if i < len(toks) else ""


def modifies(path: str, cmd: str, depth: int = 0) -> bool:
    """Could this bash command have changed `path`? Conservative: anything but a read of it or a run of it."""
    if BULK.search(cmd):
        return True
    base, segs = posixpath.basename(path), segments(cmd)
    names = lambda t: bool(re.search(rf"(?<![\w.-]){re.escape(base)}(?![\w.-])", t or ""))  # noqa: E731
    if segs is None:
        return names(cmd)
    if any("$(" in t or "`" in t for s in segs if isinstance(s, list) for t in s):
        return True  # a quoted or backquoted command substitution stays one token, so it isn't parsed: it could run anything
    seen = False
    for raw, toks in ((s, words(s)) for s in segs if isinstance(s, list) and words(s)):  # raw keeps wrappers' own -o targets
        w, named = posixpath.basename(toks[0]), [t for t in toks if names(t)]
        tgts = [raw[j + 1] for j, t in enumerate(raw[:-1]) if t.startswith(">") or t in ("-o", "-O", "--output")] + [t[9:] for t in raw if t.startswith("--output=")]
        parent = any(t.startswith("/") and path.startswith(posixpath.normpath(t).rstrip("/") + "/") for t in toks)
        inplace = w == "sed" and any(t.startswith("--in-place") or (t[:1] == "-" and t[:2] != "--" and "i" in t[1:]) for t in toks)
        glob = any(set(t) & set("*?[") for t in toks) and (w in ("rm", "mv", "cp", "perl", "find") or inplace)
        kind, v = script_of(toks) if INTERP.fullmatch(w) else (None, None)
        if (w in ("xargs", "eval", "parallel") or (w == "git" and git_sub(toks) not in GIT_SAFE) or any(names(t) for t in tgts)
                or glob or (inplace and (named or parent)) or (w not in READ and parent)
                or (kind == "inline" and v and depth < 1 and modifies(path, v, depth + 1))):
            return True
        if not named:
            continue
        seen = True
        if w == "sed" and any(not re.fullmatch(r"[\w./+@-]+", t) or re.match(r"[wW]/", t) for t in named):
            return True
        args = [t for t in toks[1:] if not t.startswith("-")]
        cp_ok = w == "cp" and len(args) >= 2 and not any(t == "-t" or t.startswith("--target-directory") for t in toks) \
            and args[-1].startswith("/") and ".." not in args[-1].split("/") and not names(args[-1]) \
            and posixpath.normpath(args[-1]) != posixpath.dirname(path)
        once = len(named) == 1
        if not ((w in READ and w != "cp") or (kind == "script" and names(v) and once) or (names(toks[0]) and once) or cp_ok):
            return True
    return names(cmd) and not seen  # named only inside a heredoc body or similar


def may_write(path: str, cont: str, tool: str, a: dict) -> bool:
    """Could this call (whatever its result) have changed `path`?"""
    p = a.get("path")
    if tool == "text_editor":
        odd = not isinstance(p, str) or not p.startswith("/") or ".." in p.split("/")
        return container(tool) == cont and a.get("command") != "view" and (odd or posixpath.normpath(p) == path)
    hd = heredoc(bash_cmd(a)) if tool.endswith("bash") and container(tool) == cont else None
    return tool.endswith("bash") and ((hd is not None and hd[1] == path) or modifies(path, bash_cmd(a)))


def reads_file(path: str, cont: str, tool: str, a: dict) -> bool:
    """Is this call only a read (full or partial) of `path`, in its container?"""
    p = a.get("path")
    if tool == "text_editor":
        return container(tool) == cont and a.get("command") == "view" and isinstance(p, str) and p.startswith("/") \
            and ".." not in p.split("/") and posixpath.normpath(p) == path
    segs = segments(bash_cmd(a)) if tool.endswith("bash") and container(tool) == cont else None
    if not segs or modifies(path, bash_cmd(a)):
        return False
    cmds = [t for t in (words(s) for s in segs if isinstance(s, list)) if t]
    named = any(t.startswith("/") and ".." not in t.split("/") and posixpath.normpath(t) == path for c in cmds for t in c[1:])
    return named and all(posixpath.basename(c[0]) in VIEWERS for c in cmds)


def rebuild(path: str, cont: str, steps, siblings=()):
    """steps: [(message index, tool, args, output or None, error)] before the command's turn; siblings: [(tool, args)]
    of the other calls in its turn. Returns (contents, complete, reason, message indices used, display_normalized,
    other_programs_between)."""
    state, reason, used, per_msg, viewed, unverified = None, "no_in_run_evidence", [], Counter(), False, False
    for k, tool, a, out, err in steps:
        touch, mine, fresh, p, hd = True, True, False, a.get("path"), None  # mine: reads or writes this file; fresh: fully
        if tool == "text_editor" and container(tool) == cont and a.get("command") != "view" and may_write(path, cont, tool, a) \
                and (not isinstance(p, str) or posixpath.normpath(p) != path):
            mine, state, reason = False, None, "evidence_ambiguous"  # an edit through a relative or ".." path could be this file
        elif tool == "text_editor" and container(tool) == cont and isinstance(p, str) and posixpath.normpath(p) == path:
            c, o = a.get("command"), a.get("old_str")
            n = a.get("new_str", "") if c == "str_replace" else a.get("new_str", a.get("insert_text"))
            if not p.startswith("/") or ".." in p.split("/"):
                state, reason = None, "evidence_ambiguous"
            elif c == "view":
                if err:
                    state, reason = None, "evidence_ambiguous"
                elif out is None or a.get("view_range"):
                    pass  # a partial or unrecorded read
                elif any(m in out for m in TRUNC) or (g := gutter(out, (p, path))) is None:
                    state, reason = None, "evidence_truncated" if any(m in out for m in TRUNC) else "evidence_ambiguous"
                else:
                    state, reason, viewed, fresh = g, None, True, True
            elif out is None or (err and not PREWRITE.match(out)) or (not err and ERRLIKE.match(out)):
                state, reason = None, "evidence_ambiguous"
            elif err:
                pass  # the editor refused before writing
            elif c == "create" and isinstance(a.get("file_text"), str):
                state, reason, viewed, fresh = a["file_text"], None, False, True
            elif c in ("str_replace", "insert") and state is None:
                reason = "edit_without_complete_base"
            elif c == "str_replace" and isinstance(o, str) and isinstance(n, str) and "\t" not in state + o + n and state.count(o) == 1:
                state = state.replace(o, n, 1)
            elif c == "insert" and isinstance(n, str) and type(a.get("insert_line")) is int and "\t" not in state + n \
                    and 0 <= a["insert_line"] <= len(state.split("\n")):
                ls = state.split("\n")
                state = "\n".join(ls[:a["insert_line"]] + n.split("\n") + ls[a["insert_line"]:])
            else:
                state, reason = None, "evidence_ambiguous"
        elif tool.endswith("bash"):
            cmd, same = bash_cmd(a), container(tool) == cont
            hd = heredoc(cmd) if same else None
            try:
                toks = shlex.split(cmd) if "\n" not in cmd.strip() else None
            except ValueError:
                toks = None
            if same and toks and len(toks) == 2 and toks[0] == "cat" and toks[1].startswith("/") and posixpath.normpath(toks[1]) == path \
                    and ".." not in toks[1].split("/"):
                if err or (out is not None and re.search(r"(^|\n)(cat|bash|sh): ", out)):
                    state, reason = None, "evidence_ambiguous"
                elif out is None:
                    pass  # an unrecorded read
                elif any(m in out for m in TRUNC):
                    state, reason = None, "evidence_truncated"
                else:
                    state, reason, viewed, fresh = out, None, False, True
            elif hd and hd[1] == path:
                op, _, body, literal, tail = hd
                if out is None or err or not literal or (out.strip() and not tail.strip()) or SHELL_ERR.search(out) \
                        or (tail.strip() and modifies(path, tail)):
                    state, reason = None, "evidence_ambiguous"
                elif op == ">":
                    state, reason, viewed, fresh = body, None, False, True
                elif state is not None:
                    state += body
                else:
                    reason = "edit_without_complete_base"
            elif modifies(path, cmd):
                mine, state, reason = False, None, "possibly_modified_after_evidence" if state is not None else reason
            else:
                touch, mine = False, reads_file(path, cont, tool, a)
        else:
            touch = mine = False
        unverified = False if fresh else unverified or not mine
        if hd and hd[1] == path and hd[4].strip():
            unverified = True  # commands ran after the heredoc in the same call
        if touch:
            used.append(k)
            per_msg[k] += 1
            if per_msg[k] > 1:
                state, reason = None, "evidence_ambiguous"
    if state is not None and any(may_write(path, cont, t, a) for t, a in siblings):
        state, reason = None, "evidence_ambiguous"
    between = unverified or any(not reads_file(path, cont, t, a) for t, a in siblings)
    return state, state is not None, None if state is not None else reason, used, viewed and state is not None, between


def selftest() -> int:
    P, N, B = "/app/a.py", (None, None), ("k", True)  # (contents, other_programs_between if contents); N: none; B: k, flagged
    te = lambda c, k=0, **a: (k, "text_editor", {"command": c, "path": P, **a})  # noqa: E731
    sh = lambda cmd, k=0, t="bash": (k, t, {"command": cmd})  # noqa: E731
    view = "Here's the result of running `cat -n` on /app/a.py:\n     1\tx = 1\n     2\ty = 2\n     3\t\n"
    mk = (*te("create", file_text="k"), "File created", False)
    cases = [
        ([(*te("create", file_text="x = 1\ny = 2"), "File created", False)], ("x = 1\ny = 2", False)),
        ([(*te("create", file_text="x = 1\n"), "ok", False), (*te("str_replace", 1, old_str="1", new_str="3"), "ok", False)], ("x = 3\n", False)),
        ([(*te("str_replace", old_str="1", new_str="3"), "ok", False)], N),
        ([(*te("view"), view, False)], ("x = 1\ny = 2\n", False)),
        ([(*te("view"), view.replace("/app/a.py", "/app/b.py"), False)], N),
        ([(*te("create", file_text="a\n\tb"), "ok", False), (*te("str_replace", 1, old_str="a", new_str="c"), "ok", False)], N),
        ([(*te("create", file_text="a"), "ok", False), (*te("str_replace", 1, old_str="a", new_str="c"), "No replacement was performed", True)], ("a", False)),
        ([(*te("create", file_text="a"), "ok", False), (*te("str_replace", 1, old_str="a", new_str="c"), "timed out", True)], N),
        ([(*te("create", file_text="a"), "ok", False), (*te("str_replace", 1, old_str="a", new_str="c"), None, False)], N),
        ([(*te("create", file_text="a\nb"), "ok", False), (*te("insert", 1, insert_line=1, new_str="m"), "ok", False)], ("a\nm\nb", False)),
        ([(*te("create", file_text="a\nb"), "ok", False), (*te("insert", 1, insert_line=1, insert_text="m"), "ok", False)], ("a\nm\nb", False)),
        ([(*sh("cat /app/a.py"), "q\n", False)], ("q\n", False)),
        ([(*sh("cat /app/a.py"), "q\ncat: /app/a.py: Input/output error\n", False)], N),
        ([(*sh("cat > /app/a.py << 'EOF'\nz = $X\nEOF"), "", False)], ("z = $X\n", False)),
        ([(*sh("cat <<EOF > /app/a.py\nz = $X\nEOF"), "", False)], N),
        ([(*sh("cat > /app/a.py <<'EOF'\nk\nEOF\npython3 /app/a.py"), "out", False)], ("k\n", True)),
        ([(*sh("cat > '/app/a.py' <<'EOF'\nk\nEOF\nls"), "bash: line 1: /app/a.py: No such file or directory\n", False)], N),
        ([(*sh("cat > /app/a.py <<'EOF'\nk\nEOF\necho j >> /app/a.py"), "", False)], N),
        ([(*sh("cat > /app/a.py <<'EOF'\nk\nEOF\ntrue"), "cat: write error: No space left on device", False)], N),
        ([mk, (*sh("cd /app && cp a.py /tmp/a.bak && python3 a.py 2>&1 | tail > out.log", 1), "", False)], B),
        ([mk, (*sh("ls /tmp", 1), "", False), (*te("view", 2), "Here's the result of running `cat -n` on /app/a.py:\n     1\tk\n", False)], ("k", False)),
        ([mk, (2, "think", {}, None, False)], B),
        ([mk, (*te("view", 1, view_range=[1, 1]), "...", False)], ("k", False)),
        ([mk, (*sh("cat /app/a.py", 0, "db_bash"), "zz", False)], B),
        ([(*sh("cat /app/a.py", 0, "db_bash"), "zz", False)], N),
        ([mk, (*te("view", 0), "Here's the result of running `cat -n` on /app/a.py:\n     1\tk\n", False)], N),
        ([(*sh("echo k > /app/a.py", 0), "", False), (*sh("cat /app/a.py", 0), "j", False)], N),
        ([mk, (*sh("cat /app/a.py", 1), "cat: /app/a.py: No such file or directory", False)], N),
        ([mk, (*sh("cat /app/a.py", 1), None, True)], N),
        ([mk, (*te("view", 1), "The path /app/a.py does not exist.", True)], N),
        ([mk, (*te("view", 1), None, True)], N),
        ([mk, (*te("create", 1, path="/app/x/../a.py", file_text="j"), "ok", False)], N),
        ([mk, (1, "text_editor", {"command": "create", "file_text": "j"}, "ok", False)], N),
    ]
    writes = ["sed -i s/k/j/ a.py", "python3 -c \"open('a.py', 'w')\"", "rm -rf /app/", "python3 - <<'E'\nopen('/app/a.py','w')\nE",
              "git checkout .", "sort -o /app/a.py /app/a.py", "cd /app && find . -name x -delete", "timeout 5 sed -i s/k/j/ /app/a.py",
              "cat /o>/app/a.py", "cat /o >&/app/a.py", "sed --in-place=.bak s/k/j/ /app/a.py", "sed -n 'w /app/a.py' /o",
              "git -c core.x=1 reset --hard", "git --git-dir .git reset --hard", "sudo -n rm /app/a.py", "sed -i s/a/b/ /app/*.py",
              "cp /o /app/a.py --force", "cd /app && cp /tmp/a.py .", "bash -c 'rm -rf /app'", "ls | xargs rm", "printf '' x > /app/a.py",
              "sed -Ei 's/k/j/' /app/a.py", "rm /app/[ab].py", "env time -o /app/a.py cat /app/a.py", "curl -o /app/a.py http://x",
              "cat \"$(printf new > /app/a.py; printf /dev/null)\"", "X=`rm /app/*`; cat /app/a.py"]
    cases += [([mk, (*sh(w, 1), "", False)], N) for w in writes]
    for i, (steps, want) in enumerate(cases):
        r = rebuild(P, "default", steps)
        assert r[0] == want[0] and r[1] == (r[0] is not None) and (r[0] is None or r[5] == want[1]), ("evidence selftest case", i)
    sib = lambda c, t="bash": rebuild(P, "default", [mk], [(t, {"command": c})])  # noqa: E731
    checks = [sib("echo j > /app/a.py")[:3] + sib("echo j > /app/a.py")[5:] == (None, False, "evidence_ambiguous", True),
              sib("python3 /app/a.py")[:3] + sib("python3 /app/a.py")[5:] == ("k", True, None, True), sib("cat /app/a.py")[5] is False,
              sib("cd /app && head -5 /app/a.py | grep x")[5] is False, sib("cat /app/a.py", "db_bash")[5] is True,
              rebuild(P, "default", [mk, (*sh("head /app/a.py", 1), "k", False)])[5] is False, rebuild(P, "default", [(*te("view"), view, False)])[4]]
    assert all(checks), ("evidence selftest check", checks.index(False))
    return len(cases) + len(checks)
