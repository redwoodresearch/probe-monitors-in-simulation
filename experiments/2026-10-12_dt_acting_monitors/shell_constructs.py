"""Which commands stay inside the shell grammar the context rules model, for the blinded context supplement (2026-10-12).

Codex's reviews kept finding shell spellings that context_parse.py and context_evidence.py misread, and lists of bad
constructs kept missing some, so this is an allowlist (the orchestrator: "anything it can't model must fail closed").
A row whose command, or any bash-type call from its earliest evidence on (from the run's start if the directory came
from `pwd`) or in the command's own turn, is outside it loses its contents (construct_outside_shell_model).
- Text: words of [A-Za-z0-9._/=:,@%+-], plus quoted text on one line with no < or > (double-quoted: no $ ` \\ either);
  no word with a ".." path part; a quoted word that is only operator characters is outside.
- Commands joined by && || | ; or newlines; a line can't end in an operator; no empty command.
- Redirections only after a command's words: > >> < (fd 0-2) or &>, to an unquoted plain word, or >&1 / >&2.
- Heredocs: one per line, outside quotes, an identifier delimiter ending its word, a closing line, a literal body.
- Each command: NAME=value assignments (not CDPATH, PATH, IFS, HOME, BASH_*, ENV, ZDOTDIR, ...), then an unquoted command
  word on the list below (a ./path or a *.sh/*.py/... path of [A-Za-z0-9_.+@/-] counts as a script run). sudo only with -n -E -H -u USER; env only
  with assignments; timeout DURATION; nohup and time bare; none of these may wrap cd, and none may be path-qualified.
  bash/sh/zsh/dash only as `bash SCRIPT ...` (not under /dev or /proc). cd and pushd take one plain argument and popd
  none, unqualified; in the row's own command only && may follow a cd (| or && after that).
  npm, yarn, pnpm and npx never in the row's own command. Writers (rm rmdir mv cp tee dd truncate) take only plain flags
  (after --, everything is an operand), no operand with //, and a relative operand that could name a row's file or one
  of its directories (".", or a run of its path parts) is outside, as is such a relative value of -o -O -d -t --output
  --target --target-directory --prefix --out-dir for any program. No flag cluster with an attached o/O value. find
  without -exec/-ok/-delete/-fprint/-fls; git without -c/--exec-path/--config-env/-o, config or (abbreviated) --output
  or --file; sed, perl and ruby without in-place backup suffixes or an abbreviated --in-place; make without long options
  in the row's own command; curl only with flags that write no file, or -o to a literal absolute path; no wget.
- Only printable ASCII, tabs and newlines; a redirection needs a space before it unless its fd digit is there, and its
  target ends at a space, ;, |, && or the line end and isn't all digits. A referenced path starting with // is outside.
Known limits (Codex's fourth review of this file was the last, by the orchestrator's call on the evening of 2026-10-12):
programs' own effects (a script, python -c, awk or sed programs, make recipes, npm scripts, an installer's target
directory, output-file conventions of programs not listed above) are out of scope, as in context_evidence.py, except the
output forms the evidence rules model; files reached through symlinks made before the window; and any spelling a later
review would find.
"""

import posixpath
import re

from context_parse import HEREDOC_OP

PLAIN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/=:,@%+-")
SHELLS = {"bash", "sh", "zsh", "dash"}
PROGRAMS = set("""pytest py.test pip pip3 node nodejs ts-node tsx ruby perl php java javac go cargo rustc gcc g++ cc clang make
npm yarn pnpm npx ls cat head tail less more grep egrep fgrep rg wc sort uniq cut tr diff cmp comm stat file du df md5sum
sha256sum sha1sum nl od hexdump xxd strings realpath readlink dirname basename pwd echo true false test sleep date whoami
id hostname uname which sed awk jq curl ps kill pkill pgrep lsof netstat ss free psql mysql sqlite3 redis-cli mongosh mongo
mkdir touch chmod chown git find""".split())
WRITERS = {"rm": "rfRivd", "rmdir": "pv", "mv": "fivnu", "cp": "rRfivnpau", "tee": "a", "dd": "", "truncate": ""}
PACKAGE = {"npm", "yarn", "pnpm", "npx"}
BAD_VARS = {"CDPATH", "PATH", "IFS", "HOME", "OLDPWD", "PWD", "ENV", "PS4", "PROMPT_COMMAND", "SHELLOPTS", "BASHOPTS", "GLOBIGNORE",
            "ZDOTDIR"}  # ZDOTDIR: zsh runs $ZDOTDIR/.zshenv at every start (Codex on #1150)
SCRIPT = re.compile(r"\./\S+|\S+\.(sh|bash|py|js|mjs|ts|rb|pl|php)")
CURL_FLAGS = set("sSfLvikI")
CURL_VALUED = {"-X", "-H", "-d", "-u", "-m", "-A", "-e", "-w", "--request", "--header", "--data", "--data-raw", "--data-binary",
               "--data-urlencode", "--json", "--user", "--max-time", "--connect-timeout", "--retry", "--user-agent", "--write-out"}
CURL_LONG = {"--silent", "--show-error", "--fail", "--location", "--insecure", "--compressed", "--head", "--include", "--verbose"}
OUT_OPTS = {"-o", "-O", "--output", "-d", "-t", "--target", "--target-directory", "--prefix", "--out-dir", "--outdir"}
FIND_BAD = re.compile(r"-(exec|execdir|ok|okdir|delete|fprint|fprint0|fprintf|fls)")
GIT_BAD = re.compile(r"-c.*|--exec-path.*|--config-env.*|-o.*")


def lex(cmd: str):
    """([(words, next operator)], None) inside the grammar, else (None, reason). Words are (text, quoted)."""
    if re.search(r"[^\x20-\x7e\t\n]", cmd):
        return None, "character"
    lines, kept, i = cmd.split("\n"), [], 0
    while i < len(lines):
        line, i, q, ops = lines[i], i + 1, None, []
        for k, c in enumerate(line):
            if q:
                q = None if c == q else q
            elif c in "'\"":
                q = c
            elif c == "\\":
                return None, "character"
            elif line.startswith("<<", k) and line[k - 1:k] != "<":
                ops.append(k)
        if q:
            return None, "multiline_quote"
        if not ops:
            kept.append(line)
            continue
        m = HEREDOC_OP.match(line, ops[0])
        if len(ops) > 1 or not m or line[m.start() - 1:m.start()] not in ("", " ", "\t") or line.startswith("<<<", ops[0]) \
                or (m.end() < len(line) and line[m.end()] not in " \t;&|>"):
            return None, "heredoc"
        dash, d = line[m.start() + 2:m.start() + 3] == "-", m.group(2)
        end = next((j for j in range(i, len(lines)) if lines[j] == d or (dash and lines[j].lstrip("\t") == d)), None)
        if end is None or not (m.group(1) or not re.search(r"[$`\\]", "\n".join(lines[i:end]))):
            return None, "heredoc"
        kept.append(line[:m.start()] + line[m.end():])
        i = end + 1
    text, segs, seg, word, quoted, redir, k = "\n".join(kept), [], [], None, False, False, 0

    def flush():
        if word is None:
            return None
        if quoted and set(word) <= set(";&|()<>"):
            return "quoting"
        if redir or ".." in word.split("/"):
            return "redirection" if redir else "dotdot"
        seg.append((word, quoted))
        return None

    while k < len(text):
        c = text[k]
        if c in PLAIN:
            word, k = (word or "") + c, k + 1
        elif c in "'\"":
            j = text.find(c, k + 1)
            body = text[k + 1:j]
            if j < 0 or "\n" in body or set(body) & set("<>") or (c == '"' and set(body) & set("$`\\")):
                return None, "quoting"
            word, quoted, k = (word or "") + body, True, j + 1
        elif c in "<>" or text.startswith("&>", k):
            fd = word is not None and not quoted and word in ("0", "1", "2") and c in "<>"
            if word is not None and not fd:  # a word glued to a redirection (the parser drops digits there)
                return None, "redirection"
            word, quoted = None, False
            r = re.match(r"(&>|>>|>|<)(&[12](?=[ \t\n;&|]|$))?", text[k:])
            if r is None or not seg or text.startswith(("<>", ">|", "<<"), k):
                return None, "redirection"
            k += r.end()
            if not r.group(2):
                t = re.match(r"[ \t]*([^ \t\n;&|<>'\"]+)(?=[ \t\n;|]|&&|$)", text[k:])
                if t is None or not set(t.group(1)) <= PLAIN or ".." in t.group(1).split("/") or t.group(1).isdigit():
                    return None, "redirection"
                k += t.end()
            redir = True
        elif c in " \t\n;&|":
            if why := flush():
                return None, why
            word, quoted = None, False
            if c in " \t":
                k += 1
                continue
            op = "&&" if text.startswith("&&", k) else "||" if text.startswith("||", k) else c if c in ";|\n" else None
            if op is None or text.startswith("|&", k) or (not seg and op != "\n"):
                return None, "operator"
            k += len(op)
            if not seg:
                continue  # a blank line, or the line after a ;
            if op in ("&&", "||", "|") and text[k:].lstrip(" \t")[:1] in ("", "\n"):
                return None, "operator"
            segs.append((seg, op))
            seg, redir = [], False
        else:
            return None, "character"
    if why := flush():
        return None, why
    return segs + ([(seg, None)] if seg else []), None


def assignment(w) -> bool:
    return bool(re.match(r"[A-Za-z_]\w*=", w[0])) and not w[1]


def bad_var(w) -> bool:
    name = w[0].split("=", 1)[0]
    return name in BAD_VARS or name.startswith("BASH_")


def names_dir(a: str, paths: list) -> bool:
    """Could relative operand `a` name one of `paths` or one of its directories (".", or a run of its path parts)?"""
    n = posixpath.normpath(a)
    parts = [] if n == "." else n.split("/")
    return not a.startswith("/") and any(parts == p.strip("/").split("/")[s:s + len(parts)]
                                         for p in paths for s in range(len(p.strip("/").split("/"))))


def prefix_of(a: str, *longs: str) -> bool:
    """Is `a` (before any =) an abbreviation of one of the long options (GNU tools accept unambiguous prefixes)?"""
    head = a.split("=", 1)[0]
    return len(head) > 2 and any(lo.startswith(head) for lo in longs)


def check(seg: list, op, paths: list, own: bool, wrapped: bool = False):
    """A reason this command is outside, or None."""
    while seg and assignment(seg[0]):
        if bad_var(seg[0]):
            return "variable"
        seg = seg[1:]
    if not seg:
        return None
    (w, q), args = seg[0], [a for a, _ in seg[1:]]
    name = w.rsplit("/", 1)[-1]
    special = {"sudo", "env", "nohup", "time", "timeout", "cd", "pushd", "popd"}
    known = name in PROGRAMS | set(WRITERS) | SHELLS | special or re.fullmatch(r"python[\d.]*", name)
    if q:
        return "quoted_command_word"
    if "/" in w and not re.fullmatch(r"[\w.+@/-]+", w):
        return "command"  # context_parse drops a path-like command word outside these characters without a reason (Bugbot on #1145)
    if "/" in w and (name in special or not known):
        return None if "/" in w and not known and SCRIPT.fullmatch(w) else "command"
    if any(re.fullmatch(r"-[oO].+", a) for a in args) and name in ("curl", "wget", "sort"):
        return "attached_output"
    if name == "sudo":
        j = 1
        while j < len(seg) and seg[j][0].startswith("-"):
            if seg[j][0] not in ("-n", "-E", "-H", "-u") or seg[j][1]:
                return "wrapper"
            j += 2 if seg[j][0] == "-u" else 1
        return check(seg[j:], op, paths, own, True) if j < len(seg) else "wrapper"
    if name == "env":
        j = 1
        while j < len(seg) and assignment(seg[j]):
            if bad_var(seg[j]):
                return "variable"
            j += 1
        return check(seg[j:], op, paths, own, True) if j < len(seg) else None
    if name in ("nohup", "time"):
        return check(seg[1:], op, paths, own, True) if args and not args[0].startswith("-") else "wrapper"
    if name == "timeout":
        return check(seg[2:], op, paths, own, True) if len(args) > 1 and re.fullmatch(r"[\d.]+[smhd]?", args[0]) else "wrapper"
    if name in ("cd", "pushd", "popd"):
        bad = wrapped or len(args) != (name != "popd") or any(a[:1] in "-+" for a in args) or (own and op not in ("&&", None))
        return "cd_form" if bad else None
    if name in SHELLS:
        s = args[0] if args else ""
        parts = [c for c in s.split("/") if c not in ("", ".")]
        ok = s and not seg[1][1] and s[:1] not in "-+" and ("/" in s or s.endswith((".sh", ".bash"))) \
            and not (parts[:1] in (["dev"], ["proc"]) or {"fd", "stdin"} & set(parts))
        return None if ok else "shell_code"
    if own and name in PACKAGE:
        return "package_lookup"
    if name in WRITERS:
        pos, skip, ended = [], False, False
        for a in args:
            if skip:
                skip = False
            elif not ended and name == "truncate" and a == "-s":
                skip = True
            elif not ended and a == "--":
                ended = True
            elif not ended and a.startswith("-") and a != "-":
                if not (a in ("--recursive", "--force", "--verbose") or (a[1:2] != "-" and set(a[1:]) <= set(WRITERS[name]))):
                    return "writer_option"
            else:
                pos.append(a.split("=", 1)[1] if name == "dd" and "=" in a else a)
        for a in pos:
            if "//" in a:
                return "writer_operand"
            if names_dir(a, paths):
                return "relative_target"
        return None
    if not known:
        return "command"
    if name == "find" and any(FIND_BAD.fullmatch(a) for a in args):
        return "find_action"
    if any(re.fullmatch(r"-[A-Za-z]*[oO].+", a) for a in args):
        return "attached_output"  # an output file glued into a flag cluster (-soa.py, -orun)
    outs = [b for a, b in zip(args, args[1:]) if a in OUT_OPTS] + [a.split("=", 1)[1] for a in args if "=" in a and a.split("=", 1)[0] in OUT_OPTS]
    if any(names_dir(b, paths) for b in outs):
        return "relative_target"
    if name == "git" and (any(GIT_BAD.fullmatch(a) or prefix_of(a, "--output", "--file") for a in args) or "config" in args):
        return "git_option"
    if name == "sed" and any((a.startswith("--") and prefix_of(a, "--in-place") and a != "--in-place")
                             or re.fullmatch(r"-[A-Za-z]*i.+", a) for a in args):
        return "sed_backup"  # an abbreviated --in-place, or an in-place backup suffix the evidence rules don't follow
    if name in ("perl", "ruby") and any(re.fullmatch(r"-[A-Za-z]*i.+", a) for a in args):
        return "inplace_backup"
    if name == "make" and own and any(a.startswith("--") for a in args):
        return "make_option"  # GNU make accepts abbreviations such as --dir=
    if name == "curl":
        j = 0
        while j < len(args):
            a = args[j]
            if a in CURL_VALUED or (a == "-o" and j + 1 < len(args) and args[j + 1].startswith("/") and "#" not in args[j + 1]):
                j += 2  # -o only to a literal absolute path (#N would take a value from the URL)
                continue
            if a.startswith("-") and not (a in CURL_LONG or (a[1:2] != "-" and set(a[1:]) <= CURL_FLAGS)):
                return "curl_option"  # only flags that write no file (no -o, -O, -J, -c, -D, -K, --output-dir, ...)
            j += 1
    return None


def outside(cmd: str, paths: list, own: bool) -> list:
    """Reasons this command is outside the modelled grammar ([] if inside)."""
    segs, why = lex(cmd)
    if why:
        return [why]
    found, after_cd = set(), False
    for seg, op in segs:
        if r := check(seg, op, paths, own):
            found.add(r)
        if own and after_cd and op not in ("&&", "|", None):
            found.add("after_cd")  # a later ; or || could run with the cd failed
        after_cd = after_cd or any(w.rsplit("/", 1)[-1] in ("cd", "pushd", "popd") for w, _ in seg)
    return sorted(found)


def hits(calls: list, command: str, paths: list) -> list:
    """Sorted reasons the row's window (calls) or its own command (command) leaves the modelled grammar."""
    if any(p.startswith("//") for p in paths):
        return ["double_slash_path"]  # normpath keeps a leading //, which Linux treats as /
    return sorted(set(r for c in calls for r in outside(c, paths, False)) | set(outside(command, paths, True) if command else []))


def selftest() -> int:
    P = ["/app/src/a.py"]
    outs = ["echo ';' cd /o", "'builtin' cd /o", "bash -c 'cd /o'", "bash <<'E'\nrm -rf /app\nE", "cd /a\n# don't\ncd /o",
            "builtin \\\ncd /o", "cat <<EOF-TAG\nEOF\ncd /o\nEOF-TAG", "X=1 source s.sh", "sudo -uroot -i python a.py",
            "rm -- \"$F\"", "printf x >& \"$F\"", "wget -O \"$F\" u", "cd /; rm -rf ./app", "cd /; rm -rf 'app'", "cp -r r/. .",
            "cat <<EOF\n'$(printf x > /a)'\nEOF", "env -Stime -o/a cat /a", "sudo su -c true", "su -c 'cd /; rm -rf app'",
            "cd /app; > /dev/null cd /o", "command -p cd /o", "cd /o#x", "CDPATH=/s cd p", "printf x > /app/[a].py",
            "truncate -s 0 /app/*.py", "head -c 1 /dev/zero 1<> /a", ": <<123\ncd /o\n123", ": <<A <<B\nx\nA\ny\nB",
            "function f {\ncd /o\n}", "x=$(rm /a)", "a &&\nb", "a | ", "true |& cd /x", "rm -rf src", "mv x src/a.py",
            "echo '<<EOF'\nrm -rf /app\nEOF", "cat <<'E' | bash\nrm -rf /app\nE", "timeout 5 bash <<'E'\nx\nE", "nice -n 5 bash -c x",
            "echo 'a\\\nb'", "printf '\n<<E\n'; cd /; rm -rf app; printf '\nE\n'", "0<<'E' 'bash'\ncd /\nE",
            "10>/tmp/log 'bash' -c 'x'", "cd /app; X=1 >/dev/null cd /o", "cd 2>/dev/null /o && python a.py", "true >&1-",
            "true >&12", "true > /tmp/x' y'", "nice 'bash' -c 'x'", "bash +x -c 'x'", "printf 'x' | bash /dev/stdin", "ash -c x",
            "busybox ash -c x", "docker run --entrypoint=sh alpine -c x", "export CDPATH=/tmp", "CDPATH+=/tmp", "BASH_ENV=/h bash /a.sh",
            "nice -n 5 rm -rf src", "cd /; if true; then rm -rf app; fi", "coproc rm -rf app", "pushd +1", "cd /app; env cd /o",
            "cp -r --target-directory=app /tmp/src", "cp -tapp /tmp/src", "rm -rf app//src", "rm -rf app/./src", "rm -rf /t/l/../src",
            "python '/app/a&>b.py'", "echo ';'", "echo '>'", "ln -s /app /tmp/l", "find /a -delete", "git -c alias.x=y x", "printf -v X y",
            "ZDOTDIR=/tmp zsh /app/run.sh", "env ZDOTDIR=/tmp zsh /app/run.sh", "/app/run=1.sh", "./a:b.sh",
            "rm -rf /app2>&1", "cd /; /usr/bin/env rm -rf src", "/usr/bin/sudo -i x", "printf -vCDPATH /o", "curl -oa.py http://x",
            "wget -Oa.py u", "sort -oa.py b", "git config --file /app/src/a.py k v", "gunzip -f /app/src/a.py.gz",
            "tar --extract --file=/t.tgz --directory=/app", "zip -rm /t.zip src", "rm -rf //app", "printf 'x' | bash //dev/stdin",
            "cd /; bash ./dev/stdin", "cat /app/2 > 2&> /tmp/log", "curl -soa.py http://x", "wget -qOa.py u", "sort -roa.py /t",
            "gcc -orun /t/m.c", "git config -f/app/a.py x.y z", "git -C /app config -f a.py x.y z", "git diff --fil x",
            "sed --in-plac s/a/b/ /app/a.py", "sed -i.py s/a/b/ /app/a", "perl -pie s/a/b/ /a",
            "curl -o x http://x", "curl -J -O --clobber http://x", "cd /app && go build -o src .", "wget http://x/a.py", "true <<\r'E'\n\rE\ncd /\nE", "git config -f /a k v"]
    for i, c in enumerate(outs):
        assert outside(c, P, False), ("allowlist selftest: should be outside", i, c)
    assert outside("cd /app; python3 a.py", P, True) and outside("npm test", P, True) == ["package_lookup"]
    for c in ("cd /missing && true; python a.py", "cd 2&>/tmp/log && python a.py", "cd >/tmp/log && python a.py", "cd /a || x"):
        assert outside(c, P, True), ("allowlist selftest: own command should be outside", c)
    assert not outside("cd /app && python3 a.py 2>&1 | tail -3 && echo ok", P, True)
    assert outside("cd /app; rm -rf -- -r", ["/app/-r/a.py"], False) and not outside("rm -rf -- -r", P, False)
    assert hits(["ls /"], "", ["//app/src/a.py"]) == ["double_slash_path"] and outside("make --dir=/o", P, True)
    assert not outside("curl -sS -X POST -H 'Content-Type: application/json' -d '{}' http://localhost:8080/api", P, False)
    assert not outside("sed -i s/a/b/ /app/a.py && sed -n 1,5p /app/a.py", P, False)
    ins = ["cd /app && python3 run.py 2>&1 | tail -5", "cat > /a.sh <<'E'\necho $(date) `x` don't\nE", "grep -E 'a|b' /a.py",
           "python3 -c \"print(1); import os\" > /tmp/o.txt", "sudo -u pg psql -c 'select 1'", "timeout 60 python3 /app/x.py",
           "PYTHONPATH=/app env X=1 python3 -m pytest -q", "rm -rf /tmp/build build2", "ls -la /app\npwd", "python3 - <<'E'\nx = 1 << 2\nE",
           "git commit -m 'fix (a); b'", "curl -s -o /tmp/f http://x:80/y", "bash /app/run.sh --fast", "cd /x 2>/dev/null && ls",
           "ls /app;\n\npwd", "grep -n 'bash' /a.py", "./run.sh a b", "cat <<'E' > /a.txt\nhello\nE", "npm test", "cd /app && npm test"]
    for i, c in enumerate(ins):
        assert not outside(c, P, False), ("allowlist selftest: should be inside", i, c, outside(c, P, False))
    return len(outs) + len(ins) + 12
