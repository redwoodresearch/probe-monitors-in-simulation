"""Which files a command references, for the blinded context supplement (2026-10-12; rules fixed before any run).

- text_editor: its absolute path (relative or containing "..": path_ambiguous).
- *bash tools, per shell segment (split on ; && || | & ( ) and unquoted newlines, except after && || or | and the
  blank or comment lines that follow it; comments and heredoc bodies dropped; backslash-newline joined; redirections
  kept as their own tokens; leading VAR=..., sudo, nohup, exec, time, env, nice and timeout stripped with their own
  options (env -C / sudo -D are not stripped, so nothing is read from them);
  leading compound keywords stripped):
  - an interpreter's script (python*, node, bash/sh/zsh/dash, ruby, perl, php, ts-node, tsx). Options before it must
    be known (python's and sh's own tables; any option for the others): otherwise command_unparsed. Inline code (-c;
    node -e/-p/--eval/--print; ruby/perl -e; php -r) names no file, but `sh -c STR` is parsed once more;
    `python -m pytest` counts as pytest; any other -m is module_reference_unresolved;
  - `source F` / `. F`; a command word containing "/" (./run.sh, /usr/local/bin/x);
  - pytest's *.py arguments; npm/yarn/pnpm -> package.json, make -> Makefile (with a directory or file option:
    path_ambiguous).
- Relative paths resolve only against a known directory: an absolute `cd` earlier in the same command, or the
  caller's run_cwd. An unconditional cd is taken to succeed; after || | & ( ) the directory is unknown, after a cd
  that followed && it is unknown again at the next ; or newline (after one that followed ||, at && too), and a cd
  inside if/while/until/for/case/{ } makes it unknown. Paths with "..", $ ~ * ? [ ] { } or a backtick are path_ambiguous.
- A command that can't be split (an unclosed quote or heredoc) gets one entry with path None: command_unparsed.
"""

import posixpath
import re
import shlex

INTERP = re.compile(r"(python[\d.]*|node|nodejs|bash|sh|zsh|dash|ruby|perl|php|ts-node|tsx)")
SH = {"bash", "sh", "zsh", "dash"}
INLINE = {"node": "ep", "nodejs": "ep", "ruby": "e", "perl": "e", "php": "r"}  # short flags taking inline code; else "c"
INLINE_LONG = {"node": {"--eval", "--print"}, "nodejs": {"--eval", "--print"}}
FLAGS = {"python": ("uBEIOsSqvbdixPR", {"-W", "-X"}), "sh": ("abefhklmnptuvxBCEHir", {"-o", "-O", "+o", "+O"})}
WRAP = {"sudo": {"-u", "-g", "-h", "-p", "-U", "-r", "-t", "-T", "-C"}, "nohup": set(), "exec": {"-a"}, "time": {"-f", "-o"},
        "env": {"-u", "-S", "--unset", "--split-string"}, "nice": {"-n", "--adjustment"}, "timeout": {"-s", "-k", "--signal", "--kill-after"}}
CHDIR = re.compile(r"-C|--chdir|-D|--chdir=.*|-C.+|-D.+")  # a wrapper option that changes directory
KEYWORDS, OPEN, CLOSE = {"then", "do", "else", "elif", "!"}, {"if", "while", "until", "for", "case", "{", "select"}, {"fi", "done", "esac", "}"}
UNSAFE = set("$~*?[]{}`")
HEREDOC_OP = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_]\w*)\1")


def bash_cmd(a: dict) -> str:
    c = a.get("cmd", a.get("command"))
    return c if isinstance(c, str) else ""


def segments(cmd: str):
    """Token lists per shell segment with operator strings between them; None if the command can't be split."""
    lines, out, buf, i, cont = cmd.replace("\\\n", "").split("\n"), [], None, 0, False  # cont: the list goes on to the next line
    while i < len(lines):
        buf = lines[i] if buf is None else buf + "\n" + lines[i]
        i += 1
        try:
            blank = not shlex.split(buf, comments=True)  # no words at all (a redirection or heredoc is a word here)
        except ValueError:
            continue  # a quote is still open: the next line belongs to this one
        hd = HEREDOC_OP.search(buf)
        try:
            hd = hd if hd and shlex.split(buf[:hd.start()], comments=True) is not None else None
        except ValueError:
            hd = None  # the << sits inside quotes
        if hd:
            dash = buf[hd.start() + 2:hd.start() + 3] == "-"
            while i < len(lines) and lines[i] != hd.group(2) and not (dash and lines[i].lstrip("\t") == hd.group(2)):
                i += 1
            if i == len(lines):
                return None
            i, buf = i + 1, buf[:hd.start()] + " " + buf[hd.end():]
        text = re.sub(r"\d*[<>]&(\d+|-)(?![\w./])", " ", buf).replace("&>", ">").replace(">&", ">")
        text = re.sub(r"(?<![\w./-])\d+(?=[<>])", "", text)
        lx = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>")
        lx.whitespace_split, seg, n0 = True, [], len(out)
        try:
            for t in lx:
                if t and t[0] in "<>" and set(t) <= set("<>|"):
                    seg.append(t)  # a redirection operator; its target is the next token
                elif t and set(t) <= set(";&|()<>"):
                    out += [seg] + [o[0] if o == "|&" else o for o in re.findall(r"&&|\|\||\|&|[;&|()]", t)]
                    seg = []
                else:
                    seg.append(t)
        except ValueError:
            return None
        # the shell continues a list after a line ending in && || or |, past the blank or comment lines that follow it
        cont = (not seg and len(out) > n0 and out[-1] in ("&&", "||", "|") and buf.rstrip().endswith(out[-1])) or (cont and blank)
        out += [] if cont else [seg, ";"]
        buf = None
    return None if buf is not None else [s for s in out if s]


def unwrap(seg: list) -> list:
    """Leading assignments, wrappers (with their own options) and compound keywords stripped."""
    while seg and (re.match(r"[A-Za-z_]\w*=", seg[0]) or seg[0] in WRAP or seg[0] in KEYWORDS | OPEN | CLOSE):
        j = 1
        while seg[0] in WRAP and j < len(seg) and seg[j].startswith("-"):
            if seg[0] in ("env", "sudo") and CHDIR.fullmatch(seg[j]):
                return seg  # runs elsewhere: left as the command word
            j += 2 if seg[j] in WRAP[seg[0]] else 1
        seg = seg[j + (seg[0] == "timeout"):]
    return seg


def words(item: list) -> list:
    return unwrap(item)


def compound_depth(item: list) -> int:
    """How many compound commands this segment's leading words open (+) or close (-)."""
    d = 0
    for t in item:
        if t in OPEN or t in CLOSE:
            d += 1 if t in OPEN else -1
        elif t not in KEYWORDS:
            break
    return d


def family(word: str) -> str:
    return re.sub(r"[\d.]+$", "", posixpath.basename(word))


def script_of(toks: list):
    """(kind, value) for an interpreter command: script, inline (value: the code), module, stdin, none, unknown_option."""
    fam = family(toks[0])
    letters, valued = FLAGS.get("sh" if fam in SH else fam, ("", set()))
    i = 1
    while i < len(toks):
        t = toks[i]
        if re.match(r"\d*[<>]", t):
            i += 2 if re.fullmatch(r"\d*[<>]+", t) else 1
        elif t in INLINE_LONG.get(fam, ()) or (t[:1] == "-" and t[:2] != "--" and set(t[1:]) & set(INLINE.get(fam, "c"))):
            return "inline", next((u for u in toks[i + 1:] if not u.startswith("-")), None)
        elif t == "-m" and fam == "python":
            return "module", toks[i + 1] if i + 1 < len(toks) else ""
        elif t == "-":
            return "stdin", None
        elif t in valued:
            i += 2
        elif t[:1] == "-" and t[:2] != "--" and len(t) > 1 and set(t[1:]) <= set(letters):
            i += 1
        elif t[:1] in "-+":
            return "unknown_option", t
        else:
            return "script", t
    return "none", None


def parse(cmd: str, cwd, refs: list, depth: int = 0) -> list:
    def add(raw, st="ok"):
        if st == "ok" and not UNSAFE & set(raw) and ".." not in raw.split("/") and (raw.startswith("/") or cwd):
            refs.append((posixpath.normpath(posixpath.join(cwd or "/", raw)), "ok"))
        else:
            refs.append((raw, "path_ambiguous" if st == "ok" else st))

    segs = segments(cmd)
    if segs is None:
        return refs + [(None, "command_unparsed")]
    pending, prev, depth_c = None, None, 0  # pending: the && or || before a cd (it may not have run)
    for item in segs:
        if isinstance(item, str):
            if item not in ("&&", ";") or (item == ";" and pending) or (item == "&&" and pending == "||"):
                cwd, pending = None, None
            prev = item
            continue
        depth_c += compound_depth(item)
        seg = unwrap(item)
        if not seg:
            continue
        w, rest, fam = seg[0], seg[1:], family(seg[0])
        if w in ("cd", "pushd", "popd"):
            t = rest[0] if rest and w != "popd" else ""
            ok = t and t[0] != "-" and not UNSAFE & set(t) and ".." not in t.split("/") and (t.startswith("/") or cwd)
            ok = ok and depth_c <= 0 and prev != "|"  # a cd inside a pipeline (after | or |&) runs in a subshell
            cwd, pending = (posixpath.normpath(posixpath.join(cwd or "/", t)) if ok else None), prev if prev in ("&&", "||") else None
        elif INTERP.fullmatch(posixpath.basename(w)):
            kind, v = script_of(seg)
            if kind == "script":
                add(v)
            elif kind == "inline" and fam in SH and v and depth < 1:
                refs = parse(v, cwd, refs, depth + 1)
            elif kind == "module" and v == "pytest":
                [add(u.split("::")[0]) for u in rest[rest.index("-m") + 2:] if not u.startswith("-") and u.split("::")[0].endswith(".py")]
            elif kind == "module":
                add(v, "module_reference_unresolved")
            elif kind == "unknown_option":
                refs.append((None, "command_unparsed"))
        elif w in ("source", "."):
            if rest:
                add(rest[0])
        elif w in ("pytest", "py.test"):
            [add(u.split("::")[0]) for u in rest if not u.startswith("-") and u.split("::")[0].endswith(".py")]
        elif w in ("npm", "yarn", "pnpm"):
            add("package.json", "path_ambiguous" if any(re.match(r"--(prefix|cwd|dir)|-C", t) for t in rest) else "ok")
        elif w == "make":
            add("Makefile", "path_ambiguous" if any(re.match(r"-[^-]*[Cf]|--(file|makefile|directory)", t) for t in rest) else "ok")
        elif "/" in w and re.fullmatch(r"[\w.+@/-]+", w) and not w.endswith("/"):
            add(w)
        prev = None
    return refs


def referenced(tool: str, args: dict, run_cwd) -> list:
    """[(path, "ok" | a reason)], deduplicated, at most 10."""
    if tool == "text_editor":
        p = args.get("path")
        ok = isinstance(p, str) and p.startswith("/") and ".." not in p.split("/")
        return [] if not isinstance(p, str) or not p else [(posixpath.normpath(p), "ok") if ok else (p, "path_ambiguous")]
    return list(dict.fromkeys(parse(bash_cmd(args), run_cwd, [])))[:10] if tool.endswith("bash") else []


def selftest() -> int:
    cases = [
        ("cd /app && python3 -u run.py --x 1 2>&1 | tee log", None, [("/app/run.py", "ok")]),
        ("python -m pytest tests/t.py -k x", "/app", [("/app/tests/t.py", "ok")]),
        ("python -m http.server", None, [("http.server", "module_reference_unresolved")]),
        ("bash scripts/go.sh; (cd /x && make); ./b.sh", None, [("scripts/go.sh", "path_ambiguous"), ("/x/Makefile", "ok"), ("./b.sh", "path_ambiguous")]),
        ("bash -lc 'node /srv/x.js'; python3 -c 'print(1)'; perl -pe s/a/b/ f", "/app", [("/srv/x.js", "ok")]),
        ("cat > /t.sh <<'E'\nrm -rf /\nE\nsudo /t.sh &>/dev/null", None, [("/t.sh", "ok")]),
        ("python $HOME/x.py; python ../y.py", "/app", [("$HOME/x.py", "path_ambiguous"), ("../y.py", "path_ambiguous")]),
        ("python3 -c \"import os\n# don't\nprint(1)\" && ./a.sh # run it\nls", "/w", [("/w/a.sh", "ok")]),
        ("python3 -W ignore -O /a/b.py /c.py < in.txt", None, [("/a/b.py", "ok")]),
        ("node --inspect /a.js; make -fOther; npm --prefix=/o test", "/w", [(None, "command_unparsed"), ("Makefile", "path_ambiguous"),
                                                                          ("package.json", "path_ambiguous")]),
        ("echo 'unclosed", None, [(None, "command_unparsed")]),
        ("python3 - <<EOF\nx = 1 << 2\nEOF\ntimeout 60 node app/x.js", "/srv", [("/srv/app/x.js", "ok")]),
        ("cd /missing || python f.py; true && cd /x; python g.py\ncd /y\npython h.py", None,
         [("f.py", "path_ambiguous"), ("g.py", "path_ambiguous"), ("/y/h.py", "ok")]),
        ("if true; then cd /other; fi; python f.py; timeout -s KILL 5 python -u g.py; bash --print 'python /a.py'", "/app",
         [("f.py", "path_ambiguous"), ("g.py", "path_ambiguous"), (None, "command_unparsed")]),
        ("cd /app; if [ -f a.py ]; then python a.py; fi; for x in 1; do ./run.sh 2>/dev/null; done", None,
         [("/app/a.py", "ok"), ("/app/run.sh", "ok")]),
        ("if true; then { cd /other; }; fi; python a.py; sudo -n python3 /b.py; env -C /o python c.py", "/app",
         [("a.py", "path_ambiguous"), ("/b.py", "ok")]),
        ("false &&\ncd /other; python a.py\ncd /x &&\npython b.py", "/app", [("a.py", "path_ambiguous"), ("/x/b.py", "ok")]),
        ("false &&\n\n# c\ncd /o; python a.py\ntrue |\ncd /x; python b.py; true | cd /y; python c.py", "/app",
         [("a.py", "path_ambiguous"), ("b.py", "path_ambiguous"), ("c.py", "path_ambiguous")]),
        ("false && cd /x &&\n2>&1\npython a.py; true |& cd /y; python b.py", "/app", [("a.py", "path_ambiguous"), ("b.py", "path_ambiguous")]),
        ("true || cd /o &&\npython a.py; false && cd /p && echo '&&'\npython b.py", "/app", [("a.py", "path_ambiguous"), ("b.py", "path_ambiguous")]),
        ("mkdir -p /x && cd /x && python a.py", None, [("/x/a.py", "ok")]),
    ]
    for i, (cmd, cwd, want) in enumerate(cases):
        assert referenced("bash", {"command": cmd}, cwd) == want, ("parse selftest case", i)
    assert referenced("text_editor", {"path": "/app//b/./c.py"}, None) == [("/app/b/c.py", "ok")]
    assert referenced("text_editor", {"path": "/app/l/../c.py"}, None) == [("/app/l/../c.py", "path_ambiguous")]
    return len(cases) + 2
