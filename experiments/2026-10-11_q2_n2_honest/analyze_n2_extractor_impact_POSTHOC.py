"""N2, post hoc: how many calls the three extractor threads on #682/#683 touch (Codex; counts only; honest runs; free).
(1) shell calls with a simple command led by if/then/elif/else/while/until/do/{ that holds a read/write verb or one of
the redirections >, >>, < or &> (the rarer >| and &>> aren't checked; this is a reach estimate, and the v2 rerun handles all); (2) text_editor undo_edit calls; (3) log-path hits that match only through the unbounded /etc/rsyslog or
/etc/logrotate alternatives (a bounded pattern allows only .conf and .d after them).

  uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python experiments/2026-10-11_q2_n2_honest/analyze_n2_extractor_impact_POSTHOC.py <transcript cache>
"""
import glob, json, re, shlex, sys, zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
from analyze_n2_honest import COHORTS, END, LOGPATH
from probe_shift.shell_paths import HEREDOC, READERS, cmd_of, strip_comments
from probe_shift.transcripts import reader_messages
KW = {"if", "then", "elif", "else", "while", "until", "do", "{"}
VERBS = READERS | {"tee", "touch", "cp", "mv"}
BOUNDED = re.compile(LOGPATH.pattern.replace("/etc/rsyslog|/etc/logrotate", "/etc/rsyslog(?:\\.conf|\\.d)?" + END + "|/etc/logrotate(?:\\.conf|\\.d)?" + END))
assert BOUNDED.pattern != LOGPATH.pattern
def segments(cmd):
    lines, pending = [], []
    for line in cmd.replace("\\\n", " ").split("\n"):
        if pending:
            if line.strip() == pending[0]: pending.pop(0)
            continue
        lines.append(line); pending = [m.group(2) for m in HEREDOC.finditer(line)]
    lex = shlex.shlex(strip_comments("\n;\n".join(lines)), posix=True, punctuation_chars=";&|()<>"); lex.whitespace_split, lex.commenters = True, ""
    seg, out = [], []
    for t in lex:
        if t and set(t) <= set(";&|()"): out.append(seg); seg = []
        else: seg.append(t)
    return out + [seg]
cache = Path(sys.argv[1]).expanduser(); res = {}
for world, d in COHORTS.items():
    c = {"calls": 0, "shell calls": 0, "(1) shell calls with a keyword-led segment holding a read/write verb or redirection": 0,
         "(2) text_editor undo_edit calls": 0, "(3) log-path hits only via unbounded rsyslog/logrotate": 0, "log-path hits": 0,
         "runs touched by (1)": 0, "runs touched by (2)": 0, "runs touched by (3)": 0, "unparsable": 0}
    for ev in sorted(glob.glob(str(cache / d / "logs" / "*.eval"))):
        with zipfile.ZipFile(ev) as z:
            for name in sorted(x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")):
                s = json.loads(z.read(name)); hit = [False, False, False]
                for m in reader_messages(s, "system"):
                    if m["role"] != "assistant": continue
                    for tc in m.get("tool_calls") or []:
                        fn = tc["function"]; tool = (fn.get("name") if isinstance(fn, dict) else fn) or ""
                        args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
                        if isinstance(args, str): args = json.loads(args) if args.strip().startswith("{") else {"cmd": args}
                        args = args or {}; c["calls"] += 1; cmd = cmd_of(tool, args)
                        if LOGPATH.search(cmd):
                            c["log-path hits"] += 1
                            if not BOUNDED.search(cmd): c["(3) log-path hits only via unbounded rsyslog/logrotate"] += 1; hit[2] = True
                        if tool == "text_editor":
                            if args.get("command") == "undo_edit": c["(2) text_editor undo_edit calls"] += 1; hit[1] = True
                            continue
                        c["shell calls"] += 1
                        try: segs = segments(cmd)
                        except ValueError: c["unparsable"] += 1; continue
                        if any(sg and sg[0] in KW and (any(w in VERBS for w in sg) or any(t in (">", ">>", "<", "&>") for t in sg)) for sg in segs):
                            c["(1) shell calls with a keyword-led segment holding a read/write verb or redirection"] += 1; hit[0] = True
                for k, h in zip(("(1)", "(2)", "(3)"), hit): c[f"runs touched by {k}"] += h
    res[world] = c
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dest = HERE / "results" / f"n2_extractor_impact_POSTHOC_{stamp}.json"
with open(dest, "x") as fh:
    fh.write(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1)); print(dest)
