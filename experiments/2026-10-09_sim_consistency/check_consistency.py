"""Does the (simulated) machine contradict itself within a run? Three kinds of checkable pairs, script only. Read-only,
$0. README.md has the rules and the predictions (committed first).

  uv run --no-project --python 3.14 --with boto3 --with pyyaml --with numpy python -I check_consistency.py \
      --sol-cache <~/.cache/probe-shift/s3/rr-sce-sim-transcripts/transcripts> --qwen-run <copy of the strict rerun> \
      --evidence-out <a path outside git>

Writes results/consistency_<stamp>.json: counts, the resolved inputs, and each inconsistency's run key, steps and check.
The evidence itself (detail and both outputs: model and tool text) goes to --evidence-out, for S3, never results/.
"""
import argparse, collections, copy, glob, hashlib, json, os, random, re, subprocess, sys, zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from probe_shift.calls import SHELL_TOOLS, label_call, split_shell  # noqa: E402

sys.path.insert(0, str(REPO / "experiments" / "2026-10-07_sim_failure_hiding"))
from analyze_failure_hiding import failure  # noqa: E402  PR #67's rule: error field, editor message or bash error line

SEED = 20261009
SOL = {"real": "2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch1-n1000",
       "sim": "2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000"}
MANIFESTS = {"package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "Cargo.toml", "Cargo.lock", "go.mod",
             "go.sum", "pyproject.toml", "requirements.txt", "Gemfile", "Gemfile.lock", "uv.lock"}
OPTS = r"(?:\s+--?[\w-]+(?:[= ](?!(?:install|i|add|remove|rm|uninstall|update|ci|sync|get|mod)\b)[^\s-]\S*)?)*"
PKG = re.compile(r"\b(?:npm|pnpm|yarn)" + OPTS + r"\s+(?:install|i|add|remove|rm|uninstall|update|ci)\b|\bpip3?\s+install\b|"
                 r"\bcargo\s+(?:add|remove|update|build)\b|\bgo\s+(?:get|mod)\b|\buv\s+(?:add|remove|sync|pip)\b|"
                 r"\bpoetry\s+(?:add|remove)\b|\bbundle\s+(?:install|add)\b")  # package-manager commands that edit manifests
REDIR = re.compile(r"(?<![0-9&<>])>(?!>)\s*['\"]?([^\s;&|'\")<>]+)")
VOLATILE = re.compile(r"\b(?:date|ps|top|uptime|free|df|du|time|sleep|curl|wget|nc|redis-cli|psql|mysql|mongosh|"
                      r"docker|systemctl|journalctl|tail\s+-f|watch|random|uuid|ss|netstat|lsof|pgrep)\b|\$RANDOM|/proc/")
CHANGERS = re.compile(r"\b(?:rm|mv|cp|sed\s+-i|git|tee|truncate|chmod|chown|ln|install|tar|unzip|patch|make|npm|pip|"
                      r"cargo|go|uv|python3?|node|ruby|bash|sh)\b")


def text_of(c) -> str:
    return c if isinstance(c, str) else "\n".join(x.get("text") or "" for x in c or [] if isinstance(x, dict))


def calls_of(sample: dict) -> list[dict]:
    att = sample.get("attachments") or {}

    def res(v):
        if isinstance(v, str) and v.startswith("attachment://"):
            return att[v[13:]]
        if isinstance(v, dict):
            return {k: res(x) for k, x in v.items()}
        return [res(x) for x in v] if isinstance(v, list) else v

    msgs = res(sample["messages"])
    out = {m.get("tool_call_id"): m for m in msgs if m["role"] == "tool"}
    rows, step = [], 0
    for m in msgs:
        if m["role"] != "assistant":
            continue
        step += 1
        for tc in m.get("tool_calls") or []:
            if tc["function"] == "submit":
                continue
            a = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
            o = out.get(tc.get("id")) or {}
            err = o.get("error")
            # A reply counts as failed by the error field OR by failure text (simulators return failures as text)
            failed = bool(o) and bool(failure(tc["function"], o))
            rows.append({"step": step, "tool": tc["function"], "args": a, "error": bool(err) or failed,
                         "out": (err.get("message") or "") if err else text_of(o.get("content"))})
    return rows


def cmd_of(c: dict) -> str:
    return str(c["args"].get("cmd", c["args"].get("command", ""))) if c["tool"] in SHELL_TOOLS else ""


def norm_lines(s: str) -> list[str]:
    return [" ".join(x.split()) for x in s.splitlines() if x.strip()]


def strip_numbers(view: str) -> list[str]:
    """An editor view's file lines, its `cat -n` gutter removed."""
    lines = []
    for x in view.splitlines():
        m = re.match(r"^\s*\d+\t(.*)$", x)
        if m:
            lines.append(" ".join(m.group(1).split()))
    return [x for x in lines if x]


def is_read_only(c: dict) -> bool:
    try:
        return label_call(c["tool"], c["args"])["label"] == "read_only"
    except ValueError:
        return False


def writes(c: dict) -> dict:
    """{path: expected content or None}: what this call writes with known content (editor create, a heredoc redirect),
    plus paths written without known content (other redirects, tee) as None. Error replies write nothing."""
    if c["error"]:
        return {}
    a = c["args"]
    if c["tool"] == "text_editor":
        if a.get("command") == "create":
            return {str(a.get("path")): str(a.get("file_text") or "")}
        if a.get("command") in ("str_replace", "insert", "undo_edit"):  # undo_edit reverts the file: a write too
            return {str(a.get("path")): None}
        return {}
    out = {}
    for p in split_shell(cmd_of(c)):
        for t in REDIR.findall(p["text"]):
            if not t.startswith("/dev/"):
                body = p["heredocs"][0] if p["heredocs"] and p["text"].split()[0] == "cat" else None
                out[t] = body
        if p["text"].split()[:1] == ["tee"]:
            for t in p["text"].split()[1:]:
                if not t.startswith("-"):
                    out[t] = None
    return out


def full_read(c: dict) -> tuple[str, list[str]] | None:
    """(path, file lines) for a full read: an editor view with no range, or a bare `cat P`. None otherwise."""
    if c["error"] or "too long to be displayed" in c["out"]:
        return None
    a = c["args"]
    if c["tool"] == "text_editor" and a.get("command") == "view" and not a.get("view_range"):
        if "Here's the result of running `cat -n`" not in c["out"]:
            return None  # a directory listing, not a file
        return str(a.get("path")), strip_numbers(c["out"])
    m = re.fullmatch(r"\s*cat\s+(['\"]?)([^\s;&|<>*?'\"]+)\1\s*", cmd_of(c))
    return (m.group(2), norm_lines(c["out"])) if m else None


def same(p: str, q: str) -> bool:
    p, q = os.path.normpath(p if p.startswith("/") else "/app/" + p), os.path.normpath(q if q.startswith("/") else "/app/" + q)
    return p == q


def check(calls: list[dict]) -> list[dict]:
    """Every checkable pair in one run: {"kind", "i", "j", "ok", "detail"}."""
    pairs = []
    for i, c in enumerate(calls):
        a = c["args"]
        # A. write then read. A bash write whose own command goes on to run something that isn't read-only (e.g. a
        # validation script that may rewrite the file) is skipped: the file may legitimately differ afterwards.
        tail_changes = c["tool"] in SHELL_TOOLS and any(
            not is_read_only({"tool": c["tool"], "args": {"cmd": p["text"]}}) and not REDIR.search(p["text"])
            for p in split_shell(cmd_of(c))[1:])
        for path, content in ({} if tail_changes else writes(c)).items():
            exp = None
            if content is not None:
                exp = norm_lines(content)
            elif c["tool"] == "text_editor" and a.get("command") == "str_replace" and isinstance(a.get("new_str"), str):
                lines = norm_lines(a["new_str"])
                short = " ".join(a["new_str"].split())  # an empty one (a deletion) leaves nothing to look for: skipped
                exp = lines[1:-1] if len(lines) > 2 else (["\x00substring", short] if short else None)
            if not exp:
                continue
            base = os.path.basename(path)
            for j in range(i + 1, len(calls)):
                d = calls[j]
                if any(same(path, w) for w in writes(d)) or (base in cmd_of(d) and CHANGERS.search(cmd_of(d))
                                                             and not full_read(d)) or (base in MANIFESTS
                                                                                       and PKG.search(cmd_of(d))):
                    break  # P may have changed: no pair
                r = full_read(d)
                if r and same(r[0], path):
                    got = set(r[1])
                    if exp[0] == "\x00substring":  # a 1-2 line replacement: its whole text must appear in the read
                        hit = float(bool(exp[1]) and exp[1] in " ".join(" ".join(r[1]).split()))
                        detail = f"{path}: short replacement {'found' if hit else 'missing'} in the later read"
                    else:
                        hit = sum(x in got for x in exp) / len(exp)
                        detail = f"{path}: {hit:.0%} of {len(exp)} written lines in the later read"
                    pairs.append({"kind": "A", "i": i, "j": j, "ok": hit >= 0.95, "detail": detail})
                    break
        # B. create/delete then ls of the folder
        created = [p for p in writes(c)] if c["tool"] == "text_editor" and a.get("command") == "create" and not c["error"] \
            else []
        deleted = []
        if c["tool"] in SHELL_TOOLS and not c["error"]:
            # A redirect target (>, >>, 2>) exists afterwards. Conservative: only absolute targets, and only in a command
            # with no quotes or variables, nothing conditional (&&, ||, if/then, loops: the write might not run), no cd
            # (paths might be relative to it) and no rm or mv (the file might be gone again, by any spelling or glob).
            cmd = cmd_of(c)
            if not re.search(r"['\"`$]|&&|\|\||\b(?:cd|if|then|else|elif|while|until|for|case|rm|mv)\b", cmd):
                created += [t for t in re.findall(r"(?<![<>&])[12]?>>?\s*([^\s;&|<>()]+)", cmd)
                            if t.startswith("/") and not t.startswith("/dev/")]
            for p in split_shell(cmd_of(c)):
                w = p["text"].split()
                if w[:1] == ["touch"]:
                    created += [x for x in w[1:] if not x.startswith("-")]
                if w[:1] == ["rm"]:
                    deleted += [x for x in w[1:] if not x.startswith("-") and not re.search(r"[*?\[{$]", x)]
        touched = {os.path.basename(x) for q in split_shell(cmd_of(c)) if q["text"].split()[:1] in (["mv"], ["rm"])
                   for x in q["text"].split()[1:]} if c["tool"] in SHELL_TOOLS else set()
        # created, then removed or moved within the same call (any spelling of the name): no claim about a later listing
        created = [p for p in dict.fromkeys(created) if p not in deleted and os.path.basename(p) not in touched]
        for path, present in [(p, True) for p in created] + [(p, False) for p in deleted]:
            folder, base = os.path.dirname(path) or "/app", os.path.basename(path)
            if not base:
                continue
            for j in range(i + 1, len(calls)):
                d = calls[j]
                cmd = cmd_of(d)
                m = re.fullmatch(r"\s*ls((?:\s+-[A-Za-z]+)*)\s+(['\"]?)([^\s;&|<>*?'\"]+)\2\s*", cmd)
                hidden_ok = not base.startswith(".") or (m and re.search(r"-\w*[aA]", m.group(1) or ""))
                if m and same(m.group(3), folder) and not d["error"] and hidden_ok:
                    names = {t.rstrip("/*@") for t in d["out"].split()}
                    pairs.append({"kind": "B", "i": i, "j": j, "ok": (base in names) == present,
                                  "detail": f"{path} {'created' if present else 'deleted'}; later ls {folder} "
                                            f"{'lists' if base in names else 'omits'} it"})
                    break
                if (base in cmd or d["tool"] == "text_editor" and os.path.basename(str(d["args"].get("path"))) == base) \
                        and not is_read_only(d):
                    break
        # C. the same read-only command twice, only read-only calls between
        cmd = " ".join(cmd_of(c).split())
        if cmd and not c["error"] and not VOLATILE.search(cmd) and is_read_only(c):
            for j in range(i + 1, len(calls)):
                d = calls[j]
                if " ".join(cmd_of(d).split()) == cmd and not d["error"]:
                    ok = norm_lines(c["out"]) == norm_lines(d["out"])
                    pairs.append({"kind": "C", "i": i, "j": j, "ok": ok,
                                  "detail": f"`{cmd[:80]}` gave {'the same' if ok else 'a different'} output"})
                    break
                if not is_read_only(d):
                    break
    return pairs


def load_sol(cache: Path, world: str):
    for ev in sorted(glob.glob(str(cache / SOL[world] / "logs" / "*.eval"))):
        with zipfile.ZipFile(ev) as z:
            for n in z.namelist():
                if n.startswith("samples/") and n.endswith(".json"):
                    s = json.loads(z.read(n))
                    yield f"sol-{world}/{Path(ev).stem}/{s['id']}", calls_of(s)


def load_qwen(run_dir: Path):
    launch = json.loads((run_dir / "launch.json").read_text())
    for r in launch["runs"]:
        counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
        if not counted:
            continue
        a = counted[0] if r["arm"] == "honest_a" else counted[-1]
        (ev,) = sorted((run_dir / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])).glob("*.eval"))
        with zipfile.ZipFile(ev) as z:
            (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
            s = json.loads(z.read(n))
        world = "qwen-real" if r["real"] else f"qwen-sim@{r['simulator']}"
        yield world, f"{world}/{r['arm']}/{r['task']}", calls_of(s)


def plant(runs: dict) -> list[tuple[str, list[dict], str]]:
    """6 planted inconsistencies (one on a 1-2 line str_replace) in copies of real Qwen runs: for checkable pairs of each kind, the later output is
    altered so the pair contradicts the earlier call."""
    out, rng = [], random.Random(SEED)
    items = [(k, c, p) for k, c in sorted(runs.items()) for p in check(c) if p["ok"]]
    for kind, n, short in (("A", 2, False), ("A", 1, True), ("B", 2, False), ("C", 1, False)):
        pool = [x for x in items if x[2]["kind"] == kind and ("short replacement" in x[2]["detail"]) == short]
        for k, c, p in rng.sample(pool, min(n, len(pool))):
            c2 = copy.deepcopy(c)
            d = c2[p["j"]]
            if kind == "A":  # the later read no longer shows what was written
                d["out"] = (re.sub(r"(?m)^(\s*\d+\t).*$", r"\1PLANTED", d["out"]) if d["tool"] == "text_editor"
                            else "PLANTED CONTENT")
            elif kind == "B":
                base = os.path.basename(p["detail"].split()[0])
                d["out"] = (d["out"].replace(base, "")) if "created" in p["detail"] else d["out"] + "\n" + base
            else:
                d["out"] = d["out"] + "\nPLANTED EXTRA LINE"
            out.append((k, c2, kind))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol-cache", type=Path, required=True)
    ap.add_argument("--qwen-run", type=Path, required=True)
    ap.add_argument("--evidence-out", type=Path, required=True, help="JSONL of each inconsistency's text (outside git)")
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    qwen = {}
    worlds = collections.defaultdict(dict)
    for world, key, calls in load_qwen(args.qwen_run):
        worlds[world][key] = calls
    print(f"script commit {commit[:12]} | seed {SEED}")
    planted = plant(worlds["qwen-real"])
    caught = sum(any(not p["ok"] and p["kind"] == kind for p in check(c)) for _, c, kind in planted)
    print(f"positive control: {len(planted)} inconsistencies planted in copies of real Qwen runs "
          f"({collections.Counter(k for _, _, k in planted)}); flagged: {caught} of {len(planted)}")
    # Real Qwen runs may have no 1-2 line str_replace pair, so that path also gets a synthetic pair: consistent, then planted
    edit = {"step": 1, "tool": "text_editor", "error": False, "out": "edited",
            "args": {"command": "str_replace", "path": "/app/x.py", "old_str": "a = 0", "new_str": "a = 1"}}
    synth = [check([edit, {"step": 2, "tool": "text_editor", "error": False, "args": {"command": "view", "path": "/app/x.py"},
                           "out": f"Here's the result of running `cat -n` on /app/x.py:\n     1\timport os\n     2\t{v}"}])
             for v in ("a = 1", "a = 0")]
    synth_ok = [p["ok"] for ps in synth for p in ps] == [True, False]
    print(f"short str_replace, synthetic pair: consistent read passes and planted read is flagged: {synth_ok}")
    for w in ("real", "sim"):
        for key, calls in load_sol(args.sol_cache, w):
            worlds[f"sol-{w}"][key] = calls
    inputs = {"qwen_run": str(args.qwen_run.resolve()),
              "qwen_launch_json_sha256": hashlib.sha256((args.qwen_run / "launch.json").read_bytes()).hexdigest(),
              "sol_cache": str(args.sol_cache.resolve()), "sol_runs": SOL,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = {"script_commit": commit, "inputs": inputs, "seed": SEED, "made": stamp, "planted": len(planted),
           "planted_flagged": caught, "short_replacement_synthetic_ok": synth_ok, "worlds": {}, "inconsistencies": []}
    bad_examples, evidence = collections.defaultdict(list), []
    print("\nper world: checkable pairs and inconsistent share, by check")
    for w in sorted(worlds):
        tally = collections.Counter()
        for key, calls in worlds[w].items():
            for p in check(calls):
                tally[(p["kind"], "n")] += 1
                if not p["ok"]:
                    tally[(p["kind"], "bad")] += 1
                    bad_examples[w].append((key, calls[p["i"]]["step"], calls[p["j"]]["step"], p["kind"], p["detail"],
                                            calls[p["j"]]["out"][:200]))
                    out["inconsistencies"].append({"world": w, "run": key, "step_i": calls[p["i"]]["step"],
                                                   "step_j": calls[p["j"]]["step"], "check": p["kind"]})
                    evidence.append({**out["inconsistencies"][-1], "detail": p["detail"],
                                     "output_i": calls[p["i"]]["out"], "output_j": calls[p["j"]]["out"]})
        row = {k: {"pairs": tally[(k, "n")], "inconsistent": tally[(k, "bad")]} for k in "ABC"}
        out["worlds"][w] = {"runs": len(worlds[w]), **row}
        print(f"  {w:16s} runs {len(worlds[w]):4d} | " + " | ".join(
            f"{k}: {row[k]['inconsistent']}/{row[k]['pairs']} = "
            f"{(row[k]['inconsistent'] / row[k]['pairs'] if row[k]['pairs'] else float('nan')):.3f}" for k in "ABC"))
    rng = random.Random(SEED)
    print(f"\n3 inconsistencies per world, drawn with seed {SEED} (run, steps i -> j, check, detail, later output head):")
    for w in sorted(bad_examples):
        for key, si, sj, kind, det, o in rng.sample(bad_examples[w], min(3, len(bad_examples[w]))):
            print(f"  [{key}, steps {si} -> {sj}, {kind}] {det} | later output: {' '.join(o.split())[:160]!r}")
    args.evidence_out.write_text("".join(json.dumps(e) + "\n" for e in evidence))
    out["evidence_sha256"] = hashlib.sha256(args.evidence_out.read_bytes()).hexdigest()  # the text, on S3
    res = HERE / "results" / f"consistency_{stamp}.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(out, indent=1))
    print(f"-> {res.relative_to(REPO)}\n-> evidence: {args.evidence_out}")


if __name__ == "__main__":
    main()
