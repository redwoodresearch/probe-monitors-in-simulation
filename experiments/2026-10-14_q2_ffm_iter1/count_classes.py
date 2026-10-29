"""RULE.md step (a), POST HOC: outcome and format classes of honest tool outputs, real (REAL1, REAL2) vs simulated
full + F9 + F11 (BASE). Modal CPU; counts only: no transcript text leaves the container. The result JSON goes to the
Volume and, when the local client is still attached, to results/ here.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/count_classes.py
"""
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
FOLDER = "experiments/2026-10-14_q2_ffm_iter1"
if modal.is_local():
    sys.path.insert(0, str(REPO / "src"))
APP = "probe-shift_aswin-surya_sub-q2-ffm1_counts_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
# (path on the Volume, sha256 from the stage records on aswin/q2-f10-tabs @ 51846989, arm, runs)
BASE_SRC = ("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_4.jsonl",
            "eaa15a3c136de18916ae66b4bfd28a9a9779f7d14e94dc9e83cb82e20c7bdd0c", "sim-honest-fx-fullF9", 437)
F11_SRC = ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_3.jsonl",
           "78445677dc32700b49ccd9f9d30e73fe096c895584ce5bce109db0b663df3e91", "sim-honest-fx-fullF9F11", 226)
B7_SHA = "8bdaa55d0a395e5a36415bb001f1eff13c9eaabb9ff9f5c71e56bdd26e9e5042"  # b7 @ ea0e8f31 = 54cbb57e (its broad result)
OUT = "2026-10-14_q2_ffm_iter1/results"
B, SEED, MIN_SHARE, MARGIN = 2000, 20261013, 0.05, 0.03

I, M = re.I, re.M
CLASSES = [  # (name, family scope, pattern or None for a special rule); RULE.md "(a) Counts"
    ("te_not_exist", "te", re.compile(r"does not exist\. Please provide a valid path")),
    ("te_exists", "te", re.compile(r"File already exists at|Cannot overwrite files")),
    ("te_no_replace", "te", re.compile(r"No replacement was performed")),
    ("te_multiple", "te", re.compile(r"Multiple occurrences of old_str")),
    ("te_bad_range", "te", re.compile(r"Invalid `view_range`|Invalid `insert_line`")),
    ("te_errno", "te", re.compile(r"Ran into \[Errno \d+\]|is a directory\. ")),
    ("te_dir_listing", "te", re.compile(r"Here are the files and directories")),
    ("te_created", "te", re.compile(r"File created successfully")),
    ("b_cmd_not_found", "bash", re.compile(r"command not found")),
    ("b_no_such_file", "bash", re.compile(r"No such file or directory|cannot access")),
    ("b_permission", "bash", re.compile(r"Permission denied")),
    ("b_traceback", "bash", re.compile(r"Traceback \(most recent call last\)")),
    ("b_exc_line", "bash", re.compile(r"^[A-Z][A-Za-z]+Error: ", M)),
    ("b_module_missing", "bash", re.compile(r"ModuleNotFoundError|No module named")),
    ("b_syntax", "bash", re.compile(r"syntax error|SyntaxError")),
    ("b_connection", "bash", re.compile(r"Connection refused|Could not resolve host|Failed to connect|curl: \(\d+\)")),
    ("b_timeout", "bash", re.compile(r"timed out", I)),
    ("b_test_fail", "bash", re.compile(r"\b[1-9]\d* failed\b|^FAILED|^FAIL\s|ERROR at (?:setup|teardown)", M)),  # RULE literal (Codex r1)
    ("b_test_pass", "bash", re.compile(r"\b\d+ passed\b|All tests passed")),
    ("b_error_prefix", "bash", re.compile(r"^(?:error|Error|ERROR|E|fatal): ", M)),  # RULE literal (Codex r1)
    ("b_warning", "bash", re.compile(r"warning", I)),
    ("b_http_err", "bash", re.compile(r"HTTP/\d(?:\.\d)? [45]\d\d\b")),
    ("b_http_ok", "bash", re.compile(r"HTTP/\d(?:\.\d)? 2\d\d\b")),
    ("b_json_error", "bash", re.compile(r"\"error\":")),  # RULE literal (Codex r1)
    ("b_success_word", "bash", re.compile(r"\bsuccess(?:ful(?:ly)?)?\b", I)),
    ("b_checkmark", "bash", re.compile("[✓✔✅]")),
    ("b_exit_code", "bash", re.compile(r"exit code|exit status|returned non-zero", I)),
    ("b_killed", "bash", re.compile(r"^Killed|Terminated|Segmentation fault|core dumped", M)),
    ("b_empty", "bash", None),
    ("b_no_output_note", "bash", re.compile(r"\(no output\)|\[no output\]|no output", I)),
    ("f_crlf_nondpkg", "any", None),
    ("f_crbare_nondpkg", "any", None),
    ("f_cr_dpkg", "any", None),
    ("f_ansi", "any", re.compile(r"\x1b\[")),
    ("f_truncated", "any", re.compile(r"<START_TOOL_OUTPUT>")),
    ("f_long", "any", None),
]
F_TARGETS = {"f_cr_dpkg"}  # RULE.md (iv): already an F1-F11 target (F6); reported, never ranked
DPKG = "Preparing to unpack"
CR_BARE = re.compile(r"\r(?!\n)")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "joblib==1.6.0", "boto3==1.43.113",
                      "pyyaml==6.0.3")
         .env({"AUTHOR_SPLIT_STAGE": "/tmp/author_split_stage"})
         .add_local_python_source("probe_shift")
         .add_local_file(REPO / "splits" / "exclusions_v4.json", "/root/splits/exclusions_v4.json")
         .add_local_file(HERE / "analyze_POSTHOC_author_split.py", f"/root/{FOLDER}/analyze_POSTHOC_author_split.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py", "/root/swap/rerender.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/fixes.py", "/root/fsim/fixes.py"))


class Gate(Exception):
    """A failed check; its message holds counts only."""


def say(*a):
    print(datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"), *a, flush=True)


def family(fn: str, args: dict) -> str:
    if fn == "text_editor":
        return f"te:{args.get('command') if args.get('command') in ('view', 'create', 'str_replace', 'insert', 'undo_edit') else 'other'}"
    return "bash" if fn.endswith("bash") else f"tool:{fn}"


def cmd_family(fam: str, args: dict) -> str:
    """The `\\r` outputs' source, from the call (RULE.md); first match in this order."""
    if fam != "bash":
        return fam.split(":")[0]
    c = str(args.get("cmd") or args.get("command") or "")
    if re.search(r"\b(?:apt|apt-get|dpkg)\b", c):
        return "apt/dpkg"
    if re.search(r"\bcurl\b", c):
        return "curl -i/-I/-v" if re.search(r"\bcurl\b[^|;&\n]*\s(?:-[a-zA-Z]*[iIv][a-zA-Z]*|--include|--verbose|--head)\b", c) else "curl other"
    for name, rx in (("pip", r"\bpip3?\b|-m pip\b"), ("npm/yarn/pnpm", r"\b(?:npm|yarn|pnpm|npx)\b"), ("git", r"\bgit\b"),
                     ("wget", r"\bwget\b")):
        if re.search(rx, c):
            return name
    return "other"


def in_class(name, pat, text: str) -> bool:
    if pat is not None:
        return bool(pat.search(text))
    if name == "b_empty":
        return not text.strip()
    dp = DPKG in text
    if name == "f_crlf_nondpkg":
        return not dp and "\r\n" in text
    if name == "f_crbare_nondpkg":
        return not dp and bool(CR_BARE.search(text))
    if name == "f_cr_dpkg":
        return dp and "\r" in text
    if name == "f_long":
        return len(text) > 10000
    raise Gate(f"no rule for {name}")


def scope_ok(scope: str, fam: str) -> bool:
    return scope == "any" or (scope == "te" and fam.startswith("te:")) or (scope == "bash" and fam == "bash")


def tally_run(msgs, fixes):
    """Per run: class -> outputs in it; family -> outputs; lengths per family; `\\r` outputs by command family."""
    c = Counter()
    lengths = defaultdict(list)
    for x in fixes.calls_of(msgs):
        if x["tool"] == "submit":
            continue
        t = msgs[x["msg"]].get("content")
        if not isinstance(t, str):
            c["outputs with non-text content (skipped)"] += 1
            continue
        fam = family(x["tool"], x["args"])
        c[f"family|{fam}"] += 1
        lengths[fam].append(len(t))
        for name, scope, pat in CLASSES:
            if scope_ok(scope, fam) and in_class(name, pat, t):
                c[f"class|{name}"] += 1
        if "\r" in t:
            c[f"cr by command|{cmd_family(fam, x['args'])}"] += 1
    n_tool = sum(1 for m in msgs if m["role"] == "tool")
    c["tool messages"] += n_tool
    return c, lengths


def read_rows(job) -> list[dict]:
    """One v4 shard's wanted real honest rows (filtered in the worker, so only those cross processes)."""
    path, want = job
    out = []
    for line in open(path):
        it = json.loads(line)
        if it["arm"] == "real-honest" and it["key"] in want:
            out.append(it)
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=16, memory=65536, timeout=3600, retries=0)
def count(git: str, stamp: str) -> dict:
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    sys.path[:0] = ["/root/swap", "/root/fsim", f"/root/{FOLDER}"]
    import fixes

    src = Path(f"/root/{FOLDER}/analyze_POSTHOC_author_split.py")
    if hashlib.sha256(src.read_bytes()).hexdigest() != B7_SHA:
        raise Gate("b7's analysis module differs from its pin")
    import analyze_POSTHOC_author_split as A

    sim, first, second, ex_sha = A.cohort_rows()
    tasks = sorted(first)
    key_of = {r["cell"]: k for k, r in sim.items()}
    if len(tasks) != 437 or set(key_of) != set(tasks) or len(key_of) != len(sim):
        raise Gate("not 437 tasks with one simulated honest run each")
    shas = {"exclusions_v4.json": ex_sha, "analyze_POSTHOC_author_split.py": B7_SHA}

    def pinned(spec):
        p, h, arm, n = spec
        raw = (Path("/acts") / p).read_bytes()
        shas[p] = hashlib.sha256(raw).hexdigest()
        if shas[p] != h:
            raise Gate(f"{p}: sha256 differs from its stage record")
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        if len(rows) != n or {r["arm"] for r in rows} != {arm} or len({r["key"] for r in rows}) != n:
            raise Gate(f"{p}: not {n} runs of {arm}")
        return {r["key"]: r["messages"] for r in rows}

    base, f11 = pinned(BASE_SRC), pinned(F11_SRC)
    if set(base) != set(sim):
        raise Gate("BASE keys are not b7's 437 simulated honest keys")
    if not set(f11) <= set(base):
        raise Gate("F11 keys outside BASE")
    base |= f11
    want = {r["key"] for r in [*first.values(), *second.values()]}
    shard_files = sorted(str(p) for p in (Path("/acts") / V4 / "inputs").glob("shard_*.jsonl"))
    real = {}
    with ProcessPoolExecutor(min(16, len(shard_files))) as ex:
        for rows in ex.map(read_rows, [(p, want) for p in shard_files]):
            for it in rows:
                if it["key"] in real:
                    raise Gate("a real run appears twice in the v4 inputs")
                real[it["key"]] = it["messages"]
    if set(real) != want:
        raise Gate("wanted real runs missing from the v4 inputs")
    say("rows loaded", len(base), len(real))

    arms = {"REAL1": {t: real[first[t]["key"]] for t in tasks},
            "REAL2": {t: real[second[t]["key"]] for t in sorted(second)},
            "BASE": {t: base[key_of[t]] for t in tasks}}
    per = {a: {t: tally_run(m, fixes) for t, m in d.items()} for a, d in arms.items()}
    names = [n for n, _, _ in CLASSES]
    scope_of = {n: s for n, s, _ in CLASSES}

    def fam_total(a, scope):
        return sum(v for t in per[a] for k, v in per[a][t][0].items()
                   if k.startswith("family|") and scope_ok(scope, k.split("|", 1)[1]))

    g = np.random.default_rng(SEED)
    W = np.stack([np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)]).astype(float)
    t2 = [i for i, t in enumerate(tasks) if t in second]
    W2 = W[:, t2]
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    table = {}
    for n in names:
        has = {a: np.array([per[a][t][0][f"class|{n}"] > 0 for t in (tasks if a != "REAL2" else [tasks[i] for i in t2])], float)
               for a in arms}
        d = has["BASE"] - has["REAL1"]
        dc = has["REAL2"] - has["REAL1"][t2]
        row = {"scope": scope_of[n]}
        for a in arms:
            outs = sum(per[a][t][0][f"class|{n}"] for t in per[a])
            den = fam_total(a, scope_of[n])
            row[a] = {"runs": int(has[a].sum()), "of runs": len(has[a]), "share of runs": float(has[a].mean()),
                      "outputs": int(outs), "family outputs": int(den), "per-output rate": outs / den if den else None}
        row["delta BASE-REAL1 (points)"] = {"point": 100 * float(d.mean()), "ci95": [100 * x for x in pct(W @ d / W.sum(1))]}
        row["control REAL2-REAL1 (points)"] = {"point": 100 * float(dc.mean()), "ci95": [100 * x for x in pct(W2 @ dc / W2.sum(1))],
                                               "tasks": len(t2)}
        dp, ci, cp = row["delta BASE-REAL1 (points)"]["point"], row["delta BASE-REAL1 (points)"]["ci95"], row["control REAL2-REAL1 (points)"]["point"]
        row["eligible"] = {"(i) share >= 5% in BASE or REAL1": max(row["BASE"]["share of runs"], row["REAL1"]["share of runs"]) >= MIN_SHARE,
                           "(ii) interval excludes 0": ci[0] > 0 or ci[1] < 0,
                           "(iii) |delta| >= |control| + 3": abs(dp) >= abs(cp) + 100 * MARGIN,
                           "(iv) not an F1-F11 target": n not in F_TARGETS}
        row["eligible"]["all"] = all(row["eligible"].values())
        table[n] = row
    ranked = sorted((n for n in names if table[n]["eligible"]["all"]),
                    key=lambda n: (-abs(table[n]["delta BASE-REAL1 (points)"]["point"]),
                                   -abs((table[n]["BASE"]["per-output rate"] or 0) - (table[n]["REAL1"]["per-output rate"] or 0))))
    census = {a: Counter() for a in arms}
    cr = {a: Counter() for a in arms}
    cr_runs = {a: Counter() for a in arms}
    lens = {a: defaultdict(list) for a in arms}
    for a in arms:
        for t, (c, L) in per[a].items():
            for k, v in c.items():
                if k.startswith("family|") or k in ("tool messages", "outputs with non-text content (skipped)"):
                    census[a][k] += v
                if k.startswith("cr by command|"):
                    cr[a][k.split("|", 1)[1]] += v
                    cr_runs[a][k.split("|", 1)[1]] += 1
            for f, xs in L.items():
                lens[a][f] += xs
    length = {a: {f: {"outputs": len(xs), "median chars": float(np.median(xs)), "p90 chars": float(np.percentile(xs, 90))}
                  for f, xs in sorted(lens[a].items())} for a in arms}
    out = {"POST HOC": True, "status": "preliminary", "rule": "RULE.md (this folder) @ c6414e10", "git": git, "stamp": stamp,
           "inputs sha256": shas, "runs": {a: len(d) for a, d in arms.items()}, "draws": B, "seed": SEED,
           "census (outputs by family)": {a: dict(sorted(c.items())) for a, c in census.items()},
           "output length by family": length,
           "outputs with \\r by command family": {a: dict(sorted(c.items())) for a, c in cr.items()},
           "runs with \\r by command family": {a: dict(sorted(c.items())) for a, c in cr_runs.items()},
           "classes": table, "ranked eligible classes": [
               {"rank": i + 1, "class": n, "delta (points)": table[n]["delta BASE-REAL1 (points)"]["point"],
                "BASE share": table[n]["BASE"]["share of runs"], "REAL1 share": table[n]["REAL1"]["share of runs"]}
               for i, n in enumerate(ranked)]}
    dst = Path("/acts") / OUT / f"counts_{stamp}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    acts.commit()
    say("written", dst.name, "eligible", len(ranked))
    return out


@app.local_entrypoint()
def main():
    me = Path(__file__).resolve()
    for f in (me, HERE / "RULE.md", HERE / "analyze_POSTHOC_author_split.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = count.remote(git, stamp)
    p = HERE / "results" / f"counts_{stamp}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", p, "ranked:", [r["class"] for r in out["ranked eligible classes"]][:5])
