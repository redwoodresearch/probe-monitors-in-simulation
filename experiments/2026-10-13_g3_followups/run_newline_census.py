"""G3 follow-ups (3), the newline census (PLAN.md (3), POST HOC; preliminary until Codex): counts only, in one Modal CPU
container (8 CPUs; about $0.05). Per tool output of every Volume-staged world (PLAN.md (3)'s list: the 939 cells' first real,
second real, as simulated, repaired, fix-full and N; real and simulated honest and their fix-full and N versions; the
September on-policy set), its family and measures() (trailing newlines by two definitions, internal blank-line runs, b7's
blank lines, error-like text), counted over all outputs, Qwen3-32B-kept rows (where a Qwen3-32B dump exists) and first
outputs. No text, digest or key leaves the container; an exception crosses as its type, file and line.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-13_g3_followups/run_newline_census.py \\
      --q36-runs <#36 runs.json> --fix <fix939 per-run jsonl> --fixfull-stage <stage_2026-10-13_142154.json> --honest-errors <honest_errors jsonl>
"""
import ast
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
G3 = HERE.parent / "2026-10-13_g3_new_reader"
LLAMA_ROWS = ("a887ae1c", "experiments/2026-10-13_q2_llama_gap_where/run_rows.py")
APP = "probe-shift_aswin-surya_sub-q2-g3f_census_kill-2026-11-13"
LA = "probe-shift_aswin-surya_sub-la-activations"
DUMP, V4 = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
FIX_INPUTS, FIX_INPUTS_SHA = "2026-10-11_q3_fix939_inputs_v1", "57cfbfbf1e4571690769605d2093d73edc93b958e06fed5e8efe8fa88587530d"
FIX_DUMP, FIX_ARM = "2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "sim-attack-note-sr-both"
FIXFULL, FIXFULL_REC = "2026-10-13_q2_fixfull_inputs_v2", ("82909df0", "experiments/2026-10-13_q2_fixfull/results/stage_2026-10-13_142154.json")
FIXFULL_ARMS = ("sim-attack-fx-full", "sim-attack-fx-NL", "sim-honest-fx-full", "sim-honest-fx-NL")
SEPT = "2026-10-10_la_september-onpolicy-l0_v1"
ERRORS = ("experiments/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl",
          "59ac3381a8e717a22c643679f717d047ee63e8db4063d3b40100f64688e0a6ff")
PINS = {"q36 runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
        "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd",
        "honest_errors_2026-10-13_130542.jsonl": "a3e7d8956079f9169b1d7ca1877848912aa8e16fad28d53da28a0c343d5fcab7"}
FAMILY = ("bash", "view", "create", "str_replace", "insert", "editor-other", "other")
EDITORS, CMDS, LENGTH = ("text_editor", "str_replace_based_edit_tool"), ("view", "create", "str_replace", "insert"), (100, 1000, 5000)
RUNS = re.compile(r"(?:\r?\n){2,}")

app = modal.App(APP)
vol = modal.Volume.from_name(LA, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12")


# ---------- copied unchanged from ../2026-10-13_q2_llama_gap_where/run_rows.py @ a887ae1c (main checks it) ----------
def categories(msgs):
    """Per tool output, by (step, action) as positions() numbers them: [family, length bin, ending, position, tool name];
    and how many calls' arguments weren't a JSON object (their family is editor-other or other)."""
    outs, step, action, calls, bad = [], -1, 0, [], 0
    for m in msgs:
        if m["role"] == "assistant":
            step, action, calls = step + 1, 0, m.get("tool_calls") or []
        elif m["role"] == "tool":
            fn = (calls[action].get("function") or {}) if action < len(calls) else {}
            name, args = fn.get("name"), fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            if not isinstance(args, dict):
                bad += 1; args = {}
            cmd = args.get("command")
            fam = 0 if name == "bash" else (1 + CMDS.index(cmd) if isinstance(cmd, str) and cmd in CMDS else 5) if name in EDITORS else 6
            c = m.get("content")
            text = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
            ch = text[-1] if text else ""
            end = 0 if not text else 1 if ch in "\n\r" else 2 if ch.isspace() else 3 if ch.isalnum() else 4
            outs.append([step, action, fam, sum(len(text) >= t for t in LENGTH), end, name]); action += 1
    n = len(outs)
    return {(o[0], o[1]): [o[2], o[3], o[4], 0 if 3 * k < n else 1 if 3 * k < 2 * n else 2, o[5]] for k, o in enumerate(outs)}, n, bad
# ---------- end of the copy ----------


def measures(t: str, error_line) -> tuple:
    """(trailing newlines, mine: "\\n"/"\\r" ending the text; b7's: "\\n" inside the trailing whitespace; internal runs of
    two or more line breaks, leading and trailing newlines stripped; b7's blank_lines; error-like text: 1 if any whole line
    matches error_line)."""
    inner_b7 = t.strip("\n").split("\n") if t.strip("\n") else []
    return (len(t) - len(t.rstrip("\n\r")), t[len(t.rstrip()):].count("\n"), len(RUNS.findall(t.strip("\n\r"))),
            sum(x.strip() == "" for x in inner_b7), int(any(error_line.fullmatch(x) for x in t.split("\n"))))


def family(fam: int, name) -> str:
    return "other shell" if fam == 6 and isinstance(name, str) and name.endswith("_bash") else FAMILY[fam]


def per_run(msgs, error_line):
    """Per tool output, in order: (step, action, family, *measures). Text joined as categories() joins it."""
    cats, _, _ = categories(msgs)
    out, step, action = [], -1, 0
    for m in msgs:
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            c = m.get("content")
            t = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
            fam, _, _, _, name = cats[(step, action)]
            out.append((step, action, family(fam, name), *measures(t, error_line))); action += 1
    return out


def _shard(args):
    """One staged shard: the needed runs' per_run() rows (arm, key) -> rows; the file's sha256; items seen per arm."""
    path, need, pattern = args
    error_line = re.compile(pattern, re.I)
    data = Path(path).read_bytes()
    res, arms = {}, {}
    for line in data.splitlines():
        it = json.loads(line)
        arms[it["arm"]] = arms.get(it["arm"], 0) + 1
        if need is None or (it["arm"], it["key"]) in need:
            if (it["arm"], it["key"]) in res:
                raise ValueError("a staged transcript appears twice")
            res[(it["arm"], it["key"])] = per_run(it["messages"], error_line)
    return path, hashlib.sha256(data).hexdigest(), res, arms


@app.function(image=image, volumes={"/la": vol}, cpu=8, memory=32768, timeout=1800, retries=0)
def census(job: dict) -> dict:
    try:
        return _census(job)
    except Exception as e:  # type, file and line only
        import traceback
        f = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"census failed: {type(e).__name__} at {Path(f.filename).name}:{f.lineno}") from None


def _census(job):
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import Pool
    L = Path("/la")
    pat = job["error line pattern"]
    rec = {"inputs sha256": {}, "items per arm in each staged folder": {}}
    q36_need = {(a, k) for a, k, _ in job["q36 units"]}
    v4_need = {("sim-honest", k) for k in job["sim honest keys"]}
    tasks = [(str(p), q36_need, pat) for p in sorted((L / DUMP / "inputs").glob("shard_*.jsonl"))]
    tasks += [(str(p), v4_need, pat) for p in sorted((L / V4 / "inputs").glob("shard_*.jsonl"))]
    tasks += [(str(L / FIX_INPUTS / "inputs" / "shard_0.jsonl"), None, pat)]
    tasks += [(str(L / FIXFULL / s), None, pat) for s, x in sorted(job["fixfull shards"].items()) if x["arm"] in FIXFULL_ARMS]
    sept_files = json.loads((L / SEPT / "manifest.json").read_text())["files"]
    tasks += [(str(L / SEPT / f), None, pat) for f in sorted(sept_files)]
    with Pool(8) as pool:
        got = pool.map(_shard, tasks, chunksize=1)
    rows = {}
    for path, h, res, arms in got:
        rel = str(Path(path).relative_to(L)); rec["inputs sha256"][rel] = h
        folder = rel.split("/")[0]
        acc = rec["items per arm in each staged folder"].setdefault(folder, {})
        for a, n in arms.items():
            acc[a] = acc.get(a, 0) + n
        if folder == FIX_INPUTS:
            if h != FIX_INPUTS_SHA:
                raise ValueError("fix939's inputs are not the pinned ones")
            res = {("repaired", k): v for (a, k), v in res.items() if k in set(job["fix keys"])}
        elif folder == FIXFULL:
            x = job["fixfull shards"][rel.removeprefix(FIXFULL + "/")]
            if h != x["sha256"] or set(arms) != {x["arm"]} or arms[x["arm"]] != x["runs"]:
                raise ValueError("a fix-full shard differs from its staging record")
        elif folder == SEPT:
            if h != sept_files[rel.removeprefix(SEPT + "/")]:
                raise ValueError("a September shard differs from its manifest")
        for ak in res:
            if ak in rows:
                raise ValueError("a run staged twice across shards")
        rows |= res
    if not q36_need | v4_need <= set(rows):
        raise ValueError("a needed staged transcript is missing")
    if sum(a == "repaired" for a, _ in rows) != len(job["fix keys"]):
        raise ValueError("a FIX key has no fix939 input")
    for a, ks in job["fixfull keys"].items():
        if {k for x, k in rows if x == a} != set(ks):
            raise ValueError("fix-full's staged keys differ from its record's")
    # Qwen3-32B kept rows: #36's dump metas (e1, e2, sim, real honest); fix939's own dump for the repaired runs
    def kept(a, k, fn):
        p = L / FIX_DUMP / "meta" / FIX_ARM / f"{fn}.json" if a == "repaired" else L / DUMP / "acts" / a / f"{fn}.json"
        m = json.loads(p.read_text())
        if m["key"] != k:
            raise ValueError("a meta of another run")
        return (a, k), {(p_["step"], p_["action"]) for p_ in m["positions"] if p_["kind"] == "output_end" and p_["kept"]}
    want = [(a, k, fn) for a, k, fn in job["q36 units"]] + [("repaired", k, job["fix filenames"][k]) for k in job["fix keys"]]
    with ThreadPoolExecutor(32) as ex:
        keep = dict(ex.map(lambda u: kept(*u), want))
    errs = {(a, k): {tuple(x) for x in v} for a, kv in job["errored"].items() for k, v in kv.items()}
    base = {"repaired": "sim-attack", "sim-attack-fx-full": "sim-attack", "sim-attack-fx-NL": "sim-attack",
            "sim-honest-fx-full": "sim-honest", "sim-honest-fx-NL": "sim-honest"}  # error flags of the run each was made from
    world = {"real-attack-e1": "first real", "real-attack-e2": "second real", "sim-attack": "as simulated", "repaired": "repaired (823)",
             "sim-attack-fx-full": "fix-full", "sim-attack-fx-NL": "N", "real-honest": "honest: real", "sim-honest": "honest: as simulated",
             "sim-honest-fx-full": "honest: fix-full", "sim-honest-fx-NL": "honest: N", "real-attack-r1": "September: real run 1",
             "real-attack-r2": "September: real run 2", "sim-attack-sept-fixed": "September: fixed simulated"}
    fix_keys, cover, R939 = set(job["fix keys"]), {(a, k): n for a, k, n in job["error table outputs"]}, "repaired (939: the 823 repaired + the 116 unchanged)"
    C, seen_err, bad = Counter(), 0, Counter()
    for (a, k), outs in rows.items():
        ws = [world[a]] + (["as simulated (the 823's originals)"] if a == "sim-attack" and k in fix_keys else
                           ["as simulated (the 116 unchanged)", R939] if a == "sim-attack" else [R939] if a == "repaired" else [])
        ek, sept = (base.get(a, a), k), a.startswith(("real-attack-r", "sim-attack-sept"))
        e, staged = None if sept else errs.get(ek, set()), {(st, ac) for st, ac, *_ in outs}
        if not sept:  # per output: the table has one row per staged output and every errored pair is a staged output
            if ek not in cover:
                raise ValueError("a run without error-flag coverage")
            bad[f"{a} | error-table rows != staged outputs (runs)"] += cover[ek] != len(outs)
            bad[f"{a} | errored pairs matching no staged output (runs)"] += not e <= staged
        kp = keep.get((a, k))
        bad[f"{a} | kept pairs matching no staged output (runs)"] += kp is not None and not kp <= staged
        for j, (st, ac, fam, tm, tb, ir, bl, el) in enumerate(outs):
            raw = fam
            if e is not None and (st, ac) in e:
                fam, seen_err = "error (field set)", seen_err + 1
            fams = [fam] + (["shell (bash + other shell)", "shell, error-like text" if el else "shell, no error-like text"] if fam in ("bash", "other shell") else []) + ["all families"]
            fams += ["shell, error field ignored"] if raw in ("bash", "other shell") else ["view, error field ignored"] if raw == "view" else []
            sets = ["all outputs"] + (["Qwen3-32B-kept rows"] if kp is not None and (st, ac) in kp else []) + (["first output"] if j == 0 else [])
            for w in ws:
                for s in sets:
                    for f in fams:
                        p = f"{w} | {s} | {f}"
                        C[f"{p} | outputs"] += 1
                        C[f"{p} | trailing (mine) | {min(tm, 3)}"] += 1
                        C[f"{p} | trailing (b7) | {min(tb, 3)}"] += 1
                        C[f"{p} | trailing (b7) | sum"] += tb
                        C[f"{p} | internal runs | outputs with one or more"] += ir > 0
                        C[f"{p} | internal runs | sum"] += ir
                        C[f"{p} | blank lines (b7) | sum"] += bl
    rec["join checks (runs failing each, by arm; original arms must be 0)"] = {x: n for x, n in sorted(bad.items())}
    if any(n for x, n in bad.items() if x.split(" | ")[0] in ("real-attack-e1", "real-attack-e2", "sim-attack", "real-honest", "sim-honest") or "kept" in x):
        raise ValueError("a per-output join check failed")
    rec["errored flags matched to staged outputs (error (field set) rows, all worlds and sets)"] = seen_err
    rec["counts (world | set | family | measure | value)"] = dict(sorted(C.items()))
    return rec


@app.local_entrypoint()
def main(q36_runs: str, fix: str, fixfull_stage: str, honest_errors: str):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True)  # noqa: E731
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md"):
        if git("ls-files", "--error-unmatch", str(f)).returncode or git("diff", "--quiet", "HEAD", "--", str(f)).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    def defs(src):
        return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "categories"}
    if not defs(me.read_text()) or defs(me.read_text()) != defs(git("show", f"{LLAMA_ROWS[0]}:{LLAMA_ROWS[1]}").stdout):
        raise SystemExit("categories() differs from the Llama run_rows.py's")
    for p, n in ((q36_runs, "q36 runs.json"), (fix, Path(fix).name), (honest_errors, Path(honest_errors).name)):
        if sha(p) != PINS.get(n):
            raise SystemExit(f"{n} is not the pinned file")
    if Path(fixfull_stage).read_bytes() != git("show", f"{FIXFULL_REC[0]}:{FIXFULL_REC[1]}").stdout.encode():
        raise SystemExit("not fix-full's staging record @ 82909df0")
    from probe_shift.scrub import STDERR_LINE
    pattern = STDERR_LINE.pattern.replace(r")\n?", ")")  # the fixed-sim evening's ERROR_LINE (fixes_round2.py), whole line, any case
    st = json.loads(Path(fixfull_stage).read_text())
    fit = {r["cell"] for r in json.loads((G3 / "results" / "run_scores_2026-10-13_111525" / "runs.json").read_text()) if r["arm"] == "real-attack-e2"}
    q36 = json.loads(Path(q36_runs).read_text())
    units = [(r["arm"], r["key"], r["filename"]) for r in q36 if (r["arm"] != "real-honest" and r["cell"] in fit) or r["arm"] == "real-honest"]
    if sum(a != "real-honest" for a, _, _ in units) != 3 * 939 or sum(a == "real-honest" for a, _, _ in units) != 990:
        raise SystemExit("not 939 cells' three runs and #36's 990 real honest runs")
    sim_fn = {r["key"]: r["filename"] for r in q36 if r["arm"] == "sim-attack"}
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    if len(fix_keys) != 823 or not set(fix_keys) <= {k for a, k, _ in units if a == "sim-attack"}:
        raise SystemExit("not fix939's 823 FIX keys among the 939 simulated runs")
    hon = st["runs by arm"]["sim-honest-fx-full"]
    if len(hon) != 437 or set(hon) != set(st["runs by arm"]["sim-honest-fx-NL"]):
        raise SystemExit("not fix-full's 437 simulated honest runs")
    errored, cover, n_rows = {}, set(), {}
    want = {(a, k) for a, k, _ in units} | {("sim-honest", k) for k in hon}
    for path, arms in ((Path(HERE.parents[1] / ERRORS[0]), ("real-attack-e1", "real-attack-e2", "sim-attack")), (Path(honest_errors), ("real-honest", "sim-honest"))):
        if path.name == Path(ERRORS[0]).name and sha(path) != ERRORS[1]:
            raise SystemExit("not the committed attack error table")
        for line in open(path):
            r = json.loads(line)
            if r["arm"] in arms and (r["arm"], r["key"]) in want:
                cover.add((r["arm"], r["key"])); n_rows[(r["arm"], r["key"])] = n_rows.get((r["arm"], r["key"]), 0) + 1
                if r["error"]:
                    errored.setdefault(r["arm"], {}).setdefault(r["key"], []).append([r["step"], r["action"]])
    if cover != want:
        raise SystemExit(f"{len(want - cover)} runs have no error-flag rows")
    job = {"q36 units": units, "sim honest keys": hon, "fix keys": fix_keys, "fix filenames": {k: sim_fn[k] for k in fix_keys},
           "fixfull shards": dict(st["shards"]),
           "fixfull keys": {a: st["runs by arm"][a] for a in FIXFULL_ARMS}, "errored": errored,
           "error table outputs": sorted([a, k, n] for (a, k), n in n_rows.items()), "error line pattern": pattern}
    res = census.remote(job)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"POST HOC": True, "status": "preliminary until Codex", "plan": "PLAN.md (3)", "git": git("rev-parse", "HEAD").stdout.strip(), "app": APP,
           "unit": "one tool output, its staged text (Inspect's error message when the error field is set, else the output; list content joined)",
           "inputs sha256 (laptop)": PINS | {Path(ERRORS[0]).name: ERRORS[1], "fix-full staging record": sha(fixfull_stage)},
           "error line": "scrub.STDERR_LINE as fixes_round2.ERROR_LINE (whole line, any case)",
           "notes": {"error (field set)": "from the committed error tables; fix-full, N and repaired runs take the flags of the as-simulated run they were made from; "
                                          "September runs have no error table, so their errored outputs stay in their tool family; "
                                          "'shell, error field ignored' and 'view, error field ignored' count every shell or view output whatever its flag, in "
                                          "every world, for comparisons that include September (Codex)",
                     "Qwen3-32B-kept rows": "#36's dump for the 939 cells' runs and real honest; fix939's own dump for the repaired runs; none for the other worlds",
                     "measures": "trailing (mine): newline characters ending the text, 0/1/2/3 = three or more; trailing (b7): \\n inside the trailing whitespace, "
                                 "same bins and its sum; internal runs: maximal runs of two or more line breaks in the text with leading and trailing "
                                 "newlines stripped; blank lines (b7): inner lines empty or whitespace only, sum"}, **res}
    with open(dest := HERE / "results" / f"newline_census_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(dest)
