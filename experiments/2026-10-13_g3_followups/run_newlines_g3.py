"""G3 follow-ups, POST HOC (0d's newline questions, 13:30 PT): trailing newlines of tool outputs, counts only, in one Modal CPU
container (about $0.02). For the 939 cells' first real, second real and as-simulated runs (#36's staged inputs) and the 823
repaired runs (fix939's pinned inputs): per tool output, its family (categories(), copied from the Llama run_rows.py and
checked) and how many newline characters ("\\n" or "\\r") end its raw content (0, 1, >= 2), counted for all outputs and for
the kept rows (G3's dumps' metas; repaired: all outputs only). The unit is the staged tool message's raw content, which Seed's
template renders verbatim. No text, digest or key leaves the container; an exception crosses as its type, file and line.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-13_g3_followups/run_newlines_g3.py \\
      --q36-runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import ast
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
G3 = HERE.parent / "2026-10-13_g3_new_reader"
LLAMA_ROWS = ("a887ae1c", "experiments/2026-10-13_q2_llama_gap_where/run_rows.py")
APP = "probe-shift_aswin-surya_sub-q2-g3f_newlines_kill-2026-11-13"
LA, G3_ACTS = "probe-shift_aswin-surya_sub-la-activations", "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_INPUTS, FIX_INPUTS_SHA = "2026-10-11_q3_fix939_inputs_v1", "57cfbfbf1e4571690769605d2093d73edc93b958e06fed5e8efe8fa88587530d"
DUMPS = {"real-attack-e1": "2026-10-13_g3_real_seed-oss-36b_v1", "real-attack-e2": "2026-10-13_g3_real_seed-oss-36b_v1", "sim-attack": "2026-10-13_g3_sim_seed-oss-36b_v1"}
PINS = {"q36 runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
        "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}
FAMILY = ("bash", "view", "create", "str_replace", "insert", "editor-other", "other")
EDITORS, CMDS, LENGTH = ("text_editor", "str_replace_based_edit_tool"), ("view", "create", "str_replace", "insert"), (100, 1000, 5000)

app = modal.App(APP)
vols = {"/la": modal.Volume.from_name(LA, version=2).read_only(), "/g3acts": modal.Volume.from_name(G3_ACTS, version=2).read_only()}
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


def trailing(msgs):
    """Per tool output: (step, action, number of newline characters ending its raw content, capped at 2)."""
    out, step, action = [], -1, 0
    for m in msgs:
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            c = m.get("content")
            t = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
            out.append((step, action, min(len(t) - len(t.rstrip("\n\r")), 2))); action += 1
    return out


@app.function(image=image, volumes=vols, cpu=4, memory=32768, timeout=1800, retries=0)
def count(units: list[dict], fix_keys: list[str]) -> dict:
    try:
        return _count(units, fix_keys)
    except Exception as e:  # type, file and line only
        import traceback
        f = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"count failed: {type(e).__name__} at {Path(f.filename).name}:{f.lineno}") from None


def _count(units, fix_keys):
    from collections import Counter
    need = {(u["arm"], u["key"]) for u in units}
    msgs = {}
    for p in sorted((Path("/la") / DUMP / "inputs").glob("shard_*.jsonl")):
        for line in p.read_bytes().splitlines():
            it = json.loads(line)
            if (it["arm"], it["key"]) in need:
                if (it["arm"], it["key"]) in msgs:
                    raise ValueError("a staged transcript appears twice")
                msgs[(it["arm"], it["key"])] = it["messages"]
    if set(msgs) != need:
        raise ValueError("a needed staged transcript is missing")
    fx = Path("/la") / FIX_INPUTS / "inputs" / "shard_0.jsonl"
    if hashlib.sha256(fx.read_bytes()).hexdigest() != FIX_INPUTS_SHA:
        raise ValueError("fix939's inputs are not the pinned ones")
    fk = set(fix_keys)
    for it in map(json.loads, fx.read_text().splitlines()):
        if it["key"] in fk:
            msgs[("repaired", it["key"])] = it["messages"]
    if sum(a == "repaired" for a, _ in msgs) != len(fk):
        raise ValueError("a FIX key has no fix939 input")
    c = Counter()
    for (a, k), m in msgs.items():
        cats, n, _ = categories(m)
        kept = None
        if a != "repaired":
            u = next(x for x in units if x["arm"] == a and x["key"] == k)
            meta = json.loads((Path("/g3acts") / DUMPS[a] / "meta" / a / f"{u['filename']}.json").read_text())
            if meta["key"] != k:
                raise ValueError("a meta of another run")
            kept = {(p["step"], p["action"]) for p in meta["positions"] if p["kind"] == "output_end" and p["kept"]}
        for st, ac, nl in trailing(m):
            fam = FAMILY[cats[(st, ac)][0]]
            c[f"{a} | all outputs | {fam} | {nl}"] += 1
            if kept is not None and (st, ac) in kept:
                c[f"{a} | kept rows | {fam} | {nl}"] += 1
    return {"counts (arm | set | family | trailing newlines: 0, 1, 2 = two or more)": dict(sorted(c.items()))}


@app.local_entrypoint()
def main(q36_runs: str, fix: str):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True)  # noqa: E731
    me = Path(__file__).resolve()
    if git("ls-files", "--error-unmatch", str(me)).returncode or git("diff", "--quiet", "HEAD", "--", str(me)).returncode:
        raise SystemExit("commit this script before running")
    def defs(src):
        return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "categories"}
    if defs(me.read_text()) != defs(git("show", f"{LLAMA_ROWS[0]}:{LLAMA_ROWS[1]}").stdout) or not defs(me.read_text()):
        raise SystemExit("categories() differs from the Llama run_rows.py's")
    if sha(q36_runs) != PINS["q36 runs.json"] or sha(fix) != PINS[Path(fix).name]:
        raise SystemExit("not #36's runs.json or fix939's per-run file")
    fit = {r["cell"] for r in json.loads((G3 / "results" / "run_scores_2026-10-13_111525" / "runs.json").read_text()) if r["arm"] == "real-attack-e2"}
    units = [{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in json.loads(Path(q36_runs).read_text())
             if r["arm"] in DUMPS and r["cell"] in fit]
    if len(units) != 3 * 939:
        raise SystemExit("not 939 cells' three runs")
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    res = count.remote(units, fix_keys)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"POST HOC": True, "git": git("rev-parse", "HEAD").stdout.strip(), "app": APP, "unit": "the staged tool message's raw content (Seed's template renders it verbatim)",
           "inputs sha256": PINS | {f"{FIX_INPUTS}/inputs/shard_0.jsonl": FIX_INPUTS_SHA}, **res}
    with open(dest := HERE / "results" / f"newlines_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
