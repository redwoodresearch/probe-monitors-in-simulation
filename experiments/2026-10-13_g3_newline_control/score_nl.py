"""G3 newline-only controls, POST HOC (PLAN.md here): scores one edited arm (nlstrip or nladd) with G3's frozen detector, in one
Modal CPU container (score, one .remote() call), beside the same runs' unedited rows from G3's own dumps.
  - Run level: `_score` and `filename` are copied verbatim from ../2026-10-13_g3_new_reader/g3_score_sim.py (22bd11f9) and
    checked by AST source against it before anything runs; EXTRACT holds this job's two edited dumps. Every nested model (f, v),
    v != f, on the run's kept rows, at all 7 layers.
  - Row level (new, `_rows`): per kept row of the edited run and of the same run's unedited dump (G3's real dump for nladd, its
    simulated dump for nlstrip), each nested model's logit at the fold's chosen layer, paired by (step, action); per row its
    family and its index among the run's tool outputs (categories(), the Llama run_rows.py's @ a887ae1c, AST-checked), and
    whether it was edited and whether it lies before the run's first edited output (first 5 and last 5 are independent) (edit decisions recomputed with
    run_extract_g3_nl.py's edit() rule, AST-checked against it).
The probes read must be forecast.json's (sha256 pins @ 6d7b1f50). No text leaves the container; an exception crosses as its
type only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_g3_newline_control/score_nl.py --which nlstrip|nladd --fit-runs <fit runs.json> \\
      --runs experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json --check <extract_check record>
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
APP = "probe-shift_aswin-surya_sub-q2-g3nl_score_kill-2026-11-12"
WORK = "probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
LA = "probe-shift_aswin-surya_sub-la-activations"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
EXTRACT = {"nlstrip": ("2026-10-13_g3_nlstrip_seed-oss-36b_v1", "sim-attack-nlstrip", 939),
           "nladd": ("2026-10-13_g3_nladd_seed-oss-36b_v1", "real-attack-e2-nladd", 939)}
ORIGINAL = {"nlstrip": ("2026-10-13_g3_sim_seed-oss-36b_v1", "sim-attack"), "nladd": ("2026-10-13_g3_real_seed-oss-36b_v1", "real-attack-e2")}
LAYERS = (16, 24, 32, 40, 48, 56, 63)
FIT_RUN = "2026-10-13_111525"
RUNS_PIN = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
FORECAST = ("6d7b1f50", "experiments/2026-10-13_g3_new_reader/forecast.json", "4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77")
FROZEN_SCORER = ("22bd11f9", "experiments/2026-10-13_g3_new_reader/g3_score_sim.py")
LLAMA_ROWS = ("a887ae1c", "experiments/2026-10-13_q2_llama_gap_where/run_rows.py")
EDITORS, CMDS, LENGTH = ("text_editor", "str_replace_based_edit_tool"), ("view", "create", "str_replace", "insert"), (100, 1000, 5000)
EDIT_FAMS = (2, 3, 4)
HOW = {"nlstrip": "strip", "nladd": "add"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
work = modal.Volume.from_name(WORK, version=2).read_only()
acts = modal.Volume.from_name(ACTS, version=2).read_only()
la = modal.Volume.from_name(LA, version=2).read_only()


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


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


def edited_outputs(msgs, how):
    """run_extract_g3_nl.py's edit() decision, without the edit: the (step, action) of the outputs it changes, in order."""
    cats, _, _ = categories(msgs)
    step, action, done = -1, 0, []
    for m in msgs:
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            fam = cats[(step, action)][0]
            c = m.get("content")
            text = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
            new = text.rstrip("\n\r") if how == "strip" else text if text.endswith(("\n", "\r")) else text + "\n"
            if fam in EDIT_FAMS and new != text:
                done.append((step, action))
            action += 1
    return done


@app.function(image=image, volumes={"/work": work, "/acts": acts, "/la": la}, cpu=16, memory=65536, timeout=3600, retries=0)
def score(mode: str, items: list[dict], layers: list[int], run_id: str, chosen: dict) -> dict:
    try:
        return {"run level": _score(mode, items, layers, run_id), "rows": _rows(mode, items, run_id, chosen)}
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"score failed: {type(e).__name__}") from None


def _score(mode, items, layers, RUN):
    import numpy as np
    run, arm, _ = EXTRACT[mode]
    root = Path("/acts") / run
    metas = {}
    for j, it in enumerate(items):
        mp = root / "meta" / arm / f"{filename(it['key'])}.json"
        if not mp.exists():  # a failed extraction: its scores stay NaN and it is counted, for the masks
            continue
        m = json.loads(mp.read_text())
        if m["key"] != it["key"] or m["arm"] != arm:
            raise ValueError("meta mismatch")
        idx = json.loads((root / "chunks" / f"{m['chunk']}.json").read_text())  # G3: the chunk holds only this job's arm,
        if any(t["arm"] != arm for t in idx["transcripts"]):                   # checked before any array is read
            raise ValueError("a chunk holds another arm")
        blk = [t for t in idx["transcripts"] if t["key"] == it["key"]]
        rows_ = [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]]
        if len(blk) != 1 or blk[0]["row_start"] != m["row_start"] or any(not (m["row_start"] <= r < m["row_start"] + blk[0]["n_rows"]) for r in rows_):
            raise ValueError("a run's rows lie outside its own block")
        metas[j] = (m["chunk"], [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]])
    out, shas = {}, {}
    for l in layers:
        f = Path("/work") / "runs" / RUN / f"probes_L{l}.npz"
        shas[str(l)] = hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(f) as zf:
            z = {k: zf[k] for k in zf.files}
        S = np.full((5, 5, len(items)), np.nan); by_chunk = {}
        for j, (c, rows) in metas.items():
            by_chunk.setdefault(c, []).append((j, rows))
        for c, js in by_chunk.items():
            X = np.load(root / "by_layer" / f"L{l}" / f"{c}.npy")  # G3's planted and controls dumps hold only real runs
            for j, rows in js:
                if not rows:
                    continue
                x, f_ = X[rows].astype(np.float32), items[j]["fold"]
                for v in range(5):
                    if v != f_:
                        k = f"output_end__task_nested_{f_}_{v}__"
                        S[f_, v, j] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
        out[str(l)] = S.tolist()
    return {"scores": out, "probes sha256": shas}


def _rows(mode, items, RUN, chosen):
    """Per kept row, edited and unedited, each nested model's logit at the fold's chosen layer; columns as in COLS."""
    import numpy as np
    run, arm, _ = EXTRACT[mode]
    orun, oarm = ORIGINAL[mode]
    need = {it["key"] for it in items}
    msgs = {}
    for p in sorted((Path("/la") / DUMP / "inputs").glob("shard_*.jsonl")):
        for line in p.read_bytes().splitlines():
            it = json.loads(line)
            if it["arm"] == oarm and it["key"] in need:
                if it["key"] in msgs:
                    raise ValueError("a staged transcript appears twice")
                msgs[it["key"]] = it["messages"]
    if set(msgs) != need:
        raise ValueError("a needed staged transcript is missing")
    probes = {}
    for l in sorted(set(chosen.values())):
        with np.load(Path("/work") / "runs" / RUN / f"probes_L{l}.npz") as zf:
            probes[l] = {k: zf[k] for k in zf.files}
    cache = {}

    def arr(root, l, chunk):
        if (root, l, chunk) not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[(root, l, chunk)] = np.load(root / "by_layer" / f"L{l}" / f"{chunk}.npy", mmap_mode="r")
        return cache[(root, l, chunk)]

    def logits(root, a, key, f, l):
        m = json.loads((root / "meta" / a / f"{filename(key)}.json").read_text())
        if m["key"] != key or m["arm"] != a:
            raise ValueError("meta mismatch")
        kept = [(q["step"], q["action"], q["row"]) for q in m["positions"] if q["kind"] == "output_end" and q["kept"]]
        if not kept:
            return {}
        X = np.asarray(arr(root, l, m["chunk"])[[m["row_start"] + r for _, _, r in kept]]).astype(np.float32)
        z = probes[l]; L = np.full((len(kept), 5), np.nan)
        for v in range(5):
            if v != f:
                k = f"output_end__task_nested_{f}_{v}__"
                L[:, v] = ((((X - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64))
        return {(s, a_): L[i] for i, (s, a_, _) in enumerate(kept)}

    rows, missing = [], {"edited run without a meta": 0, "rows kept in one dump only": 0}
    for j, it in enumerate(items):
        f = it["fold"]; l = int(chosen[str(f)])
        if not (Path("/acts") / run / "meta" / arm / f"{filename(it['key'])}.json").exists():
            missing["edited run without a meta"] += 1
            continue
        e = logits(Path("/acts") / run, arm, it["key"], f, l)
        o = logits(Path("/acts") / orun, oarm, it["key"], f, l)
        cats, n, _ = categories(msgs[it["key"]])
        order = {sa: k for k, sa in enumerate(sorted(cats))}
        ed = edited_outputs(msgs[it["key"]], HOW[mode]); eds = set(ed); first = ed[0] if ed else None
        missing["rows kept in one dump only"] += len(set(e) ^ set(o))
        for sa in sorted(set(e) & set(o)):
            k = order[sa]  # first 5 and last 5 are independent memberships (they overlap in runs of under 10 outputs)
            nb = sum(1 for x_ in ed if x_ < sa)  # edited outputs before this one
            rows.append([j, sa[0], sa[1], cats[sa][0], int(k < 5), int(k >= n - 5), int(sa in eds), int(first is None or sa < first), nb,
                         *e[sa].tolist(), *o[sa].tolist()])
    return {"columns": COLS, "rows": rows, "missing": missing}


COLS = ["item", "step", "action", "family", "in the first 5 outputs", "in the last 5 outputs", "edited", "before the first edit", "edited outputs before it",
        *[f"edited logit v{v}" for v in range(5)], *[f"unedited logit v{v}" for v in range(5)]]


def git(*a, check=True):
    return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=check)


def defs(src, names):
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in names}


@app.local_entrypoint()
def main(which: str, runs: str, fit_runs: str, check: str):
    import numpy as np
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if which not in EXTRACT:
        raise SystemExit("--which nlstrip|nladd")
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md", HERE / "run_extract_g3_nl.py"):
        if git("ls-files", "--error-unmatch", str(f), check=False).returncode or git("diff", "--quiet", "HEAD", "--", str(f), check=False).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    src = me.read_text()
    if defs(src, ("filename", "_score")) != defs(git("show", f"{FROZEN_SCORER[0]}:{FROZEN_SCORER[1]}").stdout, ("filename", "_score")):
        raise SystemExit("filename or _score differs from the frozen simulated scorer's")
    if defs(src, ("categories",)) != defs(git("show", f"{LLAMA_ROWS[0]}:{LLAMA_ROWS[1]}").stdout, ("categories",)) or not defs(src, ("categories",)):
        raise SystemExit("categories() differs from the Llama run_rows.py's")
    ext = (HERE / "run_extract_g3_nl.py").read_text()
    body = lambda s, name: ast.get_source_segment(s, next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == name))  # noqa: E731
    if "text.rstrip(\"\\n\\r\") if how == \"strip\" else text if text.endswith((\"\\n\", \"\\r\")) else text + \"\\n\"" not in body(ext, "edit") \
            or "if fam in EDIT_FAMS and new != text:" not in body(ext, "edit"):
        raise SystemExit("the extractor's edit() rule differs from edited_outputs()'s")
    fcb = subprocess.run(["git", "show", f"{FORECAST[0]}:{FORECAST[1]}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(fcb).hexdigest() != FORECAST[2]:
        raise SystemExit("forecast.json at its commit differs from the acknowledged one")
    fz = json.loads(fcb)["frozen"]
    if fz["fit run"] != FIT_RUN or sha(fit_runs) != fz["inputs of x sha256"]["runs.json"] or sha(runs) != RUNS_PIN:
        raise SystemExit("not the frozen fit's runs.json, or not #36's runs.json")
    run, arm, expect = EXTRACT[which]
    ck = json.loads(Path(check).read_text())
    if ck.get("run") != run or ck["meta files by arm"] != {arm: expect} or ck["kept readouts (meta)"] != ck["rows (chunks)"] or any(ck["problems"].values()) \
            or ck["non-finite values: every layer (chunk indexes)"] or ck["non-finite values: first and last layer recounted"]:
        raise SystemExit("--check is not a clean full check of this edited dump")
    e2_fold = {r["cell"]: int(r["fold"]) for r in json.loads(Path(fit_runs).read_text()) if r["arm"] == "real-attack-e2"}
    orig = ORIGINAL[which][1]
    base = sorted((r for r in json.loads(Path(runs).read_text()) if r["arm"] == orig), key=lambda r: r["key"])
    if len(base) != 939 or {r["cell"] for r in base} != set(e2_fold):
        raise SystemExit("not 939 runs, one per frozen cell")
    items = [{"key": r["key"], "fold": e2_fold[r["cell"]], "cell": r["cell"]} for r in base]
    chosen = {str(k): int(v) for k, v in fz["chosen layer by test fold"].items()}
    res = score.remote(which, items, list(LAYERS), FIT_RUN, chosen)
    rl = res["run level"]
    if any(rl["probes sha256"][str(l)] != fz["probes sha256 by layer"][str(l)] for l in LAYERS):
        raise SystemExit("the probes read differ from the frozen ones")
    S = {l: np.array(rl["scores"][str(l)]) for l in LAYERS}
    unavailable = {str(l): sum(not np.isfinite(np.delete(S[l][it["fold"], :, j], it["fold"])).all() for j, it in enumerate(items)) for l in LAYERS}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"scores_{which}_{stamp}.npz"
    R = np.array(res["rows"]["rows"], float)
    np.savez(npz, keys=np.array([it["key"] for it in items]), cells=np.array([it["cell"] for it in items]), folds=np.array([it["fold"] for it in items]),
             rows=R, **{f"L{l}": S[l] for l in LAYERS})
    rec = {"plan": "PLAN.md (this folder), POST HOC", "git": git("rev-parse", "HEAD").stdout.strip(), "which": which, "extraction run": run, "arm": arm,
           "unedited dump": list(ORIGINAL[which]), "frozen fit": FIT_RUN, "forecast.json": {"commit": FORECAST[0], "sha256 of forecast.json": FORECAST[2]},
           "probes": rl["probes sha256"], "chosen layer by test fold": chosen, "runs": len(items), "row columns": res["rows"]["columns"],
           "row count": len(R), "missing": res["rows"]["missing"], "runs without four finite nested scores, by layer": unavailable,
           "extraction check": {"file": Path(check).name, "sha256 of the check": sha(check)},
           "inputs": {"#36 runs.json": RUNS_PIN, "fit runs.json": sha(fit_runs)},
           "scores file": npz.name, "sha256 of the scores file": sha(npz),
           "status": "unevaluable" if any(unavailable.values()) or res["rows"]["missing"]["edited run without a meta"] else "scored"}
    with open(dest := HERE / "results" / f"scores_{which}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
    if rec["status"] != "scored":
        raise SystemExit(1)
