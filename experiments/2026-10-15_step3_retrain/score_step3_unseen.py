"""Step 3 unseen-launch control (RULE.md): score S_new (su311) and IL (il311, ilold) messages with the step-3 recipes at L48 (R, C),
using famexp pass 2's own messages (sha256 equal to famexp's committed pass-2 records) and its row code. Raw logits of all five fold
models per row (`score_by_model`, N x 5) plus `fold` (the main's Sol #36 fold, -1 if none); no run score here. Writes only to the
step-3 work Volume: /step3/<spec>/unseen_<stamp>/<d9_arm>/<variant>__<R|C>.npz + record.json. IL files are never read here.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --fit-stamp <stamp> --runs-file <json>
(--runs-file: {"<d9_arm>": "<message path>"}; each message's sha256 must equal famexp's pass2_<d9_arm>_*.json message_sha256.)
"""

import glob
import hashlib
import inspect
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
FAM = REPO / "experiments/2026-10-14_p2_family_expansion"
APP = "probe-shift_aswin-surya_sub-step3_score_kill-2026-10-22"
SPEC = "2026-10-14_163027"
L = 48
RECIPES = ("r1only", "r2only", "r12")
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
R36 = ("c24106ca0", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json", "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")
if modal.is_local():
    sys.path.insert(0, str(FAM))
    import run_famexp as rf  # materializes and sha-checks the pinned 2e row code into /tmp/famexp_build
    import run_famexp_pass2 as p2

    BUILD = rf.BUILD
else:
    BUILD = Path("/tmp/famexp_build")


def _x2e():
    sys.path.insert(0, "/root/x2e")
    import run_correctedb as rc

    return rc, rc.rat


def _models(spec, fit, L, readout):
    """(variant, fold) -> weights at layer L and readout: pass 1's fit record (own_a, adapted; certified only) or 2e's b_asis file."""
    pair, out = spec["layer_choice_a"], {}
    for x in fit:
        if x["layer"] == L and x["readout"] == readout and x["certified"]:
            out[(x["variant"], x["fold"])] = ("p1", x["sha256"]["weights"], f"{x['variant']}__{readout}__L{L}__f{x['fold']}.safetensors")
    if readout == "C" and L == 63:
        out |= {("b_asis", f): ("b_asis", None, B_ASIS["C"]) for f in range(5)}
    if readout == "R":
        out |= {("b_asis", f): ("b_asis", None, B_ASIS["R"].format(L=L)) for f in range(5) if L in pair[f]}
    return out


B_ASIS = {"R": "unused/{L}", "C": "unused"}  # never reached: the spec copy passed below has empty layer pairs and L != 63


def _score_rows(own, L, task_fold, spec, fit, fit_run, base, readout, cache):
    """Rows of `own` at L, scored by every (variant, fold) model there: {variant: (names, units, call_index, folds, S [N x 5])}, weights."""
    import numpy as np
    from safetensors.numpy import load_file

    rc, rat = _x2e()
    models = _models(spec, fit, L, readout)
    if not models:
        return {}, {}
    try:
        X, names, folds = (rat._b_rows if readout == "R" else rat._b_rows_cmd)(own, L, task_fold)
    except ValueError as e:  # 2e's loader cannot concatenate zero arrays: no staged epoch has a row in this readout
        if "need at least one array to concatenate" not in str(e):
            raise
        return {}, {}  # every run of the message is then listed as runs_without_rows (orchestrate2), never imputed
    ci, _ = rc._row_calls(own, cmd=(readout == "C"))
    X32, X = X.astype(np.float32), X.astype(np.float64)
    out, wsha = {}, {}
    for v in sorted({k[0] for k in models}):
        S = np.full((len(names), 5), np.nan)
        for f in range(5):
            if (v, f) not in models:
                continue
            kind, want, path = models[(v, f)]
            p = (base / f"run_{fit_run}" / "weights" / path) if kind == "p1" else Path("/sol", path)
            if p not in cache:
                b_ = p.read_bytes()
                cache[p] = (hashlib.sha256(b_).hexdigest(), load_file(str(p)) if kind == "p1" else dict(np.load(io.BytesIO(b_))))
            h, Z = cache[p]
            if want and h != want:
                raise SystemExit(f"{v} f{f} L{L}: pass-1 weights sha256 differs from the fit record")
            pre = "" if kind == "p1" else next((q for q in ("b_asis__", "b_asis_tight__") if f"{q}f{f}__W" in Z), None)
            if pre is None:
                raise SystemExit(f"b_asis weights: no b_asis(_tight)__f{f}__W key in {path}")
            w = {k: Z[f"{pre}f{f}__{k}" if pre else k] for k in ("W", "b", "mean", "std")}
            Xs = ((X - w["mean"]) / w["std"]) if kind == "p1" else ((X32 - w["mean"].astype(np.float32)) / w["std"].astype(np.float32)).astype(np.float64)
            S[:, f] = Xs @ np.asarray(w["W"], np.float64).reshape(-1) + float(np.asarray(w["b"]).reshape(-1)[0])
            wsha[f"{v}|{readout}|L{L}|f{f}"] = h
        out[v] = (names, np.asarray(rc._units(names), np.int64), np.asarray(ci, np.int64), np.asarray(folds, np.int64), S)
    return out, wsha


def _own_spec(epochs, readout, skip):
    return {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": {e["key"]: [e["world"], e["cell"]] for e in epochs},
            "arm_of": {e["key"]: e["arm"] for e in epochs}, "run_of": {e["key"]: e["run"] for e in epochs}} | ({"skip": skip} if readout == "C" else {})


if modal.is_local():  # the copied functions must be famexp pass 2's, byte for byte
    for fn in (_models, _score_rows, _own_spec):
        if inspect.getsource(fn) != inspect.getsource(getattr(p2, fn.__name__)):
            raise SystemExit(f"{fn.__name__} differs from run_famexp_pass2.py's")

app = modal.App(APP)
WORK = modal.Volume.from_name("probe-shift_aswin-surya_sub-step3_work_kill-2026-11-13", version=2)
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0").add_local_dir(BUILD, "/root/x2e")


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=900, retries=0, scaledown_window=2, volumes={"/out": WORK})
def score_msg(fit_stamp: str, fit_run: str, msg: dict, task_fold: dict, stamp: str) -> dict:
    """famexp pass 2's staged-input checks and fixed-token skips (score_layer), then the step-3 models at L48, R and C."""
    import numpy as np
    rc, _ = _x2e()
    base = Path("/out/step3") / SPEC
    spec = json.loads((base / f"spec_{SPEC}.json").read_text())
    spec_np = spec | {"layer_choice_a": [[] for _ in spec["layer_choice_a"]]}  # no b_asis models here
    fit = [x for g in json.loads((base / f"fit_{fit_stamp}.json").read_text())["groups"] for x in g["fits"] if x["layer"] == L]
    if {g["run"] for g in json.loads((base / f"fit_{fit_stamp}.json").read_text())["groups"]} != {fit_run}:
        raise SystemExit("the fit record spans more than one run folder (a retried group): extend this scorer first")
    eps, fixed, bad, cfgs = {}, set(spec["fixed_ids"]), [], {}
    for e in msg["epochs"]:
        if e["staged_sha256"] == "from_config":
            if e["run"] not in cfgs:
                cfgs[e["run"]] = json.loads((Path("/b") / e["run"] / "config.json").read_text())["inputs_sha256"]
            e["staged_sha256"] = cfgs[e["run"]][f"{e['arm']}.jsonl"]
    for path, h in sorted({(e["staged"], e["staged_sha256"]) for e in msg["epochs"]}):
        raw = Path("/b", path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"staged inputs {path}: sha256 differs from the message")
        eps |= {(path, x["key"]): x for x in (json.loads(y) for y in raw.decode().splitlines() if y)}
    for e in msg["epochs"]:
        st = eps.get((e["staged"], e["key"]))
        mt = json.loads((Path("/b") / e["run"] / "meta" / e["arm"] / f"{rc.stem(e['key'])}.json").read_text()) if st else None
        if not st or mt["key"] != e["key"] or mt.get("inputs_sha256") != e["staged_sha256"] or [(c["irid"], c["call_index"], c["kind"]) for c in st["calls"]] != \
                [(c["irid"], c["call_index"], c["kind"]) for c in mt["calls"]]:
            bad.append(e["key"])
    if bad:
        raise SystemExit(f"{len(bad)} epochs missing from staging or inconsistent with the extraction meta, e.g. {bad[:2]}")
    skip = {e["key"]: [[r["irid"], r["call_index"]] for r in eps[(e["staged"], e["key"])]["calls"]
                       if r["kind"] == "call_end" and r["readout"] is not None and eps[(e["staged"], e["key"])]["ids"][r["readout"]] in fixed] for e in msg["epochs"]}
    od = base / f"unseen_{stamp}" / msg["d9_arm"]
    rec, cache = {"d9_arm": msg["d9_arm"], "files": {}, "weights": {}, "runs_without_rows": {}}, {}
    keys = {e["world"] + "|" + e["cell"] for e in msg["epochs"]}
    for readout in ("R", "C"):
        res, wsha = _score_rows(_own_spec(msg["epochs"], readout, skip), L, task_fold, spec_np, fit, fit_run, base, readout, cache)
        rec["weights"] |= wsha
        for v, (names, units, ci, folds, S) in res.items():
            buf = io.BytesIO()
            np.savez(buf, key=np.asarray(names), unit=units, call_index=ci, fold=folds, score_by_model=S)
            fp = od / f"{v}__{readout}.npz"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.open("xb").write(buf.getvalue())
            rec["files"][fp.name] = hashlib.sha256(buf.getvalue()).hexdigest()
            rec["runs_without_rows"][f"{v}|{readout}"] = sorted(keys - set(names.tolist()))
    rec["complete"] = len(rec["files"]) == 2 * len(RECIPES)
    (od / "record.json").open("x").write(json.dumps(rec, indent=1))
    WORK.commit()
    return rec


@app.function(image=img, cpu=(2, 2), memory=(4096, 4096), timeout=2400, retries=0, scaledown_window=2, volumes={"/out": WORK})
def orchestrate(fit_stamp: str, fit_run: str, msgs: dict, folds: dict, stamp: str) -> dict:
    out = {}
    for attempt in (1, 2):
        todo = [a for a in msgs if a not in out or "error" in out[a]]
        calls = {a: score_msg.with_options(volumes={"/out": WORK, "/b": modal.Volume.from_name(msgs[a]["volume"], version=2).read_only()}).spawn(
            fit_stamp, fit_run, msgs[a], folds[a], stamp if attempt == 1 else f"{stamp}_a2") for a in todo}
        for a, c in calls.items():
            try:
                r = c.get()
                out[a] = {k: v for k, v in r.items() if k != "weights"} | {"attempt": attempt}
            except BaseException as e:  # noqa: BLE001 - recorded, retried once into its own folder
                out[a] = {"error": type(e).__name__, "detail": str(e)[:300], "attempt": attempt}
    raw = json.dumps(out, indent=1).encode()
    p = Path("/out/step3") / SPEC / f"unseen_{stamp}.json"
    p.open("xb").write(raw)
    WORK.commit()
    return {"record": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "per_message": {a: {k: v for k, v in r.items() if k not in ("files", "runs_without_rows")} for a, r in out.items()}}


@app.local_entrypoint()
def main(fit_stamp: str, runs_file: str):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted changes: commit and push first")
    show = lambda c, p: git("show", f"{c}:{p}")  # noqa: E731
    pins = {}
    for nm, (c, p, h) in {"q2": Q2, "t196": T196, "r36": R36}.items():
        raw = show(c, p)
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"{nm} sha256 differs from its pin")
        pins[nm] = json.loads(raw)
    main_of = {c["key"]: c["main"] for c in pins["q2"]["cells"]} | {c["key"]: c["main"] for c in pins["t196"]["cells"]}
    fom = {r["main_task"]: r["fold"] for r in pins["r36"]}
    t196_mains = {c["main"] for c in pins["t196"]["cells"]}
    if {c["main"] for c in pins["q2"]["cells"]} & t196_mains:
        raise SystemExit("an S_new main is also a T196 main: its fold models may have trained on it")
    runs = json.loads(Path(runs_file).read_text())
    msgs, folds, shas = {}, {}, {}
    for arm, mp in sorted(runs.items()):
        recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(FAM / "results" / f"pass2_{arm}_2026-*.json"))) if "_note_" not in f]
        want = {r_["message_sha256"] for r_ in recs if r_.get("result", {}).get("gate_pass") is True}
        raw = Path(mp).read_bytes()
        if hashlib.sha256(raw).hexdigest() not in want or len(want) != 1:
            raise SystemExit(f"{arm}: message sha256 is not famexp pass 2's (one passing record expected)")
        m = json.loads(raw)
        if m["d9_arm"] != arm or any(e["stratum"] == "S_newmains" for e in m["epochs"]):
            raise SystemExit(f"{arm}: d9_arm mismatch or S_newmains rows (out of scope)")
        msgs[arm], shas[arm] = m, hashlib.sha256(raw).hexdigest()
        folds[arm] = {e["cell"]: fom.get(main_of[e["cell"]], -1) for e in m["epochs"]}
    fit = json.loads(Path(HERE / "results" / f"fit_{fit_stamp}.json").read_text())["result"]
    fit_run = fit["run"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = orchestrate.remote(fit_stamp, fit_run, msgs, folds, stamp)
    rec = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp, "app": APP, "fit_stamp": fit_stamp, "message_sha256": shas,
           "no_fold_cells": {a: sorted(c for c, f in fd.items() if f < 0) for a, fd in folds.items()}, "result": out}
    (HERE / "results" / f"unseen_{stamp}.json").open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "no_fold_cells"} | {"no_fold_cells_n": {a: len(v) for a, v in rec["no_fold_cells"].items()}}, indent=1)[:5000])
