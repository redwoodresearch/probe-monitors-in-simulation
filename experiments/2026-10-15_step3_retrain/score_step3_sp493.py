"""Step-3 recipes (r1only, r2only, r12; L48; R and C) on S_paper493 (RULE Addendum P; 09's rulings relayed by 4f 10-15/16). FINAL second
refresh only; every row SEALED until 09's analysis note. Input: messages from build_sp493_message.py over d9's chain routing records, each
equal by sha256 to a committed adapter record with passing gates. Admission and routing are sp493_rules.py's (09's regexes verbatim; the
names record and the _try rule; the facts cell list; t987 refused by name; g4's T196-main rule; step-3 folds: T196 mains held out by
runs36, p311 and P_new mains fold -1), re-checked here at launch with the by-key held-out gate and its controls, the g1/g2 one-set
assert and a census. Same staged-input checks, fixed-token skips, step-3 models and row code as pass 2 (byte-identical copies).
Output (create-only): real and sim-attack arms convaudit_work:/scaleup/<folder>/step3/, sim-honest /simh/sp493_g<k>/step3/<folder>/;
per-row npz (key, run_key, unit, call_index, irid, token, n_tokens, score, fold, score_by_model), manifest_step3.json last. One
remote call per message (spawned; no map: RULE F.8).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --runs-file <json>
(--runs-file: [{"message": <path>, "source_record": "<commit>:<adapter record path>"}, ...])
"""

import hashlib
import importlib.util
import inspect
import io
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
FAM = REPO / "experiments/2026-10-14_p2_family_expansion"
APP = "probe-shift_aswin-surya_sub-step3_sp493_kill-2026-10-22"
SPEC = "2026-10-14_163027"
SPEC_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
FIT3 = {"stamp": "2026-10-15_114402", "run": "fit_2026-10-15_114402", "volume_record_sha256": "0aa4ad7db54aea16597588eda368c43053684456ef463efd59c3761eccfa2bed"}
VARIANTS = [f"step3_{r}_L48" for r in ("r1only", "r2only", "r12")]
LABEL = "S_paper493: FINAL second refresh only; SEALED until 09's analysis note; POST HOC"

if modal.is_local():
    sys.path.insert(0, str(FAM))
    import run_famexp as rf
    import run_famexp_pass2 as p2

    BUILD = rf.BUILD
else:
    BUILD = Path("/tmp/famexp_build")
B_ASIS = {"R": "unused/{L}", "C": "unused"}  # never reached: empty layer pairs, L != 63


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


if modal.is_local():
    for fn in (_models, _score_rows, _own_spec):
        if inspect.getsource(fn) != inspect.getsource(getattr(p2, fn.__name__)):
            raise SystemExit(f"{fn.__name__} differs from run_famexp_pass2.py's")

app = modal.App(APP)
OUTV = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
W3 = modal.Volume.from_name("probe-shift_aswin-surya_sub-step3_work_kill-2026-11-13", version=2).read_only()
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0").add_local_dir(BUILD, "/root/x2e")


def _row_meta(own: dict, cmd: bool):
    """Per row, in _row_calls order (same map order, same filters): the staged epoch key, irid, readout token and the epoch's n_tokens."""
    rc, _ = _x2e()
    rk, ir, tk, nt = [], [], [], []
    for key in own["map"]:
        m = json.loads((Path("/b") / own["run_of"][key] / "meta" / own["arm_of"][key] / f"{rc.stem(key)}.json").read_text())
        if cmd:
            skip = {tuple(x) for x in own.get("skip", {}).get(key, [])}
            cs = [c for c in m["calls"] if c["kind"] == own["kind"] and c.get("row") is not None and (c["irid"], c["call_index"]) not in skip]
        else:
            cs = [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
        if any(not isinstance(c.get("readout"), int) or not 0 <= c["readout"] < m["n_tokens"] for c in cs):
            raise SystemExit(f"{key}: a unit without a readout position inside the epoch")
        rk += [key] * len(cs)
        ir += [str(c["irid"]) for c in cs]
        tk += [c["readout"] for c in cs]
        nt += [m["n_tokens"]] * len(cs)
    return rk, ir, tk, nt


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=1200, retries=0, scaledown_window=2, volumes={"/out": OUTV, "/w3": W3})
def score(msg: dict, task_fold: dict, main_of_t196: dict, head: dict) -> dict:
    import numpy as np

    rc, _ = _x2e()
    base3 = Path("/w3/step3") / SPEC
    if hashlib.sha256((base3 / f"fit_{FIT3['stamp']}.json").read_bytes()).hexdigest() != FIT3["volume_record_sha256"]:
        raise RuntimeError("the step-3 fit record on the Volume differs from its committed copy")
    if hashlib.sha256((base3 / f"spec_{SPEC}.json").read_bytes()).hexdigest() != SPEC_SHA:
        raise RuntimeError("the spec copy differs from famexp's prep sha256 (fixed_ids drive the command skips)")
    spec = json.loads((base3 / f"spec_{SPEC}.json").read_text())
    # the held-out gate's premise: the fit's own fold of every T196 task equals runs36 by its main, so a mapped main's fold model held it out
    fot_of_main = {}
    for t, f in spec["fot"].items():
        if t in main_of_t196:
            fot_of_main.setdefault(main_of_t196[t], set()).add(f)
    bad_fot = sorted({m for c, m in head["main_of_cell"].items() if task_fold[c] >= 0 and fot_of_main.get(m) != {task_fold[c]}})
    if bad_fot or any(e["world"] == "sp493_t196" and task_fold[e["cell"]] != spec["fot"].get(e["cell"], -9) for e in msg["epochs"]):
        raise RuntimeError(f"routing differs from the fit's held-out folds for mains {bad_fot[:3]}")
    spec_np = spec | {"layer_choice_a": [[] for _ in spec["layer_choice_a"]]}
    fit3 = [x for g in json.loads((base3 / f"fit_{FIT3['stamp']}.json").read_text())["groups"] for x in g["fits"] if x["layer"] == 48]
    eps, fixed, bad = {}, set(spec["fixed_ids"]), []
    for path, h in sorted({(e["staged"], e["staged_sha256"]) for e in msg["epochs"]}):
        raw = Path("/b", path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise RuntimeError(f"staged inputs {path}: sha256 differs from the message")
        eps |= {(path, x["key"]): x for x in (json.loads(y) for y in raw.decode().splitlines() if y)}
    for e in msg["epochs"]:
        st = eps.get((e["staged"], e["key"]))
        mt = json.loads((Path("/b") / e["run"] / "meta" / e["arm"] / f"{rc.stem(e['key'])}.json").read_text()) if st else None
        if not st or mt["key"] != e["key"] or mt.get("inputs_sha256") != e["staged_sha256"] or [(c["irid"], c["call_index"], c["kind"]) for c in st["calls"]] != \
                [(c["irid"], c["call_index"], c["kind"]) for c in mt["calls"]]:
            bad.append(e["key"])
    if bad:
        raise RuntimeError(f"{len(bad)} epochs missing from staging or inconsistent with the meta, e.g. {bad[:2]}")
    skip = {e["key"]: [[r["irid"], r["call_index"]] for r in eps[(e["staged"], e["key"])]["calls"]
                       if r["kind"] == "call_end" and r["readout"] is not None and eps[(e["staged"], e["key"])]["ids"][r["readout"]] in fixed] for e in msg["epochs"]}
    folders = sorted({e["folder"] for e in msg["epochs"]})
    ods = {fo: Path("/out") / head["layout"][fo] for fo in folders}
    keys_msg = {e["world"] + "|" + e["cell"] for e in msg["epochs"]}
    for fo in folders:
        if head["layout"][fo].startswith("simh/"):  # one sim-honest reader convention: each key once per g batch
            for mp in sorted((ods[fo].parent).glob("*/manifest_step3.json")):
                if set(json.loads(mp.read_text())["keys"]) & keys_msg:
                    raise RuntimeError(f"{mp}: a key of this message is already scored in this batch")
        if ods[fo].exists() and any(ods[fo].iterdir()):
            raise RuntimeError(f"{ods[fo]} already holds files: never overwritten")
    keys_of = {fo: {e["world"] + "|" + e["cell"] for e in msg["epochs"] if e["folder"] == fo} for fo in folders}
    fo_of = {e["world"] + "|" + e["cell"]: e["folder"] for e in msg["epochs"]}
    blobs, files, counts, missing = ({fo: {} for fo in folders} for _ in range(4))
    wsha = {}
    for readout in ("R", "C"):
        own = _own_spec(msg["epochs"], readout, skip)
        res, ws = _score_rows(own, 48, task_fold, spec_np, fit3, FIT3["run"], base3, readout, {})
        wsha |= ws
        meta = _row_meta(own, readout == "C") if res else None
        for v in VARIANTS:
            if v not in res:
                for fo in folders:
                    missing[fo][f"{v}|{readout}"] = sorted(keys_of[fo])
                continue
            names, units, ci, folds, S = res[v]
            names = names.astype(str)
            rk, ir, tk, nt = (np.asarray(x) for x in meta)
            ci_ref, _ = rc._row_calls(own, cmd=(readout == "C"))
            if len(rk) != len(names) or list(ci_ref) != ci.tolist() or [str(own["map"][k][0]) + "|" + str(own["map"][k][1]) for k in rk.tolist()] != names.tolist():
                raise RuntimeError(f"{v} {readout}: the per-row meta does not line up with the scored rows")
            if not np.isfinite(S).all() or any(folds[i] != task_fold[k.split("|", 1)[1]] for i, k in enumerate(names.tolist())):
                raise RuntimeError(f"{v} {readout}: a non-finite score or a row not routed by its main")
            for fo in folders:
                sel = np.array([fo_of[k] == fo for k in names.tolist()])
                if not sel.any():
                    missing[fo][f"{v}|{readout}"] = sorted(keys_of[fo])
                    continue
                f_, S_ = folds[sel], S[sel]
                sc = np.where(f_ >= 0, S_[np.arange(len(f_)), np.clip(f_, 0, 4)], np.nan)
                buf = io.BytesIO()
                np.savez(buf, key=names[sel], run_key=rk[sel], unit=units[sel], call_index=ci[sel], irid=ir[sel], token=tk[sel].astype(np.int64),
                         n_tokens=nt[sel].astype(np.int64), score=sc, fold=f_, score_by_model=S_)
                blobs[fo][f"{v}__{readout}.npz"] = buf.getvalue()
                files[fo][f"{v}__{readout}.npz"] = hashlib.sha256(buf.getvalue()).hexdigest()
                counts[fo][f"{v}__{readout}"] = {"n_runs": len(set(rk[sel].tolist())), "n_units": int(sel.sum()), "n_units_mapped": int((f_ >= 0).sum()),
                                                 "n_units_fold_minus1": int((f_ < 0).sum())}  # 09 am.1 section 2's per-probe x readout census
                gone = sorted(keys_of[fo] - set(names[sel].tolist()))
                if gone:
                    missing[fo][f"{v}|{readout}"] = gone
    out = {}
    for fo, od in ods.items():
        od.mkdir(parents=True, exist_ok=True)
        for fn, b_ in sorted(blobs[fo].items()):
            (od / fn).open("xb").write(b_)
        man = {k: v for k, v in head.items() if k not in ("layout", "main_of_cell")} | {
            "folder": fo, "files": files[fo], "counts": counts[fo], "runs_without_rows": missing[fo], "keys": sorted(keys_of[fo]),
            "routing": {k: head["routing"][k] for k in sorted(keys_of[fo])}, "weights_sha256": wsha, "complete": not missing[fo] and len(files[fo]) == 2 * len(VARIANTS)}
        mb = json.dumps(man, indent=1, default=str).encode()
        (od / "manifest_step3.json").open("xb").write(mb)
        out[fo] = {"manifest": str(od / "manifest_step3.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest(), "counts": counts[fo], "complete": man["complete"],
                   "keys": sorted(keys_of[fo])}  # read back by later launches' g1/g2 one-set assert
    OUTV.commit()
    return out


@app.local_entrypoint()
def main(runs_file: str):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    on_origin = lambda c: any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c).decode().splitlines())  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/step3-retrain") \
            or git("ls-files", "--error-unmatch", str(Path(__file__).relative_to(REPO))).strip() == b"":
        raise SystemExit("uncommitted changes, HEAD not pushed, or this scorer untracked")
    sp = importlib.util.spec_from_file_location("sp493_rules", HERE / "sp493_rules.py")
    R = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(R)
    P = R.load()
    jobs, sets = [], {}
    for rp in sorted((HERE / "results").glob("sp493_*.json")):  # earlier committed sp493 records join the g1/g2 one-set assert
        r_ = json.loads(rp.read_text())
        for fo, x in (r_.get("result") or {}).items():
            sets.setdefault(fo.split("__")[0], set()).update(x.get("keys", []))
    for job in json.loads(Path(runs_file).read_text()):
        raw = Path(job["message"]).read_bytes()
        rc_, rp_ = job["source_record"].split(":", 1)
        if not on_origin(rc_):
            raise SystemExit(f"{rp_}: the adapter record's commit is not on origin")
        rb_ = git("show", f"{rc_}:{rp_}")
        rec = json.loads(rb_)
        if hashlib.sha256(raw).hexdigest() != rec.get("message_sha256") or not rec.get("gate") or not all(g.get("pass") is True for g in rec["gate"].values()) \
                or "sample_error_null" not in rec["gate"]:
            raise SystemExit(f"{rp_}: message sha256 differs from the committed adapter record's, or a gate (incl. sample_error) did not pass")
        msg = json.loads(raw)
        folders = sorted({e["folder"] for e in msg["epochs"]})
        why = {fo: R.refuse_folder(fo, P) for fo in folders}
        if any(why.values()) or len({fo.split("__")[0] for fo in folders}) != 1:
            raise SystemExit(f"{rp_}: refused folders {[(f, w) for f, w in why.items() if w]} or more than one staged set")
        routing, cells, mains = {}, {}, {}
        for e in msg["epochs"]:
            if R.world_of(e["folder"], e["d9_world"]) != e["world"] or R.STRATUM[R.kind_of(e["folder"])] != e["stratum"]:
                raise SystemExit(f"{e['key']}: world or stratum differs from 09's map")
            main, fold, reason = R.admit_cell(e["world"], e["cell"], P)
            if reason:
                raise SystemExit(f"{e['key']}: refused ({reason}); the adapter must list it, never route it (fail closed)")
            origin = "t196" if e["world"] == "sp493_t196" else P["facts"][e["cell"]]["main_origin"]
            routing[e["world"] + "|" + e["cell"]] = {"fold": fold, "main": main, "main_origin": origin,
                                                     "basis": "T196 main: the step-3 fold model that held it out (runs36)" if fold >= 0 else
                                                     (R.P_NEW_LABEL if origin == "new" else "+311 main: step-3 never trained on it: fold -1, all five models")}
            cells[e["cell"]], mains[e["cell"]] = fold, main
            sets.setdefault(e["folder"].split("__")[0], set()).add(e["world"] + "|" + e["cell"])
        gate = R.heldout_gate(list(zip(mains.values(), cells.values())), P)
        if not gate["pass"]:
            raise SystemExit(f"{rp_}: the held-out gate or its controls failed: {gate}")
        layout = {fo: (f"simh/{re.match(r'^(sp493_g[0-9]+)_simh', fo).group(1)}/step3/{fo}" if R.kind_of(fo) == "sim_honest" else f"scaleup/{fo}/step3")
                  for fo in folders}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        head = {"label": "step-3 recipes on S_paper493 (RULE Addendum P): SEALED; hashes, counts and routing only", "row_label": LABEL, "status": "POST HOC, SEALED",
                "scorer_git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp, "message_sha256": rec["message_sha256"],
                "source_record": {"commit": rc_, "path": rp_, "sha256": hashlib.sha256(rb_).hexdigest()}, "routing_record": rec["pins"].get("routing"),
                "pins": {n: {"commit": c, "path": p, "sha256_prefix": h} for n, (c, p, h) in {"regex": R.REGEX_PIN, "facts": R.FACTS_PIN, "names": R.NAMES_PIN,
                                                                                               "t196": R.T196_PIN, "r36": R.R36_PIN}.items()},
                "refused_by_name": msg.get("refused", []), "heldout_gate": gate, "census_cells": R.census({msg["epochs"][0]["stratum"]: list(zip(mains.values(), cells.values()))}),
                "routing": routing, "layout": layout, "main_of_cell": mains, "step3_fit": FIT3,
                "schema": "per-row only: key (world|cell), run_key, unit, call_index, irid, token, n_tokens, score (designated model; NaN on fold -1), fold, score_by_model"}
        jobs.append((msg, cells, head))
    dup = R.one_set_per_cell(sets)
    if dup:
        raise SystemExit(f"STOP: a sim-attack cell is in two admitted sets (09 00:2x / 00:58): {dup[:3]}")
    tb = R.tail_assert(sets, P)  # 09's per-set half (01:2x): per-row group and launch; partial releases score; the union is final_tail_assert.py's
    if tb:
        raise SystemExit(f"STOP: 09's per-set tail assert fails: {tb[:3]}")
    main_of_t196 = {k: c["main"] for k, c in P["t196"].items()}
    calls = [(m, h, score.with_options(volumes={"/out": OUTV, "/w3": W3, "/b": modal.Volume.from_name(m["volume"], version=2).read_only()}).spawn(m, t, main_of_t196, h))
             for m, t, h in jobs]  # one call per message (no map: RULE F.8)
    failed = []
    for m, h, fc in calls:
        p = HERE / "results" / f"sp493_{m['d9_arm']}_{h['stamp (PT)']}.json"
        try:
            out = fc.get()
        except (Exception, SystemExit) as ex:  # a remote gate's SystemExit too (Codex cutfull-b r1)  # recorded and re-raised below; the other messages' records are still written
            failed.append(f"{m['d9_arm']}: {type(ex).__name__}: {ex}")
            p.with_name(p.stem + "_FAILED.json").open("x").write(json.dumps({k: v for k, v in h.items() if k not in ("layout", "main_of_cell")} | {"error": failed[-1]}, indent=1))
            continue
        p.open("x").write(json.dumps({k: v for k, v in h.items() if k not in ("layout", "main_of_cell")} | {"function_call": fc.object_id, "result": out}, indent=1))
        print(json.dumps({fo: {k: v for k, v in r.items() if k != "counts"} for fo, r in out.items()}, indent=1))
    if failed:
        raise SystemExit(f"{len(failed)} of {len(jobs)} messages FAILED: {failed}")
