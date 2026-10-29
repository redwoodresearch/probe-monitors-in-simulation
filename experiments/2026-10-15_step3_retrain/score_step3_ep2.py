"""Step 3 recipes on epoch-2 REAL-HONEST roots (09's Append 38 @ 0f5d1b429; sealed scoring permitted by 09's READ_LOG @ 96b6c4fb4).
SEALED: per-row scores only, nothing derived; no epoch-2 outcome is read by anyone before 09's Append 38 section 4 pool-key artifact is
committed and verified. Input: ONE message (pass-2 schema; one extraction root, one routed folder per arm) whose sha256 equals a
committed adapter record (build_ep2_message.py) with passing gates. Same staged-input checks, fixed-token skips and step-3 models
(step3_{r1only,r2only,r12}_L48, R and C) as score_step3_simh_batch.py. Routing by main (runs36): T196 cells use the fold model that
held the task out (checked against the fit's fot); +311 cells their mapped held-out model or fold -1; nm360 cells fold -1 (all five
kept). Output, create-only, on convaudit_work: /ep2/<pool>/step3/<folder>/step3_<recipe>_L48__<R|C>.npz (pass 1's schema +
run_score_by_model) and manifest_step3.json last.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --message <json> --source-record <commit>:<path>
"""

import hashlib
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
APP = "probe-shift_aswin-surya_sub-step3_ep2_kill-2026-10-22"
SPEC = "2026-10-14_163027"
SPEC_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
FIT3 = {"stamp": "2026-10-15_114402", "run": "fit_2026-10-15_114402", "volume_record_sha256": "0aa4ad7db54aea16597588eda368c43053684456ef463efd59c3761eccfa2bed"}
VARIANTS = [f"step3_{r}_L48" for r in ("r1only", "r2only", "r12")]
APPEND_38, EP2_GATE = "0f5d1b429", "96b6c4fb4"  # 09's Append 38 and its READ_LOG ruling "epoch-2 scoring may run sealed"
_FOLDER_RE_BASE = re.compile(r"^ep2_(t196|311_l1|311_l2|nm)_v1(_try[0-9]+)?__honest_(a|b)$")  # 09's tested regex, verbatim
# CUT lift (Aswin via 0d/2e, 10-15 19:0x), flags OFF: d9's names <original_target>_cutfull_v1(_try<n>)?__<arm> on the same output roots; gate
# sets <original_target>_cutfull_gate_v1__<arm> are never scored or pooled. 09's ONE tested regex extension goes into CUTFULL["regex"] verbatim
# when relayed; until then (and unless --cutfull is passed) every cutfull folder is refused and nothing else changes.
CUTFULL = {"enabled": False, "regex": None}
GATE_SET_RE = re.compile(r"_cutfull_gate_v1(?:_try[0-9]+)?__")


class _Grammar:
    """FOLDER_RE with the cutfull hook: a cutfull name is admitted only with the flag ON and 09's regex set, and is then matched on its
    canonical (cut) form, so batch / pool groups and every rule keyed on them are unchanged; output paths keep the real folder name."""

    def fullmatch(self, f):
        if GATE_SET_RE.search(f):
            return None
        if "_cutfull_v1" in f:
            if not CUTFULL["enabled"] or not CUTFULL["regex"] or not re.fullmatch(CUTFULL["regex"], f):
                return None
            f = f.replace("_cutfull_v1", "_v1", 1)
        return _FOLDER_RE_BASE.fullmatch(f)


FOLDER_RE = _Grammar()
STRATUM = {"t196": "S_old", "311_l1": "S_new", "311_l2": "S_new", "nm": "S_newmains"}
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
NEWMAINS = ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
R36 = ("c24106ca0", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json", "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")
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


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=1200, retries=0, scaledown_window=2, volumes={"/out": OUTV, "/w3": W3})
def score(msg: dict, task_fold: dict, head: dict) -> dict:
    import numpy as np
    CUTFULL["enabled"] = bool(head.get("cutfull"))  # the launch's flag, in-container
    rc, _ = _x2e()
    base3 = Path("/w3/step3") / SPEC
    if hashlib.sha256((base3 / f"fit_{FIT3['stamp']}.json").read_bytes()).hexdigest() != FIT3["volume_record_sha256"]:
        raise SystemExit("the step-3 fit record on the Volume differs from its committed copy")
    if hashlib.sha256((base3 / f"spec_{SPEC}.json").read_bytes()).hexdigest() != SPEC_SHA:
        raise SystemExit("the spec copy differs from famexp's prep sha256 (fixed_ids drive the command skips)")
    spec = json.loads((base3 / f"spec_{SPEC}.json").read_text())
    spec_np = spec | {"layer_choice_a": [[] for _ in spec["layer_choice_a"]]}
    fit3 = [x for g in json.loads((base3 / f"fit_{FIT3['stamp']}.json").read_text())["groups"] for x in g["fits"] if x["layer"] == 48]
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
    if head["pool"] == "t196":  # T196 runs: the designated model must be the one that held the task out (spec fot = runs36 by main)
        bad_f = [e["cell"] for e in msg["epochs"] if task_fold[e["cell"]] != spec["fot"].get(e["cell"], -9)]
        if bad_f:
            raise SystemExit(f"T196 routing differs from the fit's held-out folds: {bad_f[:3]}")
    skip = {e["key"]: [[r["irid"], r["call_index"]] for r in eps[(e["staged"], e["key"])]["calls"]
                       if r["kind"] == "call_end" and r["readout"] is not None and eps[(e["staged"], e["key"])]["ids"][r["readout"]] in fixed] for e in msg["epochs"]}
    excl = {}
    for p_ in msg.get("provenance", []):  # pass 2's staging exclusions: kept per folder, never dropped (Codex simhb r1 item 4); same grammar and batch
        if any(not FOLDER_RE.fullmatch(x["folder"]) or FOLDER_RE.fullmatch(x["folder"]).group(1) != head["pool"] for x in p_.get("excluded_by_staging", [])):
            raise SystemExit("a staging exclusion outside this batch's folder grammar")
        for x in p_.get("excluded_by_staging", []):
            excl.setdefault(x["folder"], []).append(x)
    folders = sorted({e["folder"] for e in msg["epochs"]} | set(excl))
    ods = {fo: Path("/out/ep2") / FOLDER_RE.fullmatch(fo).group(1) / "step3" / fo for fo in folders}  # 09 Append 38; layout as 25h section 11
    for od in ods.values():
        if od.exists() and any(od.iterdir()):
            raise SystemExit(f"{od} already holds files: never overwritten")
    keys_of = {fo: {e["world"] + "|" + e["cell"] for e in msg["epochs"] if e["folder"] == fo} for fo in folders}
    blobs, files, counts, missing = ({fo: {} for fo in folders} for _ in range(4))
    wsha = {}
    fo_of = {e["world"] + "|" + e["cell"]: e["folder"] for e in msg["epochs"]}
    for readout in ("R", "C"):
        res, ws = _score_rows(_own_spec(msg["epochs"], readout, skip), 48, task_fold, spec_np, fit3, FIT3["run"], base3, readout, {})
        wsha |= ws
        for v in VARIANTS:
            if v not in res:
                for fo in folders:
                    missing[fo][f"{v}|{readout}"] = sorted(keys_of[fo])
                continue
            names, units, ci, folds, S = res[v]
            names = names.astype(str)
            if not np.isfinite(S).all() or any(folds[i] != task_fold[k.split("|", 1)[1]] for i, k in enumerate(names.tolist())):
                raise SystemExit(f"{v} {readout}: a non-finite score or a row not routed by its main")
            for fo in folders:
                sel = np.array([fo_of[k] == fo for k in names.tolist()])
                if not sel.any():
                    missing[fo][f"{v}|{readout}"] = sorted(keys_of[fo])
                    continue
                k_, u_, c_, S_, f_ = names[sel], units[sel], ci[sel], S[sel], folds[sel]
                score = np.where(f_ >= 0, S_[np.arange(len(f_)), np.clip(f_, 0, 4)], np.nan)  # the designated model's ROW score (NaN on fold -1)
                rk, first = np.unique(k_, return_index=True)  # run keys and their folds for counting only; no run aggregate is computed (SEALED)
                rfo = f_[first]
                buf = io.BytesIO()
                np.savez(buf, key=k_, unit=u_, call_index=c_, score=score, score_by_model=S_, fold=f_)  # per-row only (09: no derived quantities)
                blobs[fo][f"{v}__{readout}.npz"] = buf.getvalue()
                files[fo][f"{v}__{readout}.npz"] = hashlib.sha256(buf.getvalue()).hexdigest()
                counts[fo][f"{v}__{readout}"] = {"n_runs": len(rk), "n_units": len(k_), "n_runs_fold_minus1": int((rfo < 0).sum())}
                gone = sorted(keys_of[fo] - set(rk.tolist()))
                if gone:
                    missing[fo][f"{v}|{readout}"] = gone
    out = {}
    for fo, od in ods.items():
        od.mkdir(parents=True, exist_ok=True)
        for fn, b_ in sorted(blobs[fo].items()):
            (od / fn).open("xb").write(b_)
        man = head | {"folder": fo, "excluded_by_staging": excl.get(fo, []), "files": files[fo], "counts": counts[fo], "runs_without_rows": missing[fo], "keys": sorted(keys_of[fo]),
                      "routing": {k: {"fold": task_fold[k.split("|", 1)[1]], "basis": "mapped: held-out step-3 model" if task_fold[k.split("|", 1)[1]] >= 0 else "unmapped: fold -1, all five models"}
                                  for k in sorted(keys_of[fo])}, "weights_sha256": wsha, "complete": not missing[fo] and len(files[fo]) == 2 * len(VARIANTS)}
        mb = json.dumps(man, indent=1, default=str).encode()
        (od / "manifest_step3.json").open("xb").write(mb)
        out[fo] = {"manifest": str(od / "manifest_step3.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest(), "counts": counts[fo], "complete": man["complete"]}
    OUTV.commit()
    return out




@app.local_entrypoint()
def main(message: str, source_record: str, cutfull: bool = False, profile: str = "cut32k"):
    if profile != "cut32k":  # the full-length refit profile (RULE Addendum F) is not pinned yet: refused
        raise SystemExit(f"profile {profile!r} is not pinned (only cut32k: the fit on cut rows)")
    if cutfull and not CUTFULL["regex"]:
        raise SystemExit("--cutfull needs 09's tested cutfull regex in CUTFULL['regex'] (not yet relayed)")
    CUTFULL["enabled"] = bool(cutfull)
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/step3-retrain"):
        raise SystemExit("uncommitted changes, or HEAD is not pushed")
    for c_ in (APPEND_38, EP2_GATE):
        if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c_).decode().splitlines()):
            raise SystemExit(f"09's commit {c_} is not on origin")
    rc_, rp_ = source_record.split(":", 1)
    if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", rc_).decode().splitlines()):
        raise SystemExit("the source record's commit is not on origin")
    rb_ = git("show", f"{rc_}:{rp_}")
    rec = json.loads(rb_)
    raw = Path(message).read_bytes()
    if hashlib.sha256(raw).hexdigest() != rec.get("message_sha256") or not rec.get("gate") or not all(g.get("pass") is True for g in rec["gate"].values()):
        raise SystemExit("the message sha256 differs from the committed adapter record's, or its gates did not pass")
    msg = json.loads(raw)
    keys = [e["key"] for e in msg["epochs"]]
    pools = {FOLDER_RE.fullmatch(e["folder"]).group(1) if FOLDER_RE.fullmatch(e["folder"]) else None for e in msg["epochs"]}
    if None in pools or len(pools) != 1 or len(set(keys)) != len(keys) or len({(e["world"], e["cell"]) for e in msg["epochs"]}) != len(msg["epochs"]):
        raise SystemExit("a folder outside 09's ep2 grammar, mixed pools, or a duplicate key / world|cell")
    (pool,) = pools
    if any(e["world"] not in ("honest_a", "honest_b") or FOLDER_RE.fullmatch(e["folder"]).group(3) != e["world"][-1] or e["stratum"] != STRATUM[pool] for e in msg["epochs"]):
        raise SystemExit("a world, folder letter or stratum mismatch")
    for p_ in msg.get("provenance", []):  # H200 on every shard of every contributing root
        dr = p_.get("device_record") or {}
        drec = json.loads(git("show", f"{dr['commit']}:{dr['path']}"))
        if not drec.get("shards") or any((sh.get("gpu") or sh.get("gpu_name")) != "NVIDIA H200" for sh in drec["shards"]):
            raise SystemExit(f"{p_.get('run')}: the device record does not show NVIDIA H200 on every shard")
    pins = {}
    for nm, (c, p, h) in {"q2": Q2, "newmains": NEWMAINS, "t196": T196, "r36": R36}.items():
        b = git("show", f"{c}:{p}")
        if hashlib.sha256(b).hexdigest() != h:
            raise SystemExit(f"{nm} sha256 differs from its pin")
        pins[nm] = json.loads(b)
    lists = {"S_old": {c["key"]: c["main"] for c in pins["t196"]["cells"]}, "S_new": {c["key"]: c["main"] for c in pins["q2"]["cells"]},
             "S_newmains": {c["key"]: c["main"] for c in pins["newmains"]["cells"]}}
    main_of = lists[STRATUM[pool]]
    if any(e["cell"] not in main_of for e in msg["epochs"]):
        raise SystemExit("a cell outside its pool's pinned list")
    t196_mains = set(lists["S_old"].values())
    if pool != "t196" and any(main_of[e["cell"]] in t196_mains for e in msg["epochs"]):
        raise SystemExit("a non-T196 cell's main is a T196 main: the step-3 models may have trained on it")
    fom = {}
    for r in pins["r36"]:
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    task_fold = {e["cell"]: fom.get(main_of[e["cell"]], -1) for e in msg["epochs"]}
    if pool == "t196" and any(f < 0 for f in task_fold.values()):
        raise SystemExit("a T196 cell without a runs36 fold")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"label": "step-3 recipes on an epoch-2 real-honest root: SEALED per 09's Append 38 sections 4/5 (per-row scores only; nothing derived; "
                     "unread until 09's pool-key artifact is committed and verified)", "SEALED": True,
            "scorer_git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp, "append_38": APPEND_38, "ep2_scoring_gate": EP2_GATE, "pool": pool,
            "message_sha256": rec["message_sha256"], "source_record": {"commit": rc_, "path": rp_, "sha256": hashlib.sha256(rb_).hexdigest()}, "step3_fit": FIT3,
            "cutfull": bool(cutfull), "profile": profile,
            "run_transform_for_readers": {"R": "per model, mean of reply rows", "C": "per model, max of content-only command rows"},
            "schema": "per-row only (SEALED, 09 Append 38 / READ_LOG @ 96b6c4fb4): key, unit, call_index, score (designated model's row score; NaN on fold -1), "
                      "score_by_model (n x 5), fold. No run_score or run_score_by_model: run aggregates are left to readers after 09 verifies the pool keys"}
    out = score.with_options(volumes={"/out": OUTV, "/w3": W3, "/b": modal.Volume.from_name(msg["volume"], version=2).read_only()}).remote(msg, task_fold, head)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"ep2_{msg['d9_arm']}_{stamp}.json").open("x").write(json.dumps(head | {"result": out,
        "unmapped_cells": sorted(c for c, f in task_fold.items() if f < 0)}, indent=1))
    print(json.dumps({fo: {k: v for k, v in r.items() if k != "counts"} | {"counts": r["counts"]} for fo, r in out.items()}, indent=1))
