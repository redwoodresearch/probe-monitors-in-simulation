"""own_all_L48 (09's Append 40 + amendment 1 @ 01b2d13db; Aswin "ok own all 15"): own_frac100_L48's recipe (pass 1's pinned spec 5e2409a7…;
layer 48; R and C; the same rows, preprocessing and certified fit) trained on every universe main of T196, +311 and nm360 (results/
own_all_inputs_*.json: universe, folds, per-fold main lists, training manifest), 5 fold models by main, and every arm scored ONLY by its main's
fold model; mains outside the universe are routed fold -1 with all five models' logits (09 b6a6b6b48 §3). Three fractions (amendment 3 @ de33ce99e):
own_all_L48, own_all25_L48 and own_all10_L48 (run_famexp._train_tasks's seeded subsample of each fold's training mains).

  fit:   PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --mode fit --inputs <inputs record>
  score: ... --mode score --inputs <inputs record> --fit-stamp <stamp> --messages <list file>     (--dry: every local check, nothing remote)

Jobs: rows (CPU, one per volume: 2e's own loaders _b_rows / _b_rows_cmd, staged-sha and meta checks as pass 2), fit (H100: 30 models, each with
its training manifest; then the refit-match controls: own_frac100 / 25 / 10_L48 refit on T196 through this code must reproduce pass 1's
registered open-pack scores within TOL_REFIT, fixed here before any run), score (CPU, one per message / T196 / the A25 sim-honest runs:
artifact-bound gate per row with a misrouted negative, b6a6b6b48 §4). Every output is create-only, its manifest last. Keys, shas, counts and gate results only are returned or printed: never a score.

--loo (09 inclusion amendment 2 @ 8b9ca793f; POST HOC): own_all_LOO_L48, a fold-map override. Folds are task sets (origins): 0 = S_old (T196),
1 = S_new (+311), 2 = S_newmains (nm360). Model o trains on own_all's existing rows (/own_all/2026-10-15_173309/rows, sha-pinned; no new rows)
of the OTHER origins' run1 + honest_a, dropping by key every main of set o (count asserted 0), and scores set o only. Same recipe, gates and
records; outputs under own_all_loo trees, never own_all's. Its refit-match control refits own_all_L48 from its own manifest and must reproduce
its registered weights within TOL_REFIT.
"""

import ast
import collections
import hashlib
import io
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_ownall_kill-2026-11-14"
L, C_REG, TOL_REFIT = 48, 0.01, 1e-6  # layer 48; own_a's C (pass 1's C); refit-match tolerance on logits, fixed before any run (amendment 1 §4)
SPEC_STAMP, FIT_P1 = "2026-10-14_163027", "2026-10-14_175636"  # pass 1's spec and fit (the registered own_frac100_L48)
SPEC_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
VOL_T196, VOL_SU = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
BUILD = Path("/tmp/famexp_build")
VOL_Q3 = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"  # the A25 sim-honest dumps (run_owns.py's prep spec names /q3/<run>)
OWNS, OWNS_SHA = "2026-10-14_231856", "0521496e4a05be415ad4aacf18653eeb511021a2d0c448711169fdd328e57a8a"  # prep_owns record @ b8bccb29c .result.spec_sha256
DUMPS = ("231f31404", "experiments/2026-10-14_p2_own_sim_trained/dumps.json", "ff7b154ff5d7f82abe5e6ca8c3be8b1ba3d328fd2780e96c2a5053605a4c512f")
COUNTED_SHA = "80753f44aa66e78445f233e43e18c96fd01913246afcc03420b9322c447f4082"  # the 149 counted A25 sim-honest runs (simh_counted_keys_primary_v3)
LOO_VAR, ORIGINS = "own_all_LOO_L48", ("S_old", "S_new", "S_newmains")  # 09 inclusion amendment 2: fold o = origin index
ROWS_STAMP = "2026-10-15_173309"  # own_all_L48's fit (fit record 0b0fccdc… @ 66800141f): its row dumps and registered weights
ROWS_SHA = {("T196", "R"): "8b375c6174a12740165cd483a6cfa1be58c9cec6caed95387bdc82c48ba53fcb", ("T196", "C"): "26edc3300421ee915f08d9f0d8f3cff0488d5c66c596286c016df2e6aff478f7",
            ("scaleup", "R"): "ec50594b39d6f63e83df0918c9773e25c9c60921975248610fcf11bb1a1ebd24", ("scaleup", "C"): "09cbe4995372aa100ca682d53d8ac3c481756a3bf42c311f9001f63b9a56e308"}
OWN_ALL_FIT_SHA = "0b0fccdc8ef7cb8558b2aea8437a4ac2e2c303eba71752cddbc0609f7adf9843"


RS_VARS = {"own_all_rs_L48": 1.0, "own_all_rs25_L48": 0.25, "own_all_rs10_L48": 0.10}  # 09 inclusion amendment 3: own_all{,25,10}'s recipes on the re-split folds
RS_LABEL = ("own_all_rs_L48 (09 inclusion amendment 3 @ 4298ae33a + 3a @ eb18fe1c4; POST HOC): headline on-policy probe chosen by the author after own_all v1's values were seen; "
            "launch-disjoint sensitivity (LOO) beside (disclosure NOTE: experiments/2026-10-14_priority1_spec_09/READ_LOG.md @ 300b4008b, 0d's decision log); trained on other mains' run1 and honest_a across the included sets; "
            "trained on the three earlier sets; S_paper493 scored, not trained on; The v1 diagnostic (own_all v1, three older sets) did not detect the specified differential "
            "launch pattern; it does not establish negligible launch effects (e.g. own_all's S_old reply catch DiD at 1% spans about −0.0 to +10.7 points, @ c6f0b432d); "
            "re-split = the pinned fold list @ ec56720ad (sha256 c2fc5d40eb84a34a…); PRELIMINARY")
RS_FOLDS = "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json"
UNCUT_VARS = {"own_all_rs_uncut_L48": 1.0, "own_all_rs25_uncut_L48": 0.25, "own_all_rs10_uncut_L48": 0.10}  # 4f: the three fractions, own_all_rs{,25,10}'s rule
UNCUT_LABEL = ("own_all_rs_uncut_L48 (09 full-length lead RULE @ 36b9b47df; uncut §6 / §7): new probe trained on full transcripts (OOD class (i)); POST HOC; not the registered "
               "probe; T196 rows from the fixed-template staging (render differs from the registered rows on {n} epochs); trained on the three earlier sets; re-split folds "
               "@ ec56720ad; PRELIMINARY")
RS_FIT = ("2026-10-16_003049", "9ff00c432c84")  # own_all_rs phase A: fit stamp, Volume fit-record sha256 prefix (checked in full against the git fit record)


def _cfg(inputs, stamp):
    """(number of models, variants, fold of main, fit dir, output subdir): own_all's 5 folds by main, --loo's 3 origin folds, or --rs's re-split folds."""
    if inputs.get("rs_uncut"):
        return 5, UNCUT_VARS, {m: v[0] for m, v in inputs["rs"]["by_main"].items()}, Path("/out/own_all_rs_uncut") / stamp, "own_all_rs_uncut"
    if inputs.get("rs"):
        return 5, RS_VARS, {m: v[0] for m, v in inputs["rs"]["by_main"].items()}, Path("/out/own_all_rs") / stamp, "own_all_rs"
    if inputs.get("loo"):
        return 3, {LOO_VAR: 1.0}, inputs["loo"]["fold_of_main"], Path("/out/own_all_loo") / stamp, "own_all_loo"
    return 5, VARIANTS, {m: v[0] for m, v in inputs["folds"]["by_main"].items()}, Path("/out/own_all") / stamp, "own_all"
if modal.is_local():  # run_famexp's import materializes the pinned 2e row code and torch_probe into BUILD, refusing any byte difference
    sys.path.insert(0, str(HERE))
    import cutfull as cf  # famexp's 09 Append 41 amd 2 grammar @ 5c3484152, unchanged (its self-test runs on import)
    import run_famexp  # noqa: F401

app = modal.App(APP)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2)
WORK = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "safetensors==0.8.0").add_local_dir(BUILD, "/root/x2e"))
SIMH_BATCH = re.compile(r"^(simh311_q[1-4]|simh_nm360_(?:s1|s2|s3|s4d|s4r|s5)|sp493_g[0-9]+|simh_t196fill)_")  # sp493_g<k>: 09's S_paper493 batch (Q1); simh_t196fill: the A25 fill-in
SP493_FACTS = ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json", "07caa527db89ec51")  # sha256 prefix (4f's relay)
SP493_OUT = {"t987_model_registry": "no_usable_reference (refindex_L3 @ 627400b22); out of the S_paper493 attack cohort for every row (09 READ_LOG @ 65e738c7a / 4e173ffb9)"}
RS_PENDING = {"t619_model_registry": "pending re-stage: sp493_b_main_v1_try4 is superseded when the fallback re-stage routes (it recovers t619); b_main is rescored then (4f, 10-16)"}
RS_P_NEW = ("S_paper493's P_new mains are in no training set: each is scored by the model of its listed fold in the pinned fold list @ ec56720ad (out of sample for "
            "every fold model); P_rep mains (T196 / +311 mains) by their own held-out fold (09 amendment 3a §f)")


def _x2e():
    sys.path.insert(0, "/root/x2e")
    import run_correctedb as rc
    import torch_probe_pinned as tp

    return rc, rc.rat, tp


def _own(epochs, readout, skip):
    return {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": {e["key"]: [e["world"], e["cell"]] for e in epochs},
            "arm_of": {e["key"]: e["arm"] for e in epochs}, "run_of": {e["key"]: e["run"] for e in epochs}} | ({"skip": skip} if readout == "C" else {})


def _skip(epochs, spec):
    """Staged-input and extraction-meta checks (pass 2's score_layer, Codex round 11) and the C skip rows (fixed call tokens) for scale-up epochs."""
    rc, _, _ = _x2e()
    eps, cfgs, bad, fixed = {}, {}, [], set(spec["fixed_ids"])
    for e in epochs:
        if e["staged_sha256"] == "from_config":
            cfgs.setdefault(e["run"], json.loads((Path("/b") / e["run"] / "config.json").read_text())["inputs_sha256"])
            e["staged_sha256"] = cfgs[e["run"]][f"{e['arm']}.jsonl"]
    for path, h in sorted({(e["staged"], e["staged_sha256"]) for e in epochs}):
        raw = Path("/b", path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"staged inputs {path}: sha256 differs")
        eps |= {(path, x["key"]): x for x in (json.loads(y) for y in raw.decode().splitlines() if y)}
    for e in epochs:
        st = eps.get((e["staged"], e["key"]))
        mt = json.loads((Path("/b") / e["run"] / "meta" / e["arm"] / f"{rc.stem(e['key'])}.json").read_text()) if st else None
        if not st or mt["key"] != e["key"] or mt.get("inputs_sha256") != e["staged_sha256"] or \
                [(c["irid"], c["call_index"], c["kind"]) for c in st["calls"]] != [(c["irid"], c["call_index"], c["kind"]) for c in mt["calls"]]:
            bad.append(e["key"])
    if bad:
        raise SystemExit(f"{len(bad)} epochs missing from staging or inconsistent with the extraction meta, e.g. {bad[:2]}")
    return {e["key"]: [[r["irid"], r["call_index"]] for r in eps[(e["staged"], e["key"])]["calls"]
                       if r["kind"] == "call_end" and r["readout"] is not None and eps[(e["staged"], e["key"])]["ids"][r["readout"]] in fixed] for e in epochs}


def _load(epochs, readout, skip, cell_fold):
    """2e's rows at L48 for these epochs (pass 1 / pass 2's loaders): X float32, names '<world>|<cell>', call_index, unit, fold; empty if none."""
    import numpy as np

    rc, rat, _ = _x2e()
    own = _own(epochs, readout, skip)
    try:
        X, names, folds = (rat._b_rows if readout == "R" else rat._b_rows_cmd)(own, L, cell_fold)
    except ValueError as e:
        if "need at least one array to concatenate" not in str(e):
            raise
        return np.zeros((0, 0), np.float32), np.array([], str), np.array([], np.int64), np.array([], np.int64), np.array([], np.int64)
    ci, _ = rc._row_calls(own, cmd=(readout == "C"))
    return X, np.asarray(names), np.asarray(ci, np.int64), np.asarray(rc._units(names), np.int64), np.asarray(folds, np.int64)


def _npz(path, **arrs):
    import numpy as np

    buf = io.BytesIO()
    np.savez(buf, **arrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.open("xb").write(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _spec():
    p = Path("/out/famexp") / SPEC_STAMP / f"spec_{SPEC_STAMP}.json"
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != SPEC_SHA:
        raise SystemExit("pass 1's spec differs from its pin")
    return json.loads(b)


@app.function(image=img, cpu=(8, 8), memory=(65536, 65536), timeout=1800, retries=0, scaledown_window=2, volumes={"/out": OUT})
def rows(stamp: str, pool: str, epochs: list, cell_fold: dict) -> dict:
    """Training rows of one volume (T196 or scale-up), both readouts, written to /out/own_all/<stamp>/rows/<pool>__<R|C>.npz."""
    spec = _spec()
    skip = {k: spec["skip"].get(k, []) for k in (e["key"] for e in epochs)} if pool == "T196" else _skip(epochs, spec)
    if pool == "T196":  # T196 rows carry pass 1's r36 fold (spec fot) for every cell, one-world cells included (the refit-match control's folds)
        cell_fold = spec["fot"]
    res = {}
    for readout in ("R", "C"):
        X, names, ci, units, folds = _load(epochs, readout, skip, cell_fold)
        res[readout] = {"rows": int(len(names)), "runs": len(set(names.tolist())),
                        "sha256": _npz(Path("/out/own_all") / stamp / "rows" / f"{pool}__{readout}.npz", X=X, names=names, call_index=ci, unit=units, fold=folds)}
    OUT.commit()
    return res


def _fit_one(tp, X, y, tr, st, dev):
    """pass 1's run_famexp._fit_one @ b5a0586fa, verbatim: certified fit on rows tr, standardized by rows st (float64)."""
    import torch

    t0 = time.time()
    Xs = X[torch.as_tensor(st, device=dev)].double()
    mean, std = Xs.mean(0), Xs.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    del Xs
    Xt = (X[torch.as_tensor(tr, device=dev)].double() - mean) / std
    W, b, cert = tp.fit(Xt, torch.as_tensor(y[tr], device=dev, dtype=torch.float64), C_REG, allow_uncertified=True, return_certificate=True)
    torch.cuda.synchronize()
    return (W, b, mean, std), {k: cert[k] for k in ("n", "max_grad", "certified", "n_evals")}, time.time() - t0


VARIANTS = {"own_all_L48": 1.0, "own_all25_L48": 0.25, "own_all10_L48": 0.10}  # Append 40 + amendment 3 (@ de33ce99e): own_frac100 / 25 / 10's recipes
CONTROLS = {1.0: "own_frac100_L48", 0.25: "own_frac25_L48", 0.10: "own_frac10_L48"}  # the registered pass-1 fits each refit must reproduce


def _subsample(tasks, k, v):
    """run_famexp._train_tasks @ b5a0586fa, verbatim rule: the fold's training tasks, sorted, seeded order, first ceil(v x n)."""
    import math
    import random

    t = sorted(tasks)
    random.Random(f"famexp|own_a|f{k}").shuffle(t)
    return set(t[: math.ceil(v * len(t))])


def _prep_sha(mean, std):
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(mean, np.float64).tobytes() + np.ascontiguousarray(std, np.float64).tobytes()).hexdigest()


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(65536, 65536), timeout=2700, retries=0, scaledown_window=2, volumes={"/out": OUT})
def fit(stamp: str, inputs: dict) -> dict:
    """own_all{,25,10}_L48: 30 models (3 fractions x R, C x 5 folds) on universe mains with fold != k (fractions: the seeded subsample of the fold's
    training mains), each with its training manifest; in-job STOP; then the refit-match controls on T196 (own_frac100 / 25 / 10_L48)."""
    import numpy as np
    import torch
    from safetensors.numpy import load_file, save_file

    OUT.reload()
    _, _, tp = _x2e()
    dev, od = "cuda", Path("/out/own_all") / stamp
    fold, main_of = {m: v[0] for m, v in inputs["folds"]["by_main"].items()}, inputs["main_of"]
    want = {int(k): v for k, v in inputs["folds"]["training_mains_sha256"].items()}
    sha_l = lambda xs: hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()  # noqa: E731
    rec, ctrl = {"fits": [], "gate": {}}, {}
    for readout in ("R", "C"):
        parts = [np.load(od / "rows" / f"{p}__{readout}.npz") for p in ("T196", "scaleup")]
        names = np.concatenate([p["names"] for p in parts])
        world = np.array([u.split("|", 1)[0] for u in names])
        cells = np.array([u.split("|", 1)[1] for u in names])
        mains = np.array([main_of[c] for c in cells])
        uni = np.array([m in fold for m in mains])
        rf = np.array([fold.get(m, -1) for m in mains])
        y = np.where(world == "run1", 1.0, np.where(world == "honest_a", 0.0, np.nan))
        X = torch.from_numpy(np.concatenate([p["X"] for p in parts])).to(dev)
        for k in range(5):
            full = uni & (rf != k) & ~np.isnan(y)
            if sha_l(set(mains[full].tolist())) != want[k]:  # in-job STOP: own_all's full training mains = the committed list for fold k
                raise SystemExit(f"STOP: fold model {k} ({readout}) training mains differ from the committed list")
            for var, v in VARIANTS.items():
                tt = _subsample(set(cells[full].tolist()), k, v)
                tr = np.flatnonzero(full & np.isin(cells, sorted(tt)))
                tm = sorted(set(mains[tr].tolist()))
                if {m for m in tm if fold[m] == k} or sha_l(tm) != sha_l(set(tm) & set(mains[full].tolist())):
                    raise SystemExit(f"STOP: {var} f{k} ({readout}) trains on a fold-{k} main or outside the fold's training pool")
                mp = od / "training" / f"{var}__{readout}__f{k}.json"
                mp.parent.mkdir(parents=True, exist_ok=True)
                mp.open("x").write(json.dumps(tm))
                m_, cert, secs = _fit_one(tp, X, y, tr, tr, dev)
                wp = od / "weights" / f"{var}__{readout}__L48__f{k}.safetensors"
                wp.parent.mkdir(parents=True, exist_ok=True)
                arr = {n: np.ascontiguousarray(t.cpu().numpy()) for n, t in zip(("W", "b", "mean", "std"), m_, strict=True)}
                save_file(arr, str(wp))
                rec["fits"].append({"variant": var, "fraction": v, "readout": readout, "layer": L, "fold": k, "rows": int(len(tr)), "runs": len(set(names[tr].tolist())),
                                    "mains": len(tm), "training_manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(), "seconds": secs, **cert,
                                    "weights_sha256": hashlib.sha256(wp.read_bytes()).hexdigest(), "preproc_sha256": _prep_sha(arr["mean"], arr["std"])})
        # refit-match controls (amendment 1 §4, amendment 3 §2): pass 1's own_frac rule on T196's rows (every task with run1 / honest_a rows, r36 folds)
        t = parts[0]
        Xt_, nt = torch.from_numpy(t["X"]).to(dev), t["names"]
        wt = np.array([u.split("|", 1)[0] for u in nt])
        tk = np.array([u.split("|", 1)[1] for u in nt])
        yt = np.where(wt == "run1", 1.0, np.where(wt == "honest_a", 0.0, np.nan))
        ft = t["fold"]
        for v, cname in CONTROLS.items():
            for f in range(5):
                tt = _subsample({tk[i] for i in np.flatnonzero((ft != f) & ~np.isnan(yt))}, f, v)
                tr = np.flatnonzero(~np.isnan(yt) & (ft != f) & np.isin(tk, sorted(tt)))
                m_, cert, _ = _fit_one(tp, Xt_, yt, tr, tr, dev)
                sc = np.flatnonzero(ft == f)
                z = tp.decision((Xt_[torch.as_tensor(sc, device=dev)].double() - m_[2]) / m_[3], m_[0], m_[1]).cpu().numpy()[:, 0]
                reg = np.load(Path("/out/famexp") / SPEC_STAMP / f"run_{FIT_P1}" / "open" / f"{cname}__{readout}__L48__f{f}.npz")
                ref = {(k_, int(u), int(c)): float(s_) for k_, u, c, s_ in zip(reg["key"], reg["unit"], reg["call_index"], reg["score"], strict=True)}
                got = [(str(nt[i]), int(t["unit"][i]), int(t["call_index"][i])) for i in sc]
                miss = [g for g in got if g not in ref]
                d = [abs(ref[g] - float(x)) for g, x in zip(got, z, strict=True) if g in ref]
                w1 = load_file(str(Path("/out/famexp") / SPEC_STAMP / f"run_{FIT_P1}" / "weights" / f"{cname}__{readout}__L48__f{f}.safetensors"))
                ctrl[f"{cname}|{readout}|f{f}"] = {"rows_compared": len(d), "rows_missing_from_registered": len(miss), "max_abs_diff": max(d) if d else None,
                                                  "weights_max_abs_diff": float(max(np.abs(np.asarray(w1[n], np.float64) - t_.cpu().numpy().astype(np.float64)).max()
                                                                                   for n, t_ in zip(("W", "b", "mean", "std"), m_, strict=True))),
                                                  "certified": bool(cert["certified"]), "pass": bool(d) and not miss and max(d) <= TOL_REFIT}
    rec["gate"] = {"in_job_stop": "passed (full training mains = committed list per fold; every fraction inside its fold's pool; no fold-k main)",
                   "refit_match": ctrl, "tol_refit": TOL_REFIT, "refit_match_pass": all(v["pass"] for v in ctrl.values()),
                   "all_certified": all(bool(f["certified"]) for f in rec["fits"])}
    rec["inputs_sha256"] = inputs["_sha256"]
    p = od / "fit_record.json"
    p.open("x").write(json.dumps(rec, indent=1, default=float))
    OUT.commit()
    return rec | {"fit_record_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(65536, 65536), timeout=2700, retries=0, scaledown_window=2, volumes={"/out": OUT})
def fit_loo(stamp: str, inputs: dict) -> dict:
    """--loo: own_all_LOO_L48 (R, C x 3 origin models) on own_all's sha-pinned rows; in-job STOP (training mains = the committed list per
    origin; 0 mains of set o in model o's training); then the refit-match control: own_all_L48's 10 fits refit from its own fold lists through
    this code must reproduce its registered weights within TOL_REFIT."""
    import numpy as np
    import torch
    from safetensors.numpy import load_file, save_file

    OUT.reload()
    _, _, tp = _x2e()
    dev, od, rd_ = "cuda", Path("/out/own_all_loo") / stamp, Path("/out/own_all") / ROWS_STAMP
    if hashlib.sha256((rd_ / "fit_record.json").read_bytes()).hexdigest() != OWN_ALL_FIT_SHA:
        raise SystemExit("own_all_L48's fit record differs from its pin")
    lo = inputs["loo"]
    ofold, main_of, k5 = lo["fold_of_main"], inputs["main_of"], {m: v[0] for m, v in inputs["folds"]["by_main"].items()}
    set_mains = {int(o): set(v) for o, v in lo["set_mains"].items()}
    want_o = {int(o): v for o, v in lo["training_mains_sha256"].items()}
    want_k = {int(k): v for k, v in inputs["folds"]["training_mains_sha256"].items()}
    sha_l = lambda xs: hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()  # noqa: E731
    reg = {(f["readout"], f["fold"]): f for f in json.loads((rd_ / "fit_record.json").read_text())["fits"] if f["variant"] == "own_all_L48"}
    rec, ctrl, dropped = {"fits": [], "gate": {}}, {}, {}
    for readout in ("R", "C"):
        parts = []
        for p in ("T196", "scaleup"):
            b = (rd_ / "rows" / f"{p}__{readout}.npz").read_bytes()
            if hashlib.sha256(b).hexdigest() != ROWS_SHA[(p, readout)]:
                raise SystemExit(f"own_all's rows {p}__{readout}: sha256 differs from the pin")
            parts.append(np.load(io.BytesIO(b)))
        names = np.concatenate([p["names"] for p in parts])
        world = np.array([u.split("|", 1)[0] for u in names])
        mains = np.array([main_of[u.split("|", 1)[1]] for u in names])
        uni = np.array([m in k5 for m in mains])
        y = np.where(world == "run1", 1.0, np.where(world == "honest_a", 0.0, np.nan))
        X = torch.from_numpy(np.concatenate([p["X"] for p in parts])).to(dev)
        if not all(m in ofold for m in mains[uni].tolist()):
            raise SystemExit("STOP: a training main has no origin")
        of = np.array([ofold.get(m, -1) for m in mains])
        for o in range(3):
            cand = uni & (of != o) & ~np.isnan(y)
            drop = cand & np.isin(mains, sorted(set_mains[o]))  # amendment 2 item 3: every main of set o is dropped by key, whatever set its row comes from
            dropped[f"{readout}|o{o}"] = len(set(mains[drop].tolist()))
            if dropped[f"{readout}|o{o}"]:
                raise SystemExit(f"STOP: LOO-3 model o{o} ({readout}): {dropped[f'{readout}|o{o}']} training mains also appear in set o (asserted 0)")
            tr = np.flatnonzero(cand & ~drop)
            tm = sorted(set(mains[tr].tolist()))
            if sha_l(tm) != want_o[o] or set(tm) & set_mains[o]:  # in-job STOP: training mains = the committed list for origin o
                raise SystemExit(f"STOP: LOO model o{o} ({readout}) training mains differ from the committed list")
            mp = od / "training" / f"{LOO_VAR}__{readout}__f{o}.json"
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.open("x").write(json.dumps(tm))
            m_, cert, secs = _fit_one(tp, X, y, tr, tr, dev)
            wp = od / "weights" / f"{LOO_VAR}__{readout}__L48__f{o}.safetensors"
            wp.parent.mkdir(parents=True, exist_ok=True)
            arr = {n: np.ascontiguousarray(t.cpu().numpy()) for n, t in zip(("W", "b", "mean", "std"), m_, strict=True)}
            save_file(arr, str(wp))
            rec["fits"].append({"variant": LOO_VAR, "fraction": 1.0, "readout": readout, "layer": L, "fold": o, "origin_held_out": ORIGINS[o], "rows": int(len(tr)),
                                "runs": len(set(names[tr].tolist())), "mains": len(tm), "training_manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
                                "seconds": secs, **cert, "weights_sha256": hashlib.sha256(wp.read_bytes()).hexdigest(), "preproc_sha256": _prep_sha(arr["mean"], arr["std"])})
        rf = np.array([k5.get(m, -1) for m in mains])
        for k in range(5):  # refit-match control (amendment 2 item 4): own_all_L48 from its own fold lists, vs its registered weights
            full = uni & (rf != k) & ~np.isnan(y)
            if sha_l(set(mains[full].tolist())) != want_k[k]:
                raise SystemExit(f"STOP: refit-match control f{k} ({readout}): training mains differ from own_all's committed list")
            m_, cert, _ = _fit_one(tp, X, y, np.flatnonzero(full), np.flatnonzero(full), dev)
            w1 = load_file(str(rd_ / "weights" / f"own_all_L48__{readout}__L48__f{k}.safetensors"))
            if hashlib.sha256((rd_ / "weights" / f"own_all_L48__{readout}__L48__f{k}.safetensors").read_bytes()).hexdigest() != reg[(readout, k)]["weights_sha256"]:
                raise SystemExit(f"own_all_L48 {readout} f{k}: registered weights differ from its fit record")
            dmax = float(max(np.abs(np.asarray(w1[n], np.float64) - t_.cpu().numpy().astype(np.float64)).max() for n, t_ in zip(("W", "b", "mean", "std"), m_, strict=True)))
            ctrl[f"own_all_L48|{readout}|f{k}"] = {"rows": int(full.sum()), "registered_rows": reg[(readout, k)]["rows"], "weights_max_abs_diff": dmax,
                                                   "certified": bool(cert["certified"]), "pass": dmax <= TOL_REFIT and int(full.sum()) == reg[(readout, k)]["rows"]}
    rec["gate"] = {"in_job_stop": "passed (training mains = committed list per origin; 0 mains of set o in model o's training)", "dropped_mains_set_o": dropped,
                   "refit_match": ctrl, "tol_refit": TOL_REFIT, "refit_match_pass": all(v["pass"] for v in ctrl.values()),
                   "all_certified": all(bool(f["certified"]) for f in rec["fits"])}
    rec["inputs_sha256"] = inputs["_sha256"]
    rec["loo_sha256"] = lo["_sha256"]
    rec["rows_from"] = {"fit_stamp": ROWS_STAMP, "fit_record_sha256": OWN_ALL_FIT_SHA, "rows_sha256": {f"{p}__{r}": h for (p, r), h in ROWS_SHA.items()}}
    p = od / "fit_record.json"
    p.open("x").write(json.dumps(rec, indent=1, default=float))
    OUT.commit()
    return rec | {"fit_record_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(65536, 65536), timeout=2700, retries=0, scaledown_window=2, volumes={"/out": OUT})
def fit_rs(stamp: str, inputs: dict) -> dict:
    """--rs: own_all_rs{,25,10}_L48 (R, C x 5 re-split folds) on own_all's sha-pinned rows; in-job STOP (training mains = the committed list per
    fold; no fold-k main; each fraction inside its fold's pool); then the refit-match control: own_all v1's 30 fits refit on v1's folds through
    this code must reproduce v1's registered weights within TOL_REFIT."""
    import numpy as np
    import torch
    from safetensors.numpy import load_file, save_file

    OUT.reload()
    _, _, tp = _x2e()
    dev, od, rd_ = "cuda", Path("/out/own_all_rs") / stamp, Path("/out/own_all") / ROWS_STAMP
    if hashlib.sha256((rd_ / "fit_record.json").read_bytes()).hexdigest() != OWN_ALL_FIT_SHA:
        raise SystemExit("own_all_L48's fit record differs from its pin")
    rs, main_of = inputs["rs"], inputs["main_of"]
    fold, k1 = {m: v[0] for m, v in rs["by_main"].items()}, {m: v[0] for m, v in inputs["folds"]["by_main"].items()}
    want = {int(k): v for k, v in rs["training_mains_sha256"].items()}
    want1 = {int(k): v for k, v in inputs["folds"]["training_mains_sha256"].items()}
    sha_l = lambda xs: hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()  # noqa: E731
    reg = {(f["variant"], f["readout"], f["fold"]): f for f in json.loads((rd_ / "fit_record.json").read_text())["fits"]}
    rec, ctrl = {"fits": [], "gate": {}}, {}
    for readout in ("R", "C"):
        parts = []
        for p in ("T196", "scaleup"):
            b = (rd_ / "rows" / f"{p}__{readout}.npz").read_bytes()
            if hashlib.sha256(b).hexdigest() != ROWS_SHA[(p, readout)]:
                raise SystemExit(f"own_all's rows {p}__{readout}: sha256 differs from the pin")
            parts.append(np.load(io.BytesIO(b)))
        names = np.concatenate([p["names"] for p in parts])
        world = np.array([u.split("|", 1)[0] for u in names])
        cells = np.array([u.split("|", 1)[1] for u in names])
        mains = np.array([main_of[c] for c in cells])
        if not all(m in fold for m in set(mains.tolist())):
            raise SystemExit("STOP: a row's main has no re-split fold")
        uni = np.array([m in k1 for m in mains])  # own_all v1's training universe (mains with run1 AND honest_a); the dumps also hold one-arm mains' rows, never trained (as v1)
        rf, rf1 = np.array([fold[m] for m in mains]), np.array([k1.get(m, -1) for m in mains])
        y = np.where(world == "run1", 1.0, np.where(world == "honest_a", 0.0, np.nan))
        X = torch.from_numpy(np.concatenate([p["X"] for p in parts])).to(dev)
        for k in range(5):
            full = uni & (rf != k) & ~np.isnan(y)
            if sha_l(set(mains[full].tolist())) != want[k]:  # in-job STOP: training mains = the committed re-split list for fold k
                raise SystemExit(f"STOP: re-split fold model {k} ({readout}) training mains differ from the committed list")
            for var, v in RS_VARS.items():
                tt = _subsample(set(cells[full].tolist()), k, v)
                tr = np.flatnonzero(full & np.isin(cells, sorted(tt)))
                tm = sorted(set(mains[tr].tolist()))
                if {m for m in tm if fold[m] == k} or not set(tm) <= set(mains[full].tolist()):
                    raise SystemExit(f"STOP: {var} f{k} ({readout}) trains on a fold-{k} main or outside the fold's training pool")
                mp = od / "training" / f"{var}__{readout}__f{k}.json"
                mp.parent.mkdir(parents=True, exist_ok=True)
                mp.open("x").write(json.dumps(tm))
                m_, cert, secs = _fit_one(tp, X, y, tr, tr, dev)
                wp = od / "weights" / f"{var}__{readout}__L48__f{k}.safetensors"
                wp.parent.mkdir(parents=True, exist_ok=True)
                arr = {n: np.ascontiguousarray(t.cpu().numpy()) for n, t in zip(("W", "b", "mean", "std"), m_, strict=True)}
                save_file(arr, str(wp))
                rec["fits"].append({"variant": var, "fraction": v, "readout": readout, "layer": L, "fold": k, "rows": int(len(tr)), "runs": len(set(names[tr].tolist())),
                                    "mains": len(tm), "training_manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(), "seconds": secs, **cert,
                                    "weights_sha256": hashlib.sha256(wp.read_bytes()).hexdigest(), "preproc_sha256": _prep_sha(arr["mean"], arr["std"])})
        for k in range(5):  # refit-match control: own_all v1's fits on v1's folds, vs v1's registered weights
            full = uni & (rf1 != k) & ~np.isnan(y)
            if sha_l(set(mains[full].tolist())) != want1[k]:
                raise SystemExit(f"STOP: refit-match control f{k} ({readout}): training mains differ from own_all v1's committed list")
            for var, v in VARIANTS.items():
                tr = np.flatnonzero(full & np.isin(cells, sorted(_subsample(set(cells[full].tolist()), k, v))))
                m_, cert, _ = _fit_one(tp, X, y, tr, tr, dev)
                wpath = rd_ / "weights" / f"{var}__{readout}__L48__f{k}.safetensors"
                if hashlib.sha256(wpath.read_bytes()).hexdigest() != reg[(var, readout, k)]["weights_sha256"]:
                    raise SystemExit(f"{var} {readout} f{k}: v1's registered weights differ from its fit record")
                w1 = load_file(str(wpath))
                dmax = float(max(np.abs(np.asarray(w1[n], np.float64) - t_.cpu().numpy().astype(np.float64)).max() for n, t_ in zip(("W", "b", "mean", "std"), m_, strict=True)))
                ctrl[f"{var}|{readout}|f{k}"] = {"rows": int(len(tr)), "registered_rows": reg[(var, readout, k)]["rows"], "weights_max_abs_diff": dmax,
                                                 "certified": bool(cert["certified"]), "pass": dmax <= TOL_REFIT and int(len(tr)) == reg[(var, readout, k)]["rows"]}
    rec["gate"] = {"in_job_stop": "passed (training mains = committed re-split list per fold; every fraction inside its fold's pool; no fold-k main)",
                   "refit_match": ctrl, "tol_refit": TOL_REFIT, "refit_match_pass": all(v["pass"] for v in ctrl.values()),
                   "all_certified": all(bool(f["certified"]) for f in rec["fits"])}
    rec["inputs_sha256"], rec["rs_sha256"] = inputs["_sha256"], rs["_sha256"]
    rec["rows_from"] = {"fit_stamp": ROWS_STAMP, "fit_record_sha256": OWN_ALL_FIT_SHA, "rows_sha256": {f"{p}__{r}": h for (p, r), h in ROWS_SHA.items()}}
    p = od / "fit_record.json"
    p.open("x").write(json.dumps(rec, indent=1, default=float))
    OUT.commit()
    return rec | {"fit_record_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


@app.function(image=img, cpu=(8, 8), memory=(65536, 65536), timeout=3000, retries=0, scaledown_window=2, volumes={"/out": OUT})
def rows_uncut(stamp: str, name: str, epochs: list, cell_fold: dict) -> dict:
    """own_all_rs_uncut's training rows from one uncut root (the cutfull Volume at /b): 2e's loaders, the staged / meta checks (_skip), full
    consumption and per-row token positions (_cut_tokens), written to /out/own_all_rs_uncut/<stamp>/rows/<name>__<R|C>.npz (create-only)."""
    import numpy as np

    spec = _spec()
    rc = _x2e()[0]
    skip = _skip(epochs, spec)
    res = {}
    for readout in ("R", "C"):
        X, names, ci, units, folds = _load(epochs, readout, skip, cell_fold)
        if folds.tolist() != [cell_fold[u.split("|", 1)[1]] for u in names.tolist()] or (folds < 0).any():
            raise SystemExit(f"{name} {readout}: the loader's folds differ from the re-split fold of each row's main, or a row has no fold")
        tk, n_tok = _cut_tokens(epochs, readout, skip, rc)
        if [t[0] for t in tk] != names.tolist() or [t[2] for t in tk] != ci.tolist():
            raise SystemExit(f"{name} {readout}: token positions do not align with the rows")
        tok = np.array([t[3] for t in tk], np.int64)
        rk = np.array([t[1] for t in tk])
        res[readout] = {"rows": int(len(names)), "runs": len(set(names.tolist())), "rows_past_32768": int((tok >= 32768).sum()),
                        "runs_with_rows_past_32768": len(set(rk[tok >= 32768].tolist())),
                        "sha256": _npz(Path("/out/own_all_rs_uncut") / stamp / "rows" / f"{name}__{readout}.npz", X=X, names=names, call_index=ci, unit=units, token=tok, run_key=rk)}
    OUT.commit()
    return res


def _uncut_ctx(stamp, inputs, readout):
    """Shared by gate_uncut / fit_uncut: paths, own_all_rs's pinned fit record (registered fits of this readout), folds and the row-array builder."""
    import numpy as np

    od, rd_, rs_d = Path("/out/own_all_rs_uncut") / stamp, Path("/out/own_all") / ROWS_STAMP, Path("/out/own_all_rs") / RS_FIT[0]
    rsb = (rs_d / "fit_record.json").read_bytes()
    if hashlib.sha256(rsb).hexdigest() != inputs["rs_uncut"]["rs_fit_record_sha256"]:
        raise SystemExit("own_all_rs's fit record differs from its pin")
    rsreg = {(f["variant"], f["fold"]): f for f in json.loads(rsb)["fits"] if f["readout"] == readout}
    main_of, fold = inputs["main_of"], {m: v[0] for m, v in inputs["rs"]["by_main"].items()}
    k1 = {m: v[0] for m, v in inputs["folds"]["by_main"].items()}

    def _arrays(parts):
        names = np.concatenate([p["names"] for p in parts])
        cells = np.array([u.split("|", 1)[1] for u in names])
        mains = np.array([main_of[c] for c in cells])
        world = np.array([u.split("|", 1)[0] for u in names])
        y = np.where(world == "run1", 1.0, np.where(world == "honest_a", 0.0, np.nan))
        return names, cells, mains, y, np.array([m in k1 for m in mains]), np.array([fold[m] for m in mains])

    return od, rd_, rs_d, rsreg, fold, _arrays


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(98304, 98304), timeout=3600, retries=0, scaledown_window=2, volumes={"/out": OUT})
def gate_uncut(stamp: str, inputs: dict, readout: str) -> dict:
    """(1) the GATE for ONE readout (09 uncut §7 / lead RULE §1): a refit-match of the new code on own_all_rs's REGISTERED rows (own_all's sha-pinned
    dumps) vs its registered weights, all three fractions x 5 folds, TOL_REFIT. Writes gate/<readout>.json (create-only) and saves no weights; fit_uncut
    saves nothing unless BOTH readouts' gate files pass (Codex r15: a barrier before any uncut weights are written)."""
    import numpy as np
    import torch
    from safetensors.numpy import load_file

    OUT.reload()
    _, _, tp = _x2e()
    dev = "cuda"
    od, rd_, rs_d, rsreg, _, _arrays = _uncut_ctx(stamp, inputs, readout)
    reg = []
    for p in ("T196", "scaleup"):
        b = (rd_ / "rows" / f"{p}__{readout}.npz").read_bytes()
        if hashlib.sha256(b).hexdigest() != ROWS_SHA[(p, readout)]:
            raise SystemExit(f"own_all's rows {p}__{readout}: sha256 differs from the pin")
        reg.append(np.load(io.BytesIO(b)))
    names, cells, mains, y, uni, rf = _arrays(reg)
    X = torch.from_numpy(np.concatenate([p["X"] for p in reg])).to(dev)
    ctrl = {}
    for k in range(5):
        full = uni & (rf != k) & ~np.isnan(y)
        for var, v in RS_VARS.items():
            tr = np.flatnonzero(full & np.isin(cells, sorted(_subsample(set(cells[full].tolist()), k, v))))
            m_, cert, _ = _fit_one(tp, X, y, tr, tr, dev)
            wpath = rs_d / "weights" / f"{var}__{readout}__L48__f{k}.safetensors"
            if hashlib.sha256(wpath.read_bytes()).hexdigest() != rsreg[(var, k)]["weights_sha256"]:
                raise SystemExit(f"{var} {readout} f{k}: registered own_all_rs weights differ from its fit record")
            w1 = load_file(str(wpath))
            dmax = float(max(np.abs(np.asarray(w1[n], np.float64) - t_.cpu().numpy().astype(np.float64)).max() for n, t_ in zip(("W", "b", "mean", "std"), m_, strict=True)))
            ctrl[f"{var}|{readout}|f{k}"] = {"rows": int(len(tr)), "registered_rows": rsreg[(var, k)]["rows"], "weights_max_abs_diff": dmax, "pass": dmax <= TOL_REFIT and int(len(tr)) == rsreg[(var, k)]["rows"]}
    g = {"readout": readout, "refit_match_registered": ctrl, "pass": all(c["pass"] for c in ctrl.values()), "tol_refit": TOL_REFIT, "uncut_sha256": inputs["rs_uncut"]["_sha256"]}
    gp = od / "gate" / f"{readout}.json"
    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.open("x").write(json.dumps(g, indent=1, default=float))
    OUT.commit()
    if not g["pass"]:
        raise SystemExit(f"STOP: refit-match on the REGISTERED rows failed: {json.dumps({k: c for k, c in ctrl.items() if not c['pass']})}")
    return g | {"gate_sha256": hashlib.sha256(gp.read_bytes()).hexdigest()}


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(98304, 98304), timeout=5400, retries=0, scaledown_window=2, volumes={"/out": OUT})
def fit_uncut(stamp: str, inputs: dict, readout: str, gate_shas: dict) -> dict:
    """own_all_rs{,25,10}_uncut_L48 for ONE readout on the pinned re-split folds (09 uncut §6 / §7), only after BOTH readouts' gate files (gate_uncut)
    pass with the shas the entrypoint got back: (2) the in-job STOP (training mains = the committed re-split lists); (3) the fits on the uncut rows;
    (4) DESCRIPTIVE only: the 100% fit on the uncut rows with token < 32768, its weight difference from the registered own_all_rs_L48 (never a gate)."""
    import numpy as np
    import torch
    from safetensors.numpy import load_file, save_file

    OUT.reload()
    _, _, tp = _x2e()
    dev = "cuda"
    od, _, rs_d, rsreg, fold, _arrays = _uncut_ctx(stamp, inputs, readout)
    ctrl = {}
    for rd in ("R", "C"):  # the barrier: both readouts' gates, read back from the Volume by the shas gate_uncut returned
        gb = (od / "gate" / f"{rd}.json").read_bytes()
        g = json.loads(gb)
        if hashlib.sha256(gb).hexdigest() != gate_shas[rd] or g["readout"] != rd or g["uncut_sha256"] != inputs["rs_uncut"]["_sha256"] or not g["pass"] or \
                not all(c["pass"] for c in g["refit_match_registered"].values()) or len(g["refit_match_registered"]) != 15:
            raise SystemExit(f"STOP: the {rd} gate file is missing, differs from its sha, or did not pass; no uncut weights are written")
        ctrl |= g["refit_match_registered"]
    want = {int(k): v for k, v in inputs["rs"]["training_mains_sha256"].items()}
    sha_l = lambda xs: hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()  # noqa: E731
    reg_w = {k: load_file(str(rs_d / "weights" / f"own_all_rs_L48__{readout}__L48__f{k}.safetensors")) for k in range(5)}
    for k in range(5):
        if hashlib.sha256((rs_d / "weights" / f"own_all_rs_L48__{readout}__L48__f{k}.safetensors").read_bytes()).hexdigest() != rsreg[("own_all_rs_L48", k)]["weights_sha256"]:
            raise SystemExit(f"own_all_rs_L48 {readout} f{k}: registered weights differ from its fit record")
    # (2)-(4) the uncut rows
    parts, tokp, rkp = [], [], []
    for nm_, h in inputs["rs_uncut"]["rows"][readout].items():
        b = (od / "rows" / f"{nm_}__{readout}.npz").read_bytes()
        if hashlib.sha256(b).hexdigest() != h:
            raise SystemExit(f"uncut rows {nm_}__{readout}: sha256 differs from its rows job")
        z = np.load(io.BytesIO(b))
        parts.append(z)
        tokp.append(z["token"])
        rkp.append(z["run_key"])
    names, cells, mains, y, uni, rf = _arrays(parts)
    tok, rk = np.concatenate(tokp), np.concatenate(rkp)
    trn = uni & ~np.isnan(y)  # the training units (run1 + honest_a rows of training mains), any fold
    past = {"cut": "none", "training_rows": int(trn.sum()), "training_rows_past_32768": int((trn & (tok >= 32768)).sum()),
            "training_runs": len(set(rk[trn].tolist())), "training_runs_with_rows_past_32768": len(set(rk[trn & (tok >= 32768)].tolist()))}
    X = torch.from_numpy(np.concatenate([p["X"] for p in parts])).to(dev)
    fits, desc = [], {}
    for k in range(5):
        full = uni & (rf != k) & ~np.isnan(y)
        if sha_l(set(mains[full].tolist())) != want[k]:  # in-job STOP: the uncut training mains = the committed re-split list for fold k
            raise SystemExit(f"STOP: uncut fold model {k} ({readout}) training mains differ from the committed list")
        for var, v in UNCUT_VARS.items():
            tt = _subsample(set(cells[full].tolist()), k, v)
            tr = np.flatnonzero(full & np.isin(cells, sorted(tt)))
            tm = sorted(set(mains[tr].tolist()))
            if {m for m in tm if fold[m] == k} or not set(tm) <= set(mains[full].tolist()):
                raise SystemExit(f"STOP: {var} f{k} ({readout}) trains on a fold-{k} main or outside the fold's training pool")
            mp = od / "training" / f"{var}__{readout}__f{k}.json"
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.open("x").write(json.dumps(tm))
            m_, cert, secs = _fit_one(tp, X, y, tr, tr, dev)
            wp = od / "weights" / f"{var}__{readout}__L48__f{k}.safetensors"
            wp.parent.mkdir(parents=True, exist_ok=True)
            arr = {n: np.ascontiguousarray(t.cpu().numpy()) for n, t in zip(("W", "b", "mean", "std"), m_, strict=True)}
            save_file(arr, str(wp))
            fits.append({"variant": var, "fraction": v, "readout": readout, "layer": L, "fold": k, "rows": int(len(tr)), "rows_past_32768": int((tok[tr] >= 32768).sum()),
                         "runs": len(set(names[tr].tolist())), "mains": len(tm), "training_manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(), "seconds": secs, **cert,
                         "weights_sha256": hashlib.sha256(wp.read_bytes()).hexdigest(), "preproc_sha256": _prep_sha(arr["mean"], arr["std"])})
        trc = np.flatnonzero(full & (tok < 32768))  # (4) DESCRIPTIVE: the uncut pipeline cut at 32768, vs the registered weights
        m_, _, _ = _fit_one(tp, X, y, trc, trc, dev)
        desc[f"own_all_rs_L48|{readout}|f{k}"] = {"rows": int(len(trc)), "registered_rows": rsreg[("own_all_rs_L48", k)]["rows"],
                                                  "weights_max_abs_diff_vs_registered": float(max(np.abs(np.asarray(reg_w[k][n], np.float64) - t_.cpu().numpy().astype(np.float64)).max()
                                                                                                  for n, t_ in zip(("W", "b", "mean", "std"), m_, strict=True)))}
    OUT.commit()
    return {"fits": fits, "refit_match_registered": ctrl, "descriptive_cut32768": desc, "training_units_past_32768": {readout: past}}


@app.function(image=img, cpu=(1, 1), memory=(2048, 2048), timeout=600, retries=0, scaledown_window=2, volumes={"/out": OUT})
def finalize_uncut(stamp: str, inputs: dict, parts: list, gate_shas: dict, notes: dict) -> dict:
    """The uncut fit record (score() reads it): both readouts' fits, the gate (with both gate files' shas), the descriptive refit, cut = none and the
    training units past token 32768 per readout (4f for 2e's OOD table), and the launch-time by-key checks (notes); create-only."""
    OUT.reload()
    fits = [f for p in parts for f in p["fits"]]
    ctrl = {k: v for p in parts for k, v in p["refit_match_registered"].items()}
    rec = {"fits": fits, "gate": {"in_job_stop": "passed (uncut training mains = committed re-split lists per fold; every fraction inside its fold's pool; no fold-k main)",
                                  "refit_match": ctrl, "refit_match_on": "own_all_rs's REGISTERED rows (09 uncut §7 / lead RULE §1)", "tol_refit": TOL_REFIT,
                                  "refit_match_pass": all(v["pass"] for v in ctrl.values()), "all_certified": all(bool(f["certified"]) for f in fits)},
           "descriptive_cut32768": {k: v for p in parts for k, v in p["descriptive_cut32768"].items()}, "inputs_sha256": inputs["_sha256"], "rs_sha256": inputs["rs"]["_sha256"],
           "uncut_sha256": inputs["rs_uncut"]["_sha256"], "label": inputs["rs_uncut"]["label"], "gate_files_sha256": gate_shas,
           "cut": "none", "training_units_past_32768": {k: v for p in parts for k, v in p["training_units_past_32768"].items()}, "checks": notes}
    p = Path("/out/own_all_rs_uncut") / stamp / "fit_record.json"
    p.open("x").write(json.dumps(rec, indent=1, default=float))
    OUT.commit()
    return rec | {"fit_record_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


LOO_LABEL = "own_all_LOO_L48 scores (09 inclusion amendment 2 @ 8b9ca793f): new probe, trained on the other task sets; POST HOC; never replaces or pools with own_all_L48's rows; PRELIMINARY"


def _tgt(kind, fo, fit_stamp, sub):
    """A folder's output dir: own_all's layout, with own_all_loo in place of own_all under --loo (separate trees)."""
    return {"scaleup": Path("/work/scaleup") / fo / sub, "simh": Path("/work/simh") / (SIMH_BATCH.match(fo)[1] if SIMH_BATCH.match(fo) else "?") / sub / fo,
            "t196": Path("/out/famexp") / SPEC_STAMP / f"{sub}_{fit_stamp}" / fo, "a25simh": Path("/work/simh/primary_v3") / sub}[kind]


def _cut_dir(batch, sub, fo, uncut):
    """A cutfull output dir, /work/cutfull/<batch>/<sub>/<folder>/: an own_all_rs_uncut record goes ONLY to its own sub-folder and a registered record
    never does (4f / ownall-il, 10-16: the uncut headline must not land in, or be mistaken for, the registered (B) tree)."""
    if bool(uncut) != (sub == "own_all_rs_uncut"):
        raise SystemExit(f"{sub}: an own_all_rs_uncut record is written only under /cutfull/<batch>/own_all_rs_uncut/, and nothing else is (uncut={bool(uncut)})")
    return Path("/work/cutfull") / batch / sub / fo


def _cut_dir_selftest():  # the negative test: uncut into a registered folder (or the reverse) is refused
    assert _cut_dir("b", "own_all_rs_uncut", "f", True) == Path("/work/cutfull/b/own_all_rs_uncut/f")
    assert _cut_dir("b", "own_all_rs", "f", False) == Path("/work/cutfull/b/own_all_rs/f")
    for sub_, u_ in (("own_all_rs", True), ("own_all", True), ("own_all_loo", True), ("own_all_rs_uncut", False)):
        try:
            _cut_dir("b", sub_, "f", u_)
        except SystemExit:
            continue
        raise AssertionError(f"_cut_dir accepted sub {sub_} with uncut={u_}")


_cut_dir_selftest()


def _gate(main, manifest_mains):
    """The artifact-bound held-out gate (b6a6b6b48 §4): a violation iff the scored main is in its scoring model's training manifest."""
    return main in manifest_mains


def _logits(X64, w):
    import numpy as np

    return ((X64 - w["mean"]) / w["std"]) @ np.asarray(w["W"], np.float64).reshape(-1) + float(np.asarray(w["b"]).reshape(-1)[0])


def _cut_tokens(fe, readout, skip, rc):
    """The cutfull line (2e's DECISIVE_DG_DRIVER_SPEC @ a9ac396e3): per scored row, (call_index, token = the unit's readout position), in 2e's
    _b_rows / _b_rows_cmd order (run_famexp_cutfull._tokens @ 5c3484152, same filters), and per epoch n_tokens = its staged length; each epoch's
    extraction meta must say cut "full" with tokens_total = the staged length, and n_tokens must equal the registered tokens_before (full
    consumption, routing spec v2 @ c75141175), else STOP: a withheld epoch fails the routing revision."""
    out, n_tok, bad = [], {}, []
    staged = {}
    for path, h in sorted({(e["staged"], e["staged_sha256"]) for e in fe}):
        raw = Path("/b", path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"staged inputs {path}: sha256 differs")
        staged |= {(path, x["key"]): x for x in (json.loads(y) for y in raw.decode().splitlines() if y)}
    for e in fe:
        st = staged[(e["staged"], e["key"])]
        mt = json.loads((Path("/b") / e["run"] / "meta" / e["arm"] / f"{rc.stem(e['key'])}.json").read_text())
        n_tok[e["key"]] = len(st["ids"])
        if mt.get("cut") != "full" or mt.get("tokens_total") != len(st["ids"]) or e.get("tokens_before") != len(st["ids"]):
            bad.append(e["key"])
        if readout == "C":
            sk = {tuple(x) for x in skip.get(e["key"], [])}
            cs = [c for c in mt["calls"] if c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in sk]
        else:
            cs = [c for c in mt["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
        out += [(f"{e['world']}|{e['cell']}", e["key"], c["call_index"], c["readout"]) for c in cs]
    if bad:
        raise SystemExit(f"STOP (full consumption): {len(bad)} epochs are not cut 'full' with tokens_total = staged length = registered tokens_before, e.g. {bad[:2]}")
    return out, n_tok


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2, volumes={"/out": OUT, "/work": WORK})
def score(fit_stamp: str, fit_sha: str, inputs: dict, msg: dict, msg_sha: str, kind: str, writer: dict, line: str = "") -> dict:
    """One message (or T196 / the A25 sim-honest runs): universe mains scored ONLY by their fold's model; mains outside the universe routed
    fold -1 (all five models' logits, b6a6b6b48 §3); every row carries its model, preprocessing and training-manifest shas; the artifact-bound
    gate checks every scored main against THAT manifest, and a deliberately misrouted example must fail it; create-only, manifest last."""
    import numpy as np
    from safetensors.numpy import load_file

    OUT.reload()
    WORK.reload()
    K, VARS, fold, od, sub = _cfg(inputs, fit_stamp)
    if hashlib.sha256((od / "fit_record.json").read_bytes()).hexdigest() != fit_sha:
        raise SystemExit("fit record sha256 differs")
    frec = json.loads((od / "fit_record.json").read_text())
    if not frec["gate"]["refit_match_pass"] or not frec["gate"]["all_certified"] or frec["inputs_sha256"] != inputs["_sha256"] or \
            (inputs.get("loo") and frec.get("loo_sha256") != inputs["loo"]["_sha256"]) or (inputs.get("rs") and frec.get("rs_sha256") != inputs["rs"]["_sha256"]) or \
            (inputs.get("rs_uncut") and frec.get("uncut_sha256") != inputs["rs_uncut"]["_sha256"]):
        raise SystemExit("the fit is not usable here: refit-match control failed, an uncertified model, or another inputs record")
    main_of = inputs["main_of"]
    art = {}
    for f in frec["fits"]:  # every model's artifacts, re-hashed: weights, preprocessing, training manifest
        mp = od / "training" / f"{f['variant']}__{f['readout']}__f{f['fold']}.json"
        wp = od / "weights" / f"{f['variant']}__{f['readout']}__L48__f{f['fold']}.safetensors"
        if hashlib.sha256(mp.read_bytes()).hexdigest() != f["training_manifest_sha256"] or hashlib.sha256(wp.read_bytes()).hexdigest() != f["weights_sha256"]:
            raise SystemExit(f"{f['variant']} {f['readout']} f{f['fold']}: an artifact differs from the fit record")
        w = load_file(str(wp))
        if _prep_sha(w["mean"], w["std"]) != f["preproc_sha256"]:
            raise SystemExit("preprocessing sha256 differs from the fit record")
        art[(f["variant"], f["readout"], f["fold"])] = (w, set(json.loads(mp.read_text())), f)
    spec = _spec()
    so = None
    if kind == "a25simh":  # the 149 counted A25 sim-honest runs, from run_owns.py's prep spec (as run_famexp_simh_ownfam.py reads them)
        pth = Path("/out/owns") / OWNS / f"spec_owns_{OWNS}.json"
        if hashlib.sha256(pth.read_bytes()).hexdigest() != OWNS_SHA:
            raise SystemExit("run_owns.py's prep spec differs from its pin")
        so = json.loads(pth.read_text())
        msg = {"epochs": [{"key": k, "world": wt[0], "cell": wt[1], "run": so["run_of"][k], "arm": so["arm_of"][k], "folder": "primary_v3"}
                          for k, wt in so["map"].items() if wt[0] == "simh_full"]}
        if sorted(e["cell"] for e in msg["epochs"]) != sorted(inputs["a25_counted_tasks"]) or len(msg["epochs"]) != 149:
            raise SystemExit("the prep's simh_full epochs are not exactly the 149 counted runs")
    eps = msg["epochs"]
    if inputs.get("rs"):  # 3a (c): every scored arm's and pool's main is in the pinned fold list, else STOP (never a silent fold -1)
        miss = sorted({main_of.get(e["cell"], f"<no main: {e['cell']}>") for e in eps if main_of.get(e["cell"]) not in fold})
        if miss:
            raise SystemExit(f"STOP: {len(miss)} scored mains are not in the re-split fold list, e.g. {miss[:3]}")
    skip = {e["key"]: spec["skip"].get(e["key"], []) for e in eps} if kind == "t196" else {e["key"]: so["skip"][e["key"]] for e in eps} if so else _skip(eps, spec)
    cell_fold = {e["cell"]: fold.get(main_of[e["cell"]], -1) for e in eps}
    cand = [main_of[e["cell"]] for e in eps if main_of[e["cell"]] in fold] + sorted(fold)  # this message's universe mains first, then any universe main
    neg = {}
    for var in VARS:  # a deliberately misrouted example per fraction x readout: a main scored by an other-fold model that trained on it
        for readout in ("R", "C"):
            m_, g_ = next(((m, g) for m in cand for g in range(K) if g != fold[m] and m in art[(var, readout, g)][1]), (None, None))
            if m_ is None or _gate(m_, art[(var, readout, g_)][1]) is not True:
                raise SystemExit(f"the misrouted negative for {var} {readout} did not fail the gate: the gate cannot be trusted")
            neg[f"{var}|{readout}"] = {"main": m_, "designated_fold": fold[m_], "misrouted_to_fold": g_, "gate_flags_violation": True,
                                       "model_weights_sha256": art[(var, readout, g_)][2]["weights_sha256"]}
    by_folder = {}
    for e in eps:
        by_folder.setdefault(e["folder"], []).append(e)
    out = {}
    if line and (line != "cutfull" or not inputs.get("rs") or msg.get("volume") != "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14" or not msg.get("batch")):
        raise SystemExit("the cutfull line scores only --rs cutfull messages on the cutfull Volume, with a routing batch")
    rc_ = _x2e()[0]
    for fo, fe in sorted(by_folder.items()):
        tgt = _cut_dir(msg["batch"], sub, fo, inputs.get("rs_uncut")) if line else _tgt(kind, fo, fit_stamp, sub)  # 09 @ 77a770eec: separate cutfull tree
        if tgt.exists():
            raise SystemExit(f"{tgt}: exists (create-only)")
        files, counts, viol = {}, {}, 0
        for readout in ("R", "C"):
            X, names, ci, units, folds = _load(fe, readout, skip, cell_fold)
            if line:
                tk, n_tok = _cut_tokens(fe, readout, skip, rc_)
                if [t[0] for t in tk] != names.tolist() or [t[2] for t in tk] != ci.tolist():
                    raise SystemExit(f"{fo} {readout}: token positions do not align with the scored rows")
                run_key = np.array([t[1] for t in tk])
                tok = np.array([t[3] for t in tk], np.int64)
                ntk = np.array([n_tok[k_] for k_ in run_key.tolist()], np.int64)
                if ((tok < 0) | (tok >= ntk)).any():
                    raise SystemExit(f"{fo} {readout}: a token position outside 0 <= token < n_tokens")
            if inputs.get("rs") and (folds == -1).any():  # amendment 3 §4: no fold -1 unit, by construction
                raise SystemExit(f"STOP: {fo} {readout}: {int((folds == -1).sum())} rows have no re-split fold")
            X64 = X.astype(np.float64)
            rmain = np.array([main_of[u.split("|", 1)[1]] for u in names])
            for var in VARS:
                n = len(names)
                s, sbm = np.full(n, np.nan), np.full((n, K), np.nan)
                wsha, psha, tsha = (np.full(n, "", dtype=object) for _ in range(3))
                wsha5, psha5, tsha5 = (np.full((n, K), "", dtype=object) for _ in range(3))
                j = np.flatnonzero(folds == -1)  # out of the universe: fold -1, all K models' logits, each model's identity
                for k in range(K):
                    w, man_k, fr = art[(var, readout, k)]
                    i = np.flatnonzero(folds == k)
                    viol += sum(_gate(m, man_k) for m in set(rmain[i].tolist()))
                    viol += sum(_gate(m, man_k) for m in set(rmain[j].tolist()))  # a fold -1 main must be in NONE of the five manifests
                    if len(i):
                        s[i] = _logits(X64[i], w)
                        wsha[i], psha[i], tsha[i] = fr["weights_sha256"], fr["preproc_sha256"], fr["training_manifest_sha256"]
                    if len(j):
                        sbm[j, k] = _logits(X64[j], w)
                        wsha5[j, k], psha5[j, k], tsha5[j, k] = fr["weights_sha256"], fr["preproc_sha256"], fr["training_manifest_sha256"]
                if viol:
                    raise SystemExit(f"STOP: {fo} {var} {readout}: {viol} scored mains are in their scoring model's training manifest")
                if not (np.isfinite(s[folds >= 0]).all() and np.isfinite(sbm[folds == -1]).all()):
                    raise SystemExit(f"{fo}: non-finite scores")
                files[f"{var}__{readout}.npz"] = _npz(tgt / f"{var}__{readout}.npz", key=names, unit=units, call_index=ci, fold=folds, score=s, score_by_model=sbm,
                                                     model_weights_sha256=wsha.astype(str), preproc_sha256=psha.astype(str), training_manifest_sha256=tsha.astype(str),
                                                     model_weights_sha256_by_model=wsha5.astype(str), preproc_sha256_by_model=psha5.astype(str),
                                                     training_manifest_sha256_by_model=tsha5.astype(str),
                                                     **({"run_key": run_key, "token": tok, "n_tokens": ntk} if line else {}))
            wd = np.array([u.split("|", 1)[0] for u in names])
            counts[readout] = {"rows": int(len(names)), "rows_mapped": int((folds >= 0).sum()), "rows_fold_minus1": int((folds == -1).sum()), "runs": len(set(names.tolist())),
                               **({"rows_past_32768": int((tok >= 32768).sum())} if line else {}),
                               "by_world": {w_: {"mapped": int(((wd == w_) & (folds >= 0)).sum()), "fold_minus1": int(((wd == w_) & (folds == -1)).sum())} for w_ in sorted(set(wd.tolist()))},
                               "per_run": {u: [int(((names == u) & (folds >= 0)).sum()), int(((names == u) & (folds == -1)).sum())] for u in sorted(set(names.tolist()))},
                               "runs_without_rows": sorted({f"{e['world']}|{e['cell']}" for e in fe} - set(names.tolist()))}
        man = {"label": (inputs["rs_uncut"]["label"] if inputs.get("rs_uncut") else RS_LABEL if inputs.get("rs") else LOO_LABEL if inputs.get("loo") else "own_all{,25,10}_L48 scores (09 Append 40 + amendments 1, 3 @ de33ce99e; b6a6b6b48 §3-4): new recipe; POST HOC; trained after results were read; SEALED"),
               "folder": fo, "kind": kind, "fit_stamp": fit_stamp, "fit_record_sha256": fit_sha, "message_sha256": msg_sha, "inputs_sha256": inputs["_sha256"],
               "models": {f"{v}|{r}|f{k}": {"weights_sha256": fr["weights_sha256"], "preproc_sha256": fr["preproc_sha256"], "training_manifest_sha256": fr["training_manifest_sha256"]}
                          for (v, r, k), (_, _, fr) in sorted(art.items())},
               "counts": counts, "files": files, "fold_minus1_out_of_universe": sorted({f"{e['world']}|{e['cell']}" for e in fe if cell_fold[e["cell"]] < 0}),
               "gate": {"in_fold_violations": viol, "misrouted_negative": neg},
               "s4_orphan_rows_never_pooled": sorted(f"{e['world']}|{e['cell']}" for e in fe if e["key"].removesuffix("#e0") in inputs["s4_orphan_records"]),
               "routing": ("re-split folds (amendment 3 §4): every main scored ONLY by its fold's model; no fold -1" if inputs.get("rs") else
                           "set o's mains: model o only (score; o = the main's origin, 0 S_old / 1 S_new / 2 S_newmains); mains of no origin: fold -1, all three models (score_by_model)"
                           if inputs.get("loo") else "universe mains: the model of the main's fold only (score); out-of-universe mains: fold -1, all five models (score_by_model, b6a6b6b48 §3)"),
               "writer": writer} | ({"sealed_epoch2": "SEALED (09 Append 38 §4/§5): per-row scores only, nothing derived; unread until 09 verifies the epoch-2 pool-key artifact"}
                                    if fo.startswith("ep2_") else {}) | (
            {"line": "cutfull", "cutfull_label": "uncut: all tokens; POST HOC (09 Append 41); a sensitivity, never replaces a registered row; "
                                                   + ("own_all_rs_uncut weights (their own fit)" if inputs.get("rs_uncut") else "registered own_all_rs weights, no refit"),
             "layout": f"09 @ 77a770eec / ruling (c): /cutfull/<routing batch>/{sub}/<cutfull folder>/", "batch": msg["batch"], "original_folder": msg["original_of"][fo],
             "routing_record": msg["routing_pin"], "attestation": msg["attest_pin"], "inventory": msg["inventory_pin"], "cutfull_rule": msg["rule"], "record": msg["record"],
             "weights": {"fit_stamp": fit_stamp, "fit_record_sha256": fit_sha, "path": f"/{sub}/{fit_stamp}/weights/ on probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"},
             "schema": "per row key (world|task), run_key (staged epoch key), unit, call_index, token (readout position), n_tokens (epoch length), score, score_by_model (n x 5), fold, plus the model sha columns"}
            if line else {})
        (tgt / "manifest_own_all.json").open("x").write(json.dumps(man, indent=1))
        out[fo] = {"manifest_sha256": hashlib.sha256((tgt / "manifest_own_all.json").read_bytes()).hexdigest(), "in_fold_violations": viol, "kind": kind,
                   **({"path": str(tgt), "batch": msg["batch"], "line": "cutfull"} if line else {}),
                   "s4_orphan_rows": len(man["s4_orphan_rows_never_pooled"]),
                   "counts": {r: {k: v for k, v in c.items() if k != "runs_without_rows"} | {"runs_without_rows": len(c["runs_without_rows"])} for r, c in counts.items()}}
    OUT.commit()
    WORK.commit()
    return {"folders": out, "misrouted_negative": neg}


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2, volumes={"/out": OUT, "/work": WORK})
def verify_rows(fit_stamp: str, fit_sha: str, folders: list, loo: bool = False, rs: bool = False, uncut: bool = False) -> dict:
    """09's artifact check (4f's relay): every emitted row's three shas, read back from the Volume npz files (sha-checked against their
    manifests), equal the fit record's entries for its designated model, or the five-model set for fold -1 rows. Reads fold and sha columns only."""
    import numpy as np

    OUT.reload()
    WORK.reload()
    K, sub = (3, "own_all_loo") if loo else (5, "own_all_rs_uncut") if (rs and uncut) else (5, "own_all_rs") if rs else (5, "own_all")  # uncut: its own tree
    od = Path("/out") / sub / fit_stamp
    if hashlib.sha256((od / "fit_record.json").read_bytes()).hexdigest() != fit_sha:
        raise SystemExit("fit record sha256 differs")
    fits = {(f["variant"], f["readout"], f["fold"]): f for f in json.loads((od / "fit_record.json").read_text())["fits"]}
    cnt, bad = {}, []
    for kind, fo, man_sha, *rest in folders:
        batch = rest[0] if rest else ""  # the cutfull line: the path is BUILT from a checked batch and cutfull folder name, never taken from a record (Codex r9)
        if batch and not (re.fullmatch(r"[a-z0-9][a-z0-9_-]*", batch) and re.fullmatch(r"[a-z0-9_]+?_cutfull_v1(?:_try[0-9]+)?__[A-Za-z0-9_@.]+", fo)):
            raise SystemExit(f"{fo}: not a cutfull batch / folder name")
        tgt = _cut_dir(batch, sub, fo, uncut) if batch else _tgt(kind, fo, fit_stamp, sub)
        mb = (tgt / "manifest_own_all.json").read_bytes()
        if hashlib.sha256(mb).hexdigest() != man_sha:
            bad.append(f"{fo}: manifest sha256 differs from the score record")
            continue
        man = json.loads(mb)
        if (man.get("line") == "cutfull") != bool(batch) or (batch and man.get("batch") != batch):  # item 4: one line per verify, both ways (Codex r9)
            bad.append(f"{fo}: a {man.get('line') or 'registered'}-line pack in a {'cutfull' if batch else 'registered'} verify")
            continue
        if {k: v for k, v in man["models"].items()} != {f"{v}|{r}|f{k}": {"weights_sha256": f["weights_sha256"], "preproc_sha256": f["preproc_sha256"],
                                                                         "training_manifest_sha256": f["training_manifest_sha256"]} for (v, r, k), f in fits.items()}:
            bad.append(f"{fo}: the manifest's model map differs from the fit record")
        for name, h in man["files"].items():
            b = (tgt / name).read_bytes()
            if hashlib.sha256(b).hexdigest() != h:
                bad.append(f"{fo}/{name}: npz sha256 differs from the manifest")
                continue
            var, readout = name.removesuffix(".npz").split("__")
            z = np.load(io.BytesIO(b))
            fold = z["fold"]
            cols = {c: z[c] for c in ("model_weights_sha256", "preproc_sha256", "training_manifest_sha256",
                                      "model_weights_sha256_by_model", "preproc_sha256_by_model", "training_manifest_sha256_by_model")}
            c_ = cnt.setdefault(f"{var}|{readout}", {"rows": 0, "match": 0, "mismatch": 0, "rows_fold_minus1": 0})
            for i in range(len(fold)):
                k = int(fold[i])
                if k >= 0:
                    f = fits[(var, readout, k)]
                    ok = (cols["model_weights_sha256"][i], cols["preproc_sha256"][i], cols["training_manifest_sha256"][i]) == (f["weights_sha256"], f["preproc_sha256"], f["training_manifest_sha256"]) \
                        and not any(cols[c][i].any() for c in ("model_weights_sha256_by_model", "preproc_sha256_by_model", "training_manifest_sha256_by_model"))
                else:
                    c_["rows_fold_minus1"] += 1
                    if rs:
                        bad.append(f"{fo}/{name} row {i}: fold -1 under --rs")
                    ok = all((cols["model_weights_sha256_by_model"][i][g], cols["preproc_sha256_by_model"][i][g], cols["training_manifest_sha256_by_model"][i][g])
                             == (fits[(var, readout, g)]["weights_sha256"], fits[(var, readout, g)]["preproc_sha256"], fits[(var, readout, g)]["training_manifest_sha256"]) for g in range(K)) \
                        and not (cols["model_weights_sha256"][i] or cols["preproc_sha256"][i] or cols["training_manifest_sha256"][i])
                c_["rows"] += 1
                c_["match" if ok else "mismatch"] += 1
                if not ok and len(bad) < 20:
                    bad.append(f"{fo}/{name} row {i}")
    return {"folders": len(folders), "counts": cnt, "problems": bad}


@app.function(image=img, cpu=(2, 2), memory=(8192, 8192), timeout=1800, retries=0, scaledown_window=2, volumes={"/out": OUT, "/work": WORK})
def coverage_units(fit_stamp: str, folders: list) -> dict:
    """3a (c)'s coverage assert, read side: each folder's manifest (sha-checked against its score record) -> per readout, per run (world|cell)
    [mapped units, fold -1 units]. Counts only; no score is read."""
    OUT.reload()
    WORK.reload()
    out, bad = {}, []
    for kind, fo, man_sha in folders:
        mb = (_tgt(kind, fo, fit_stamp, "own_all_rs") / "manifest_own_all.json").read_bytes()
        if hashlib.sha256(mb).hexdigest() != man_sha:
            bad.append(f"{fo}: manifest sha256 differs from the score record")
            continue
        man = json.loads(mb)
        out[fo] = {"kind": kind, "per_run": {rd: c["per_run"] for rd, c in man["counts"].items()}, "runs_without_rows": {rd: c["runs_without_rows"] for rd, c in man["counts"].items()}}
    return {"folders": out, "problems": bad}


def _folder_set(fo, cell, origin):
    """The set a scored unit belongs to (3a (c)): S_paper493 for its own folders (so g4's T196-keyed epochs count under S_paper493), else the cell's origin."""
    return "S_paper493" if fo.startswith("sp493_") else origin.get(cell)


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2, volumes={"/out": OUT, "/work": WORK})
def negative_e2e(fit_stamp: str, fit_sha: str, inputs: dict, msg: dict, msg_sha: str, writer: dict) -> dict:
    """09's END-TO-END negative: a few real rows sent through the real scoring path (score's own code, in-process) with their mains
    deliberately re-routed to a fold model that TRAINED on them, into a clearly named test folder; the run must fail closed."""
    import traceback

    K, _, fold, _, _ = _cfg(inputs, fit_stamp)
    bad_in = json.loads(json.dumps(inputs))
    moved = {}
    for e in msg["epochs"]:
        m = inputs["main_of"][e["cell"]]
        if m in fold and m not in moved:
            moved[m] = {"true_fold": fold[m], "routed_to": (fold[m] + 1) % K}
            if inputs.get("rs"):
                bad_in["rs"]["by_main"][m] = [moved[m]["routed_to"], "MISROUTED (negative test)"]
            elif inputs.get("loo"):
                bad_in["loo"]["fold_of_main"][m] = moved[m]["routed_to"]
            else:
                bad_in["folds"]["by_main"][m] = [moved[m]["routed_to"], "MISROUTED (negative test)"]
    test = {"epochs": [dict(e, folder=f"own_all_negative_test__{e['folder']}") for e in msg["epochs"]], "volume": msg.get("volume")}
    try:
        score.local(fit_stamp, fit_sha, bad_in, test, msg_sha, "scaleup", writer | {"negative_test": True})
    except SystemExit as e:  # only the row-level training-overlap STOP on the rerouted test folder counts (Codex round 181)
        written = [str(p) for p in Path("/work/scaleup").glob("own_all_negative_test__*")]
        want = str(e).startswith("STOP: own_all_negative_test__") and "scored mains are in their scoring model's training manifest" in str(e)
        return {"failed_closed": bool(want), "error": str(e), "expected_error": "STOP: own_all_negative_test__<folder> <variant> <readout>: <n> scored mains are in their scoring model's training manifest",
                "moved": moved, "rows_epochs": len(msg["epochs"]), "test_outputs_left": written}
    except Exception:
        return {"failed_closed": False, "error": traceback.format_exc()[-800:], "moved": moved}
    return {"failed_closed": False, "error": None, "moved": moved, "note": "the misrouted scoring did NOT fail: the gate is broken"}


@app.local_entrypoint()
def main(mode: str, inputs: str, fit_stamp: str = "", messages: str = "", t196_spec: str = "", census: str = "", followup: str = "", dry: bool = False, loo: bool = False,
         rs: bool = False, cutfull: bool = False, routing_commit: str = "", inventory_b: str = "", refuse: str = "", message_pins: str = "", uncut: bool = False):
    _launch(mode, inputs, fit_stamp, messages, t196_spec, census, dry, followup, loo, rs, cutfull, routing_commit, inventory_b, refuse, message_pins, uncut)


CUT_ROUTING = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
CUT_WORLD = {"run1": "run1", "run2": "run2", "sim": "sim", "honest_a": "honest_a", "honest_b": "honest_b", "sim_honest": "simh_311"}  # registered key labels (as famexp's)
SIMH311 = re.compile(r"^simh311_q[1-4]_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")
SU311 = re.compile(r"^su311_(q[0-9]+r?)_(main|run2|sim)(_s[0-9]+)?_v1(_try[0-9]+)?__(honest_a|honest_b|attack|sim_attack)$")


def _cut_jobs(git_, routing_commit, inv_b=None, refuse=()):
    """Cutfull messages from d9's routing records (highest revision per set, attestation pointer failing {} for that revision): (a'') restricted to
    inventory v3's planned folders and checked by famexp's cutfull grammar (09's (a)-(d), d9's original_folders, inventory v3 by key); with inv_b
    (commit, path, sha256 of 2e's inventory_b), also the (b) routed sets and the S_old (b) sets (d9's S_old attestations, latest revision, failing
    empty), whose originals are the closed list d9 attests (09's (c) applied to that list)."""
    jobs_b = _cut_jobs_b(git_, routing_commit, inv_b, refuse) if inv_b else ([], {})
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", routing_commit, "origin/aswin/q3-cut-option"]).returncode != 0:  # exact ref (Codex r12)
        raise SystemExit("--routing-commit is not on origin/aswin/q3-cut-option")
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    invb = show(*cf.INVENTORY[:2])
    if hashlib.sha256(invb).hexdigest() != cf.INVENTORY[2]:
        raise SystemExit("inventory v3 differs from its pin")
    inv_runs = json.loads(invb)["runs"]
    planned = {r["uncut_planned"]["folder"] for r in inv_runs}
    best = {}
    for p in git_("ls-tree", "-r", "--name-only", routing_commit, CUT_ROUTING).split():
        m = re.fullmatch(r".*/(.+?__.+?_cutfull)(?:_r([0-9]+))?\.json", p)
        if m and int(m[2] or 1) > best.get(m[1], (0, ""))[0]:
            best[m[1]] = (int(m[2] or 1), p)
    jobs, skipped = [], {}
    for stem_, (k, p) in sorted(best.items()):
        rb = show(routing_commit, p)
        rec = json.loads(rb)
        arms = [dict(a, token=a["meta_arm"].split("@", 1)[0]) for a in rec["arms"]]
        for a in arms:
            a["folder"] = f"{rec['staged_set']}__{a['token']}"
        if not {a["folder"] for a in arms} <= planned:
            skipped[stem_] = "outside inventory v3 (not (a''))"
            continue
        ap = p[:-5] + "_attest.json"
        try:
            atb = show(routing_commit, ap)
        except subprocess.CalledProcessError:
            skipped[stem_] = f"revision {k}: no attestation pointer"
            continue
        at = json.loads(atb)
        if at.get("failing") != {} or at.get("revision") != k or not str(at.get("routing_record", "")).startswith(p):
            raise SystemExit(f"{stem_}: the attestation pointer does not attest this revision with failing {{}}")
        if rec.get("volume") != cf.VOLUME or rec.get("cut") != "full" or rec.get("revision") != k or not str(rec.get("repro_gate", "")).startswith("experiments/") or "cf5c4f5e" not in str(rec.get("step0_key_list", "cf5c4f5e")):
            raise SystemExit(f"{stem_}: not a cutfull record of this revision on the cutfull Volume with its step-0 gate")
        by_folder = {a["folder"]: a for a in arms}
        eps = []
        for r in inv_runs:
            a = by_folder.get(r["uncut_planned"]["folder"])
            if a:
                w_ = ast.literal_eval(a["world"]) if isinstance(a["world"], str) else a["world"]
                if w_ not in ([r["world"]], [CUT_WORLD[r["world"]]]):  # d9 names the sim-honest arm's world by its registered label (simh_311)
                    raise SystemExit(f"{stem_} {a['folder']}: the inventory world {r['world']} is not the arm's {w_}")
                eps.append({"key": r["key"], "world": CUT_WORLD[r["world"]], "cell": r["task"], "folder": a["folder"], "staged": a["staged_inputs_file"],
                            "staged_sha256": a["staged_sha"], "run": rec["extraction_root"], "arm": a["meta_arm"], "tokens_before": r["tokens_before"], "eval_sha256": r.get("eval_sha256")})
        for a in arms:
            if sum(e["folder"] == a["folder"] for e in eps) != a["epochs"]:
                raise SystemExit(f"{stem_} {a['folder']}: inventory v3 runs differ from the routing record's epochs")
        msg = {"volume": rec["volume"], "batch": rec["batch"], "original_folders": rec["original_folders"], "epochs": eps}
        accepted = lambda x: bool(SIMH311.fullmatch(x) or SU311.fullmatch(x))  # noqa: E731
        orig = cf.split(msg, True, accepted)  # 09's (a)-(d) on every folder, d9's original_folders, inventory v3 by key
        msg |= {"original_of": {e["folder"]: o["folder"] for e, o in zip(msg["epochs"], orig["epochs"], strict=True)},
                "routing_pin": {"path": p, "commit": routing_commit, "sha256": hashlib.sha256(rb).hexdigest(), "revision": k},
                "attest_pin": {"path": ap, "commit": routing_commit, "sha256": hashlib.sha256(atb).hexdigest()},
                "inventory_pin": {"commit": cf.INVENTORY[0], "path": cf.INVENTORY[1], "sha256": cf.INVENTORY[2]}, "rule": cf.RULE,
                "record": {x: rec.get(x) for x in ("staged_set", "extraction_root", "extraction_record", "repro_gate", "step0_key_list", "chat_template_sha256", "stage_record")}}
        kinds = {("simh" if SIMH311.fullmatch(o) else "scaleup") for o in msg["original_of"].values()}
        if len(kinds) != 1:
            raise SystemExit(f"{stem_}: mixed sim-honest and scale-up folders")
        mb = json.dumps(msg, sort_keys=True).encode()
        jobs.append((msg, hashlib.sha256(mb).hexdigest(), kinds.pop(), rec["volume"]))
    return jobs + jobs_b[0], skipped | jobs_b[1]


SOLD_ATT_TC = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/attest/attestations_sold__q4f_main_cutfull_r3.json"  # template_class by key (T196 training)
CHAIN_REF = ("origin/aswin/q3-overnight-chain", "experiments/2026-10-14_q3_overnight_chain/config")
Q4B_MAN = ("948ecd8e8", "experiments/2026-10-13_q4_why_weaker_POSTHOC/results/manifest_q4b_eval_upload_v1.json")
HA25F_INV = ("85447113f", "experiments/2026-10-15_cut_sensitivity/results/inventory_ha25f_v1_2026-10-16_025414.json",
             "a626647f118a84ca181e54dc69872a31b9ebc3c90793c49192e56ef9d3ce8c08")  # 2e's 25f H_A inventory (4f, 10-16): by-key cross-check of the 3 roots


def _uncut_training(git_, routing_commit, inv_b, inp, chain_commit=None):
    """own_all_rs_uncut's training epochs BY KEY (09 uncut §6): every own_all training epoch (run1 + honest_a of T196 / S_new / S_newmains) must have
    an attested uncut source ((a'') routing, (b) routing, S_old attestation, the 25f H_A cutfull sets), with the same run key AND .eval sha256 as the
    registered upload manifests (lead RULE §1); the two redraw2 no-policy-call honest_a references carry 0 rows in both fits and are listed, not missing
    (09 §7). Returns (epochs by uncut root, a report); any other missing or unequal key is a STOP (the probe is UNAVAILABLE). chain_commit: the fit's
    pinned chain commit for the registered upload manifests (scoring rebuilds the fit's identity from it); None at fit time (resolved and recorded)."""
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    src = {}
    for m, _, _, _ in _cut_jobs(git_, routing_commit, inv_b, ())[0]:
        for e in m["epochs"]:
            src.setdefault(e["key"], e)
    for p in git_("ls-tree", "-r", "--name-only", routing_commit, CUT_ROUTING).split():  # the 25f H_A cutfull sets (one epoch each; key checked in the rows job)
        m_ = re.fullmatch(r".*/ha25f__(su311_q[0-9]_main_ha25f)_cutfull(?:_r([0-9]+))?\.json", p)
        if not m_:
            continue
        others = [x for x in git_("ls-tree", "-r", "--name-only", routing_commit, CUT_ROUTING).split() if re.fullmatch(rf".*/ha25f__{m_[1]}_cutfull(?:_r[0-9]+)?\.json", x)]
        if max(int(re.search(r"_r([0-9]+)\.json$", x)[1]) if "_cutfull_r" in x else 1 for x in others) != int(m_[2] or 1):
            continue  # not the highest revision (r1 of these sets is void: d9's glob error)
        rec = json.loads(show(routing_commit, p))
        at = json.loads(show(routing_commit, p[:-5] + "_attest.json"))
        if at.get("failing") != {} or at.get("revision") != rec.get("revision") or not str(at.get("routing_record", "")).startswith(p) or rec.get("cut") != "full":
            raise SystemExit(f"{p}: the 25f H_A cutfull set is not attested at this revision")
        (a,) = rec["arms"]
        cell = rec["members"][0].split("/")[-1]
        tk = next(e for e in inp["training_epochs"] if e["cell"] == cell and str(e.get("ha25f_selected")) == "True")
        mp_, mc_ = rec["manifest"].split(" @ ")  # the routed source's own upload manifest, read independently of the registered pin (Codex r15)
        mf = [f for f in json.loads(show(mc_, mp_))["files"] if f["task"] == cell and f["world"] == "honest_a"]
        sp_, sc_ = rec["stage_record"].split(" @ ")  # full consumption: the cutfull stage record's staged length (one epoch per set)
        sg = json.loads(show(sc_, sp_))["main"]["assembled"]
        cov = sg["coverage"][a["meta_arm"]]
        if len(mf) != 1 or f"{mf[0]['path']}#e0" != tk["key"] or sg["epochs"] != {a["meta_arm"]: 1} or cov["runs"] != 1 or cov["totals"]["truncated"] != 0 or \
                cov["totals"]["tokens_before"] != cov["totals"]["tokens_after"] or sg["staged_inputs_sha256"][f"{a['meta_arm']}.jsonl"] != a["staged_sha"]:
            raise SystemExit(f"{p}: the 25f source manifest or cutfull stage record does not give exactly this one epoch, untruncated, at the routed staged sha")
        src.setdefault(tk["key"], {"key": tk["key"], "world": "honest_a", "cell": cell, "staged": a["staged_inputs_file"], "staged_sha256": a["staged_sha"],
                                   "run": rec["extraction_root"], "arm": a["meta_arm"], "tokens_before": cov["totals"]["tokens_before"], "eval_sha256": mf[0]["sha256"]})
    hb = show(HA25F_INV[0], HA25F_INV[1])
    if hashlib.sha256(hb).hexdigest() != HA25F_INV[2]:
        raise SystemExit("inventory_ha25f_v1 differs from its pin")
    ha_inv = {r["key"]: r for r in json.loads(hb)["runs"]}
    ha_keys = sorted(e["key"] for e in inp["training_epochs"] if str(e.get("ha25f_selected")) == "True")
    ha_bad = [k for k in ha_keys if k not in ha_inv or k not in src or (ha_inv[k]["tokens_before"], ha_inv[k]["full_len"], ha_inv[k]["uncut_planned"]["staged_sha256"],
                                                                         ha_inv[k]["uncut_planned"]["extraction_root"], ha_inv[k]["world"])
              != (src[k]["tokens_before"], src[k]["tokens_before"], src[k]["staged_sha256"], src[k]["run"], "honest_a")]
    if ha_bad or set(ha_inv) != set(ha_keys):
        raise SystemExit(f"STOP: the 25f H_A training sources differ by key from inventory_ha25f_v1: {ha_bad or sorted(set(ha_inv) ^ set(ha_keys))}")
    reg_sha = {}
    chain = chain_commit or git_("rev-parse", CHAIN_REF[0]).strip()
    for f in git_("ls-tree", "--name-only", f"{chain}:{CHAIN_REF[1]}").split():
        if re.match(r"^manifest_(su311_q[1-4]r?_main|nm360_s[0-9a-z]+_main)(_s[0-9]+)?\.json$", f):
            reg_sha |= {x["path"]: x["sha256"] for x in json.loads(show(chain, f"{CHAIN_REF[1]}/{f}")).get("files", [])}
    reg_sha |= {x["path"]: x["sha256"] for x in json.loads(show(*Q4B_MAN))["files"]}
    reg_sha |= {e["key"].removesuffix("#e0"): inp["pins"]["ha25f_selected"][e["cell"]] for e in inp["training_epochs"] if str(e.get("ha25f_selected")) == "True"}
    nopol = sorted(e["key"] for e in inp["training_epochs"] if e["world"] == "honest_a" and e["cell"] in ("nm222_slack_clone", "nm203_slack_clone") and "_redraw2/" in e["key"])
    missing, unequal, by_root = [], [], {}
    for e in inp["training_epochs"]:
        if e["key"] in nopol:
            continue
        u = src.get(e["key"])
        if u is None:
            missing.append(e["key"])
            continue
        rs_ = reg_sha.get(e["key"].removesuffix("#e0"))
        if u["world"] != e["world"] or u["cell"].removesuffix("_r1") != e["cell"] or not rs_ or not u.get("eval_sha256") or rs_ != u["eval_sha256"]:  # missing = STOP (Codex r15)
            unequal.append(e["key"])
            continue
        by_root.setdefault(u["run"], []).append(dict(u, cell=e["cell"]))
    tc = json.loads(show(routing_commit, SOLD_ATT_TC))["template_class"]
    t196_keys = {e["key"] for e in inp["training_epochs"] if e["pool"] == "T196"}
    changed = sorted(k for arm in ("attack", "honest_a") for k, v in tc[arm].items() if v != "fixed, render unchanged" and k in t196_keys)
    rep_ = {"needed": len(inp["training_epochs"]) - len(nopol), "found": sum(len(v) for v in by_root.values()), "missing": missing, "unequal_key_or_eval_sha": unequal,
            "no_policy_call_refs_listed_not_missing": nopol, "roots": {r: len(v) for r, v in sorted(by_root.items())},
            "registered_eval_sha_checked": sum(1 for v in by_root.values() for e in v if reg_sha.get(e["key"].removesuffix("#e0"))),
            "template_class_render_changed": changed, "template_class_source": {"path": SOLD_ATT_TC, "commit": routing_commit}, "chain_commit": chain,
            "ha25f_inventory_check": {"inventory": {"commit": HA25F_INV[0], "path": HA25F_INV[1], "sha256": HA25F_INV[2]}, "keys": len(ha_keys), "mismatches": 0,
                                      "fields": "tokens_before = full_len = the cutfull stage record's, staged sha256, extraction root, world"}}
    if missing or unequal:
        raise SystemExit(f"STOP (own_all_rs_uncut UNAVAILABLE, 09 §6): {len(missing)} training keys without an attested uncut source, {len(unequal)} with an unequal key / world / .eval sha, e.g. {(missing + unequal)[:3]}")
    return by_root, rep_


IB_FIX = ("09b070617", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
          "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03")  # 2e's fix_v1: v2's 77 nm360_s2_run2_s1 rows -> r4, by key (4f, 10-16)
SOLD_STD = ("703ccbe0f3fb", {"q4f_main_cutfull_r3": "764df967", "q4f_run2_cutfull_r2": "0e718de3", "q4f_c_cutfull_r2": "41d4bbb6", "rh1e4_stage_cutfull_r2": "b37c94f7",
                             "simh_q4f_v1_full_b1_cutfull_r2": "76339b31", "simh_q4f_v1_full_b2_cutfull_r2": "72ab3a14"})  # d9's standard S_old (b) routing records (4f)


def _uncut_checks(git_, by_root, ib_u, inp):
    """Launch-time by-key checks for the uncut fit record (4f, 10-16), STOP on any mismatch: (1) inventory_b_fix_v1 differs from the fit's v2 pin on
    exactly 77 rows, none a training row; (2) every T196 (S_old) training epoch, read from the attestations, equals d9's standard S_old routing records
    @ 703ccbe0f3fb by key: staged sha256, staged file, extraction root, meta arm."""
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    train = {e["key"]: e for v in by_root.values() for e in v}
    v2b, fxb = show(ib_u[0], ib_u[1]), show(IB_FIX[0], IB_FIX[1])
    if hashlib.sha256(v2b).hexdigest() != ib_u[2] or hashlib.sha256(fxb).hexdigest() != IB_FIX[2]:
        raise SystemExit("inventory_b v2 or fix_v1 differs from its pin")
    v2, fx = ({r["key"]: r for r in json.loads(b)["runs"]} for b in (v2b, fxb))
    chg = sorted(k for k in v2 if json.dumps(v2[k], sort_keys=True) != json.dumps(fx.get(k), sort_keys=True))
    if set(v2) != set(fx) or len(chg) != 77 or set(chg) & set(train):
        raise SystemExit(f"STOP: fix_v1 vs v2: same keys {set(v2) == set(fx)}, {len(chg)} changed rows (want 77), {len(set(chg) & set(train))} of them training rows")
    std, roots = {}, set()
    for nm, pfx in SOLD_STD[1].items():
        b = show(SOLD_STD[0], f"{CUT_ROUTING}/sold__{nm}.json")
        if not hashlib.sha256(b).hexdigest().startswith(pfx):
            raise SystemExit(f"sold__{nm}.json differs from 4f's relayed sha")
        r = json.loads(b)
        roots.add(r["extraction_root"])
        std |= {k: (a["staged_sha"], a["staged_inputs_file"], r["extraction_root"], a["meta_arm"]) for a in r["arms"] for k in a["keys"]}
    t196 = sorted(k for k in train if k in {e["key"] for e in inp["training_epochs"] if e["pool"] == "T196"})
    sold = sorted(k for k, u in train.items() if u["run"] in roots)
    bad = [k for k in sold if std.get(k) != (train[k]["staged_sha256"], train[k]["staged"], train[k]["run"], train[k]["arm"])]
    if bad or sold != t196:
        raise SystemExit(f"STOP: the uncut fit's S_old training inputs differ from d9's standard records by key: {len(bad)} unequal, e.g. {bad[:2]}; "
                         f"{len(set(sold) ^ set(t196))} T196 keys outside the standard roots")
    return {"inventory_b_fix_v1": {"pin": {"commit": IB_FIX[0], "path": IB_FIX[1], "sha256": IB_FIX[2]}, "rows_changed_vs_v2": len(chg),
                                   "changed_original_folders": sorted({v2[k]["original_folder"] for k in chg}), "changed_rows_in_training": 0},
            "sold_standard_records": {"commit": SOLD_STD[0], "records": sorted(SOLD_STD[1]), "t196_training_epochs": len(t196), "equal_by_key": len(sold), "mismatches": 0,
                                      "fields": "staged sha256, staged inputs file, extraction root, meta arm"}}


INV_WORLD = {"run1": "run1", "run2": "run2", "sim": "sim", "honest_a": "honest_a", "honest_b": "honest_b", "honest_c": "honest_c", "sim_honest": None, "relaunch": None}
SOLD_ATT = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/attest"


def _cut_jobs_b(git_, routing_commit, inv_b, refuse=()):
    """(b): every routed set whose planned folders are inventory_b's, plus d9's S_old sets (attestation-bound). Epochs come from inventory_b by key."""
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    ib = show(inv_b[0], inv_b[1])
    if hashlib.sha256(ib).hexdigest() != inv_b[2]:
        raise SystemExit("inventory_b differs from its pin")
    runs = json.loads(ib)["runs"]
    by_planned = {}
    for r in runs:
        by_planned.setdefault(r["uncut_planned"]["folder"], []).append(r)
    jobs, skipped = [], {}

    def _job(stem_, folders, originals, staged_set, root, shas, volume, batch, pins, meta_arm_of):
        eps = []
        for tok, fo in sorted(folders.items()):
            rs_ = by_planned.get(fo, [])
            if not rs_:
                raise SystemExit(f"{stem_} {fo}: no inventory_b run plans this folder")
            for r in rs_:
                w = INV_WORLD.get(r["world"], "?")
                if w is None or w == "?":
                    w = {"sim_honest": "simh_" + ("full" if "q4f" in fo else "nm360" if "nm360" in fo else "311")}.get(r["world"], r["world"])
                ma = meta_arm_of[tok]
                eps.append({"key": r["key"], "world": w, "cell": r["task"], "folder": fo, "staged": f"{staged_set}/inputs/{ma}.jsonl", "staged_sha256": shas[f"{ma}.jsonl"],
                            "run": root, "arm": ma, "tokens_before": r.get("tokens_before", r.get("full_len")), "eval_sha256": r.get("eval_sha256")})
        if volume != cf.VOLUME:
            raise SystemExit(f"{stem_}: not on the cutfull Volume")
        msg = {"volume": volume, "batch": batch, "original_folders": originals, "epochs": eps,
               "original_of": {folders[t]: originals[t] for t in folders}, "rule": cf.RULE + "; (b): originals = d9's attested closed list", "inventory_pin": {"commit": inv_b[0], "path": inv_b[1], "sha256": inv_b[2]}} | pins
        mb = json.dumps(msg, sort_keys=True).encode()
        kd = {"simh" if e["world"].startswith("simh_") else "scaleup" for e in eps}
        if len(kd) != 1:
            raise SystemExit(f"{stem_}: mixed sim-honest and other worlds")
        jobs.append((msg, hashlib.sha256(mb).hexdigest(), kd.pop(), volume))

    best = {}
    for p in git_("ls-tree", "-r", "--name-only", routing_commit, CUT_ROUTING).split():
        m = re.fullmatch(r".*/(.+?__.+?_cutfull)(?:_r([0-9]+))?\.json", p)
        if m and int(m[2] or 1) > best.get(m[1], (0, ""))[0]:
            best[m[1]] = (int(m[2] or 1), p)
    for stem_, (k, p) in sorted(best.items()):
        if any(stem_.split("__", 1)[1].removesuffix("_cutfull") == t for t in refuse):
            skipped[stem_] = "REFUSED by name (4f / 0d: 21g re-extraction not attested yet)"
            continue
        if any(t.startswith("only=") for t in refuse) and f"only={stem_.split('__', 1)[1].removesuffix('_cutfull')}" not in refuse:
            continue  # --refuse only=<target>: a follow-up of named sets only
        rb = show(routing_commit, p)
        rec = json.loads(rb)
        folders = {a["meta_arm"].split("@", 1)[0]: re.sub(r"_cutfull_v1_try[0-9]+$", "_cutfull_v1", rec["staged_set"]) + "__" + a["meta_arm"].split("@", 1)[0] for a in rec["arms"]}
        if not set(folders.values()) <= set(by_planned):
            continue  # not a (b) set ((a'') is built above)
        ap = p[:-5] + "_attest.json"
        try:
            atb = show(routing_commit, ap)
        except subprocess.CalledProcessError:
            skipped[stem_] = f"(b) revision {k}: no attestation pointer"
            continue
        at = json.loads(atb)
        if at.get("failing") != {} or at.get("revision") != k or rec.get("cut") != "full" or rec.get("revision") != k or not str(at.get("routing_record", "")).startswith(p):
            raise SystemExit(f"{stem_}: (b) attestation does not attest THIS routing record's revision with failing {{}} (Codex r11)")
        shas = {Path(a["staged_inputs_file"]).name: a["staged_sha"] for a in rec["arms"]}
        if any(Path(a["staged_inputs_file"]).parent.parent.name != rec["staged_set"] for a in rec["arms"]):
            raise SystemExit(f"{stem_}: staged inputs outside the record's staged set")
        _job(stem_, folders, rec["original_folders"], rec["staged_set"], rec["extraction_root"], shas, rec["volume"], rec["batch"],
             {"routing_pin": {"path": p, "commit": routing_commit, "sha256": hashlib.sha256(rb).hexdigest(), "revision": k},
              "attest_pin": {"path": ap, "commit": routing_commit, "sha256": hashlib.sha256(atb).hexdigest()},
              "record": {x: rec.get(x) for x in ("staged_set", "extraction_root", "extraction_record", "repro_gate", "step0_key_list", "chat_template_sha256", "stage_record")}},
             {a["meta_arm"].split("@", 1)[0]: a["meta_arm"] for a in rec["arms"]})
    bestA = {}
    for p in git_("ls-tree", "-r", "--name-only", routing_commit, SOLD_ATT).split():
        m = re.fullmatch(r".*/attestations_sold__(.+?_cutfull)(?:_r([0-9]+))?\.json", p)
        if m and int(m[2] or 1) > bestA.get(m[1], (0, ""))[0]:
            bestA[m[1]] = (int(m[2] or 1), p)
    for stem_, (k, p) in sorted(bestA.items()):
        if stem_.removesuffix("_cutfull") in refuse or (any(t.startswith("only=") for t in refuse) and f"only={stem_.removesuffix('_cutfull')}" not in refuse):
            skipped[f"sold {stem_}"] = "REFUSED by name (not a registered own_all_rs root, e.g. the rh1e4 relaunch), or outside an only= follow-up"
            continue
        ab = show(routing_commit, p)
        at = json.loads(ab)
        if at.get("failing") not in (0, {}, []) or at.get("revision") != k:
            raise SystemExit(f"sold {stem_}: the S_old attestation is not this revision with failing empty")
        cfb = at["cutfull"]
        sp_, sc_ = cfb["stage_record"].split(" @ ")
        sr = json.loads(show(sc_, sp_))
        shas = next(v["assembled"]["staged_inputs_sha256"] for v in sr.values() if isinstance(v, dict) and "assembled" in v)
        meta_arm_of = {Path(f).stem.split("@", 1)[0]: Path(f).stem for f in shas}
        folders = {t: fo for t, fo in cfb["folders"].items()}
        if not set(folders.values()) <= set(by_planned):
            skipped[f"sold {stem_}"] = "a planned folder outside inventory_b"
            continue
        _job(f"sold {stem_}", folders, at["original_folders"], cfb["staged_set"], cfb["extraction_root"], shas, cfb["volume"], "sold",
             {"routing_pin": {"path": p, "commit": routing_commit, "sha256": hashlib.sha256(ab).hexdigest(), "revision": k, "kind": "S_old attestation"},
              "attest_pin": {"path": p, "commit": routing_commit, "sha256": hashlib.sha256(ab).hexdigest()},
              "record": {"staged_set": cfb["staged_set"], "extraction_root": cfb["extraction_root"], "stage_record": cfb["stage_record"], "template_class": "in the S_old attestation, by key",
                         "repro_gate": at.get("repro_gate")}}, meta_arm_of)
    return jobs, skipped


def _launch(mode: str, inputs: str, fit_stamp: str = "", messages: str = "", t196_spec: str = "", census: str = "", dry: bool = False, followup: str = "", loo: bool = False,
            rs: bool = False, cutfull: bool = False, routing_commit: str = "", inventory_b: str = "", refuse: str = "", message_pins: str = "", uncut: bool = False):
    """Every local check; with dry=True nothing is spawned (the no-Modal dry run calls this directly). loo: 09 inclusion amendment 2's override."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    if loo and rs:
        raise SystemExit("--loo and --rs are exclusive")
    if git("status", "--porcelain", "--untracked-files=no") or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/own-all-loo" if (loo or rs) else "origin/aswin/p2-family-expansion"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    ib = Path(inputs).read_bytes()
    rel = str(Path(inputs).resolve().relative_to(REPO))
    if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True).stdout != ib:
        raise SystemExit("the inputs record is not committed as read")
    inp = json.loads(ib)
    if not inp["gate"]["pass"]:
        raise SystemExit("the held-out gate on the new fold lists did not pass")
    universe = set(inp["folds"]["by_main"])
    lists = {int(k): v for k, v in inp["folds"]["lists"].items()}
    train = {k: sorted(m for j, ms in lists.items() if j != k for m in ms) for k in range(5)}
    if any(hashlib.sha256(json.dumps(train[k]).encode()).hexdigest() != inp["folds"]["training_mains_sha256"][str(k)] for k in range(5)):
        raise SystemExit("training lists differ from the inputs record")
    main_of, origin = {}, {}
    for (c_, p_, s_), og in ((tuple(inp["pins"]["t196"]), "S_old"), (tuple(inp["pins"]["plus311"]), "S_new"), (tuple(inp["pins"]["nm360"]), "S_newmains")):
        b = subprocess.run(["git", "-C", str(REPO), "show", f"{c_}:{p_}"], capture_output=True, check=True).stdout
        if hashlib.sha256(b).hexdigest() != s_:
            raise SystemExit(f"{p_}: sha256 differs")
        main_of |= {c["key"]: c["main"] for c in json.loads(b)["cells"]}
        origin |= {c["key"]: og for c in json.loads(b)["cells"]}
    payload = {"folds": {"by_main": inp["folds"]["by_main"], "training_mains_sha256": inp["folds"]["training_mains_sha256"], "training_lists": train},
               "main_of": main_of, "_sha256": hashlib.sha256(ib).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    writer = {"commit": git("rev-parse", "HEAD"), "app": APP, "app_id": app.app_id or "", "stamp": stamp}
    lo = None
    if loo:  # 09 inclusion amendment 2: fold = origin; every main must have exactly one origin (checked, not assumed)
        og_of = {}
        for c_, m_ in main_of.items():
            og_of.setdefault(m_, set()).add(origin[c_])
        multi = sorted(m_ for m_, v in og_of.items() if len(v) != 1)
        if multi:
            raise SystemExit(f"STOP: {len(multi)} mains have more than one origin, e.g. {multi[:3]}")
        fom = {m_: ORIGINS.index(next(iter(v))) for m_, v in sorted(og_of.items())}
        if not universe <= set(fom):
            raise SystemExit("STOP: a training-universe main has no origin")
        set_mains = {o: sorted(m_ for m_, x in fom.items() if x == o) for o in range(3)}
        tr_o = {o: sorted(m_ for m_ in universe if fom[m_] != o) for o in range(3)}
        ep = inp["training_epochs"]
        launches = {o: {} for o in range(3)}
        for e in ep:  # each model's training launches (first path component of the epoch key), by world: the eligibility file's assert
            if e["world"] in ("run1", "honest_a") and main_of[e["cell"]] in universe:
                for o in range(3):
                    if fom[main_of[e["cell"]]] != o:
                        launches[o].setdefault(e["key"].split("/", 1)[0], {}).setdefault(e["world"], 0)
                        launches[o][e["key"].split("/", 1)[0]][e["world"]] += 1
        lo = {"fold_of_main": fom, "set_mains": {str(o): v for o, v in set_mains.items()},
              "training_mains_sha256": {str(o): hashlib.sha256(json.dumps(v).encode()).hexdigest() for o, v in tr_o.items()},
              "training_lists": {str(o): v for o, v in tr_o.items()}, "training_launches": {str(o): v for o, v in launches.items()},
              "origins": list(ORIGINS), "mains_with_two_origins": 0,
              "dropped_mains_set_o_planned": {str(o): len(set(tr_o[o]) & set(set_mains[o])) for o in range(3)}}
        lo["_sha256"] = hashlib.sha256(json.dumps(lo, sort_keys=True).encode()).hexdigest()
        if any(lo["dropped_mains_set_o_planned"].values()):
            raise SystemExit("STOP: a model's training list contains a main of its held-out set")
        payload["loo"] = lo
        universe = set(fom)  # under --loo every main with an origin is mapped (fold = origin); fold -1 = no origin
        pfx = "own_all_loo"
    elif rs:  # 09 inclusion amendment 3: the committed re-split fold list (every main of the four sets), read from HEAD by path and checked
        fb = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{RS_FOLDS}"], capture_output=True, check=True).stdout
        fl = json.loads(fb)
        bym = fl["by_main"]
        if fl["asserts"]["mains_without_fold"] or fl["asserts"]["mains_not_in_exactly_one_fold"] or set(main_of.values()) - set(bym):
            raise SystemExit("STOP: the re-split fold list does not cover every main of the committed cell lists, or its asserts failed")
        if any(sorted(m for m, v in bym.items() if v[0] == k) != fl["lists"][str(k)] for k in range(5)):
            raise SystemExit("STOP: the re-split lists differ from by_main")
        tr_k = {k: sorted(m for m in universe if bym[m][0] != k) for k in range(5)}  # training = own_all v1's universe mains (run1 + honest_a epochs), re-split
        in_fold = [m for m in universe if m in set(tr_k[bym[m][0]])]
        pos_fail = [m for m in universe if any(m not in set(tr_k[g]) for g in range(5) if g != bym[m][0])]
        neg = {k: len(set(fl["lists"][str(k)]) & set(tr_k[k])) for k in range(5)}
        if in_fold or pos_fail or any(neg.values()):
            raise SystemExit("STOP: the held-out gate on the re-split training lists failed")
        fb2 = subprocess.run(["git", "-C", str(REPO), "show", f"{SP493_FACTS[0]}:{SP493_FACTS[1]}"], capture_output=True, check=True).stdout
        if not hashlib.sha256(fb2).hexdigest().startswith(SP493_FACTS[2]):
            raise SystemExit("the S_paper493 facts file differs from its pin")
        sp_cells = {r["key"]: r["main"] for r in json.loads(fb2)["rows"] if r["key"] not in SP493_OUT}
        if {c for c, x in fl["cell_set"].items() if x == "S_paper493"} != set(sp_cells) or set(sp_cells) & set(main_of) or set(sp_cells.values()) - set(bym):
            raise SystemExit("STOP: S_paper493's cells differ from the fold list's, collide with another set's cell keys, or have unfolded mains")
        main_of |= sp_cells
        origin |= dict.fromkeys(sp_cells, "S_paper493")
        rs_p = {"by_main": bym, "fold_list": {"path": RS_FOLDS, "commit": git("log", "-1", "--format=%H", "--", RS_FOLDS), "sha256": hashlib.sha256(fb).hexdigest()},
                "training_lists": {str(k): v for k, v in tr_k.items()}, "training_mains_sha256": {str(k): hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in tr_k.items()},
                "gate": {"in_fold_violations": len(in_fold), "positive_control_failures": len(pos_fail), "negative_control_counts": neg,
                         "positive_control_counts": {k: len(tr_k[k]) for k in range(5)}, "pass": True}}
        rs_p["_sha256"] = hashlib.sha256(json.dumps(rs_p, sort_keys=True).encode()).hexdigest()
        payload["rs"] = rs_p
        universe = set(bym)  # every main of the four sets has a fold: no fold -1 unit
        pfx = "own_all_rs"
        if uncut:  # own_all_rs_uncut_L48 (09 uncut §6 / §7, lead RULE @ 36b9b47df): the same pinned folds, uncut run1 + honest_a of the three sets
            if not (routing_commit and inventory_b):
                raise SystemExit("--uncut needs --routing-commit and --inventory-b")
            ib_u, fit_rc, fit_chain = tuple(inventory_b.split(":", 2)), routing_commit, None
            if mode != "fit":  # Codex r15: the fit's FROZEN training identity (its own routing commit, inventory and chain commit), never this run's scoring routing
                fr_ = json.loads(git("show", f"HEAD:{(HERE / 'results' / f'own_all_rs_uncut_fit_{fit_stamp}.json').relative_to(REPO)}"))["rs_uncut"]
                fit_rc, ib_u, fit_chain = fr_["routing_commit"], (fr_["inventory_b"]["commit"], fr_["inventory_b"]["path"], fr_["inventory_b"]["sha256"]), fr_["training_report"]["chain_commit"]
            by_root, urep = _uncut_training(git, fit_rc, ib_u, inp, fit_chain)
            rsfit = json.loads((HERE / "results" / f"own_all_rs_fit_{RS_FIT[0]}.json").read_text())["fit"]["fit_record_sha256"]
            if not rsfit.startswith(RS_FIT[1]):
                raise SystemExit("own_all_rs's git fit record does not name the pinned Volume fit record")
            un = {"label": UNCUT_LABEL.format(n=len(urep["template_class_render_changed"])), "rs_fit_record_sha256": rsfit, "routing_commit": fit_rc,
                  "inventory_b": {"commit": ib_u[0], "path": ib_u[1], "sha256": ib_u[2]}, "training_report": urep}
            un["_sha256"] = hashlib.sha256(json.dumps(un, sort_keys=True).encode()).hexdigest()
            if mode != "fit" and un["_sha256"] != fr_["_sha256"]:
                raise SystemExit("STOP: the rebuilt own_all_rs_uncut training identity differs from the fit record's")
            uchecks = _uncut_checks(git, by_root, ib_u, inp) if mode == "fit" else {}
            payload["rs_uncut"] = un
            pfx = "own_all_rs_uncut"
    else:
        pfx = "own_all"
    if mode == "fit" and rs and uncut:
        print(json.dumps({"mode": "fit --rs --uncut", "stamp": stamp, "roots": len(by_root), "epochs": sum(len(v) for v in by_root.values()),
                          "template_class_render_changed": len(urep["template_class_render_changed"]), "no_policy_call_refs": urep["no_policy_call_refs_listed_not_missing"],
                          "registered_eval_sha_checked": urep["registered_eval_sha_checked"], "uncut_sha256": payload["rs_uncut"]["_sha256"], "tol_refit": TOL_REFIT,
                          "chain_commit": urep["chain_commit"], "checks": uchecks}))
        if dry:
            return
        names = sorted(by_root)
        rres = list(rows_uncut.with_options(volumes={"/out": OUT, "/b": modal.Volume.from_name(cf.VOLUME, version=2).read_only()}).starmap(
            [(stamp, nm_, by_root[nm_], {e["cell"]: rs_p["by_main"][main_of[e["cell"]]][0] for e in by_root[nm_]}) for nm_ in names]))
        payload["rs_uncut"]["rows"] = {rd: {nm_: r_[rd]["sha256"] for nm_, r_ in zip(names, rres, strict=True)} for rd in ("R", "C")}
        gates = [c.get() for c in [gate_uncut.spawn(stamp, payload, rd) for rd in ("R", "C")]]  # the barrier: both gates pass (else STOP) before any fit
        gate_shas = {g["readout"]: g["gate_sha256"] for g in gates}
        calls = [fit_uncut.spawn(stamp, payload, rd, gate_shas) for rd in ("R", "C")]  # two H100s in parallel (4f / Aswin: cap waived)
        parts = [c.get() for c in calls]
        res = finalize_uncut.remote(stamp, payload, parts, gate_shas, uchecks)
        rec = {"label": payload["rs_uncut"]["label"] + "; fit: rows by uncut root, 30 models (own_all_rs{,25,10}_uncut_L48 x R, C x 5 pinned folds), the in-job STOP, the GATE "
                        "(refit-match of the new code on own_all_rs's REGISTERED rows vs its registered weights) and the descriptive cut-32768 refit; keys, shas, counts and gate results only",
               "stamp (PT)": stamp, "writer": writer, "inputs": {"path": rel, "sha256": payload["_sha256"]}, "rs": {k: v for k, v in rs_p.items() if k != "by_main"} | {"mains": len(rs_p["by_main"])},
               "rs_uncut": payload["rs_uncut"], "rows": {nm_: r_ for nm_, r_ in zip(names, rres, strict=True)}, "gates": gates, "checks": uchecks, "fit": res}
        (HERE / "results" / f"own_all_rs_uncut_fit_{stamp}.json").open("x").write(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"refit_match_pass": res["gate"]["refit_match_pass"], "all_certified": res["gate"]["all_certified"],
                          "fits": [(f["variant"], f["readout"], f["fold"], f["rows"], f["rows_past_32768"], f["certified"]) for f in res["fits"]]}, default=float))
        return
    if mode == "fit" and rs:
        print(json.dumps({"mode": "fit --rs", "stamp": stamp, "fold_list": rs_p["fold_list"], "training_mains": {k: len(v) for k, v in tr_k.items()},
                          "gate": rs_p["gate"], "rs_sha256": rs_p["_sha256"], "tol_refit": TOL_REFIT}))
        if dry:
            return
        res = fit_rs.remote(stamp, payload)
        rec = {"label": RS_LABEL + "; fit: 30 models (own_all_rs{,25,10}_L48 x R, C x 5 re-split folds) from own_all's sha-pinned rows, in-job STOP gate and the refit-match "
                                   "control (own_all v1's 30 fits on v1's folds vs its registered weights); keys, shas, counts and gate results only",
               "stamp (PT)": stamp, "writer": writer, "inputs": {"path": rel, "sha256": payload["_sha256"]}, "rs": {k: v for k, v in rs_p.items() if k != "by_main"} | {"mains": len(rs_p["by_main"])},
               "fit": res}
        (HERE / "results" / f"own_all_rs_fit_{stamp}.json").open("x").write(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"refit_match_pass": res["gate"]["refit_match_pass"], "all_certified": res["gate"]["all_certified"],
                          "fits": [(f["variant"], f["readout"], f["fold"], f["rows"], f["mains"], f["certified"]) for f in res["fits"]]}, default=float))
        return
    if mode == "fit" and loo:
        print(json.dumps({"mode": "fit --loo", "stamp": stamp, "set_mains": {ORIGINS[o]: len(v) for o, v in set_mains.items()},
                          "training_mains": {ORIGINS[o]: len(v) for o, v in tr_o.items()}, "dropped_planned": lo["dropped_mains_set_o_planned"],
                          "training_launches": {ORIGINS[o]: sorted(v) for o, v in launches.items()}, "loo_sha256": lo["_sha256"], "tol_refit": TOL_REFIT}))
        if dry:
            return
        res = fit_loo.remote(stamp, payload)
        rec = {"label": "own_all_LOO_L48 fit (09 inclusion amendment 2 @ 8b9ca793f; POST HOC): 6 models (R, C x 3 origins) from own_all's sha-pinned rows, in-job STOP gate and the "
                        "refit-match control (own_all_L48 refit vs its registered weights); keys, shas, counts and gate results only",
               "stamp (PT)": stamp, "writer": writer, "inputs": {"path": rel, "sha256": payload["_sha256"]}, "loo": lo, "fit": res}
        (HERE / "results" / f"own_all_loo_fit_{stamp}.json").open("x").write(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"refit_match_pass": res["gate"]["refit_match_pass"], "all_certified": res["gate"]["all_certified"], "dropped": res["gate"]["dropped_mains_set_o"],
                          "fits": [(f["variant"], f["readout"], f["fold"], f["rows"], f["mains"], f["certified"]) for f in res["fits"]]}, default=float))
        return
    if mode == "fit":
        cell_fold = {c: inp["folds"]["by_main"][m][0] if m in universe else -1 for c, m in main_of.items()}
        ep = inp["training_epochs"]
        plan = {p: [e for e in ep if (e["pool"] == "T196") == (p == "T196")] for p in ("T196", "scaleup")}
        print(json.dumps({"mode": "fit", "stamp": stamp, "epochs": {p: len(v) for p, v in plan.items()}, "universe": len(universe), "tol_refit": TOL_REFIT}))
        if dry:
            return
        r = {p: rows.with_options(volumes={"/out": OUT, "/b": modal.Volume.from_name(VOL_T196 if p == "T196" else VOL_SU, version=2).read_only()}).spawn(stamp, p, v, cell_fold)
             for p, v in plan.items()}
        rr = {p: c.get() for p, c in r.items()}
        res = fit.remote(stamp, payload)
        rec = {"label": "own_all_L48 fit: rows, 10 models, in-job STOP gate and refit-match control; keys, shas, counts and gate results only",
               "stamp (PT)": stamp, "writer": writer, "inputs": {"path": rel, "sha256": payload["_sha256"]}, "rows": rr, "fit": res}
        (HERE / "results" / f"own_all_fit_{stamp}.json").open("x").write(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"rows": rr, "refit_match_pass": res["gate"]["refit_match_pass"], "all_certified": res["gate"]["all_certified"],
                          "fits": [(f["variant"], f["readout"], f["fold"], f["rows"], f["certified"]) for f in res["fits"]]}, default=float))
        return
    if mode == "coverage":  # 3a (c), by key: every fold-list main of the four sets has scored units in its set's arms / pools; gaps listed with reasons
        if not rs:
            raise SystemExit("--mode coverage is an --rs check")
        fl = json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{RS_FOLDS}"], capture_output=True, check=True).stdout)
        frec = json.loads((HERE / "results" / f"{pfx}_fit_{fit_stamp}.json").read_text())
        folders, recs_, kinds_by_sha = [], [], {}
        for sp_ in followup.split(","):
            b = Path(sp_).read_bytes()
            if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{Path(sp_).resolve().relative_to(REPO)}"], capture_output=True).stdout != b:
                raise SystemExit(f"{sp_}: not committed as read")
            r_ = json.loads(b)
            if r_["fit_record_sha256"] != frec["fit"]["fit_record_sha256"]:
                raise SystemExit(f"{sp_}: another fit")
            recs_.append({"path": str(Path(sp_).resolve().relative_to(REPO)), "sha256": hashlib.sha256(b).hexdigest()})
            folders += [(x["kind"], fo, x["manifest_sha256"]) for r in r_["results"].values() for fo, x in r["folders"].items()]
            kinds_by_sha |= {k.split("|", 1)[1]: k.split("|", 1)[0] for k in r_["results"]}
        sup = set()  # superseded folders (append-only SUPERSEDED NOTEs, committed): excluded from coverage, never read
        for np_ in [x for x in census.split(",") if x] if census else []:
            nb_ = Path(np_).read_bytes()
            if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{Path(np_).resolve().relative_to(REPO)}"], capture_output=True).stdout != nb_:
                raise SystemExit(f"{np_}: superseded NOTE not committed as read")
            sup |= {fo for v in json.loads(nb_)["superseded_folders"].values() for fo in v}
        folders = [f_ for f_ in folders if f_[1] not in sup]
        eps = []  # every scored epoch, from the messages those records name (sha-checked)
        for mp in [x for x in Path(messages).read_text().split() if x]:
            mb = Path(mp).read_bytes()
            h = hashlib.sha256(mb).hexdigest()
            if h not in kinds_by_sha:
                raise SystemExit(f"{mp}: no given score record scored this message")
            kd = kinds_by_sha.pop(h)
            eps += [(kd, e) for e in json.loads(mb)["epochs"] if e["folder"] not in sup]
        if set(kinds_by_sha) - {SPEC_SHA, OWNS_SHA}:
            raise SystemExit(f"{len(set(kinds_by_sha) - {SPEC_SHA, OWNS_SHA})} scored messages were not given")
        print(json.dumps({"mode": "coverage", "folders": len(folders), "score_records": recs_, "message_epochs": len(eps)}))
        if dry:
            return
        cu = coverage_units.remote(fit_stamp, folders)
        if cu["problems"]:
            raise SystemExit(f"STOP: {cu['problems'][:3]}")
        units = collections.defaultdict(lambda: collections.Counter())  # (set, main) -> world -> mapped units (R + C)
        fm1, runs_zero = 0, collections.defaultdict(list)
        for fo, x in cu["folders"].items():
            for rd, pr in x["per_run"].items():
                for u, (mp_, f1) in pr.items():
                    w, c = u.split("|", 1)
                    fm1 += f1
                    st = _folder_set(fo, c, origin)
                    if st:
                        units[(st, main_of[c])][w] += mp_
            for rd, lst in x["runs_without_rows"].items():
                for u in lst:
                    st = _folder_set(fo, u.split("|", 1)[1], origin)
                    if st:
                        runs_zero[(st, main_of[u.split("|", 1)[1]])].append(f"{rd}:{u} in {fo}")
        epm = collections.defaultdict(list)
        for kind, e in eps:
            st = _folder_set(e["folder"], e["cell"], origin)
            if st:
                epm[(st, main_of[e["cell"]])].append(f"{e['world']}|{e['cell']}")
        if t196_spec:  # T196's epochs come from pass 1's pinned spec (all worlds), as score mode builds them
            sb_ = Path(t196_spec).read_bytes()
            if hashlib.sha256(sb_).hexdigest() != SPEC_SHA:
                raise SystemExit("--t196-spec: sha256 differs from pass 1's spec")
            for w, c in json.loads(sb_)["map"].values():
                epm[("S_old", main_of[c])].append(f"{w}|{c}")
        set_mains, cells_of = collections.defaultdict(set), collections.defaultdict(set)
        for c, st in fl["cell_set"].items():
            set_mains[st].add(main_of[c])
            cells_of[(st, main_of[c])].add(c)
        out = {}
        for st in ("S_old", "S_new", "S_newmains", "S_paper493"):
            gaps = {}
            for m in sorted(set_mains[st]):
                tot = sum(units[(st, m)].values())
                pend = sorted(c for c in cells_of[(st, m)] if c in RS_PENDING)
                if tot == 0 and pend:
                    gaps[m] = {"reason": RS_PENDING[pend[0]], "pending_cells": pend, "runs": sorted(set(epm[(st, m)]))[:8]}
                elif tot == 0:
                    gaps[m] = {"reason": ("runs routed and scored, 0 units (e.g. zero-call / empty runs): " + ", ".join(sorted(set(runs_zero[(st, m)]))[:6])) if epm[(st, m)]
                               else "no routed run of this main in the set's scored arms or pools", "runs": sorted(set(epm[(st, m)]))[:8]}
            out[st] = {"fold_list_mains": len(set_mains[st]), "mains_with_scored_units": len(set_mains[st]) - len(gaps), "gaps": gaps,
                       "worlds": dict(collections.Counter(w for m in set_mains[st] for w, n in units[(st, m)].items() if n > 0))}
        not_in_list = sorted({m for (st, m) in units if m not in fl["by_main"]})
        rec = {"label": RS_LABEL + "; 3a (c) coverage assert by key: every fold-list main of the four sets vs scored units in its set's arms and pools; keys and counts only",
               "stamp (PT)": stamp, "writer": writer, "fit_record_sha256": frec["fit"]["fit_record_sha256"], "fold_list": {"path": RS_FOLDS, "sha256": hashlib.sha256(
                   subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{RS_FOLDS}"], capture_output=True, check=True).stdout).hexdigest()},
               "score_records": recs_, "fold_minus1_units": fm1, "scored_mains_not_in_fold_list": not_in_list, "sets": out, "pending_restage": RS_PENDING,
               "pass": fm1 == 0 and not not_in_list, "superseded_folders_excluded": sorted(sup), "superseded_notes": census,
               "note": "a gap is a fold-list main with no scored unit in its set; each is listed with its reason, never silently missing"}
        (HERE / "results" / f"{pfx}_coverage_assert_{stamp}.json").open("x").write(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"pass": rec["pass"], "fold_minus1_units": fm1, "sets": {st: [v["fold_list_mains"], v["mains_with_scored_units"], len(v["gaps"])] for st, v in out.items()}}))
        return
    if mode == "verify":  # 09's row-level artifact check + the end-to-end negative, over the given score records (comma-separated)
        frec = json.loads((HERE / "results" / f"{pfx}_fit_{fit_stamp}.json").read_text())
        fit_sha = frec["fit"]["fit_record_sha256"]
        folders, recs_ = [], []
        for sp_ in followup.split(","):
            b = Path(sp_).read_bytes()
            if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{Path(sp_).resolve().relative_to(REPO)}"], capture_output=True).stdout != b:
                raise SystemExit(f"{sp_}: not committed as read")
            recs_.append({"path": str(Path(sp_).resolve().relative_to(REPO)), "sha256": hashlib.sha256(b).hexdigest()})
            fx = [(x["kind"], fo, x["manifest_sha256"], x.get("batch", "")) for r in json.loads(b)["results"].values() for fo, x in r["folders"].items()]
            if any(bool(f_[3]) != cutfull for f_ in fx) or (cutfull and not rs):  # one line per verify (Codex r9): --cutfull verifies only cutfull records, and only under --rs
                raise SystemExit(f"{sp_}: a {'registered' if cutfull else 'cutfull'}-line folder in a {'cutfull' if cutfull else 'registered'} verify")
            folders += fx
        s4c, s4p = inp["pins"]["s4_source_map"]
        payload["s4_orphan_records"] = sorted(x["record"] for x in json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"{s4c}:{s4p}"], capture_output=True, check=True).stdout)["outside_source_map"])
        mp = [x for x in Path(messages).read_text().split() if x][0]
        mb = Path(mp).read_bytes()
        m = json.loads(mb)
        m["epochs"] = m["epochs"][:3]
        print(json.dumps({"mode": "verify", "folders": len(folders), "negative_message": Path(mp).name, "negative_epochs": len(m["epochs"])}))
        if dry:
            return
        v = verify_rows.remote(fit_stamp, fit_sha, folders, loo, rs, uncut)
        n = negative_e2e.with_options(volumes={"/out": OUT, "/work": WORK, "/b": modal.Volume.from_name(m["volume"], version=2).read_only()}).remote(
            fit_stamp, fit_sha, payload, m, hashlib.sha256(mb).hexdigest(), writer)
        rec = {"label": ("own_all_LOO_L48" if loo else "own_all_rs_uncut_L48" if (rs and uncut) else "own_all_rs_L48" if rs else "own_all") + " artifact check (09): every emitted row's model / preprocessing / training-manifest sha256 vs the fit record, and the end-to-end misrouting negative; keys and counts only",
               "stamp (PT)": stamp, "writer": writer, "score_records": recs_,
               "fit_record": f"Volume fit record {fit_sha[:8]}…; git file {hashlib.sha256((HERE / 'results' / f'{pfx}_fit_{fit_stamp}.json').read_bytes()).hexdigest()[:8]}…" + ("" if (loo or rs) else " (@ 66800141f)"),
               **({} if (loo or rs) else {"fit_record_label_note": "the git fit record's label says '10 models' (written before amendment 3's fractions were added); it holds 30 fits (own_all, own_all25, own_all10 x R, C x 5 folds)"}),
               "rows": v, "negative_e2e": n, "pass": not v["problems"] and all(c["mismatch"] == 0 for c in v["counts"].values()) and n["failed_closed"] and not n.get("test_outputs_left")}
        (HERE / "results" / f"{pfx}_verify_{stamp}.json").open("x").write(json.dumps(rec, indent=1, default=float))
        print(json.dumps({"pass": rec["pass"], "counts": v["counts"], "problems": v["problems"][:5], "negative": {k: n.get(k) for k in ("failed_closed", "error", "test_outputs_left")}}))
        return
    if mode != "score" or not fit_stamp:
        raise SystemExit("--mode fit | score | verify (score / verify need --fit-stamp)")
    frec = json.loads((HERE / "results" / f"{pfx}_fit_{fit_stamp}.json").read_text())
    fit_sha = frec["fit"]["fit_record_sha256"]
    if loo and frec["loo"]["_sha256"] != lo["_sha256"]:
        raise SystemExit("the LOO fold source differs from the fit's")
    if rs and frec["rs"]["_sha256"] != payload["rs"]["_sha256"]:
        raise SystemExit("the re-split fold source differs from the fit's")
    if not frec["fit"]["gate"]["refit_match_pass"] or not frec["fit"]["gate"]["all_certified"] or frec["fit"]["inputs_sha256"] != payload["_sha256"]:
        raise SystemExit("the fit is not usable: refit-match control failed, an uncertified model, or another inputs record")
    db = subprocess.run(["git", "-C", str(REPO), "show", f"{DUMPS[0]}:{DUMPS[1]}"], capture_output=True, check=True).stdout
    full = [d for d in json.loads(db)["dumps"] if d["world"] == "simh_full"] if hashlib.sha256(db).hexdigest() == DUMPS[2] else []
    if {d["counted_keys"]["sha256"] for d in full} != {COUNTED_SHA}:
        raise SystemExit("dumps.json's simh_full counted-key pin differs")
    ck = full[0]["counted_keys"]
    cb = subprocess.run(["git", "-C", str(REPO), "show", f"{ck['commit']}:{ck['path']}"], capture_output=True, check=True).stdout
    if hashlib.sha256(cb).hexdigest() != ck["sha256"]:
        raise SystemExit("the counted-key file differs from its pin")
    payload["a25_counted_tasks"] = [x["task"] for x in json.loads(cb)["keys"] if x.get("usable") is True and x.get("world") == "simh_full"]
    recs = {}
    for p in (HERE / "results").glob("pass2_*.json"):
        if "TEST" not in p.name and "note" not in p.name and "postcheck" not in p.name:
            recs[json.loads(p.read_text())["message_sha256"]] = ("scaleup", p.name)
    for p in (HERE / "results").glob("simh_scaleup_*.json"):
        if "note" not in p.name and "pins" not in p.name:
            recs[json.loads(p.read_text())["pins"]["message"]["sha256"]] = ("simh", p.name)
    for mp_ in [x for x in message_pins.split(",") if x]:  # a message COMMITTED on aswin/p2-family-expansion before its own record lands: <commit>:<path>:<sha256>:<scaleup|simh>
        c_, pth, sh_, kd_ = mp_.split(":")
        if kd_ not in ("scaleup", "simh") or subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", c_, "origin/aswin/p2-family-expansion"]).returncode != 0:
            raise SystemExit(f"--message-pins {mp_}: bad kind, or the commit is not on origin/aswin/p2-family-expansion (exact ref)")
        mb_ = subprocess.run(["git", "-C", str(REPO), "show", f"{c_}:{pth}"], capture_output=True, check=True).stdout
        if hashlib.sha256(mb_).hexdigest() != sh_:
            raise SystemExit(f"--message-pins {pth} @ {c_}: sha256 differs from the pin")
        fos = {e["folder"] for e in json.loads(mb_)["epochs"]}
        if (kd_ == "simh") != all(bool(SIMH_BATCH.match(f_)) and f_.endswith("__sim_honest") for f_ in fos) or (sh_ in recs and recs[sh_][0] != kd_):  # Codex r12
            raise SystemExit(f"--message-pins {pth}: kind {kd_} disagrees with its folders, or with the message's committed record")
        recs.setdefault(sh_, (kd_, f"{pth} @ {c_} (message pin)"))
    jobs, cut_skipped = [], {}
    if cutfull:  # the cutfull line (09 Append 41 amd 2): built from d9's attested routing records, an addendum under the same fit; never the registered line
        if not (rs and (followup or uncut) and routing_commit and mode == "score"):
            raise SystemExit("--cutfull: --rs score mode with --followup <parent score record> (or --uncut, the probe's own first record) and --routing-commit only")
        ib_ = tuple(inventory_b.split(":", 2)) if inventory_b else None  # "<commit>:<path>:<sha256>"; with it, ONLY (b) is built (Codex r11 scope)
        jobs, cut_skipped = _cut_jobs(git, routing_commit, ib_, tuple(x for x in refuse.split(",") if x))
        if ib_ and not uncut:
            a2 = {h for _, h, _, _ in _cut_jobs(git, routing_commit)[0]}
            jobs = [j for j in jobs if j[1] not in a2]
        pfx = "own_all_rs_uncut_cutfull" if uncut else "own_all_rs_cutfull"
    for mp in [x for x in Path(messages).read_text().split() if x] if not cutfull else []:
        mb = Path(mp).read_bytes()
        h = hashlib.sha256(mb).hexdigest()
        if h not in recs:
            raise SystemExit(f"{mp}: no committed record names this message")
        m = json.loads(mb)
        jobs.append((m, h, recs[h][0], m["volume"]))
    parent = None
    if followup:  # an ADDENDUM: only the listed messages (roots routed after the parent run), under the same fit; T196 / A25 are not re-run
        pb = Path(followup).read_bytes()
        prel = str(Path(followup).resolve().relative_to(REPO))
        if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{prel}"], capture_output=True).stdout != pb or json.loads(pb)["fit_record_sha256"] != fit_sha:
            raise SystemExit("--followup: the parent score record is not committed as read, or it used another fit")
        done = {k.split("|", 1)[1] for k in json.loads(pb)["results"]}
        if any(h in done for _, h, _, _ in jobs):
            raise SystemExit("--followup: a message the parent record already scored")
        parent = {"path": prel, "sha256": hashlib.sha256(pb).hexdigest()}
    elif cutfull and uncut:  # own_all_rs_uncut's first cutfull record: no registered T196 / A25 jobs on the cutfull line
        parent = None
    else:
        sb = Path(t196_spec).read_bytes()  # T196: every epoch of pass 1's spec (all worlds), one folder per world (sim rows apart: pass 1's firewall)
        if hashlib.sha256(sb).hexdigest() != SPEC_SHA:
            raise SystemExit("--t196-spec: sha256 differs from pass 1's spec")
        sp = json.loads(sb)
        t196_eps = [{"key": k, "world": w, "cell": c, "run": sp["run_of"][k], "arm": sp["arm_of"][k], "folder": f"t196__{w}"} for k, (w, c) in sp["map"].items()]
        jobs.append(({"epochs": t196_eps, "volume": VOL_T196}, SPEC_SHA, "t196", VOL_T196))
        jobs.append(({"epochs": [], "volume": VOL_Q3}, OWNS_SHA, "a25simh", VOL_Q3))  # epochs built in-job from run_owns.py's pinned prep spec
    # b6a6b6b48 §3: coverage census of the calibration pools' mains against the training universe, per origin and pool, committed before scoring.
    # Pool membership is the admitted one: staging exclusions are absent from the messages; setup-failure / unusable-honest keys and the S4 source
    # map's orphans (09: authoritative for every arm and pool) are dropped by key; H_A is honest_a on H_sim's cells (25f), with t309 / t325 / t389's
    # 25f selected attempts; S_old's H_sim is the 149 counted A25 runs, its H_A / H_B pass 1's spec epochs on those cells / all its honest_b.
    s4c, s4p = inp["pins"]["s4_source_map"]
    s4b = subprocess.run(["git", "-C", str(REPO), "show", f"{s4c}:{s4p}"], capture_output=True, check=True).stdout
    payload["s4_orphan_records"] = sorted(x["record"] for x in json.loads(s4b)["outside_source_map"])
    orphans = set(payload["s4_orphan_records"])
    bad = {k for j in jobs for p_ in j[0].get("provenance", []) for k in ((p_.get("setup_failures") or {}).get("keys") or []) + ((p_.get("unusable_honest") or {}).get("keys") or [])}
    adm = [(e, j[2]) for j in jobs for e in j[0]["epochs"] if e["key"] not in bad and e["key"].removesuffix("#e0") not in orphans]
    addendum = bool(parent or cutfull)  # an addendum or a cutfull record: no registered A25 primary_v3 rows of its own
    hsim_cells = {e["cell"] for e, k in adm if k == "simh"} | (set() if addendum else set(payload["a25_counted_tasks"]))
    pools = {"H_sim": {main_of[c] for c in hsim_cells},
             "H_A": {main_of[e["cell"]] for e, _ in adm if e["world"] == "honest_a" and e["cell"] in hsim_cells},
             "H_B": {main_of[e["cell"]] for e, _ in adm if e["world"] == "honest_b"}}
    cell_of = {m: c for c, m in main_of.items()}
    ogs = ("S_old", "S_new", "S_newmains") + (("S_paper493",) if rs else ())
    eog = (lambda fo, c: "S_paper493" if fo.startswith("sp493_") else origin[c]) if rs else (lambda fo, c: origin[c])  # noqa: E731
    if rs:  # pools as (origin, main) by EPOCH: P_rep mains are shared with T196 / +311, so a main alone does not name its origin
        pe = {"H_sim": [(e, k) for e, k in adm if k == "simh"], "H_A": [(e, k) for e, k in adm if e["world"] == "honest_a" and e["cell"] in hsim_cells],
              "H_B": [(e, k) for e, k in adm if e["world"] == "honest_b"]}
        pools_og = {pl: {og: {main_of[e["cell"]] for e, _ in es if eog(e["folder"], e["cell"]) == og} | ({main_of[t] for t in payload["a25_counted_tasks"]} if (pl == "H_sim" and og == "S_old" and not addendum) else set())
                         for og in ogs} for pl, es in pe.items()}
    cen = {"label": ("own_all_rs_L48 coverage census (09 inclusion amendment 3 §4): calibration-pool mains vs the re-split fold list (every main folded), per origin and pool; keys and counts only" if rs else
                     "own_all_LOO_L48 coverage census (09 inclusion amendment 2; b6a6b6b48 §3 form): calibration-pool mains vs the LOO fold map (fold = origin), per origin and pool; keys and counts only" if loo else
                     "own_all coverage census (09 b6a6b6b48 §3): calibration-pool mains vs own_all's training universe, per origin and pool; keys and counts only")
                    + ("; APPEND-ONLY ADDENDUM: the new roots' rows only; pools with no new rows are unchanged from the parent run's census" if parent else ""),
           "inputs_sha256": payload["_sha256"], "fit_stamp": fit_stamp, "parent_score_record": parent,
           "pools": ({pl: {og: {"mains": len(ms), "in_universe": len(ms & universe), "out_of_universe_fold_minus1": sorted(ms - universe)} for og, ms in v.items()}
                      for pl, v in pools_og.items()} if rs else
                     {pl: {og: {"mains": sum(origin[cell_of[m]] == og for m in ms), "in_universe": sum(origin[cell_of[m]] == og and m in universe for m in ms),
                                "out_of_universe_fold_minus1": sorted(m for m in ms if origin[cell_of[m]] == og and m not in universe)}
                           for og in ("S_old", "S_new", "S_newmains")} for pl, ms in pools.items()})} | ({"p_new_rule": RS_P_NEW} if rs else {})
    for pl in cen["pools"].values():
        for og, x in pl.items():
            x["withheld_no_mapped_mains"] = x["in_universe"] == 0 and (x["mains"] > 0 or not addendum)  # empty pools included (an addendum only reports new rows)
    cb_ = json.dumps(cen, indent=1)
    print(json.dumps({"mode": "score", "messages": len(jobs), "by_kind": {k: sum(j[2] == k for j in jobs) for k in ("scaleup", "simh", "t196", "a25simh")}, "fit_stamp": fit_stamp,
                      "epochs": sum(len(j[0]["epochs"]) for j in jobs), "fold_minus1_epochs": sum(1 for j in jobs for e in j[0]["epochs"] if main_of[e["cell"]] not in universe),
                      "census": {pl: {og: [x["mains"], x["in_universe"]] for og, x in v.items()} for pl, v in cen["pools"].items()}}))
    if dry:
        Path(census).open("x").write(cb_)  # commit it, then launch scoring with the same --census path
        return
    crel = str(Path(census).resolve().relative_to(REPO))
    if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{crel}"], capture_output=True).stdout.decode() != cb_:
        raise SystemExit("the coverage census is not committed, or differs from the one recomputed now")
    calls = [score.with_options(volumes={"/out": OUT, "/work": WORK, ("/q3" if k == "a25simh" else "/b"): modal.Volume.from_name(v, version=2).read_only()})
             .spawn(fit_stamp, fit_sha, payload, m, h, k, writer, "cutfull" if cutfull else "") for m, h, k, v in jobs]
    res = [c.get() for c in calls]
    rec = {"label": ((payload["rs_uncut"]["label"] if uncut else RS_LABEL) + "; scoring: manifests, counts and gate results; no values") if rs else (LOO_LABEL + "; scoring: manifests, counts and gate results; no values") if loo else "own_all{,25,10}_L48 scoring: manifests, counts and gate results; no values", "stamp (PT)": stamp, "writer": writer,
           "fit_stamp": fit_stamp, "fit_record_sha256": fit_sha, "census": {"path": crel, "sha256": hashlib.sha256(cb_.encode()).hexdigest()}, "parent_score_record": parent,
           "results": {f"{k}|{h}": r for (_, h, k, _), r in zip(jobs, res, strict=True)}} | ({"p_new_rule": RS_P_NEW} if rs else {}) | (
        {"line": "cutfull", "cutfull_label": "uncut: all tokens; POST HOC (09 Append 41); a sensitivity, never replaces a registered row; "
                                               + ("own_all_rs_uncut weights (their own fit)" if uncut else "registered own_all_rs weights, no refit"),
         "routing_commit": routing_commit, "skipped_routing_records": cut_skipped,
         "messages": {h: {"batch": m["batch"], "routing_pin": m["routing_pin"], "attest_pin": m["attest_pin"], "original_of": m["original_of"]} for m, h, _, _ in jobs}} if cutfull else {})
    # unit-level census after scoring (b6a6b6b48 §3): usable units of each ADMITTED pool x origin x readout, from the per-run counts of the run's own
    # folder (so S4 orphans and excluded runs never count); zero entries kept; a pool x origin x readout with no mapped unit is WITHHELD
    per = {(fo, rd, u): v for r in res for fo, x in r["folders"].items() for rd, c in x["counts"].items() for u, v in c["per_run"].items()}
    mem = {"H_sim": [(e["folder"], f"{e['world']}|{e['cell']}", e["cell"]) for e, k in adm if k == "simh"] + ([] if addendum else [("primary_v3", f"simh_full|{t}", t) for t in payload["a25_counted_tasks"]]),
           "H_A": [(e["folder"], f"{e['world']}|{e['cell']}", e["cell"]) for e, _ in adm if e["world"] == "honest_a" and e["cell"] in hsim_cells],
           "H_B": [(e["folder"], f"{e['world']}|{e['cell']}", e["cell"]) for e, _ in adm if e["world"] == "honest_b"]}
    ucen = {}
    for pl, ms in mem.items():
        for og in ogs:
            for rd in ("R", "C"):
                runs = [(fo, u) for fo, u, c in ms if eog(fo, c) == og]
                mapped = sum(per.get((fo, rd, u), [0, 0])[0] for fo, u in runs)
                ucen[f"{pl}|{og}|{rd}"] = {"admitted_runs": len(runs), "runs_with_units": sum(1 for fo, u in runs if sum(per.get((fo, rd, u), [0, 0]))),
                                           "mapped_units": mapped, "fold_minus1_units": sum(per.get((fo, rd, u), [0, 0])[1] for fo, u in runs),
                                           "withheld": mapped == 0 and (bool(runs) or not addendum)}
    rec["unit_census"] = ucen
    for r in res:  # keep the record small: per-run counts stay in each folder's manifest-side return only as totals
        for x in r["folders"].values():
            for c in x["counts"].values():
                c["per_run"] = len(c["per_run"])
    (HERE / "results" / f"{pfx}_score_{stamp}.json").open("x").write(json.dumps(rec, indent=1, default=float))
    print(json.dumps({"messages": len(res), "folders": sum(len(r["folders"]) for r in res)}))
