"""RULE.md "Own-family scoring of the A25 sim-honest runs" (09's registration @ 51a612b29; scoring only, sealed, no value read).
own_frac{10,25,100}_L48 (pass 1's frozen fold models, fit 2026-10-14_175636, through run_famexp_pass2._score_rows @ 9d006b058) and
own_a (2e's tight weights of 164934, through run_owna507_score.score_all @ 724e3dbf0) on exactly the 149 counted A25 sim-honest
runs (simh_counted_keys_primary_v3, full sha256 80753f44…, via dumps.json @ 231f31404), rows from run_owns.py's multi-dump prep
(spec_owns_2026-10-14_231856, sha-pinned by its committed prep record). Routing: each run by its MAIN, as on the real side:
own_frac* own_a_adapted = r36 (mapped T196 mains -> the designated held-out model; a verified-unmapped main would take the
five-model ensemble, with exclusion evidence: here every main must be mapped); own_a its T196 training-fold map. Hard gates
before scoring: every weight sha256 equals its committed hash (own_a: owna507_2026-10-15_031024.json @ 54e18e68b
.gate_b.weights_sha256; own_frac: the fit record pinned by sha, checked file by file); both dumps' extraction evidence (reader
config, staged-input hashes, H200 on every shard, the reference reproduction and longest-context read-twice gates on every shard, format checks,
completeness for all 149); the two positive-control gates of run_famexp_newarm.py (own_frac L48 honest_b vs pass 1's open pack;
own_a 2e's honest-B gate part 2). Output (SEALED, sim side): convaudit_work:/simh/primary_v3/famexp/<variant>__<R|C>.npz in pass 1's
schema (key, unit, call_index, score [designated; NaN on fold -1], score_by_model [n x 5], fold, run_key, run_fold, run_score
[NaN on fold -1]) plus run_score_by_model [N_runs x 5]; run transform per model on raw logits: R = mean of the run's reply rows,
C = max of its content-only command rows (09 (3)); manifest_famexp.json last. No threshold, rate, catch or comparison; returns
and logs carry hashes, counts and gate results only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run <this file> --r36 <runs36.json>
"""

import hashlib
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
APP = "probe-shift_aswin-surya_sub-p2probes_simh-ownfam_kill-2026-11-13"
BUILD = Path("/tmp/famexp_simh_ownfam_build")
P2 = ("9d006b058", "experiments/2026-10-14_p2_family_expansion/run_famexp_pass2.py", "1ad9b16f0e94e1dc248ecd4dbbe444cd6dda87e28c3d17c5532c7e184ce66e53")
OA = ("724e3dbf0", "experiments/2026-10-15_fpr_sweep_507_prep/run_owna507_score.py", "7ebfc56f2321fab09a0fe1c8e526154fb8ffb09247faf3f3ecab1b55d42acbc2")
CB = ("cc53afc3a", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json", "c002d5f822e49d7877f4e8b79a80ca9dd2e2f16c664b921d3ad12cb3da4c49ac")
OA_REC = ("54e18e68b", "experiments/2026-10-15_fpr_sweep_507_prep/results/owna507_2026-10-15_031024.json", "fa0c87be5d065842f6d4be58027ef8e5884b5ed5d53bc1a88ddf30d69366fe1a")
BMAN = ("69ea6174f", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json", "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
DUMPS = ("231f31404", "experiments/2026-10-14_p2_own_sim_trained/dumps.json", "ff7b154ff5d7f82abe5e6ca8c3be8b1ba3d328fd2780e96c2a5053605a4c512f")
PREP = ("b8bccb29c", "experiments/2026-10-14_p2_own_sim_trained/results/prep_owns_2026-10-14_231856.json")
COUNTED_SHA = "80753f44aa66e78445f233e43e18c96fd01913246afcc03420b9322c447f4082"
SE = "experiments/2026-10-14_q3_q35_simhonest_purchase/results"
EVIDENCE = {"simh_full_b1": {"extract": ("d48da1200", f"{SE}/extract_simh_simh_full_b1_a1_2026-10-14_205202.json", "dc4a545a4f628273ccba64d422612f0a372953d434fb028b004c345e9b1a72d7"),
                             "format": ("d48da1200", f"{SE}/format_check_simh_full_b1_2026-10-14_211537.json", "9097907028155342fc2adb465aec033262e04716be7144668bf54f8667e3d718")},
            "simh_full_b2": {"extract": ("1d1675c4d", f"{SE}/extract_simh_simh_full_b2_a1_2026-10-14_230610.json", "86636ebbd27d85390dafeb70a49feae0a1301cb171c5802a983ec30471faa023"),
                             "format": ("68456a747", f"{SE}/format_check_simh_full_b2_2026-10-14_230552.json", "7e646ec06d700bc1702345b1d36e938481552b1b8374c9535b738c0e1ef412c4")}}
R36_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
SPEC, FIT, OWNS = "2026-10-14_163027", "2026-10-14_175636", "2026-10-14_231856"
PIN_S = {"spec": "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61", "fit": "024fbb0ac04067a08c6dabc49e524a1b2841ab090b2c7f4acdeaf41345034745",
         "open_pack": ("pack_open_2026-10-14_180501", "a0765577c34182c48f7827492f9b745c76201da57c03088686d7f2f434c01e9f")}
OWN_FRAC = [f"own_frac{p}_L48" for p in (10, 25, 100)]
OWN_A_FILES = {"R": "own_a_tight__avg_top2", "C": "own_a_tight__cmd_A"}
OUT_DIR = "/simh/primary_v3/famexp"
REPRO = {"min_cosine": 0.9999, "max_abs_diff": 0.125}  # the extraction gate's tolerances (Amendment 23 step 2's reproduction rule)
TOL = 1e-9

if modal.is_local():
    sys.path.insert(0, str(HERE))
    import run_famexp

    BUILD.mkdir(exist_ok=True)
    for name, (commit, path, want) in list(run_famexp.PINNED.items()) + [("p2pass2.py", P2), ("owna507.py", OA)]:
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
        if hashlib.sha256(raw).hexdigest() != want:
            raise SystemExit(f"{name}: pinned bytes differ ({commit}:{path})")
        (BUILD / name).write_bytes(raw)

app = modal.App(APP)
OUTV = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)
VOLS = {"/out": OUTV, "/s": modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only(),
        "/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only(),
        "/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only(),
        "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(),
        "/q3": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13", version=2).read_only()}
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0", "torch==2.8.0", "transformers==5.19.0", "jinja2")
       .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift").add_local_dir(BUILD, "/root/x2e"))


def _plain(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _runs(key, fold, score, sbm, readout):
    """09 (3): per model on raw logits, R = mean of the run's rows, C = max (as run_famexp_pack._save @ 47fab1bac lines 64-68); the
    designated column is NaN on fold -1 (run_score NaN there; run_score_by_model carries the five)."""
    import numpy as np

    keys, inv = np.unique(key, return_inverse=True)
    rf = np.full(len(keys), -2, np.int64)
    rf[inv] = fold
    if any(len(set(fold[inv == i].tolist())) != 1 for i in range(len(keys))):
        raise SystemExit("a run has rows from two folds")
    if readout == "R":
        rsm = np.stack([np.bincount(inv, weights=sbm[:, j]) / np.bincount(inv) for j in range(5)], axis=1)
    else:
        rsm = np.full((len(keys), 5), -np.inf)
        for j in range(5):
            np.maximum.at(rsm[:, j], inv, sbm[:, j])
    rs = np.where(rf >= 0, rsm[np.arange(len(keys)), np.clip(rf, 0, 4)], np.nan)
    return keys, rf, rs, rsm


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2, volumes=VOLS)
def run(cfg: dict) -> dict:
    import numpy as np
    import torch

    sys.path.insert(0, "/root/x2e")
    import owna507 as oa
    import p2pass2 as p2

    for m, want in ((p2, P2[2]), (oa, OA[2])):
        if _sha(m.__file__) != want:
            raise SystemExit(f"{m.__name__}: imported bytes differ from the pin")
    base = Path("/s/famexp") / SPEC
    if _sha(base / f"spec_{SPEC}.json") != PIN_S["spec"] or _sha(base / f"fit_{FIT}.json") != PIN_S["fit"]:
        raise SystemExit("pass 1's spec or fit manifest differs from its pin")
    spec = json.loads((base / f"spec_{SPEC}.json").read_text())
    fit = [x for g in json.loads((base / f"fit_{FIT}.json").read_text())["groups"] for x in g["fits"]]
    fot, fixed = spec["fot"], set(spec["fixed_ids"])
    _, fid = oa._fixed_ids()
    if set(fid) != fixed:
        raise SystemExit("2e's tokenizer fixed call-token ids differ from pass 1's fixed_ids")
    w, wsha = oa.load_weights(cfg["own_a"])
    want_w = cfg["own_a"]["committed_weights_sha256"]
    if {k: v for k, v in wsha.items()} != {k: v for k, v in want_w.items() if not k.startswith("C|")} or \
            {nm: cfg["own_a"]["weights_C"][nm]["sha256"] for nm in ("tight", "outer")} != {nm: want_w[f"C|{nm}"] for nm in ("tight", "outer")}:
        raise SystemExit("own_a weight sha256s differ from owna507_2026-10-15_031024.json's committed gate_b.weights_sha256")
    choice, fold_t196 = cfg["own_a"]["choice"], cfg["own_a"]["fold_t196"]
    if any(fold_t196[t] != f for t, f in fot.items() if t in fold_t196):
        raise SystemExit("own_a's T196 training-fold map differs from pass 1's fot (r36 by main)")
    # extraction evidence in the dumps themselves: staged inputs, complete.json (completeness), metas for every counted run
    owns_p = Path("/s/owns") / OWNS / f"spec_owns_{OWNS}.json"
    if _sha(owns_p) != cfg["owns_spec_sha256"]:
        raise SystemExit("run_owns.py's prep spec differs from its committed sha256")
    so = json.loads(owns_p.read_text())
    eps = [{"key": k, "world": wt[0], "task": wt[1], "arm": so["arm_of"][k], "extraction_run": Path(so["run_of"][k]).name, "run_of": so["run_of"][k], "skip": so["skip"][k]}
           for k, wt in so["map"].items() if wt[0] == "simh_full"]
    counted = cfg["counted_tasks"]
    if sorted(e["task"] for e in eps) != sorted(counted) or len(eps) != 149:
        raise SystemExit(f"the prep's simh_full epochs ({len(eps)}) are not exactly the 149 counted runs")
    ev = {}
    for d in cfg["dumps"]:
        root, inp = Path("/q3") / d["run"], Path("/q3") / d["staging"] / "inputs" / f"{d['arm']}.jsonl"
        raw = inp.read_bytes()
        done = json.loads((root / "complete.json").read_text())
        mine = [e for e in eps if e["extraction_run"] == d["run"]]
        if hashlib.sha256(raw).hexdigest() != d["staged_sha256"] or done.get("inputs_sha256", {}).get(f"{d['arm']}.jsonl") != d["staged_sha256"] or \
                done.get("epochs") != d["epochs"] or len(mine) != d["epochs"] or any(e["run_of"] != str(root) for e in mine):
            raise SystemExit(f"{d['run']}: staged inputs, complete.json or epoch count differ from the pins")
        sk2 = oa._skip_file(raw, {e["key"] for e in mine}, fid)[0]
        if any(sorted(map(tuple, sk2[e["key"]])) != sorted(map(tuple, e["skip"])) for e in mine):
            raise SystemExit(f"{d['run']}: 2e's skip rule and the prep's skip rule disagree")
        for e in mine:
            e["skip_oa"] = sk2[e["key"]]
        ev[d["run"]] = {"staged_sha256": d["staged_sha256"], "complete_epochs": done.get("epochs"), "counted_runs": len(mine)}
    if any(t not in fot for t in counted):
        raise SystemExit("a counted sim-honest task is not a mapped T196 cell (an unmapped main would need ensemble evidence: stop)")
    # the positive-control gates (run_famexp_newarm.py's): own_frac L48 honest_b vs pass 1's open pack; own_a 2e's honest-B gate part 2
    gate = {}
    pk_dir = base / PIN_S["open_pack"][0]
    if _sha(pk_dir / "manifest.json") != PIN_S["open_pack"][1]:
        raise SystemExit("pass 1's open pack manifest differs from its pin")
    pk_files = {Path(x["npz"]["path"]).name: x["npz"]["sha256"] for x in json.loads((pk_dir / "manifest.json").read_text())["variants"].values()}
    torch.set_num_threads(8)
    hb = {k: v for k, v in spec["map"].items() if v[0] == "honest_b"}
    for readout in ("R", "C"):
        own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": hb, "arm_of": {k: spec["arm_of"][k] for k in hb},
               "run_of": {k: spec["run_of"][k] for k in hb}} | ({"skip": {k: spec["skip"][k] for k in hb}} if readout == "C" else {})
        res, _ = p2._score_rows(own, 48, fot, spec, fit, FIT, base, readout, {})
        for v in OWN_FRAC:
            names, units, ci, folds, S = res[v]
            fn = f"{v}__{readout}.npz"
            b_ = (pk_dir / fn).read_bytes()
            if hashlib.sha256(b_).hexdigest() != pk_files[fn]:
                raise SystemExit(f"open pack {fn}: sha256 differs from its manifest")
            z = np.load(io.BytesIO(b_))
            sel = np.char.startswith(z["key"].astype(str), "honest_b|")
            ref = {(k, int(u), int(c)): (float(s), int(f)) for k, u, c, s, f in zip(z["key"][sel].astype(str), z["unit"][sel], z["call_index"][sel], z["score"][sel], z["fold"][sel])}
            des = np.where(folds >= 0, S[np.arange(len(folds)), np.clip(folds, 0, 4)], np.nan)
            mine = {(k, int(u), int(c)): (float(s), int(f)) for k, u, c, s, f in zip(names.astype(str), units, ci, des, folds)}
            dm = max((abs(mine[u][0] - ref[u][0]) for u in set(mine) & set(ref)), default=float("inf"))
            ok = set(mine) == set(ref) and all(mine[u][1] == ref[u][1] for u in mine) and dm <= TOL
            gate[f"{v}|{readout}"] = {"rows_mine": len(mine), "rows_pack": len(ref), "max_abs_diff": dm, "pass": ok}
    hb_eps = cfg["own_a"]["hb_epochs"]
    sk_b = {}
    for rel, h in sorted({(e["staged_inputs_file"], e["staged_inputs_sha256"]) for e in hb_eps}):
        raw = (Path("/b") / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"{rel}: staged sha256")
        sk_b |= oa._skip_file(raw, {e["key"] for e in hb_eps if e["staged_inputs_file"] == rel}, fid)[0]
    torch.set_num_threads(4)
    sc = oa.score_all("/b", hb_eps, sk_b, choice, w)

    def ref_rows(e):
        b_ = (Path("/sol") / e["path"]).read_bytes()
        if hashlib.sha256(b_).hexdigest() != e["sha256"]:
            raise SystemExit(f"{e['path']}: sha256")
        return np.load(io.BytesIO(b_))

    zc = ref_rows(cfg["own_a"]["ref_C"])
    for ro, readout in (("avg_top2", "R"), ("cmd_A", "C")):
        s = sc[ro]
        if ro == "cmd_A":
            ref = {}
            for k, i, c, v in zip(zc["key"], zc["irid"], zc["call_index"], zc["score"]):
                if (str(k), str(i), int(c)) in ref:
                    raise SystemExit("own_a C reference: a duplicate row identity (2e's gate)")
                ref[(str(k), str(i), int(c))] = float(v)
        else:
            parts = {}
            for e in cfg["own_a"]["ref_R"].values():
                z = ref_rows(e)
                for k, i, c, v in zip(z["key"], z["irid"], z["call_index"], z["score"]):
                    parts.setdefault((str(k), str(i), int(c)), []).append(float(v))
            if any(len(v) != 2 for u, v in parts.items() if u[0].startswith("honest_b|")):
                raise SystemExit("own_a R reference: an honest_b row without exactly two layer contributions (2e's gate)")
            ref = {u: float(np.mean(v)) for u, v in parts.items()}
        ids = list(zip(s["key"].tolist(), s["irid"], s["call_index"].tolist()))
        if len(set(ids)) != len(ids):
            raise SystemExit(f"own_a {ro}: a duplicate row identity in the scale-up path (2e's gate)")
        des = [s["score_by_model"][i, fold_t196[k.split("|", 1)[1]]] for i, k in enumerate(s["key"].tolist())]
        mine, ref_hb = dict(zip(ids, des)), {u for u in ref if u[0].startswith("honest_b|")}
        dm = max((abs(mine[u] - ref[u]) for u in set(mine) & ref_hb), default=float("inf"))
        gate[f"own_a|{readout}"] = {"rows_mine": len(mine), "rows_reference": len(ref_hb), "max_abs_diff": dm, "pass": set(mine) == ref_hb and len(mine) > 0 and dm <= TOL}
    if not all(g["pass"] for g in gate.values()):
        raise SystemExit(f"positive-control gate failed, nothing written: {json.dumps({k: g for k, g in gate.items() if not g['pass']}, default=str)}")
    # scoring (sealed): rows in memory, coverage checked, then written create-only, manifest last
    od = Path("/out") / OUT_DIR.lstrip("/")
    if od.exists() and any(od.iterdir()):
        raise SystemExit(f"{od} already holds files: never overwritten")
    joins = sorted(f"simh_full|{e['task']}" for e in eps)
    files, counts, dropped, blobs, wsha_f = {}, {}, {}, {}, {}

    def build(name, key, unit, ci, sbm, fold, readout):
        if not np.isfinite(sbm).all() or (fold < 0).any():
            raise SystemExit(f"{name}: a non-finite score or an unmapped run")
        if any(fold[i] != fot[k.split("|", 1)[1]] for i, k in enumerate(key.tolist())):
            raise SystemExit(f"{name}: a row is not routed to the fold model that held out its main")
        score = sbm[np.arange(len(fold)), fold]
        rk, rf, rs, rsm = _runs(key, fold, score, sbm, readout)
        buf = io.BytesIO()
        np.savez(buf, key=key, unit=unit, call_index=ci, score=score, score_by_model=sbm, fold=fold, run_key=rk, run_fold=rf, run_score=rs, run_score_by_model=rsm)
        blobs[f"{name}.npz"] = buf.getvalue()
        files[f"{name}.npz"] = hashlib.sha256(buf.getvalue()).hexdigest()
        counts[name] = {"n_runs": len(rk), "n_units": len(key)}
        dropped[name] = sorted(set(joins) - set(rk.tolist()))

    tf = {e["task"]: fot[e["task"]] for e in eps}
    for readout in ("R", "C"):
        own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": {e["key"]: [e["world"], e["task"]] for e in eps},
               "arm_of": {e["key"]: e["arm"] for e in eps}, "run_of": {e["key"]: e["run_of"] for e in eps}} | (
            {"skip": {e["key"]: e["skip"] for e in eps}} if readout == "C" else {})
        res, ws = p2._score_rows(own, 48, tf, spec, fit, FIT, base, readout, {})
        wsha_f |= ws
        for v in OWN_FRAC:
            names, units, ci, folds, S = res[v]
            build(f"{v}__{readout}", names.astype(str), units, ci, S, folds, readout)
    sc = oa.score_all("/q3", eps, {e["key"]: e["skip_oa"] for e in eps}, choice, w)
    for ro, readout in (("avg_top2", "R"), ("cmd_A", "C")):
        s = sc[ro]
        fold = np.array([fold_t196[k.split("|", 1)[1]] for k in s["key"].tolist()], np.int64)
        build(OWN_A_FILES[readout], s["key"].astype(str), s["unit"], s["call_index"], s["score_by_model"], fold, readout)
    if any(dropped.values()) or any(c["n_runs"] != 149 for c in counts.values()) or len(blobs) != 2 * (len(OWN_FRAC) + 1):
        raise SystemExit(f"coverage is not all 149 counted runs in every file, nothing written: {json.dumps({k: v for k, v in dropped.items() if v})}")
    od.mkdir(parents=True, exist_ok=True)
    for fn, b_ in sorted(blobs.items()):
        (od / fn).open("xb").write(b_)
    man = {"label": "own-family scoring of the A25 sim-honest runs (09 @ 51a612b29): SEALED sim-side pack; hashes, counts and gate results only",
           "status": "POST HOC end-to-end inputs; not in Append 25's frozen five, X_err_full or its sim-honest calibration (09 (4))",
           "out_dir": OUT_DIR, "volume": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", "files": files, "counts": counts, "runs_without_rows": dropped,
           "keys_scored": joins, "routing": {j: {"fold": fot[j.split("|", 1)[1]], "basis": "T196 cell, mapped main: the designated held-out model (own_frac: r36 / own_a_adapted; own_a: its T196 training-fold map, equal)"} for j in joins},
           "gate": gate, "weights_sha256": {"own_frac": wsha_f, "own_a": wsha, "own_a_committed_equal": True}, "dumps_evidence_in_job": ev,
           "inputs": cfg["pins"], "run_transform": {"R": "per model, mean of the run's reply rows", "C": "per model, max of the run's content-only command rows",
           "designated": "run_score = the designated model's run transform; NaN on fold -1 (none here)", "code": "this file's _runs (= run_famexp_pack._save @ 47fab1bac lines 64-68, per model)"},
           "schema": "pass 1's: key, unit, call_index, score, score_by_model (n x 5, fold models 0..4), fold, run_key, run_fold, run_score; extension run_score_by_model (N_runs x 5)",
           "code": {"own_frac": f"{P2[1]} @ {P2[0]} _score_rows", "own_a": f"{OA[1]} @ {OA[0]} score_all", "rows_prep": f"run_owns.py prep spec_owns_{OWNS}"}}
    mb = json.dumps(_plain(man), indent=1).encode()
    (od / "manifest_famexp.json").open("xb").write(mb)
    OUTV.commit()
    return _plain({"stage": "written", "gate": gate, "manifest": str(od / "manifest_famexp.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest(),
                   "files": files, "counts": counts, "runs_without_rows": dropped, "n_scored": len(joins), "dumps_evidence_in_job": ev})


@app.local_entrypoint()
def main(r36: str):
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout

    def show(pin):
        b = git("show", f"{pin[0]}:{pin[1]}")
        if len(pin) > 2 and hashlib.sha256(b).hexdigest() != pin[2]:
            raise SystemExit(f"{pin[1]}: sha256 differs from its pin")
        return json.loads(b)

    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/p2-family-expansion"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", "51a612b29").decode().splitlines()):
        raise SystemExit("09's registration @ 51a612b29 is not on origin")
    cb, oarec, bman, t196, dumps, prep = show(CB), show(OA_REC), show(BMAN), show(T196), show(DUMPS), show(PREP)
    r36b = Path(r36).read_bytes()
    if hashlib.sha256(r36b).hexdigest() != R36_SHA:
        raise SystemExit("runs36: sha256 differs")
    fom = {}
    for r in json.loads(r36b):
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    choice = cb["layer_choice_own_a"]
    if choice != [[49, 50], [49, 50], [50, 51], [49, 50], [49, 50]]:
        raise SystemExit("layer_choice_own_a differs from 2e's")
    tav, cmd_a = cb["saved"]["tight"]["avg"], cb["saved"]["cmd"]["cmdA"]
    need = [f"{L}|{f}" for f in range(5) for L in choice[f]]
    full = [d for d in dumps["dumps"] if d["world"] == "simh_full"]
    if {d["counted_keys"]["sha256"] for d in full} != {COUNTED_SHA} or sorted(d["run"] for d in full) != ["simh_full_b1", "simh_full_b2"]:
        raise SystemExit("dumps.json's simh_full dumps or counted-key pin differ")
    ck = full[0]["counted_keys"]
    counted = show((ck["commit"], ck["path"], ck["sha256"]))
    tasks = [x["task"] for x in counted["keys"] if x.get("usable") is True and x.get("world") == "simh_full"]
    if len(tasks) != 149 or len(set(tasks)) != 149:
        raise SystemExit("the counted set is not 149 distinct usable simh_full tasks")
    ev = {}
    for d in full:  # Append 25's input requirements for each contributing dump, from its committed extraction and format records
        ex, fc = show(EVIDENCE[d["run"]]["extract"]), show(EVIDENCE[d["run"]]["format"])
        # two record kinds per shard: reference reproductions (batch C rows re-read: min_cosine, max_abs_diff) and the shard's own
        # longest-context read-twice check (own_longest_read_twice_min_cosine); both are required on every shard
        refs = [[r for r in (s.get("reproduction") or []) if "reference" in r] for s in ex["shards"]]
        longs = [[r for r in (s.get("reproduction") or []) if "own_longest_read_twice_min_cosine" in r] for s in ex["shards"]]
        other = [r for s in ex["shards"] for r in (s.get("reproduction") or []) if "reference" not in r and "own_longest_read_twice_min_cosine" not in r]
        okx = (ex["config"]["run"] == d["run"] and ex["config"]["inputs"] == d["staging"] and ex["config"]["inputs_sha256"] == {f"{d['arm']}.jsonl": d["staged_sha256"]}
               and ex["config"]["model"] == "Qwen/Qwen3.5-27B" and {s.get("gpu") for s in ex["shards"]} == {"NVIDIA H200"}
               and all(s.get("device_count") == 1 for s in ex["shards"]) and not other and all(refs) and all(len(x) == 1 for x in longs)
               and all(r["min_cosine"] >= REPRO["min_cosine"] and r["max_abs_diff"] <= REPRO["max_abs_diff"] for x in refs for r in x)
               and all(x[0]["own_longest_read_twice_min_cosine"] >= REPRO["min_cosine"] for x in longs)
               and sum(s["epochs"] for s in ex["shards"]) == d["epochs"] and fc.get("passes") is True and fc.get("n") == d["epochs"])
        ev[d["run"]] = {"extract_record": dict(zip(("commit", "path", "sha256"), EVIDENCE[d["run"]]["extract"])), "format_check": dict(zip(("commit", "path", "sha256"), EVIDENCE[d["run"]]["format"])),
                        "reader": {k: ex["config"].get(k) for k in ("model", "pad_to", "render", "gate_arm_prefix")}, "shards": len(ex["shards"]), "gpu": "NVIDIA H200 on every shard",
                        "reference_reproductions": sum(map(len, refs)), "longest_context_checks": sum(map(len, longs)),
                        "longest_context_tokens_max": max((x[0]["tokens"] for x in longs if x), default=None), "passes": okx}
        if not okx:
            raise SystemExit(f"{d['run']}: extraction evidence fails Append 25's input requirements: {json.dumps(ev[d['run']])}")
    own_a = {"choice": choice, "fold_t196": {c["key"]: fom[c["main"]] for c in t196["cells"]},
             "weights_R": {k: {"path": tav[k]["weights"], "sha256": tav[k]["weights_sha256"]} for k in need},
             "weights_C": {"tight": cmd_a["own_a_tight_weights"], "outer": cmd_a["own_a_weights"]},
             "ref_R": {k: {"path": tav[k]["rows"], "sha256": tav[k]["rows_sha256"]} for k in need},
             "ref_C": {"path": cmd_a["rows|own_a_tight"]["path"], "sha256": cmd_a["rows|own_a_tight"]["sha256"]},
             "committed_weights_sha256": oarec["gate_b"]["weights_sha256"],
             "hb_epochs": [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_file", "staged_inputs_sha256")}
                           for e in bman["epochs"] if e["world"] == "honest_b"]}
    pins = {k: dict(zip(("commit", "path", "sha256"), v)) for k, v in (("p2_scorer", P2), ("owna507", OA), ("cb_record", CB), ("owna507_record", OA_REC),
                                                                       ("correctedb_manifest", BMAN), ("t196", T196), ("dumps", DUMPS), ("prep_record", PREP))} | {
        "counted_keys": ck, "runs36": {"sha256": R36_SHA}, "registration_09": {"commit": "51a612b29", "path": "experiments/2026-10-14_priority1_spec_09/RULE.md",
        "section": "Registration: own-family scoring of the A25 sim-honest runs"}, "pass1": {"spec": [SPEC, PIN_S["spec"]], "fit": [FIT, PIN_S["fit"]],
        "open_pack": list(PIN_S["open_pack"])}, "owns_spec": [OWNS, prep["result"]["spec_sha256"]], "dump_evidence": ev}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = run.remote({"own_a": own_a, "counted_tasks": tasks, "dumps": full, "owns_spec_sha256": prep["result"]["spec_sha256"], "pins": pins})
    rec = {"label": "own-family scoring of the A25 sim-honest runs (sealed): hashes, counts and gate results only", "git": git("rev-parse", "HEAD").decode().strip(),
           "stamp (PT)": stamp, "app": APP, "app_id": app.app_id, "pins": pins} | out
    dest = HERE / "results" / f"simh_ownfam_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1, default=str))
    print(dest, json.dumps({k: rec.get(k) for k in ("stage", "manifest_sha256", "counts", "n_scored")}), json.dumps({k: g["pass"] for k, g in out["gate"].items()}))
