"""RULE.md addendum "Sim-honest scoring of the step-3 recipes" (under 09's A25 registration @ 51a612b29; scoring only, sealed, POST HOC).
The step-3 recipes (r1only, r2only, r12; L48; R and C) from fit 2026-10-15_114402, through run_famexp_pass2._score_rows @ 9d006b058,
on exactly the 149 counted A25 sim-honest runs (simh_counted_keys_primary_v3, sha256 80753f44…, via dumps.json @ 231f31404), rows
from run_owns.py's multi-dump prep (spec_owns_2026-10-14_231856). This is famexp's run_famexp_simh_ownfam.py @ a02df5169 with its
inputs and gates, own_a removed and the step-3 fit swapped in. Routing by MAIN (09 (1)): every counted run is a mapped T196 cell and
uses the step-3 fold model that held out its main (r36). Hard gates before scoring: the step-3 fit record on the Volume names the
committed RULE (sha256 d073d238…), fit-path hash and run, all 90 fits certified, every weight file equals its recorded sha256; both
dumps' extraction evidence (as famexp); 2e's skip rule equals the prep's; a positive control: T196 honest_b re-scored here equals the
fit job's own held-out open rows (1e-9). Output (SEALED, sim side): convaudit_work:/simh/primary_v3/step3/step3_<recipe>_L48__<R|C>.npz in
pass 1's schema plus run_score_by_model (09 (2)); run transform per model on raw logits, R = mean of reply rows, C = max of content-only
command rows (09 (3)); manifest_step3.json last. No threshold, rate, catch or comparison; returns carry hashes, counts and gates only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run <this file>
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
FAM = REPO / "experiments/2026-10-14_p2_family_expansion"
APP = "probe-shift_aswin-surya_sub-step3_simh_kill-2026-10-22"
BUILD = Path("/tmp/step3_simh_build")
P2 = ("9d006b058", "experiments/2026-10-14_p2_family_expansion/run_famexp_pass2.py", "1ad9b16f0e94e1dc248ecd4dbbe444cd6dda87e28c3d17c5532c7e184ce66e53")
OA = ("724e3dbf0", "experiments/2026-10-15_fpr_sweep_507_prep/run_owna507_score.py", "7ebfc56f2321fab09a0fe1c8e526154fb8ffb09247faf3f3ecab1b55d42acbc2")  # its skip rule only
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
DUMPS = ("231f31404", "experiments/2026-10-14_p2_own_sim_trained/dumps.json", "ff7b154ff5d7f82abe5e6ca8c3be8b1ba3d328fd2780e96c2a5053605a4c512f")
PREP = ("b8bccb29c", "experiments/2026-10-14_p2_own_sim_trained/results/prep_owns_2026-10-14_231856.json")
R36 = ("c24106ca0", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json", "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")
COUNTED_SHA = "80753f44aa66e78445f233e43e18c96fd01913246afcc03420b9322c447f4082"
SE = "experiments/2026-10-14_q3_q35_simhonest_purchase/results"
EVIDENCE = {"simh_full_b1": {"extract": ("d48da1200", f"{SE}/extract_simh_simh_full_b1_a1_2026-10-14_205202.json", "dc4a545a4f628273ccba64d422612f0a372953d434fb028b004c345e9b1a72d7"),
                             "format": ("d48da1200", f"{SE}/format_check_simh_full_b1_2026-10-14_211537.json", "9097907028155342fc2adb465aec033262e04716be7144668bf54f8667e3d718")},
            "simh_full_b2": {"extract": ("1d1675c4d", f"{SE}/extract_simh_simh_full_b2_a1_2026-10-14_230610.json", "86636ebbd27d85390dafeb70a49feae0a1301cb171c5802a983ec30471faa023"),
                             "format": ("68456a747", f"{SE}/format_check_simh_full_b2_2026-10-14_230552.json", "7e646ec06d700bc1702345b1d36e938481552b1b8374c9535b738c0e1ef412c4")}}
SPEC, OWNS = "2026-10-14_163027", "2026-10-14_231856"
PIN_SPEC = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
FIT3 = {"stamp": "2026-10-15_114402", "run": "fit_2026-10-15_114402", "rule_sha256": "d073d238849b93b9a93b214977b3919c8988df4dd434b3f76ac63136a246b2dc",
        "fit_path_sha256": "fcf85fb4d4456bf59700b952795df662165f33c0db6933a2d0e07029e214c76e",
        "volume_record_sha256": "0aa4ad7db54aea16597588eda368c43053684456ef463efd59c3761eccfa2bed"}  # = committed results/fit_2026-10-15_114402_volume_record.json
RECIPES = ("r1only", "r2only", "r12")
VARIANTS = [f"step3_{r}_L48" for r in RECIPES]
OUT_DIR = "/simh/primary_v3/step3"
REPRO = {"min_cosine": 0.9999, "max_abs_diff": 0.125}
TOL = 1e-9

if modal.is_local():
    sys.path.insert(0, str(FAM))
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
        "/w3": modal.Volume.from_name("probe-shift_aswin-surya_sub-step3_work_kill-2026-11-13", version=2).read_only(),
        "/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only(),
        "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(),
        "/q3": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13", version=2).read_only()}
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0", "torch==2.8.0", "transformers==5.19.0", "jinja2")
       .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift").add_local_dir(BUILD, "/root/x2e"))


def _plain(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _runs(key, fold, score, sbm, readout):
    """09 (3): per model on raw logits, R = mean of the run's rows, C = max; the designated column is NaN on fold -1 (famexp's _runs)."""
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

    sys.path.insert(0, "/root/x2e")
    import owna507 as oa
    import p2pass2 as p2

    for m, want in ((p2, P2[2]), (oa, OA[2])):
        if _sha(m.__file__) != want:
            raise SystemExit(f"{m.__name__}: imported bytes differ from the pin")
    if _sha(Path("/s/famexp") / SPEC / f"spec_{SPEC}.json") != PIN_SPEC:
        raise SystemExit("pass 1's spec differs from its pin")
    spec = json.loads((Path("/s/famexp") / SPEC / f"spec_{SPEC}.json").read_text())
    spec_np = spec | {"layer_choice_a": [[] for _ in spec["layer_choice_a"]]}  # no b_asis models in _score_rows
    base3 = Path("/w3/step3") / SPEC
    if _sha(base3 / f"fit_{FIT3['stamp']}.json") != FIT3["volume_record_sha256"]:
        raise SystemExit("the step-3 fit record on the Volume differs from its committed copy's sha256")
    frec = json.loads((base3 / f"fit_{FIT3['stamp']}.json").read_text())
    roster = sorted((x["variant"], x["readout"], x["layer"], x["fold"]) for g in frec["groups"] for x in g["fits"])
    if roster != sorted((f"step3_{r}_L{L}", rd, L, f) for r in RECIPES for rd in ("R", "C") for L in (16, 32, 48) for f in range(5)) or \
            not all(x["certified"] for g in frec["groups"] for x in g["fits"]):
        raise SystemExit("the step-3 fit roster is not exactly 3 recipes x R/C x L16/32/48 x 5 folds, all certified")
    if frec["rule"]["sha256"] != FIT3["rule_sha256"] or frec["fit_path_sha256"] != FIT3["fit_path_sha256"] or frec["run"] != FIT3["run"] \
            or frec["failed_groups"] or frec["uncertified"] or len([x for g in frec["groups"] for x in g["fits"]]) != 90 or {g["run"] for g in frec["groups"]} != {FIT3["run"]}:
        raise SystemExit("the step-3 fit record on the Volume does not match the committed RULE, fit path, run, or 90 certified fits")
    fit3 = [x for g in frec["groups"] for x in g["fits"] if x["layer"] == 48]
    fot, fixed = spec["fot"], set(spec["fixed_ids"])
    if any(fot.get(t) != f for t, f in cfg["fold_t196"].items()) or set(fot) != set(cfg["fold_t196"]):
        raise SystemExit("the spec's fot differs from the independently pinned T196 -> main -> r36 fold map")
    _, fid = oa._fixed_ids()
    if set(fid) != fixed:
        raise SystemExit("2e's tokenizer fixed call-token ids differ from pass 1's fixed_ids")
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
        ev[d["run"]] = {"staged_sha256": d["staged_sha256"], "complete_epochs": done.get("epochs"), "counted_runs": len(mine)}
    if any(t not in fot for t in counted):
        raise SystemExit("a counted sim-honest task is not a mapped T196 cell: stop")
    # positive control: T196 honest_b re-scored on CPU here == the step-3 fit job's own held-out open rows (GPU), designated model
    gate = {}
    hb = {k: v for k, v in spec["map"].items() if v[0] == "honest_b"}
    for readout in ("R", "C"):
        own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": hb, "arm_of": {k: spec["arm_of"][k] for k in hb},
               "run_of": {k: spec["run_of"][k] for k in hb}} | ({"skip": {k: spec["skip"][k] for k in hb}} if readout == "C" else {})
        res, _ = p2._score_rows(own, 48, fot, spec_np, fit3, FIT3["run"], base3, readout, {})
        for v in VARIANTS:
            names, units, ci, folds, S = res[v]
            ref = {}
            for x in sorted((x for x in fit3 if x["variant"] == v and x["readout"] == readout), key=lambda x: x["fold"]):
                p = base3 / f"run_{FIT3['run']}" / "open" / f"{v}__{readout}__L48__f{x['fold']}.npz"
                if _sha(p) != x["sha256"]["open"]:
                    raise SystemExit(f"{p.name}: sha256 differs from the fit record")
                z = np.load(p)
                sel = np.char.startswith(z["key"].astype(str), "honest_b|")
                ref |= {(k, int(u), int(c)): (float(s), int(f)) for k, u, c, s, f in zip(z["key"][sel].astype(str), z["unit"][sel], z["call_index"][sel], z["score"][sel], z["fold"][sel])}
            des = np.where(folds >= 0, S[np.arange(len(folds)), np.clip(folds, 0, 4)], np.nan)
            mine = {(k, int(u), int(c)): (float(s), int(f)) for k, u, c, s, f in zip(names.astype(str), units, ci, des, folds)}
            dm = max((abs(mine[u][0] - ref[u][0]) for u in set(mine) & set(ref)), default=float("inf"))
            ok = set(mine) == set(ref) and all(mine[u][1] == ref[u][1] for u in mine) and dm <= TOL
            gate[f"{v}|{readout}"] = {"rows_mine": len(mine), "rows_fit_job": len(ref), "max_abs_diff": dm, "pass": ok}
    if not all(g["pass"] for g in gate.values()):
        raise SystemExit(f"positive-control gate failed, nothing written: {json.dumps({k: g for k, g in gate.items() if not g['pass']}, default=str)}")
    od = Path("/out") / OUT_DIR.lstrip("/")
    if od.exists() and any(od.iterdir()):
        raise SystemExit(f"{od} already holds files: never overwritten")
    joins = sorted(f"simh_full|{e['task']}" for e in eps)
    files, counts, dropped, blobs, wsha = {}, {}, {}, {}, {}

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
        res, ws = p2._score_rows(own, 48, tf, spec_np, fit3, FIT3["run"], base3, readout, {})
        wsha |= ws
        for v in VARIANTS:
            names, units, ci, folds, S = res[v]
            build(f"{v}__{readout}", names.astype(str), units, ci, S, folds, readout)
    if any(dropped.values()) or any(c["n_runs"] != 149 for c in counts.values()) or len(blobs) != 2 * len(VARIANTS):
        raise SystemExit(f"coverage is not all 149 counted runs in every file, nothing written: {json.dumps({k: v for k, v in dropped.items() if v})}")
    od.mkdir(parents=True, exist_ok=True)
    for fn, b_ in sorted(blobs.items()):
        (od / fn).open("xb").write(b_)
    man = {"label": "step-3 recipes on the A25 sim-honest runs (under 09's registration @ 51a612b29): SEALED sim-side pack; hashes, counts and gate results only",
           "status": "POST HOC end-to-end inputs (step 3 item 0 recipes); not in Append 25's frozen five, X_err_full or its sim-honest calibration (09 (4))",
           "out_dir": OUT_DIR, "volume": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", "files": files, "counts": counts, "runs_without_rows": dropped,
           "keys_scored": joins, "routing": {j: {"fold": fot[j.split("|", 1)[1]], "basis": "T196 cell, mapped main: the step-3 fold model that held out its main (r36)"} for j in joins},
           "gate": gate, "weights_sha256": wsha, "step3_fit": FIT3 | {"volume": "probe-shift_aswin-surya_sub-step3_work_kill-2026-11-13"}, "dumps_evidence_in_job": ev,
           "inputs": cfg["pins"], "run_transform": {"R": "per model, mean of the run's reply rows", "C": "per model, max of the run's content-only command rows",
           "designated": "run_score = the designated model's run transform; NaN on fold -1 (none here)", "code": "this file's _runs (= famexp's run_famexp_simh_ownfam.py @ a02df5169)"},
           "schema": "pass 1's: key, unit, call_index, score, score_by_model (n x 5, fold models 0..4), fold, run_key, run_fold, run_score; extension run_score_by_model (N_runs x 5)",
           "code": {"step3": f"{P2[1]} @ {P2[0]} _score_rows with the step-3 fit record", "rows_prep": f"run_owns.py prep spec_owns_{OWNS}"}}
    mb = json.dumps(_plain(man), indent=1).encode()
    (od / "manifest_step3.json").open("xb").write(mb)
    OUTV.commit()
    return _plain({"stage": "written", "gate": gate, "manifest": str(od / "manifest_step3.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest(),
                   "files": files, "counts": counts, "runs_without_rows": dropped, "n_scored": len(joins), "dumps_evidence_in_job": ev})


@app.local_entrypoint()
def main():
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout  # noqa: E731

    def show(pin):
        b = git("show", f"{pin[0]}:{pin[1]}")
        if len(pin) > 2 and hashlib.sha256(b).hexdigest() != pin[2]:
            raise SystemExit(f"{pin[1]}: sha256 differs from its pin")
        return json.loads(b)

    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/step3-retrain"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not origin/aswin/step3-retrain")
    if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", "51a612b29").decode().splitlines()):
        raise SystemExit("09's registration @ 51a612b29 is not on origin")
    loc = json.loads((HERE / "results" / f"fit_{FIT3['stamp']}.json").read_text())["result"]
    if loc["rule"]["sha256"] != FIT3["rule_sha256"] or loc["fit_path_sha256"] != FIT3["fit_path_sha256"] or loc["run"] != FIT3["run"] or loc["fits"] != 90:
        raise SystemExit("the committed step-3 fit record differs from this file's pins")
    dumps, prep = show(DUMPS), show(PREP)
    fom = {}
    for r in show(R36):  # independent routing evidence: T196 cell -> main -> r36 fold, no main in two folds
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    fold_t196 = {c["key"]: fom[c["main"]] for c in show(T196)["cells"]}
    full = [d for d in dumps["dumps"] if d["world"] == "simh_full"]
    if {d["counted_keys"]["sha256"] for d in full} != {COUNTED_SHA} or sorted(d["run"] for d in full) != ["simh_full_b1", "simh_full_b2"]:
        raise SystemExit("dumps.json's simh_full dumps or counted-key pin differ")
    ck = full[0]["counted_keys"]
    counted = show((ck["commit"], ck["path"], ck["sha256"]))
    tasks = [x["task"] for x in counted["keys"] if x.get("usable") is True and x.get("world") == "simh_full"]
    if len(tasks) != 149 or len(set(tasks)) != 149:
        raise SystemExit("the counted set is not 149 distinct usable simh_full tasks")
    if any(t not in fold_t196 for t in tasks):
        raise SystemExit("a counted task is not a T196 cell")
    ev = {}
    for d in full:  # famexp's evidence checks, unchanged
        ex, fc = show(EVIDENCE[d["run"]]["extract"]), show(EVIDENCE[d["run"]]["format"])
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
                        "shards": len(ex["shards"]), "gpu": "NVIDIA H200 on every shard", "reference_reproductions": sum(map(len, refs)),
                        "longest_context_checks": sum(map(len, longs)), "passes": okx}
        if not okx:
            raise SystemExit(f"{d['run']}: extraction evidence fails Append 25's input requirements")
    pins = {k: dict(zip(("commit", "path", "sha256"), v)) for k, v in (("p2_scorer", P2), ("owna507_skip_rule", OA), ("t196", T196), ("dumps", DUMPS), ("prep_record", PREP), ("runs36", R36))} | {
        "counted_keys": ck, "registration_09": {"commit": "51a612b29", "path": "experiments/2026-10-14_priority1_spec_09/RULE.md",
                                                "section": "Registration: own-family scoring of the A25 sim-honest runs"},
        "pass1_spec": [SPEC, PIN_SPEC], "owns_spec": [OWNS, prep["result"]["spec_sha256"]], "step3_fit": FIT3, "dump_evidence": ev}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = run.remote({"fold_t196": fold_t196, "counted_tasks": tasks, "dumps": full, "owns_spec_sha256": prep["result"]["spec_sha256"], "pins": pins})
    rec = {"label": "step-3 recipes on the A25 sim-honest runs (sealed): hashes, counts and gate results only", "git": git("rev-parse", "HEAD").decode().strip(),
           "stamp (PT)": stamp, "app": APP, "app_id": app.app_id, "pins": pins} | out
    dest = HERE / "results" / f"simh_step3_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1, default=str))
    print(dest, json.dumps({k: rec.get(k) for k in ("stage", "manifest_sha256", "counts", "n_scored")}), json.dumps({k: g["pass"] for k, g in out["gate"].items()}))
