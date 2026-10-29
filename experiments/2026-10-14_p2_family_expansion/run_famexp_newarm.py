"""RULE.md "New-arm scoring" (no fitting): the own-family probes on arms they never saw.
  own_frac{10,25,100}_L{16,32,48}: pass 1's frozen fold models (fit 2026-10-14_175636), rows and arithmetic of pass 2's scorer
      (run_famexp_pass2._score_rows @ P2[0], imported from the pinned bytes).
  own_a: 2e's tight weights of correctedb_scores_2026-10-14_164934 (@ cc53afc3a) through 2e's run_owna507_score.score_all
      (@ OA[0], imported from the pinned bytes); 2e's skip rule (_skip_file with the tokenizer's fixed call-token ids).
Routing: every run is scored by the fold model that HELD OUT its main (T196 cell -> Sol #36 fold of its main; own_a's T196
training-fold map is the same r36 map, checked). Gates before any write: (1) own_frac: T196 honest_b rebuilt through this path
vs pass 1's open pack, every row within 1e-9 (score and fold); (2) own_a: 2e's honest-B gate part 2 (score_all on corrected B's
honest_b vs 164934's stored rows, within 1e-9). Output per probe x readout, one npz: per row key, unit, call_index, score (the
designated model's), score_by_model (rows x 5), fold; per run run_key, run_fold, run_score (R = mean, C = max of the designated
rows, as run_famexp_pack._save @ 47fab1bac lines 64-68), run_score_by_model (runs x 5, same transform per model); plus
manifest_famexp.json LAST. Files are created exclusively. Keys, shas and counts only are returned or printed: never a score.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run <this file> --mode relaunch --r36 <runs36.json>
"""

import hashlib
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
APP = "probe-shift_aswin-surya_sub-p2probes_newarm_kill-2026-11-13"
BUILD = Path("/tmp/famexp_newarm_build")
P2 = ("9d006b058", "experiments/2026-10-14_p2_family_expansion/run_famexp_pass2.py", "1ad9b16f0e94e1dc248ecd4dbbe444cd6dda87e28c3d17c5532c7e184ce66e53")
OA = ("724e3dbf0", "experiments/2026-10-15_fpr_sweep_507_prep/run_owna507_score.py", "7ebfc56f2321fab09a0fe1c8e526154fb8ffb09247faf3f3ecab1b55d42acbc2")
CB = ("cc53afc3a", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json", "c002d5f822e49d7877f4e8b79a80ca9dd2e2f16c664b921d3ad12cb3da4c49ac")
BMAN = ("69ea6174f", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json", "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
R36_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
SPEC, FIT = "2026-10-14_163027", "2026-10-14_175636"
PIN_S = {"spec": "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61", "fit": "024fbb0ac04067a08c6dabc49e524a1b2841ab090b2c7f4acdeaf41345034745",
         "open_pack": ("pack_open_2026-10-14_180501", "a0765577c34182c48f7827492f9b745c76201da57c03088686d7f2f434c01e9f")}
OWN_FRAC = [f"own_frac{p}_L{L}" for p in (10, 25, 100) for L in (16, 32, 48)]
OWN_A_FILES = {"R": "own_a_tight__avg_top2", "C": "own_a_tight__cmd_A"}  # 2e's per-row file names (the 507 fork reads them)
TOL = 1e-9
RH = "/ref/relaunch/famexp"  # 09 @ 51a612b29
RELAUNCH = {"volume": "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14", "run": "rh1e4_extract_v1", "arm": "honest_b",
            "staged": "rh1e4_stage_v1/inputs/honest_b.jsonl", "staged_sha": "8a150d13903df37c974e6d2ffd3f6fe4b114fb1ad1a328858a21d50ff4eabb78",
            "counted": ("9cd23708e", "experiments/2026-10-14_q3_q35_realhonest_1e4/config/manifest_rh1e4_eval_upload_v1.json",
                        "9785118d230c4ec4e67bb73ea6821547c672705689c8bc3d519bef56b49068ac"),
            "stage_record": ("7bbc0e3df", "experiments/2026-10-14_q3_q35_realhonest_1e4/results/rh_stage_rh_2026-10-14_231559.json",
                             "8d8ad3507ae22b4a43934c18b501acbb2a765e00f4f4edef91a58d12778bede9"),
            "extract_record": ("27491ffbc", "experiments/2026-10-14_q3_q35_realhonest_1e4/results/extract_rh_rh1e4_extract_v1_a1_2026-10-14_232922.json",
                               "7d74617fa69a8ff7391be1c163ca4c3c89030c83f631b8a36a439f993634c895"),
            "expected_dropped": ["honest_r2|t27_slack_clone", "honest_r4|t27_slack_clone", "honest_r2|t83_auto_workflow", "honest_r3|t83_auto_workflow",
                                 "honest_r5|t83_auto_workflow"], "expected_scored": 958}

if modal.is_local():  # the pinned bundle: pass 1's 2e row code (run_famexp.PINNED, byte-checked) + the two scorers' modules
    sys.path.insert(0, str(HERE))
    import run_famexp

    BUILD.mkdir(exist_ok=True)
    for name, (commit, path, want) in list(run_famexp.PINNED.items()) + [("p2pass2.py", P2), ("owna507.py", OA)]:
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
        if hashlib.sha256(raw).hexdigest() != want:
            raise SystemExit(f"{name}: pinned bytes differ ({commit}:{path})")
        (BUILD / name).write_bytes(raw)

app = modal.App(APP)
VOLS = {"/out": modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2),
        "/s": modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only(),
        "/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only(),
        "/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only(),
        "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(),
        "/rh": modal.Volume.from_name(RELAUNCH["volume"]).read_only()}
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0", "torch==2.8.0", "transformers==5.19.0", "jinja2")
       .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift").add_local_dir(BUILD, "/root/x2e"))


def _mods():
    sys.path.insert(0, "/root/x2e")
    import owna507 as oa
    import p2pass2 as p2

    for m, (_, _, want) in ((p2, P2), (oa, OA)):  # the imported bytes are the pinned ones
        if hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() != want:
            raise SystemExit(f"{m.__name__}: imported bytes differ from the pin")
    return p2, oa


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _runs(key, fold, score, sbm, readout):
    """Run level of the designated rows and of each model: R = mean, C = max (run_famexp_pack._save @ 47fab1bac lines 64-68)."""
    import numpy as np

    keys, inv = np.unique(key, return_inverse=True)
    rf = np.full(len(keys), -2, np.int64)
    rf[inv] = fold
    if any(len(set(fold[inv == i].tolist())) != 1 for i in range(len(keys))):
        raise SystemExit("a run has rows from two folds")
    cols = np.column_stack([score, sbm])
    if readout == "R":
        out = np.stack([np.bincount(inv, weights=cols[:, j]) / np.bincount(inv) for j in range(cols.shape[1])], axis=1)
    else:
        out = np.full((len(keys), cols.shape[1]), -np.inf)
        for j in range(cols.shape[1]):
            np.maximum.at(out[:, j], inv, cols[:, j])
    out[rf < 0, 0] = np.nan  # fold -1: no designated run score (09: NaN for fold -1)
    return keys, rf, out[:, 0], out[:, 1:]


def _plain(x):
    """JSON-plain copy of a return value (numpy scalars -> Python), so a client without numpy can read it."""
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _designated(sbm, fold):
    import numpy as np

    return np.where(fold >= 0, sbm[np.arange(len(fold)), np.clip(fold, 0, 4)], np.nan)


@app.function(image=img, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2, volumes=VOLS)
def run(cfg: dict) -> dict:
    import numpy as np
    import torch

    p2, oa = _mods()
    base = Path("/s/famexp") / SPEC
    if _sha(base / f"spec_{SPEC}.json") != PIN_S["spec"] or _sha(base / f"fit_{FIT}.json") != PIN_S["fit"]:
        raise SystemExit("pass 1's spec or fit manifest differs from its pin")
    spec = json.loads((base / f"spec_{SPEC}.json").read_text())
    fit = [x for g in json.loads((base / f"fit_{FIT}.json").read_text())["groups"] for x in g["fits"]]
    pk_dir = base / PIN_S["open_pack"][0]
    if _sha(pk_dir / "manifest.json") != PIN_S["open_pack"][1]:
        raise SystemExit("pass 1's open pack manifest differs from its pin")
    pk_files = {Path(x["npz"]["path"]).name: x["npz"]["sha256"] for x in json.loads((pk_dir / "manifest.json").read_text())["variants"].values()}
    fot, fixed = spec["fot"], set(spec["fixed_ids"])
    _, fid = oa._fixed_ids()
    if set(fid) != fixed:
        raise SystemExit("2e's tokenizer fixed call-token ids differ from pass 1's fixed_ids")
    w, wsha = oa.load_weights(cfg["own_a"])
    choice, fold_t196 = cfg["own_a"]["choice"], cfg["own_a"]["fold_t196"]
    if any(fold_t196[t] != f for t, f in fot.items() if t in fold_t196):
        raise SystemExit("own_a's T196 training-fold map differs from pass 1's fot (r36 by main)")
    torch.set_num_threads(8)
    gate = {}
    # gate 1: own_frac through this path on T196 honest_b vs pass 1's open pack (score and fold, row for row)
    hb = {k: v for k, v in spec["map"].items() if v[0] == "honest_b"}
    for readout in ("R", "C"):
        own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": hb, "arm_of": {k: spec["arm_of"][k] for k in hb},
               "run_of": {k: spec["run_of"][k] for k in hb}} | ({"skip": {k: spec["skip"][k] for k in hb}} if readout == "C" else {})
        for L in (16, 32, 48):
            res, _ = p2._score_rows(own, L, fot, spec, fit, FIT, base, readout, {})
            for v in (x for x in OWN_FRAC if x.endswith(f"_L{L}")):
                names, units, ci, folds, S = res[v]
                fn = f"{v}__{readout}.npz"
                b_ = (pk_dir / fn).read_bytes()
                if hashlib.sha256(b_).hexdigest() != pk_files[fn]:
                    raise SystemExit(f"open pack {fn}: sha256 differs from its manifest")
                z = np.load(io.BytesIO(b_))
                sel = np.char.startswith(z["key"].astype(str), "honest_b|")
                ref = {(k, int(u), int(c)): (float(s), int(f)) for k, u, c, s, f in zip(z["key"][sel].astype(str), z["unit"][sel], z["call_index"][sel], z["score"][sel], z["fold"][sel])}
                mine = {(k, int(u), int(c)): (float(s), int(f)) for k, u, c, s, f in zip(names.astype(str), units, ci, _designated(S, folds), folds)}
                dm = max((abs(mine[u][0] - ref[u][0]) for u in set(mine) & set(ref)), default=float("inf"))
                gate[f"{v}|{readout}"] = {"rows_mine": len(mine), "rows_pack": len(ref), "same_rows": set(mine) == set(ref),
                                          "same_folds": all(mine[u][1] == ref[u][1] for u in set(mine) & set(ref)), "max_abs_diff": dm}
                gate[f"{v}|{readout}"]["pass"] = gate[f"{v}|{readout}"]["same_rows"] and gate[f"{v}|{readout}"]["same_folds"] and dm <= TOL
    # gate 2: own_a, 2e's honest-B gate part 2 (score_all on corrected B's honest_b vs 164934's stored rows)
    hb_eps = cfg["own_a"]["hb_epochs"]
    sk_b = {}
    for rel, h in sorted({(e["staged_inputs_file"], e["staged_inputs_sha256"]) for e in hb_eps}):
        raw = (Path("/b") / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"{rel}: staged sha256")
        sk_b |= oa._skip_file(raw, {e["key"] for e in hb_eps if e["staged_inputs_file"] == rel}, fid)[0]
    torch.set_num_threads(4)  # 2e's production configuration
    sc = oa.score_all("/b", hb_eps, sk_b, choice, w)
    ref_rows = lambda e: np.load(io.BytesIO(_chk(Path("/sol") / e["path"], e["sha256"])))
    zc = ref_rows(cfg["own_a"]["ref_C"])
    for ro, readout in (("avg_top2", "R"), ("cmd_A", "C")):
        s = sc[ro]
        des = [s["score_by_model"][i, fold_t196[k.split("|", 1)[1]]] for i, k in enumerate(s["key"].tolist())]
        if ro == "cmd_A":
            ref = {}
            for k, i, c, v in zip(zc["key"], zc["irid"], zc["call_index"], zc["score"]):
                if (str(k), str(i), int(c)) in ref:
                    raise SystemExit("own_a C reference: a duplicate (key, irid, call_index) (2e's gate)")
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
            raise SystemExit(f"own_a {ro}: a duplicate (key, irid, call_index) in the scale-up path (2e's gate)")
        mine, ref_hb = dict(zip(ids, des)), {u for u in ref if u[0].startswith("honest_b|")}
        dm = max((abs(mine[u] - ref[u]) for u in set(mine) & ref_hb), default=float("inf"))
        gate[f"own_a|{readout}"] = {"rows_mine": len(mine), "rows_reference": len(ref_hb), "same_rows": set(mine) == ref_hb, "max_abs_diff": dm,
                                    "pass": set(mine) == ref_hb and len(mine) > 0 and dm <= TOL}
    if not all(g["pass"] for g in gate.values()):  # score-free diagnostics only (row counts, same-row flags, max |diff|)
        raise SystemExit(f"positive-control gate failed, nothing written: {json.dumps({k: g for k, g in gate.items() if not g['pass']}, default=str)}")
    # the arm: staged epochs keyed by the counted-key manifest (09's key A), metas checked, routed by the held-out fold of the main
    a = cfg["arm"]
    raw = (Path("/rh") / a["staged"]).read_bytes()
    done = json.loads((Path("/rh") / a["run"] / "complete.json").read_text())
    if hashlib.sha256(raw).hexdigest() != a["staged_sha"] or done.get("inputs_sha256", {}).get(Path(a["staged"]).name) != a["staged_sha"]:
        raise SystemExit("staged inputs differ from their pin or from complete.json")
    staged = [json.loads(x) for x in raw.decode().splitlines() if x]
    eps, fails = [], []
    for ep in staged:
        ent = a["entries"].get(ep["key"])
        if ent is None:
            fails.append(f"{ep['task']}: staged key not in the counted-key manifest")
            continue
        rm = re.fullmatch(r"(.+)_r([1-5])", ep["task"])
        if not rm or ent["world"] != f"honest_r{rm.group(2)}" or ent["task"] != ep["task"] or rm.group(1) not in fot:
            fails.append(f"{ep['task']}: not <base>_r<i> for world {ent['world']}, or base not a T196 cell")
            continue
        m = json.loads((Path("/rh") / a["run"] / "meta" / a["arm"] / f"{hashlib.sha1(ep['key'].encode()).hexdigest()[:20]}.json").read_text())
        if m["key"] != ep["key"] or m.get("inputs_sha256") != a["staged_sha"] or [(c["irid"], c["call_index"], c["kind"]) for c in ep["calls"]] != \
                [(c["irid"], c["call_index"], c["kind"]) for c in m["calls"]]:
            fails.append(f"{ep['task']}: meta differs from the staged epoch")
            continue
        eps.append({"key": ep["key"], "world": ent["world"], "task": rm.group(1), "arm": a["arm"], "extraction_run": a["run"], "launch_task": ep["task"],
                    "skip_pass1": [[c["irid"], c["call_index"]] for c in ep["calls"] if c["kind"] == "call_end" and c["readout"] is not None and ep["ids"][c["readout"]] in fixed]})
    joins = [f"{e['world']}|{e['task']}" for e in eps]
    if len(set(joins)) != len(joins) or len(eps) != len(staged) or set(a["entries"]) - {e["key"] for e in staged} != set(a["excluded"]) or fails:
        raise SystemExit(f"arm keys: {len(eps)} of {len(staged)} staged usable, duplicates {len(joins) - len(set(joins))}, coverage or fails {fails[:4]}")
    sk_oa = oa._skip_file(raw, {e["key"] for e in eps}, fid)[0]
    if any(sorted(map(tuple, sk_oa[e["key"]])) != sorted(map(tuple, e["skip_pass1"])) for e in eps):
        raise SystemExit("2e's skip rule and pass 1's skip rule disagree on an arm epoch")
    tf = {e["task"]: fot[e["task"]] for e in eps}
    od = Path("/out") / a["out_dir"].lstrip("/")
    if od.exists() and any(od.iterdir()):
        raise SystemExit(f"{od} already holds files: never overwritten")
    files, counts, dropped, blobs = {}, {}, {}, {}

    def write(name, key, unit, ci, sbm, fold, readout):  # built in memory; written only after the coverage check below
        score = _designated(sbm, fold)
        if not (np.isfinite(sbm).all() and np.isfinite(score).all()):
            raise SystemExit(f"{name}: non-finite score")
        if any(fold[i] != fot[k.split("|", 1)[1]] for i, k in enumerate(key.tolist())):
            raise SystemExit(f"{name}: a row is not routed to the fold model that held out its main")
        rk, rf, rs, rsm = _runs(key, fold, score, sbm, readout)
        buf = io.BytesIO()
        np.savez(buf, key=key, unit=unit, call_index=ci, score=score, score_by_model=sbm, fold=fold, run_key=rk, run_fold=rf, run_score=rs, run_score_by_model=rsm)
        blobs[f"{name}.npz"] = buf.getvalue()
        files[f"{name}.npz"] = hashlib.sha256(buf.getvalue()).hexdigest()
        counts[name] = {"n_runs": len(rk), "n_units": len(key)}
        dropped[name] = sorted(set(joins) - set(rk.tolist()))

    for readout in ("R", "C"):
        own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": {e["key"]: [e["world"], e["task"]] for e in eps},
               "arm_of": {e["key"]: a["arm"] for e in eps}, "run_of": {e["key"]: f"/rh/{a['run']}" for e in eps}} | (
            {"skip": {e["key"]: e["skip_pass1"] for e in eps}} if readout == "C" else {})
        for L in (16, 32, 48):
            res, ws = p2._score_rows(own, L, tf, spec, fit, FIT, base, readout, {})
            wsha |= ws
            for v in (x for x in OWN_FRAC if x.endswith(f"_L{L}")):
                names, units, ci, folds, S = res[v]
                write(f"{v}__{readout}", names.astype(str), units, ci, S, folds, readout)
    sc = oa.score_all("/rh", eps, sk_oa, choice, w)
    for ro, readout in (("avg_top2", "R"), ("cmd_A", "C")):
        s = sc[ro]
        fold = np.array([fold_t196[k.split("|", 1)[1]] for k in s["key"].tolist()], np.int64)
        write(OWN_A_FILES[readout], s["key"].astype(str), s["unit"], s["call_index"], s["score_by_model"], fold, readout)
    exp_drop = set(a["expected_dropped"])
    bad = {n: {"dropped_not_expected": sorted(set(d) - exp_drop), "expected_not_dropped": sorted(exp_drop - set(d))} for n, d in dropped.items() if set(d) != exp_drop}
    if bad or any(c["n_runs"] != a["expected_scored"] for c in counts.values()) or len(blobs) != 2 * (len(OWN_FRAC) + 1):
        raise SystemExit(f"coverage differs from the expected 958 / 5 drops, nothing written: {json.dumps(bad)} runs {sorted({c['n_runs'] for c in counts.values()})}")
    od.mkdir(parents=True, exist_ok=True)
    for fn, b_ in sorted(blobs.items()):
        (od / fn).open("xb").write(b_)
    routing = {j: {"main_fold": fot[j.split("|", 1)[1]], "basis": "T196 cell: the fold model that held out its main (Sol #36 fold of the main)"} for j in sorted(joins)}
    man = {"label": "own-family new-arm scores (RULE.md 'New-arm scoring'); keys, shas and counts in this manifest, never a score", "mode": cfg["mode"],
           "out_dir": a["out_dir"], "volume": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", "files": files, "counts": counts,
           "runs_without_rows": dropped, "keys_scored": sorted(set(joins) - set().union(*map(set, dropped.values()))), "routing": routing,
           "row_meta_of": {f"{e['world']}|{e['task']}": {"launch_task": e["launch_task"]} for e in eps}, "gate": gate, "weights_sha256": wsha,
           "fixed_call_token_ids_equal": True, "inputs": cfg["pins"], "run_transform": {"R": "mean of the run's rows", "C": "max of the run's rows",
           "code": "run_famexp_pack._save @ 47fab1bac lines 64-68 (designated column); the same per model column for run_score_by_model; fold -1: NaN"},
           "code": {"own_frac": f"{P2[1]} @ {P2[0]} _score_rows (sha256 {P2[2]})", "own_a": f"{OA[1]} @ {OA[0]} score_all (sha256 {OA[2]})"}}
    mb = json.dumps(man, indent=1, default=str).encode()
    (od / "manifest_famexp.json").open("xb").write(mb)
    VOLS["/out"].commit()
    return _plain({"stage": "written", "gate": gate, "manifest": str(od / "manifest_famexp.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest(), "files": files,
            "counts": counts, "runs_without_rows": dropped, "keys_scored": man["keys_scored"], "n_staged": len(eps)})


def _chk(p: Path, want: str) -> bytes:
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != want:
        raise SystemExit(f"{p}: sha256 differs from its pin")
    return b


@app.local_entrypoint()
def main(mode: str = "relaunch", r36: str = ""):
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/p2-family-expansion"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on its upstream")
    show = lambda pin: json.loads(_pin(git("show", f"{pin[0]}:{pin[1]}"), pin[2], pin[1]))
    if mode != "relaunch":
        raise SystemExit("only --mode relaunch is registered (the sim-honest mode waits for its own amendment)")
    cb, bman, t196 = show(CB), show(BMAN), show(T196)
    r36b = Path(r36).read_bytes()
    _pin(r36b, R36_SHA, "runs36")
    fom = {}
    for r in json.loads(r36b):
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    choice = cb["layer_choice_own_a"]
    if choice != [[49, 50], [49, 50], [50, 51], [49, 50], [49, 50]]:
        raise SystemExit("layer_choice_own_a differs from 2e's")
    tav, cmd_a = cb["saved"]["tight"]["avg"], cb["saved"]["cmd"]["cmdA"]
    need = [f"{L}|{f}" for f in range(5) for L in choice[f]]
    own_a = {"choice": choice, "fold_t196": {c["key"]: fom[c["main"]] for c in t196["cells"]},
             "weights_R": {k: {"path": tav[k]["weights"], "sha256": tav[k]["weights_sha256"]} for k in need},
             "weights_C": {"tight": cmd_a["own_a_tight_weights"], "outer": cmd_a["own_a_weights"]},
             "ref_R": {k: {"path": tav[k]["rows"], "sha256": tav[k]["rows_sha256"]} for k in need},
             "ref_C": {"path": cmd_a["rows|own_a_tight"]["path"], "sha256": cmd_a["rows|own_a_tight"]["sha256"]},
             "hb_epochs": [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_file", "staged_inputs_sha256")}
                           for e in bman["epochs"] if e["world"] == "honest_b"]}
    counted, stage, ext = show(RELAUNCH["counted"]), show(RELAUNCH["stage_record"]), show(RELAUNCH["extract_record"])
    if {str(s.get("device")) for s in ext["shards"]} != {"NVIDIA H200"} or ext["config"]["run"] != RELAUNCH["run"] or \
            ext["config"]["inputs_sha256"].get("honest_b.jsonl") != RELAUNCH["staged_sha"] or stage["c"]["assembled"]["staged_inputs_sha256"] != {"honest_b.jsonl": RELAUNCH["staged_sha"]}:
        raise SystemExit("extraction record (H200 on every shard, run, staged sha) or stage record differs")
    entries = {f"{x['path']}#e0": {"world": x["world"], "task": x["task"]} for x in counted["files"] if x["arm_dir"] == RELAUNCH["arm"]}
    if len(entries) != len(counted["files"]):
        raise SystemExit("counted-key manifest: duplicate or non-honest_b entries")
    arm = {"run": RELAUNCH["run"], "arm": RELAUNCH["arm"], "staged": RELAUNCH["staged"], "staged_sha": RELAUNCH["staged_sha"], "entries": entries,
           "excluded": sorted(stage["c"]["c_gate"]["excluded"]), "out_dir": RH, "expected_dropped": RELAUNCH["expected_dropped"], "expected_scored": RELAUNCH["expected_scored"]}
    pins = {k: {"commit": v[0], "path": v[1], "sha256": v[2]} for k, v in (("p2_scorer", P2), ("owna507", OA), ("cb_record", CB), ("correctedb_manifest", BMAN),
                                                                             ("t196", T196), ("counted_keys", RELAUNCH["counted"]), ("stage_record", RELAUNCH["stage_record"]),
                                                                             ("extract_record", RELAUNCH["extract_record"]))} | {
        "runs36": {"sha256": R36_SHA}, "pass1": {"spec": [SPEC, PIN_S["spec"]], "fit": [FIT, PIN_S["fit"]], "open_pack": list(PIN_S["open_pack"])},
        "dump": {"volume": RELAUNCH["volume"], "run": RELAUNCH["run"], "staged": RELAUNCH["staged"], "staged_sha256": RELAUNCH["staged_sha"]},
        "rule": "09 @ 51a612b29 (aswin/convergence-audit-09): /ref/relaunch/famexp, key A honest_r<i>|<base>, launch_task in row_meta; p2 Amendment 16 @ 70831ab3d"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = run.remote({"mode": mode, "own_a": own_a, "arm": arm, "pins": pins})
    exp = set(RELAUNCH["expected_dropped"])
    chk = {n: {"dropped_equal_expected": set(d) == exp, "dropped_minus_expected": sorted(set(d) - exp), "expected_minus_dropped": sorted(exp - set(d))}
           for n, d in (out.get("runs_without_rows") or {}).items()}
    rec = {"label": "own-family new-arm scoring (RULE.md 'New-arm scoring'): keys, shas and counts only", "git": git("rev-parse", "HEAD").decode().strip(),
           "stamp (PT)": stamp, "app": APP, "mode": mode, "pins": pins, "expected": {"dropped": sorted(exp), "scored": RELAUNCH["expected_scored"]},
           "drop_check": chk, "keys_scored_count": len(out.get("keys_scored", [])),
           "keys_scored_equal_expected": len(out.get("keys_scored", [])) == RELAUNCH["expected_scored"] and all(c["dropped_equal_expected"] for c in chk.values())} | out
    dest = HERE / "results" / f"newarm_{mode}_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1, default=str))
    print(dest, json.dumps({k: rec.get(k) for k in ("stage", "manifest_sha256", "counts", "keys_scored_count", "keys_scored_equal_expected", "drop_check")}),
          json.dumps({k: {x: g[x] for x in ("pass", "rows_mine", "same_rows")} for k, g in out["gate"].items()}))


def _pin(b: bytes, want: str, what: str) -> bytes:
    if hashlib.sha256(b).hexdigest() != want:
        raise SystemExit(f"{what}: sha256 differs from its pin")
    return b
