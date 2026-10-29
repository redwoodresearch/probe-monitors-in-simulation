"""RULE.md "Own-family scoring of the scale-up sim-honest batches": own_frac{10,25,100}_L48 and own_a tight on the sim-honest runs
of each +311 batch (Q1-Q4) and of nm360, one pass-2 style message per batch (make_pass2_message.py, part kind `simh`). Same code
paths, transform and schema as run_famexp_simh_ownfam.py (09's A25 registration @ 51a612b29), with the scale-up routing:
  own_frac*: own_a_adapted = r36 by main (pass 2's per-probe rule): a main in Sol #36 -> its designated held-out model; a main in
      no fold model's training mains -> fold -1 (score NaN, run_score NaN, score_by_model / run_score_by_model the five models);
  own_a: no scale-up main is an own_a training main (T196 mains), so every row is fold -1 (2e's run_owna507_score.py rule; fold_r36
      informational) - checked by key against the five training-main lists.
Gates before scoring: the imported scorer bytes; own_a weight sha256s = owna507_2026-10-15_031024.json's committed hashes; staged
inputs, extraction metas and complete.json against the message; 2e's and pass 1's skip rules agree; the two positive controls
(own_frac L48 honest_b vs pass 1's open pack; own_a 2e's honest-B gate part 2). Output (09's names, Append 25h §9): convaudit_work:/simh/<batch>/famexp/ (own_frac<p>_L48__<R|C>.npz) and
/simh/<batch>/own_a/ (own_a_tight__avg_top2.npz, own_a_tight__cmd_A.npz), each with the d9 routed folder level (Append 25h §11) (per row key, unit,
call_index, score, score_by_model, fold, fold_r36; per run run_key, run_fold, run_score, run_score_by_model; R = mean, C = max per
model) and manifest_<famexp|own_a>.json last in each; create-only, every target checked before any write. Runs without rows are listed by key, never imputed. Hashes, counts and gate
results only are returned or printed.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run <this file> \\
      --message <msg.json> --message-sha <sha256> --r36 <runs36.json> --out-famexp '/simh/<batch>/famexp/{folder}' --out-own-a '/simh/<batch>/own_a/{folder}' \\
      --gates <verify_roots out>,<verify_roots_secondary out> [--standing <READ_LOG commit>:<line sha256>,...]  (`_rlr` keys admitted by 09's standing rule)
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
APP = "probe-shift_aswin-surya_sub-p2probes_simh-scaleup_kill-2026-11-13"
BUILD = Path("/tmp/famexp_simh_scaleup_build")
P2 = ("9d006b058", "experiments/2026-10-14_p2_family_expansion/run_famexp_pass2.py", "1ad9b16f0e94e1dc248ecd4dbbe444cd6dda87e28c3d17c5532c7e184ce66e53")
OA = ("724e3dbf0", "experiments/2026-10-15_fpr_sweep_507_prep/run_owna507_score.py", "7ebfc56f2321fab09a0fe1c8e526154fb8ffb09247faf3f3ecab1b55d42acbc2")
CB = ("cc53afc3a", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json", "c002d5f822e49d7877f4e8b79a80ca9dd2e2f16c664b921d3ad12cb3da4c49ac")
OA_REC = ("54e18e68b", "experiments/2026-10-15_fpr_sweep_507_prep/results/owna507_2026-10-15_031024.json", "fa0c87be5d065842f6d4be58027ef8e5884b5ed5d53bc1a88ddf30d69366fe1a")
BMAN = ("69ea6174f", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json", "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc")
T196 = ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
Q2 = ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")
NEWMAINS = ("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e")
NM_BUILT = ("6feeee4b7", "experiments/2026-10-14_q3_new_mains_launch/config/nm_built.json", "93d9b538710b6bd930e7df01b0f5296d417550a0f0a35bcb6eb696166cf39a04")
R36_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
SPEC, FIT = "2026-10-14_163027", "2026-10-14_175636"
PIN_S = {"spec": "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61", "fit": "024fbb0ac04067a08c6dabc49e524a1b2841ab090b2c7f4acdeaf41345034745",
         "open_pack": ("pack_open_2026-10-14_180501", "a0765577c34182c48f7827492f9b745c76201da57c03088686d7f2f434c01e9f")}
OWN_FRAC = [f"own_frac{p}_L48" for p in (10, 25, 100)]
OWN_A_FILES = {"R": "own_a_tight__avg_top2", "C": "own_a_tight__cmd_A"}
TOL = 1e-9
PINS_REC = "experiments/2026-10-14_p2_family_expansion/results/simh_scaleup_pins_2026-10-15.json"  # 25f §2's pins, committed before any scoring
# 09's routed-folder grammar (25h §11 + `_rlr` @ 91485ce80); an `_rlr` folder holds only the ruled replacement keys of its parent batch
FOLDER_RES = (r"^simh_nm360_(s1|s2|s3|s4d|s4r|s5)_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$",  # 09 @ 502c1b05e, verbatim
              r"^simh311_q[1-4]_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$")  # 09 @ e2a473d2e, verbatim
RLR_RE = r"_sim_rlr([2-9]|[1-9][0-9]+)?_v1"  # a k-th replacement root: k = 1 is `_rlr`, k >= 2 is `_rlr<k>` (`_rlr1`, `_rlr0` fail FOLDER_RES)
RLR_K = {"simh_311|t293_media_processing": 2, "simh_nm360|nm107_media_processing": 2,  # 09 @ 502c1b05e §2 moves these to second launches; every other ruled key: k = 1
         "simh_311|t298_user_tracking": 2, "simh_311|t310_file_indexing": 2}  # 09 @ 322282c09: Q3_rlr launch 1 had no rows (EC2 stop), so Q3's keys live in rlr2
_RL, _TE = ("infrastructure (RateLimitError)", "13be06982"), ("unknown (terminal error)", "c10a14948")
RLR_KEYS = {"simh311_q1": {"simh_311|t462_file_indexing": (_RL[0], "c5420f2f5"), "simh_311|t242_model_registry": (_RL[0], "c5420f2f5"),
                           "simh_311|t237_media_processing": _TE, "simh_311|t300_user_tracking": _TE, "simh_311|t389_model_registry": _TE},
            "simh311_q2": {"simh_311|t271_slack_clone": (_RL[0], "c5420f2f5"), "simh_311|t410_file_indexing": (_RL[0], "c5420f2f5"),
                           "simh_311|t293_media_processing": (_TE[0], "dd93a41cd \u00a71"), "simh_311|t240_model_registry": (_RL[0], "81f75b763"),
                           "simh_311|t275_shared_dotfiles": (_RL[0], "81f75b763"), "simh_311|t457_media_processing": (_RL[0], "81f75b763")},
            "simh311_q3": {"simh_311|t298_user_tracking": _RL, "simh_311|t310_file_indexing": (_RL[0], "c5420f2f5")},
            "simh311_q4": {"simh_311|t361_file_indexing": _TE, "simh_311|t411_media_processing": _TE, "simh_311|t413_media_processing": _TE},
            "simh_nm360_s3": {"simh_nm360|nm103_file_indexing": _RL, "simh_nm360|nm134_media_processing": _TE},
            "simh_nm360_s4d": {"simh_nm360|nm107_media_processing": _TE, "simh_nm360|nm258_tiktok": _TE}}
RLR_ROOT_CHECKS = [("98ffa9e40", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/simh_rl_originals_roots_2026-10-15.json",  # d9's admission gates (@ c5420f2f5 §4, @ c10a14948 §2)
                    "d3da1531e396390195fc5a84f49cfd6767b2a082950e079bc937c33fd1fb4b67"),
                   ("2b66c9dbe", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/simh_terminal_error_originals_roots_2026-10-15.json",
                    "af088cee5345f696a685f10e5be5e0df44bc6087cab2bf804adc8aebfccbe21c"),
                   ("a3987df44", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/simh_terminal_error_original_root_t293_2026-10-15.json",  # t293 (@ dd93a41cd \u00a71)
                    "ee64329a4e38d680f632700a590b2f357a4b9b360fb3c30c4a81f12163e4ff50"),
                   ("3a81548f5", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/simh_rl_originals_roots_q2_t240_t275_t457_2026-10-15.json",  # @ 81f75b763 \u00a71
                    "0c34124cb1757f838137075273aaf90f1c371c59eefe18125a5161f4cc02e7e7")]
READ_LOG = "experiments/2026-10-14_priority1_spec_09/READ_LOG.md"  # 09's standing rules (@ dd93a41cd §2, @ 81f75b763 §2): a STANDING_RULE_ADD line admits exactly its key
ADD_RE = r"- STANDING_RULE_ADD key=(\S+\|\S+) class=(unknown|infrastructure) m_r=([0-9]+) evidence=([0-9a-f]{7,40}) rootcheck=([0-9a-f]{7,40}) root=(\S+)"
ADD_CLASS = {"unknown": ("unknown (terminal error; standing rule)", "dd93a41cd \u00a72"), "infrastructure": ("infrastructure (RateLimitError; standing rule)", "81f75b763 \u00a72")}

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
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0", "torch==2.8.0", "transformers==5.19.0", "jinja2")
       .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift").add_local_dir(BUILD, "/root/x2e"))
VOLS = {"/out": OUTV, "/s": modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only(),
        "/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only(),
        "/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only(),
        "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(),
        "/d": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", version=2).read_only()}


def _plain(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _runs(key, fold, sbm, readout):
    """Per model on raw logits: R = mean of the run's rows, C = max (run_famexp_pack._save @ 47fab1bac lines 64-68); run_score = the
    designated model's, NaN on fold -1 (09's A25 schema @ 51a612b29)."""
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
    return keys, rf, np.where(rf >= 0, rsm[np.arange(len(keys)), np.clip(rf, 0, 4)], np.nan), rsm


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
    if wsha != {k: v for k, v in want_w.items() if not k.startswith("C|")} or \
            {nm: cfg["own_a"]["weights_C"][nm]["sha256"] for nm in ("tight", "outer")} != {nm: want_w[f"C|{nm}"] for nm in ("tight", "outer")}:
        raise SystemExit("own_a weight sha256s differ from owna507_2026-10-15_031024.json's committed gate_b.weights_sha256")
    choice, fold_t196 = cfg["own_a"]["choice"], cfg["own_a"]["fold_t196"]
    # the message's epochs: staged inputs and extraction metas (as pass 2's score_layer), skips by both rules
    msg = cfg["message"]
    eps, staged = msg["epochs"], {}
    for path, h in sorted({(e["staged"], e["staged_sha256"]) for e in eps}):
        raw = (Path("/d") / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise SystemExit(f"staged inputs {path}: sha256 differs from the message")
        staged[path] = raw
    sk1, sk2, bad, calls = {}, {}, [], {}
    for path, raw in staged.items():
        mine = {e["key"] for e in eps if e["staged"] == path}
        sk2 |= oa._skip_file(raw, mine, fid)[0]
        for x in (json.loads(y) for y in raw.decode().splitlines() if y):
            if x["key"] in mine:
                sk1[x["key"]] = [[r["irid"], r["call_index"]] for r in x["calls"] if r["kind"] == "call_end" and r["readout"] is not None and x["ids"][r["readout"]] in fixed]
                calls[x["key"]] = [(r["irid"], r["call_index"], r["kind"]) for r in x["calls"]]
    for e in eps:  # as pass 2's score_layer: staged coverage, meta identity, ordered calls, complete.json
        run_ = Path("/d") / e["run"]
        mt = json.loads((run_ / "meta" / e["arm"] / f"{hashlib.sha1(e['key'].encode()).hexdigest()[:20]}.json").read_text())
        done = json.loads((run_ / "complete.json").read_text())
        if e["key"] not in calls or mt["key"] != e["key"] or mt.get("inputs_sha256") != e["staged_sha256"] or \
                [(c["irid"], c["call_index"], c["kind"]) for c in mt["calls"]] != calls[e["key"]] or done.get("inputs_sha256", {}).get(f"{e['arm']}.jsonl") != e["staged_sha256"]:
            bad.append(e["key"])
    if bad or any(sorted(map(tuple, sk1[k])) != sorted(map(tuple, sk2[k])) for k in sk1):
        raise SystemExit(f"{len(bad)} epochs inconsistent with metas / complete.json, or the two skip rules disagree")
    # the positive controls (as run_famexp_simh_ownfam.py)
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
            gate[f"{v}|{readout}"] = {"rows_mine": len(mine), "rows_pack": len(ref), "max_abs_diff": dm,
                                      "pass": set(mine) == set(ref) and all(mine[u][1] == ref[u][1] for u in mine) and dm <= TOL}
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
    # scoring per output folder (one per staged set x meta arm), rows in memory, then create-only writes, manifest last
    tf_frac, fr36 = cfg["fold_own_frac"], cfg["fold_r36"]
    out = {}
    fos = sorted({e["folder"] for e in eps})
    if len(fos) > 1 and not all("{folder}" in cfg["out"][s] for s in ("famexp", "own_a")):
        raise SystemExit("several root folders in one message: the output templates must contain {folder}")
    targets = {(fo, s): Path("/out") / cfg["out"][s].format(folder=fo).lstrip("/") for fo in fos for s in ("famexp", "own_a")}
    if any(od.exists() and any(od.iterdir()) for od in targets.values()):
        raise SystemExit(f"an output folder already holds files, nothing written: {sorted(str(od) for od in targets.values() if od.exists() and any(od.iterdir()))}")
    for fo in fos:
        fe = [e for e in eps if e["folder"] == fo]
        joins = sorted(f"{e['world']}|{e['cell']}" for e in fe)
        files, counts, dropped, blobs = {}, {}, {}, {}

        def build(name, key, unit, ci, sbm, fold, readout, blobs=blobs, files=files, counts=counts, dropped=dropped, joins=joins):  # called in this iteration only
            if not np.isfinite(sbm).all():
                raise SystemExit(f"{name}: a non-finite score")
            score = np.where(fold >= 0, sbm[np.arange(len(fold)), np.clip(fold, 0, 4)], np.nan)
            rk, rf, rs, rsm = _runs(key, fold, sbm, readout)
            buf = io.BytesIO()
            np.savez(buf, key=key, unit=unit, call_index=ci, score=score, score_by_model=sbm, fold=fold, fold_r36=np.array([fr36[k.split("|", 1)[1]] for k in key.tolist()], np.int64),
                     run_key=rk, run_fold=rf, run_score=rs, run_score_by_model=rsm)
            blobs[f"{name}.npz"] = buf.getvalue()
            files[f"{name}.npz"] = hashlib.sha256(buf.getvalue()).hexdigest()
            counts[name] = {"n_runs": len(rk), "n_units": len(key)}
            dropped[name] = sorted(set(joins) - set(rk.tolist()))

        for readout in ("R", "C"):
            own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": {e["key"]: [e["world"], e["cell"]] for e in fe},
                   "arm_of": {e["key"]: e["arm"] for e in fe}, "run_of": {e["key"]: f"/d/{e['run']}" for e in fe}} | ({"skip": {e["key"]: sk1[e["key"]] for e in fe}} if readout == "C" else {})
            try:
                res, _ = p2._score_rows(own, 48, tf_frac, spec, fit, FIT, base, readout, {})
            except ValueError as e_:  # 2e's loader cannot concatenate zero arrays: no run of this folder has a row in this readout
                if "need at least one array to concatenate" not in str(e_):
                    raise
                res = {}
            for v in OWN_FRAC:
                if v in res:
                    names, units, ci, folds, S = res[v]
                    if any(folds[i] != tf_frac[k.split("|", 1)[1]] for i, k in enumerate(names.astype(str).tolist())):
                        raise SystemExit(f"{fo} {v}: a row is not routed by its main")
                    build(f"{v}__{readout}", names.astype(str), units, ci, S, folds, readout)
                else:
                    dropped[f"{v}__{readout}"] = joins
        sc = oa.score_all("/d", [{"key": e["key"], "world": e["world"], "task": e["cell"], "arm": e["arm"], "extraction_run": e["run"]} for e in fe],
                          {e["key"]: sk2[e["key"]] for e in fe}, choice, w)
        for ro, readout in (("avg_top2", "R"), ("cmd_A", "C")):
            s = sc[ro]
            if len(s["key"]):
                build(OWN_A_FILES[readout], s["key"].astype(str), s["unit"], s["call_index"], s["score_by_model"], np.full(len(s["key"]), -1, np.int64), readout)
            else:
                dropped[OWN_A_FILES[readout]] = joins
        written = {}
        for s, names_ in (("famexp", [f"{v}__{r}" for v in OWN_FRAC for r in ("R", "C")]), ("own_a", list(OWN_A_FILES.values()))):
            od = targets[(fo, s)]
            od.mkdir(parents=True, exist_ok=True)
            fs = {f"{n}.npz": files[f"{n}.npz"] for n in names_ if f"{n}.npz" in files}
            for fn in sorted(fs):
                (od / fn).open("xb").write(blobs[fn])
            man = {"label": f"{s} scores of a scale-up sim-honest root (RULE.md 'Own-family scoring of the scale-up sim-honest batches'); SEALED; hashes, counts and gate results only",
                   "folder": fo, "scorer_dir": str(od), "volume": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", "files": fs,
                   "counts": {n: counts[n] for n in names_ if n in counts}, "runs_without_rows": {n: dropped[n] for n in names_ if n in dropped}, "keys_staged": joins,
               "routing": {j: {"own_frac_fold": tf_frac[j.split("|", 1)[1]], "own_a_fold": -1, "fold_r36": fr36[j.split("|", 1)[1]]} for j in joins},
               "routing_evidence": cfg["routing_evidence"], "gate": gate, "weights_sha256": {"own_a": wsha, "own_a_committed_equal": True}, "message_sha256": cfg["message_sha256"],
               "inputs": cfg["pins"], "run_transform": {"R": "per model, mean of the run's reply rows", "C": "per model, max of the run's content-only command rows",
               "designated": "run_score = the designated model's; NaN on fold -1"},
                   "schema": "per row key, unit, call_index, score (NaN on fold -1), score_by_model (n x 5), fold, fold_r36 (informational); per run run_key, run_fold, run_score, run_score_by_model (N_runs x 5)"}
            mb = json.dumps(_plain(man), indent=1).encode()
            (od / f"manifest_{s}.json").open("xb").write(mb)
            written[s] = {"manifest": str(od / f"manifest_{s}.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest()}
            if s == "own_a":  # 2e's condition (as the nm360 own_a fill): a create-only writer sidecar beside the own_a files
                side = {"writer": {"job": "famexp own-family scoring of a scale-up sim-honest root (run_famexp_simh_scaleup.py)", "commit": cfg["writer"]["commit"],
                                   "app_id": cfg["writer"]["app_id"], "code": f"{OA[1]} @ {OA[0]} (sha256 {OA[2]}), score_all in-process"},
                        "gates": {"own_a_honest_b_gate_part2": {k: gate[k] for k in gate if k.startswith("own_a|")}, "finite": True,
                                  "weights_equal_committed": True, "routing": "every row fold -1 (no scale-up main is an own_a training main), checked by key"},
                        "manifest_own_a_sha256": written[s]["manifest_sha256"], "files": fs}
                sb = json.dumps(_plain(side), indent=1).encode()
                (od / "writer_famexp.json").open("xb").write(sb)
                written[s]["writer_sidecar_sha256"] = hashlib.sha256(sb).hexdigest()
        OUTV.commit()
        out[fo] = {"written": written, "counts": counts, "runs_without_rows": dropped, "files": files}
    return _plain({"stage": "written", "gate": gate, "folders": out})


@app.local_entrypoint()
def main(message: str, message_sha: str, r36: str, out_famexp: str, out_own_a: str, gates: str, standing: str = ""):
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout

    def show(pin):
        b = git("show", f"{pin[0]}:{pin[1]}")
        if hashlib.sha256(b).hexdigest() != pin[2]:
            raise SystemExit(f"{pin[1]}: sha256 differs from its pin")
        return json.loads(b)

    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/p2-family-expansion"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    for t_, sc_ in ((out_famexp, "famexp"), (out_own_a, "own_a")):  # 09 Append 25h §11: /simh/<batch>/<scorer>/<d9 routed folder>/, one per root
        if not re.fullmatch(r"/simh/(?:simh311_q[1-4]|simh_nm360_(?:s1|s2|s3|s4d|s4r|s5))/" + sc_ + r"/\{folder\}", t_):  # 09's batch names (25h §9)
            raise SystemExit(f"--out-{sc_.replace('_', '-')}: must be /simh/<batch>/{sc_}/{{folder}} (09 Append 25h §11)")
    gate_files, prim, sec = {}, [], {}
    for g_ in (x for x in gates.split(",") if x):  # the pre-launch root gates (verify_roots.py / verify_roots_secondary.py outputs), pinned by sha256
        gb_ = Path(g_).read_bytes()
        gate_files[Path(g_).name] = hashlib.sha256(gb_).hexdigest()
        txt, i_, dec = gb_.decode(), 0, json.JSONDecoder()
        while txt[i_:].strip():
            o_, n_ = dec.raw_decode(txt[i_:].lstrip())
            i_ = len(txt) - len(txt[i_:].lstrip()) + n_
            if "roots" in o_ and "nm_pins_agree" in o_:
                sec |= {r_["root"]: r_ | {"_file_pass": o_.get("pass")} for r_ in o_["roots"].values()}
            elif "root" in o_ and "pass" in o_:
                prim.append(o_)
    mb = Path(message).read_bytes()
    if hashlib.sha256(mb).hexdigest() != message_sha:
        raise SystemExit("message sha256 differs")
    msg = json.loads(mb)
    cb, oarec, bman, t196, q2, nm, nb = show(CB), show(OA_REC), show(BMAN), show(T196), show(Q2), show(NEWMAINS), show(NM_BUILT)
    if {c["key"]: c["main"] for c in nm["cells"]} != {c["key"]: c["main"] for c in nb["cells"]}:
        raise SystemExit("the new-main list and d9's built map disagree")
    main_of = {c["key"]: c["main"] for c in q2["cells"]} | {c["key"]: c["main"] for c in nm["cells"]}
    stratum = {c["key"]: "S_new" for c in q2["cells"]} | {c["key"]: "S_newmains" for c in nm["cells"]}
    d9_stratum = {"S_new": ("S_new", "simh_311"), "S_newmains": ("S_newmains", "simh_nm360")}  # d9's manifests name the sim-honest stratum after the world
    if any(e["cell"] not in main_of or e["stratum"] not in d9_stratum[stratum[e["cell"]]] for e in msg["epochs"]) or len({(e["world"], e["cell"]) for e in msg["epochs"]}) != len(msg["epochs"]):
        raise SystemExit("an epoch outside the +311 / new-main lists, a stratum mismatch, or a duplicate world|cell")
    want_world = {"S_new": "simh_311", "S_newmains": "simh_nm360"}  # d9 @ fd40f4db4 / 09 Append 25g @ e6ded1811
    batches = {out_famexp.split("/")[2], out_own_a.split("/")[2]}
    if len(batches) != 1 or any(not e["folder"].startswith(f"{next(iter(batches))}_") for e in msg["epochs"]):
        raise SystemExit("--out-famexp / --out-own-a name one batch, and every routed folder of the message must belong to it")
    if any(e["world"] != want_world[stratum[e["cell"]]] or not str(e["arm"]).startswith("sim_honest") or not e["folder"].endswith("__sim_honest") for e in msg["epochs"]):
        raise SystemExit("not a sim-honest message: world simh_311 (S_new) / simh_nm360 (S_newmains), sim_honest arm and folder required")
    if any(not any(re.fullmatch(x_, e["folder"]) for x_ in FOLDER_RES) for e in msg["epochs"]):
        raise SystemExit("a routed folder outside 09's grammar (25h §11, `_rlr` @ 91485ce80, `_rlr<k>` @ 502c1b05e / e2a473d2e)")
    rlr = {}
    if standing and not any(re.search(RLR_RE, e["folder"]) for e in msg["epochs"]):
        raise SystemExit("--standing applies only to an `_rlr` message")
    if any(re.search(RLR_RE, e["folder"]) for e in msg["epochs"]):  # 09 @ 91485ce80 §2 (+ `_rlr<k>` @ 502c1b05e): the regex admits nothing; ruled keys of the parent batch, staged out in d9's root check
        if not all(re.search(RLR_RE, e["folder"]) for e in msg["epochs"]):
            raise SystemExit("an `_rlr` message must hold only `_rlr` folders")
        allowed, checks = RLR_KEYS.get(next(iter(batches)), {}), {}
        for c_, p_, s_ in RLR_ROOT_CHECKS:
            if not any(x.strip().startswith("origin/") for x in git("branch", "-r", "--contains", c_).decode().splitlines()):
                raise SystemExit(f"{c_} (d9's root check) is not on origin")
            checks |= {r_["key"]: r_ | {"_check": c_} for r_ in show((c_, p_, s_))["rows"]}
        root_of, added = {}, {}
        for s_ in (x for x in standing.split(",") if x):  # <READ_LOG commit>:<sha256 of the line>: one pinned STANDING_RULE_ADD line per key
            c_, h_ = s_.split(":")
            hit = [x.strip() for x in git("show", f"{c_}:{READ_LOG}").decode().splitlines() if hashlib.sha256(x.strip().encode()).hexdigest() == h_]
            m_ = re.fullmatch(ADD_RE, hit[0]) if len(hit) == 1 else None
            if not m_ or any(not any(x.strip().startswith("origin/") for x in git("branch", "-r", "--contains", y_).decode().splitlines()) for y_ in (c_, m_[4], m_[5])):
                raise SystemExit(f"--standing {s_}: no single STANDING_RULE_ADD line with that sha256 in {READ_LOG} @ {c_}, or it, its evidence or its root check is not on origin")
            b_, c1_ = m_[1].split("|", 1)  # 09 @ 87cbaca6a: key=<batch>|<cell>; lines before it: key=<world>|<cell>
            w_ = "simh_311" if b_ == "simh_311" or b_.startswith("simh311_") else "simh_nm360" if b_ == "simh_nm360" or b_.startswith("simh_nm360_") else None
            if w_ is None or (b_ not in ("simh_311", "simh_nm360") and b_ != next(iter(batches))):
                raise SystemExit(f"--standing {s_}: key {m_[1]} names another batch or no sim-honest world")
            k1 = f"{w_}|{c1_}"
            rc_ = [r_ | {"_check": m_[5]} for p_ in git("show", "--format=", "--name-only", m_[5]).decode().split() if p_.endswith(".json")
                   for r_ in json.loads(git("show", f"{m_[5]}:{p_}")).get("rows", []) if r_.get("key") == k1]
            if len(rc_) != 1 or k1 in allowed or k1 in added:
                raise SystemExit(f"--standing {s_}: root check {m_[5]} lists {k1} {len(rc_)} times, or the key is admitted twice")
            added[k1], checks[k1], root_of[k1] = (ADD_CLASS[m_[2]][0], f"{ADD_CLASS[m_[2]][1]}; READ_LOG @ {c_}"), rc_[0], m_[6]
            added[k1] += ({"line_sha256": h_, "line_key": m_[1], "m_r": int(m_[3]), "evidence": m_[4]},)
        for e in msg["epochs"]:
            k_, o_ = f"{e['world']}|{e['cell']}", checks.get(f"{e['world']}|{e['cell']}")
            a_ = allowed.get(k_) or added.get(k_)
            kk_ = int(re.search(RLR_RE, e["folder"])[1] or 1)
            if k_ in allowed and kk_ != RLR_K.get(k_, 1):
                raise SystemExit(f"`_rlr` key {k_}: ruled to replacement root k={RLR_K.get(k_, 1)} (09 @ 502c1b05e), routed in k={kk_}")
            if not a_ or not o_ or o_["staged"] is not False or not o_["routed_folder"].startswith(f"{next(iter(batches))}_") \
                    or (k_ in root_of and re.sub(r"_try[0-9]+__", "__", e["folder"]) != root_of[k_]):
                raise SystemExit(f"`_rlr` key {k_}: not a ruled or standing-rule key of this batch and root, or no staged-out original in d9's root check")
            rlr[k_] = {"class": a_[0], "ruling": a_[1], "grammar": "91485ce80 + 502c1b05e", "replacement_root_k": kk_, "original_root_check": o_["_check"], "original_folder": o_["routed_folder"]} | (a_[2] if len(a_) > 2 else {})
        if set(added) - set(rlr):
            raise SystemExit(f"--standing names keys absent from the message: {sorted(set(added) - set(rlr))}")
    roots = {(p_["run"], p_["staged_set"]) for p_ in msg["provenance"]}
    okp = {r_["root"] for r_ in prim if r_.get("pass") is True and not r_.get("fails")}
    oks = {(r_["root"], r_.get("staged_set")) for r_ in sec.values() if r_.get("pass") is True and r_.get("_file_pass") is True}
    if not roots or any(r_ not in okp for r_, _ in roots) or any(x_ not in oks for x_ in roots):
        raise SystemExit(f"--gates: every root of the message needs a passing verify_roots.py and verify_roots_secondary.py output (roots {sorted(roots)})")
    r36b = Path(r36).read_bytes()
    if hashlib.sha256(r36b).hexdigest() != R36_SHA:
        raise SystemExit("runs36: sha256 differs")
    rows36 = json.loads(r36b)
    fom = {}
    for r in rows36:
        if fom.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    t196_main = {c["key"]: c["main"] for c in t196["cells"]}
    fold_t196 = {c: fom[m] for c, m in t196_main.items()}
    cells = sorted({e["cell"] for e in msg["epochs"]})
    tf_frac = {c: fom.get(main_of[c], -1) for c in cells}  # pass 2's own/adapted rule (own_a_adapted = r36)
    frac_train = [{t196_main[c] for c, g in fold_t196.items() if g != f} | {r["main_task"] for r in rows36 if r["fold"] != f} for f in range(5)]
    owna_train = [{t196_main[c] for c, g in fold_t196.items() if g != f} for f in range(5)]  # 2e's: T196 mains, r36 folds
    bad = [c for c in cells if (tf_frac[c] >= 0 and main_of[c] in frac_train[tf_frac[c]]) or (tf_frac[c] < 0 and any(main_of[c] in x for x in frac_train))
           or any(main_of[c] in x for x in owna_train)]
    if bad:
        raise SystemExit(f"STOP: a cell would be scored by a model that trained on its main: {bad[:3]}")
    for c_ in ("fd163c4c7", "5aa9f32bf"):  # 09's Append 25h and 0d's attestation (25h §10): the pre-scoring gate, on origin
        if not any(x.strip().startswith("origin/") for x in git("branch", "-r", "--contains", c_).decode().splitlines()):
            raise SystemExit(f"{c_} (Append 25h / its attestation) is not on origin")
    pmap = json.loads(git("show", f"HEAD:{PINS_REC}"))["25f_s2_pins"]["per_probe_routing_maps"]["map"]  # the committed, independently derived maps
    if any(pmap[c]["own_frac"] != tf_frac[c] or pmap[c]["own_a"] != -1 for c in cells):
        raise SystemExit("the routing differs from the committed per-probe routing map (25f §2)")
    evidence = {"own_frac": {"mapped": sum(f >= 0 for f in tf_frac.values()), "fold_minus1_excluded_by_all_five": sum(f < 0 for f in tf_frac.values())},
                "own_a": {"fold_minus1_excluded_by_all_five": len(cells), "training_mains_sha256": [hashlib.sha256(json.dumps(sorted(x)).encode()).hexdigest() for x in owna_train]}}
    choice = cb["layer_choice_own_a"]
    if choice != [[49, 50], [49, 50], [50, 51], [49, 50], [49, 50]]:
        raise SystemExit("layer_choice_own_a differs from 2e's")
    tav, cmd_a = cb["saved"]["tight"]["avg"], cb["saved"]["cmd"]["cmdA"]
    need = [f"{L}|{f}" for f in range(5) for L in choice[f]]
    own_a = {"choice": choice, "fold_t196": fold_t196, "weights_R": {k: {"path": tav[k]["weights"], "sha256": tav[k]["weights_sha256"]} for k in need},
             "weights_C": {"tight": cmd_a["own_a_tight_weights"], "outer": cmd_a["own_a_weights"]},
             "ref_R": {k: {"path": tav[k]["rows"], "sha256": tav[k]["rows_sha256"]} for k in need},
             "ref_C": {"path": cmd_a["rows|own_a_tight"]["path"], "sha256": cmd_a["rows|own_a_tight"]["sha256"]},
             "committed_weights_sha256": oarec["gate_b"]["weights_sha256"],
             "hb_epochs": [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_file", "staged_inputs_sha256")} for e in bman["epochs"] if e["world"] == "honest_b"]}
    pins = {k: dict(zip(("commit", "path", "sha256"), v)) for k, v in (("p2_scorer", P2), ("owna507", OA), ("cb_record", CB), ("owna507_record", OA_REC), ("correctedb_manifest", BMAN),
                                                                       ("t196", T196), ("plus311", Q2), ("new_mains", NEWMAINS), ("nm_built", NM_BUILT))} | {
        "runs36": {"sha256": R36_SHA}, "message": {"path": Path(message).name, "sha256": message_sha, "d9_arm": msg.get("d9_arm")},
        "pins_25f_s2": {"path": PINS_REC, "sha256": hashlib.sha256(git("show", f"HEAD:{PINS_REC}")).hexdigest()}, "append_25h": "fd163c4c7", "attestation_25h_s10": "5aa9f32bf",
        "pass1": {"spec": [SPEC, PIN_S["spec"]], "fit": [FIT, PIN_S["fit"]], "open_pack": list(PIN_S["open_pack"])}, "message_provenance": msg.get("provenance"), "rlr_replacements": rlr}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = run.remote({"message": msg, "message_sha256": message_sha, "own_a": own_a, "fold_own_frac": tf_frac, "fold_r36": {c: fom.get(main_of[c], -1) for c in cells},
                      "routing_evidence": evidence, "out": {"famexp": out_famexp, "own_a": out_own_a}, "pins": pins | {"root_gates": gate_files},
                      "writer": {"commit": git("rev-parse", "HEAD").decode().strip(), "app_id": app.app_id or ""}})
    rec = {"label": "own-family scoring of a scale-up sim-honest batch: hashes, counts and gate results only", "git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp,
           "app": APP, "app_id": app.app_id, "pins": {k: v for k, v in pins.items() if k != "message_provenance"}, "routing_evidence": evidence} | out
    dest = HERE / "results" / f"simh_scaleup_{msg.get('d9_arm')}_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1, default=str))
    print(dest, json.dumps({fo: {"manifests": {s: w_["manifest_sha256"] for s, w_ in x["written"].items()}, "counts": x["counts"], "drops": {k: v for k, v in x["runs_without_rows"].items() if v}}
                            for fo, x in out["folders"].items()}), json.dumps({k: g["pass"] for k, g in out["gate"].items()}))
