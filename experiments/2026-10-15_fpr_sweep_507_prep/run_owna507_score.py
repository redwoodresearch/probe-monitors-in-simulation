"""own_a tight on the 507 scale-up arms: BLIND scoring (2e's approval (a1), 2026-10-15 ~02:3x PT, under 0d's rule; RULE.md Q9).

Targets: config/owna_arms.json, the 58 Sol arm folders with policy runs (the 64 Sol folders minus the 6 no-policy-call nm360
main_v1 honest remainders). Every reply row (avg_top2: model f = mean of its two chosen layers) and command row (cmd_A, L63) is
scored by all five own_a tight fold models of correctedb_scores_2026-10-14_164934 (run_correctedb.py @ 18c21b556: per
(layer, fold) tight refits for the reply readout; per fold tight, else outer, at L63 for command A). own_a trained on corrected
B (T196) only, and no scale-up main is a T196 main, so every row is fold -1 (Append 21 section 4 / 21b): score NaN,
score_by_model (rows x 5) on every row. fold_r36 is informational only. Output: convaudit_work:/scaleup/<arm>/own_a/ in the Sol
per-row schema, files created exclusively, manifest_owna.json LAST.

Gates before any write:
  B (once): corrected B's own_a rows of 164934 rebuilt bit for bit (reply per (layer, fold); command per fold), and the
    scale-up code path on honest B within 1e-9 of them.
  per arm: H200 provenance as the Sol manifest records it (and every chunk read in it); staged keys = Sol's (counted by d9 per
    the Sol gate); rows aligned with Sol's per-row files (key, unit, call_index; fold_r36 = Sol's fold); every score finite;
    0 scale-up mains in own_a's training mains.
Firewall: shas, counts and gate results only. No score value is printed, summarized or returned.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_owna507_score.py [--only arm1,arm2]
"""

import hashlib
import io
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the container the script sits at /root
APP = "probe-shift_aswin-surya_sub-owna507_kill-2026-11-14"
SOLVOL, BVOL, DUMPVOL, OUTVOL, HFVOL = ("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
                                        "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14",
                                        "probe-shift_aswin-surya_hf-cache")
SP = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad")
PINS = {  # local inputs, full sha256 (the Sol scorer's launch inputs; 164934's record names r36, the C manifest and the corrected map)
    "cb_record": ("git", "cc53afc3a:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json",
                  "c002d5f822e49d7877f4e8b79a80ca9dd2e2f16c664b921d3ad12cb3da4c49ac"),
    "t196": ("git", "HEAD:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
             "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "cells311": ("git", "d75f997f3:experiments/2026-10-14_q35_sol_avg_POSTHOC/config/scaleup_tasks_plus311_2026-10-14_125135.json",
                 "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
    "cellsnm": ("git", "d75f997f3:experiments/2026-10-14_q35_sol_avg_POSTHOC/config/nm360_cells_nm_built_2026-10-14.json",
                "93d9b538710b6bd930e7df01b0f5296d417550a0f0a35bcb6eb696166cf39a04"),
    "r36": ("path", ("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/wt_nanors_restack/"
                     "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"),
            "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"),
    "b_manifest": ("path", str(SP / "cb/correctedb_manifest_v1.json"), "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc"),
    "c_upload": ("path", str(SP / "cc/manifest_q4f_c_eval_upload_v1.json"), "98cbbbd52aeb73cfce884b08350130ddd7cc23bf637d1603896f3ae0137be157"),
    "c_stage": ("path", str(SP / "cc/q4f_stage_c.json"), "7864132d252fa225eb6019e0b9558669b121b9ea374c0d77b5de6b95424556a0"),
    "c_extract": ("path", str(SP / "cc/extract_q4b_q4b_extract_c_v1_a1_2026-10-14_121640.json"), "d362ab2c23f0d05105901955aaa335eca98e70cb8422ce3285ae863116ac43f1"),
}
Q4_ATTEST = "d76d4ae1836db7aeb154dfc817ce530fff6f4c7386636e450561391d809fa96e"  # q4_h200_attestation_2026-10-14.json @ 3172e9c88 (09's Revision 6i)
FOLD_R36_LABEL = "informational only; not a leakage-relevant routing for own_a (none of own_a's five models saw any scale-up main)"
app = modal.App(APP)
sol_vol = modal.Volume.from_name(SOLVOL, version=2).read_only()
b_vol = modal.Volume.from_name(BVOL, version=2).read_only()
d_vol = modal.Volume.from_name(DUMPVOL, version=2).read_only()
out_vol = modal.Volume.from_name(OUTVOL, version=2)
hf_vol = modal.Volume.from_name(HFVOL).read_only()
tok_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3")
             .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
torch_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3")  # run_avg_transfer.cpu_torch_image


def stem(key):
    return hashlib.sha1(key.encode()).hexdigest()[:20]


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- shared row and scoring helpers (container side)
def select(root, e, kind, skip):
    """The Sol scorer's row selection (run_scaleup_score.score_all): reply = output_end calls with row_mean; command = call_end
    calls with row, minus the fixed-token rows. Returns [(irid, src_irid, call_index, row)], chunk, row_start."""
    m = json.loads((Path(root) / e["extraction_run"] / "meta" / e["arm"] / f"{stem(e['key'])}.json").read_text())
    if m["key"] != e["key"]:
        raise ValueError("meta key mismatch")
    if kind == "R":
        cs = [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]
        rr = [c["row_mean"] for c in cs]
    else:
        sk = {tuple(x) for x in skip[e["key"]]}
        cs = [c for c in m["calls"] if c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in sk]
        rr = [c["row"] for c in cs]
    return [(str(c["irid"]), str(c["src_irid"]), int(c["call_index"]), r) for c, r in zip(cs, rr)], m["chunk"], m["row_start"]


def gather(root, items, L):
    """items: [(extraction_run, chunk, row_start, [row offsets])] -> float32 X in that order."""
    import numpy as np

    X = [np.asarray(np.load(Path(root) / xr / "by_layer" / f"L{L}" / f"{ch}.npy", mmap_mode="r")[[rs + q for q in rr]]).astype(np.float32)
         for xr, ch, rs, rr in items]
    X = np.concatenate(X) if X else np.zeros((0, 1), np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"L{L}: non-finite activation rows")
    return X


def load_weights(cfg):
    """{("R", L, f): (kind, W, b, mean, std)} for every (layer, fold) of the reply choice; {("C", f): ...} at L63. sha-checked."""
    import numpy as np

    out, shas = {}, {}
    for k, e in cfg["weights_R"].items():
        b_ = (Path("/sol") / e["path"]).read_bytes()
        if sha_b(b_) != e["sha256"]:
            raise ValueError(f"weights {k}: sha256")
        z = np.load(io.BytesIO(b_))
        L, f = map(int, k.split("|"))
        out[("R", L, f)] = ("tight", z["W"], z["b"], z["mean"], z["std"])
        shas[f"R|{k}"] = e["sha256"]
    zc = {}
    for nm in ("tight", "outer"):
        b_ = (Path("/sol") / cfg["weights_C"][nm]["path"]).read_bytes()
        if sha_b(b_) != cfg["weights_C"][nm]["sha256"]:
            raise ValueError(f"weights C {nm}: sha256")
        zc[nm] = np.load(io.BytesIO(b_))
    for f in range(5):
        nm = "tight" if f"own_a__f{f}__W" in zc["tight"].files else "outer"
        out[("C", f)] = (nm, *(zc[nm][f"own_a__f{f}__{p}"] for p in ("W", "b", "mean", "std")))
    return out, shas


def logit(XT, w):
    """run_correctedb arithmetic: tight = (((X - mean)/std).double() @ W + b); outer = decision((X - mean)/std, W, b).double()."""
    import torch

    kind, W, b, mean, std = w
    Xs = (XT - torch.from_numpy(mean)) / torch.from_numpy(std)
    if kind == "tight":
        return (Xs.double() @ torch.from_numpy(W.reshape(-1, 1)) + torch.from_numpy(b.reshape(-1))).numpy()[:, 0]
    return (Xs @ torch.from_numpy(W.reshape(-1, 1)) + torch.from_numpy(b.reshape(-1))).double().numpy()[:, 0]


def score_all(root, epochs, skip, choice, w):
    """Every row scored by the five own_a models: R model f = mean of its two layers' logits; C model f = L63. Rows in epoch order."""
    import numpy as np
    import torch

    out = {}
    for ro, kind, Ls in (("avg_top2", "R", sorted({L for c in choice for L in c})), ("cmd_A", "C", [63])):
        items, rows, empty = [], [], []
        for e in epochs:
            rr, ch, rs = select(root, e, kind, skip)
            if not rr:
                empty.append(f"{e['world']}|{e['task']}")
                continue
            items.append((e["extraction_run"], ch, rs, [r[3] for r in rr]))
            rows += [(f"{e['world']}|{e['task']}", r[0], r[1], r[2], ch) for r in rr]
        lg = {}
        for L in (Ls if items else []):
            XT = torch.from_numpy(gather(root, items, L))
            for f in range(5):
                if kind == "C" or L in choice[f]:
                    lg[(f, L)] = logit(XT, w[("C", f)] if kind == "C" else w[("R", L, f)])
        sbm = np.stack([np.mean([lg[(f, L)] for L in choice[f]], axis=0) if kind == "R" else lg[(f, 63)] for f in range(5)], axis=1) if rows else np.zeros((0, 5))
        if not np.isfinite(sbm).all():
            raise ValueError(f"{ro}: non-finite scores")
        seen, unit = {}, []
        for r in rows:
            unit.append(seen.get(r[0], 0))
            seen[r[0]] = unit[-1] + 1
        out[ro] = {"key": np.array([r[0] for r in rows], dtype=str), "irid": [r[1] for r in rows], "src_irid": [r[2] for r in rows],
                   "call_index": np.array([r[3] for r in rows], np.int64), "unit": np.array(unit, np.int64), "chunks": sorted({r[4] for r in rows}),
                   "score_by_model": sbm, "no_rows": empty}
    return out


# ---------------------------------------------------------------- remote functions
def _fixed_ids():
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    from probe_shift.q1_eval import fixed_call_tokens

    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], local_files_only=True)
    fixed = fixed_call_tokens("q4b_main_v1")
    return fixed, set(tok.convert_tokens_to_ids(fixed))


def _skip_file(b, keys, fixed_ids):
    """run_asis_fits.b_skip's rule (as run_scaleup_score.skip_rows): call_end rows whose readout token is a fixed call token."""
    sk, seen, pairs = {}, set(), []
    for line in b.decode().splitlines():
        ep = json.loads(line)
        if keys is not None and ep["key"] not in keys:
            continue
        if ep["key"] in seen:
            raise ValueError("a key twice in one staged file")
        seen.add(ep["key"])
        pairs.append((ep["key"], ep.get("task")))
        sk[ep["key"]] = [[r["irid"], r["call_index"]] for r in ep["calls"] if r["kind"] == "call_end" and r["readout"] is not None and ep["ids"][r["readout"]] in fixed_ids]
    if keys is not None and seen != keys:
        raise ValueError(f"{len(keys - seen)} requested epochs not staged")
    return sk, pairs


@app.function(image=tok_image, cpu=(4, 4), memory=(32768, 32768), timeout=900, retries=0, scaledown_window=2, volumes={"/b": b_vol, "/hf": hf_vol})
def skips_b(b_want: dict, b_shas: dict) -> dict:
    """Corrected B + C: the fixed-token command rows to drop, per epoch (staged sha256 checked)."""
    fixed, fid = _fixed_ids()
    out = {}
    for run, arms_ in b_want.items():
        for arm, keys in arms_.items():
            rel = f"{run}/inputs/{arm}.jsonl"
            b = (Path("/b") / rel).read_bytes()
            if rel in b_shas and b_shas[rel] != sha_b(b):
                raise ValueError(f"{rel}: staged sha256")
            out |= _skip_file(b, set(keys), fid)[0]
    return {"skip": out, "fixed_tokens": fixed}


@app.function(image=tok_image, cpu=(2, 2), memory=(8192, 8192), timeout=420, retries=0, scaledown_window=2, max_containers=16, volumes={"/d": d_vol, "/hf": hf_vol})
def skips_arm(a: dict) -> dict:
    """One arm's staged epochs: (key, task) pairs and the fixed-token command rows; the staged file's sha256 must be the Sol
    manifest's dump sha."""
    _, fid = _fixed_ids()
    rel = f"{a['staged_run']}/inputs/{a['meta_arm']}.jsonl"
    b = (Path("/d") / rel).read_bytes()
    if sha_b(b) != a["staged_sha"]:
        raise ValueError(f"{a['arm']}: {rel} sha256 differs from the Sol manifest's dump sha")
    sk, pairs = _skip_file(b, None, fid)
    return {"pairs": pairs, "skip": sk, "dropped": sum(len(v) for v in sk.values()), "staged_rel": rel}


@app.function(image=torch_image, cpu=(16, 16), memory=(65536, 65536), timeout=1800, retries=0, scaledown_window=2, volumes={"/sol": sol_vol, "/b": b_vol})
def gate_b(epochs: list, skip: dict, choice: list, cfg: dict, fold_t196: dict) -> dict:
    """Corrected B: (1) the exact path, 164934's own_a tight rows rebuilt row for row (same rows, order, shapes and arithmetic):
    reply per (layer, fold), command per fold; (2) the scale-up path (score_all) on honest B, within 1e-9 of the same rows."""
    import numpy as np
    import torch

    torch.set_num_threads(16)
    w, wsha = load_weights(cfg)
    em = {f"{e['world']}|{e['task']}": e for e in epochs}
    res = {}

    def ref_rows(e):
        b_ = (Path("/sol") / e["path"]).read_bytes()
        if sha_b(b_) != e["sha256"]:
            raise ValueError(f"reference rows {e['path']}: sha256")
        return np.load(io.BytesIO(b_))

    def rebuild(z, kind, wkey_of):  # rows of z in order -> exact logits per fold group
        keys, ir, ci = [str(k) for k in z["key"]], [str(x) for x in z["irid"]], z["call_index"].tolist()
        got = np.full(len(keys), np.nan)
        bad_sel = 0
        for f in sorted(set(z["fold"].tolist())):
            ix = np.flatnonzero(z["fold"] == f)
            items, cur = [], None
            for i in ix:  # consecutive rows of one key form one run's block, in meta order
                if cur is None or cur[0] != keys[i]:
                    cur = [keys[i], []]
                    items.append(cur)
                cur[1].append(i)
            gi = []
            for k, rows_i in items:
                rr, ch, rs = select("/b", em[k], kind, skip)
                if [(r[0], r[2]) for r in rr] != [(ir[i], ci[i]) for i in rows_i]:
                    bad_sel += 1
                    continue
                gi.append((em[k]["extraction_run"], ch, rs, [r[3] for r in rr]))
            if bad_sel:
                break
            for L in ({63} if kind == "C" else {int(x) for x in z["layer"][ix]}):
                XT = torch.from_numpy(gather("/b", gi, L))
                got[ix] = logit(XT, w[wkey_of(L, f)])
        return got, bad_sel

    worst, n_eq, n_all, bad = 0.0, 0, 0, []
    for k, e in cfg["ref_R"].items():
        L, f = map(int, k.split("|"))
        z = ref_rows(e)
        if set(z["fold"].tolist()) != {f} or set(z["layer"].tolist()) != {L}:
            bad.append(f"R {k}: fold/layer arrays")
            continue
        got, bs = rebuild(z, "R", lambda L_, f_: ("R", L_, f_))
        if bs:
            bad.append(f"R {k}: {bs} runs whose row selection differs")
            continue
        eq = got == z["score"]
        n_eq, n_all, worst = n_eq + int(eq.sum()), n_all + len(eq), max(worst, float(np.max(np.abs(got - z["score"]))))
    zc = ref_rows(cfg["ref_C"])
    gotc, bs = rebuild(zc, "C", lambda L_, f_: ("C", f_))
    if bs:
        bad.append(f"C: {bs} runs whose row selection differs")
    else:
        eq = gotc == zc["score"]
        n_eq_c, worst_c = int(eq.sum()), float(np.max(np.abs(gotc - zc["score"])))
    res["exact"] = {"R": {"rows": n_all, "bit_identical": n_eq, "max_abs_diff": worst},
                    "C": {"rows": len(zc["score"]), "bit_identical": None if bs else n_eq_c, "max_abs_diff": None if bs else worst_c,
                          "folds_tight": [f for f in range(5) if w[("C", f)][0] == "tight"]},
                    "problems": bad, "passes": not bad and n_eq == n_all and n_all > 0 and not bs and n_eq_c == len(zc["score"])}
    # (2) the scale-up path on honest B: designated model = the run's T196 fold; reference R = mean of the fold's two layer rows
    hb = [e for e in epochs if e["world"] == "honest_b"]
    torch.set_num_threads(4)  # the production (score_arm) configuration
    sc = score_all("/b", hb, skip, choice, w)
    d = {}
    for ro in ("avg_top2", "cmd_A"):
        s = sc[ro]
        des = np.array([s["score_by_model"][i, fold_t196[k.split("|", 1)[1]]] for i, k in enumerate(s["key"].tolist())])
        ref = {}
        if ro == "cmd_A":
            for k, ir_, ci_, v in zip(zc["key"].tolist(), zc["irid"].tolist(), zc["call_index"].tolist(), zc["score"].tolist()):
                if (str(k), str(ir_), int(ci_)) in ref:
                    raise ValueError("C reference: a duplicate (key, irid, call_index)")
                ref[(str(k), str(ir_), int(ci_))] = v
        else:
            parts = {}
            for e in cfg["ref_R"].values():
                z = ref_rows(e)
                for k, ir_, ci_, v in zip(z["key"].tolist(), z["irid"].tolist(), z["call_index"].tolist(), z["score"].tolist()):
                    parts.setdefault((str(k), str(ir_), int(ci_)), []).append(v)
            if any(len(v) != 2 for u, v in parts.items() if u[0].startswith("honest_b|")):
                raise ValueError("R reference: an honest_b row without exactly two layer contributions")
            ref = {u: float(np.mean(v)) for u, v in parts.items()}
        ids = list(zip(s["key"].tolist(), s["irid"], s["call_index"].tolist()))
        if len(set(ids)) != len(ids):
            raise ValueError(f"{ro}: a duplicate (key, irid, call_index) in the scale-up path")
        mine = dict(zip(ids, des.tolist()))
        ref_hb = {u for u in ref if u[0].startswith("honest_b|")}
        dm = max((abs(mine[u] - ref[u]) for u in set(mine) & ref_hb), default=float("inf"))
        d[ro] = {"rows_mine": len(mine), "rows_reference": len(ref_hb), "max_abs_diff": dm,
                 "passes": set(mine) == ref_hb and len(mine) > 0 and dm <= 1e-9}
    res["scale_up_path_honest_b"] = d | {"passes": all(v["passes"] for v in d.values())}
    res["weights_sha256"] = wsha | {f"C|{nm}": cfg["weights_C"][nm]["sha256"] for nm in ("tight", "outer")}
    res["pass"] = res["exact"]["passes"] and res["scale_up_path_honest_b"]["passes"]
    return res


@app.function(image=torch_image, cpu=(4, 4), memory=(16384, 16384), timeout=480, retries=0, scaledown_window=2, max_containers=16,
              volumes={"/sol": sol_vol, "/d": d_vol, "/work": out_vol})
def score_arm(a: dict, sk: dict, choice: list, cfg: dict, main_of: dict, fold_r36_of_main: dict, train: dict, stamp: str, git: str) -> dict:
    """One arm: score, gate, then write /work/scaleup/<arm>/own_a/ (exclusive, manifest last). Returns shas, counts, gates only."""
    import numpy as np
    import torch

    torch.set_num_threads(4)
    arm = a["arm"]
    sdir = Path("/work") / "scaleup" / arm / a["sol_dir"]
    mb = (sdir / "manifest_sol.json").read_bytes()
    if sha_b(mb) != a["sol_manifest_sha256"]:
        raise ValueError(f"{arm}: Sol manifest sha256 differs from the pinned one")
    sm = json.loads(mb)
    (world,) = sm["arm_to_world"].values()
    if a["extract_record"]["run"] != sm["dump"]["extract_run"] or a["extract_record"]["sha256"] != sm["dump"]["manifest"]["sha256"]:
        raise ValueError(f"{arm}: the pinned extraction record is not the one the Sol manifest names")
    g = {}
    # epochs from the staged pairs, mapped exactly as the Sol scorer did; keys must equal Sol's staged keys
    eps = []
    for k, t in sk["pairs"]:
        w_, c_ = ("run2", t[:-3]) if t.endswith("_r2") and sm.get("r2_suffix_world") == "run2" else (world, t)
        eps.append({"key": k, "world": w_, "task": c_, "arm": sm["dump"]["meta_arm"], "extraction_run": sm["dump"]["extract_run"]})
    keys = {f"{e['world']}|{e['task']}" for e in eps}
    g["staged_keys_equal_sol"] = {"n": len(keys), "only_mine": sorted(keys - set(sm["folds"]))[:5], "only_sol": sorted(set(sm["folds"]) - keys)[:5],
                                  "sol_counted_gate_passes": bool(sm["gates"]["staged_keys_counted_by_d9"]["passes"]),
                                  "passes": keys == set(sm["folds"]) and len(keys) == len(eps) and bool(sm["gates"]["staged_keys_counted_by_d9"]["passes"])}
    g["skip_rows_equal_sol"] = {"mine": sk["dropped"], "sol": sum(sm["skip_dropped_rows"].values()), "passes": sk["dropped"] == sum(sm["skip_dropped_rows"].values())}
    # routing: own_a's training mains are T196 mains; no scale-up main may be among them (explicit count)
    mains = {main_of[e["task"]] for e in eps}
    union_train = set().union(*map(set, train["mains"]))
    g["owna_training_overlap"] = {"scale_up_mains": len(mains), "in_owna_training_mains": len(mains & union_train), "passes": not (mains & union_train)}
    fr36 = {f"{e['world']}|{e['task']}": fold_r36_of_main.get(main_of[e["task"]], -1) for e in eps}
    g["fold_r36_equals_sol_fold"] = {"passes": all(sm["folds"][k] == v for k, v in fr36.items())}
    if not all(x["passes"] for x in g.values()):
        return {"arm": arm, "published": False, "gates": g}
    w, wsha = load_weights(cfg)
    res = score_all("/d", eps, sk["skip"], choice, w)
    # alignment with Sol's per-row files (key, unit, call_index; fold): reads those arrays only, never a score array
    al = {}
    for ro in ("avg_top2", "cmd_A"):
        fb = (sdir / f"a_tight__{ro}.npz").read_bytes()
        if sha_b(fb) != sm["files"][f"a_tight__{ro}.npz"]:
            raise ValueError(f"{arm}: Sol {ro} npz sha256 differs from its manifest")
        z = np.load(io.BytesIO(fb))
        r = res[ro]
        same = (len(z["key"]) == len(r["key"]) and np.array_equal(z["key"].astype(str), r["key"]) and np.array_equal(z["unit"], r["unit"])
                and np.array_equal(z["call_index"], r["call_index"]) and np.array_equal(z["fold"], np.array([fr36[k] for k in r["key"].tolist()], np.int64)))
        al[ro] = {"rows": len(r["key"]), "runs": len(set(r["key"].tolist())), "runs_without_rows": len(r["no_rows"]), "passes": bool(same)}
    g["rows_aligned_with_sol"] = al | {"passes": all(v["passes"] for v in al.values())}
    # H200: the Sol manifest's recorded provenance, and every chunk read here inside it
    gp = sm["gpu_provenance"]
    chunks = sorted(set(res["avg_top2"]["chunks"]) | set(res["cmd_A"]["chunks"]))
    pairs = [(int(m_.group(1)), int(m_.group(2))) for c in chunks if (m_ := re.match(r"shard(\d+)_a(\d+)_chunk", c))]
    if "union" in gp:  # the attempt records' union, as the Sol scorer bound it
        allowed = {tuple(p) for p in gp["union"]["ok_pairs"] + gp["union"]["eligible_partial_pairs"]}
    else:  # one record: every chunk must be the sha-verified extraction record's attempt (local entrypoint), as complete.json agrees
        att = int(a["extract_record"]["attempt"])
        comp = json.loads((Path("/d") / sm["dump"]["extract_run"] / "complete.json").read_text())
        if comp["inputs_sha256"].get(f"{sm['dump']['meta_arm']}.jsonl") != sm["dump"]["sha256"] or int(comp["attempt"]) != att:
            raise ValueError(f"{arm}: complete.json does not name this staged set's sha256 or the recorded attempt")
        allowed = {(int(s), att) for s in gp["shard_ids"]}
    h200 = (set(gp["gpu_per_shard"].values()) == {"NVIDIA H200"} and set(sm["dump"]["gpu"].values()) == {"NVIDIA H200"} and len(pairs) == len(chunks)
            and {s for s, _ in pairs} <= set(sm["shards_scored"]) and set(pairs) <= allowed)
    q4 = sm["dump"]["extract_run"] in ("su311_q4_main_extract_v1", "su311_q4_run2_extract_v1")
    side_ok, side_sha = True, None
    if q4:  # Revision 6i: the Sol manifest's roots_read.json sidecar, its contents bound to this manifest and the pinned attestation
        sp = sdir / "roots_read.json"
        sd_ = json.loads(sp.read_text()) if sp.exists() else {}
        side_sha = sha_b(sp.read_bytes()) if sp.exists() else "MISSING"
        side_ok = (sd_.get("root") == "v1" and sd_.get("manifest_sol_sha256") == a["sol_manifest_sha256"] and sd_.get("extract_run") == sm["dump"]["extract_run"]
                   and sd_.get("arm") == arm and (sd_.get("attestation") or {}).get("sha256") == Q4_ATTEST)
    g["h200"] = {"chunks": len(chunks), "union": "union" in gp, "q4_sidecar_sha256": side_sha, "passes": bool(h200) and side_ok}
    g["finite"] = {"passes": True, "note": "score_all raises on any non-finite activation row or score"}
    if not all(x["passes"] for x in g.values()):
        return {"arm": arm, "published": False, "gates": g}
    # outputs, Sol per-row schema
    blobs = {}
    js = {"run_scores": {}, "cmd_A": {}, "by_model": {}, "cmd_A_by_model": {}}
    for ro in ("avg_top2", "cmd_A"):
        r = res[ro]
        n = len(r["key"])
        buf = io.BytesIO()
        np.savez(buf, key=r["key"], score=np.full(n, np.nan), unit=r["unit"], call_index=r["call_index"], fold=np.full(n, -1, np.int64),
                 fold_r36=np.array([fr36[k] for k in r["key"].tolist()], np.int64), score_by_model=r["score_by_model"])
        blobs[f"own_a_tight__{ro}.npz"] = buf.getvalue()
        for k in dict.fromkeys(r["key"].tolist()):
            mk = r["key"] == k
            mean_ = r["score_by_model"][mk].mean(axis=0)
            if not np.isfinite(mean_).all():
                raise ValueError(f"{arm}: a non-finite run mean ({ro})")
            (js["run_scores"] if ro == "avg_top2" else js["cmd_A"])[k] = float("nan")
            (js["by_model"] if ro == "avg_top2" else js["cmd_A_by_model"])[k] = [float(x) for x in mean_]
    blobs["own_a_tight__avg_top2.json"] = json.dumps(js, indent=1).encode()
    for f in range(5):
        blobs[f"training_mains__own_a_tight__f{f}.json"] = train["files"][f].encode()
    files = {fn: sha_b(b) for fn, b in blobs.items()}
    tsha = [sha_b(train["files"][f].encode()) for f in range(5)]
    man = {"label": "own_a tight on the 507 scale-up (blind; 2e's approval (a1)); raw scores, counts, shas and gates only", "scorer_git": git, "stamp": stamp,
           "arm": arm, "arm_to_world": sm["arm_to_world"], "sol_manifest": {"path": f"/scaleup/{arm}/{a['sol_dir']}/manifest_sol.json", "sha256": a["sol_manifest_sha256"]},
           "dump": sm["dump"], "staged_rel": sk["staged_rel"], "skip_dropped_rows": sk["dropped"],
           "folds": {k: -1 for k in sorted(keys)}, "run_folds": {k: -1 for k in sorted(keys)},
           "fold_r36": {"by_run": fr36, "label": FOLD_R36_LABEL},
           "routing": {mn: {"excluded_by_all_five": True, "training_manifest_sha256": tsha} for mn in sorted(mains)},
           "routing_rule": "own_a's fold map is its training map (T196 mains, corrected B run1 vs honest_a, r36 folds); a scale-up main is in none of the "
                           "five training lists, so it is fold -1 (Append 21 section 4 / 21b): score NaN, score_by_model = the five models' raw scores",
           "training_manifest_sha256_by_fold": {str(f): tsha[f] for f in range(5)},
           "weights": {"record": cfg["record"], "sha256": wsha | {f"C|{nm}": cfg["weights_C"][nm]["sha256"] for nm in ("tight", "outer")},
                       "C_kind_by_fold": {str(f): w[("C", f)][0] for f in range(5)}, "layer_choice_own_a": choice},
           "fit_sha256": {str(f): sha_b(b"".join(np.asarray(x, np.float64).tobytes() for L in sorted(choice[f]) for x in w[("R", L, f)][1:3])
                                       + b"".join(np.asarray(x, np.float64).tobytes() for x in w[("C", f)][1:3])) for f in range(5)},
           "fit_sha256_method": "per fold model f: sha256 over float64 bytes of W then b at each layer in sorted(layer_choice_own_a[f]) (reply), then W, b at L63 (command A)",
           "gates": g, "gate_b": cfg["gate_b_summary"], "runs_without_rows": {ro: res[ro]["no_rows"] for ro in ("avg_top2", "cmd_A")},
           "files": files, "file_bytes": {fn: len(b) for fn, b in blobs.items()},
           "formats": {"npz": "own_a_tight__<avg_top2|cmd_A>.npz: per-row key, score (designated: NaN, every row is fold -1), unit, call_index, fold (-1), fold_r36 "
                              "(informational), score_by_model (rows x 5, own_a fold models 0..4; avg_top2 under model f = mean of its two chosen layers)",
                       "json": "own_a_tight__avg_top2.json: run_scores / cmd_A (NaN, fold -1), by_model / cmd_A_by_model = the five models' raw run means"},
           "not_done_here": "no per-fold z-scoring, ensembles, thresholds, false-alarm rates or comparisons (09, Append 21b's order)"}
    mbytes = json.dumps(man, indent=1).encode()
    out_vol.reload()
    od = Path("/work") / "scaleup" / arm / "own_a"
    names = list(blobs) + ["manifest_owna.json"]
    if od.exists() and any((od / fn).exists() for fn in names):
        raise ValueError(f"{arm}: own_a outputs already exist; an operator decision is needed")
    od.mkdir(parents=True, exist_ok=True)
    (od / ".publishing").mkdir(exist_ok=False)
    for fn, b in list(blobs.items()) + [("manifest_owna.json", mbytes)]:
        with (od / fn).open("xb") as fh:
            fh.write(b)
    (od / ".publishing").rmdir()
    out_vol.commit()
    return {"arm": arm, "published": True, "gates": g, "files": files, "manifest_sha256": sha_b(mbytes), "world": world,
            "rows": {ro: al[ro]["rows"] for ro in al}, "runs": {ro: al[ro]["runs"] for ro in al}}


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=(1, 1), memory=(2048, 2048), timeout=7200, retries=0, scaledown_window=2, volumes={"/work": out_vol})
def driver(cfg: dict, stamp: str, git: str) -> dict:
    """Server-side orchestration (a dropped client does not stop it): skips, the corrected-B gate, then the arms."""
    arms = []
    for a in cfg["arms"]:
        mb = (Path("/work") / "scaleup" / a["arm"] / a["sol_dir"] / "manifest_sol.json").read_bytes()
        if sha_b(mb) != a["sol_manifest_sha256"]:
            raise ValueError(f"{a['arm']}: Sol manifest sha256 differs from the pinned one")
        sm = json.loads(mb)
        arms.append({"arm": a["arm"], "staged_run": sm["dump"]["staged_run"], "meta_arm": sm["dump"]["meta_arm"], "staged_sha": sm["dump"]["sha256"]})
    skb = skips_b.remote(cfg["b_want"], cfg["b_shas"])
    rec = {"stage": "gate_b", "fixed_tokens": skb["fixed_tokens"]}
    gb = gate_b.remote(cfg["epochs_b"], skb["skip"], cfg["choice"], cfg, cfg["fold_t196"])
    rec["gate_b"] = gb
    if not gb["pass"]:
        return rec | {"stop": "corrected-B gate failed: nothing scored or written"}
    cfg = cfg | {"gate_b_summary": {"exact": gb["exact"], "scale_up_path_honest_b": gb["scale_up_path_honest_b"]}}
    rec["stage"] = "arms"
    sks = list(skips_arm.map(arms, return_exceptions=True))
    outs, todo = [], []
    for a, s in zip(cfg["arms"], sks):
        if isinstance(s, dict):
            todo.append((a, s))
        else:
            outs.append({"arm": a["arm"], "published": False, "error": f"skips: {type(s).__name__}: {str(s)[:500]}"})
    for (a, _), r in zip(todo, score_arm.starmap([(a, s, cfg["choice"], cfg, cfg["main_of"], cfg["fold_r36_of_main"], cfg["train"], stamp, git)
                                                    for a, s in todo], return_exceptions=True)):
        outs.append(r if isinstance(r, dict) else {"arm": a["arm"], "published": False, "error": f"{type(r).__name__}: {str(r)[:500]}"})
    rec |= {"stage": "done", "arms": outs, "published": sum(o.get("published", False) for o in outs), "of": len(outs)}
    return rec


@app.local_entrypoint()
def main(only: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    head = git("rev-parse", "HEAD").decode().strip()
    raw = {}
    for k, (kind, src, s) in PINS.items():
        b = git("show", src) if kind == "git" else Path(src).read_bytes()
        if sha_b(b) != s:
            raise SystemExit(f"{k}: sha256 mismatch")
        raw[k] = json.loads(b)
    rec_ = raw["cb_record"]
    choice = rec_["layer_choice_own_a"]
    if choice != [[49, 50], [49, 50], [50, 51], [49, 50], [49, 50]]:
        raise SystemExit("layer_choice_own_a differs from part (c)'s")
    tav, cmdA = rec_["saved"]["tight"]["avg"], rec_["saved"]["cmd"]["cmdA"]
    need = [f"{L}|{f}" for f in range(5) for L in choice[f]]
    if any(k not in tav for k in need):
        raise SystemExit("a (layer, fold) of the reply choice has no tight refit in 164934")
    fold_of_main = {}
    for r in raw["r36"]:
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("r36: a main in two folds")
    t196 = {c["key"]: c["main"] for c in raw["t196"]["cells"]}
    fold_t196 = {t: fold_of_main[m] for t, m in t196.items()}
    # corrected B + C epochs, as the Sol scorer's gate (run_scaleup_score._main)
    mf = raw["b_manifest"]
    epochs = [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run")} for e in mf["epochs"]]
    shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"]}
    up, st, ex = raw["c_upload"], raw["c_stage"], raw["c_extract"]
    excl = set(st["c"]["c_gate"]["excluded"])
    epochs += [{"key": f"{f_['path']}#e0", "world": "honest_c", "task": f_["task"], "arm": "honest_b", "extraction_run": ex["config"]["run"],
                "staged_inputs_run": ex["config"]["inputs"]} for f_ in up["files"] if f"{f_['path']}#e0" not in excl]
    shas |= {f"{ex['config']['inputs']}/inputs/{a}": s for a, s in ex["config"]["inputs_sha256"].items()}
    if len({e["key"] for e in epochs}) != len(epochs) or len({(e["world"], e["task"]) for e in epochs}) != len(epochs):
        raise SystemExit("duplicate epochs in corrected B + C")
    want = {}
    for e in epochs:
        want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    # own_a's training mains per fold: corrected B run1 / honest_a runs of T196 tasks with r36 fold != f (a superset of the fitted rows)
    tr_tasks = sorted({e["task"] for e in epochs if e["world"] in ("run1", "honest_a")})
    train_files = [json.dumps({"probe": "own_a_tight", "fold_model": f, "training_mains": sorted({t196[t] for t in tr_tasks if fold_t196[t] != f}),
                               "source": "corrected B (correctedb_manifest_v1) run1 and honest_a runs of T196 tasks with r36 fold != f; 164934 @ cc53afc3a"}, indent=1)
                   for f in range(5)]
    main_of = {c["key"]: c["main"] for c in raw["cells311"]["cells"]} | {c["key"]: c["main"] for c in raw["cellsnm"]["cells"]}
    tr_union = set().union(*(set(json.loads(x)["training_mains"]) for x in train_files))
    overlap = {"scale_up_mains": len(set(main_of.values())), "owna_training_mains": len(tr_union), "intersection": len(set(main_of.values()) & tr_union),
               "basis": "every +311 (q2 @ 0caadbfb) and nm360 (nm_built) cell's main vs the union of own_a's five training-main lists, by key"}
    if overlap["intersection"]:
        raise SystemExit(f"{overlap['intersection']} scale-up mains are own_a training mains: fold -1 routing does not hold")
    arms_cfg = json.loads((HERE / "config" / "owna_arms.json").read_text())["arms"]
    if only:
        want_ = set(only.split(","))
        if not want_ <= {a["arm"] for a in arms_cfg}:
            raise SystemExit(f"--only names arms outside config/owna_arms.json: {sorted(want_ - {a['arm'] for a in arms_cfg})}")
        arms_cfg = [a for a in arms_cfg if a["arm"] in want_]
    if len({a["arm"] for a in arms_cfg}) != len(arms_cfg):
        raise SystemExit("an arm is listed twice")
    for a in arms_cfg:  # d9's extraction record named by the Sol manifest: sha256, run and attempt, read here (Codex fix-check item 1)
        er = a["extract_record"]
        b = Path(er["path"]).read_bytes()
        rj = json.loads(b)
        if sha_b(b) != er["sha256"] or rj["config"]["run"] != er["run"] or rj["config"]["attempt"] != er["attempt"]:
            raise SystemExit(f"{a['arm']}: the extraction record differs from the pinned one")
    if not 0 < len(arms_cfg) <= 58:
        raise SystemExit("between 1 and 58 arms")
    rec_path = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json"
    cfg = {"arms": arms_cfg, "choice": choice, "epochs_b": epochs, "b_want": want, "b_shas": shas, "fold_t196": fold_t196,
           "main_of": main_of, "fold_r36_of_main": fold_of_main, "train": {"files": train_files, "mains": [json.loads(x)["training_mains"] for x in train_files]},
           "weights_R": {k: {"path": tav[k]["weights"], "sha256": tav[k]["weights_sha256"]} for k in need},
           "ref_R": {k: {"path": tav[k]["rows"], "sha256": tav[k]["rows_sha256"]} for k in need},
           "weights_C": {"tight": cmdA["own_a_tight_weights"], "outer": cmdA["own_a_weights"]},
           "ref_C": {"path": cmdA["rows|own_a_tight"]["path"], "sha256": cmdA["rows|own_a_tight"]["sha256"]},
           "record": {"commit": "cc53afc3a", "path": rec_path, "sha256": PINS["cb_record"][2]}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = driver.remote(cfg, stamp, head)
    rec = {"label": "own_a tight on the 507 scale-up, blind: shas, counts and gate results only", "git": head, "stamp": stamp,
           "script_sha256": sha_b(Path(__file__).read_bytes()), "pins": {k: v[2] for k, v in PINS.items()}, "arms_requested": len(arms_cfg),
           "owna_training_overlap_all_scaleup_mains": overlap, "fold_r36_label": FOLD_R36_LABEL} | out
    blob = json.dumps(rec, indent=1, default=str).encode()
    dest = HERE / "results" / f"owna507_{stamp}.json"
    dest.open("xb").write(blob)
    print(json.dumps({"record": str(dest.relative_to(REPO)), "sha256": sha_b(blob), "stage": out.get("stage"), "stop": out.get("stop"),
                      "gate_b_pass": out.get("gate_b", {}).get("pass"), "published": out.get("published"), "of": out.get("of")}), flush=True)
