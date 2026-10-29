"""honest_all route C rerun: independent value check of the four layer-48 run-level AUROCs (rows 75-78 and 95-98).

Spec: honest_all/PRESTATED_2026-10-20_130735.md (sha256 54bf522c3becbad9400d77ae83048d91b65f7d597e5c3811da13a3808a73cf9e) plus
PRESTATED_ADDENDUM_2026-10-20_131806.md (sha256 63e6dd03099d4ce94e634490e1cb2698cecad975112afb24f256283832b50f6e) plus
PRESTATED_ADDENDUM_C_2026-10-20_134810.md (sha256 c485637904e1d2f3e48e6f239d9401d615dc3afbc211e4d91743a85ed0a864c1) plus
PRESTATED_ADDENDUM_D_2026-10-20_140146.md (sha256 16486c817b1e1d1f0dd67b8e859a9078f1c8dbf598649a53ded32c3db1336284) plus
PRESTATED_ADDENDUM_E_2026-10-20_141519.md (sha256 66bbcbaea2451de08897803d4948d5132f5356cde2faeae3aa9b8db0da11dcc2) plus
PRESTATED_ADDENDUM_F_2026-10-20_161905.md (sha256 444e44c3ff2260bc3d19510cb704113b7e332e32c3e04e0572041bea1db0dc75) (hash-shards + replay-only) plus
PRESTATED_ADDENDUM_G_2026-10-20_163306.md (sha256 5772826235e2df3cd4298f762ea13e1d7745f36a2f2b37e0e4db308015e3ce9c) plus
PRESTATED_ADDENDUM_H_2026-10-20_164719.md (sha256 3ae3c451bd1c8ec311d724a1a375b1f59895785d30b59efc263a360e192b04f2) (G erratum) plus
PRESTATED_ADDENDUM_H2_2026-10-20_165900.md (sha256 4307d806c207ca7df8dbdef6b2fc448cac0011653c4ce40d5106f6036db03b74) (H erratum); later addenda rule where they differ.
Written from spec text only; the producer's code is not opened. Values stay in the containers and on the output Volume;
stdout carries booleans, counts, shas and paths only.

  M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
  "$M" run --detach check_honest_all.py --mode real --pinned <dir> --git-head <sha> > <fresh log> 2>&1
  modes: real | test | test-ruling (exit 0 expected) | test-fail-value | test-fail-pilot | test-fail-sha | test-fail-token | test-fail-defgap (exit 1 expected)

The pinned dir holds the committed inputs (PRESTATED section 2), materialized by the launch script with `git show`:
record.json, simpool.json, provenance.jsonl, release.json, folds.json, pilot.json, sources_v2.json and
routing/<commit>/<file name> for the 606 routing entries of sources_v2. The local side only reads their bytes.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-rerun_honall_kill-2026-11-19"
OUT_VOL = "probe-shift_aswin-surya_sub-rerun_honall-out_kill-2026-11-19"
ACTS_VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
W_VOL = "probe-shift_aswin-surya_sub-honsimdet_routec_w_kill-2026-11-15"
W_DIR = "xfer/honest_sim_detector_all/2026-10-18_121211"
TOK_REPO, TOK_REV = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
LAYER = 48
CELLS = ("command|L48", "reply|L48")
PROBLEMS = {"simh_vs_b": (("sim",), ("B",)), "a_vs_b": (("A",), ("B",))}
C = 0.01                      # RULE L42
TOL_GRAD, NEWTON_CAP = 1e-6, 100  # RULE L44 bar; Newton step cap (PRESTATED step 5)
TOL_REFIT, TOL_REPLAY, TOL_DEF = 1e-3, 1e-4, 1e-9
TOL_STRICT, TOL_REPLAY_FP32 = 1e-5, 1e-3  # addendum: strict run-score check; fp32-level replay tier
GRAD_SMALL, GAP_SMALL_REL, GAP_NEG_REL = 1e-4, 1e-4, 1e-9  # addendum diagnosis-map thresholds; fitter check
DHAT_SMALL = 1e-3  # 09's reading: max|Delta-hat| <= 1e-3
RESID_FLOOR, RESID_FRAC = 1e-6, 0.05  # RESID_EXPLAINED: max|resid| <= max(RESID_FLOOR, RESID_FRAC * max|Delta-hat|)
MU_TOL_REL, SIGMA_TOL_EXTRA = 1e-5, 1e-6  # mu_same: |dmu| <= MU_TOL_REL*max(1,|mu|); sigma_same: |ratio-1| <= 1/n_train + SIGMA_TOL_EXTRA
REFIT_LABEL = "independent refit (registered recipe + SPEC-SILENT defaults)"
ROW_ROLE = {"simh_vs_b": "beside_cleanroom_4b", "a_vs_b": "only_fit_check"}
HALF_DP, BOUNDARY = 0.005, 1e-3
BATCH = 16384
PRINTED_ROWS = {75: ("reply|L48", "simh_vs_b"), 76: ("command|L48", "simh_vs_b"), 77: ("reply|L48", "simh_vs_b"),
                78: ("command|L48", "simh_vs_b"), 95: ("reply|L48", "a_vs_b"), 96: ("command|L48", "a_vs_b"),
                97: ("reply|L48", "a_vs_b"), 98: ("command|L48", "a_vs_b")}
SIMPOOL_ITEM = "5_table7_row5_control"
PIN_SHA = {  # PRESTATED section 2
    "record.json": "10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46",
    "simpool.json": "2e17de0c852612d8a2b1d83468997c0cb6f787dcfee9a3c93eef6edee711fd11",
    "provenance.jsonl": "220d25d21943226791637581afda92faac595a3b15b9b6d6bbd3fd1a1e575971",
    "release.json": "8ea44f19639cbee9a964f57a6003d965a171a9e9facc275a20eec891a582fa2d",
    "folds.json": "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299",
    "pilot.json": "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27",
    "sources_v2.json": "d69ad9c41a091d6485cf8070b094cd62f2b820a74328e2d6540a76ffea8221ac",
}
PIN_SRC = {  # path @ commit of each pinned file (PRESTATED section 2); routing/<commit>/<name> come from sources_v2
    "record.json": "experiments/2026-10-16_honest_sim_detector_all/results/honest_all_routec_2026-10-18_121211.json@596bc767e9f2bcb97ac187024ff7efd411b2d6f5",
    "simpool.json": "experiments/2026-10-20_routec_simpool_numbers/results/routec_simpool_numbers_2026-10-20_020501.json@104087089d65a4561f7c15634017b283edb68714",
    "provenance.jsonl": "experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final17_0141.jsonl@98b5054fbf8306e54711b1a7bacd72b5f5e0af2a",
    "release.json": "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.2_2026-10-20.json@aee5aee0999260a25133221436e7ea1b319ca4a9",
    "folds.json": "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json@ec56720adde66baabe642a7fc842172c83737679",
    "pilot.json": "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json@344829af6dc4cd0473757398d542bb8e1f72f512",
    "sources_v2.json": "experiments/2026-10-16_honest_sim_detector_all/config/sources_v2.json@596bc767e9f2bcb97ac187024ff7efd411b2d6f5",
}
NPZ_SHA = {  # handoff L60-63 (== record saved_weights sha256)
    ("command|L48", "simh_vs_b"): "6a94d8219357d084c401953afd03a4c15e1c12ffaf44b0ce14b88a206679b1a0",
    ("command|L48", "a_vs_b"): "288e0bdb85d2340e801df168ce70ce9373d00715108983f42fd9d1f2b3f57064",
    ("reply|L48", "simh_vs_b"): "b4bef98d29b1c917c4f234365671df66ccb5e8a4c0e1d6353e49c0af34f9ef69",
    ("reply|L48", "a_vs_b"): "c6a65afa35550391609e34fca7296d85da3de23fe9ba511bc06d4397ef08b6f1",
}
REAL_EXPECT = {"sim": 869, "B": 1714, "A": 1516, "sim_scored": 868, "zero_row": 1, "pilot": 20, "sold_sim": 191, "staged": 109,
               "routing": 606}
MODES = ("real", "test", "test-ruling", "test-fail-value", "test-fail-pilot", "test-fail-sha", "test-fail-token", "test-fail-defgap",
         "hash-shards", "replay", "test-replay", "test-replay-fail-fold", "test-replay-fail-producer", "test-replay-fail-manifest")  # F, G, H
NPZ_NAMES = {"w": {"w", "coef", "coef_", "weights", "weight"}, "b": {"b", "intercept", "intercept_", "bias"},
             "mean": {"mean", "mu", "x_mean", "means"}, "std": {"std", "sd", "scale", "sigma", "x_std", "stds"},
             "folds": {"folds", "fold", "fold_ids"}}

LIGHT_PKGS = ("numpy==2.2.6", "scipy==1.15.3", "orjson==3.10.18", "tokenizers==0.21.1", "huggingface_hub==0.33.0")
image_light = modal.Image.debian_slim(python_version="3.12").pip_install(*LIGHT_PKGS)  # prep + real orchestrator (addendum C5)
image = modal.Image.debian_slim(python_version="3.12").pip_install(*LIGHT_PKGS, "scikit-learn==1.6.1", "torch==2.8.0")  # fit + toys
app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS_VOL)
w_vol = modal.Volume.from_name(W_VOL)
out_vol = modal.Volume.from_name(OUT_VOL, create_if_missing=True)


class Stop(Exception):
    pass


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


_STAGE = {"s": "init"}  # coarse, value-free stage name for sanitized crash reports (addendum C3)


def stage(s: str) -> None:
    _STAGE["s"] = s


def emit(lines: list, s: str) -> None:
    """Keep a value-free line for the final print and show it live in the app log (progress for long runs)."""
    lines.append(s)
    print("HONALL-LIVE " + s, flush=True)


def meta_name(key: str) -> str:  # extract_chain.py @ afefd8817 L102-103
    return hashlib.sha1(key.encode()).hexdigest()[:20]


def auroc(pos, neg) -> float:
    """Pooled Mann-Whitney AUROC, class 1 = pos, ties counted 1/2."""
    import numpy as np
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def half_up_2dp(x: float) -> str:
    """Round-half-up of the exact float value to 2 dp, as a string (the toy's printed values; addendum E)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def printed_unchanged(x: float, printed: str) -> bool:
    """09 (addendum C1): round-half-up of the exact float value to 2 dp equals the printed value, compared numerically (Decimal)."""
    from decimal import ROUND_HALF_UP, Decimal
    return Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) == Decimal(printed)


BOUNDARY_CASES = [(0.525, "0.52", False), (0.52 + 0.005, "0.52", False), (0.625, "0.62", False), (0.625, "0.63", True),
                  (0.515, "0.52", True), (0.5249999999, "0.52", True), (0.53, "0.53", True), (0.5349999999, "0.54", False)]


def cell_file(cell: str, problem: str) -> str:
    return f"{W_DIR}/w_{cell.replace('|', '_')}__{problem}.npz"


# ----------------------------------------------------------------------------------------------- prep (CPU, per staged file)

def prep_one(acts_root: Path, shard_dir: Path, job: dict) -> dict:
    """Parse one staged file; for pooled epochs check meta and gather layer-48 rows (PRESTATED step 3)."""
    import numpy as np
    import orjson

    stage("prep:read_and_parse")
    raw = (acts_root / job["staged_key"]).read_bytes()
    out = {"idx": job["idx"], "staged_key": job["staged_key"], "sha_ok": sha256(raw) == job["staged_sha"], "n_epochs": 0,
           "fixed_all": 0, "fixed_matched": 0, "matched": [], "dup_in_file": 0, "meta_fail": {}, "counts": {}, "d": None}
    if not out["sha_ok"]:
        return out
    wanted, fixed = set(job["wanted"]), set(job["fixed_ids"])
    eps = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        ep = orjson.loads(line)
        out["n_epochs"] += 1
        ids = ep["ids"]
        flags = [c["kind"] == "call_end" and c["readout"] is not None and ids[c["readout"]] in fixed for c in ep["calls"]]
        out["fixed_all"] += sum(flags)
        if ep["key"] in wanted:
            out["dup_in_file"] += ep["key"] in eps
            eps[ep["key"]] = {"arm": ep["arm"], "fixed": flags,
                              "calls": [{f: c.get(f) for f in ("kind", "readout", "irid", "src_irid", "call_index")}
                                        | {"span": bool(c.get("span"))} for c in ep["calls"]]}
            out["fixed_matched"] += sum(flags)
    del raw
    stage("prep:meta_and_rows")
    root = acts_root / job["extraction_root"]
    chunk_idx, chunk_arr, gathered = {}, {}, {}
    for key in sorted(eps):
        st = eps[key]
        fails = []
        mp = root / "meta" / job["meta_arm"] / f"{meta_name(key)}.json"
        if not mp.exists():
            out["meta_fail"][key] = ["missing"]
            continue
        m = json.loads(mp.read_text())
        if m.get("key") != key:
            fails.append("key")
        if m.get("arm") != st["arm"]:
            fails.append("arm")
        if m.get("inputs_sha256") != job["staged_sha"]:
            fails.append("inputs_sha256")
        mc = m.get("calls") or []
        if len(mc) != len(st["calls"]):
            fails.append("n_calls")
        cmd, rep = [], []
        for c, s, fx in zip(mc, st["calls"], st["fixed"]):
            if any(c.get(f) != s[f] for f in ("kind", "readout", "irid", "src_irid", "call_index")):
                fails.append("call_fields")
                break
            if (c.get("row") is not None) != (s["readout"] is not None):
                fails.append("row_presence")
                break
            if ("row_mean" in c) != (s["readout"] is not None and s["kind"] == "output_end" and s["span"]):
                fails.append("row_mean_presence")
                break
            if s["kind"] == "call_end" and c.get("row") is not None and not fx:
                cmd.append(c["row"])
            if "row_mean" in c:
                rep.append(c["row_mean"])
        if any(r >= m.get("n_rows", -1) or r < 0 for r in cmd + rep):
            fails.append("row_range")
        ch = m.get("chunk")
        if not ch:
            out["meta_fail"][key] = fails + ["chunk_name"]
            continue
        if ch not in chunk_idx:
            chunk_idx[ch] = json.loads((root / "chunks" / f"{ch}.json").read_text())
            chunk_arr[ch] = np.load(root / "by_layer" / f"L{job['layer']}" / f"{ch}.npy", mmap_mode="r")
        arr = chunk_arr[ch]
        if arr.dtype != np.float16 or arr.ndim != 2 or arr.shape[0] != chunk_idx[ch]["rows"]:
            fails.append("chunk_array")
        ent = [e for e in chunk_idx[ch]["epochs"] if e["key"] == key]
        if len(ent) != 1 or ent[0]["row_start"] != m.get("row_start") or ent[0]["n_rows"] != m.get("n_rows"):
            fails.append("chunk_index")
        if fails:
            out["meta_fail"][key] = fails
            continue
        arr, r0 = chunk_arr[ch], m["row_start"]
        out["d"] = int(arr.shape[1])
        gathered[key] = (np.asarray(arr[[r0 + r for r in cmd]]) if cmd else np.zeros((0, arr.shape[1]), np.float16),
                         np.asarray(arr[[r0 + r for r in rep]]) if rep else np.zeros((0, arr.shape[1]), np.float16))
        out["counts"][key] = [len(cmd), len(rep)]
    out["matched"] = sorted(eps)
    stage("prep:write")
    keys = sorted(gathered)
    if keys:
        shard_dir.mkdir(parents=True, exist_ok=True)
        stem = shard_dir / f"{job['idx']:03d}"
        if stem.with_suffix(".npz").exists() or stem.with_suffix(".json").exists():
            raise FileExistsError("shard exists (results are append-only); use a fresh stamp")
        tmp = shard_dir / f".{job['idx']:03d}.tmp.npz"
        np.savez(tmp, X_cmd=np.concatenate([gathered[k][0] for k in keys]), X_rep=np.concatenate([gathered[k][1] for k in keys]),
                 cmd_counts=np.array([gathered[k][0].shape[0] for k in keys], np.int64),
                 rep_counts=np.array([gathered[k][1].shape[0] for k in keys], np.int64))
        os.replace(tmp, stem.with_suffix(".npz"))
        stem.with_suffix(".json").write_text(json.dumps({"keys": keys, "staged_key": job["staged_key"], "staged_sha": job["staged_sha"]}))
    out["shard"] = f"{job['idx']:03d}" if keys else None
    return out


@app.function(image=image_light, cpu=(2, 2), memory=(8192, 8192), timeout=300, startup_timeout=120, scaledown_window=2, max_containers=109,
              retries=0,  # addendum E: request = limit; one round of 109
              volumes={"/acts": acts_vol.read_only(), "/out": out_vol})
def prep(job: dict) -> dict:
    err = None
    try:
        res = prep_one(Path("/acts"), Path("/out") / job["stamp"] / "shards", job)
        out_vol.commit()
    except Exception as e:  # sanitized (addendum C3): type and stage only, raised below, outside the except
        err = f"{type(e).__name__} at {_STAGE['s']}"
    if err is not None:
        raise RuntimeError(f"prep {job['idx']:03d} failed: {err}")
    return res


# ------------------------------------------------------------------------------------------------- fit and replay (GPU)

def objective(X, tr, y, mu, sd, th, grad, hess):
    """scikit-learn's L2 objective G = 0.5|w|^2 + C sum log(1+exp(-s(w.z+b))), unpenalized b, on rows tr: value, gradient, Hessian (float64)."""
    import torch
    d, device = X.shape[1], X.device
    w, b = th[:d], th[d]
    f = 0.5 * float(w @ w)
    g = torch.zeros(d + 1, dtype=torch.float64, device=device)
    H = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)
    if grad:
        g[:d] = w
    if hess:
        H[:d, :d] = torch.eye(d, dtype=torch.float64, device=device)
    for s in range(0, len(tr), BATCH):
        Z = (X[tr[s:s + BATCH]].double() - mu) / sd
        yb = y[s:s + BATCH]
        z = Z @ w + b
        f += C * float((torch.nn.functional.softplus(z) - yb * z).sum())
        if grad or hess:
            p = torch.sigmoid(z)
        if grad:
            r = p - yb
            g[:d] += C * (Z.T @ r)
            g[d] += C * r.sum()
        if hess:
            q = p * (1 - p)
            Zq = Z * q[:, None]
            H[:d, :d] += C * (Z.T @ Zq)
            H[:d, d] += C * Zq.sum(0)
            H[d, d] += C * q.sum()
    if hess:
        H[d, :d] = H[:d, d]
    return f, g, H


def newton_fit(X, tr, y, mu, sd, device):
    """Minimize `objective` with float64 Newton + Armijo until max|grad| <= TOL_GRAD (RULE L44) or NEWTON_CAP steps."""
    import torch
    d = X.shape[1]
    theta = torch.zeros(d + 1, dtype=torch.float64, device=device)

    def evaluate(th, grad, hess):
        return objective(X, tr, y, mu, sd, th, grad, hess)

    steps, gmax = 0, float("inf")
    for it in range(NEWTON_CAP + 1):
        f, g, H = evaluate(theta, True, True)
        gmax = float(g.abs().max())
        if gmax <= TOL_GRAD or it == NEWTON_CAP:
            break
        step = torch.cholesky_solve(-g[:, None], torch.linalg.cholesky(H))[:, 0]
        slope = float(g @ step)
        t = 1.0
        if -slope > 1e-8:
            while evaluate(theta + t * step, False, False)[0] > f + 1e-4 * t * slope and t > 1e-10:
                t *= 0.5
        theta = theta + t * step
        steps += 1
    return theta[:d], theta[d], {"max_abs_grad": gmax, "newton_steps": steps, "certified": gmax <= TOL_GRAD}


def fold_stats(X, tr):
    """mu, sigma over the training rows, two-pass float64, ddof 0, sigma == 0 -> 1 (SPEC-SILENT defaults 2-3)."""
    import torch
    mu = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    for s in range(0, len(tr), BATCH):
        mu += X[tr[s:s + BATCH]].double().sum(0)
    mu /= len(tr)
    var = torch.zeros_like(mu)
    for s in range(0, len(tr), BATCH):
        var += ((X[tr[s:s + BATCH]].double() - mu) ** 2).sum(0)
    sd = (var / len(tr)).sqrt()
    sd[sd == 0] = 1.0
    return mu, sd


def row_logits(X, rows, mu, sd, w, b):
    import torch
    out = []
    for s in range(0, len(rows), BATCH):
        Xb = X[rows[s:s + BATCH]].double()
        out.append(((Xb - mu) / sd if mu is not None else Xb) @ w + b)
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64, device=X.device)


def run_means(logits, row_run_np, rows_np, n_runs):
    import numpy as np
    s = np.zeros(n_runs)
    n = np.zeros(n_runs)
    np.add.at(s, row_run_np[rows_np], logits)
    np.add.at(n, row_run_np[rows_np], 1)
    return s, n


def resolve_npz(z, d):
    """PRESTATED step 6: weights (5, d), intercept (5,), optional mean/std (5, d), optional fold ids. None if unresolved."""
    import numpy as np
    found = {}
    for role, names in NPZ_NAMES.items():
        hits = [k for k in z.files if k.lower() in names]
        if len(hits) > 1:
            return None
        if hits:
            found[role] = hits[0]
    if "w" not in found or "b" not in found or (("mean" in found) != ("std" in found)):
        return None
    W, B = np.asarray(z[found["w"]], np.float64), np.asarray(z[found["b"]], np.float64)
    W = W.reshape(5, d) if W.size == 5 * d and W.shape in ((5, d), (5, 1, d), (5, d, 1)) else None
    B = B.reshape(5) if B.shape in ((5,), (5, 1)) else None
    if W is None or B is None:
        return None
    lay = {"w": W, "b": B, "mode": "stored" if "mean" in found else "recompute+raw", "keys": found}
    if "mean" in found:
        M, S = np.asarray(z[found["mean"]], np.float64), np.asarray(z[found["std"]], np.float64)
        if M.shape not in ((5, d), (5, 1, d)) or S.shape not in ((5, d), (5, 1, d)):  # never guess a transposed layout (A8)
            return None
        lay["mean"], lay["std"] = M.reshape(5, d), S.reshape(5, d)
    if "folds" in found:
        F = np.asarray(z[found["folds"]]).reshape(-1)
        if F.shape != (5,):
            return None
        lay["folds"] = [int(v) for v in F]
    return lay


def replay_score(ctx: dict, mode: str, rotate: bool) -> dict:
    """Score each held-out fold with a stored model (rotate: fold folds[i] by model i+1 mod 5, control N1) and apply the replay rule."""
    import numpy as np
    import torch
    order, lay, folds, device = ctx["order"], ctx["lay"], ctx["folds"], ctx["device"]
    s, n = np.zeros(len(order)), np.zeros(len(order))
    for i, fid in enumerate(folds):
        j = (i + 1) % 5 if rotate else i
        te = np.nonzero(ctx["row_in"] & (ctx["row_fold"] == fid))[0]
        tet = torch.from_numpy(te).to(device)
        wj, bj = torch.from_numpy(lay["w"][j]).to(device), float(lay["b"][j])
        if mode == "stored":
            mu, sd = torch.from_numpy(lay["mean"][j]).to(device), torch.from_numpy(lay["std"][j]).to(device)
        elif mode == "recompute":
            mu, sd = ctx["fits"][folds[j]][0], ctx["fits"][folds[j]][1]
        else:
            mu, sd = None, None
        si, ni = run_means(row_logits(ctx["X"], tet, mu, sd, wj, bj).cpu().numpy(), ctx["row_run"], te, len(order))
        s += si
        n += ni
    sc = {order[i]: float(s[i] / n[i]) for i in range(len(order)) if ctx["in_p"][i] and n[i] > 0}
    runs, pos, neg, rec = ctx["runs"], ctx["pos"], ctx["neg"], ctx["rec"]
    a = auroc([x for k, x in sc.items() if runs[k][0] in pos], [x for k, x in sc.items() if runs[k][0] in neg])
    dmax = max((abs(sc[k] - rec["scores"][k]) for k in set(sc) & set(rec["scores"])), default=float("inf"))
    same_keys, da = set(sc) == set(rec["scores"]), abs(a - rec["auroc"])
    tier = ("tight" if same_keys and dmax <= TOL_REPLAY and da <= TOL_REPLAY else
            "fp32" if same_keys and dmax <= TOL_REPLAY_FP32 and da <= TOL_REPLAY_FP32 else None)  # addendum A13
    return {"auroc": a, "run_scores": sc, "max_abs_run_score_diff": dmax, "tier": tier, "match": tier is not None,
            "diff_bucket": bucket(dmax, (1e-4, 1e-3, 1e-2))}


def discordant(a, b, pos):
    """Pairs i < j that a and b order differently (a tie counts as its own order): over all pairs, and over pos-neg pairs only."""
    import numpy as np
    a, b, pos = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(pos, bool)
    n, tot, pn = len(a), 0, 0
    jj = np.arange(n)[None, :]
    for s in range(0, n, 512):
        e = min(s + 512, n)
        bad = (np.sign(a[s:e, None] - a[None, :]) != np.sign(b[s:e, None] - b[None, :])) & (jj > np.arange(s, e)[:, None])
        tot += int(bad.sum())
        pn += int((bad & (pos[s:e, None] != pos[None, :])).sum())
    return tot, pn


def bracket(x: float, ts=(1e-4, 1e-3, 1e-2)) -> str:
    """Booleans x <= t for each threshold t (value-free)."""
    return ",".join(f"le_{t:g}={str(x <= t).upper()}" for t in ts)


def bucket(x: float, ts) -> str:
    """The smallest threshold t with x <= t, else gt_<largest> (value-free)."""
    for t in ts:
        if x <= t:
            return f"le_{t:g}"
    return f"gt_{ts[-1]:g}"


def fit_quality(ctx: dict, mode: str, my_scores: dict, stored_scores: dict, trs: dict, ys: dict) -> dict:
    """PRESTATED addendum A9-A10. Per fold: the stored model in my coordinates, grad G and Hessian of MY objective at it on MY training rows;
    Delta-hat = run mean of x~' H^-1 grad G (first-order stored-minus-refit). Also the stats check and the rank comparison."""
    import numpy as np
    import torch
    X, lay, folds, fits, order = ctx["X"], ctx["lay"], ctx["folds"], ctx["fits"], ctx["order"]
    d, device = X.shape[1], X.device
    dh_s, dh_n = np.zeros(len(order)), np.zeros(len(order))
    per_fold = []
    for i, fid in enumerate(folds):
        mu, sd, w_m, b_m = fits[fid]
        tr, y = trs[fid], ys[fid]
        ws, bs = torch.from_numpy(lay["w"][i]).to(device), float(lay["b"][i])
        if mode == "stored":
            ms, ss = torch.from_numpy(lay["mean"][i]).to(device), torch.from_numpy(lay["std"][i]).to(device)
        elif mode == "recompute":
            ms, ss = mu, sd
        else:
            ms, ss = torch.zeros_like(mu), torch.ones_like(sd)
        # gradient of the pre-stated objective at the stored (w, b), on this fold's training rows, standardized with the STORED mu/sigma (B1b)
        _, g_sc, _ = objective(X, tr, y, ms, ss, torch.cat([ws, torch.tensor([bs], dtype=torch.float64, device=device)]), True, False)
        # the stored model in MY coordinates: ws.(x - ms)/ss + bs = wp.z + bp with z = (x - mu)/sd (the same logit function)
        wp = ws * sd / ss
        bp = bs + float((ws * (mu - ms) / ss).sum())
        th_s = torch.cat([wp, torch.tensor([bp], dtype=torch.float64, device=device)])
        j_s, g, H = objective(X, tr, y, mu, sd, th_s, True, True)
        j_m = objective(X, tr, y, mu, sd, torch.cat([w_m, b_m.reshape(1)]), False, False)[0]
        v = torch.cholesky_solve(g[:, None], torch.linalg.cholesky(H))[:, 0]
        te = np.nonzero(ctx["row_in"] & (ctx["row_fold"] == fid))[0]
        s_, n_ = run_means(row_logits(X, torch.from_numpy(te).to(device), mu, sd, v[:d], v[d]).cpu().numpy(), ctx["row_run"], te, len(order))
        dh_s += s_
        dh_n += n_
        f = {"fold": fid, "grad_stored_coords": float(g_sc.abs().max()), "grad_my_coords": float(g.abs().max()),
             "J_stored": j_s, "J_mine": j_m, "J_gap": j_s - j_m, "n_train": int(len(tr)),
             "fitter_ok": j_s - j_m >= -GAP_NEG_REL * max(1.0, abs(j_m)),
             "gap_small": 0.0 <= j_s - j_m <= GAP_SMALL_REL * max(1.0, abs(j_m)), "grad_small": float(g_sc.abs().max()) <= GRAD_SMALL}
        if mode == "stored":  # stored vs recomputed standardization (B1b); same = within fp32 storage / the ddof factor
            m_np, s_np, mu_np, sd_np = lay["mean"][i], lay["std"][i], mu.cpu().numpy(), sd.cpu().numpy()
            f |= {"max_abs_dmu": float(np.abs(mu_np - m_np).max()), "max_abs_sigma_ratio_m1": float(np.abs(s_np / sd_np - 1.0).max()),
                  "mu_same": bool(np.all(np.abs(mu_np - m_np) <= MU_TOL_REL * np.maximum(1.0, np.abs(m_np)))),
                  "sigma_same": bool(float(np.abs(s_np / sd_np - 1.0).max()) <= 1.0 / len(tr) + SIGMA_TOL_EXTRA)}
        per_fold.append(f)
    keys = sorted(set(my_scores) & set(stored_scores))
    dhat = np.array([dh_s[ctx["pos_of"][k]] / dh_n[ctx["pos_of"][k]] for k in keys])
    diff = np.array([stored_scores[k] - my_scores[k] for k in keys])
    max_res, max_dh, max_diff = float(np.abs(diff - dhat).max()), float(np.abs(dhat).max()), float(np.abs(diff).max())
    swaps, swaps_pn = discordant([stored_scores[k] for k in keys], [my_scores[k] for k in keys],
                                 [ctx["runs"][k][0] in ctx["pos"] for k in keys])
    stats_ok = all(f["mu_same"] and f["sigma_same"] for f in per_fold) if mode == "stored" else None
    allf = lambda k: all(f[k] for f in per_fold)  # noqa: E731
    if mode != "stored":
        branch = "unclassified_no_stored_stats"
    elif not all(f["sigma_same"] for f in per_fold):
        branch = "standardization_default_differs_G1"
    elif allf("grad_small") and allf("gap_small") and allf("mu_same"):
        branch = "same_objective_producer_fit_unconverged"
    elif not any(f["grad_small"] for f in per_fold) and not any(f["gap_small"] for f in per_fold):
        branch = "definition_gap_G4"
    else:
        branch = "unclassified"
    return {"n_runs": len(keys), "max_abs_residual": max_res, "max_abs_dhat": max_dh, "max_abs_stored_minus_refit": max_diff,
            "resid_explained": max_res <= max(RESID_FLOOR, RESID_FRAC * max_dh), "per_fold": per_fold, "stats_ok": stats_ok,
            "fitter_ok": allf("fitter_ok"), "branch": branch,
            "rank_identical": swaps == 0, "swaps_all_pairs": swaps, "swaps_pos_neg": swaps_pn}


def fit_and_compare(payload: dict, device: str) -> dict:
    """Refit (step 5), replay (step 6), comparisons and negative controls N1/N2. Returns values (for the record) and value-free lines."""
    import numpy as np
    import torch

    stage("fit:load")
    shard_dir, w_root, my_w_dir = Path(payload["shard_dir"]), Path(payload["w_root"]), Path(payload["my_w_dir"])
    runs, ek2rk = payload["runs"], payload["evalkey_to_run"]
    order = sorted(runs)
    pos_of = {rk: i for i, rk in enumerate(order)}
    cls_arr = np.array([runs[rk][0] for rk in order])
    fold_arr = np.array([runs[rk][1] for rk in order])
    parts = {c: [] for c in CELLS}
    rr = {c: [] for c in CELLS}
    for f in payload["shard_files"]:
        side = json.loads((shard_dir / f"{f}.json").read_text())
        z = np.load(shard_dir / f"{f}.npz")
        for cell, xk, nk in (("command|L48", "X_cmd", "cmd_counts"), ("reply|L48", "X_rep", "rep_counts")):
            parts[cell].append(z[xk])
            rr[cell].extend(pos_of[ek2rk[k]] for k, n in zip(side["keys"], z[nk]) for _ in range(int(n)))
    lines, vals, flags, info = [], {}, {}, {}
    for cell in CELLS:
        Xnp = np.concatenate(parts[cell])
        X = torch.from_numpy(Xnp).to(device)
        del Xnp
        row_run = np.asarray(rr[cell], np.int64)
        d = X.shape[1]
        for prob, (pos, neg) in PROBLEMS.items():
            stage(f"fit:refit {cell} {prob}")
            rec = payload["record"][cell][prob]
            in_p = np.isin(cls_arr, pos + neg)
            y_run = np.isin(cls_arr, pos).astype(np.float64)
            row_in, row_fold = in_p[row_run], fold_arr[row_run]
            s_my, n_my = np.zeros(len(order)), np.zeros(len(order))
            certs, fits, trs, ys = {}, {}, {}, {}
            for k in range(5):
                tr = np.nonzero(row_in & (row_fold != k))[0]
                te = np.nonzero(row_in & (row_fold == k))[0]
                trt, tet = torch.from_numpy(tr).to(device), torch.from_numpy(te).to(device)
                mu, sd = fold_stats(X, trt)
                yt = torch.from_numpy(y_run[row_run[tr]]).to(device)
                w, b, cert = newton_fit(X, trt, yt, mu, sd, device)
                certs[k] = cert
                fits[k] = (mu, sd, w, b)
                trs[k], ys[k] = trt, yt
                print(f"HONALL-LIVE STAGE fit {cell} {prob} fold={k} certified={str(cert['certified']).upper()}", flush=True)
                s, n = run_means(row_logits(X, tet, mu, sd, w, b).cpu().numpy(), row_run, te, len(order))
                s_my += s
                n_my += n
            my_scores = {order[i]: float(s_my[i] / n_my[i]) for i in range(len(order)) if in_p[i] and n_my[i] > 0}
            my_auc = auroc([v for k_, v in my_scores.items() if runs[k_][0] in pos], [v for k_, v in my_scores.items() if runs[k_][0] in neg])
            my_w_dir.mkdir(parents=True, exist_ok=True)
            wp = my_w_dir / f"my_w_{cell.replace('|', '_')}__{prob}.npz"
            with open(wp, "xb") as fh:
                np.savez(fh, w=np.stack([fits[k][2].cpu().numpy() for k in range(5)]), b=np.array([float(fits[k][3]) for k in range(5)]),
                         mean=np.stack([fits[k][0].cpu().numpy() for k in range(5)]), std=np.stack([fits[k][1].cpu().numpy() for k in range(5)]),
                         folds=np.arange(5))
            common = sorted(set(my_scores) & set(rec["scores"]))
            sdiff = max((abs(my_scores[k_] - rec["scores"][k_]) for k_ in common), default=float("inf"))
            refit_match = abs(my_auc - rec["auroc"]) <= TOL_REFIT
            ncert = sum(c["certified"] for c in certs.values())
            tag = f"{cell} {prob}"
            keys_eq = set(my_scores) == set(rec["scores"])
            emit(lines, f"HONALL REFIT {tag} tight_tier_1e-3={'MATCH' if refit_match else 'MISMATCH'} "
                        f"auroc_diff_bucket={bucket(abs(my_auc - rec['auroc']), (1e-4, 1e-3, 5e-3))} certified={ncert}/5 runs={len(my_scores)}"
                        + ("" if ncert == 5 else f" uncertified ({5 - ncert} of 5 fits)"))
            emit(lines, f"HONALL REFIT_SCORES {tag} keys_equal={str(keys_eq).upper()} "
                        f"within_1e-3={str(sdiff <= 1e-3).upper()} strict_within_1e-5={str(sdiff <= TOL_STRICT).upper()}")
            flags[f"certified {tag}"] = ncert
            info[tag] = {"refit_tight": refit_match, "strict": keys_eq and sdiff <= TOL_STRICT, "fitq": None}
            v = {"refit": {"auroc": my_auc, "run_scores": my_scores, "certificates": certs, "max_abs_run_score_diff": sdiff,
                           "weights_file": str(wp)}, "record": {"auroc": rec["auroc"]}}
            stage(f"fit:replay {tag}")
            # ---- replay with the stored weights
            npz_path = w_root / payload["saved"][cell][prob]["path"]
            raw = npz_path.read_bytes()
            sha_ok = sha256(raw) == payload["npz_sha"][cell][prob]
            lay = None
            if sha_ok:
                zz = np.load(npz_path)
                shapes = {}
                for k_ in sorted(zz.files):
                    try:
                        shapes[k_] = list(zz[k_].shape)
                    except ValueError:  # object arrays need pickle; never loaded
                        shapes[k_] = "object"
                emit(lines, f"HONALL NPZ {tag} sha=MATCH keys={sorted(zz.files)} shapes={list(shapes.values())}")
                if "object" not in shapes.values():
                    lay = resolve_npz(zz, d)
            else:
                emit(lines, f"HONALL NPZ {tag} sha=MISMATCH")
            replay = {"resolved": lay is not None}
            if lay is None:
                emit(lines, f"HONALL REPLAY {tag} UNRESOLVED")
                flags[f"replay {tag}"] = False
            else:
                folds = lay.get("folds") or [int(x) for x in payload["saved"][cell][prob]["folds"]]
                modes = ["stored"] if lay["mode"] == "stored" else ["recompute", "raw"]
                ctx = {"X": X, "row_run": row_run, "row_in": row_in, "row_fold": row_fold, "folds": folds, "lay": lay, "fits": fits,
                       "order": order, "pos_of": pos_of, "in_p": in_p, "runs": runs, "pos": pos, "neg": neg, "rec": rec, "device": device}
                per_mode = {m: replay_score(ctx, m, False) for m in modes}
                hits = [m for m in modes if per_mode[m]["match"]]
                ok = len(hits) == 1
                emit(lines, f"HONALL REPLAY {tag} {'MATCH' if ok else 'MISMATCH'} mode={hits[0] if ok else '/'.join(modes)} "
                            f"tier={per_mode[hits[0]]['tier'] if ok else 'none'} modes_matching={len(hits)} "
                            f"max_diff_bucket={'/'.join(per_mode[m]['diff_bucket'] for m in modes)}")
                flags[f"replay {tag}"] = ok
                rot = replay_score(ctx, hits[0] if ok else modes[0], True)
                emit(lines, f"HONALL NEGCTRL N1 {tag} {'UNEXPECTED_PASS' if rot['match'] else 'FAIL_AS_PREDICTED'}")
                flags[f"N1 {tag}"] = not rot["match"]
                replay |= {"layout_keys": lay["keys"], "folds": folds, "modes": per_mode, "rotated": {"match": rot["match"]}}
                if ok:  # addendum: fit-quality reading (09) and diagnosis map (reviewer B1)
                    stage(f"fit:fitq {tag}")
                    fq = fit_quality(ctx, hits[0], my_scores, per_mode[hits[0]]["run_scores"], trs, ys)
                    info[tag]["fitq"] = fq
                    replay["fit_quality"] = fq
                    pf = fq["per_fold"]
                    emit(lines, f"HONALL FITQ {tag} resid_explained={str(fq['resid_explained']).upper()} "
                                f"resid[{bracket(fq['max_abs_residual'])}] dhat[{bracket(fq['max_abs_dhat'])}] "
                                f"stored_minus_refit[{bracket(fq['max_abs_stored_minus_refit'])}] rank_identical={str(fq['rank_identical']).upper()} "
                                f"swaps_all_pairs={fq['swaps_all_pairs']} swaps_pos_neg={fq['swaps_pos_neg']} runs={fq['n_runs']}")
                    emit(lines, f"HONALL FITQ_STORED {tag} fitter_ok={str(fq['fitter_ok']).upper()} "
                                f"grad_stored_coords_small={sum(f['grad_small'] for f in pf)}/5 "
                                f"grad_stored_coords_max[{bracket(max(f['grad_stored_coords'] for f in pf), (1e-6, 1e-4, 1e-2))}] "
                                f"J_gap_small={sum(f['gap_small'] for f in pf)}/5 J_gap_nonneg={sum(f['fitter_ok'] for f in pf)}/5 "
                                f"stats={'NA' if fq['stats_ok'] is None else str(fq['stats_ok']).upper()} "
                                + ("" if fq["stats_ok"] is None else
                                   f"mu_same={sum(f['mu_same'] for f in pf)}/5 sigma_same={sum(f['sigma_same'] for f in pf)}/5 "
                                   f"dmu_max[{bracket(max(f['max_abs_dmu'] for f in pf), (1e-6, 1e-4, 1e-2))}] "
                                   f"sigma_ratio_m1_max[{bracket(max(f['max_abs_sigma_ratio_m1'] for f in pf), (1e-6, 1e-4, 1e-2))}] ")
                                + f"branch={fq['branch']}")
            v["replay"] = replay
            n2 = abs(my_auc - (rec["auroc"] + 0.01)) <= TOL_REFIT  # the REFIT comparator, given the record value + 0.01
            emit(lines, f"HONALL NEGCTRL N2 {tag} {'UNEXPECTED_PASS' if n2 else 'FAIL_AS_PREDICTED'}")
            flags[f"N2 {tag}"] = not n2
            vals.setdefault(cell, {})[prob] = v
        del X
        if device == "cuda":
            torch.cuda.empty_cache()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    return {"lines": lines, "values": vals, "flags": flags, "info": info, "device": gpu}


@app.function(image=image, gpu="H100", cpu=(8, 8), memory=(65536, 65536), timeout=1500, startup_timeout=240, scaledown_window=2,
              retries=0,  # addendum E: request = limit
              volumes={"/w": w_vol.read_only(), "/out": out_vol})
def fit(payload: dict) -> dict:
    err = None
    try:
        out_vol.reload()
        print("HONALL-LIVE STAGE fit start", flush=True)
        res = fit_and_compare(payload, "cuda")
        out_vol.commit()
    except Exception as e:  # sanitized (addendum C3)
        err = f"{type(e).__name__} at {_STAGE['s']}"
    if err is not None:
        raise RuntimeError(f"fit failed: {err}")
    return res


# --------------------------------------------------------------------------------------------- orchestration and gates

def fixed_token_ids():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    p = hf_hub_download(TOK_REPO, "tokenizer.json", revision=TOK_REV)
    tok = Tokenizer.from_file(p)
    ids = {s: tok.token_to_id(s) for s in ("0", "n")}
    ok = all(v is not None for v in ids.values()) and ids["0"] != ids["n"] and all(tok.decode([v]) == s for s, v in ids.items())
    return ok, ids, sha256(Path(p).read_bytes())


def run_keys_in(obj, universe):
    """Run keys anywhere in a nested structure (keys or string leaves), restricted to known run keys."""
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in universe:
                found.add(k)
            found |= run_keys_in(v, universe)
    elif isinstance(obj, list):
        for v in obj:
            found |= run_keys_in(v, universe)
    elif isinstance(obj, str) and obj in universe:
        found.add(obj)
    return found


def pipeline(mode, blobs, meta, acts_root: Path, w_root: str, expect, pin_sha, prep_runner, fit_runner, out_root: Path) -> dict:
    stamp = meta["stamp"]
    run_dir = out_root / stamp
    lines, res = [], {"label": f"PRELIMINARY route C rerun: honest_all value check; {REFIT_LABEL}. SPEC-SILENT defaults: class weights none, "
                               "ddof 0, sigma 0 -> 1, sklearn sum objective, unpenalized intercept, mu/sigma from the problem's training rows "
                               "only, float64 standardization",
                      "mode": mode, "stamp": stamp, "meta": meta, "prestated_sha256": {
                          "PRESTATED_2026-10-20_130735.md": "54bf522c3becbad9400d77ae83048d91b65f7d597e5c3811da13a3808a73cf9e",
                          "PRESTATED_ADDENDUM_2026-10-20_131806.md": "63e6dd03099d4ce94e634490e1cb2698cecad975112afb24f256283832b50f6e",
                          "PRESTATED_ADDENDUM_C_2026-10-20_134810.md": "c485637904e1d2f3e48e6f239d9401d615dc3afbc211e4d91743a85ed0a864c1",
                          "PRESTATED_ADDENDUM_D_2026-10-20_140146.md": "16486c817b1e1d1f0dd67b8e859a9078f1c8dbf598649a53ded32c3db1336284",
                          "PRESTATED_ADDENDUM_E_2026-10-20_141519.md": "66bbcbaea2451de08897803d4948d5132f5356cde2faeae3aa9b8db0da11dcc2"},
                      "config": {"C": C, "tol_grad": TOL_GRAD, "newton_cap": NEWTON_CAP, "tol_refit": TOL_REFIT, "tol_replay": TOL_REPLAY,
                                 "tol_def": TOL_DEF, "half_dp": HALF_DP, "boundary": BOUNDARY, "batch": BATCH, "layer": LAYER,
                                 "acts_volume": ACTS_VOL, "weights_volume": W_VOL, "weights_dir": W_DIR, "tokenizer": [TOK_REPO, TOK_REV],
                                 "pinned_sources": PIN_SRC if mode == "real" else "toy", "npz_sha256": {f"{c}::{p}": h for (c, p), h in NPZ_SHA.items()},
                                 "addendum": {"tol_strict_run_score": TOL_STRICT, "tol_replay_fp32": TOL_REPLAY_FP32,
                                              "grad_small_stored_coords": GRAD_SMALL, "gap_small_rel": GAP_SMALL_REL, "gap_negative_rel": GAP_NEG_REL,
                                              "dhat_small": DHAT_SMALL, "resid_floor": RESID_FLOOR, "resid_frac": RESID_FRAC,
                                              "mu_tol_rel": MU_TOL_REL, "sigma_tol": "1/n_train + " + repr(SIGMA_TOL_EXTRA),
                                              "auroc_diff_buckets": [1e-4, 1e-3, 5e-3], "replay_diff_buckets": [1e-4, 1e-3, 1e-2],
                                              "brackets": [1e-4, 1e-3, 1e-2], "grad_brackets": [1e-6, 1e-4, 1e-2],
                                              "printed_rule": "Decimal(x).quantize(0.01, ROUND_HALF_UP) == Decimal(printed)",
                                              "timeouts_s": {"prep": 300, "fit": 1500, "orchestrate": 3300, "orchestrate_toy": 600},
                                              "cpu_request_limit": {"prep": [2, 2], "fit": [8, 8], "orchestrate": [1, 1], "orchestrate_toy": [1, 1]},
                                              "memory_mib_request_limit": {"prep": [8192, 8192], "fit": [65536, 65536], "orchestrate": [2048, 2048], "orchestrate_toy": [2048, 2048]},
                                              "startup_timeouts_s": {"prep": 120, "fit": 240, "orchestrate": 120, "orchestrate_toy": 180},
                                              "scaledown_window_s": 2, "prep_max_containers": 109,
                                              "hard_bound_usd": 4.519147}},
                      "gates": {}, "diag": {}, "inputs": {}}
    exitbad = []

    def gate(name, ok, detail=""):
        stage(f"gate {name}")
        emit(lines, f"HONALL GATE {name} {'PASS' if ok else 'STOP'} {detail}".rstrip())
        res["gates"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            raise Stop(name)

    def diag(name, ok, detail="", counts=True):
        emit(lines, f"HONALL DIAG {name} {str(bool(ok)).upper()} {detail}".rstrip())
        res["diag"][name] = {"ok": bool(ok), "detail": detail}
        if counts and not ok:
            exitbad.append(name)

    under_out = str(out_root).startswith("/out")

    def finish(code):
        # no reload here: it could drop this container's uncommitted toy writes; commits of distinct files from other containers merge
        run_dir.mkdir(parents=True, exist_ok=True)
        rp = run_dir / f"honest_all_rerun_{stamp}.json"
        body = json.dumps(res, default=str).encode()
        with open(rp, "xb") as fh:
            fh.write(body)
        if under_out:
            out_vol.commit()
        where = f"{OUT_VOL}:{rp.relative_to('/out')}" if under_out else str(rp)
        lines.append("HONALL RESULT " + json.dumps({"exit": code, "path": where, "sha256": sha256(body)}))
        return {"lines": lines, "exit": code}

    try:
        if meta.get("toy_ref_ok") is not None:  # addendum C1/D2: the boundary control runs first, in all 7 toy modes
            nb = sum(printed_unchanged(x, p) == want for x, p, want in BOUNDARY_CASES)
            emit(lines, f"HONALL TOYBOUNDARY cases_as_predicted={nb}/{len(BOUNDARY_CASES)} {str(nb == len(BOUNDARY_CASES)).upper()}")
            res["toy_boundary_ok"] = nb == len(BOUNDARY_CASES)
            if nb != len(BOUNDARY_CASES):
                exitbad.append("toyboundary")
        # G-PIN: pinned files, the 606 routing versions, the 4 stored-weight npz files, and the record's saved_weights entries
        routing_entries = []
        bad = [n for n, h in pin_sha.items() if n not in blobs or sha256(blobs[n]) != h]
        if not bad:
            sources = json.loads(blobs["sources_v2.json"])
            routing_entries = [e for e in sources["files"] if e.get("role") == "routing"]
            for e in routing_entries:
                n = f"routing/{e['commit']}/{Path(e['path']).name}"
                if n not in blobs or sha256(blobs[n]) != e["sha256"]:
                    bad.append(n)
        npz_expect = meta.get("npz_sha") or {f"{c}::{p}": h for (c, p), h in NPZ_SHA.items()}
        for c in CELLS:
            for p in PROBLEMS:
                f = Path(w_root) / cell_file(c, p)
                if not f.is_file() or sha256(f.read_bytes()) != npz_expect[f"{c}::{p}"]:
                    bad.append(f"npz {c} {p}")
        if not bad:
            rec = json.loads(blobs["record.json"])
            for c in CELLS:
                for p in PROBLEMS:
                    sw = rec["cells"][c]["saved_weights"][p]
                    if sw.get("path") != cell_file(c, p) or sw.get("sha256") != npz_expect[f"{c}::{p}"]:
                        bad.append(f"saved_weights {c} {p}")
        res["inputs"] = {n: sha256(b) for n, b in blobs.items()}
        gate("G-PIN", not bad and len(routing_entries) == expect["routing"],
             f"pinned={len(pin_sha)} routing={len(routing_entries)} npz=4 bad={len(bad)}")
        okset = rec["settings"]["C"] == C and rec["settings"]["problems"] == [
            {"name": "simh_vs_b", "pos": ["sim"], "neg": ["B"]}, {"name": "a_vs_b", "pos": ["A"], "neg": ["B"]}]
        emit(lines, f"HONALL CHECK settings_C_and_problems {str(okset).upper()}")
        res["diag"]["settings"] = okset
        # G-CLASS
        run_keys = rec["run_keys"]
        universe = set(run_keys)
        cls = {rk: rk.split("|", 1)[0] for rk in run_keys}
        cnt = {c: sum(v == c for v in cls.values()) for c in ("sim", "B", "A")}
        gate("G-CLASS", cnt == {c: expect[c] for c in ("sim", "B", "A")} and len(run_keys) == sum(cnt.values()),
             f"sim={cnt['sim']} B={cnt['B']} A={cnt['A']} total={len(run_keys)} pred=target_informed")
        # G-SCORES
        ok, det, zero = True, [], {}
        for cell in CELLS:
            pr = rec["cells"][cell]["problems"]
            ks = {p: set(pr[p]["run_scores"]) for p in PROBLEMS}
            zero[cell] = run_keys_in(rec["zero_row_runs"][cell].get("runs", []), universe)
            c1 = (sum(cls[k] == "sim" for k in ks["simh_vs_b"]), sum(cls[k] == "B" for k in ks["simh_vs_b"]), len(ks["simh_vs_b"]))
            c2 = (sum(cls[k] == "A" for k in ks["a_vs_b"]), sum(cls[k] == "B" for k in ks["a_vs_b"]), len(ks["a_vs_b"]))
            ok &= c1 == (expect["sim_scored"], expect["B"], expect["sim_scored"] + expect["B"])
            ok &= c2 == (expect["A"], expect["B"], expect["A"] + expect["B"])
            ok &= len(zero[cell]) == expect["zero_row"] and all(cls[k] == "sim" for k in zero[cell])
            ok &= (ks["simh_vs_b"] | ks["a_vs_b"] | zero[cell]) == universe
            det.append(f"{cell}:simh={c1[0]}+{c1[1]},avb={c2[0]}+{c2[1]},zero={len(zero[cell])}")
        ok &= zero[CELLS[0]] == zero[CELLS[1]]
        gate("G-SCORES", ok, " ".join(det))
        # G-PILOT
        pilot = json.loads(blobs["pilot.json"])["keys"]
        base_of = {}
        for rk, ek in run_keys.items():
            if cls[rk] == "sim":
                base_of.setdefault(ek.split("#", 1)[0].rsplit("/", 1)[-1], []).append(rk)
        hits = sum(len(base_of.get(Path(p["volume_path"]).name, [])) == 1 for p in pilot)
        sold = sum(cls[rk] == "sim" and rec["run_set"].get(rk) == "S_old" for rk in run_keys)
        gate("G-PILOT", len(pilot) == expect["pilot"] and hits == expect["pilot"] and sold == expect["sold_sim"],
             f"pilot_keys={len(pilot)} matched={hits} sold_sim={sold} pred=target_informed")
        # G-FOLD
        folds = json.loads(blobs["folds.json"])
        fold_of = {}
        for k, mains in folds["lists"].items():
            for m in mains:
                fold_of.setdefault(m, set()).add(int(k))
        have = sum(rec["run_main"].get(rk) in fold_of and len(fold_of[rec["run_main"][rk]]) == 1 for rk in run_keys)
        same = sum(rec["run_main"].get(rk) in fold_of and fold_of[rec["run_main"][rk]] == {int(rec["run_folds"][rk])} for rk in run_keys)
        fids = sorted({int(k) for k in folds["lists"]})
        gate("G-FOLD", have == len(run_keys) and same == len(run_keys) and fids == [0, 1, 2, 3, 4],
             f"mains_in_list={have} folds_equal={same} of {len(run_keys)}")
        runs = {rk: [cls[rk], next(iter(fold_of[rec["run_main"][rk]]))] for rk in run_keys}
        # G-ROUTE
        staged = rec["skip"]["staged_sha256"]
        routings = [json.loads(blobs[f"routing/{e['commit']}/{Path(e['path']).name}"]) for e in routing_entries]
        route, nres = {}, 0
        for sk, sh in staged.items():
            opts = {(r["extraction_root"], a["meta_arm"]) for r in routings for a in r.get("arms", [])
                    if a.get("staged_inputs_file") == sk and a.get("staged_sha") == sh}
            if len(opts) == 1:
                route[sk] = next(iter(opts))
                nres += 1
        gate("G-ROUTE", len(staged) == expect["staged"] and nres == len(staged), f"staged={len(staged)} resolved={nres}")
        # G-TOKEN
        tok_ok, fixed_ids, tok_sha = meta["fixed"] if "fixed" in meta else fixed_token_ids()
        res["inputs"]["tokenizer.json"] = tok_sha
        res["config"]["fixed_token_ids"] = fixed_ids
        gate("G-TOKEN", tok_ok, f"tokenizer_sha256={tok_sha}")
        # D4: the record's auroc vs its own run_scores (definition check, G6)
        d4 = 0
        for cell in CELLS:
            for p, (pos, neg) in PROBLEMS.items():
                rs = rec["cells"][cell]["problems"][p]["run_scores"]
                a = auroc([v for k, v in rs.items() if cls[k] in pos], [v for k, v in rs.items() if cls[k] in neg])
                d4 += abs(a - rec["cells"][cell]["problems"][p]["run_level"]["auroc"]) <= TOL_DEF
        diag("D4 record_auroc_is_pooled_over_run_scores", d4 == 4, f"{d4}/4", counts=False)
        # B1a: the record's own certificates (observed_certified; per-fold observed column, "column 0 = observed labels"; uncut_gates.failures)
        fails = (rec.get("uncut_gates") or {}).get("failures")
        n_fail = len(fails) if isinstance(fails, (list, dict)) else -1
        stored_cert = {}
        for cell in CELLS:
            for p in PROBLEMS:
                per = {}
                for e in rec["cells"][cell].get("certificates", []):
                    cols = e.get("columns") if isinstance(e, dict) else None
                    if e.get("problem") == p and isinstance(cols, list) and 0 in cols:
                        j = cols.index(0)
                        per.setdefault(int(e["fold"]), []).append((bool(e["certified_per_column"][j]), float(e["max_grad_per_column"][j])))
                parsed = sorted(per) == [0, 1, 2, 3, 4] and all(len(v) == 1 for v in per.values())
                n_ok = sum(v[0][0] for v in per.values()) if parsed else -1
                n_le = sum(v[0][1] <= TOL_GRAD for v in per.values()) if parsed else -1
                oc = rec["cells"][cell]["problems"][p].get("observed_certified")
                ok_c = parsed and n_ok == 5 and n_le == 5 and oc is True
                stored_cert[f"{cell} {p}"] = {"ok": ok_c, "parsed": parsed, "observed_certified": oc, "folds_certified": n_ok,
                                              "folds_max_grad_le_1e-6": n_le, "per_fold": per}
                emit(lines, f"HONALL STOREDCERT {cell} {p} observed_certified={str(oc).upper()} parsed={str(parsed).upper()} "
                            f"folds_certified={n_ok}/5 folds_max_grad_le_1e-6={n_le}/5 pass={str(ok_c).upper()}")
        emit(lines, f"HONALL STOREDCERT uncut_gates_failures_count={n_fail}")
        res["stored_cert"] = {"by_cell_problem": stored_cert, "uncut_gates_failures_count": n_fail}
        # prep
        wanted = sorted(run_keys.values())
        ek2rk = {ek: rk for rk, ek in run_keys.items()}
        jobs = [{"idx": i, "staged_key": sk, "staged_sha": staged[sk], "extraction_root": route[sk][0], "meta_arm": route[sk][1],
                 "wanted": wanted, "fixed_ids": [fixed_ids["0"], fixed_ids["n"]], "layer": LAYER, "stamp": stamp}
                for i, sk in enumerate(sorted(staged))]
        print(f"HONALL-LIVE STAGE prep start files={len(jobs)}", flush=True)
        stage("prep")
        preps = prep_runner(jobs)
        print(f"HONALL-LIVE STAGE prep done files={len(preps)}", flush=True)
        gate("G-STAGED", sum(p["sha_ok"] for p in preps) == len(jobs), f"{sum(p['sha_ok'] for p in preps)}/{len(jobs)}")
        occ = {}
        for p in preps:
            for k in p["matched"]:
                occ[k] = occ.get(k, 0) + 1
        once = sum(occ.get(ek, 0) == 1 for ek in wanted)
        gate("G-KEY", once == len(wanted) and sum(p["dup_in_file"] for p in preps) == 0, f"{once}/{len(wanted)}")
        nfail = sum(len(p["meta_fail"]) for p in preps)
        reasons = sorted({r for p in preps for rs in p["meta_fail"].values() for r in rs})
        gate("G-META", nfail == 0, f"{len(wanted) - nfail}/{len(wanted)} fail_reasons={reasons}")
        res["prep"] = [{k: p[k] for k in ("idx", "staged_key", "n_epochs", "fixed_all", "fixed_matched", "shard", "d")} for p in preps]
        counts = {ek2rk[k]: c for p in preps for k, c in p["counts"].items()}
        # D1 / D2 / D3
        for ci, cell in enumerate(CELLS):
            rpr = rec["cells"][cell]["rows_per_run"]
            eq = sum(counts[rk][ci] == (rpr.get(rk, 0) if isinstance(rpr.get(rk, 0), (int, float)) else -1) for rk in run_keys)
            diag(f"D1 rows_per_run {cell}", eq == len(run_keys), f"{eq}/{len(run_keys)}")
            myzero = {rk for rk in run_keys if counts[rk][ci] == 0}
            diag(f"D2 zero_row_runs {cell}", myzero == zero[cell], f"mine={len(myzero)} record={len(zero[cell])}")
        dr = rec["skip"].get("dropped_rows", {})
        d3a = sum(dr.get(p["staged_key"]) == p["fixed_all"] for p in preps)
        d3b = sum(dr.get(p["staged_key"]) == p["fixed_matched"] for p in preps)
        diag("D3a dropped_rows_all_epochs", d3a == len(preps), f"{d3a}/{len(preps)} pred=target_informed", counts=False)
        diag("D3b dropped_rows_matched_epochs", d3b == len(preps), f"{d3b}/{len(preps)} pred=target_informed", counts=False)
        # fit + replay
        payload = {"shard_dir": str(run_dir / "shards"), "shard_files": sorted(p["shard"] for p in preps if p["shard"]),
                   "w_root": w_root, "my_w_dir": str(run_dir / "my_weights"), "runs": runs, "evalkey_to_run": ek2rk,
                   "record": {c: {p: {"scores": rec["cells"][c]["problems"][p]["run_scores"],
                                      "auroc": rec["cells"][c]["problems"][p]["run_level"]["auroc"]} for p in PROBLEMS} for c in CELLS},
                   "saved": {c: {p: rec["cells"][c]["saved_weights"][p] for p in PROBLEMS} for c in CELLS},
                   "npz_sha": {c: {p: meta.get("npz_sha", {}).get(f"{c}::{p}", NPZ_SHA[(c, p)]) for p in PROBLEMS} for c in CELLS}}
        stage("fit")
        fr = fit_runner(payload)
        lines.extend(fr["lines"])
        res["values"], res["fit_device"] = fr["values"], fr["device"]
        ncert = sum(v for k, v in fr["flags"].items() if k.startswith("certified"))
        diag("D5 certified_folds", ncert == 20, f"{ncert}/20")  # counts toward exit (Codex R1 finding; PRESTATED amendment A1)
        for k, v in fr["flags"].items():
            if not k.startswith("certified") and not v:
                exitbad.append(k)
        stage("printed")
        # PRINTED and C-LINK
        prov = {}
        for line in blobs["provenance.jsonl"].decode().splitlines():
            if line.strip():
                o = json.loads(line)
                prov[o["row"]] = o
        simpool = json.loads(blobs["simpool.json"])
        res["printed"] = {}
        emit(lines, f"HONALL LABEL {REFIT_LABEL}")
        for row, (cell, prob) in PRINTED_ROWS.items():
            o = prov.get(row, {})
            want_kp = (["cells", cell, "problems", prob, "run_level", "auroc"] if prob == "simh_vs_b"
                       else ["items", SIMPOOL_ITEM, "run_level_auroc", "routec", cell])
            nums = re.findall(r"\d+\.\d+", str(o.get("printed", "")))
            if len(nums) != 1 or o.get("key_path") != want_kp:
                emit(lines, f"HONALL PRINTED row={row} UNPARSEABLE")
                exitbad.append(f"printed {row}")
                continue
            pv = float(nums[0])
            mine = fr["values"][cell][prob]["refit"]["auroc"]
            recv = rec["cells"][cell]["problems"][prob]["run_level"]["auroc"]
            m_ok, r_ok = printed_unchanged(mine, nums[0]), printed_unchanged(recv, nums[0])  # exact round-half-up (addendum C1)
            near = abs(abs(mine - pv) - HALF_DP) < BOUNDARY or abs(abs(recv - pv) - HALF_DP) < BOUNDARY
            role = ROW_ROLE[prob]
            emit(lines, f"HONALL PRINTED row={row} refit_unchanged_2dp={'MATCH' if m_ok else 'MISMATCH'} record={'MATCH' if r_ok else 'MISMATCH'} "
                        f"near_boundary={str(near).upper()} role={role}")
            res["printed"][row] = {"printed": pv, "refit_match": m_ok, "record_match": r_ok, "near_boundary": near, "role": role}
            if not r_ok:  # the refit side enters FIT_VERDICT (09's reading)
                exitbad.append(f"printed_record {row}")
        link = all(simpool["items"][SIMPOOL_ITEM]["run_level_auroc"]["routec"][c] == rec["cells"][c]["problems"]["a_vs_b"]["run_level"]["auroc"]
                   for c in CELLS)
        emit(lines, f"HONALL LINK rows95-98 a_vs_b_equal={str(link).upper()}")
        res["link_rows95_98"] = link
        if not link:
            exitbad.append("link")
        stage("verdict")
        # FIT_VERDICT per cell x problem: 09's reading (decisive for the printed numbers), PRESTATED addendum A9-A12
        printed_ok = len(res["printed"]) == len(PRINTED_ROWS) and all(v["refit_match"] for v in res["printed"].values())
        res["fit_verdict"] = {}
        for tag, inf in fr["info"].items():
            fq, cert = inf["fitq"], stored_cert[tag]
            if not printed_ok and inf["strict"]:
                verdict = "UNRESOLVED"
            elif inf["strict"]:
                verdict = "PASS_STRICT"
            elif fq is None:
                verdict = "UNRESOLVED"
            elif not fq["fitter_ok"]:
                verdict = "STOP_FITTER_CHECK"
            elif not fq["resid_explained"]:
                verdict = "RECIPE_NOT_FULLY_REGISTERED"
            elif (fr["flags"].get(f"replay {tag}") and fq["stats_ok"] is True and cert["ok"]
                  and (fq["rank_identical"] or fq["max_abs_dhat"] <= DHAT_SMALL) and printed_ok):
                verdict = "PASS_RULING"
            else:
                verdict = "UNRESOLVED"
            branch = fq["branch"] if fq else "no_replay"
            emit(lines, f"HONALL FIT_VERDICT {tag} {verdict} strict={str(inf['strict']).upper()} printed_unchanged_8={str(printed_ok).upper()} "
                        f"stored_cert={str(cert['ok']).upper()} diagnosis={branch}")
            res["fit_verdict"][tag] = {"verdict": verdict, "diagnosis": branch, "strict": inf["strict"], "refit_tight": inf["refit_tight"]}
            if verdict not in ("PASS_STRICT", "PASS_RULING"):
                exitbad.append(f"fit_verdict {tag}")
        if meta.get("toy_ref_ok") is not None:  # counts toward exit (Codex R1 finding; PRESTATED amendment A2)
            tr_ok = bool(meta["toy_ref_ok"](run_dir / "my_weights"))
            emit(lines, f"HONALL TOYREF weights_within_1e-6 {str(tr_ok).upper()}")
            res["toy_ref_ok"] = tr_ok
            if not tr_ok:
                exitbad.append("toyref")
        res["exit_reasons"] = exitbad
        return finish(1 if exitbad else 0)
    except Stop as e:
        res["stopped_at"] = str(e)
        return finish(1)
    except Exception as e:  # sanitized (addendum C3): no raw traceback reaches the log
        own = isinstance(e, RuntimeError) and str(e).startswith(("prep ", "fit failed"))
        crash = {"type": type(e).__name__, "stage": _STAGE["s"], "remote": str(e) if own else None}
    res["crashed"] = crash
    emit(lines, f"HONALL CRASH stage={crash['stage']} type={crash['type']}" + (f" remote=[{crash['remote']}]" if crash["remote"] else ""))
    return finish(1)


def crash_result(err) -> dict:
    return {"lines": [f"HONALL CRASH stage={err[1]} type={err[0]}", "HONALL RESULT " + json.dumps({"exit": 1, "path": None, "sha256": None})],
            "exit": 1}


@app.function(image=image_light, cpu=(1, 1), memory=(2048, 2048), timeout=3300, startup_timeout=120, scaledown_window=2,
              retries=0,  # addendum E: request = limit; timeout covers the worst-case wall time
              volumes={"/out": out_vol, "/w": w_vol.read_only()})
def orchestrate(mode: str, blobs: dict, meta: dict) -> dict:
    """Real run: gates, prep.map, fit.remote, comparisons (addendum C2: timeout 2700 s, ~1.7x the expected ~25 min wall time)."""
    if mode != "real":
        raise SystemExit("orchestrate runs the real mode only")
    err = None
    try:
        return pipeline(mode, blobs, meta, Path("/acts"), "/w", REAL_EXPECT, PIN_SHA,
                        lambda jobs: list(prep.map(jobs)), lambda payload: fit.remote(payload), Path("/out"))
    except Exception as e:  # sanitized (addendum C3); pipeline() already catches its own errors
        err = (type(e).__name__, _STAGE["s"])
    return crash_result(err)


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=600, startup_timeout=180, scaledown_window=2,
              retries=0,  # addendum E: request = limit
              volumes={"/out": out_vol})
def orchestrate_toy(mode: str, meta: dict) -> dict:
    """Toy controls on CPU (addendum C2: timeout 600 s each)."""
    err = None
    try:
        return toy(mode, meta)
    except Exception as e:  # sanitized (addendum C3)
        err = (type(e).__name__, _STAGE["s"])
    return crash_result(err)


# ----------------------------------------------------------------------------------------------------- toy controls

def toy(mode: str, meta: dict) -> dict:
    """PRESTATED section 7: a toy problem in the real layout; scikit-learn builds the toy 'record' and npz."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    tok_ok, fixed_ids, tok_sha = fixed_token_ids()
    rng = np.random.default_rng(20261020)
    base = Path("/tmp/toy")
    acts, wroot = base / "acts", base / "w"
    d, mains = 16, [f"main{i}" for i in range(10)]
    fold_of = {m: i % 5 for i, m in enumerate(mains)}
    arms = {"sim": "sim_honest", "B": "honest_b", "A": "honest_a"}
    sset, eroot = "toy_cutfull_v1", "toy_cutfull_extract_v1"
    fixed = [fixed_ids["0"], fixed_ids["n"]]
    content = np.array([i for i in range(1000, 3000) if i not in fixed])
    shift = {"sim": (0.6, 0.4), "B": (0.0, 0.0), "A": (0.15, 0.1)}
    runs, eps = {}, {a: [] for a in arms}
    for cls, n in (("sim", 13), ("B", 12), ("A", 12)):
        for j in range(n + 1):  # the last one per class is an extra epoch outside the pool
            main = mains[(j + {"sim": 0, "B": 3, "A": 6}[cls]) % 10]
            ek = f"toy_{cls}/{arms[cls]}@x/t{j:02d}_{main}/f{cls}{j:02d}.eval#e0"
            rk = f"{cls}|t{j:02d}_{main}|e1|{hashlib.sha1(ek.encode()).hexdigest()[:8]}"
            ids = rng.choice(content, 400).tolist()
            calls, nc = [], int(rng.integers(3, 7))
            for t in range(nc):
                p0 = 40 + t * 60
                cut = cls == "sim" and j == 0
                ce = None if cut or (t > 0 and rng.random() < 0.1) else p0 + 10
                if ce is not None and t > 0 and rng.random() < 0.4:  # call 0 always keeps a content call_end row
                    ids[ce] = int(fixed[int(rng.integers(0, 2))])
                calls += [{"irid": f"r{t}", "kind": "command", "readout": None if cut else p0 + 5, "src_irid": f"r{t}", "call_index": 0},
                          {"irid": f"r{t}", "kind": "call_end", "readout": ce, "src_irid": f"r{t}", "call_index": 0},
                          {"irid": f"r{t + 1}", "kind": "output_end", "readout": None if cut else p0 + 30,
                           "span": None if cut else [p0 + 20, p0 + 31], "src_irid": f"r{t}", "call_index": 0}]
            eps[cls].append({"key": ek, "arm": arms[cls], "ids": ids, "calls": calls})
            if j < n:
                runs[rk] = {"cls": cls, "ek": ek, "main": main, "set": "S_old" if j < 5 else "S_new"}
    # rows per epoch, as the extractor lays them out
    rows = {}
    for cls, lst in eps.items():
        for ep in lst:
            kept = [c for c in ep["calls"] if c["readout"] is not None]
            means = [c for c in kept if c["kind"] == "output_end" and c.get("span")]
            X = rng.normal(size=(len(kept) + len(means), d))
            for i, c in enumerate(kept):
                if c["kind"] == "call_end":
                    X[i, :2] += shift[cls][0]
            for i in range(len(means)):
                X[len(kept) + i, 2:4] += shift[cls][1]
            rows[ep["key"]] = X.astype(np.float16)
            ri, mi = iter(range(len(kept))), iter(range(len(kept), len(kept) + len(means)))
            mid = {id(c) for c in means}
            ep["meta_calls"] = [{k: v for k, v in c.items()} | {"row": next(ri) if c["readout"] is not None else None}
                                | ({"row_mean": next(mi)} if id(c) in mid else {}) for c in ep["calls"]]
            ep["fixed"] = [c["kind"] == "call_end" and c["readout"] is not None and ep["ids"][c["readout"]] in fixed for c in ep["calls"]]
    if mode == "test-fail-token":  # move one fixed call_end readout to a content token AFTER the reference flags are taken
        for ep in eps["B"]:
            hit = [c for c, fx in zip(ep["calls"], ep["fixed"]) if fx]
            if hit:
                ep["ids"][hit[0]["readout"]] = int(content[0])
                break
    # staged files, chunks, metas
    staged_sha = {}
    for cls, lst in eps.items():
        p = acts / sset / "inputs" / f"{arms[cls]}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"".join(json.dumps({k: ep[k] for k in ("key", "arm", "ids", "calls")}).encode() + b"\n" for ep in lst))
        staged_sha[f"{sset}/inputs/{arms[cls]}.jsonl"] = sha256(p.read_bytes())
    allep = [ep for lst in eps.values() for ep in lst]
    for ci in range(0, len(allep), 4):
        grp, name = allep[ci:ci + 4], f"shard000_a1_chunk{ci // 4:03d}"
        start, ent = 0, []
        for ep in grp:
            ep["chunk"], ep["row_start"] = name, start
            ent.append({"arm": ep["arm"], "key": ep["key"], "row_start": start, "n_rows": len(rows[ep["key"]])})
            start += len(rows[ep["key"]])
        (acts / eroot / "by_layer" / "L48").mkdir(parents=True, exist_ok=True)
        (acts / eroot / "chunks").mkdir(parents=True, exist_ok=True)
        np.save(acts / eroot / "by_layer" / "L48" / f"{name}.npy", np.concatenate([rows[ep["key"]] for ep in grp]))
        (acts / eroot / "chunks" / f"{name}.json").write_text(json.dumps({"layers": [48], "rows": start, "epochs": ent}))
        for ep in grp:
            sk = f"{sset}/inputs/{ep['arm']}.jsonl"
            mdir = acts / eroot / "meta" / ep["arm"]
            mdir.mkdir(parents=True, exist_ok=True)
            (mdir / f"{meta_name(ep['key'])}.json").write_text(json.dumps(
                {"key": ep["key"], "arm": ep["arm"], "n_rows": len(rows[ep["key"]]), "calls": ep["meta_calls"], "chunk": ep["chunk"],
                 "row_start": ep["row_start"], "inputs_sha256": staged_sha[sk]}))
    # scikit-learn reference = the toy "record"
    by_ek = {ep["key"]: ep for ep in allep}
    rk_list = sorted(runs)
    rec = {"run_keys": {rk: runs[rk]["ek"] for rk in rk_list}, "run_main": {rk: runs[rk]["main"] for rk in rk_list},
           "run_folds": {rk: fold_of[runs[rk]["main"]] for rk in rk_list}, "run_set": {rk: runs[rk]["set"] for rk in rk_list},
           "settings": {"C": 0.01, "problems": [{"name": "simh_vs_b", "pos": ["sim"], "neg": ["B"]}, {"name": "a_vs_b", "pos": ["A"], "neg": ["B"]}]},
           "skip": {"staged_sha256": dict(staged_sha), "dropped_rows": {}}, "zero_row_runs": {}, "cells": {}}
    for sk in staged_sha:
        arm = Path(sk).stem
        rec["skip"]["dropped_rows"][sk] = sum(sum(by_ek[runs[rk]["ek"]]["fixed"]) for rk in rk_list if by_ek[runs[rk]["ek"]]["arm"] == arm)
    ref_w = {}
    for cell in CELLS:
        rr = {}
        for rk in rk_list:
            ep = by_ek[runs[rk]["ek"]]
            if cell == "command|L48":
                idx = [mc["row"] for mc, fx in zip(ep["meta_calls"], ep["fixed"]) if mc["kind"] == "call_end" and mc["row"] is not None and not fx]
            else:
                idx = [mc["row_mean"] for mc in ep["meta_calls"] if "row_mean" in mc]
            rr[rk] = rows[ep["key"]][idx].astype(np.float64)
        zero = [rk for rk in rk_list if len(rr[rk]) == 0]
        rec["zero_row_runs"][cell] = {"n": len(zero), "runs": zero}
        cellrec = {"rows_per_run": {rk: len(rr[rk]) for rk in rk_list if len(rr[rk])}, "problems": {}, "saved_weights": {}, "certificates": []}
        for prob, (pos, neg) in PROBLEMS.items():
            members = [rk for rk in rk_list if runs[rk]["cls"] in pos + neg and len(rr[rk])]
            scores, W, B, M, S, Wref = {}, [], [], [], [], []
            for k in range(5):
                tr = [rk for rk in members if fold_of[runs[rk]["main"]] != k]
                te = [rk for rk in members if fold_of[runs[rk]["main"]] == k]
                Xtr = np.concatenate([rr[rk] for rk in tr])
                ytr = np.concatenate([[float(runs[rk]["cls"] in pos)] * len(rr[rk]) for rk in tr])
                sc = StandardScaler().fit(Xtr)
                lr = LogisticRegression(C=0.01, solver="newton-cholesky", tol=1e-12, max_iter=1000).fit(sc.transform(Xtr), ytr)
                Wref.append(lr.coef_[0])
                ws, bs = lr.coef_[0].copy(), float(lr.intercept_[0])
                if mode == "test-fail-defgap":  # the toy "producer" uses another C: a definition gap (G4)
                    lr2 = LogisticRegression(C=10.0, solver="newton-cholesky", tol=1e-12, max_iter=1000).fit(sc.transform(Xtr), ytr)
                    ws, bs = lr2.coef_[0].copy(), float(lr2.intercept_[0])
                if mode == "test-ruling":  # the toy "producer" stopped short of the optimum: same objective, unconverged
                    ws = ws + 3e-5 * rng.standard_normal(d)
                    bs = bs + 3e-5 * float(rng.standard_normal())
                for rk in te:
                    scores[rk] = float((sc.transform(rr[rk]) @ ws + bs).mean())
                W.append(ws)
                B.append(bs)
                M.append(sc.mean_)
                S.append(sc.scale_)
                cellrec["certificates"].append({"problem": prob, "fold": k, "columns": [0], "certified_per_column": [True],
                                                "max_grad_per_column": [5e-7]})
            a = float(roc_auc_score([float(runs[rk]["cls"] in pos) for rk in scores], [scores[rk] for rk in scores]))
            wp = wroot / cell_file(cell, prob)
            wp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(wp, w=np.stack(W), b=np.array(B), mean=np.stack(M), std=np.stack(S), folds=np.arange(5))
            ref_w[(cell, prob)] = np.stack(Wref)
            cellrec["problems"][prob] = {"run_scores": scores, "run_level": {"auroc": a}, "observed_certified": True}
            cellrec["saved_weights"][prob] = {"path": cell_file(cell, prob), "sha256": sha256(wp.read_bytes()), "folds": [0, 1, 2, 3, 4]}
        rec["cells"][cell] = cellrec
    rec["uncut_gates"] = {"failures": []}
    prov = [{"row": r, "key_path": (["cells", c, "problems", p, "run_level", "auroc"] if p == "simh_vs_b" else
                                    ["items", SIMPOOL_ITEM, "run_level_auroc", "routec", c]),
             "printed": half_up_2dp(rec["cells"][c]["problems"][p]["run_level"]["auroc"])} for r, (c, p) in PRINTED_ROWS.items()]
    simpool = {"items": {SIMPOOL_ITEM: {"run_level_auroc": {"routec": {c: rec["cells"][c]["problems"]["a_vs_b"]["run_level"]["auroc"] for c in CELLS}}}}}
    pilot = {"keys": [{"volume_path": "simh_evals_v1/" + runs[rk]["ek"].split("#")[0]} for rk in rk_list if runs[rk]["cls"] == "sim"][5:8]}
    folds = {"lists": {str(k): [m for m in mains if fold_of[m] == k] for k in range(5)}}
    if mode == "test-fail-value":
        pr = rec["cells"]["command|L48"]["problems"]["simh_vs_b"]
        pr["run_level"]["auroc"] += 0.01
        k0 = sorted(pr["run_scores"])[0]
        pr["run_scores"][k0] += 0.05
    if mode == "test-fail-pilot":
        pilot["keys"] = pilot["keys"][1:]
    if mode == "test-fail-sha":
        sk = f"{sset}/inputs/sim_honest.jsonl"
        rec["skip"]["staged_sha256"][sk] = "0" * 64
        staged_sha[sk] = "0" * 64
    routing = {"extraction_root": eroot, "arms": [{"meta_arm": arms[c], "staged_inputs_file": f"{sset}/inputs/{arms[c]}.jsonl",
                                                    "staged_sha": staged_sha[f"{sset}/inputs/{arms[c]}.jsonl"]} for c in arms]}
    rblob = json.dumps(routing).encode()
    blobs = {"record.json": json.dumps(rec).encode(), "simpool.json": json.dumps(simpool).encode(),
             "provenance.jsonl": "".join(json.dumps(o) + "\n" for o in prov).encode(), "release.json": b"{}",
             "folds.json": json.dumps(folds).encode(), "pilot.json": json.dumps(pilot).encode(), "routing/toyc/toy.json": rblob}
    blobs["sources_v2.json"] = json.dumps({"files": [{"role": "routing", "commit": "toyc", "path": "x/routing/toy.json", "sha256": sha256(rblob)}]}).encode()
    pin = {n: sha256(b) for n, b in blobs.items() if not n.startswith("routing/")}
    expect = {"sim": 13, "B": 12, "A": 12, "sim_scored": 12, "zero_row": 1, "pilot": 3, "sold_sim": 5, "staged": 3, "routing": 1}

    def ref_ok(my_dir):
        return all(float(np.abs(np.load(my_dir / f"my_w_{c.replace('|', '_')}__{p}.npz")["w"] - ref_w[(c, p)]).max()) <= 1e-6
                   for c in CELLS for p in PROBLEMS)

    meta = meta | {"fixed": (tok_ok, fixed_ids, tok_sha), "toy_ref_ok": ref_ok,
                   "npz_sha": {f"{c}::{p}": rec["cells"][c]["saved_weights"][p]["sha256"] for c in CELLS for p in PROBLEMS},
                   "stamp": f"{meta['stamp']}_{mode}"}
    out_root = Path("/out") / "toy"  # toy shards, weights and record live on the output Volume under toy/<stamp>_<mode>/
    # persist the toy producer artifacts and synthetic inputs beside the toy outputs (Codex r4): pinned blobs, producer npz, staged/meta/chunks
    import shutil
    keep = out_root / meta["stamp"] / "toy_inputs"
    if keep.exists():
        raise FileExistsError("toy inputs exist (results are append-only); use a fresh stamp")
    for n, b in blobs.items():
        (keep / "pinned" / n).parent.mkdir(parents=True, exist_ok=True)
        (keep / "pinned" / n).write_bytes(b)
    shutil.copytree(wroot, keep / "producer_w")
    shutil.copytree(acts, keep / "acts")
    res = pipeline("test" if mode in REPLAY_TOYS else mode, blobs, meta, acts, str(wroot), expect, pin,
                   lambda jobs: [prep_one(acts, out_root / meta["stamp"] / "shards", j) for j in jobs],
                   lambda payload: fit_and_compare(payload, "cpu"), out_root)
    if mode not in REPLAY_TOYS:
        return res
    fp = first_pass_gate(res)  # addendum G4: a failing first pass (e.g. TOYBOUNDARY, TOYREF) fails the replay toy; no replay runs
    if fp is not None:
        return fp
    # ---- addendum F5 toys: the producer npz rewritten in the per-fold layout, then the replay-only pipeline on this toy's stored outputs
    first = [ln for ln in res["lines"] if ln.startswith("HONALL RESULT ")]
    wpf = base / "w_perfold"
    for c in CELLS:
        for p in PROBLEMS:
            z = np.load(wroot / cell_file(c, p))
            arrs = {}
            for k in range(5):
                if mode == "test-replay-fail-fold" and k == 4 and (c, p) == (CELLS[0], "simh_vs_b"):
                    continue  # one npz lacks fold 4: G-LAYOUT must STOP
                wk = z["w"][k] * (1.5 if mode == "test-replay-fail-producer" else 1.0)  # addendum G2: producer weights changed, my_weights not
                arrs |= {f"f{k}__W": wk, f"f{k}__b": np.array([z["b"][k]]), f"f{k}__mean": z["mean"][k], f"f{k}__std": z["std"][k]}
            dst = wpf / cell_file(c, p)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, **arrs)
    src_stamp = meta["stamp"]
    result_rel = f"{src_stamp}/honest_all_rerun_{src_stamp}.json"
    rec2 = json.loads(blobs["record.json"])
    npz2 = {}
    for c in CELLS:
        for p in PROBLEMS:
            h = sha256((wpf / cell_file(c, p)).read_bytes())
            rec2["cells"][c]["saved_weights"][p]["sha256"] = h
            npz2[f"{c}::{p}"] = h
    blobs2 = dict(blobs) | {"record.json": json.dumps(rec2).encode()}
    m_items = manifest_of(out_root, wpf, src_stamp, result_rel)
    if mode == "test-replay-fail-manifest":  # addendum H: one stray file listed; canonical and sha-consistent, so only coverage can fail
        m_items.append({"set": "my_weights", "path": f"{src_stamp}/my_weights/stray.npz", "size": 1, "sha256": "0" * 64})
    mb = manifest_bytes(m_items)
    blobs2["replay_manifest.json"] = mb
    keep2 = out_root / src_stamp / "toy_replay_inputs"  # addendum G6: persist the rewritten inputs (outside the manifest's sets)
    if keep2.exists():
        raise FileExistsError("toy replay inputs exist (results are append-only); use a fresh stamp")
    shutil.copytree(wpf, keep2 / "producer_w_perfold")
    (keep2 / "record.json").write_bytes(blobs2["record.json"])
    (keep2 / "replay_manifest.json").write_bytes(mb)
    pin2 = {n: sha256(b) for n, b in blobs2.items() if not n.startswith("routing/") and n != "replay_manifest.json"}
    meta2 = {k: v for k, v in meta.items() if k not in ("toy_ref_ok", "fixed")} | {
        "npz_sha": npz2, "manifest_sha": sha256(mb), "stamp": f"{src_stamp}_r", "toy_first_pass": first}
    out = replay_pipeline("toy-" + mode, blobs2, meta2, str(wpf), out_root, src_stamp, result_rel,
                          sha256((out_root / result_rel).read_bytes()), expect, pin2,
                          lambda payload: replay_and_compare(payload, "cpu"), out_root)
    out["lines"] = [f"HONALL TOY_FIRST_PASS {ln[len('HONALL RESULT '):]}" for ln in first] + out["lines"]
    return out


# ------------------------------------------------------------- addendum F: the hash-shards pin and the replay-only run (09's route A)

REAL_STAMP = "2026-10-20_152036"
REAL_RESULT = f"{REAL_STAMP}/honest_all_rerun_{REAL_STAMP}.json"
REAL_RESULT_SHA = "57d32200b5f3db4dbe492580c59a6b4708e75547a5b922eeaf0478e42bcaa64b"
PERFOLD_SUFFIXES = ("W", "b", "mean", "std")
REPLAY_TOYS = ("test-replay", "test-replay-fail-fold", "test-replay-fail-producer", "test-replay-fail-manifest")  # G2, H


def file_sha256(p: Path) -> tuple:
    """(size, sha256) of a file, streamed."""
    h, n = hashlib.sha256(), 0
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 23), b""):
            h.update(blk)
            n += len(blk)
    return n, h.hexdigest()


def manifest_of(out_root: Path, w_root: Path, stamp: str, result_rel: str) -> list:
    """F3: {set, path, size, sha256} of every file the replay reads: the run's shards and my_weights, its results record, the 4 producer npz."""
    items = []
    for sset in ("shards", "my_weights"):
        for p in sorted((out_root / stamp / sset).iterdir()):
            if p.is_file():
                n, h = file_sha256(p)
                items.append({"set": sset, "path": f"{stamp}/{sset}/{p.name}", "size": n, "sha256": h})
    n, h = file_sha256(out_root / result_rel)
    items.append({"set": "result", "path": result_rel, "size": n, "sha256": h})
    for c in CELLS:
        for pr in PROBLEMS:
            n, h = file_sha256(w_root / cell_file(c, pr))
            items.append({"set": "producer_npz", "path": cell_file(c, pr), "size": n, "sha256": h})
    return items


def manifest_bytes(items: list) -> bytes:
    """Canonical serialization; the manifest pin is sha256 of these bytes."""
    return (json.dumps(items, sort_keys=True, separators=(",", ":")) + "\n").encode()


@app.function(image=image_light, cpu=(2, 2), memory=(8192, 8192), timeout=600, startup_timeout=120, scaledown_window=2,
              retries=0,  # addendum F6; reads only (read-only mounts)
              volumes={"/out": out_vol.read_only(), "/w": w_vol.read_only()})
def hash_shards() -> dict:
    """F3 phase 1: the manifest of the real run's stored outputs and the producer npz. Prints paths, sizes and sha256 only."""
    err = None
    try:
        stage("hash:manifest")
        items = manifest_of(Path("/out"), Path("/w"), REAL_STAMP, REAL_RESULT)
        stage("hash:record")
        myrec = json.loads((Path("/out") / REAL_RESULT).read_bytes())
        shard_ids = sorted(p["shard"] for p in myrec["prep"] if p.get("shard"))
    except Exception as e:  # sanitized (addendum C3)
        err = f"{type(e).__name__} at {_STAGE['s']}"
    if err is not None:
        raise RuntimeError(f"hash failed: {err}")
    by = {s: [i for i in items if i["set"] == s] for s in ("shards", "my_weights", "result", "producer_npz")}
    want_npz = {cell_file(c, p): NPZ_SHA[(c, p)] for c in CELLS for p in PROBLEMS}
    want_myw = {f"my_w_{c.replace('|', '_')}__{p}.npz" for c in CELLS for p in PROBLEMS}
    want_sh = {f"{s}.npz" for s in shard_ids} | {f"{s}.json" for s in shard_ids}
    checks = [("G-HASH-RESULT", len(by["result"]) == 1 and by["result"][0]["sha256"] == REAL_RESULT_SHA, ""),
              ("G-HASH-NPZ", len(by["producer_npz"]) == 4 and all(want_npz.get(i["path"]) == i["sha256"] for i in by["producer_npz"]), ""),
              ("G-HASH-SHARDS", {Path(i["path"]).name for i in by["shards"]} == want_sh, f"files={len(by['shards'])} expected={len(want_sh)}"),
              ("G-HASH-MYW", {Path(i["path"]).name for i in by["my_weights"]} == want_myw, f"files={len(by['my_weights'])}")]
    lines = [f"HONALL HASH {i['set']} {i['path']} size={i['size']} sha256={i['sha256']}" for i in items]
    lines += [f"HONALL GATE {n} {'PASS' if ok else 'STOP'} {d}".rstrip() for n, ok, d in checks]
    mb = manifest_bytes(items)
    ok = all(ok for _, ok, _ in checks)
    lines.append("HONALL RESULT " + json.dumps({"exit": 0 if ok else 1, "manifest_sha256": sha256(mb), "files": len(items)}))
    return {"lines": lines, "exit": 0 if ok else 1, "manifest": mb.decode()}


def parse_per_fold(z, d: int):
    """F1: exactly the 20 keys f{k}__{W,b,mean,std}, k = 0..4; W, mean, std of shape (d,), b of shape (1,) or (); finite; std > 0.
    Stacked in k order to the step-6 'stored' form (w (5, d), b (5,), mean (5, d), std (5, d), folds [0..4]). Returns (layout, reason)."""
    import numpy as np
    if set(z.files) != {f"f{k}__{s}" for k in range(5) for s in PERFOLD_SUFFIXES}:
        return None, "keys"
    W, B, M, S = [], [], [], []
    for k in range(5):
        try:
            raw = {s_: np.asarray(z[f"f{k}__{s_}"]) for s_ in PERFOLD_SUFFIXES}
        except ValueError:  # object arrays need pickle; never loaded
            return None, "object"
        if any(a.dtype.kind != "f" for a in raw.values()):  # addendum G5: integer, boolean or complex arrays STOP before any cast
            return None, "dtype"
        w, b = raw["W"].astype(np.float64), raw["b"].astype(np.float64)
        m, s = raw["mean"].astype(np.float64), raw["std"].astype(np.float64)
        if w.shape != (d,) or m.shape != (d,) or s.shape != (d,) or b.shape not in ((1,), ()):
            return None, "shapes"
        if not (np.isfinite(w).all() and np.isfinite(b).all() and np.isfinite(m).all() and np.isfinite(s).all()):
            return None, "nonfinite"
        if not (s > 0).all():
            return None, "std_not_positive"
        W.append(w)
        B.append(float(b.reshape(-1)[0]))
        M.append(m)
        S.append(s)
    return ({"w": np.stack(W), "b": np.array(B), "mean": np.stack(M), "std": np.stack(S), "folds": [0, 1, 2, 3, 4], "mode": "stored",
             "keys": {"layout": "per-fold f{k}__{W,b,mean,std}, k = 0..4"}}, "ok")


def folds_exact(f) -> bool:
    """Addendum G3: the recorded saved_weights folds list itself: exactly the Python ints [0, 1, 2, 3, 4] (no bool, float or coercion)."""
    return isinstance(f, list) and len(f) == 5 and all(type(x) is int for x in f) and f == [0, 1, 2, 3, 4]


def manifest_coverage(items: list, src_stamp: str, result_rel: str, npz_expect: dict, result_sha: str):
    """Addendum G1: exact coverage by name. Returns (ok, shard stems, detail)."""
    sets = {s: sum(i["set"] == s for i in items) for s in ("shards", "my_weights", "result", "producer_npz")}
    paths_of = {s: [i["path"] for i in items if i["set"] == s] for s in sets}
    all_paths = [i["path"] for i in items]
    myw_expect = {f"{src_stamp}/my_weights/my_w_{c.replace('|', '_')}__{p}.npz" for c in CELLS for p in PROBLEMS}
    stems = sorted({Path(q).stem for q in paths_of["shards"]})
    shard_expect = {f"{src_stamp}/shards/{s}{ext}" for s in stems for ext in (".npz", ".json")}
    ok = (len(all_paths) == len(set(all_paths)) and len(items) == sum(sets.values())
          and set(paths_of["my_weights"]) == myw_expect and paths_of["result"] == [result_rel]
          and set(paths_of["producer_npz"]) == {cell_file(c, p) for c in CELLS for p in PROBLEMS}
          and len(stems) > 0 and set(paths_of["shards"]) == shard_expect and sets["shards"] == 2 * len(stems)
          and all(i["sha256"] == npz_expect[f"{c}::{p}"] for i in items if i["set"] == "producer_npz"
                  for c in CELLS for p in PROBLEMS if i["path"] == cell_file(c, p))
          and all(i["sha256"] == result_sha for i in items if i["set"] == "result"))
    detail = (f"files={len(items)} unique={len(set(all_paths))} shard_stems={len(stems)} my_weights={sets['my_weights']} "
              f"producer_npz={sets['producer_npz']}")
    return ok, stems, detail


def stems_match(stems: list, prep_ids: list) -> bool:
    """Addendum G1: the manifest's shard stems equal the record's prep shard ids (unique), as G-HASH-SHARDS."""
    return sorted(stems) == sorted(prep_ids) and len(set(prep_ids)) == len(prep_ids)


def first_pass_gate(res: dict):
    """Addendum G4: a replay toy whose first pass exited non-zero STOPs here (no replay). Returns the STOP result, or None to continue."""
    if res["exit"] != 0:
        res["lines"].insert(0, "HONALL GATE G-TOY-FIRST-PASS STOP")
        return res
    return None


def unit_negatives() -> list:
    """Addendum H (09's N3 ruling): in-container unit negatives of the pure gate functions, on fixtures; each must STOP or return false
    as predicted (positive sanity cases included). Returns [(name, as_predicted)]."""
    import io
    import numpy as np
    st, rr = "S", "S/honest_all_rerun_S.json"
    good = ([{"set": "shards", "path": f"{st}/shards/000{e}", "size": 1, "sha256": "x"} for e in (".npz", ".json")]
            + [{"set": "my_weights", "path": f"{st}/my_weights/my_w_{c.replace('|', '_')}__{p}.npz", "size": 1, "sha256": "x"}
               for c in CELLS for p in PROBLEMS]
            + [{"set": "result", "path": rr, "size": 1, "sha256": "r"}]
            + [{"set": "producer_npz", "path": cell_file(c, p), "size": 1, "sha256": f"n{c}{p}"} for c in CELLS for p in PROBLEMS])
    npz_exp = {f"{c}::{p}": f"n{c}{p}" for c in CELLS for p in PROBLEMS}
    one_ext = [i for i in good if i["path"] != f"{st}/shards/000.json"]
    stray = good + [{"set": "my_weights", "path": f"{st}/my_weights/stray.npz", "size": 1, "sha256": "x"}]
    # addendum H2: two fixtures that v9's count checks would pass and only G1's name checks catch
    first_myw = f"{st}/my_weights/my_w_{CELLS[0].replace('|', '_')}__simh_vs_b.npz"
    renamed = [(i | {"path": f"{st}/my_weights/stray.npz"}) if i["path"] == first_myw else i for i in good]  # still 4 my_weights entries
    two_stems = [i for i in good if i["set"] != "shards"] + [{"set": "shards", "path": f"{st}/shards/000.npz", "size": 1, "sha256": "x"},
                                                             {"set": "shards", "path": f"{st}/shards/001.json", "size": 1, "sha256": "x"}]

    def npz_of(arrs):
        buf = io.BytesIO()
        np.savez(buf, **arrs)
        buf.seek(0)
        return np.load(buf)

    d = 3
    base = {f"f{k}__{s}": (np.ones(1) if s == "b" else np.ones(d)) for k in range(5) for s in PERFOLD_SUFFIXES}
    ints = dict(base) | {"f2__W": np.ones(d, dtype=np.int64)}
    objs = dict(base) | {"f1__mean": np.array([1.0, "x", 2.0], dtype=object)}
    fp_fail = first_pass_gate({"exit": 1, "lines": ["HONALL RESULT {}"]})
    return [("coverage_exact_passes", manifest_coverage(good, st, rr, npz_exp, "r")[0] is True),
            ("coverage_stem_one_extension_fails", manifest_coverage(one_ext, st, rr, npz_exp, "r")[0] is False),
            ("coverage_stray_file_fails", manifest_coverage(stray, st, rr, npz_exp, "r")[0] is False),
            ("coverage_myw_stray_name_same_count_fails", manifest_coverage(renamed, st, rr, npz_exp, "r")[0] is False),
            ("coverage_two_one_ext_stems_even_count_fails", manifest_coverage(two_stems, st, rr, npz_exp, "r")[0] is False),
            ("stems_equal_passes", stems_match(["000"], ["000"]) is True),
            ("stem_not_in_record_fails", stems_match(["000", "001"], ["000"]) is False),
            ("folds_0to4_passes", folds_exact([0, 1, 2, 3, 4]) is True),
            ("folds_0to3_fails", folds_exact([0, 1, 2, 3]) is False),
            ("folds_0to5_fails", folds_exact([0, 1, 2, 3, 4, 5]) is False),
            ("folds_float_fails", folds_exact([0.0, 1.0, 2.0, 3.0, 4.0]) is False),
            ("layout_float_passes", parse_per_fold(npz_of(base), d)[0] is not None),
            ("layout_int_dtype_stops", parse_per_fold(npz_of(ints), d) == (None, "dtype")),
            ("layout_object_dtype_stops", parse_per_fold(npz_of(objs), d)[0] is None),
            ("first_pass_fail_stops", fp_fail is not None and fp_fail["exit"] == 1 and fp_fail["lines"][0] == "HONALL GATE G-TOY-FIRST-PASS STOP"),
            ("first_pass_ok_continues", first_pass_gate({"exit": 0, "lines": []}) is None)]


def replay_and_compare(payload: dict, device: str) -> dict:
    """F2: REPLAY and N1 score the shard rows with the PRODUCER's per-fold npz only (B3 tiers vs the record's run_scores). my_weights (my
    refit) enter only Delta-hat, RESID_EXPLAINED, rank_identical and the stats / J-gap / gradient diagnostics (fit_quality). No refit."""
    import numpy as np
    import torch

    out_root, w_root = Path(payload["out_root"]), Path(payload["w_root"])
    gates, lines, vals, flags, info = [], [], {}, {}, {}

    def ret():
        gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        return {"gates": gates, "lines": lines, "values": vals, "flags": flags, "info": info, "device": gpu}

    stage("replay:manifest")  # G-MANIFEST: every pinned file's size and sha256 (F3)
    bad = sum(file_sha256((w_root if it["set"] == "producer_npz" else out_root) / it["path"]) != (it["size"], it["sha256"])
              for it in payload["manifest"])
    gates.append(("G-MANIFEST", bad == 0, f"files={len(payload['manifest'])} mismatched={bad}"))
    if bad:
        return ret()
    stage("replay:result")
    raw = (out_root / payload["result_rel"]).read_bytes()
    gates.append(("G-RESULT", sha256(raw) == payload["result_sha"], ""))
    if sha256(raw) != payload["result_sha"]:
        return ret()
    myrec = json.loads(raw)
    prep_ids = sorted(p["shard"] for p in myrec["prep"] if p.get("shard"))
    stems_ok = stems_match(payload["shard_files"], prep_ids)  # addendum G1, as G-HASH-SHARDS (unit-tested in unit_negatives)
    gates.append(("G-STEMS", stems_ok, f"stems={len(payload['shard_files'])} prep_shards={len(prep_ids)}"))
    if not stems_ok:
        return ret()
    ds = {p.get("d") for p in myrec["prep"] if p.get("shard")}
    gates.append(("G-WIDTH", len(ds) == 1 and None not in ds, f"widths={len(ds)}"))
    if len(ds) != 1 or None in ds:
        return ret()
    d = int(next(iter(ds)))
    stage("replay:layout")  # G-LAYOUT on all 4 producer npz BEFORE any scoring (09 condition 3)
    lays, why = {}, []
    for c in CELLS:
        for p in PROBLEMS:
            lay, r = parse_per_fold(np.load(w_root / cell_file(c, p)), d)
            if lay is None:
                why.append(f"{c} {p}:{r}")
            else:
                lays[(c, p)] = lay
    folds_ok = all(folds_exact(payload["saved"][c][p]["folds"]) for c in CELLS for p in PROBLEMS)
    gates.append(("G-LAYOUT", not why and folds_ok, f"parsed={len(lays)}/4 record_folds_0to4={str(folds_ok).upper()} reasons={why}"))
    if why or not folds_ok:
        return ret()
    stage("replay:my_weights")  # my refit (F2): only for Delta-hat / RESID / rank / stats
    myw, bad_myw = {}, []
    for c in CELLS:
        for p in PROBLEMS:
            z = np.load(out_root / payload["src_stamp"] / "my_weights" / f"my_w_{c.replace('|', '_')}__{p}.npz")
            ok = (sorted(z.files) == ["b", "folds", "mean", "std", "w"] and z["w"].shape == (5, d) and z["b"].shape == (5,)
                  and z["mean"].shape == (5, d) and z["std"].shape == (5, d) and [int(x) for x in z["folds"]] == [0, 1, 2, 3, 4])
            if not ok:
                bad_myw.append(f"{c} {p}")
            else:
                myw[(c, p)] = {k: np.asarray(z[k], np.float64) for k in ("w", "b", "mean", "std")}
    gates.append(("G-MYW", not bad_myw, f"loaded={len(myw)}/4 bad={bad_myw}"))
    if bad_myw:
        return ret()
    stage("replay:load")
    runs, ek2rk = payload["runs"], payload["evalkey_to_run"]
    order = sorted(runs)
    pos_of = {rk: i for i, rk in enumerate(order)}
    cls_arr = np.array([runs[rk][0] for rk in order])
    fold_arr = np.array([runs[rk][1] for rk in order])
    shard_dir = out_root / payload["src_stamp"] / "shards"
    parts = {c: [] for c in CELLS}
    rr = {c: [] for c in CELLS}
    for f in payload["shard_files"]:
        side = json.loads((shard_dir / f"{f}.json").read_text())
        z = np.load(shard_dir / f"{f}.npz")
        for cell, xk, nk in (("command|L48", "X_cmd", "cmd_counts"), ("reply|L48", "X_rep", "rep_counts")):
            parts[cell].append(z[xk])
            rr[cell].extend(pos_of[ek2rk[k]] for k, n in zip(side["keys"], z[nk]) for _ in range(int(n)))
    myfit_bad = []
    for cell in CELLS:
        X = torch.from_numpy(np.concatenate(parts[cell])).to(device)
        row_run = np.asarray(rr[cell], np.int64)
        for prob, (pos, neg) in PROBLEMS.items():
            stage(f"replay:score {cell} {prob}")
            tag = f"{cell} {prob}"
            rec = payload["record"][cell][prob]
            mv = myrec["values"][cell][prob]["refit"]
            my_scores, my_auc = mv["run_scores"], mv["auroc"]
            in_p = np.isin(cls_arr, pos + neg)
            y_run = np.isin(cls_arr, pos).astype(np.float64)
            row_in, row_fold = in_p[row_run], fold_arr[row_run]
            fits, trs, ys = {}, {}, {}
            s_chk, n_chk = np.zeros(len(order)), np.zeros(len(order))
            for k in range(5):
                tr = np.nonzero(row_in & (row_fold != k))[0]
                te = np.nonzero(row_in & (row_fold == k))[0]
                mw = myw[(cell, prob)]
                fits[k] = tuple(torch.from_numpy(a).to(device) for a in (mw["mean"][k], mw["std"][k], mw["w"][k])) + \
                    (torch.tensor(float(mw["b"][k]), dtype=torch.float64, device=device),)
                trs[k] = torch.from_numpy(tr).to(device)
                ys[k] = torch.from_numpy(y_run[row_run[tr]]).to(device)
                s, n = run_means(row_logits(X, torch.from_numpy(te).to(device), fits[k][0], fits[k][1], fits[k][2], fits[k][3]).cpu().numpy(),
                                 row_run, te, len(order))
                s_chk += s
                n_chk += n
            chk = {order[i]: float(s_chk[i] / n_chk[i]) for i in range(len(order)) if in_p[i] and n_chk[i] > 0}
            dmy = max((abs(chk[k_] - my_scores[k_]) for k_ in set(chk) & set(my_scores)), default=float("inf"))
            myfit_ok = set(chk) == set(my_scores) and dmy <= 1e-9
            if not myfit_ok:
                myfit_bad.append(tag)
            sdiff = max((abs(my_scores[k_] - rec["scores"][k_]) for k_ in set(my_scores) & set(rec["scores"])), default=float("inf"))
            strict = set(my_scores) == set(rec["scores"]) and sdiff <= TOL_STRICT
            refit_tight = abs(my_auc - rec["auroc"]) <= TOL_REFIT
            emit(lines, f"HONALL MYFIT {tag} reproduces_record_refit={str(myfit_ok).upper()} strict_within_1e-5={str(strict).upper()} "
                        f"tight_tier_1e-3={'MATCH' if refit_tight else 'MISMATCH'} "
                        f"auroc_diff_bucket={bucket(abs(my_auc - rec['auroc']), (1e-4, 1e-3, 5e-3))}")
            info[tag] = {"refit_tight": refit_tight, "strict": strict, "fitq": None}
            lay = lays[(cell, prob)]
            ctx = {"X": X, "row_run": row_run, "row_in": row_in, "row_fold": row_fold, "folds": lay["folds"], "lay": lay, "fits": fits,
                   "order": order, "pos_of": pos_of, "in_p": in_p, "runs": runs, "pos": pos, "neg": neg, "rec": rec, "device": device}
            rp = replay_score(ctx, "stored", False)  # PRODUCER weights + PRODUCER mean/std on the shard rows (F2)
            emit(lines, f"HONALL REPLAY {tag} {'MATCH' if rp['match'] else 'MISMATCH'} mode=stored tier={rp['tier'] or 'none'} "
                        f"modes_matching={int(rp['match'])} max_diff_bucket={rp['diff_bucket']}")
            flags[f"replay {tag}"] = rp["match"]
            rot = replay_score(ctx, "stored", True)  # N1 on real data: must fail at the fp32 tier
            emit(lines, f"HONALL NEGCTRL N1 {tag} {'UNEXPECTED_PASS' if rot['match'] else 'FAIL_AS_PREDICTED'}")
            flags[f"N1 {tag}"] = not rot["match"]
            n2 = abs(my_auc - (rec["auroc"] + 0.01)) <= TOL_REFIT
            emit(lines, f"HONALL NEGCTRL N2 {tag} {'UNEXPECTED_PASS' if n2 else 'FAIL_AS_PREDICTED'}")
            flags[f"N2 {tag}"] = not n2
            v = {"replay": {"layout": lay["keys"], "stored": rp, "rotated": {"match": rot["match"], "tier": rot["tier"]}},
                 "my_refit_reproduced": {"ok": myfit_ok, "max_abs_diff": dmy}, "strict_max_abs_diff": sdiff}
            if rp["match"]:
                stage(f"replay:fitq {cell} {prob}")
                fq = fit_quality(ctx, "stored", my_scores, rp["run_scores"], trs, ys)
                info[tag]["fitq"] = fq
                v["fit_quality"] = fq
                pf = fq["per_fold"]
                emit(lines, f"HONALL FITQ {tag} resid_explained={str(fq['resid_explained']).upper()} "
                            f"resid[{bracket(fq['max_abs_residual'])}] dhat[{bracket(fq['max_abs_dhat'])}] "
                            f"stored_minus_refit[{bracket(fq['max_abs_stored_minus_refit'])}] rank_identical={str(fq['rank_identical']).upper()} "
                            f"swaps_all_pairs={fq['swaps_all_pairs']} swaps_pos_neg={fq['swaps_pos_neg']} runs={fq['n_runs']}")
                emit(lines, f"HONALL FITQ_STORED {tag} fitter_ok={str(fq['fitter_ok']).upper()} "
                            f"grad_stored_coords_small={sum(f['grad_small'] for f in pf)}/5 "
                            f"grad_stored_coords_max[{bracket(max(f['grad_stored_coords'] for f in pf), (1e-6, 1e-4, 1e-2))}] "
                            f"J_gap_small={sum(f['gap_small'] for f in pf)}/5 J_gap_nonneg={sum(f['fitter_ok'] for f in pf)}/5 "
                            f"stats={'NA' if fq['stats_ok'] is None else str(fq['stats_ok']).upper()} "
                            + ("" if fq["stats_ok"] is None else
                               f"mu_same={sum(f['mu_same'] for f in pf)}/5 sigma_same={sum(f['sigma_same'] for f in pf)}/5 "
                               f"dmu_max[{bracket(max(f['max_abs_dmu'] for f in pf), (1e-6, 1e-4, 1e-2))}] "
                               f"sigma_ratio_m1_max[{bracket(max(f['max_abs_sigma_ratio_m1'] for f in pf), (1e-6, 1e-4, 1e-2))}] ")
                            + f"branch={fq['branch']}")
            vals.setdefault(cell, {})[prob] = v
        del X
        if device == "cuda":
            torch.cuda.empty_cache()
    gates.append(("G-MYFIT", not myfit_bad, f"reproduced={4 - len(myfit_bad)}/4"))
    return ret()


@app.function(image=image, gpu="H100", cpu=(8, 8), memory=(65536, 65536), timeout=900, startup_timeout=240, scaledown_window=2,
              retries=0,  # addendum F6: request = limit; one call, no fan-out
              volumes={"/w": w_vol.read_only(), "/out": out_vol.read_only()})
def replay_fit(payload: dict) -> dict:
    err = None
    try:
        print("HONALL-LIVE STAGE replay start", flush=True)
        res = replay_and_compare(payload, "cuda")
    except Exception as e:  # sanitized (addendum C3)
        err = f"{type(e).__name__} at {_STAGE['s']}"
    if err is not None:
        raise RuntimeError(f"fit failed: replay {err}")
    return res


def replay_pipeline(mode, blobs, meta, w_root: str, out_root: Path, src_stamp: str, result_rel: str, result_sha: str, expect, pin_sha,
                    replay_runner, write_root: Path) -> dict:
    """Addendum F: gates, ONE replay call, then PRINTED / LINK / FIT_VERDICT (B5 unchanged) and the exit rule (B6 as amended)."""
    stamp = meta["stamp"]
    run_dir = write_root / f"{stamp}_replay"
    lines, res = [], {"prestated_F_sha256": "444e44c3ff2260bc3d19510cb704113b7e332e32c3e04e0572041bea1db0dc75",
                      "prestated_G_sha256": "5772826235e2df3cd4298f762ea13e1d7745f36a2f2b37e0e4db308015e3ce9c",
                      "prestated_H_sha256": "3ae3c451bd1c8ec311d724a1a375b1f59895785d30b59efc263a360e192b04f2",
                      "prestated_H2_sha256": "4307d806c207ca7df8dbdef6b2fc448cac0011653c4ce40d5106f6036db03b74",
                      "label": f"PRELIMINARY route C replay-only (addendum F): the producer's per-fold npz on the stored shards of {src_stamp}; "
                               f"{REFIT_LABEL}", "mode": mode, "stamp": stamp, "meta": meta, "src_stamp": src_stamp,
                      "config": {"tol_replay": TOL_REPLAY, "tol_replay_fp32": TOL_REPLAY_FP32, "tol_strict": TOL_STRICT, "tol_refit": TOL_REFIT,
                                 "dhat_small": DHAT_SMALL, "resid_floor": RESID_FLOOR, "resid_frac": RESID_FRAC, "layout": "per-fold (F1)",
                                 "timeouts_s": {"hash_shards": 600, "replay_fit": 900, "orchestrate_replay": 1800, "orchestrate_toy": 600},
                                 "startup_timeouts_s": {"hash_shards": 120, "replay_fit": 240, "orchestrate_replay": 120, "orchestrate_toy": 180},
                                 "scaledown_window_s": 2, "hard_bound_usd": 1.66874336, "pinned_sources": PIN_SRC if mode == "replay" else "toy"},
                      "gates": {}, "diag": {}}
    exitbad = []
    under_out = str(write_root).startswith("/out")

    def gate(name, ok, detail=""):
        stage(f"gate {name}")
        emit(lines, f"HONALL GATE {name} {'PASS' if ok else 'STOP'} {detail}".rstrip())
        res["gates"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            raise Stop(name)

    def diag(name, ok, detail="", counts=True):
        emit(lines, f"HONALL DIAG {name} {str(bool(ok)).upper()} {detail}".rstrip())
        res["diag"][name] = {"ok": bool(ok), "detail": detail}
        if counts and not ok:
            exitbad.append(name)

    def finish(code):
        run_dir.mkdir(parents=True, exist_ok=True)
        rp = run_dir / f"honest_all_replay_{stamp}.json"
        body = json.dumps(res, default=str).encode()
        with open(rp, "xb") as fh:
            fh.write(body)
        if under_out:
            out_vol.commit()
        where = f"{OUT_VOL}:{rp.relative_to('/out')}" if under_out else str(rp)
        lines.append("HONALL RESULT " + json.dumps({"exit": code, "path": where, "sha256": sha256(body)}))
        return {"lines": lines, "exit": code}

    crash = None
    try:
        if mode.startswith("toy-"):  # addendum H: in-container unit negatives of the pure gate functions (no extra launch)
            un = unit_negatives()
            for name, ok in un:
                emit(lines, f"HONALL UNITNEG {name} {'AS_PREDICTED' if ok else 'UNEXPECTED'}")
            res["unit_negatives"] = dict(un)
            if not all(ok for _, ok in un):
                exitbad.append("unit_negatives")
        # G-PIN: the pinned committed inputs, the 606 routing versions, the 4 producer npz and the record's saved_weights entries
        bad = [n for n, h in pin_sha.items() if n not in blobs or sha256(blobs[n]) != h]
        if mode == "replay" and "npz_sha" in meta:  # addendum H (N6): in the real replay the npz shas come only from the pinned NPZ_SHA
            bad.append("npz_sha override in replay mode")
        routing_entries = []
        if not bad:
            sources = json.loads(blobs["sources_v2.json"])
            routing_entries = [e for e in sources["files"] if e.get("role") == "routing"]
            bad += [e["path"] for e in routing_entries if sha256(blobs.get(f"routing/{e['commit']}/{Path(e['path']).name}", b"")) != e["sha256"]]
        npz_expect = meta.get("npz_sha") or {f"{c}::{p}": h for (c, p), h in NPZ_SHA.items()}
        for c in CELLS:
            for p in PROBLEMS:
                f = Path(w_root) / cell_file(c, p)
                if not f.is_file() or sha256(f.read_bytes()) != npz_expect[f"{c}::{p}"]:
                    bad.append(f"npz {c} {p}")
        if not bad:
            rec = json.loads(blobs["record.json"])
            bad += [f"saved_weights {c} {p}" for c in CELLS for p in PROBLEMS
                    if rec["cells"][c]["saved_weights"][p].get("path") != cell_file(c, p)
                    or rec["cells"][c]["saved_weights"][p].get("sha256") != npz_expect[f"{c}::{p}"]]
        gate("G-PIN", not bad and len(routing_entries) == expect["routing"],
             f"pinned={len(pin_sha)} routing={len(routing_entries)} npz=4 bad={len(bad)}")
        # G-MANIFEST-FILE: the committed manifest (phase 1) matches the launch's --manifest-sha and is canonical (F3)
        mb = blobs.get("replay_manifest.json", b"")
        items = json.loads(mb) if mb else []
        # addendum G1: EXACT coverage, not counts (manifest_coverage, unit-tested in unit_negatives)
        cov_ok, stems, cov_detail = manifest_coverage(items, src_stamp, result_rel, npz_expect, result_sha)
        okm = bool(mb) and sha256(mb) == meta.get("manifest_sha") and manifest_bytes(items) == mb and cov_ok
        gate("G-MANIFEST-FILE", okm, cov_detail)
        # addendum G1: the bytes of the real run's record are hashed before any use here (D5, PRINTED read it below)
        myrec_raw = (out_root / result_rel).read_bytes()
        gate("G-RESULT-LOCAL", sha256(myrec_raw) == result_sha, "")
        myrec = json.loads(myrec_raw)
        prep_ids = sorted(q["shard"] for q in myrec["prep"] if q.get("shard"))
        gate("G-STEMS-LOCAL", stems_match(stems, prep_ids), f"stems={len(stems)} prep_shards={len(prep_ids)}")  # before any GPU spend
        # runs: classes from the record keys, folds from the fold list (as in the real run's G-CLASS / G-FOLD)
        run_keys = rec["run_keys"]
        cls = {rk: rk.split("|", 1)[0] for rk in run_keys}
        cnt = {c: sum(v == c for v in cls.values()) for c in ("sim", "B", "A")}
        gate("G-CLASS", cnt == {c: expect[c] for c in ("sim", "B", "A")}, f"sim={cnt['sim']} B={cnt['B']} A={cnt['A']} total={len(run_keys)}")
        fold_of = {}
        for k, mains in json.loads(blobs["folds.json"])["lists"].items():
            for m in mains:
                fold_of.setdefault(m, set()).add(int(k))
        same = sum(rec["run_main"].get(rk) in fold_of and fold_of[rec["run_main"][rk]] == {int(rec["run_folds"][rk])} for rk in run_keys)
        gate("G-FOLD", same == len(run_keys), f"folds_equal={same} of {len(run_keys)}")
        runs = {rk: [cls[rk], next(iter(fold_of[rec["run_main"][rk]]))] for rk in run_keys}
        # the record's own certificates (B2), unchanged
        stored_cert = {}
        for cell in CELLS:
            for p in PROBLEMS:
                per = {}
                for e in rec["cells"][cell].get("certificates", []):
                    cols = e.get("columns") if isinstance(e, dict) else None
                    if e.get("problem") == p and isinstance(cols, list) and 0 in cols:
                        j = cols.index(0)
                        per.setdefault(int(e["fold"]), []).append((bool(e["certified_per_column"][j]), float(e["max_grad_per_column"][j])))
                parsed = sorted(per) == [0, 1, 2, 3, 4] and all(len(v) == 1 for v in per.values())
                n_ok = sum(v[0][0] for v in per.values()) if parsed else -1
                n_le = sum(v[0][1] <= TOL_GRAD for v in per.values()) if parsed else -1
                oc = rec["cells"][cell]["problems"][p].get("observed_certified")
                ok_c = parsed and n_ok == 5 and n_le == 5 and oc is True
                stored_cert[f"{cell} {p}"] = {"ok": ok_c}
                emit(lines, f"HONALL STOREDCERT {cell} {p} observed_certified={str(oc).upper()} parsed={str(parsed).upper()} "
                            f"folds_certified={n_ok}/5 folds_max_grad_le_1e-6={n_le}/5 pass={str(ok_c).upper()}")
        fails = (rec.get("uncut_gates") or {}).get("failures")  # B2's record-side count, restored (Codex r9 follow-up)
        n_fail = len(fails) if isinstance(fails, (list, dict)) else -1
        emit(lines, f"HONALL STOREDCERT uncut_gates_failures_count={n_fail}")
        res["stored_cert"] = {"by_cell_problem": stored_cert, "uncut_gates_failures_count": n_fail}  # addendum H (N1), as pipeline()
        shard_files = stems
        payload = {"out_root": str(out_root), "w_root": w_root, "src_stamp": src_stamp, "result_rel": result_rel, "result_sha": result_sha,
                   "manifest": items, "shard_files": shard_files, "runs": runs, "evalkey_to_run": {ek: rk for rk, ek in run_keys.items()},
                   "record": {c: {p: {"scores": rec["cells"][c]["problems"][p]["run_scores"],
                                      "auroc": rec["cells"][c]["problems"][p]["run_level"]["auroc"]} for p in PROBLEMS} for c in CELLS},
                   "saved": {c: {p: rec["cells"][c]["saved_weights"][p] for p in PROBLEMS} for c in CELLS}}
        stage("replay")
        fr = replay_runner(payload)  # the single replay call site (addendum H, N7): one call, no fan-out, by reading; no counter assert
        for name, ok, detail in fr["gates"]:
            gate(name, ok, detail)
        lines.extend(fr["lines"])
        res["values"], res["fit_device"] = fr["values"], fr["device"]
        for k, v in fr["flags"].items():
            if not v:
                exitbad.append(k)
        # D5: my refit's certificates, as recorded by the real run
        # myrec: the bytes hashed by G-RESULT-LOCAL above (addendum G1); never re-read unverified
        ncert = sum(bool(c.get("certified")) for cell in CELLS for p in PROBLEMS
                    for c in myrec["values"][cell][p]["refit"]["certificates"].values())
        diag("D5 certified_folds", ncert == 20, f"{ncert}/20")
        # PRINTED (C1 exact) and LINK, unchanged
        stage("printed")
        prov = {}
        for line in blobs["provenance.jsonl"].decode().splitlines():
            if line.strip():
                o = json.loads(line)
                prov[o["row"]] = o
        simpool = json.loads(blobs["simpool.json"])
        res["printed"] = {}
        emit(lines, f"HONALL LABEL {REFIT_LABEL}")
        for row, (cell, prob) in PRINTED_ROWS.items():
            o = prov.get(row, {})
            want_kp = (["cells", cell, "problems", prob, "run_level", "auroc"] if prob == "simh_vs_b"
                       else ["items", SIMPOOL_ITEM, "run_level_auroc", "routec", cell])
            nums = re.findall(r"\d+\.\d+", str(o.get("printed", "")))
            if len(nums) != 1 or o.get("key_path") != want_kp:
                emit(lines, f"HONALL PRINTED row={row} UNPARSEABLE")
                exitbad.append(f"printed {row}")
                continue
            pv = float(nums[0])
            mine = myrec["values"][cell][prob]["refit"]["auroc"]
            recv = rec["cells"][cell]["problems"][prob]["run_level"]["auroc"]
            m_ok, r_ok = printed_unchanged(mine, nums[0]), printed_unchanged(recv, nums[0])
            near = abs(abs(mine - pv) - HALF_DP) < BOUNDARY or abs(abs(recv - pv) - HALF_DP) < BOUNDARY
            emit(lines, f"HONALL PRINTED row={row} refit_unchanged_2dp={'MATCH' if m_ok else 'MISMATCH'} record={'MATCH' if r_ok else 'MISMATCH'} "
                        f"near_boundary={str(near).upper()} role={ROW_ROLE[prob]}")
            res["printed"][row] = {"refit_match": m_ok, "record_match": r_ok, "near_boundary": near, "role": ROW_ROLE[prob]}
            if not r_ok:
                exitbad.append(f"printed_record {row}")
        link = all(simpool["items"][SIMPOOL_ITEM]["run_level_auroc"]["routec"][c] == rec["cells"][c]["problems"]["a_vs_b"]["run_level"]["auroc"]
                   for c in CELLS)
        emit(lines, f"HONALL LINK rows95-98 a_vs_b_equal={str(link).upper()}")
        res["link_rows95_98"] = link
        if not link:
            exitbad.append("link")
        # FIT_VERDICT: B5, unchanged
        stage("verdict")
        printed_ok = len(res["printed"]) == len(PRINTED_ROWS) and all(v["refit_match"] for v in res["printed"].values())
        res["fit_verdict"] = {}
        for tag, inf in fr["info"].items():
            fq, cert = inf["fitq"], stored_cert[tag]
            if not printed_ok and inf["strict"]:
                verdict = "UNRESOLVED"
            elif inf["strict"]:
                verdict = "PASS_STRICT"
            elif fq is None:
                verdict = "UNRESOLVED"
            elif not fq["fitter_ok"]:
                verdict = "STOP_FITTER_CHECK"
            elif not fq["resid_explained"]:
                verdict = "RECIPE_NOT_FULLY_REGISTERED"
            elif (fr["flags"].get(f"replay {tag}") and fq["stats_ok"] is True and cert["ok"]
                  and (fq["rank_identical"] or fq["max_abs_dhat"] <= DHAT_SMALL) and printed_ok):
                verdict = "PASS_RULING"
            else:
                verdict = "UNRESOLVED"
            branch = fq["branch"] if fq else "no_replay"
            emit(lines, f"HONALL FIT_VERDICT {tag} {verdict} strict={str(inf['strict']).upper()} printed_unchanged_8={str(printed_ok).upper()} "
                        f"stored_cert={str(cert['ok']).upper()} diagnosis={branch}")
            res["fit_verdict"][tag] = {"verdict": verdict, "diagnosis": branch, "strict": inf["strict"], "refit_tight": inf["refit_tight"]}
            if verdict not in ("PASS_STRICT", "PASS_RULING"):
                exitbad.append(f"fit_verdict {tag}")
        res["exit_reasons"] = exitbad
        return finish(1 if exitbad else 0)
    except Stop as e:
        res["stopped_at"] = str(e)
        return finish(1)
    except Exception as e:  # sanitized (addendum C3)
        own = isinstance(e, RuntimeError) and str(e).startswith(("prep ", "fit failed"))
        crash = {"type": type(e).__name__, "stage": _STAGE["s"], "remote": str(e) if own else None}
    res["crashed"] = crash
    emit(lines, f"HONALL CRASH stage={crash['stage']} type={crash['type']}" + (f" remote=[{crash['remote']}]" if crash["remote"] else ""))
    return finish(1)


@app.function(image=image_light, cpu=(1, 1), memory=(2048, 2048), timeout=1800, startup_timeout=120, scaledown_window=2,
              retries=0,  # addendum F6: request = limit
              volumes={"/out": out_vol, "/w": w_vol.read_only()})
def orchestrate_replay(blobs: dict, meta: dict) -> dict:
    """Addendum F phase 2 (real data): one replay_fit call on the pinned stored outputs of REAL_STAMP."""
    err = None
    try:
        return replay_pipeline("replay", blobs, meta, "/w", Path("/out"), REAL_STAMP, REAL_RESULT, REAL_RESULT_SHA, REAL_EXPECT, PIN_SHA,
                               lambda payload: replay_fit.remote(payload), Path("/out"))
    except Exception as e:  # sanitized (addendum C3); replay_pipeline() already catches its own errors
        err = (type(e).__name__, _STAGE["s"])
    return crash_result(err)


@app.local_entrypoint()
def main(mode: str = "real", pinned: str = "", git_head: str = "", stamp: str = "", manifest_out: str = "", manifest: str = "",
         manifest_sha: str = ""):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode}")
    if mode == "hash-shards":  # addendum F3 phase 1: writes the manifest (paths, sizes, sha256 only) to a NEW local file
        mo = Path(manifest_out)
        if not manifest_out or mo.exists() or not mo.parent.is_dir():
            raise SystemExit("--manifest-out must name a new file in an existing folder")
        out = hash_shards.remote()
        with open(mo, "x") as fh:
            fh.write(out["manifest"])
        for line in out["lines"]:
            print(line)
        sys.exit(out["exit"])
    stamp = stamp or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"stamp": stamp, "git_head": git_head, "script_sha256": sha256(Path(__file__).read_bytes())}
    blobs = {}
    if mode in ("real", "replay"):
        root = Path(pinned)
        if not root.is_dir():
            raise SystemExit("--pinned must be the materialized inputs folder")
        blobs = {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
    if mode == "replay":  # addendum F3 phase 2: the committed manifest and its sha256 (the pin)
        if not manifest or not manifest_sha or "replay_manifest.json" in blobs:
            raise SystemExit("--manifest and --manifest-sha are required")
        blobs["replay_manifest.json"] = Path(manifest).read_bytes()
        meta["manifest_sha"] = manifest_sha
        out = orchestrate_replay.remote(blobs, meta)
    elif mode == "real":
        out = orchestrate.remote(mode, blobs, meta)
    else:
        out = orchestrate_toy.remote(mode, meta)
    for line in out["lines"]:
        print(line)
    sys.exit(out["exit"])
