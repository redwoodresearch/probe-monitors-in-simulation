"""refit3 (0d 10-16; option B after launch 2's G4 STOP): certify the 3 uncertified observed fold fits of 4f's all-sets attack record by a targeted refit,
leaving every other fit as recorded. Record: attack_all_2026-10-16_042630.json @ 59acf625e (sha256 46e8f7d5...); targets, max |grad| vs
the 1e-6 bar: command|L63 c fold 4 (1.1175e-6), command|L63 d fold 2 (1.3237e-6), reply|L48 d fold 4 (1.0168e-6).

Method: EXACT NEWTON (0d, option 1, after launch 3's G5 STOP showed an L-BFGS warm start does not certify these fits). A target's
stored solution (4f's saved weights npz: W, b and the training rows' mean and std) starts exact Newton steps on torch_probe.fit's
objective G (C = 0.01, float64, intercept unpenalized; fit_newton), then torch_probe.certify. G is strictly convex with a unique optimum,
so this is a post hoc numerical repair of the same estimator. 0d's rule, fixed before any downstream score: a fit is accepted iff
certify's max |grad| < 1e-6 AND every Cholesky solve's relative residual <= 1e-8; if any of the 3 fails, NOTHING is replaced and only
the Newton deltas are written, as a sensitivity (refit3_sensitivity_<stamp>.json). The 37 certified observed fits and every null fit are NOT refit: their stored weights and the record's values are reused.
(The held global cap raise, 960ab0ed3, moved certified fits too: Codex cap r1, 291c3e119.)

The record lacks two inputs, rebuilt here with gates: the per-row out-of-fold scores (call-level metrics need them; rebuilt by scoring
each fold's test rows with that fold's stored weights ON AN H100 WITH 4f's EXACT TORCH OPS AND SHAPES, fit_a2's S and S2, so the
replay can be bitwise; launch 2's CPU rebuild differed from the in-run scores at ~1e-14 and moved call-level AUROCs by up to 1.4e-8) and the per-run skip list (rebuilt by 4f's skip stage). Gates, each a STOP:
  G1 inputs: 4f's build_inputs, run REMOTELY on the shipped committed files (config/sources_v1.json), gives the record's run keys, folds,
     mains and sets by name.
  G2 skip: 4f's skip stage reproduces the record's dropped_rows and staged sha256s; meta checked = 3,939.
  G3 rows: per affected cell, rows, rows per run, runs without rows and chunks read equal the record's.
  G4 stored: every npz sha256 equals the record's; the replayed run scores and held-out run 2 run scores reproduce the record's (G4a,
     <= 1e-9, finite); a no-refit baseline reproduces EVERY output the patch replaces (G4b, <= 1e-9, finite); certify() of each target's
     stored weights on its training rows reproduces the record's max |grad| (relative 1e-6).
  G5 new: every Newton target is accepted by the rule above (else the sensitivity path; no optimizer iteration after a STOP).
Then, for the affected cells only, 4f's own metric code recomputes run/call AUROC and catch, both bootstraps, the paired c - d bootstrap,
the held-out run 2 block (c fold models) and the null summaries (null draws reused). A NEW record is written: the old one with those cells
patched, the superseded certificates kept, a refit3 block (gates, deltas, new weights' shas) and the list of every changed key path. 4f's
files are never edited. Option B: replay_cell (H100: load, replay, warm-started refits) returns per-row scores; gate_cell (CPU) runs G4a / G4b
strictly at 1e-9 (on failure EVERY failing leaf is printed and raised) and 4f's metric code. A failed stage cancels every outstanding call.
Run from this folder:
  PYTHONPATH=src <modal> run experiments/2026-10-16_attack_refit3/run_refit3.py::main_r3 --stage dry|skip|fit
"""

import copy
import hashlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_attack_all as ra  # noqa: E402  (4f's copy @ 59acf625e, unchanged)

base, a1, ui = ra.base, ra.a1, ra.ui
TARGETS = {"command|L63": [("c_simatk_vs_run1", 4), ("d_run2_vs_run1", 2)], "reply|L48": [("d_run2_vs_run1", 4)]}
TOL, REL = 1e-9, 1e-6
W3VOL = "probe-shift_aswin-surya_sub-attackrefit3_w_kill-2026-11-15"
w3_vol = modal.Volume.from_name(W3VOL, version=2, create_if_missing=True)
w4f_vol = modal.Volume.from_name(ra.SOLVOL, version=2).read_only()
SOURCES = "config/sources_v1.json"
app = ra.app
_files = [(HERE / f, f"/root/{f}") for f in ("run_simh_control.py", "run_controls_a1.py", "uncut_inputs.py", "uncut_inputs_all.py", "run_attack_all.py")]  # (this entry file is mounted by modal run)
_files += [(p, f"/root/config/{p.name}") for p in sorted((HERE / "config").glob("inputs_all_*_v1.json"))]  # build_inputs reads HERE/config
img = ra.base.gpu_image  # torch + numpy (CPU use here)
tok_img = ra.base.tok_image  # (launch 1 fix) every container imports this entry file, so every image needs run_attack_all and its helpers
for _f in _files:
    img, tok_img = img.add_local_file(*_f), tok_img.add_local_file(*_f)


def blob_get(data: dict, commit: str, path: str) -> bytes:
    """A shipped committed file by (commit or unique prefix, path)."""
    m = [k for k in data if k.endswith(":" + path) and k.split(":", 1)[0].startswith(commit)]
    if len(m) != 1:
        raise KeyError(f"{path} @ {commit}: {len(m)} shipped matches")
    return data[m[0]]


@app.function(image=img, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0)
def prep_r3(data: dict) -> dict:
    """G1: 4f's build_inputs on the shipped files (its git reads redirected to them), checked against the record by run name."""
    def git_json(_repo, spec):
        b = blob_get(data, spec["commit"], spec["path"])
        if hashlib.sha256(b).hexdigest() != spec["sha256"]:
            raise ValueError(f"{spec['path']}: sha256 differs from the citing config's")
        return json.loads(b)

    ra._gitb = lambda c, p: blob_get(data, c, p)
    ui.git_json = ra.uia.git_json = git_json
    u = ra.build_inputs(dry=False)
    (rk,) = [k for k in data if k.endswith(":" + "experiments/2026-10-16_cmd_sim_detector_uncut_all/results/attack_all_2026-10-16_042630.json")]
    rec = json.loads(data[rk])
    got = {"run_keys": {e["name"]: e["key"] for e in u["epochs"]}, "run_folds": {e["name"]: e["fold"] for e in u["epochs"]},
           "run_main": {e["name"]: e["task"] for e in u["epochs"]}, "run_set": {e["name"]: e["set"] for e in u["epochs"]}}
    bad = [k for k, v in got.items() if v != rec[k]]
    if bad:
        raise ValueError(f"G1 STOP: rebuilt inputs differ from the record's {bad}")
    return u


@app.function(image=tok_img, cpu=(4, 4), memory=(32768, 32768), timeout=3600, retries=0, volumes={"/u": ra.u_vol, "/hf": base.hf_vol})
def skip_r3(want: dict, shas: dict, full_len: dict, roots: dict) -> dict:
    return ui.uncut_skip("/u", want, shas, full_len, roots)


NEWTON = {"max_steps": 20, "tol": 1e-10, "chol_rel_residual_bar": 1e-8, "accept_max_grad_below": 1e-6, "armijo": 1e-4, "chunk_rows": 8192}


def fit_newton(X, Y, C: float, W0, b0):
    """(0d 10-16, option 1; rule fixed before any downstream score) Exact Newton on torch_probe.fit's objective
    G = (sum logloss + ||w||^2 / (2C)) / n from the stored solution (W0, b0): augmented design [X, 1]; Hessian
    X~^T diag(p (1 - p)) X~ / n + diag(1, ..., 1, 0) / (C n) (the intercept unpenalized; every intercept cross term kept); Cholesky
    solve with its relative residual ||H delta + g|| / ||g|| reported per step; Armijo backtracking on G. Stops at max |grad| <= 1e-10,
    after 20 steps, or when the line search finds no decrease. The first Hessian's eigenvalues give its condition number. Then
    torch_probe.certify at the result. Accepted iff certify's max |grad| < 1e-6 AND every step's Cholesky relative residual <= 1e-8."""
    import math

    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp

    n, d = X.shape
    y = Y.reshape(n, -1)[:, 0].to(X.dtype)
    w, b = torch.as_tensor(W0, dtype=X.dtype).reshape(d).clone(), torch.as_tensor(b0, dtype=X.dtype).reshape(-1)[0].clone()
    G = lambda w_, b_: float((F.binary_cross_entropy_with_logits(X @ w_ + b_, y, reduction="sum") + (w_ * w_).sum() / (2 * C)) / n)  # noqa: E731

    def grad(w_, b_):
        r = torch.sigmoid(X @ w_ + b_) - y
        return torch.cat([(X.T @ r + w_ / C) / n, (r.sum() / n).reshape(1)])

    g, log, cond = grad(w, b), [], None
    for step in range(NEWTON["max_steps"]):
        gmax = float(g.abs().max())
        if gmax <= NEWTON["tol"]:
            break
        s = torch.sigmoid(X @ w + b)
        s = s * (1 - s)
        H = torch.zeros(d + 1, d + 1, dtype=X.dtype, device=X.device)
        for i in range(0, n, NEWTON["chunk_rows"]):
            Xc = X[i:i + NEWTON["chunk_rows"]]
            H[:d, :d] += Xc.T @ (Xc * s[i:i + NEWTON["chunk_rows"], None])
        H[:d, d] = H[d, :d] = X.T @ s
        H[d, d] = s.sum()
        H /= n
        H[:d, :d] += torch.eye(d, dtype=X.dtype, device=X.device) / (C * n)
        if cond is None:
            ev = torch.linalg.eigvalsh(H)
            cond = {"min_eig": float(ev[0]), "max_eig": float(ev[-1]), "condition": float(ev[-1] / ev[0])}
        L, info = torch.linalg.cholesky_ex(H)
        if int(info) != 0:  # not positive definite in float64: recorded; the fit is then rejected by the rule (no crash, no retry)
            log.append({"step": step, "max_grad_before": gmax, "chol_rel_residual": None, "status": "cholesky_failed", "stop": f"Cholesky failed (info {int(info)})"})
            break
        delta = torch.cholesky_solve((-g).reshape(-1, 1), L).reshape(-1)
        res = float(torch.linalg.norm(H @ delta + g) / torch.linalg.norm(g))
        G0, slope, t = G(w, b), float(g @ delta), 1.0
        while t >= 2.0 ** -30:
            w1, b1 = w + t * delta[:d], b + t * delta[d]
            G1 = G(w1, b1)
            if G1 <= G0 + NEWTON["armijo"] * t * slope:
                break
            t /= 2
        else:
            log.append({"step": step, "max_grad_before": gmax, "chol_rel_residual": res, "stop": "line search found no decrease"})
            break
        w, b = w1, b1
        g = grad(w, b)
        log.append({"step": step, "max_grad_before": gmax, "t": t, "chol_rel_residual": res, "G_before": G0, "G_after": G1, "max_grad_after": float(g.abs().max())})
    W, bb = w.reshape(d, 1), b.reshape(1)
    cert = tp.certify(X, y.reshape(n, 1), W, bb, C)
    failed = any(x.get("status") == "cholesky_failed" for x in log)  # (Codex n1 P1) an explicit status, never a non-finite number
    rs = [x["chol_rel_residual"] for x in log if x.get("chol_rel_residual") is not None]
    allfin = all(math.isfinite(r) for r in rs)  # (Codex n2 P2) every residual finite, so max() cannot hide a NaN
    rmax = max(rs) if rs and allfin else None
    ok = bool(not failed and allfin and rmax is not None and math.isfinite(cert["max_grad"]) and cert["max_grad"] < NEWTON["accept_max_grad_below"]
              and rmax <= NEWTON["chol_rel_residual_bar"])
    san = lambda o: {k: san(v) for k, v in o.items()} if isinstance(o, dict) else [san(v) for v in o] if isinstance(o, list) else (  # noqa: E731
        None if isinstance(o, float) and not math.isfinite(o) else o)  # (Codex n1 P1) no non-finite number reaches the strict JSON writer
    log, cond = san(log), san(cond)
    return W, bb, cert | {"n_steps": len(log), "newton_log": log, "chol_rel_residual_max": rmax, "hessian_condition_first_step": cond,
                          "accepted": ok, "rule": NEWTON, "start": "exact Newton from the stored solution"}


def heldout_block(np, prep, runs, row_run, h2: dict, hr2: dict) -> dict:
    """fit_a2's held-out run 2 block @ 59acf625e, verbatim in substance, on 1-D out-of-fold arrays."""
    c = prep["c_simatk_vs_run1"]
    cs = {runs[j]["name"]: float(c["oof"][i]) for i, j in enumerate(c["ids"])}
    sim = {n: v for n, v in cs.items() if n.startswith("sim|")}
    r1 = {n: v for n, v in cs.items() if n.startswith("run1|")}
    task = {r["name"]: r["task"] for r in runs}.__getitem__
    hold = {}
    for name, pos, neg in (("sim_vs_heldout_run2", sim, h2), ("heldout_run2_vs_run1", h2, r1)):
        sc, yy = np.array([*pos.values(), *neg.values()]), np.array([1] * len(pos) + [0] * len(neg))
        hold[name] = {"auroc": base.auroc(sc, yy), "n_pos": len(pos), "n_neg": len(neg), "boot": ra.boot_pairs(sc, yy, [task(n) for n in [*pos, *neg]])}
    rr = c["pos"][row_run[c["rows"]]]
    names = [runs[c["ids"][i]]["name"] for i in rr]
    sim_rows = [float(v) for n, v in zip(names, c["oof_row"]) if n.startswith("sim|")]
    r1_rows = [float(v) for n, v in zip(names, c["oof_row"]) if n.startswith("run1|")]
    sim_t, r1_t = [task(n) for n in names if n.startswith("sim|")], [task(n) for n in names if n.startswith("run1|")]
    h_rows = [v for n in sorted(hr2) for v in hr2[n]]
    h_t = [task(n) for n in sorted(hr2) for _ in hr2[n]]
    for name, pos, pt, neg, nt in (("sim_vs_heldout_run2", sim_rows, sim_t, h_rows, h_t), ("heldout_run2_vs_run1", h_rows, h_t, r1_rows, r1_t)):
        sc, yy = np.array([*pos, *neg]), np.array([1] * len(pos) + [0] * len(neg))
        hold[name]["call_level"] = {"auroc": base.auroc(sc, yy), "n_pos_rows": len(pos), "n_neg_rows": len(neg), "boot": ra.boot_pairs(sc, yy, [*pt, *nt])}
    return {"n_runs": len(h2), "run_scores": h2, "contrasts": hold,
            "note": "run 2 runs scored with the (c) sim-vs-run1 fold model of their main task's fold (task unseen in that model's training); "
                    "sim and run 1 scores are (c)'s out-of-fold scores; class 1 = first world named"}


FIELDS = ("run_level", "call_level", "run_scores", "run_auroc_boot", "call_auroc_boot", "null")


def cmp(a, b, path="") -> list:
    """(review r1) Every leaf of a against b: numbers must both be finite and within TOL; other leaves equal; same keys and lengths."""
    import math

    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))[:5]}"]
        return [x for k in sorted(a) for x in cmp(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        return [f"{path}: lengths {len(a)} != {len(b)}"] if len(a) != len(b) else [x for i, (u, v) in enumerate(zip(a, b)) for x in cmp(u, v, f"{path}/{i}")]
    if num(a) and num(b):
        return [] if math.isfinite(a) and math.isfinite(b) and abs(a - b) <= TOL else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def prep_meta(runs: list, row_run, np) -> dict:
    """fit_a2's per-problem index arrays (run ids, labels, positions, rows, row positions, folds), out-of-fold arrays empty."""
    prep = {}
    for pr in ra.PROBLEMS:
        ids = np.array([j for j, r in enumerate(runs) if r["batch"] in pr["pos"] + pr["neg"]])
        y = np.array([int(runs[j]["batch"] in pr["pos"]) for j in ids], np.int64)
        pos = np.full(len(runs), -1)
        pos[ids] = np.arange(len(ids))
        rows = np.flatnonzero(np.isin(row_run, ids))
        rpos = np.full(len(row_run), -1)
        rpos[rows] = np.arange(len(rows))
        prep[pr["name"]] = {"ids": ids, "y": y, "pos": pos, "rows": rows, "rpos": rpos, "fold": np.array([runs[j]["fold"] for j in ids]),
                            "oof": np.full(len(ids), np.nan), "oof_row": np.full(len(rows), np.nan)}
    return prep


def _replay_cell(cell: str, epochs: list, skip: dict, allowed: dict, rec_cell: dict, stamp: str, guard) -> dict:
    """(option B, 0d 10-16) On an H100, 4f's exact torch ops and shapes: every fold's test rows standardized by torch_probe.standardize on
    that fold's full training and test row matrices and scored as Xte @ W + b with W [d, 1] (fit_a2's S), held-out run 2 rows as
    ((X2 - mu_) / sd_) @ W[:, 0] + b[0] (fit_a2's S2), using the STORED weights; then the warm-started refits, scored the same way.
    Returns per-row score arrays only; gates and metrics are computed on CPU (gate_cell). The cost guard is 4f's."""
    if not guard.put(f"replay|{cell}", time.time(), skip_if_exists=True):
        raise RuntimeError(f"{cell}: a rescheduled attempt; refused by the cost guard")
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    t0 = time.time()
    ro, L = cell.split("|")[0], int(cell.split("|L")[1])
    X16, row_run, keep, no_rows, chunks = base._load(L, ro, epochs, skip, allowed)
    runs = [epochs[i] for i in keep]
    rpr = {r["name"]: int((row_run == j).sum()) for j, r in enumerate(runs)}
    if len(row_run) != rec_cell["rows"] or rpr != rec_cell["rows_per_run"] or sorted(no_rows) != sorted(rec_cell["runs_without_rows"]) \
            or chunks != rec_cell["chunks_read"]:
        raise ValueError(f"G3 STOP {cell}: rows, rows per run, zero-row runs or chunks differ from the record")
    t_load = time.time() - t0
    XT = torch.from_numpy(X16).cuda()
    del X16
    prep = prep_meta(runs, row_run, np)
    sw = {}
    for p in prep:
        spec = rec_cell["saved_weights"][p]
        raw = Path("/w4f", spec["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError(f"G4 STOP {cell}|{p}: stored weights sha256 differs from the record's")
        z = np.load(io.BytesIO(raw))
        sw[p] = {f: tuple(np.asarray(z[f"f{f}__{k}"], np.float64) for k in ("W", "b", "mean", "std")) for f in range(5)}
    gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).cuda()  # noqa: E731
    targets = dict.fromkeys(TARGETS[cell])
    base_rows, new_rows, std_diff, certs_old, certs_new, wnew = {}, {}, {}, [], [], {}
    r2_base, r2_new = {}, {}
    for p, d in prep.items():
        base_rows[p], new_rows[p] = d["oof_row"].copy(), d["oof_row"].copy()
        for f in range(5):
            tr_runs, te_runs = d["ids"][d["fold"] != f], d["ids"][d["fold"] == f]
            tr, te = np.flatnonzero(np.isin(row_run, tr_runs)), np.flatnonzero(np.isin(row_run, te_runs))
            Xtr, Xte = tp.standardize(XT[gpu(tr)].double(), XT[gpu(te)].double())  # fit_a2's line, verbatim in effect
            Xr = XT[gpu(tr)].double()  # fit_a2's mu_ / sd_ (the held-out run 2 standardization)
            mu_, sd_ = Xr.mean(0), Xr.std(0, unbiased=False)
            sd_ = sd_.where(sd_ > 0, sd_.new_ones(()))
            del Xr
            Wn, bn, mn, sn = sw[p][f]
            std_diff[f"{p}|{f}"] = [float((mu_ - gpu(mn)).abs().max()), float((sd_ - gpu(sn)).abs().max())]  # diagnostic: 0 if 4f-exact
            W, b = gpu(Wn).reshape(-1, 1), gpu(bn).reshape(-1)
            S = (Xte @ W + b).cpu().numpy()[:, 0]
            if not np.isfinite(S).all():
                raise ValueError(f"{cell}: non-finite replayed scores")
            base_rows[p][d["rpos"][te]] = S
            new_rows[p][d["rpos"][te]] = S
            r2 = np.array([j for j, r in enumerate(runs) if r["batch"] == "run2" and r["fold"] == f], np.int64) if p == "c_simatk_vs_run1" else None
            if r2 is not None:
                rows2 = np.flatnonzero(np.isin(row_run, r2))
                S2 = (((XT[gpu(rows2)].double() - mu_) / sd_) @ W[:, 0] + b.reshape(-1)[0]).cpu().numpy()
                r2_base[f] = (rows2, S2)
                r2_new[f] = (rows2, S2)
            if (p, f) in targets:
                if std_diff[f"{p}|{f}"] != [0.0, 0.0]:  # (Codex b1 P2) the refit must use exactly the stored scalers it saves
                    raise ValueError(f"G4 STOP {cell}|{p}|fold {f}: recomputed mean/std differ from the stored ones {std_diff[f'{p}|{f}']}")
                old = [c for c in rec_cell["certificates"] if c.get("priority") == "P1" and c["problem"] == p and c["fold"] == f]
                if len(old) != 1 or old[0]["certified_per_column"][0]:
                    raise ValueError(f"{cell}|{p}|fold {f}: not exactly one uncertified observed certificate in the record")
                certs_old.append(old[0])
                Ytr = gpu(d["y"][d["pos"][row_run[tr]]].astype(np.float64))
                c0 = tp.certify(Xtr, Ytr, W, b, ra.C0)
                if not (np.isfinite(c0["max_grad"]) and abs(c0["max_grad"] - old[0]["max_grad"]) <= REL * old[0]["max_grad"]):
                    raise ValueError(f"G4 STOP {cell}|{p}|fold {f}: stored weights' max |grad| {c0['max_grad']!r} is not the record's {old[0]['max_grad']!r}")
                ts = time.time()
                W1, b1, cert = fit_newton(Xtr, Ytr, ra.C0, W, b)
                acc = bool(np.isfinite(cert["max_grad"]) and cert["accepted"])  # (0d's rule) G5: no raise; a rejected fit is reported, never used
                S1 = (Xte @ W1 + b1).cpu().numpy()[:, 0]
                if not np.isfinite(S1).all():
                    raise ValueError(f"{cell}: non-finite refit scores")
                new_rows[p][d["rpos"][te]] = S1
                if r2 is not None:
                    r2_new[f] = (r2_new[f][0], (((XT[gpu(r2_new[f][0])].double() - mu_) / sd_) @ W1[:, 0] + b1.reshape(-1)[0]).cpu().numpy())
                w1, bb1 = W1[:, 0].cpu().numpy().astype(np.float64), b1.reshape(-1)[:1].cpu().numpy().astype(np.float64)
                certs_new.append({"priority": "P1", "problem": p, "C": ra.C0, "fold": f, "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                                  "n_train_runs": int(len(tr_runs)), "n_test_runs": int(len(te_runs)), "n_newton_steps": int(cert["n_steps"]), "columns": [0],
                                  "max_grad_per_column": [float(cert["max_grad"])], "certified_per_column": [acc], "max_grad": float(cert["max_grad"]),
                                  "uncertified_columns": [] if acc else [0], "accepted": acc, "seconds": round(time.time() - ts, 2), "start": cert["start"],
                                  "gpu": torch.cuda.get_device_name(0), "newton_log": cert["newton_log"], "chol_rel_residual_max": cert["chol_rel_residual_max"],
                                  "hessian_condition_first_step": cert["hessian_condition_first_step"], "rule": cert["rule"],
                                  "columns_note": "column 0 = observed labels; refit3: post hoc numerical repair, exact Newton from the stored solution",
                                  "stored_solution_max_grad_recomputed": c0["max_grad"],
                                  "max_abs_dcoef": float(max(np.abs(w1 - Wn).max(), np.abs(bb1 - bn).max())),
                                  "max_abs_dscore_test_rows": float(np.abs(S1 - S).max())})
                buf = io.BytesIO()
                np.savez(buf, **{f"f{f}__{k}": v for k, v in zip(("W", "b", "mean", "std"), (w1, bb1, mn, sn))})
                rel = f"xfer/attack_refit3/{stamp}/w_{ro}_L{L}__{p}__fold{f}.npz"
                Path("/w3", rel).parent.mkdir(parents=True, exist_ok=True)
                Path("/w3", rel).open("xb").write(buf.getvalue())
                wnew[f"{p}|{f}"] = {"volume": W3VOL, "path": rel, "sha256": hashlib.sha256(buf.getvalue()).hexdigest(), "keys": f"f{f}__W/b/mean/std"}
            del Xtr, Xte
            torch.cuda.empty_cache()
    if set(targets) != {(c["problem"], c["fold"]) for c in certs_new}:
        raise ValueError(f"{cell}: not every target was refit")
    w3_vol.commit()
    return {"cell": cell, "keep": keep, "row_run": row_run, "base_rows": base_rows, "new_rows": new_rows, "r2_base": r2_base, "r2_new": r2_new,
            "std_diff_vs_stored": std_diff, "certs_old": certs_old, "certs_new": certs_new, "weights_new": wnew, "gpu": torch.cuda.get_device_name(0),
            "timing": {"load_seconds": round(t_load), "seconds_total": round(time.time() - t0)}}


def _gate_cell(cell: str, epochs: list, rec_cell: dict, rep: dict, guard) -> dict:
    """G4a / G4b on the replayed rows (strict 1e-9; on failure EVERY failing leaf is printed and returned in the error), then the refit
    outputs, by 4f's own metric code (null draws reused; only their summaries recomputed)."""
    if not guard.put(f"gate|{cell}", time.time(), skip_if_exists=True):  # (Codex b1 P2) the cost guard covers the CPU stage too
        raise RuntimeError(f"{cell}: a rescheduled gate attempt; refused by the cost guard")
    import numpy as np

    runs, row_run = [epochs[i] for i in rep["keep"]], np.asarray(rep["row_run"])
    prep = prep_meta(runs, row_run, np)

    def runmeans(d, rows):
        loc = d["pos"][row_run[d["rows"]]]
        tot, cnt = np.zeros(len(d["ids"])), np.bincount(loc, minlength=len(d["ids"]))
        np.add.at(tot, loc, rows)
        if (cnt == 0).any():
            raise ValueError("a run without rows")
        return tot / cnt

    def r2_scores(r2):
        h2, hr2 = {}, {}
        for f in range(5):
            rows2, S2 = r2[f]
            rr2 = row_run[np.asarray(rows2)]
            for j in sorted(set(rr2.tolist())):
                m_ = rr2 == j
                h2[runs[j]["name"]], hr2[runs[j]["name"]] = float(np.asarray(S2)[m_].mean()), np.asarray(S2)[m_].tolist()
        return h2, hr2

    for p, d in prep.items():
        d["oof_row"] = np.asarray(rep["base_rows"][p], np.float64)
        if not np.isfinite(d["oof_row"]).all():
            raise ValueError(f"{cell}|{p}: a row without a replayed score")
        d["oof"] = runmeans(d, d["oof_row"])
    h2, hr2 = r2_scores(rep["r2_base"])
    rec_rs = {p: rec_cell["problems"][p]["run_scores"] for p in prep}
    g4a = cmp({p: {runs[j]["name"]: float(d["oof"][i]) for i, j in enumerate(d["ids"])} for p, d in prep.items()} | {"heldout": h2},
              rec_rs | {"heldout": rec_cell["heldout_run2"]["run_scores"]})
    exact_runs = {p: int(sum(float(d["oof"][i]) == rec_rs[p][runs[j]["name"]] for i, j in enumerate(d["ids"]))) for p, d in prep.items()}
    if g4a:
        for x in g4a:
            print("G4a FAIL", x, flush=True)
        raise ValueError(f"G4 STOP {cell}: {len(g4a)} run-score leaves differ (all printed above): {g4a[:20]}")
    for p, d in prep.items():  # byte identity: the record's own run-level values
        d["oof"] = np.array([rec_rs[p][runs[j]["name"]] for j in d["ids"]], np.float64)
    h2 = dict(rec_cell["heldout_run2"]["run_scores"])
    hit = {p for p, _ in TARGETS[cell]}
    has_c = "c_simatk_vs_run1" in hit

    def outputs() -> dict:
        o = {"problems": {}}
        for p in sorted(hit):
            d = prep[p]
            rrun = d["pos"][row_run[d["rows"]]]
            r = {"run_level": a1._metrics(d["oof"], d["y"], d["fold"]), "call_level": a1._metrics(d["oof_row"], d["y"][rrun], d["fold"][rrun])}
            for lvl in ("run_level", "call_level"):
                for a in ("0.01", "0.05"):
                    r[lvl][f"catch_{a}"]["fpr_note"] = ra.FPR_NOTE
            r["run_scores"] = {runs[j]["name"]: float(d["oof"][i]) for i, j in enumerate(d["ids"])}
            r["run_auroc_boot"] = ra.boot_pairs(d["oof"], d["y"], [runs[j]["task"] for j in d["ids"]])
            r["call_auroc_boot"] = ra.boot_pairs(d["oof_row"], d["y"][rrun], [runs[d["ids"][i]]["task"] for i in rrun])
            nul = copy.deepcopy(rec_cell["problems"][p]["null"])
            for lvl in ("run_level", "call_level"):
                for a in (0.01, 0.05):
                    nul[f"{lvl}_catch_{a:g}_summary"] = a1._summ(np.array(nul[f"{lvl}_catch_{a:g}"]), r[lvl][f"catch_{a:g}"]["tpr"])
                nul[f"{lvl}_auroc_summary"] = a1._summ(np.array(nul[f"{lvl}_auroc"]), r[lvl]["auroc"])
            r["null"] = nul
            o["problems"][p] = r
        pp = {p: {"ids": d["ids"], "pos": d["pos"], "rows": d["rows"], "Y": d["y"][None, :], "fold": d["fold"], "oof": d["oof"][None, :],
                  "oof_row": d["oof_row"][None, :]} for p, d in prep.items()}
        o["paired_c_minus_d"] = ra._paired(pp, runs, row_run)
        if has_c:
            o["heldout_run2"] = heldout_block(np, prep, runs, row_run, h2, hr2)
        return o

    ref = {"problems": {p: {k: rec_cell["problems"][p][k] for k in FIELDS} for p in sorted(hit)}, "paired_c_minus_d": rec_cell["paired_c_minus_d"]}
    ref |= {"heldout_run2": rec_cell["heldout_run2"]} if has_c else {}
    g4b = cmp(outputs(), ref)
    if g4b:
        for x in g4b:
            print("G4b FAIL", x, flush=True)
        raise ValueError(f"G4 STOP {cell}: {len(g4b)} replaced output leaves differ from the record (ALL printed above): {g4b}")
    stored = {p: (d["oof"].copy(), d["oof_row"].copy()) for p, d in prep.items()}
    for p, f in TARGETS[cell]:  # the refit folds' new rows and run means; the (c) model's held-out run 2 rows of that fold
        d = prep[p]
        te_runs = d["ids"][d["fold"] == f]
        te = np.flatnonzero(np.isin(row_run, te_runs))
        d["oof_row"][d["rpos"][te]] = np.asarray(rep["new_rows"][p], np.float64)[d["rpos"][te]]
        d["oof"][d["pos"][te_runs]] = runmeans(d, d["oof_row"])[d["pos"][te_runs]]
        if p == "c_simatk_vs_run1":
            h_, hr_ = r2_scores({ff: (rep["r2_new"][ff] if ff == f else rep["r2_base"][ff]) for ff in range(5)})
            for n, v in h_.items():
                if runs[[r["name"] for r in runs].index(n)]["fold"] == f:
                    h2[n], hr2[n] = v, hr_[n]
    new = outputs()
    changed = {p: {"runs": int(np.sum(d["oof"] != stored[p][0])), "rows": int(np.sum(d["oof_row"] != stored[p][1]))} for p, d in prep.items()}
    tdel = {}  # (0d) per refit fold: max |delta run score| on that fold's held-out runs
    for p, f in TARGETS[cell]:
        ix = prep[p]["pos"][prep[p]["ids"][prep[p]["fold"] == f]]
        tdel[f"{p}|{f}"] = {"max_abs_drun_score_fold_test_runs": float(np.abs(prep[p]["oof"][ix] - stored[p][0][ix]).max()), "n_runs": int(len(ix))}
    eff = {p: {str(f): rep["weights_new"].get(f"{p}|{f}") or {"volume": ra.SOLVOL, "path": rec_cell["saved_weights"][p]["path"],
                                                              "sha256": rec_cell["saved_weights"][p]["sha256"], "keys": f"f{f}__W/b/mean/std"} for f in range(5)}
           for p in sorted(hit)}
    return {"cell": cell, "gates": {"G3": "pass", "G4a": "pass (run scores, held-out run 2 run scores; finite, <= 1e-9)", "exact_run_scores": exact_runs,
                                    "G4b": "pass (every replaced output leaf; finite, <= 1e-9)", "G5_accepted": {f"{x['problem']}|{x['fold']}": x["accepted"] for x in rep["certs_new"]},
                                    "gpu": rep["gpu"],
                                    "std_diff_vs_stored": rep["std_diff_vs_stored"]}, "certs_old": rep["certs_old"], "certs_new": rep["certs_new"],
            "weights_new": rep["weights_new"], "effective_weights": eff, "outputs": new, "changed": changed, "timing": rep["timing"],
            "target_deltas": tdel}


# (Codex b2 P2) the polled stages: a worker's own TimeoutError is re-raised as RuntimeError, so the driver's poll (get(timeout=5) ->
# TimeoutError) can never mistake a finished failure for a pending result
@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(49152, 49152), timeout=3600, retries=0, startup_timeout=ra.START_GPU, scaledown_window=ra.IDLE_S,
              volumes={"/u": ra.u_vol, "/w4f": w4f_vol, "/w3": w3_vol})
def replay_cell(cell: str, epochs: list, skip: dict, allowed: dict, rec_cell: dict, stamp: str, guard) -> dict:
    try:
        return _replay_cell(cell, epochs, skip, allowed, rec_cell, stamp, guard)
    except TimeoutError as e:
        raise RuntimeError(f"worker TimeoutError: {e}") from e


@app.function(image=img, cpu=(4, 4), memory=(16384, 16384), timeout=3600, retries=0, scaledown_window=ra.IDLE_S)
def gate_cell(cell: str, epochs: list, rec_cell: dict, rep: dict, guard) -> dict:
    try:
        return _gate_cell(cell, epochs, rec_cell, rep, guard)
    except TimeoutError as e:
        raise RuntimeError(f"worker TimeoutError: {e}") from e


@app.function(image=img, cpu=(2, 2), memory=(16384, 16384), timeout=900, retries=0)
def patch_r3(raw: bytes, res: dict, epochs: list, expected: dict, accepted: bool) -> dict:
    """(0d's rule) accepted = every Newton refit passed; else NOTHING is replaced and only the sensitivity (deltas, certificates) returns.
    The new record: 4f's with the affected cells' outputs replaced, the superseded certificates kept, the effective weights manifest,
    the uncut gates recomputed (they must now pass), and every changed key path listed and checked (the invariance check)."""
    rec = json.loads(raw)
    new = copy.deepcopy(rec)
    deltas = {}
    num = lambda v, o: [x - y for x, y in zip(v, o)] if isinstance(v, list) else v - o  # noqa: E731
    for cell, r in res.items():
        c, o = new["cells"][cell], r["outputs"]
        olds = {(x["problem"], x["fold"]) for x in r["certs_old"]}
        keep_ = [x for x in c["certificates"] if not (x.get("priority") == "P1" and (x.get("problem"), x.get("fold")) in olds)]
        c["certificates"] = keep_ + r["certs_new"]
        c["refit3"] = {"superseded_certificates": r["certs_old"], "weights_new": r["weights_new"], "gates": r["gates"], "timing": r["timing"],
                       "effective_weights": r["effective_weights"], "changed_runs_rows": r["changed"],
                       "effective_weights_rule": "per problem and fold, the npz to score with: refit folds from this job's Volume, the rest 4f's saved_weights"}
        for p, v in o["problems"].items():
            q, old = c["problems"][p], copy.deepcopy(c["problems"][p])
            for k in FIELDS:
                q[k] = v[k]
            unc = sorted({(x["fold"], u) for x in c["certificates"] if x.get("problem") == p for u in x.get("uncertified_columns", [])})
            q["observed_certified"], q["uncertified_fold_columns"] = not any(u == 0 for _, u in unc), [list(x) for x in unc]
            deltas[f"{cell}|{p}"] = {f"{lvl}|{m}": num(*(x[lvl][m] if m == "auroc" else x[lvl][m]["tpr"] for x in (v, old)))
                                     for lvl in ("run_level", "call_level") for m in ("auroc", "catch_0.01", "catch_0.05")} | {
                f"{b}|ci95": num(v[b]["ci95"], old[b]["ci95"]) for b in ("run_auroc_boot", "call_auroc_boot")}
        po, pn = c["paired_c_minus_d"]["observed"], o["paired_c_minus_d"]["observed"]
        deltas[f"{cell}|paired"] = {m: {"c_minus_d": pn[m]["c_minus_d"] - po[m]["c_minus_d"], "ci95": num(pn[m]["ci95"], po[m]["ci95"])} for m in po}
        c["paired_c_minus_d"] = o["paired_c_minus_d"]
        if "heldout_run2" in o:
            ho, hn = c["heldout_run2"]["contrasts"], o["heldout_run2"]["contrasts"]
            deltas[f"{cell}|heldout_run2"] = {k: {"auroc": hn[k]["auroc"] - ho[k]["auroc"], "ci95": num(hn[k]["boot"]["ci95"], ho[k]["boot"]["ci95"]),
                                                  "call_auroc": hn[k]["call_level"]["auroc"] - ho[k]["call_level"]["auroc"],
                                                  "call_ci95": num(hn[k]["call_level"]["boot"]["ci95"], ho[k]["call_level"]["boot"]["ci95"])} for k in ho}
            c["heldout_run2"] = o["heldout_run2"]
    targets_report = {cell: {"certificates": r["certs_new"], "target_deltas": r["target_deltas"]} for cell, r in res.items()}
    if not accepted:
        return {"label": "refit3 SENSITIVITY ONLY (0d's rule): a Newton refit failed the acceptance rule, so nothing is replaced; the record stands as "
                         "'uncertified (3 of 40 fits; max |grad| 1.02-1.32e-6 vs the 1e-6 bar)' and these Newton deltas are a sensitivity check",
                "accepted": False, "deltas": deltas, "targets": targets_report}
    new["uncut_gates"] = ui.check_cells(new["cells"], epochs, expected)
    if new["uncut_gates"]["failures"]:  # 4f's record failed only on these 3 observed fits (all null draws certified): now none may fail
        raise ValueError(f"GATE STOP: uncut gates still fail: {new['uncut_gates']['failures'][:5]}")

    def paths(a, b, pre=""):
        if isinstance(a, dict) and isinstance(b, dict):
            return [x for k in sorted(set(a) | set(b)) for x in (paths(a[k], b[k], f"{pre}/{k}") if k in a and k in b else [f"{pre}/{k}"])]
        return [] if a == b else [pre]
    changed = paths(rec, new)
    for cell, r in res.items():  # (review r1) non-target certificates and every null draw array untouched
        oc = [x for x in rec["cells"][cell]["certificates"] if not (x.get("priority") == "P1" and (x.get("problem"), x.get("fold")) in
                                                                   {(y["problem"], y["fold"]) for y in r["certs_old"]})]
        if oc != new["cells"][cell]["certificates"][:len(oc)]:
            raise ValueError(f"INVARIANCE STOP {cell}: a non-target certificate changed")
        for p in r["outputs"]["problems"]:
            a_, b_ = rec["cells"][cell]["problems"][p]["null"], new["cells"][cell]["problems"][p]["null"]
            if {k: v for k, v in a_.items() if not k.endswith("_summary")} != {k: v for k, v in b_.items() if not k.endswith("_summary")}:
                raise ValueError(f"INVARIANCE STOP {cell}|{p}: a null draw array changed")
        for p in [*r["outputs"]["problems"], "heldout"]:  # changed run scores must be test runs of the refit folds only
            pre = f"/cells/{cell}/problems/{p}/run_scores/" if p != "heldout" else f"/cells/{cell}/heldout_run2/run_scores/"
            tf = {f for q, f in TARGETS[cell] if q == (p if p != "heldout" else "c_simatk_vs_run1")}
            bad = [x for x in changed if x.startswith(pre) and rec["run_folds"][x[len(pre):]] not in tf]
            if bad:
                raise ValueError(f"INVARIANCE STOP: run scores changed outside the refit folds: {bad[:5]}")
    allowed = tuple(f"/cells/{cell}/{k}" for cell in res for k in ("certificates", "refit3", "paired_c_minus_d")) + tuple(
        f"/cells/{cell}/heldout_run2" for cell, r in res.items() if "heldout_run2" in r["outputs"]) + tuple(
        f"/cells/{cell}/problems/{p}/{k}" for cell, r in res.items() for p in r["outputs"]["problems"]
        for k in (*FIELDS, "observed_certified", "uncertified_fold_columns")) + ("/uncut_gates",)
    stray = [x for x in changed if not x.startswith(allowed)]
    if stray:
        raise ValueError(f"INVARIANCE STOP: changes outside the refit's fields: {stray[:10]}")
    rep_g = [x["max_grad"] for r in res.values() for x in r["certs_new"]]  # (Codex n1 P2) the precision claim only if the numbers show it
    tset = {(cell, p, f) for cell, ts in TARGETS.items() for p, f in ts}
    oth_g = [x["max_grad"] for cell, c in rec["cells"].items() for x in c["certificates"] if x.get("priority") == "P1" and (cell, x["problem"], x["fold"]) not in tset]
    precise = len(oth_g) == 37 and max(rep_g) < min(oth_g)
    new["refit3"] = {"label": "Post hoc numerical repair of 3 of 40 fits (exact Newton from the stored solution); all 40 meet the common 1e-6 "
                              "criterion" + ("; the 3 repaired fits are more precise than the other 37" if precise else "") + ". The null fits stay as "
                              "recorded (4f's record certifies every null draw). PRELIMINARY until 0d's recheck",
                     "precision": {"repaired_max_grad": rep_g, "other_37_max_grad_min": min(oth_g), "other_37_max_grad_max": max(oth_g),
                                   "repaired_all_below_other_37": precise}, "accepted": True, "targets_report": targets_report, "targets": {k: [list(t) for t in v] for k, v in TARGETS.items()},
                     "deltas": deltas, "changed_paths": changed,
                     "invariance": "checked: no change outside the listed fields of the affected cells; non-target certificates and null draw arrays "
                                   "identical; changed run scores only in the refit folds (this path list excludes only the top-level /refit3 audit block)"}
    return new


@app.local_entrypoint()
def main_r3(stage: str = "dry"):
    if stage not in ("dry", "skip", "fit"):
        raise SystemExit("--stage dry|skip|fit")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no") or not git("branch", "-r", "--contains", "HEAD") or \
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", "run_refit3.py"], capture_output=True, cwd=HERE).returncode:
        raise SystemExit("commit and push first (this script tracked, the tree clean)")
    (HERE / "results").mkdir(exist_ok=True)
    src = json.loads((HERE / SOURCES).read_text())
    data = {}
    for f in src["files"]:
        b = git("show", f"{f['commit']}:{f['path']}")
        if hashlib.sha256(b).hexdigest() != f["sha256"]:
            raise SystemExit(f"{f['path']}: sha256 differs from {SOURCES}")
        data[f"{f['commit']}:{f['path']}"] = b
    (raw,) = [data[f"{f['commit']}:{f['path']}"] for f in src["files"] if f["role"] == "record"]
    u = prep_r3.remote(data)
    print(json.dumps({"G1": "pass", "epochs": len(u["epochs"])}), flush=True)
    if stage == "dry":
        return
    rec = json.loads(raw)
    sk = skip_r3.remote(u["want"], u["shas"], {e["key"]: e["full_len"] for e in u["epochs"]}, {e["staged_inputs_run"]: e["extraction_run"] for e in u["epochs"]})
    if sk["dropped_rows"] != rec["skip"]["dropped_rows"] or sk["staged_sha256"] != rec["skip"]["staged_sha256"] or sk["meta_checked"] != len(u["epochs"]):
        raise SystemExit("G2 STOP: the skip list does not reproduce the record's")
    print(json.dumps({"G2": "pass", "meta_checked": sk["meta_checked"]}), flush=True)
    if stage == "skip":
        return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    calls, t_sp, got = {}, {}, {}

    def spawn(name, fn, *args):  # (Codex b1 P1) every handle registered the moment it exists
        calls[name], t_sp[name] = fn.spawn(*args), time.time()

    def wait_all(names: list, deadline_s: float) -> dict:  # (Codex b1 P1) all calls polled together: any failure surfaces at once
        while not all(n in got for n in names):
            for n in names:
                if n in got:
                    continue
                try:
                    got[n] = calls[n].get(timeout=5)
                except TimeoutError:
                    if time.time() - t_sp[n] > deadline_s:
                        raise TimeoutError(f"{n}: past its {deadline_s:.0f} s deadline from spawn") from None
        return {n.split("|", 1)[1]: got[n] for n in names}

    with modal.Dict.ephemeral() as guard:  # 4f's cost guard, scoped to this invocation
        try:  # (launch 2) a failed stage cancels every outstanding call before the guard closes
            for cell in TARGETS:
                spawn(f"replay|{cell}", replay_cell, cell, u["epochs"], sk["skip"], u["allowed"], rec["cells"][cell], stamp, guard)
            reps = wait_all([f"replay|{c}" for c in TARGETS], 3700)
            print(json.dumps({"replay": "done", "std_diff_vs_stored_max": {c: max(max(v) for v in r["std_diff_vs_stored"].values()) for c, r in reps.items()},
                              "timing": {c: r["timing"] for c, r in reps.items()}}), flush=True)
            for cell in TARGETS:
                spawn(f"gate|{cell}", gate_cell, cell, u["epochs"], rec["cells"][cell], reps[cell], guard)
            res = wait_all([f"gate|{c}" for c in TARGETS], 3700)
        except BaseException:
            for name, c in calls.items():
                if name in got:
                    continue
                try:
                    c.cancel(terminate_containers=True)
                    print(f"cancelled {name}", flush=True)
                except Exception as e:  # noqa: BLE001 - reported; the function timeouts still bound the containers
                    print(f"cancel {name} failed: {type(e).__name__}", flush=True)
            raise
    accepted = all(x["accepted"] for r in res.values() for x in r["certs_new"])  # (0d's rule, fixed before any downstream score was read)
    new = patch_r3.remote(raw, res, u["epochs"], sk["expected_rows"], accepted)
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp, "app": base.APP}
    if accepted:
        dst = HERE / "results" / f"attack_all_refit3_{stamp}.json"
        txt = json.dumps(new | {"refit3_run": meta}, indent=1, allow_nan=False)  # (Codex n1 P1) serialized before the file is opened
        dst.open("x").write(txt)
        summ = {"deltas": new["refit3"]["deltas"], "uncut_gates_failures": new["uncut_gates"]["failures"]}
    else:
        dst = HERE / "results" / f"refit3_sensitivity_{stamp}.json"
        txt = json.dumps(new | {"refit3_run": meta}, indent=1, allow_nan=False)  # (Codex n1 P1) serialized before the file is opened
        dst.open("x").write(txt)
        summ = {"deltas": new["deltas"]}
    print(json.dumps({"record": dst.name, "accepted": accepted} | summ | {"new_certificates": {
        c: [(x["problem"], x["fold"], x["max_grad"], x["n_newton_steps"], x["chol_rel_residual_max"], x["accepted"], x["max_abs_dcoef"],
             x["max_abs_dscore_test_rows"]) for x in r["certs_new"]] for c, r in res.items()},
        "target_deltas": {c: r["target_deltas"] for c, r in res.items()}}, allow_nan=False), flush=True)
