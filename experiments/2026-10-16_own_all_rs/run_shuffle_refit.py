"""Exact-equality diagnosis for own_all_rs_uncut's descriptive cut-32768 refit (4f / 09 / 0d, 10-16; DESCRIPTIVE, not a gate): that refit fit the uncut
rows below token 32768 (the same rows as own_all's registered rows as a multiset, in another order: set-counts NOTE @ 11250f14d) and matched the registered
own_all_rs_L48 weights with max abs diff 0.0. Fold model 0 of own_all_rs_L48 (100%), R and C, all on ONE code path (run_own_all._fit_one verbatim +
the image's torch_probe_pinned.fit), every fit logged (n_evals, max_grad, certified, seconds): (A) the gate refit on the registered rows; (B) the
descriptive refit on the uncut rows < 32768, recomputed; (C) one training row dropped (the first by sorted unit, call_index, row); (D) that row's label
flipped; (E) the registered rows in a shuffled order (seed "own_all_rs|order|f0"); (F) whether (B)'s rows, gathered in fit order, are bitwise equal to the
registered rows in theirs (row hashes of X + label in sequence). Each fit is diffed against the registered weights FILE (sha-checked) and against (A).
Weights / diffs / row hashes only, no scores. One H100 job."""

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
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_ownall-shuffle_kill-2026-11-14"
BUILD = Path("/tmp/famexp_build")
L, C_REG = 48, 0.01  # run_own_all.py: layer 48, own_a's C
ROWS_STAMP = "2026-10-15_173309"
ROWS_SHA = {("T196", "R"): "8b375c6174a12740165cd483a6cfa1be58c9cec6caed95387bdc82c48ba53fcb", ("T196", "C"): "26edc3300421ee915f08d9f0d8f3cff0488d5c66c596286c016df2e6aff478f7",
            ("scaleup", "R"): "ec50594b39d6f63e83df0918c9773e25c9c60921975248610fcf11bb1a1ebd24", ("scaleup", "C"): "09cbe4995372aa100ca682d53d8ac3c481756a3bf42c311f9001f63b9a56e308"}
RS_FIT_GIT = ("experiments/2026-10-14_p2_family_expansion/results/own_all_rs_fit_2026-10-16_003049.json", "d6080ea3d83aa4ef3b59efd87abd40671c02b4522b2b919b4fb8ab7b3dc18faf")
INPUTS = "experiments/2026-10-14_p2_family_expansion/results/own_all_inputs_2026-10-15.json"
UNCUT_FIT_GIT = ("experiments/2026-10-14_p2_family_expansion/results/own_all_rs_uncut_fit_2026-10-16_030614.json", "f62f4d660109fff5d3d207a024299f3c254fbfb2c603ddc7fc62b40b38760442")
RS_FOLDS = "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json"
if modal.is_local():  # run_famexp's import materializes the pinned 2e row code and torch_probe into BUILD, refusing any byte difference (as run_own_all.py)
    sys.path.insert(0, str(REPO / "experiments" / "2026-10-14_p2_family_expansion"))
    import run_famexp  # noqa: F401
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2)
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "safetensors==0.8.0").add_local_dir(BUILD, "/root/x2e")
app = modal.App(APP)


def _fit_one(tp, X, y, tr, st, dev):
    """run_own_all._fit_one (= run_famexp._fit_one @ b5a0586fa), verbatim."""
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


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(98304, 98304), timeout=2700, retries=0, scaledown_window=2, volumes={"/out": OUT.read_only()})
def shuffle(rs_fit_sha: str, uncut: dict, main_of: dict, fold: dict, k1: list) -> dict:
    """4f's (A)-(F) for fold model 0 of own_all_rs_L48, R and C, on ONE code path (_fit_one + tp.fit), every fit logged (n_evals, max_grad, certified, seconds)."""
    import random

    import numpy as np
    import torch
    from safetensors.numpy import load_file

    sys.path.insert(0, "/root/x2e")
    import torch_probe_pinned as tp

    tp_src = Path("/root/x2e/torch_probe_pinned.py").read_bytes()
    dev, k, k1 = "cuda", 0, set(k1)
    rs_d, un_d = Path("/out/own_all_rs/2026-10-16_003049"), Path("/out/own_all_rs_uncut") / uncut["stamp"]
    fb = (rs_d / "fit_record.json").read_bytes()
    if hashlib.sha256(fb).hexdigest() != rs_fit_sha:
        raise SystemExit("own_all_rs's Volume fit record differs from the git record's")
    if hashlib.sha256((un_d / "fit_record.json").read_bytes()).hexdigest() != uncut["fit_record_sha256"]:
        raise SystemExit("the uncut Volume fit record differs from the git record's")
    reg = {(f["variant"], f["readout"], f["fold"]): f for f in json.loads(fb)["fits"]}

    def _arr(zs):
        names = np.concatenate([z["names"] for z in zs])
        ci = np.concatenate([z["call_index"] for z in zs])
        cells = np.array([u.split("|", 1)[1] for u in names])
        mains = np.array([main_of[c] for c in cells])
        world = np.array([u.split("|", 1)[0] for u in names])
        y = np.where(world == "run1", 1.0, np.where(world == "honest_a", 0.0, np.nan))
        full = np.array([m in k1 for m in mains]) & (np.array([fold[m] for m in mains]) != k) & ~np.isnan(y)
        return names, ci, y, full

    def _hashes(X, y, idx):  # per training row in fit order: sha256 of the float32 X row bytes + the label
        xs = X[idx]
        return [hashlib.sha256(xs[i].tobytes() + str(y[idx[i]]).encode()).hexdigest() for i in range(len(idx))]

    out = {}
    for rd in ("R", "C"):
        zs = []
        for p in ("T196", "scaleup"):
            b = (Path("/out/own_all") / ROWS_STAMP / "rows" / f"{p}__{rd}.npz").read_bytes()
            if hashlib.sha256(b).hexdigest() != ROWS_SHA[(p, rd)]:
                raise SystemExit(f"{p}__{rd}: rows differ from the pin")
            zs.append(np.load(io.BytesIO(b)))
        names, ci, y, full = _arr(zs)
        Xn = np.concatenate([z["X"] for z in zs])
        tr = np.flatnonzero(full)  # 100%: the subsample keeps every cell of the fold's pool
        f = reg[("own_all_rs_L48", rd, k)]
        wp = rs_d / "weights" / f"own_all_rs_L48__{rd}__L48__f{k}.safetensors"
        if len(tr) != f["rows"] or hashlib.sha256(wp.read_bytes()).hexdigest() != f["weights_sha256"]:
            raise SystemExit(f"{rd}: the fold-0 rows or the registered weights differ from the fit record")
        w_reg = {n: np.asarray(v, np.float64) for n, v in load_file(str(wp)).items()}
        first = min(tr.tolist(), key=lambda i: (str(names[i]), int(ci[i]), i))  # the stated rule: the first training row by sorted (unit, call_index, row)
        sh = tr.tolist()
        random.Random(f"own_all_rs|order|f{k}").shuffle(sh)
        y_flip = y.copy()
        y_flip[first] = 1.0 - y_flip[first]
        # the uncut rows below 32768 in fit_uncut's order (the fit record's rs_uncut.rows dict order)
        uz = []
        for nm, h in uncut["rows"][rd].items():
            b = (un_d / "rows" / f"{nm}__{rd}.npz").read_bytes()
            if hashlib.sha256(b).hexdigest() != h:
                raise SystemExit(f"uncut rows {nm}__{rd} differ from the fit record")
            uz.append(np.load(io.BytesIO(b)))
        un, uci, uy, ufull = _arr(uz)
        utok = np.concatenate([z["token"] for z in uz])
        Xu = np.concatenate([z["X"] for z in uz])
        trc = np.flatnonzero(ufull & (utok < 32768))
        hr, hu = _hashes(Xn, y, tr), _hashes(Xu, uy, trc)
        cases = {"A_gate_registered_rows": (Xn, y, tr), "B_descriptive_uncut_below_32768": (Xu, uy, trc), "C_one_row_dropped": (Xn, y, tr[tr != first]),
                 "D_one_label_flipped": (Xn, y_flip, tr), "E_shuffled_order": (Xn, y, np.array(sh))}
        res, arrs = {}, {}
        for nm, (Xh, yy, idx) in cases.items():
            X = torch.from_numpy(Xh).to(dev)
            m_, cert, secs = _fit_one(tp, X, yy, idx, idx, dev)
            del X
            torch.cuda.empty_cache()
            arrs[nm] = {n: t_.cpu().numpy().astype(np.float64) for n, t_ in zip(("W", "b", "mean", "std"), m_, strict=True)}
            res[nm] = {"rows": int(len(idx)), "n_evals": int(cert["n_evals"]), "max_grad": float(cert["max_grad"]), "certified": bool(cert["certified"]), "seconds": secs,
                       "max_abs_diff_vs_registered_file": {n: float(np.abs(w_reg[n] - a).max()) for n, a in arrs[nm].items()}}
        for nm in cases:
            res[nm]["max_abs_dW_vs_A"] = float(np.abs(arrs[nm]["W"] - arrs["A_gate_registered_rows"]["W"]).max())
        out[rd] = {"fits": res, "first_training_row_dropped_or_flipped": {"unit": str(names[first]), "call_index": int(ci[first]), "row": int(first)},
                   "shuffle_seed": f"own_all_rs|order|f{k}",
                   "F_rows_in_fit_order": {"registered_training_rows": len(hr), "uncut_training_rows_below_32768": len(hu), "bitwise_equal_in_sequence": hr == hu,
                                           "first_sequence_mismatch": next((i for i, (a, b_) in enumerate(zip(hr, hu, strict=False)) if a != b_), None),
                                           "bitwise_equal_as_multiset": sorted(hr) == sorted(hu), "x_dtype": [str(Xn.dtype), str(Xu.dtype)]}}
        del Xn, Xu
    return {"readouts": out, "torch_probe_pinned_sha256": hashlib.sha256(tp_src).hexdigest(),
            "torch_probe_fit_scan": {"sort": b"sort" in tp_src, "argsort": b"argsort" in tp_src, "cache": b"cache" in tp_src.lower(), "round": b"round(" in tp_src}}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/own-all-loo"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    fb = git("show", f"HEAD:{RS_FIT_GIT[0]}")
    if hashlib.sha256(fb).hexdigest() != RS_FIT_GIT[1]:
        raise SystemExit("the own_all_rs git fit record differs from its pin")
    fr = json.loads(fb)
    inp = json.loads(git("show", f"HEAD:{INPUTS}"))
    main_of = {}
    for c_, p_, s_ in (inp["pins"]["t196"], inp["pins"]["plus311"], inp["pins"]["nm360"]):
        b = git("show", f"{c_}:{p_}")
        if hashlib.sha256(b).hexdigest() != s_:
            raise SystemExit(f"{p_}: sha256 differs")
        main_of |= {c["key"]: c["main"] for c in json.loads(b)["cells"]}
    fl = git("show", f"HEAD:{RS_FOLDS}")
    if hashlib.sha256(fl).hexdigest() != fr["rs"]["fold_list"]["sha256"]:
        raise SystemExit("the fold list differs from the fit's")
    fold = {m: v[0] for m, v in json.loads(fl)["by_main"].items()}
    ub = git("show", f"HEAD:{UNCUT_FIT_GIT[0]}")
    if hashlib.sha256(ub).hexdigest() != UNCUT_FIT_GIT[1]:
        raise SystemExit("the uncut git fit record differs from its pin")
    ur = json.loads(ub)
    uncut = {"stamp": ur["stamp (PT)"], "fit_record_sha256": ur["fit"]["fit_record_sha256"], "rows": ur["rs_uncut"]["rows"]}
    out_ = shuffle.remote(fr["fit"]["fit_record_sha256"], uncut, main_of, fold, sorted(inp["folds"]["by_main"]))
    res = out_["readouts"]
    tp_local = (BUILD / "torch_probe_pinned.py").read_bytes()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    concl = {}
    for rd, r in res.items():
        f_ = r["fits"]
        concl[rd] = {"A_reproduces_registered_file": max(f_["A_gate_registered_rows"]["max_abs_diff_vs_registered_file"].values()) == 0.0,
                     "B_equals_registered_file": max(f_["B_descriptive_uncut_below_32768"]["max_abs_diff_vs_registered_file"].values()) == 0.0,
                     "C_drop_changes_weights_gt_1e-8": f_["C_one_row_dropped"]["max_abs_dW_vs_A"] > 1e-8, "D_flip_changes_weights_gt_1e-8": f_["D_one_label_flipped"]["max_abs_dW_vs_A"] > 1e-8,
                     "E_shuffle_dW_vs_A": f_["E_shuffled_order"]["max_abs_dW_vs_A"], "fitter_order_invariant_bitwise": f_["E_shuffled_order"]["max_abs_dW_vs_A"] == 0.0,
                     "F_rows_bitwise_equal_in_fit_order": r["F_rows_in_fit_order"]["bitwise_equal_in_sequence"],
                     "all_fits_certified": all(v["certified"] for v in f_.values())}
    note = {"label": "Exact-equality diagnosis for own_all_rs_uncut's descriptive cut-32768 refit (APPEND-ONLY NOTE, DESCRIPTIVE; 4f / 09 / 0d, 10-16): own_all_rs_L48 fold 0, "
                     "R and C, cases (A)-(F) on one code path (_fit_one + tp.fit); weights / diffs / row hashes only, no scores",
            "stamp (PT)": stamp, "writer": {"commit": git("rev-parse", "HEAD").decode().strip(), "app": APP},
            "registered_fit": {"git_record": RS_FIT_GIT[0], "sha256": RS_FIT_GIT[1], "fit_record_sha256": fr["fit"]["fit_record_sha256"]},
            "uncut_fit": {"git_record": UNCUT_FIT_GIT[0], "sha256": UNCUT_FIT_GIT[1], "fit_record_sha256": uncut["fit_record_sha256"]},
            "torch_probe_pinned": {"sha256_in_image": out_["torch_probe_pinned_sha256"], "sha256_local_build": hashlib.sha256(tp_local).hexdigest(), "scan": out_["torch_probe_fit_scan"]},
            "results": res, "conclusions": concl,
            "reading": "A = the gate (baseline); order sensitivity is read from E vs A only (Codex r21). If F is true, (B)'s rows equal the registered rows bitwise in fit order and "
                       "the 0.0 is explained by identical inputs; if F is false and E's dW is 0.0, the fitter is order-invariant; if both are false, the 0.0 stays unexplained"}
    out = REPO / "experiments" / "2026-10-14_p2_family_expansion" / "results" / f"own_all_rs_exact_equality_NOTE_{stamp}.json"
    out.open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"conclusions": concl, "path": str(out.relative_to(REPO))}))
