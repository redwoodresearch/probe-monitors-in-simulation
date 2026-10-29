"""James's check (e): was the Q1 probe undertrained at C = 0.01? A wide C search on the published Q1 features. Modal H100.

The probe as published (analyze_q1_gpu.py): output_end, all outputs, real epoch 1 vs simulated attack runs, features
standardized on the training rows, torch L-BFGS with scikit-learn's L2 logistic objective, max_iter 2000, cold start;
run score = mean of the run's output scores. Refit for every C in C_GRID and every layer in LAYERS, at scrub levels 7
(the claim) and 0 (reference). Outer split: held-out environments (10 folds, folds_v2.json). Inside each outer fold's
training environments, the main tasks of task fold (outer fold mod 5) are the validation set:
  - full model (all training environments): scores the held-out environment (test AUROC, run scores pooled over the
    10 folds as published) and its own training runs (training AUROC, per fold);
  - validation model (training environments minus the validation tasks): scores the validation runs (validation AUROC).
Selection-aware null (Aswin's rule, 2026-10-09): every fit carries 1 + N_NULL label columns, the real labels and Q1's
100 label-shuffle draws (per main task, real and simulated swap with probability 1/2), so each shuffle goes through
the identical search. Per fit, the largest gradient entry of the objective at the solution, per column (L-BFGS's own
stopping test is the joint max <= tol; see SETTINGS). Then one CPU step chooses, per label column, C per fold and layer, and C with
layer per fold, on validation AUROC (ties: the smaller C, the earlier layer), and scores those choices on the test
folds: the real result with 95% intervals from resampling main tasks, next to the same score over the 100 shuffles.

Features: the packed matrices analyze_q1_gpu.py wrote (<extraction_run>__q1gpu/packed/<manifest digest>/); the manifest
is checked against find_runs on today's splits. Per (level, layer, C) results go to <extraction_run>__q1gpu/cgrid_v1/
on the Volume, under a tag hashing this script, the probe and evaluation modules and the settings: a rerun with the
same tag skips finished ones; any change to code or settings starts a fresh tag. The summary goes to results/cgrid_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal run \
      experiments/2026-10-09_q1_method_checks/run_q1_cgrid.py::main [--pilot] [--summarize-only]
  (the ladder audit, levels 1-6: the same command with run_q1_cgrid.py::audit)
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q1-method-checks_cgrid_kill-2026-11-09"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
Q1 = ROOT / "experiments/2026-10-06_q1_segment_end"
# Level -> (Q1 config, the published analysis on it: its per-layer C = 0.01 numbers are the reproduction check).
LEVELS = {"level7": ("q1_alllayers_scrub7.yaml", "q1_gpu_2026-10-06_181139.json"),
          "level0": ("q1_alllayers.yaml", "q1_gpu_2026-10-06_190522.json")}
# Fitter audit (orchestrator's request, 2026-10-09 ~17:45 PT): the Q1 ladder's levels 1-6, published C = 0.01 analyses.
AUDIT_LEVELS = {f"level{n}": (f"q1_alllayers_scrub{n}.yaml", f"q1_gpu_2026-10-06_{stamp}.json")
                for n, stamp in zip(range(1, 7), ["190523", "190526", "190528", "190531", "190535", "190537"])}
ALL_LEVELS = LEVELS | AUDIT_LEVELS
PACKED = "d0c46e0e5516"  # manifest digest of the packed features (the same runs at both levels)
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]  # 1e-6 and 1e-5 added: pilot 2 chose 1e-4, the old bottom
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
POS = "output_end"
OUT = "cgrid_v1"  # + "_" + code_tag(): this script, torch_probe.py, q1_eval.py and the level's settings
# real vs simulated (the question) and real epoch 1 vs epoch 2 (the real-vs-real control, Aswin's rule for any difference
# claim, relayed 2026-10-09 18:30 PT: the same full search, selection included).
COMPARISONS = ["real_vs_sim", "null_e1_vs_e2"]
TOL = 0.01  # earliest-good layer: within this of the best validation AUROC (Q1's headline_tolerance)
N_NULL = 100  # label-shuffle columns fit beside the real labels: the selection-aware null runs the identical search
# Optimizer settings. tp.fit stops when the largest gradient entry of C * sum(logloss) / n + ||w||^2 / (2n) is below
# `tol`, so a fixed tol means a different convergence at every C: at C = 1e-4 the starting point already passes (no fit;
# pilot 2026-10-09 17:02 PT), at C = 0.01 it is 10x looser than scikit-learn's default (1e-4 on the gradient of the
# mean loss, which is this gradient / C). "scaled", the grid: the same minimizer found from an O(1) objective at every C.
# With u = w / sqrt(C), C * sum(logloss(Xw)) + ||w||^2 / 2 = C * [sum(logloss((sqrt(C) X) u)) + ||u||^2 / 2], so tp.fit
# runs at C = 1 on sqrt(C) X (float64) and w = sqrt(C) u; L-BFGS's loss-change test then means the same at every C, and
# its gradient test tol = 1e-4 * min(sqrt(C), 1) is scikit-learn's default (1e-4 on the gradient of the mean loss: the
# weights' gradient here is sqrt(C) times it, the intercept's equal) or stricter. Convergence criterion, fixed before the
# grid (Aswin, relayed 2026-10-09 17:40 PT): every label column's largest gradient entry <= tol. A fit that misses it
# after using all its iterations is refit from scratch with max_iter × 4, up to MAX_ITER_CAP; a fit that stopped early
# (pilot 3: C = 1e-6 stalls just above tol) is not refit; every fit that misses is listed. "published": Q1's float32, C as
# given, tol 1e-5, max_iter 2000, no refits; C = 0.01 only, for the reproduction gate and the published-vs-scaled
# comparison.
MAX_ITER_CAP = 80000
SETTINGS = {"scaled": {"dtype": "float64", "rescale": True, "tol_per_sqrt_C": 1e-4, "max_iter": 5000},
            "published": {"dtype": "float32", "rescale": False, "tol": 1e-5, "max_iter": 2000}}

app = modal.App(APP)
vol = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "scipy")
         .add_local_python_source("probe_shift"))


def level_cfg(level: str) -> dict:
    """The settings the Q1 config fixed for this level, plus its paired attack cells and folds."""
    import yaml

    cfg = yaml.safe_load((Q1 / "config" / ALL_LEVELS[level][0]).read_text())
    if cfg["C"] != 0.01 or cfg.get("design", "attack") != "attack" or cfg.get("match_cut_file") or cfg.get("clean_cells"):
        raise ValueError(f"{ALL_LEVELS[level][0]}: not the published Q1 setup")
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())
    ex = json.loads((ROOT / "splits" / cfg["exclusions_file"]).read_text())
    return {"extraction_run": cfg["extraction_run"], "meta_dir": cfg["meta_dir"], "seed": cfg["seed"],
            "bootstrap": cfg["bootstrap"], "max_iter": cfg["max_iter"], "paired_cells": ex["paired_attack_cells"]["cells"],
            "main_task_to_fold": folds["task_folds"]["main_task_to_fold"],
            "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}


def published(level: str, extraction_run: str, comparison: str = "real_vs_sim") -> dict:
    """{layer: held-out-environment run AUROC at C = 0.01} from the published analysis."""
    r = json.loads((Q1 / "results" / ALL_LEVELS[level][1]).read_text())
    if r["config"]["extraction_run"] != extraction_run or r["config"]["C"] != 0.01:
        raise ValueError(f"{ALL_LEVELS[level][1]} is not the C = 0.01 analysis of {extraction_run}")
    return {int(k): v["results"][POS][comparison]["held_out_environment"]["all_outputs"]["run"]["auroc"] for k, v in r["by_layer"].items()}


def headline(level: str) -> dict:
    """The published held-out-environment headline: run AUROC with its interval and the layer chosen per fold."""
    h = json.loads((Q1 / "results" / ALL_LEVELS[level][1]).read_text())["headline_held_out_environment"][POS]["real_vs_sim"]["all_outputs"]
    return {"run": h["run"], "chosen": {int(f): int(l) for f, l in h["chosen_layer_by_fold"].items()}}


def setup(cfg: dict, layer: int, comparison: str = "real_vs_sim"):
    """The checked manifest, run metadata, output_end table, scorer and per-run labels and folds."""
    import numpy as np

    from probe_shift.q1_eval import (
        Scorer,
        comparisons,
        find_runs,
        load_runs,
        step_table,
    )

    src = Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / "packed" / PACKED
    man = json.loads((src / f"manifest.L{layer}.json").read_text())
    if hashlib.sha1(json.dumps(man, sort_keys=True).encode()).hexdigest()[:12] != PACKED:
        raise ValueError("packed manifest digest mismatch")
    if json.loads(json.dumps(find_runs(cfg))) != man:
        raise ValueError("the packed manifest differs from find_runs on today's splits")
    runs = load_runs(cfg, man)
    sc = Scorer(cfg, runs, comparison)
    y = np.array([float(m["arm"] in comparisons(cfg)[comparison][1]) for m in runs], np.float32)
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    return src, runs, step_table(runs, POS), sc, y, env


def code_tag(cfgs: dict) -> str:
    """Hash of the code that makes a result and every level's settings: cached results are reused only under the same tag."""
    h = hashlib.sha256()
    for f in (Path(__file__), ROOT / "src/probe_shift/torch_probe.py", ROOT / "src/probe_shift/q1_eval.py"):
        h.update(f.read_bytes())
    h.update(json.dumps(cfgs, sort_keys=True).encode())
    return h.hexdigest()[:12]


def done_path(cfg: dict, tag: str, level: str, layer: int, C: float, setting: str, comparison: str = "real_vs_sim") -> Path:
    return Path("/acts") / f"{cfg['extraction_run']}__q1gpu" / f"{OUT}_{tag}" / level / comparison / f"L{layer}_C{C:g}_{setting}.npz"


def tol_of(setting: str, C: float) -> float:
    s = SETTINGS[setting]
    return s["tol_per_sqrt_C"] * min(C**0.5, 1.0) if s["rescale"] else s["tol"]


def labels(cfg: dict, runs: list, sc):
    """[n_runs, 1 + N_NULL]: column 0 the real labels, then Q1's label-shuffle draws (analyze_q1_gpu.py: per main task,
    real and simulated swap with probability 1/2; rng seed + 1, tasks in sorted order). A swap stays inside its cell,
    so every environment fold and validation-task set keeps its groups."""
    import numpy as np

    rng = np.random.default_rng(cfg["seed"] + 1)
    tasks = sorted({runs[i]["main_task"] for i in np.flatnonzero(sc.runs_in)})
    Y = np.zeros((len(runs), 1 + N_NULL), np.float32)
    Y[:, 0] = sc.y_run
    for d in range(N_NULL):
        flip = dict(zip(tasks, rng.random(len(tasks)) < 0.5))
        Y[:, 1 + d] = [1 - v if (sc.runs_in[i] and flip[runs[i]["main_task"]]) else v for i, v in enumerate(sc.y_run)]
    return Y


def col_auc(Y, S):
    """AUROC of each column of S [n, K] against the labels in the same column of Y, ties counted half (roc_auc_score)."""
    from scipy.stats import rankdata

    R = rankdata(S, axis=0)
    n1 = Y.sum(0)
    return ((R * Y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Y) - n1))


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, volumes={"/acts": vol}, timeout=2 * 3600, retries=0,
              scaledown_window=5)
def grid(cfg: dict, tag: str, level: str, layer: int, jobs: list, comparison: str = "real_vs_sim") -> dict:
    """Every (setting, C) job for one layer: per outer fold, the full and validation fits of all 1 + N_NULL label
    columns at once."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp

    t0 = time.time()
    todo = [(st, C) for st, C in jobs if not done_path(cfg, tag, level, layer, C, st, comparison).exists()]
    if not todo:
        return {"level": level, "layer": layer, "comparison": comparison, "skipped": jobs}
    src, runs, t, sc, _, env = setup(cfg, layer, comparison)
    Y = labels(cfg, runs, sc)
    X = np.load(src / f"L{layer}_{POS}.npy")
    if len(X) != len(t["run"]):
        raise ValueError(f"packed L{layer}: {len(X)} rows, expected {len(t['run'])}")
    X = torch.from_numpy(X).cuda().float()
    print(f"{level} L{layer}: loaded in {time.time() - t0:.0f} s", flush=True)
    use = sc.runs_in[t["run"]]  # the comparison's runs (real e1 and simulated, or real e1 and e2) of cells holding both
    efold, tfold = env[t["run"]], sc.fold[t["run"]]
    idx = lambda m: torch.from_numpy(np.flatnonzero(m)).cuda()

    def run_means(rows, S):
        """Mean output score per run, every column: (run ids, [runs, K] float64)."""
        k, inv = np.unique(t["run"][rows], return_inverse=True)
        tot = torch.zeros(len(k), S.shape[1], dtype=torch.float64, device="cuda").index_add_(0, torch.from_numpy(inv).cuda(), S.double())
        return k, (tot / torch.from_numpy(np.bincount(inv)).cuda()[:, None]).cpu().numpy()

    def grad_max(Xs, Ys, W, b, C):
        """Per column, the largest gradient entry of tp.fit's objective at (W, b) (L-BFGS's test: the joint max <= tol)."""
        W, b = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
        loss = (C * F.binary_cross_entropy_with_logits(Xs @ W + b, Ys, reduction="sum") + 0.5 * (W * W).sum()) / len(Xs)
        loss.backward()
        return torch.maximum(W.grad.abs().amax(0), b.grad.abs()).cpu().numpy()

    folds = sorted(set(efold[use]))
    K, n_f = Y.shape[1], len(folds)
    res = {job: {"test_run_score": np.full((len(runs), K), np.nan), "train_auroc": np.zeros((n_f, K)), "val_auroc": np.zeros((n_f, K)),
               "test_auroc_fold": np.zeros((n_f, K)), "grad_full": np.zeros((n_f, K)), "grad_val": np.zeros((n_f, K)),
               "evals_full": np.zeros(n_f, int), "evals_val": np.zeros(n_f, int), "max_iter_full": np.zeros(n_f, int),
               "max_iter_val": np.zeros(n_f, int), "seconds": np.zeros(n_f)} for job in todo}
    def fold_data(f, dt):
        """Outer fold f's full and validation fits: (row masks, standardized float `dt` tensors and labels)."""
        tr, te = use & (efold != f), use & (efold == f)
        va = tr & (tfold == f % 5)
        fits = {"full": (tr, te), "val": (tr & ~va, va)}
        data = {}
        for name, (a, b_) in fits.items():
            Xa, Xb = tp.standardize(X[idx(a)].to(getattr(torch, dt)), X[idx(b_)].to(getattr(torch, dt)))
            data[name] = (Xa, Xb, torch.from_numpy(Y[t["run"][a]]).cuda().to(getattr(torch, dt)))
        return fits, data

    out = {}
    # One job (setting, C) at a time over all folds, its record written as soon as it is done: a stopped container keeps
    # every finished job (2026-10-10: a stop lost a whole grid that was written only at the end). Standardizing per job
    # costs about 10% more time.
    for job in todo:
        st, C = job
        r = res[job]
        for i, f in enumerate(folds):
            fits, data = fold_data(f, SETTINGS[st]["dtype"])
            t1 = time.time()
            for name in fits:
                Xa, Xb, Ya = data[name]
                if SETTINGS[st]["rescale"]:  # fit u on sqrt(C) X at C = 1 (see SETTINGS); the scores are the same
                    Xa, Xb, C_fit = Xa * C**0.5, Xb * C**0.5, 1.0
                else:
                    C_fit = C
                mi, prev = SETTINGS[st]["max_iter"], None
                while True:  # refit with more iterations until every column passes, for the scaled grid
                    W, b, evals = tp.fit(Xa, Ya, C_fit, mi, tol=tol_of(st, C))
                    g = grad_max(Xa, Ya, W, b, C_fit)  # of the problem solved
                    # more iterations help only a fit that used them all. tp.fit returns closure evaluations, not
                    # iterations, so: no refit below mi evaluations, and stop when a refit repeats the previous count
                    # (L-BFGS is deterministic: a fit that stopped on its loss-change or line-search tests stops at the
                    # same point again). What still misses the criterion is listed.
                    if (not SETTINGS[st]["rescale"] or g.max() <= tol_of(st, C) or evals < mi or evals == prev
                            or mi * 4 > MAX_ITER_CAP):
                        break
                    mi, prev = mi * 4, evals
                r[f"evals_{name}"][i], r[f"grad_{name}"][i], r[f"max_iter_{name}"][i] = evals, g, mi
                k, M = run_means(fits[name][1], tp.decision(Xb, W, b))
                if name == "full":
                    kt, Mt = run_means(fits["full"][0], tp.decision(Xa, W, b))
                    r["train_auroc"][i] = col_auc(Y[kt], Mt)
                    r["test_auroc_fold"][i] = col_auc(Y[k], M)
                    r["test_run_score"][k] = M
                else:
                    r["val_auroc"][i] = col_auc(Y[k], M)
            r["seconds"][i] = time.time() - t1
            print(f"{level} L{layer} fold {f} {st} C {C:g}: evals {r['evals_full'][i]}/{r['evals_val'][i]} max_iter {r['max_iter_full'][i]}/{r['max_iter_val'][i]} "
                  f"grad {r['grad_full'][i].max():.1e} {r['seconds'][i]:.0f} s", flush=True)  # timing only: no scores before the gate
            del data
        s = r["test_run_score"]
        if not np.isfinite(s[sc.runs_in]).all():
            raise ValueError(f"{level} L{layer} {st} C {C:g}: a compared run got no test score")
        p = done_path(cfg, tag, level, layer, C, st, comparison)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, folds=np.array(folds), tol=tol_of(st, C), max_iter=SETTINGS[st]["max_iter"], **r)
        os.replace(p.with_suffix(".tmp"), p)  # atomic: a reader sees a whole record or none
        vol.commit()
        out[f"{st} {C:g}"] = round(float(r["seconds"].sum()))
    return {"level": level, "layer": layer, "comparison": comparison, "fit_seconds_by_C": out, "seconds": round(time.time() - t0)}  # no scores before the gate


def readout_tokens(runs: list, sc, top: int = 10) -> dict:
    """Per readout kind (output_end, call_end: every kept position, before Q1's fixed-slot call_end filter) and world,
    the most common readout tokens with their share (Aswin, relayed 2026-10-09 17:40 PT: boilerplate or content?)."""
    from collections import Counter

    import numpy as np

    from probe_shift.q1_eval import comparisons, step_table

    neg, pos = comparisons({})["real_vs_sim"]
    arm = np.array([m["arm"] for m in runs])
    out = {}
    for kind in ("output_end", "call_end"):
        t = step_table(runs, kind)
        for w, arms in (("real", neg), ("simulated", pos)):
            m = sc.runs_in[t["run"]] & np.isin(arm[t["run"]], arms)
            c, n = Counter(t["token_text"][m].tolist()), int(m.sum())
            out[f"{kind}|{w}"] = {"positions": n, "top": [[tok, k, round(k / n, 4)] for tok, k in c.most_common(top)]}
    return out


def null_summary(obs: float, null) -> dict:
    import numpy as np

    null = np.asarray(null)
    return {"observed": round(float(obs), 4), "null_mean": round(float(null.mean()), 4), "null_p95": round(float(np.percentile(null, 95)), 4),
            "null_p99": round(float(np.percentile(null, 99)), 4), "null_max": round(float(null.max()), 4),
            "p": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4), "shuffles": len(null)}


def select_level(cfg: dict, tag: str, level: str, L: list[int], Cs: list[float], ref: dict, comparison: str = "real_vs_sim") -> dict:
    """One level: per label column, C per fold and layer, then C with layer per fold, chosen on validation AUROC (ties:
    the smaller C, the earlier layer), scored on the test folds. Column 0 is the result; columns 1.. its null."""
    import numpy as np

    _, runs, _, sc, _, env = setup(cfg, L[0], comparison)
    Y = labels(cfg, runs, sc)
    R = {(l, C): dict(np.load(done_path(cfg, tag, level, l, C, "scaled", comparison))) for l in L for C in Cs}
    P = {l: dict(np.load(done_path(cfg, tag, level, l, 0.01, "published", comparison))) for l in L}
    folds = R[(L[0], Cs[0])]["folds"].tolist()
    if any(v["folds"].tolist() != folds or v["test_run_score"].shape != Y.shape for v in [*R.values(), *P.values()]):
        raise ValueError(f"{level}: cached records differ in folds or label columns")
    r_in = np.flatnonzero(sc.runs_in)
    if not all(np.isfinite(v["test_run_score"][r_in]).all() for v in [*R.values(), *P.values()]):
        raise ValueError(f"{level}: a compared run has no test score in a cached record")
    V = np.stack([np.stack([R[(l, C)]["val_auroc"] for C in Cs]) for l in L])  # [layer, C, fold, column]
    c_ix = V.argmax(1)  # [layer, fold, column]: first max, the smaller C
    top = V.max(1)  # [layer, fold, column]
    l_best = top.argmax(0)  # [fold, column]: first max, the earlier layer
    l_early = (top >= top.max(0, keepdims=True) - TOL).argmax(0)

    def nested(l_of, c_of):  # [fold, column] layer and C indices -> test AUROC per column
        s = np.full(Y.shape, np.nan)
        for i, f in enumerate(folds):
            m = env == f
            for k in range(Y.shape[1]):
                s[m, k] = R[(L[l_of[i, k]], Cs[c_of[i, k]])]["test_run_score"][m, k]
        return col_auc(Y[r_in], s[r_in]), s

    def report(l_of, c_of, choices=True):
        a, s = nested(l_of, c_of)
        d = {"test": sc.auc_ci(Y[r_in, 0], s[r_in, 0], sc.main[r_in]), "null": null_summary(a[0], a[1:])}
        if choices:
            d["choice_by_fold"] = {int(f): [L[l_of[i, 0]], Cs[c_of[i, 0]]] for i, f in enumerate(folds)}
        return d

    fi = np.arange(len(folds))[:, None]
    per_layer = {L[j]: report(np.full_like(l_best, j), c_ix[j], choices=False) | {"C_chosen_by_fold": {int(f): Cs[c_ix[j, i, 0]] for i, f in enumerate(folds)}}
                 for j in range(len(L))}
    test = {k: col_auc(Y[r_in], v["test_run_score"][r_in]) for k, v in R.items()}  # pooled test AUROC per column
    table = [{"layer": l, "C": C, "train_auroc_mean": round(float(v["train_auroc"][:, 0].mean()), 4),
              "val_auroc_mean": round(float(v["val_auroc"][:, 0].mean()), 4),
              "test": sc.auc_ci(Y[r_in, 0], v["test_run_score"][r_in, 0], sc.main[r_in]),
              "test_null_p95": round(float(np.percentile(test[(l, C)][1:], 95)), 4),
              "not_converged_full": int((v["grad_full"].max(1) > v["tol"]).sum()), "not_converged_full_real_column": int((v["grad_full"][:, 0] > v["tol"]).sum()),
              "not_converged_val": int((v["grad_val"].max(1) > v["tol"]).sum()), "not_converged_val_real_column": int((v["grad_val"][:, 0] > v["tol"]).sum()),
              "eval_limit_full": int((v["evals_full"] >= (1.25 * v["max_iter_full"]).astype(int)).sum()), "folds": len(folds),
              "seconds": round(float(v["seconds"].sum()))} for (l, C), v in sorted(R.items())]
    oracle = np.max(np.stack(list(test.values())), axis=0)  # per column, the best (layer, C) picked on the test folds
    o_key = max(test, key=lambda k: test[k][0])
    pub = {l: col_auc(Y[r_in, :1], v["test_run_score"][r_in, :1])[0] for l, v in P.items()}
    repro = {l: {"published": ref[l], "here": float(pub[l]), "diff": float(pub[l]) - ref[l]} for l in L}
    pub_vs_scaled = {l: {"published_optimizer": {"train": round(float(P[l]["train_auroc"][:, 0].mean()), 4), "val": round(float(P[l]["val_auroc"][:, 0].mean()), 4),
                                                 "test": sc.auc_ci(Y[r_in, 0], P[l]["test_run_score"][r_in, 0], sc.main[r_in])},
                         "scaled_optimizer_test": sc.auc_ci(Y[r_in, 0], R[(l, 0.01)]["test_run_score"][r_in, 0], sc.main[r_in])} for l in L}
    real_c = c_ix[:, :, 0].ravel()
    conv = {}  # the criterion fixed before the grid: every column's largest gradient entry <= tol, full and validation fits
    for col, sl in (("real", slice(0, 1)), ("shuffled", slice(1, None))):
        ok = [bool((v[f"grad_{n}"][:, sl] <= v["tol"]).all()) for v in R.values() for n in ("full", "val")]
        conv[col] = {"layer_C_fit_sets_passing": sum(ok), "of": len(ok)}
    conv["failing"] = [{"layer": l, "C": C, "fit": n, "fold": int(folds[i]), "grad_max": float(v[f"grad_{n}"][i].max()), "tol": float(v["tol"]),
                        "max_iter": int(v[f"max_iter_{n}"][i]), "evals": int(v[f"evals_{n}"][i])}
                       for (l, C), v in R.items() for n in ("full", "val") for i in range(len(folds)) if v[f"grad_{n}"][i].max() > v["tol"]]
    return {"table": table, "per_layer_C_chosen_on_validation": per_layer,
                  "best_layer_and_C": report(l_best, c_ix[l_best, fi, np.arange(Y.shape[1])[None, :]]),
                  "earliest_good_layer_and_C": report(l_early, c_ix[l_early, fi, np.arange(Y.shape[1])[None, :]]),
                  "C_chosen_at_grid_top": {"cells": int((real_c == len(Cs) - 1).sum()), "of": int(real_c.size)},
                  "C_chosen_at_grid_bottom": {"cells": int((real_c == 0).sum()), "of": int(real_c.size)},
                  "oracle_PICKED_ON_TEST": {"layer": o_key[0], "C": o_key[1], "null": null_summary(oracle[0], oracle[1:])},
                  "convergence": conv, "readout_tokens": readout_tokens(runs, sc) if comparison == "real_vs_sim" else None,
                  "reproduction_C0.01": repro, "C0.01_published_vs_scaled_optimizer": pub_vs_scaled, "reproduction_max_abs_diff": max(abs(v["diff"]) for v in repro.values())}  # unrounded


def audit_level(cfg: dict, tag: str, level: str, L: list[int], ref: dict, head: dict) -> dict:
    """One ladder level: per chosen layer, the published per-layer AUROC, the published optimizer's refit (reproduction)
    and the converged refit at C = 0.01; then the headline with the same per-fold layer choices, both ways."""
    import numpy as np

    _, runs, _, sc, _, env = setup(cfg, L[0])
    Y = labels(cfg, runs, sc)
    r_in = np.flatnonzero(sc.runs_in)
    rec = {(st, l): dict(np.load(done_path(cfg, tag, level, l, 0.01, st))) for st in ("published", "scaled") for l in L}
    if not all(np.isfinite(v["test_run_score"][r_in]).all() for v in rec.values()):
        raise ValueError(f"{level}: a compared run has no test score in a cached record")
    ci = lambda s: sc.auc_ci(Y[r_in, 0], s[r_in, 0], sc.main[r_in])
    per_layer = {}
    for l in L:
        P, S = rec[("published", l)], rec[("scaled", l)]
        per_layer[l] = {"published_file": ref[l], "published_refit": float(col_auc(Y[r_in, :1], P["test_run_score"][r_in, :1])[0]),
                        "converged_refit": ci(S["test_run_score"]),
                        # the published fits' largest real-label gradient in scikit-learn's units (this objective's / C)
                        "published_fit_grad_sklearn_units": float(max(P["grad_full"][:, 0].max(), P["grad_val"][:, 0].max()) / 0.01),
                        "converged_fits_passing": {col: int(sum((S[f"grad_{n}"][:, sl] <= S["tol"]).all(1).sum() for n in ("full", "val")))
                                                   for col, sl in (("real", slice(0, 1)), ("shuffled", slice(1, None)))},
                        "of": 2 * len(S["folds"])}

    def by_choice(st):
        s = np.full(Y.shape, np.nan)
        for f, l in head["chosen"].items():
            s[env == f] = rec[(st, l)]["test_run_score"][env == f]
        return s

    hp, hs = by_choice("published"), by_choice("scaled")
    a = col_auc(Y[r_in], hs[r_in])
    return {"per_layer": per_layer, "headline": {"published_file": head["run"], "published_refit": ci(hp), "converged_refit": ci(hs),
                                                 "converged_null_FIXED_LAYER_CHOICES": null_summary(a[0], a[1:]),
                                                 "chosen_layer_by_fold": head["chosen"]},
            "reproduction_max_abs_diff": max(abs(v["published_refit"] - v["published_file"]) for v in per_layer.values())}


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": vol}, timeout=3600, retries=0)
def audit_summary(cfgs: dict, tag: str, layers: dict, ref: dict, heads: dict) -> dict:
    vol.reload()
    return {lv: audit_level(cfg, tag, lv, layers[lv], ref[lv], heads[lv]) for lv, cfg in cfgs.items()}


@app.local_entrypoint()
def audit():
    """Fitter audit of the Q1 ladder, levels 1-6 (levels 0 and 7 come with the C search): the headline's chosen layers
    refit at C = 0.01 with the published and the scaled (converged) optimizer, real labels and the 100 shuffles.
    Writes results/fitter_audit_q1_ladder_<stamp>.json."""
    import subprocess

    cfgs = {lv: level_cfg(lv) for lv in AUDIT_LEVELS}
    tag = code_tag(cfgs)
    heads = {lv: headline(lv) for lv in cfgs}
    layers = {lv: sorted(set(h["chosen"].values())) for lv, h in heads.items()}
    print(f"cache tag {tag}; layers {layers}", flush=True)
    calls = [grid.spawn(cfgs[lv], tag, lv, l, [("published", 0.01), ("scaled", 0.01)]) for lv in cfgs for l in layers[lv]]
    for c in calls:
        print(json.dumps(c.get()), flush=True)
    ref = {lv: published(lv, cfgs[lv]["extraction_run"]) for lv in cfgs}
    out = audit_summary.remote(cfgs, tag, layers, ref, heads)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    bad = {lv: s["reproduction_max_abs_diff"] for lv, s in out.items() if s["reproduction_max_abs_diff"] > 0.002}
    dst = HERE / "results" / ("partial" if bad else "") / f"fitter_audit_q1_ladder_{stamp}{'_REPRO_FAILED' if bad else ''}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps({"meta": {"git": git, "optimizer_settings": SETTINGS, "layers": layers, "volume_results": f"{OUT}_{tag}",
                                      "configs": {lv: ALL_LEVELS[lv] for lv in cfgs}}, "levels": out}, indent=1, default=str))
    if bad:
        raise SystemExit(f"REPRODUCTION FAILED (published optimizer vs published per-layer numbers, max |diff| over 0.002): {bad}; kept at {dst}")
    for lv, s in out.items():
        h = s["headline"]
        print(f"{lv}: headline published {h['published_file']['auroc']} | published refit {h['published_refit']['auroc']} | "
              f"converged {h['converged_refit']} | null (fixed choices) {h['converged_null_FIXED_LAYER_CHOICES']}")
        for l, v in s["per_layer"].items():
            print(f"  L{l}: published {v['published_file']} refit {v['published_refit']:.4f} converged {v['converged_refit']['auroc']} "
                  f"published grad (sklearn units) {v['published_fit_grad_sklearn_units']:.1e} converged passing {v['converged_fits_passing']}/{v['of']}")
    print("->", dst)


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": vol}, timeout=3600, retries=0)
def select(cfgs: dict, tag: str, layers: dict, Cs: list[float], ref: dict, comps: list[str]) -> dict:
    vol.reload()
    return {level: {c: select_level(cfg, tag, level, layers[level], Cs, ref[level][c], c) for c in comps} for level, cfg in cfgs.items()}


@app.local_entrypoint()
def main(pilot: bool = False, summarize_only: bool = False, ladder: bool = False):
    """`--ladder`: the same search at levels 1-6 (Aswin, 2026-10-10 14:12 PT, after the levels 1-6 rule fired at level 7),
    in 16 GB containers (a layer's features need under 2 GB)."""
    import subprocess

    if ladder and pilot:
        raise SystemExit("--ladder has no pilot: the pilot is level 7, L47")
    levels = AUDIT_LEVELS if ladder else LEVELS
    cfgs = {lv: level_cfg(lv) for lv in levels}
    tag = code_tag(cfgs)  # from both levels' settings, before the pilot narrows them: the full run reuses the pilot's fits
    layers = {lv: [47] if pilot else LAYERS for lv in levels}
    Cs = [1e-6, 1e-2, 100.0] if pilot else C_GRID
    jobs = [("published", 0.01)] + [("scaled", C) for C in Cs]
    comps = ["real_vs_sim"] if pilot else COMPARISONS
    if pilot:
        cfgs, layers = {"level7": cfgs["level7"]}, {"level7": [47]}
    print(f"cache tag {tag}", flush=True)
    if not summarize_only:
        g = grid.with_options(timeout=1200) if pilot else grid.with_options(memory=16384) if ladder else grid  # the pilot: at most 20 min of H100
        calls = [g.spawn(cfgs[lv], tag, lv, l, jobs, c) for c in comps for lv in cfgs for l in layers[lv]]
        for c in calls:
            print(json.dumps(c.get()), flush=True)
    ref = {lv: {c: published(lv, cfgs[lv]["extraction_run"], c) for c in comps} for lv in cfgs}
    summary = select.remote(cfgs, tag, layers, Cs, ref, comps)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", str(HERE)], capture_output=True, text=True, check=True).stdout.strip())
    bad = {f"{lv} {c}": s["reproduction_max_abs_diff"] for lv, by in summary.items() for c, s in by.items() if s["reproduction_max_abs_diff"] > 0.002}
    dst = HERE / "results" / ("partial" if bad else "") / f"cgrid_{'ladder_' if ladder else ''}{stamp}{'_pilot' if pilot else ''}{'_REPRO_FAILED' if bad else ''}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps({"meta": {"git": git, "dirty": dirty, "pilot": pilot, "comparisons": comps, "C_grid": Cs, "optimizer_settings": SETTINGS, "layers": layers, "position": POS,
                                      "volume_results": f"{OUT}_{tag}", "packed": PACKED, "configs": {lv: ALL_LEVELS[lv] for lv in cfgs},
                                      "settings": {lv: {k: c[k] for k in ("extraction_run", "max_iter", "seed", "bootstrap")} | {"paired_cells": len(c["paired_cells"])}
                                                   for lv, c in cfgs.items()}}, "levels": summary}, indent=1, default=str))
    if bad:  # nothing else is printed: the C = 0.01 refit must match the published numbers first
        raise SystemExit(f"REPRODUCTION FAILED at C = 0.01 (max |diff| over 0.002): {bad}; summary kept for diagnosis at {dst}")
    for lv, c, s in [(lv, c, s) for lv, by in summary.items() for c, s in by.items()]:
        lv = f"{lv} {c}"
        print(f"== {lv}: reproduction max |diff| at C=0.01 {s['reproduction_max_abs_diff']:.5f}")
        for x in s["table"]:
            t = x["test"]
            print(f"{lv} L{x['layer']:>2} C {x['C']:<7g} train {x['train_auroc_mean']:.3f} val {x['val_auroc_mean']:.3f} "
                  f"test {t['auroc']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}] (null p95 {x['test_null_p95']:.3f}) "
                  f"not-converged {x['not_converged_full']}/{x['folds']} ({x['seconds']} s)")
        for k in ("best_layer_and_C", "earliest_good_layer_and_C"):
            print(f"{lv} {k}: test {s[k]['test']} | null {s[k]['null']} | choices {s[k]['choice_by_fold']}")
        for l, v in s["C0.01_published_vs_scaled_optimizer"].items():
            print(f"{lv} L{l} C 0.01: published optimizer {v['published_optimizer']} | scaled optimizer test {v['scaled_optimizer_test']}")
        print(f"{lv} convergence: real {s['convergence']['real']}, shuffled {s['convergence']['shuffled']}, failing {len(s['convergence']['failing'])}")
        for k, v in (s["readout_tokens"] or {}).items():
            print(f"{lv} readout tokens {k} ({v['positions']}): " + ", ".join(f"{tok!r} {sh:.3f}" for tok, _, sh in v["top"]))
        print(f"{lv} C at grid top {s['C_chosen_at_grid_top']}, bottom {s['C_chosen_at_grid_bottom']}; oracle (picked on test) {s['oracle_PICKED_ON_TEST']}")
    for lv, by in summary.items():  # the three columns: real vs simulated, its shuffled-label null, real vs real
        for k in ("best_layer_and_C", "earliest_good_layer_and_C"):
            rs, rr = by["real_vs_sim"][k], by.get("null_e1_vs_e2", {}).get(k)
            print(f"THREE COLUMNS {lv} {k}: real vs sim {rs['test']['auroc']} {rs['test']['ci95']} | shuffled-label null p95 "
                  f"{rs['null']['null_p95']} (p {rs['null']['p']}, {rs['null']['shuffles']} shuffles) | real e1 vs e2 "
                  f"{rr['test']['auroc'] if rr else None} {rr['test']['ci95'] if rr else ''}")
    print("->", dst)
