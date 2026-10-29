"""James's checks (c) and (d), part 3 of 4: the ladder's C search on the planted packs (PLAN.md), with its selection-aware
null. Modal H100 (16 GB), one container per (level, layer), every readout and comparison of that level in it.

Copied from claim B's step-B grid (experiments/2026-10-10_claimB_probe/run_claimB_b_grid.py at adba7a2b, itself probe step
A's grid with a readout dimension), changed only as follows:
- packs from run_pack_plants.py;
- the C search's C grid without C >= 1 (1e-6 ... 1e-1; PLAN.md Amendment 1);
- this plan's comparisons, with Amendment 1's neutral-line controls at level 7;
- scores pooled over all 10 held-out environments;
- each record also keeps the real labels' per-output test scores, for the post-plant and planted-output diagnostics;
- the summary adds the pooled family (both moving averages) and the selected real-column run scores.
Unchanged:
- the fitter (rescaled problem, gradient check <= 1e-4·min(sqrt(C), 1), refits for fits that used all their iterations);
- 1 + 100 label columns;
- the outer folds (held-out environments) and validation tasks (task fold = outer mod 5);
- run score = the mean of the run's output scores;
- converged fits only, masked before every choice;
- best and earliest-good selections.
The null swaps labels within each planted pair for the paired plants, and per main task for real vs simulated.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal run --detach \
      experiments/2026-10-10_q1_planted_checks/run_grid_plants.py::main --packs level0=<digest>,level7=<digest> \
      [--levels level7 --layers 47 --readouts output_end] [--summarize-only]
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q1plants_grid_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
POS, N_NULL, TOL, MAX_ITER_CAP, SEED, BOOT = "output_end", 100, 0.01, 80000, 0, 1000
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # PLAN.md Amendment 1: run_q1_cgrid.py's grid without C >= 1
READOUTS = ["output_end", "output_end_ema0.95", "output_end_ema0.8"]
POOLED = ["output_end_ema0.95", "output_end_ema0.8"]
MAX_ITER = 5000
COMPS = {"c_early": (("sim-attack",), ("sim-attack-tell-early",)), "c_mid": (("sim-attack",), ("sim-attack-tell-mid",)),
         "d_year": (("real-attack-e1-year2023",), ("real-attack-e1-year2027",)), "real_vs_sim": (("real-attack-e1",), ("sim-attack",)),
         "real_vs_sim_tell_early": (("real-attack-e1",), ("sim-attack-tell-early",)),
         "real_vs_sim_tell_mid": (("real-attack-e1",), ("sim-attack-tell-mid",)),
         "n_early": (("sim-attack",), ("sim-attack-neutral-early",)), "n_mid": (("sim-attack",), ("sim-attack-neutral-mid",))}
PAIRED = {"c_early", "c_mid", "d_year", "n_early", "n_mid"}  # planted pairs: the null swaps within each pair (its cell)
LEVEL_COMPS = {"level0": ["c_early", "c_mid", "d_year"],  # the bars' comparisons first, so a stop at the hard cap keeps them
               "level7": ["c_early", "c_mid", "n_early", "n_mid", "d_year", "real_vs_sim", "real_vs_sim_tell_early", "real_vs_sim_tell_mid"]}

app = modal.App(APP)
vol = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "scipy")
         .add_local_python_source("probe_shift"))


def tol_of(C: float) -> float:
    return 1e-4 * min(C**0.5, 1.0)


def setup(cfg: dict, level: str, comparison: str, layer: int):
    """The pack's runs and step table (shared by every readout), the comparison's scorer, label columns and folds."""
    import numpy as np

    from probe_shift import q1_eval
    from probe_shift.q1_eval import Scorer, step_table

    q1_eval.DESIGNS["q1plants"] = {"comparisons": COMPS}
    src = Path("/acts") / "q1plants" / level / f"packed_{cfg['packs'][level]}"
    runs = json.loads((src / f"runs.L{layer}.json").read_text())
    sc = Scorer(cfg | {"design": "q1plants"}, runs, comparison)
    rng = np.random.default_rng(SEED + 1)
    unit = "cell" if comparison in PAIRED else "main_task"
    units = sorted({runs[i][unit] for i in np.flatnonzero(sc.runs_in)})
    Y = np.zeros((len(runs), 1 + N_NULL), np.float32)
    Y[:, 0] = sc.y_run
    for d in range(N_NULL):  # swap labels per main task (real vs simulated, as Q1) or within each planted pair
        flip = dict(zip(units, rng.random(len(units)) < 0.5))
        Y[:, 1 + d] = [1 - v if (sc.runs_in[i] and flip[runs[i][unit]]) else v for i, v in enumerate(sc.y_run)]
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    return src, runs, step_table(runs, POS), sc, Y, env


def col_auc(Y, S):
    """AUROC of each column of S [n, K] against the labels in the same column of Y, ties counted half (roc_auc_score)."""
    from scipy.stats import rankdata

    R = rankdata(S, axis=0)
    n1 = Y.sum(0)
    return ((R * Y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Y) - n1))


def done_path(cfg, level, comparison, readout, layer, C) -> Path:
    return Path("/acts") / "q1plants" / level / f"grid_{cfg['packs'][level]}_{cfg['tag']}" / comparison / readout / f"L{layer}_C{C:g}.npz"


@app.function(image=image, gpu="H100!", cpu=8, memory=16384, volumes={"/acts": vol}, timeout=4 * 3600, retries=0, scaledown_window=5)
def grid(cfg: dict, level: str, layer: int, Cs: list[float], readouts: list[str]) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp

    t0, out, S = time.time(), {}, {}
    for readout in readouts:
        X = None
        for comparison in LEVEL_COMPS[level]:
            todo = [C for C in Cs if not done_path(cfg, level, comparison, readout, layer, C).exists()]
            if not todo:
                continue
            if comparison not in S:
                S[comparison] = setup(cfg, level, comparison, layer)
            src, runs, t, sc, Y, env = S[comparison]
            if X is None:
                X = torch.from_numpy(np.load(src / f"L{layer}_{readout}.npy")).cuda().float()
                if len(X) != len(t["run"]):
                    raise ValueError(f"{readout}: packed rows {len(X)} vs step table {len(t['run'])}")
            use, efold, tfold = sc.runs_in[t["run"]], env[t["run"]], sc.fold[t["run"]]
            idx = lambda m: torch.from_numpy(np.flatnonzero(m)).cuda()

            def run_means(rows, M, t=t):
                k, inv = np.unique(t["run"][rows], return_inverse=True)
                tot = torch.zeros(len(k), M.shape[1], dtype=torch.float64, device="cuda").index_add_(0, torch.from_numpy(inv).cuda(), M.double())
                return k, (tot / torch.from_numpy(np.bincount(inv)).cuda()[:, None]).cpu().numpy()

            def grad_max(Xs, Ys, W, b):  # the rescaled problem's objective at C = 1, per column
                W, b = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
                loss = (F.binary_cross_entropy_with_logits(Xs @ W + b, Ys, reduction="sum") + 0.5 * (W * W).sum()) / len(Xs)
                loss.backward()
                return torch.maximum(W.grad.abs().amax(0), b.grad.abs()).cpu().numpy()

            folds = sorted(set(efold[use]))
            K, n_f = Y.shape[1], len(folds)
            for C in todo:
                r = {k: np.zeros((n_f, K)) for k in ("train_auroc", "val_auroc", "grad_full", "grad_val")} | {
                    "test_run_score": np.full((len(runs), K), np.nan), "evals": np.zeros((n_f, 2), int), "max_iter": np.zeros((n_f, 2), int),
                    "test_out_score": np.full(len(t["run"]), np.nan)}  # the real labels' per-output test scores (diagnostics)
                for i, f in enumerate(folds):
                    tr, te = use & (efold != f), use & (efold == f)
                    va = tr & (tfold == f % 5)
                    for j, (name, a, b_) in enumerate((("full", tr, te), ("val", tr & ~va, va))):
                        if name == "val" and len(np.unique(Y[np.unique(t["run"][b_]), 0])) < 2:  # no two-class validation set
                            r["val_auroc"][i], r["grad_val"][i] = np.nan, 0.0  # NaN: never selectable (finite mask below)
                            continue
                        Xa, Xb = tp.standardize(X[idx(a)].double(), X[idx(b_)].double())
                        Xa, Xb, Ya = Xa * C**0.5, Xb * C**0.5, torch.from_numpy(Y[t["run"][a]]).cuda().double()
                        mi, prev = MAX_ITER, None
                        while True:  # refit only fits that used all their iterations; stop when a refit repeats
                            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                            g = grad_max(Xa, Ya, W, b)
                            if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                                break
                            mi, prev = mi * 4, evals
                        r[f"grad_{name}"][i], r["evals"][i, j], r["max_iter"][i, j] = g, evals, mi
                        D = tp.decision(Xb, W, b)
                        k, M = run_means(b_, D)
                        if name == "full":
                            r["test_out_score"][np.flatnonzero(b_)] = D[:, 0].double().cpu().numpy()
                            kt, Mt = run_means(a, tp.decision(Xa, W, b))
                            r["train_auroc"][i] = col_auc(Y[kt], Mt)
                            r["test_run_score"][k] = M
                        else:
                            r["val_auroc"][i] = col_auc(Y[k], M)
                        del Xa, Xb
                if not np.isfinite(r["test_run_score"][sc.runs_in]).all():
                    raise ValueError(f"{level} L{layer} {readout} {comparison} C {C:g}: a compared run got no test score")
                p = done_path(cfg, level, comparison, readout, layer, C)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p.with_suffix(".tmp"), "wb") as fh:
                    np.savez(fh, folds=np.array(folds), tol=tol_of(C), **r)
                os.replace(p.with_suffix(".tmp"), p)
                vol.commit()
                out[f"{readout} {comparison} {C:g}"] = round(time.time() - t0)
                print(f"{level} L{layer} {readout} {comparison} C {C:g} done at {round(time.time() - t0)} s; evals max {r['evals'].max()}", flush=True)
        del X
        torch.cuda.empty_cache()
    return {"level": level, "layer": layer, "done_at_s": out}


def null_summary(obs, null) -> dict:
    import numpy as np

    null = np.asarray(null)
    if not len(null):
        return {"observed": round(float(obs), 4), "shuffles": 0, "invalid": "no scorable shuffle"}
    return {"observed": round(float(obs), 4), "null_mean": round(float(null.mean()), 4), "null_p95": round(float(np.percentile(null, 95)), 4),
            "null_max": round(float(null.max()), 4), "p": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4), "shuffles": len(null)}


def summarize(cfg: dict, level: str, comparison: str, L: list[int], Cs: list[float], readouts: list[str]) -> dict:
    import numpy as np

    _, runs, t, sc, Y, env = setup(cfg, level, comparison, L[0])
    R = {(rd, l, C): dict(np.load(done_path(cfg, level, comparison, rd, l, C))) for rd in readouts for l in L for C in Cs}
    folds = R[(readouts[0], L[0], Cs[0])]["folds"].tolist()
    r_in = np.flatnonzero(sc.runs_in)
    cand = [(rd, l) for rd in readouts for l in L]  # candidates, readout-major
    V = np.stack([np.stack([R[(rd, l, C)]["val_auroc"] for C in Cs]) for rd, l in cand])  # [candidate, C, fold, column]
    ok = np.stack([np.stack([np.maximum(R[(rd, l, C)]["grad_full"], R[(rd, l, C)]["grad_val"]) <= R[(rd, l, C)]["tol"] for C in Cs]) for rd, l in cand])
    ok = ok & np.isfinite(V)  # converged fits with a defined validation AUROC only, real labels and shuffles alike
    V = np.where(ok, V, -np.inf)
    fi, cols = np.arange(len(folds))[:, None], np.arange(Y.shape[1])[None, :]
    layer_of = np.array([l for _, l in cand])
    row_run, row_sa = t["run"], list(zip(t["step"].tolist(), t["action"].tolist()))
    plant = {runs[i]["cell"]: tuple(runs[i]["plant"]) for i in range(len(runs)) if "plant" in runs[i] and runs[i]["arm"] in COMPS[comparison][1]}

    def choose(ks):
        """Per (fold, column) over candidates ks: best, and earliest good (the earliest layer within TOL of the best, then
        the best candidate at that layer). Returns {name: (candidate, C) index arrays [fold, column]} and the usable columns."""
        Vk, okk = V[ks], ok[ks]
        keep = okk.any((0, 1)).all(0)
        c_ix, top = Vk.argmax(1), Vk.max(1)  # [k, fold, column]
        best = top.argmax(0)
        good = top >= top.max(0, keepdims=True) - TOL
        first = np.where(good, layer_of[ks][:, None, None], np.iinfo(int).max).min(0)  # the earliest good layer
        early = np.where(layer_of[ks][:, None, None] == first[None], top, -np.inf).argmax(0)
        return {n: (np.asarray(ks)[k], c_ix[k, fi, cols]) for n, k in (("best", best), ("earliest_good", early))}, keep

    def ci_lo_exact(x):  # Scorer.auc_ci's own bootstrap (same draws), its 2.5th percentile unrounded
        from probe_shift.q1_eval import boot_aucs

        y0, task, m = Y[r_in, 0], sc.main[r_in], ~np.isnan(x)
        valid, a = boot_aucs(y0[m], [x[m]], task[m], sc.draws(len(np.unique(task[m]))))
        return float(np.percentile(a[0][valid], 2.5))

    def scored(sel, keep):
        if not keep[0]:
            return {"invalid": "a fold has no converged fit for the real labels"}
        k_of, c_of = sel
        s, out_s = np.full(Y.shape, np.nan), np.full(len(row_run), np.nan)
        for i, f in enumerate(folds):
            m = env == f
            for k in np.flatnonzero(keep):
                rd, l = cand[k_of[i, k]]
                s[m, k] = R[(rd, l, Cs[c_of[i, k]])]["test_run_score"][m, k]
            rd, l = cand[k_of[i, 0]]
            mr = m[row_run]
            out_s[mr] = R[(rd, l, Cs[c_of[i, 0]])]["test_out_score"][mr]
        aucs = col_auc(Y[r_in][:, keep], s[r_in][:, keep])  # column 0 the real labels, then the scorable shuffles
        res = {"all_envs": {"test": sc.auc_ci(Y[r_in, 0], s[r_in, 0], sc.main[r_in]), "null": null_summary(aucs[0], aucs[1:]),
                            "auroc_exact": float(aucs[0]), "null_max_exact": float(np.max(aucs[1:])) if len(aucs) > 1 else None,
                            "ci95_lo_exact": ci_lo_exact(s[r_in, 0])},
               "choices": {int(f): [cand[k_of[i, 0]][0], cand[k_of[i, 0]][1], Cs[c_of[i, 0]]] for i, f in enumerate(folds)},
               "shuffles_dropped": int((~keep[1:]).sum()), "null_invalid": bool((~keep[1:]).sum() > 0.05 * N_NULL),
               "scores_real_column": [None if np.isnan(x) else float(x) for x in s[:, 0]]}
        if plant and comparison in PAIRED:  # diagnostics (not bars): each pair rescored after the planted output, and on it alone
            for name, rule in (("after_plant", lambda sa, p: sa > p), ("planted_output", lambda sa, p: sa == p)):
                d = np.full(len(runs), np.nan)
                for i in r_in:
                    p = plant.get(runs[i]["cell"])
                    rows = [j for j in np.flatnonzero(row_run == i) if p is not None and rule(row_sa[j], p)]
                    if rows:
                        d[i] = np.nanmean(out_s[rows])
                res[name] = sc.auc_ci(Y[r_in, 0], d[r_in], sc.main[r_in])
        return res

    families = {"all_readouts": readouts, **{rd: [rd] for rd in readouts}, "pooled": [rd for rd in readouts if rd in POOLED]}
    out = {}
    for fam, rds in families.items():
        if not rds:
            continue
        sel, keep = choose([k for k, (r_, _) in enumerate(cand) if r_ in rds])
        out[fam] = {n: scored(v, keep) for n, v in sel.items()}
    out["convergence"] = {rd: {"entries_masked_real": int((~ok[[k for k, c in enumerate(cand) if c[0] == rd]][..., 0]).sum()),
                               "entries_masked_shuffled": int((~ok[[k for k, c in enumerate(cand) if c[0] == rd]][..., 1:]).sum()),
                               "entries_per_column": len(L) * len(Cs) * len(folds)} for rd in readouts}
    out["table"] = [{"readout": rd, "layer": l, "C": C, "converged_real": bool((np.maximum(v["grad_full"][:, 0], v["grad_val"][:, 0]) <= v["tol"]).all()),
                     "val": round(float(v["val_auroc"][:, 0].mean()), 4), "test_all_envs": sc.auc_ci(Y[r_in, 0], v["test_run_score"][r_in, 0], sc.main[r_in])}
                    for (rd, l, C), v in sorted(R.items())]
    out["runs"] = {"compared": len(r_in), "pairs_or_cells": len({runs[i]["cell"] for i in r_in}), "outer_folds": len(folds),
                   "meta": [[r["arm"], r["cell"], r["main_task"]] for r in runs]}
    return out


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": vol}, timeout=3600, retries=0)
def summary(cfg: dict, levels: list[str], layers: list[int], Cs: list[float], readouts: list[str]) -> dict:
    vol.reload()
    return {lv: {c: summarize(cfg, lv, c, layers, Cs, readouts) for c in LEVEL_COMPS[lv]
                 if all(done_path(cfg, lv, c, rd, l, C).exists() for rd in readouts for l in layers for C in Cs)} for lv in levels}


@app.local_entrypoint()
def main(packs: str, levels: str = "level0,level7", layers: str = "", readouts: str = "", summarize_only: bool = False):
    import hashlib
    import subprocess

    folds = json.loads((ROOT / "splits" / "folds_v2.json").read_text())
    code = hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "src/probe_shift/torch_probe.py").read_bytes()
                          + (ROOT / "src/probe_shift/q1_eval.py").read_bytes()).hexdigest()[:12]
    cfg = {"packs": dict(x.split("=") for x in packs.split(",")), "seed": SEED, "bootstrap": BOOT, "tag": code,
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    lvs = levels.split(",")
    ls = [int(x) for x in layers.split(",")] if layers else LAYERS
    rds = readouts.split(",") if readouts else READOUTS
    if not summarize_only:
        for c in [grid.spawn(cfg, lv, l, C_GRID, rds) for lv in lvs for l in ls]:
            print(json.dumps(c.get()), flush=True)
    res = summary.remote(cfg, lvs, ls, C_GRID, rds)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dst = HERE / "results" / f"q1plants_grid_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps({"meta": {"git": git, "code_tag": code, "packs": cfg["packs"], "levels": lvs, "layers": ls, "C_grid": C_GRID,
                                      "readouts": rds}, "levels": res}, indent=1, default=str))
    print("->", dst)
