"""Claim B's step B, part 2: the converged readout x C x layer search with its selection-aware null, on the packs from
run_claimB_b_pack.py (the fixed-sim plan's amendment 3, 43618a52 on aswin/fixed-sim). Modal H100, one container per
(level, layer), every readout and comparison of that level in it; each readout's rows are loaded once.

run_claimB_grid.py (probe step A) copied, with a readout dimension: the same fitter (rescaled problem, gradient check
<= 1e-4·min(sqrt(C), 1), refits for fits that used all their iterations), 1 + 100 label columns, outer folds
(held-out environments) and validation tasks (task fold = outer mod 5), comparisons and scoring (run score = mean of
the run's output scores; 5 test environments and all 10). C grid 1e-5 ... 1. Converged fits only: an entry whose full or
validation fit misses the gradient check is masked before the choice, real labels and shuffles alike. Reported per
comparison: the choice over readout x C x layer (best, and earliest good: the earliest layer within TOL of the best,
then its best readout and C), and the same choice within each readout. Each (comparison, readout, C) record is written
as soon as its folds finish, so a stop keeps finished work.

The null's swap unit follows the plan (amendment 2): one flip per main task for the real comparisons (as Q1), one per
planted pair for the planted checks. Probe step A's code flipped per cell in every comparison; B does not copy that.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal run --detach \
      experiments/2026-10-10_claimB_probe/run_claimB_b_grid.py::main --packs level0=<digest>,level1=<digest> \
      [--levels level1 --layers 47 --readouts output_end_mean] [--summarize-only]
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
APP = "probe-shift_aswin-surya_sub-claimb-probe_gridB_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
POS, N_NULL, TOL, MAX_ITER_CAP, SEED, BOOT = "output_end", 100, 0.01, 80000, 0, 1000
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
C_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
READOUTS = ["output_end_mean", "output_end_ema0.95", "output_end_ema0.8", "output_end"]
MAX_ITER = 5000
COMPS = {"fixed_vs_e1": (("real-attack-e1",), ("sim-attack-fixed",)), "e2_vs_e1": (("real-attack-e1",), ("real-attack-e2",)),
         "plant1": (("sim-attack-fixed",), ("sim-attack-fixed-plant1",)), "plant2": (("sim-attack-fixed",), ("sim-attack-fixed-plant2",))}
LEVEL_COMPS = {"level0": ["fixed_vs_e1", "e2_vs_e1"], "level1": ["fixed_vs_e1", "e2_vs_e1", "plant1", "plant2"]}
TEST_ENVS = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]

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

    q1_eval.DESIGNS["claimB"] = {"comparisons": COMPS}
    src = Path("/acts") / "claimB" / level / f"packedB_{cfg['packs'][level]}"
    runs = json.loads((src / f"runs.L{layer}.json").read_text())
    sc = Scorer(cfg | {"design": "claimB"}, runs, comparison)
    rng = np.random.default_rng(SEED + 1)
    unit = "cell" if comparison.startswith("plant") else "main_task"
    units = sorted({runs[i][unit] for i in np.flatnonzero(sc.runs_in)})
    Y = np.zeros((len(runs), 1 + N_NULL), np.float32)
    Y[:, 0] = sc.y_run
    for d in range(N_NULL):  # swap labels per main task (real comparisons, as Q1) or per planted pair
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
    return Path("/acts") / "claimB" / level / f"gridB_{cfg['packs'][level]}_{cfg['tag']}" / comparison / readout / f"L{layer}_C{C:g}.npz"


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, volumes={"/acts": vol}, timeout=4 * 3600, retries=0, scaledown_window=5)
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
                    "test_run_score": np.full((len(runs), K), np.nan), "evals": np.zeros((n_f, 2), int), "max_iter": np.zeros((n_f, 2), int)}
                for i, f in enumerate(folds):
                    tr, te = use & (efold != f), use & (efold == f)
                    va = tr & (tfold == f % 5)
                    for j, (name, a, b_) in enumerate((("full", tr, te), ("val", tr & ~va, va))):
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
                        k, M = run_means(b_, tp.decision(Xb, W, b))
                        if name == "full":
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

    _, runs, _, sc, Y, env = setup(cfg, level, comparison, L[0])
    R = {(rd, l, C): dict(np.load(done_path(cfg, level, comparison, rd, l, C))) for rd in readouts for l in L for C in Cs}
    folds = R[(readouts[0], L[0], Cs[0])]["folds"].tolist()
    r_in = np.flatnonzero(sc.runs_in)
    in_test = np.isin([runs[i]["environment"] for i in r_in], TEST_ENVS)
    subsets = {"test_envs": r_in[in_test], "all_envs": r_in}
    cand = [(rd, l) for rd in readouts for l in L]  # candidates, readout-major
    V = np.stack([np.stack([R[(rd, l, C)]["val_auroc"] for C in Cs]) for rd, l in cand])  # [candidate, C, fold, column]
    ok = np.stack([np.stack([np.maximum(R[(rd, l, C)]["grad_full"], R[(rd, l, C)]["grad_val"]) <= R[(rd, l, C)]["tol"] for C in Cs]) for rd, l in cand])
    V = np.where(ok, V, -np.inf)  # converged fits only (amendment 3), real labels and shuffles alike
    fi, cols = np.arange(len(folds))[:, None], np.arange(Y.shape[1])[None, :]
    layer_of = np.array([l for _, l in cand])

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

    def scored(sel, keep):
        if not keep[0]:
            return {"invalid": "a fold has no converged fit for the real labels"}
        k_of, c_of = sel
        s = np.full(Y.shape, np.nan)
        for i, f in enumerate(folds):
            m = env == f
            for k in np.flatnonzero(keep):
                rd, l = cand[k_of[i, k]]
                s[m, k] = R[(rd, l, Cs[c_of[i, k]])]["test_run_score"][m, k]
        res = {}
        for sub, r in subsets.items():
            if len(r):
                aucs = col_auc(Y[r][:, keep], s[r][:, keep])  # column 0 the real labels, then the scorable shuffles
                res[sub] = {"test": sc.auc_ci(Y[r, 0], s[r, 0], sc.main[r]), "null": null_summary(aucs[0], aucs[1:])}
        res["choices"] = {int(f): [cand[k_of[i, 0]][0], cand[k_of[i, 0]][1], Cs[c_of[i, 0]]] for i, f in enumerate(folds)}
        res["shuffles_dropped"] = int((~keep[1:]).sum())
        res["null_invalid"] = bool((~keep[1:]).sum() > 0.05 * N_NULL)
        return res

    out = {"all_readouts": {}, "by_readout": {}}
    sel, keep = choose(list(range(len(cand))))
    out["all_readouts"] = {n: scored(v, keep) for n, v in sel.items()}
    for rd in readouts:
        sel, keep = choose([k for k, (r_, _) in enumerate(cand) if r_ == rd])
        out["by_readout"][rd] = {n: scored(v, keep) for n, v in sel.items()}
    out["convergence"] = {rd: {"entries_masked_real": int((~ok[[k for k, c in enumerate(cand) if c[0] == rd]][..., 0]).sum()),
                               "entries_masked_shuffled": int((~ok[[k for k, c in enumerate(cand) if c[0] == rd]][..., 1:]).sum()),
                               "entries_per_column": len(L) * len(Cs) * len(folds)} for rd in readouts}
    out["table"] = [{"readout": rd, "layer": l, "C": C, "converged_real": bool((np.maximum(v["grad_full"][:, 0], v["grad_val"][:, 0]) <= v["tol"]).all()),
                     "val": round(float(v["val_auroc"][:, 0].mean()), 4),
                     **{f"test_{sub}": sc.auc_ci(Y[r, 0], v["test_run_score"][r, 0], sc.main[r]) for sub, r in subsets.items() if len(r)}}
                    for (rd, l, C), v in sorted(R.items())]
    out["runs"] = {"compared": len(r_in), "in_test_envs": int(in_test.sum()), "outer_folds": len(folds)}
    return out


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": vol}, timeout=3600, retries=0)
def summary(cfg: dict, levels: list[str], layers: list[int], Cs: list[float], readouts: list[str]) -> dict:
    vol.reload()
    return {lv: {c: summarize(cfg, lv, c, layers, Cs, readouts) for c in LEVEL_COMPS[lv]
                 if all(done_path(cfg, lv, c, rd, l, C).exists() for rd in readouts for l in layers for C in Cs)} for lv in levels}


@app.local_entrypoint()
def main(packs: str, levels: str = "level0,level1", layers: str = "", readouts: str = "", summarize_only: bool = False):
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
    dst = HERE / "results" / f"claimB_b_grid_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps({"meta": {"git": git, "code_tag": code, "packs": cfg["packs"], "levels": lvs, "layers": ls, "C_grid": C_GRID,
                                      "readouts": rds, "test_envs": TEST_ENVS}, "levels": res}, indent=1, default=str))
    for lv, by in res.items():
        for c, s in by.items():
            for scope, v in [("all readouts", s["all_readouts"])] + [(rd, x) for rd, x in s["by_readout"].items()]:
                for name, x in v.items():
                    if "invalid" in x:
                        print(f"{lv} {c} [{scope}] {name}: {x['invalid']}")
                        continue
                    for sub in ("test_envs", "all_envs"):
                        if sub in x:
                            print(f"{lv} {c} [{scope}] {name} {sub}: {x[sub]['test']} | null p95 {x[sub]['null'].get('null_p95')} p {x[sub]['null'].get('p')}"
                                  + (f" | NULL INVALID: {x['shuffles_dropped']} shuffles dropped" if x.get("null_invalid") else ""))
    print("->", dst)
