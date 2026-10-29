"""Claim B's probe step, part 2: the converged C x layer search with its selection-aware null, on the merged packs from
run_claimB_pack.py. Modal H100, one container per (level, layer), every comparison of that level in it (load once).

The fitter, null and selection are q3's method-test code (experiments/2026-10-09_q1_method_checks/run_q1_cgrid.py at
940a9c26), copied: output_end readout, run score = mean of the run's output scores; features standardized on each fit's
training rows; the minimizer of C·Σ logloss + ||w||²/2 found as torch_probe.fit at C = 1 on sqrt(C)·X in float64,
converged when every label column's largest gradient entry <= 1e-4·min(sqrt(C), 1) (refits with max_iter × 4 for fits
that used all their iterations, up to 80,000; failures listed); 1 + 100 label columns per fit (real labels, then swaps
within each cell with probability 1/2), so the null reruns the identical search. Outer folds: held-out environments;
inside, the main tasks of task fold (outer mod 5) are the validation set; C and layer chosen on validation AUROC.
Comparisons (negative arms, positive arms): fixed sim vs real e1, real e2 vs real e1, and at level 1 the planted twins
(each pair one cell). Scores pooled over the 5 test environments (claim B's headline) and over all environments.
Each (comparison, C) record is written as soon as its folds finish.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal run --detach \
      experiments/2026-10-10_claimB_probe/run_claimB_grid.py::main --packs level0=<digest>,level1=<digest> [--pilot]
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
APP = "probe-shift_aswin-surya_sub-claimb-probe_grid_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
POS, N_NULL, TOL, MAX_ITER_CAP, SEED, BOOT = "output_end", 100, 0.01, 80000, 0, 1000
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
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
    """Merged runs and step table for this level, the comparison's scorer, label columns and environment folds."""
    import numpy as np

    from probe_shift import q1_eval
    from probe_shift.q1_eval import Scorer, step_table

    q1_eval.DESIGNS["claimB"] = {"comparisons": COMPS}
    src = Path("/acts") / "claimB" / level / f"packed_{cfg['packs'][level]}"
    runs = json.loads((src / f"runs.L{layer}.json").read_text())
    sc = Scorer(cfg | {"design": "claimB"}, runs, comparison)
    rng = np.random.default_rng(SEED + 1)
    cells = sorted({runs[i]["cell"] for i in np.flatnonzero(sc.runs_in)})
    Y = np.zeros((len(runs), 1 + N_NULL), np.float32)
    Y[:, 0] = sc.y_run
    for d in range(N_NULL):  # swap the two labels within a cell (a planted pair, or a real and a simulated run)
        flip = dict(zip(cells, rng.random(len(cells)) < 0.5))
        Y[:, 1 + d] = [1 - v if (sc.runs_in[i] and flip[runs[i]["cell"]]) else v for i, v in enumerate(sc.y_run)]
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    return src, runs, step_table(runs, POS), sc, Y, env


def col_auc(Y, S):
    """AUROC of each column of S [n, K] against the labels in the same column of Y, ties counted half (roc_auc_score)."""
    from scipy.stats import rankdata

    R = rankdata(S, axis=0)
    n1 = Y.sum(0)
    return ((R * Y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Y) - n1))


def done_path(cfg, level, comparison, layer, C) -> Path:
    return Path("/acts") / "claimB" / level / f"grid_{cfg['packs'][level]}_{cfg['tag']}" / comparison / f"L{layer}_C{C:g}.npz"


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, volumes={"/acts": vol}, timeout=3 * 3600, retries=0, scaledown_window=5)
def grid(cfg: dict, level: str, layer: int, Cs: list[float]) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp

    t0, out, X = time.time(), {}, None
    for comparison in LEVEL_COMPS[level]:
        todo = [C for C in Cs if not done_path(cfg, level, comparison, layer, C).exists()]
        if not todo:
            continue
        src, runs, t, sc, Y, env = setup(cfg, level, comparison, layer)
        if not sc.runs_in.any():  # e.g. the planted arms in a --dry pack: nothing to fit
            out[comparison] = "no runs"
            continue
        if X is None:
            X = torch.from_numpy(np.load(src / f"L{layer}_{POS}.npy")).cuda().float()
            if len(X) != len(t["run"]):
                raise ValueError(f"packed rows {len(X)} vs step table {len(t['run'])}")
        use, efold, tfold = sc.runs_in[t["run"]], env[t["run"]], sc.fold[t["run"]]
        idx = lambda m: torch.from_numpy(np.flatnonzero(m)).cuda()

        def run_means(rows, S, t=t):
            k, inv = np.unique(t["run"][rows], return_inverse=True)
            tot = torch.zeros(len(k), S.shape[1], dtype=torch.float64, device="cuda").index_add_(0, torch.from_numpy(inv).cuda(), S.double())
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
                raise ValueError(f"{level} L{layer} {comparison} C {C:g}: a compared run got no test score")
            p = done_path(cfg, level, comparison, layer, C)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p.with_suffix(".tmp"), "wb") as fh:
                np.savez(fh, folds=np.array(folds), tol=tol_of(C), **r)
            os.replace(p.with_suffix(".tmp"), p)
            vol.commit()
            out[f"{comparison} {C:g}"] = round(time.time() - t0)
            print(f"{level} L{layer} {comparison} C {C:g} done at {out[f'{comparison} {C:g}']} s; evals max {r['evals'].max()}", flush=True)
    return {"level": level, "layer": layer, "done_at_s": out}


def null_summary(obs, null) -> dict:
    import numpy as np

    null = np.asarray(null)
    if not len(null):
        return {"observed": round(float(obs), 4), "shuffles": 0, "invalid": "no scorable shuffle"}
    return {"observed": round(float(obs), 4), "null_mean": round(float(null.mean()), 4), "null_p95": round(float(np.percentile(null, 95)), 4),
            "null_max": round(float(null.max()), 4), "p": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4), "shuffles": len(null)}


def null_against(aucs) -> dict:
    """Column 0 (the real labels) against columns 1.. (the shuffles)."""
    return null_summary(aucs[0], aucs[1:])


def summarize(cfg: dict, level: str, comparison: str, L: list[int], Cs: list[float]) -> dict:
    import numpy as np

    _, runs, _, sc, Y, env = setup(cfg, level, comparison, L[0])
    R = {(l, C): dict(np.load(done_path(cfg, level, comparison, l, C))) for l in L for C in Cs}
    folds = R[(L[0], Cs[0])]["folds"].tolist()
    r_in = np.flatnonzero(sc.runs_in)
    in_test = np.isin([runs[i]["environment"] for i in r_in], TEST_ENVS)
    subsets = {"test_envs": r_in[in_test], "all_envs": r_in}
    V = np.stack([np.stack([R[(l, C)]["val_auroc"] for C in Cs]) for l in L])  # [layer, C, fold, column]
    # converged fits only (claim B plan amendment 3): an entry whose full or validation fit misses the gradient check is
    # never chosen, for the real labels and the shuffles alike; a column with a fold left without any converged entry
    # cannot be scored: a shuffle is dropped from the null (counted), the real labels make the result invalid
    ok = np.stack([np.stack([np.maximum(R[(l, C)]["grad_full"], R[(l, C)]["grad_val"]) <= R[(l, C)]["tol"] for C in Cs]) for l in L])
    V = np.where(ok, V, -np.inf)
    keep = ok.any((0, 1)).all(0)  # [column]
    c_ix, top = V.argmax(1), V.max(1)
    choices = {"best_layer_and_C": top.argmax(0), "earliest_good_layer_and_C": (top >= top.max(0, keepdims=True) - TOL).argmax(0)}
    fi, cols = np.arange(len(folds))[:, None], np.arange(Y.shape[1])[None, :]
    out = {"choices": {}, "convergence": {"converged_only": {
        "entries_masked_real": int((~ok[..., 0]).sum()), "entries_masked_shuffled": int((~ok[..., 1:]).sum()), "entries_per_column": int(ok[..., 0].size),
        "real_labels_scorable": bool(keep[0]), "shuffles_dropped": int((~keep[1:]).sum()), "null_invalid": bool((~keep[1:]).sum() > 0.05 * (len(keep) - 1))}}}
    for name, l_of in choices.items():
        if not keep[0]:
            out[name] = {"invalid": "a fold has no converged fit for the real labels"}
            continue
        c_of = c_ix[l_of, fi, cols]
        s = np.full(Y.shape, np.nan)
        for i, f in enumerate(folds):
            m = env == f
            for k in np.flatnonzero(keep):
                s[m, k] = R[(L[l_of[i, k]], Cs[c_of[i, k]])]["test_run_score"][m, k]
        out[name] = {sub: {"test": sc.auc_ci(Y[r, 0], s[r, 0], sc.main[r]), "null": null_against(col_auc(Y[r][:, keep], s[r][:, keep]))}
                     for sub, r in subsets.items() if len(r)}
        out["choices"][name] = {int(f): [L[l_of[i, 0]], Cs[c_of[i, 0]]] for i, f in enumerate(folds)}
    for col, sl in (("real", slice(0, 1)), ("shuffled", slice(1, None))):
        ok = [bool((v[f"grad_{n}"][:, sl] <= v["tol"]).all()) for v in R.values() for n in ("full", "val")]
        out["convergence"][col] = {"layer_C_fit_sets_passing": sum(ok), "of": len(ok)}
    out["convergence"]["failing"] = [{"layer": l, "C": C, "fit": n, "fold": int(folds[i]), "grad_max": float(v[f"grad_{n}"][i].max())}
                                     for (l, C), v in R.items() for n in ("full", "val") for i in range(len(folds)) if v[f"grad_{n}"][i].max() > v["tol"]]
    out["table"] = [{"layer": l, "C": C, "converged_real": bool((np.maximum(v["grad_full"][:, 0], v["grad_val"][:, 0]) <= v["tol"]).all()),
                     "train": round(float(v["train_auroc"][:, 0].mean()), 4), "val": round(float(v["val_auroc"][:, 0].mean()), 4),
                     **{f"test_{sub}": sc.auc_ci(Y[r, 0], v["test_run_score"][r, 0], sc.main[r]) for sub, r in subsets.items() if len(r)}}
                    for (l, C), v in sorted(R.items())]
    out["runs"] = {"compared": len(r_in), "in_test_envs": int(in_test.sum()), "outer_folds": len(folds)}
    return out


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": vol}, timeout=3600, retries=0)
def summary(cfg: dict, layers: list[int], Cs: list[float]) -> dict:
    vol.reload()
    return {lv: {c: summarize(cfg, lv, c, layers, Cs) for c in LEVEL_COMPS[lv] if done_path(cfg, lv, c, layers[0], Cs[0]).exists()}
            for lv in cfg["packs"]}


@app.local_entrypoint()
def main(packs: str, pilot: bool = False, summarize_only: bool = False):
    import hashlib
    import subprocess

    folds = json.loads((ROOT / "splits" / "folds_v2.json").read_text())
    code = hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "src/probe_shift/torch_probe.py").read_bytes()
                          + (ROOT / "src/probe_shift/q1_eval.py").read_bytes()).hexdigest()[:12]
    cfg = {"packs": dict(x.split("=") for x in packs.split(",")), "seed": SEED, "bootstrap": BOOT, "tag": code,
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    layers, Cs = ([47], [1e-4, 1e-2, 100.0]) if pilot else (LAYERS, C_GRID)
    if not summarize_only:
        g = grid.with_options(timeout=1800) if pilot else grid
        for c in [g.spawn(cfg, lv, l, Cs) for lv in cfg["packs"] for l in layers]:
            print(json.dumps(c.get()), flush=True)
    res = summary.remote(cfg, layers, Cs)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dst = HERE / "results" / f"claimB_grid_{stamp}{'_pilot' if pilot else ''}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps({"meta": {"git": git, "code_tag": code, "packs": cfg["packs"], "layers": layers, "C_grid": Cs,
                                      "test_envs": TEST_ENVS, "pilot": pilot}, "levels": res}, indent=1, default=str))
    for lv, by in res.items():
        for c, s in by.items():
            for name in ("best_layer_and_C", "earliest_good_layer_and_C"):
                for sub, v in s[name].items():
                    print(f"{lv} {c} {name} {sub}: {v['test']} | null {v['null']}" if isinstance(v, dict) else f"{lv} {c} {name}: {v}")
            print(f"{lv} {c} convergence: {s['convergence']['real']} real, {s['convergence']['shuffled']} shuffled, failing {len(s['convergence']['failing'])}")
    print("->", dst)
