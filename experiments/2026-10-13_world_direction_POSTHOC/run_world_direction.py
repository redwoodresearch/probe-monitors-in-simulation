"""Remove the world direction, then refit fake/fake (b″) and mixed (c″) (RULE.md; POST HOC). One Modal H100 per layer.

Per nested model (f, v) (training runs outside folds f and v), on d9's fix-full packed features (sce_fixfull/2026-10-13_152618):
- standardize by the honest training rows (real honest + fix-full simulated honest);
- iterative nullspace projection: 11 logistic fits (C = 0.01) real-vs-simulated honest; after removing j directions the
  next fit's run-level AUROC on the held-out honest runs (folds f, v) is "realism left" at k = j; the mean-difference
  direction beside it;
- for k in (0, 1, 2, 3, 5, 10) and mean-difference k = 1: project out, then d9's _fit recipe (standardize by the cell's
  training rows, C = 0.01, 1,000 iterations, run mean) for b″ and c″; the nested scores of folds f and v are stored.
Returns scores only; no activations leave the Volume.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy modal run --detach \\
      experiments/2026-10-13_world_direction_POSTHOC/run_world_direction.py --runs <fix-full scores npz> --out <results npz>
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-worlddir_fit_kill-2026-11-13"
WORK, STAMP = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "2026-10-13_152618"
RUNS_SHA256 = "48fc5eb52546"  # prefix of d9's sce_fixfull_scores_2026-10-13_152618.npz (full check below)
C, MAX_ITER, KS, N_DIR = 0.01, 1000, (0, 1, 2, 3, 5, 10), 10
H, SAX, SHX = "real-honest", "sim-attack-fullfix", "sim-honest-fullfix"
CELLS = {"b_full": (SAX, SHX), "c_full": (SAX, H)}
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3").add_local_python_source("probe_shift")


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, timeout=7200, retries=0, volumes={"/work": work})
def run(layer: int, arm: list, fold: list) -> dict:
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool

    t0 = time.time()
    d = Path("/work/sce_fixfull") / STAMP
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    X = torch.from_numpy(np.load(d / f"L{layer}.npy")).cuda().float()
    arm, fold = np.array(arm), np.array(fold)
    n = len(arm)
    if r_run.max() >= n or np.any(np.diff(r_run) < 0):
        raise RuntimeError("rows don't match the run table")
    honest = np.isin(arm, [H, SHX])
    out = {f"{v}__{c}": np.full((5, 5, n), np.nan, np.float32) for v in [f"k{k}" for k in KS] + ["md1"] for c in CELLS}
    realism = {f"k{j}": [] for j in range(N_DIR + 1)} | {"md1": []}
    evals = []

    def fit_score(Z, train_runs, score_runs, y_run):
        tr, te = np.flatnonzero(train_runs[r_run]), np.flatnonzero(score_runs[r_run])
        Ztr = Z[torch.from_numpy(tr).cuda()]
        mean, std = Ztr.mean(0), Ztr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        W, b, ne = tp.fit((Ztr - mean) / std, torch.from_numpy(y_run[r_run[tr]]).cuda().float(), C, MAX_ITER)
        del Ztr
        evals.append(int(ne))
        s = tp.decision((Z[torch.from_numpy(te).cuda()] - mean) / std, W, b).double().cpu().numpy()[:, 0]
        if not np.isfinite(s).all():
            raise RuntimeError("non-finite scores")
        S = pool(r_run[te], s, n, "mean")
        S[~score_runs] = np.nan
        return S, W[:, 0] / std  # the direction in Z's coordinates (decision = Z @ (w / std) + const)

    def auc(pos, neg):
        a = np.concatenate([pos, neg])
        r = a.argsort().argsort().astype(float) + 1
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)  # mid-ranks for ties
        ranks = np.array([r[inv == i].mean() for i in range(len(cnt))])[inv] if len(cnt) < len(a) else r
        return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    for f, v in PAIRS:
        te = (fold == f) | (fold == v)
        htr = honest & ~te
        rows_h = torch.from_numpy(np.flatnonzero(htr[r_run])).cuda()
        mu, sd = X[rows_h].mean(0), X[rows_h].std(0, unbiased=False)
        sd = sd.where(sd > 0, sd.new_ones(()))
        Z = (X - mu) / sd
        y_world = np.where(arm == SHX, 1.0, np.where(arm == H, 0.0, np.nan))
        U = torch.zeros(X.shape[1], 0, device="cuda")
        hte = honest & te
        for j in range(N_DIR + 1):  # nullspace iteration on honest rows only
            Zp = Z - (Z @ U) @ U.T if U.shape[1] else Z
            S, w = fit_score(Zp, htr, hte, y_world)
            realism[f"k{j}"].append(auc(S[hte & (arm == SHX)], S[hte & (arm == H)]))
            if j < N_DIR:
                w = w - U @ (U.T @ w) if U.shape[1] else w
                U = torch.cat([U, (w / w.norm())[:, None]], 1)
            del Zp
        md = Z[torch.from_numpy(np.flatnonzero((htr & (arm == SHX))[r_run])).cuda()].mean(0) - Z[torch.from_numpy(np.flatnonzero((htr & (arm == H))[r_run])).cuda()].mean(0)
        md = (md / md.norm())[:, None]
        variants = {f"k{k}": U[:, :k] for k in KS} | {"md1": md}
        for name, Uk in variants.items():
            Zp = Z - (Z @ Uk) @ Uk.T if Uk.shape[1] else X  # k = 0 on the raw features: d9's exact path (the gate)
            if name == "md1":
                S, _ = fit_score(Zp, htr, hte, y_world)
                realism["md1"].append(auc(S[hte & (arm == SHX)], S[hte & (arm == H)]))
            for cell, (pos, neg) in CELLS.items():
                y = np.where(arm == pos, 1.0, np.where(arm == neg, 0.0, np.nan))
                S, _ = fit_score(Zp, ~np.isnan(y) & ~te, te, y)
                out[f"{name}__{cell}"][f, v] = S
            del Zp
        del Z
    return {"layer": layer, "arrays": out, "realism": realism, "max_evals": max(evals), "fits_at_cap": sum(e >= MAX_ITER for e in evals),
            "n_fits": len(evals), "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(runs: str, out: str, layers: str = "48"):  # L48 first (RULE.md); L63 only if L48 is cheap
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    sha = hashlib.sha256(Path(runs).read_bytes()).hexdigest()
    if not sha.startswith(RUNS_SHA256):
        raise SystemExit("not d9's fix-full scores file")
    R = np.load(runs)
    arm, fold = R["runs__arm"].tolist(), R["runs__fold"].astype(int).tolist()
    res = list(run.starmap([(int(L), arm, fold) for L in layers.split(",")]))
    with open(out, "xb") as fh:
        np.savez(fh, **{f"L{r['layer']}__{k}": a for r in res for k, a in r["arrays"].items()})
    rec = {"git": git("rev-parse", "HEAD"), "app": APP, "runs_file_sha256": sha, "stamp": STAMP, "settings": {"C": C, "max_iter": MAX_ITER, "ks": KS},
           "per_layer": [{k: r[k] for k in ("layer", "max_evals", "fits_at_cap", "n_fits", "seconds")} | {"realism_mean": {j: float(np.mean(a)) for j, a in r["realism"].items()},
                                                                                                          "realism": r["realism"]} for r in res],
           "scores_file": Path(out).name, "scores_sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest()}
    Path(out[:-4] + ".json").open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("git", "scores_sha256")} | {"per_layer": [{k: v for k, v in p.items() if k != "realism"} for p in rec["per_layer"]]}))
