"""Cell (c), mixed (RULE.md step 1 with Amendments 1-2): claim C's grid recipe (run_claim_c.grid @ main) with the training
attack arm the September SIMULATED run instead of the first real run. Deployed Modal app; one H100 per (variant, layer).

Unchanged from claim C's grid: the 196-task frozen folds, the inner validation fold (f + 1) % 5, 7 C, the rescaled fit
(sqrt(C)-scaled standardized features, C = 1 objective), the convergence tolerance and the refit loop up to 80,000
iterations, 1 + 100 label columns shuffled within each main task's training runs, run score = mean over outputs,
records `L<l>_output_end_C<C>.npz` with on_runs, on_fold, train, Y, tol, val_auroc, grad_full, grad_val, evals, on_score.
Changed: train = honest_a + every ATTEMPTED simulated run (Amendment 2.5); label y = world == "sim"; shuffle seed
[20261013, c]; output_end only; the cache per variant (swapped level 1: claimc_cache_v1; unswapped: retrain_cache_v1/
level1 and level0); and column 0's full-fold weights, intercepts and training mean and std are saved per (fold, C) in
`W_L<l>_C<C>.npz` (Amendment 2.6), so planted outputs can be scored and the rescoring checked against on_score.

  modal deploy experiments/2026-10-13_onpolicy_reply/run_cell_c.py
  uv run --no-project --python 3.12 --with "modal==1.6.1" python experiments/2026-10-13_onpolicy_reply/run_cell_c.py <labels json>
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-onpol-reply_cellc_kill-2026-11-13"
SCORE_VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
CACHES = {"swapped_L1": "claimc_cache_v1", "unswapped_L1": "retrain_cache_v1/level1", "unswapped_L0": "retrain_cache_v1/level0"}
LAYERS, KIND, C_GRID = [39, 51, 63], "output_end", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_NULL, MAX_ITER, MAX_ITER_CAP = 100, 5000, 80000

app = modal.App(APP)
score_vol = modal.Volume.from_name(SCORE_VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C: float) -> float:
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d: dict) -> dict:  # run_claim_c.frozen_folds, unchanged
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=2 * 3600, retries=0, volumes={"/score": score_vol}, scaledown_window=5)
def grid(tag: str, variant: str, layer: int, attempted: list) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    att, t0, out = set(attempted), time.time(), {}
    d = Path("/score") / CACHES[variant]
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    kn = np.flatnonzero([r["kind"] == KIND for r in rows])
    X = torch.from_numpy(np.load(d / f"on_L{layer}.npy")[kn]).cuda()
    runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in rows], return_inverse=True)
    n_run = n_run[kn]
    world = np.array([u.split("|", 1)[0] for u in runs])
    task = np.array([u.split("|", 1)[1] for u in runs])
    r_fold = np.array([fold_of[t] for t in task])
    train = (world == "honest_a") | ((world == "sim") & np.isin(runs, sorted(att)))
    y = (world == "sim").astype(float)
    Y = np.zeros((len(runs), 1 + N_NULL))
    Y[:, 0] = y
    for c in range(1, 1 + N_NULL):
        rng = np.random.default_rng([20261013, c])
        yc = y.copy()
        for t in sorted(set(task[train])):
            j = np.flatnonzero(train & (task == t))
            yc[j] = y[rng.permutation(j)]
        Y[:, c] = yc
    K = Y.shape[1]
    for C in C_GRID:
        p = Path("/score") / f"onpolreply_c_{tag}" / variant / f"L{layer}_{KIND}_C{C:g}.npz"
        if p.exists():
            continue
        r = {"val_auroc": np.full((5, K), np.nan), "grad_full": np.zeros((5, K)), "grad_val": np.zeros((5, K)),
             "evals": np.zeros((5, 2), int), "on_score": np.full((len(runs), K), np.nan)}
        keep = {"W": np.zeros((5, X.shape[1])), "b": np.zeros(5), "mean": np.zeros((5, X.shape[1])), "std": np.zeros((5, X.shape[1])), "sqrtC": C**0.5}
        for f in range(5):
            v = (f + 1) % 5
            for name, tr in (("full", train & (r_fold != f)), ("val", train & (r_fold != f) & (r_fold != v))):
                a = np.flatnonzero(tr[n_run])
                b_ = np.flatnonzero(r_fold[n_run] == f) if name == "full" else np.flatnonzero((train & (r_fold == v))[n_run])
                if name == "full":
                    Xd = X[a].double()
                    keep["mean"][f] = Xd.mean(0).cpu().numpy()
                    sd = Xd.std(0, unbiased=False)
                    keep["std"][f] = sd.where(sd > 0, sd.new_ones(())).cpu().numpy()
                    del Xd
                Xa, Xb = tp.standardize(X[a].double(), X[b_].double())
                Xa.mul_(C**0.5), Xb.mul_(C**0.5)
                Ya = torch.from_numpy(Y[n_run[a]]).cuda()
                mi, prev = MAX_ITER, None
                while True:  # refit only fits that used all their iterations; stop when a refit repeats
                    W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                    Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
                    loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                    loss.backward()
                    g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
                    if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                        break
                    mi, prev = mi * 4, evals
                r[f"grad_{name}"][f], r["evals"][f, int(name == "val")] = g, evals
                S = tp.decision(Xb, W, b).cpu().numpy()
                u, inv = np.unique(n_run[b_], return_inverse=True)
                M = np.stack([np.bincount(inv, weights=S[:, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                if name == "val":
                    R, Yv = rankdata(M, axis=0), Y[u]
                    n1 = Yv.sum(0)
                    r["val_auroc"][f] = ((R * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
                else:
                    r["on_score"][u] = M
                    keep["W"][f], keep["b"][f] = W[:, 0].cpu().numpy(), float(b[0])
                del Xa, Xb
        p.parent.mkdir(parents=True, exist_ok=True)
        for path, payload in ((p, dict(on_runs=runs, on_fold=r_fold, train=train, Y=Y, tol=tol_of(C), **r)), (p.with_name(f"W_L{layer}_C{C:g}.npz"), keep)):
            with open(path.with_suffix(".tmp"), "wb") as fh:
                np.savez(fh, **payload)
            os.replace(path.with_suffix(".tmp"), path)
        score_vol.commit()
        out[f"{C:g}"] = round(time.time() - t0)
        print(f"{variant} L{layer} C {C:g} done at {out[f'{C:g}']} s; evals max {r['evals'].max()}", flush=True)
    return out


if __name__ == "__main__":  # spawns on the deployed app
    lab = json.loads(Path(sys.argv[1]).read_text())["labels"]
    att = sorted(k for k, v in lab.items() if k.startswith("sim|") and v == "attempted")
    tag = hashlib.sha256((HERE / "run_cell_c.py").read_bytes() + json.dumps(att).encode()).hexdigest()[:12]
    fn = modal.Function.from_name(APP, "grid")
    print(json.dumps({"tag": tag, "attempted_sim": len(att), "calls": {f"{v}_L{l}": fn.spawn(tag, v, l, att).object_id for v in CACHES for l in LAYERS}}))
