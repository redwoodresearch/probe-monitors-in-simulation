"""Few real attacks in fake/fake training (RULE.md; POST HOC). One H100 container per layer, d9's fix-full packs
(sce_fixfull/2026-10-13_152618 on the work Volume, read only). Per nested model (f, v): k = 0 exactly as d9's b″ fit
(alone), then 15 variants (k in 10/30/100 x draws 0-4) as one masked batched fit; nested run scores out, no text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run --detach \\
      experiments/2026-10-13_few_real_attacks_POSTHOC/run_fewreal.py --scores <d9 sce_fixfull_scores npz> --out <results npz>
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
APP = "probe-shift_aswin-surya_sub-fewreal_fit_kill-2026-11-13"
WORK, STAMP = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "2026-10-13_152618"
D9_SCORES_SHA256 = "48fc5eb52546f8884bf1083a3d8530a6c80624b52c619678edf72c28443826be"
C, MAX_ITER, LAYERS, KS, DRAWS = 0.01, 1000, (48, 63), (10, 30, 100), 5
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
POS, NEG, E1 = "sim-attack-fullfix", "sim-honest-fullfix", "real-attack-e1"
app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scipy==1.16.2")
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, volumes={"/work": work.read_only()})
def fit_layer(layer: int, arm: list, task: list, fold: list) -> dict:
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool

    t0 = time.time()
    d = Path("/work/sce_fixfull") / STAMP
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    X = torch.from_numpy(np.load(d / f"L{layer}.npy")).cuda().float()
    arm, task, fold = np.array(arm), np.array(task), np.array(fold, int)
    n = len(arm)
    y = np.where(arm == POS, 1.0, np.where(arm == NEG, 0.0, np.nan))
    trainable = ~np.isnan(y)
    nested0 = np.full((5, 5, n), np.nan, np.float32)
    nestedk = np.full((len(KS) * DRAWS, 5, 5, n), np.nan, np.float32)
    counts, evals = {}, {"k0": [], "batched": []}

    def masked_fit(Xs, Y, M):  # each column k: C * sum_i M_ik logloss + 0.5 ||w_k||^2 (tp.fit's objective, per-column rows)
        nn_, dd = Xs.shape
        W = torch.zeros(dd, Y.shape[1], device=Xs.device, requires_grad=True)
        b = torch.zeros(Y.shape[1], device=Xs.device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=MAX_ITER, tolerance_grad=1e-5, tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")
        steps = [0]

        def closure():
            opt.zero_grad()
            loss = (C * (M * F.binary_cross_entropy_with_logits(Xs @ W + b, Y, reduction="none")).sum() + 0.5 * (W * W).sum()) / nn_
            loss.backward()
            steps[0] += 1
            return loss

        opt.step(closure)
        return W.detach(), b.detach(), steps[0]

    for f, v in PAIRS:
        te = (fold == f) | (fold == v)
        base_runs = trainable & ~te
        tr = np.flatnonzero(base_runs[r_run])
        Xtr = X[torch.from_numpy(tr).cuda()]
        mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        te_rows = np.flatnonzero(te[r_run])
        Xte = (X[torch.from_numpy(te_rows).cuda()] - mean) / std
        W0, b0, ne = tp.fit((Xtr - mean) / std, torch.from_numpy(y[r_run[tr]]).cuda().float(), C, MAX_ITER)  # d9's b″ fit, exactly
        evals["k0"].append(int(ne))
        S = pool(r_run[te_rows], tp.decision(Xte, W0, b0).double().cpu().numpy(), n, "mean")
        nested0[f, v] = np.where(te, S[:, 0], np.nan)
        # added variants: whole e1 tasks from the training folds only
        cand_tasks = np.array(sorted(set(task[(arm == E1) & ~te])))
        add = []
        for k in KS:
            for dr in range(DRAWS):
                rng = np.random.default_rng([20261016, k, dr, f, v])
                chosen, got = set(), 0
                for t in rng.permutation(cand_tasks):
                    if got >= k:
                        break
                    chosen.add(t)
                    got += int(((arm == E1) & (task == t) & ~te).sum())
                add.append((arm == E1) & np.isin(task, sorted(chosen)) & ~te)
                counts[f"k{k}_d{dr}_f{f}_v{v}"] = got
        e1_runs = np.any(add, 0)
        e1_rows = np.flatnonzero(e1_runs[r_run])
        rows = np.concatenate([tr, e1_rows])
        Xb = (X[torch.from_numpy(rows).cuda()] - mean) / std
        Y = np.concatenate([np.repeat(y[r_run[tr]][:, None], len(add), 1), np.ones((len(e1_rows), len(add)))])
        M = np.concatenate([np.ones((len(tr), len(add))), np.stack([a[r_run[e1_rows]] for a in add], 1).astype(float)])
        Wb, bb, ne = masked_fit(Xb, torch.from_numpy(Y).cuda().float(), torch.from_numpy(M).cuda().float())
        evals["batched"].append(int(ne))
        del Xb, Xtr
        Sb = pool(r_run[te_rows], tp.decision(Xte, Wb, bb).double().cpu().numpy(), n, "mean")
        for j in range(len(add)):
            nestedk[j, f, v] = np.where(te, Sb[:, j], np.nan)
    if not (np.isfinite(nested0[~np.isnan(nested0)]).all()):
        raise RuntimeError("non-finite k = 0 scores")
    return {"layer": layer, "nested0": nested0, "nestedk": nestedk, "counts": counts, "evals": evals,
            "fits_at_cap": {k: sum(x >= MAX_ITER for x in e) for k, e in evals.items()}, "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(scores: str = "", out: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if hashlib.sha256(Path(scores).read_bytes()).hexdigest() != D9_SCORES_SHA256:
        raise SystemExit("not d9's fix-full scores file")
    Z = np.load(scores)
    arm, task, fold = (Z[f"runs__{k}"].tolist() for k in ("arm", "main_task", "fold"))
    res = list(fit_layer.starmap([(L, arm, task, fold) for L in LAYERS]))
    with open(out, "xb") as fh:
        np.savez(fh, **{f"L{r['layer']}__k0": r["nested0"] for r in res}, **{f"L{r['layer']}__k": r["nestedk"] for r in res})
    rec = {"git": git("rev-parse", "HEAD"), "app": APP, "stamp": STAMP, "settings": {"C": C, "max_iter": MAX_ITER, "ks": KS, "draws": DRAWS},
           "layers": [{k: r[k] for k in ("layer", "counts", "evals", "fits_at_cap", "seconds")} for r in res],
           "scores_file": Path(out).name, "scores_sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest()}
    Path(out[:-4] + ".run.json").open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("scores_file", "scores_sha256")} | {"fits_at_cap": [r["fits_at_cap"] for r in res], "seconds": [r["seconds"] for r in res]}))
