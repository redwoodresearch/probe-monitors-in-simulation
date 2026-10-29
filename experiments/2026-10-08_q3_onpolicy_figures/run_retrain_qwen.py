"""POST-HOC step 2 of "try to rescue it" (plan Amendment 6): the attack detector retrained on Qwen's own runs, out of
fold, from the retrain caches (no new extraction). Modal: one H100 per (layer, readout); records under a new tag.

Training: real attack run 1 (label 1; attempted-only cells drop runs not labelled attempted) vs honest_a (label 0), every
output's readout labelled with its run; run score = mean over outputs. Fitter: run_retrain_grid.py's (the pinned claim
B loop: the rescaled problem at C = 1 on sqrt(C)·X, float64, converged when each column's largest gradient entry <=
1e-4·min(sqrt(C), 1), refits up to 80,000 iterations). 1 + 100 label columns: the real labels, then shuffles of run 1 vs
honest_a among each main task's training runs (seed [20261010, d]; the same in every container). Folds: the 196-task
scale-up's frozen task folds (frozen_folds on pilot_tasks_amendment9_all196.json). For held-out fold k, the full model
trains on the other 4 folds and scores every Qwen run of fold k (all worlds), by all outputs and by its first K_LEN
readouts (runs with fewer than K_LEN tool outputs in the level-0 inventory get none); the validation model trains
without fold (k + 1) % 5 and scores it (validation AUROC, run 1 vs honest_a), on which the analysis chooses layer x
readout x C. Each (level, cell, C) record is written as it finishes. `--timing`: Aswin's 10:51 PT rule: one container,
level 0, all runs, C 1e-6, 1e-2 and 1, every label column a shuffle; records under a separate tag never analyzed.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run --detach \
      experiments/2026-10-08_q3_onpolicy_figures/run_retrain_qwen.py::main --labels <label files, comma-separated> [--timing]
"""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3onpol_retrainqwen_kill-2026-11-10"
SCORE_VOL, CACHE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "retrain_cache_v1"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
LAYERS, KINDS, CELLS, LEVELS = [39, 51, 63], ["output_end", "call_end"], ["all", "attempted"], ["level0", "level1"]
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_NULL, MAX_ITER, MAX_ITER_CAP, K_LEN = 100, 5000, 80000, 11

app = modal.App(APP)
score_vol = modal.Volume.from_name(SCORE_VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C: float) -> float:
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d: dict) -> dict:  # run_score_onpolicy.frozen_folds (amendment 25), on a parsed task file
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=2 * 3600, retries=0, volumes={"/score": score_vol}, scaledown_window=5)
def grid(tag: str, layer: int, kind: str, attempted: list, timing: bool = False) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    t196 = json.loads((Path("/root") / T196).read_text())
    fold_of, main_of = frozen_folds(t196), {c["key"]: c["main"] for c in t196["cells"]}
    att = set(attempted)  # "run1|<task>" for every run-1 run labelled attempted
    counts0 = json.loads((Path("/score") / CACHE / "level0" / f"inventory.L{layer}.json").read_text())
    t0, out = time.time(), {}
    levels, cells, Cs = (["level0"], ["all"], [1e-6, 1e-2, 1.0]) if timing else (LEVELS, CELLS, C_GRID)
    for level in levels:
        d = Path("/score") / CACHE / level
        rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
        kn = np.flatnonzero([r["kind"] == kind for r in rows])
        X = torch.from_numpy(np.load(d / f"on_L{layer}.npy")[kn]).cuda()
        runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in rows], return_inverse=True)  # both readouts' runs
        n_run = n_run[kn]
        world = np.array([u.split("|", 1)[0] for u in runs])
        r_fold = np.array([fold_of[u.split("|", 1)[1]] for u in runs])
        r_main = np.array([main_of[u.split("|", 1)[1]] for u in runs])
        rank = np.zeros(len(kn), int)  # each row's order within its run
        for u in range(len(runs)):
            m = np.flatnonzero(n_run == u)
            rank[sorted(m, key=lambda i: tuple(rows[kn[i]]["order"]))] = np.arange(len(m))
        long_enough = np.array([counts0.get(u, 0) >= K_LEN for u in runs])
        for cell in cells:
            train = (world == "honest_a") | ((world == "run1") & (np.isin(runs, sorted(att)) if cell == "attempted" else True))
            y = (world == "run1").astype(float)
            Y = np.zeros((len(runs), 1 + N_NULL))
            Y[:, 0] = y
            for c in range(0 if timing else 1, 1 + N_NULL):  # --timing: column 0 is a shuffle too (no real labels)
                rng = np.random.default_rng([20261010, c + (1000 if timing and c == 0 else 0)])
                yc = y.copy()
                for m in sorted(set(r_main[train])):
                    j = np.flatnonzero(train & (r_main == m))
                    yc[j] = y[rng.permutation(j)]
                Y[:, c] = yc
            K = Y.shape[1]
            for C in Cs:
                p = Path("/score") / f"retrain_qwen_{tag}{'_TIMING' if timing else ''}" / level / cell / f"L{layer}_{kind}_C{C:g}.npz"
                if p.exists():
                    continue
                r = {"val_auroc": np.full((5, K), np.nan), "grad_full": np.zeros((5, K)), "grad_val": np.zeros((5, K)),
                     "evals": np.zeros((5, 2), int), "on_score": np.full((len(runs), K), np.nan), "on_score_k": np.full((len(runs), K), np.nan)}
                for f in range(5):
                    v = (f + 1) % 5
                    for name, tr in (("full", train & (r_fold != f)), ("val", train & (r_fold != f) & (r_fold != v))):
                        a = np.flatnonzero(tr[n_run])
                        b_ = np.flatnonzero(r_fold[n_run] == f) if name == "full" else np.flatnonzero((train & (r_fold == v))[n_run])
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
                        if name == "val":
                            u, inv = np.unique(n_run[b_], return_inverse=True)
                            M = np.stack([np.bincount(inv, weights=S[:, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                            R, Yv = rankdata(M, axis=0), Y[u]
                            n1 = Yv.sum(0)
                            r["val_auroc"][f] = ((R * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
                        else:
                            for col, sel in (("on_score", np.ones(len(b_), bool)), ("on_score_k", (rank[b_] < K_LEN) & long_enough[n_run[b_]])):
                                u, inv = np.unique(n_run[b_][sel], return_inverse=True)
                                M = np.stack([np.bincount(inv, weights=S[sel, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                                full_k = np.bincount(inv) == K_LEN if col == "on_score_k" else np.ones(len(u), bool)
                                r[col][u[full_k]] = M[full_k]
                        del Xa, Xb
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p.with_suffix(".tmp"), "wb") as fh:
                    np.savez(fh, on_runs=runs, on_fold=r_fold, long_enough=long_enough, train=train, Y=Y, tol=tol_of(C), **r)
                os.replace(p.with_suffix(".tmp"), p)
                score_vol.commit()
                out[f"{level} {cell} {C:g}"] = round(time.time() - t0)
                print(f"L{layer} {kind} {level} {cell} C {C:g} done at {out[f'{level} {cell} {C:g}']} s; evals max {r['evals'].max()}", flush=True)
    return out


@app.local_entrypoint()
def main(labels: str, timing: bool = False):
    attempted = []
    for f in labels.split(","):  # make_phase2_attempt_labels.py outputs; run 1 is every folder's attack arm but the second batch's
        if "second_attack" in Path(f).name:
            continue
        attempted += [f"run1|{r['task']}" for r in map(json.loads, Path(f).read_text().splitlines()) if r["arm"] == "attack" and r.get("label") == "attempted"]
    if len(attempted) != len(set(attempted)):
        raise ValueError("a run-1 task is labelled twice")
    tag = hashlib.sha256(Path(__file__).read_bytes() + json.dumps(sorted(attempted)).encode()).hexdigest()[:12]
    print(f"tag {tag}; run-1 runs labelled attempted: {len(attempted)}", flush=True)
    jobs = [(tag, 51, "output_end", attempted, True)] if timing else [(tag, l, k, attempted, False) for l in LAYERS for k in KINDS]
    for r in grid.starmap(jobs):
        print(json.dumps(r), flush=True)
