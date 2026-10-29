"""T1 and T2 (RULE.md Amendment 6): "test why on policy protected it". Deployed Modal app, H100 GPUs (fp64 fits), claim C's grid recipe
(run_claim_c.grid: per-output rows labelled by run, sqrt(C)-rescaled fit, refit loop, nested validation fold (f+1)%5,
1 + 100 label columns shuffled within each task's training runs, out-of-fold run score = mean over included outputs).

- A prefix k keeps each run's first k replies in turn order, in training and scoring alike ("all" keeps every reply).
- Own runs: the last-token caches (swapped level 1: claimc_cache_v1; unswapped level 0: retrain_cache_v1/level0), the
  196-task frozen folds, attempted runs only for the attack arms (claim C's labels).
- Sol's runs (T2): q2's packed L63 output_end rows (run 2026-10-12_011944, data d403ff8ad9c6), worlds e1 -> run1,
  e2 -> run2, sim-attack -> sim, honest -> honest_a. Per draw d (seed 20261013 + d): 183 main tasks with all three attack
  arms, one run per task per arm; q2's task folds.
- jobs: T1a (variant, layer): cells b (run1 vs honest_a) and c (sim vs honest_a) at prefixes 1 and 3 (shuffle seeds
  [20261010, c] and [20261013, c] as before); T1b (variant): sim vs run1 and the control run2 vs run1 at L63, prefixes
  1, 3, all (seed [20261014, c]); T2 (draw): the same two comparisons on Sol's sample.
Records: /score/onpolreply_t1t2_<tag>/<job>/L<l>_output_end_C<C>.npz (claim C's format), read by analyze_t1t2.py.

  modal deploy experiments/2026-10-13_onpolicy_reply/run_t1t2.py
  uv run --no-project --python 3.12 --with "modal==1.6.1" python experiments/2026-10-13_onpolicy_reply/run_t1t2.py <labels json>
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-onpol-reply_t1t2_kill-2026-11-13"
SCORE, Q2WORK = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q2-qwen35-reply_work_kill-2026-11-12"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
CACHE = {"swapped_L1": "claimc_cache_v1", "unswapped_L0": "retrain_cache_v1/level0"}
C_GRID, N_NULL, MAX_ITER, MAX_ITER_CAP, N_TASKS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 100, 5000, 80000, 183
Q2ARM = {"real-attack-e1": "run1", "real-attack-e2": "run2", "sim-attack": "sim", "real-honest": "honest_a"}

app = modal.App(APP)
score_vol, q2_vol = modal.Volume.from_name(SCORE, version=2), modal.Volume.from_name(Q2WORK).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C):
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d):  # run_claim_c.frozen_folds, unchanged
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


def own_data(variant, layer):
    import numpy as np
    d = Path("/score") / CACHE[variant]
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    kn = [i for i, r in enumerate(rows) if r["kind"] == "output_end" and r["world"] in ("run1", "run2", "sim", "honest_a", "honest_b")]
    X = np.load(d / f"on_L{layer}.npy", mmap_mode="r")
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    return (np.asarray(X[kn]), [f"{rows[i]['world']}|{rows[i]['task']}" for i in kn], [tuple(rows[i]["order"]) for i in kn],
            lambda u: fold_of[u.split("|", 1)[1]])


def sol_data(draw):
    import numpy as np
    d = Path("/q2/data/d403ff8ad9c6")
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    by = {}
    for i, r in enumerate(runs):
        if r["arm"] in Q2ARM:
            by.setdefault(r["main_task"], {}).setdefault(Q2ARM[r["arm"]], []).append(i)
    fold = {r["main_task"]: r["fold"] for r in runs}
    ok = sorted(m for m, a in by.items() if all(w in a for w in ("run1", "run2", "sim")))
    rng = np.random.default_rng(20261013 + draw)
    tasks = sorted(rng.choice(ok, N_TASKS, replace=False).tolist())
    pick = {}
    for m in tasks:
        for w in ("run1", "run2", "sim"):
            pick[int(rng.choice(sorted(by[m][w])))] = f"{w}|{m}"
    rr, row = t["output_end__run"], t["output_end__row"]
    sel = np.flatnonzero(np.isin(rr, sorted(pick)))
    X = np.load(d / "packed" / "L63_output_end.npy", mmap_mode="r")
    return np.asarray(X[sel]), [pick[int(rr[i])] for i in sel], [(int(row[i]), 0) for i in sel], lambda u: fold[u.split("|", 1)[1]], tasks


@app.function(image=image, gpu="H100!", cpu=4, memory=49152, timeout=2 * 3600, retries=0, volumes={"/score": score_vol, "/q2": q2_vol})
def run_jobs(tag: str, source: str, layer: int, jobs: list, attempted: list) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    extra = {}
    if source.startswith("sol_draw"):
        Xall, run_of, order, fold_fn, extra["tasks"] = sol_data(int(source[len("sol_draw"):]))
    else:
        Xall, run_of, order, fold_fn = own_data(source, layer)
    run_of, att = np.array(run_of), set(attempted)
    rank = np.zeros(len(run_of), int)  # each row's order within its run
    for u in set(run_of.tolist()):
        m = np.flatnonzero(run_of == u)
        rank[sorted(m, key=lambda i: order[i])] = np.arange(len(m))
    out = {}
    for name, pos, neg, k, seed in jobs:
        keep = np.flatnonzero(rank < k) if k else np.arange(len(run_of))
        runs, n_run = np.unique(run_of[keep], return_inverse=True)
        X = torch.from_numpy(Xall[keep]).cuda()
        world = np.array([u.split("|", 1)[0] for u in runs])
        task = np.array([u.split("|", 1)[1] for u in runs])
        r_fold = np.array([fold_fn(u) for u in runs])
        is_att = lambda w: (world == w) & (np.isin(runs, sorted(att)) if (att and not source.startswith("sol")) else True)  # noqa: E731
        train = (is_att(pos) if pos != "honest_a" else (world == pos)) | (is_att(neg) if neg != "honest_a" else (world == neg))
        y = (world == pos).astype(float)
        Y = np.zeros((len(runs), 1 + N_NULL))
        Y[:, 0] = y
        for c in range(1, 1 + N_NULL):
            rng = np.random.default_rng([seed, c])
            yc = y.copy()
            for t in sorted(set(task[train])):
                j = np.flatnonzero(train & (task == t))
                yc[j] = y[rng.permutation(j)]
            Y[:, c] = yc
        K = Y.shape[1]
        for C in C_GRID:
            p = Path("/score") / f"onpolreply_t1t2_{tag}" / name / f"L{layer}_output_end_C{C:g}.npz"
            if p.exists():
                continue
            r = {"val_auroc": np.full((5, K), np.nan), "grad_full": np.zeros((5, K)), "grad_val": np.zeros((5, K)),
                 "evals": np.zeros((5, 2), int), "on_score": np.full((len(runs), K), np.nan)}
            for f in range(5):
                v = (f + 1) % 5
                for nm, tr in (("full", train & (r_fold != f)), ("val", train & (r_fold != f) & (r_fold != v))):
                    a = np.flatnonzero(tr[n_run])
                    b_ = np.flatnonzero(r_fold[n_run] == f) if nm == "full" else np.flatnonzero((train & (r_fold == v))[n_run])
                    Xa, Xb = tp.standardize(X[a].double(), X[b_].double())
                    Xa.mul_(C**0.5), Xb.mul_(C**0.5)
                    Ya = torch.from_numpy(Y[n_run[a]]).cuda()
                    mi, prev = MAX_ITER, None
                    while True:  # as run_claim_c.grid
                        W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                        Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
                        loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                        loss.backward()
                        g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
                        if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                            break
                        mi, prev = mi * 4, evals
                    r[f"grad_{nm}"][f], r["evals"][f, int(nm == "val")] = g, evals
                    S = tp.decision(Xb, W, b).cpu().numpy()
                    u, inv = np.unique(n_run[b_], return_inverse=True)
                    M = np.stack([np.bincount(inv, weights=S[:, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                    if nm == "val":
                        R, Yv = rankdata(M, axis=0), Y[u]
                        n1 = Yv.sum(0)
                        r["val_auroc"][f] = ((R * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
                    else:
                        r["on_score"][u] = M
                    del Xa, Xb
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p.with_suffix(".tmp"), "wb") as fh:
                np.savez(fh, on_runs=runs, on_fold=r_fold, train=train, Y=Y, tol=tol_of(C), **r)
            os.replace(p.with_suffix(".tmp"), p)
            score_vol.commit()
        out[name] = {"runs": len(runs), "rows": len(keep), "train_runs": int(train.sum())}
        del X
        torch.cuda.empty_cache()
    return out | extra


def plan():
    j = []
    for v in CACHE:
        for layer in (39, 51, 63):
            j.append((v, layer, [(f"t1a_{v}_{cell}_k{k}", w, "honest_a", k, s) for cell, w, s in (("b", "run1", 20261010), ("c", "sim", 20261013)) for k in (1, 3)]))
        j.append((v, 63, [(f"t1b_{v}_{cmp}_k{k or 'all'}", pos, "run1", k, 20261014) for cmp, pos in (("real_vs_sim", "sim"), ("control", "run2")) for k in (1, 3, None)]))
    for d in range(5):
        j.append((f"sol_draw{d}", 63, [(f"t2_sol_d{d}_{cmp}_k{k or 'all'}", pos, "run1", k, 20261014) for cmp, pos in (("real_vs_sim", "sim"), ("control", "run2")) for k in (1, 3, None)]))
    return j


if __name__ == "__main__":
    lab = json.loads(Path(sys.argv[1]).read_text())["labels"]
    att = sorted(k for k, v in lab.items() if v == "attempted")
    tag = hashlib.sha256((HERE / "run_t1t2.py").read_bytes() + json.dumps(att).encode()).hexdigest()[:12]
    fn = modal.Function.from_name(APP, "run_jobs")
    calls = {f"{s}_L{l}": fn.spawn(tag, s, l, jobs, att) for s, l, jobs in plan()}
    print(json.dumps({"tag": tag, "containers": len(calls)}))
    res = {k: c.get() for k, c in calls.items()}
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dest = HERE / "results" / f"t1t2_run_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    dest.open("x").write(json.dumps({"tag": tag, "containers": res}))
    print(dest.name)
