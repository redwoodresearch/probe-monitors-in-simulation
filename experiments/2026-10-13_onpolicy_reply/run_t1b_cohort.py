"""T1b refit on the matched cohort (RULE_T1B_COHORT.md; POST HOC). A copy of run_t1t2.py's T1b path (@ 3cdcf9a2) with one change:
the cache rows are filtered to the 183 cohort tasks (all three attack arms attempted) before training, shuffling and
scoring. Then, on Modal CPU, T2's readings are recomputed from the old records (gate: equal to the committed T1/T2 values
within 1e-9) and from the new ones, with Sol's T2 records reused unchanged.

  modal deploy experiments/2026-10-13_onpolicy_reply/run_t1b_cohort.py
  uv run --no-project --python 3.12 --with "modal==1.6.1" python experiments/2026-10-13_onpolicy_reply/run_t1b_cohort.py \\
      <labels json> <t1t2 run record json> <committed t1t2 result json>
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-onpol-reply_t1bcoh_kill-2026-11-13"
SCORE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
OLD = "onpolreply_t1t2_a4c99806deae"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
CACHE = {"swapped_L1": "claimc_cache_v1", "unswapped_L0": "retrain_cache_v1/level0"}
C_GRID, N_NULL, MAX_ITER, MAX_ITER_CAP, MAX_FAIL = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 100, 5000, 80000, 0.05
KS = (1, 3, None)

app = modal.App(APP)
score_vol = modal.Volume.from_name(SCORE, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scipy")


def tol_of(C):
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d):  # run_claim_c.frozen_folds, unchanged
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


def own_data(variant, layer, cohort):  # run_t1t2.own_data with the cohort filter (the only change)
    import numpy as np
    d = Path("/score") / CACHE[variant]
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    kn = [i for i, r in enumerate(rows) if r["kind"] == "output_end" and r["world"] in ("run1", "run2", "sim", "honest_a", "honest_b") and r["task"] in cohort]
    X = np.load(d / f"on_L{layer}.npy", mmap_mode="r")
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    return (np.asarray(X[kn]), [f"{rows[i]['world']}|{rows[i]['task']}" for i in kn], [tuple(rows[i]["order"]) for i in kn],
            lambda u: fold_of[u.split("|", 1)[1]])


@app.function(image=image, gpu="H100!", cpu=4, memory=49152, timeout=3600, retries=0, volumes={"/score": score_vol})
def run_jobs(tag: str, source: str, layer: int, jobs: list, attempted: list, cohort: list) -> dict:  # run_t1t2.run_jobs, own runs only
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    Xall, run_of, order, fold_fn = own_data(source, layer, set(cohort))
    run_of, att = np.array(run_of), set(attempted)
    rank = np.zeros(len(run_of), int)
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
        is_att = lambda w: (world == w) & np.isin(runs, sorted(att))  # noqa: E731
        train = is_att(pos) | is_att(neg)
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
            p = Path("/score") / f"onpolreply_t1bcoh_{tag}" / name / f"L{layer}_output_end_C{C:g}.npz"
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
        out[name] = {"runs": len(runs), "rows": len(keep), "train_runs": int(train.sum()), "train_tasks": len(set(task[train]))}
        del X
        torch.cuda.empty_cache()
    return out


def auc(pos, neg):  # AUROC with ties counted half (analyze_retrain.wauc at unit weights)
    import numpy as np
    from scipy.stats import rankdata
    R = rankdata(np.concatenate([pos, neg]))
    return float((R[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/score": score_vol})
def analyze(new_tag: str, cohort: list, sol_tasks: dict, committed: dict) -> dict:  # analyze_t1t2.select_l63 + auc_block, readings as Amendment 7
    import numpy as np

    score_vol.reload()

    def block(d, tasks):
        R = {C: dict(np.load(d / f"L63_output_end_C{C:g}.npz")) for C in C_GRID}
        r0 = R[C_GRID[0]]
        if any(not all(np.array_equal(r[k], r0[k]) for k in ("on_runs", "Y", "on_fold", "train")) for r in R.values()):
            raise SystemExit(f"{d.name}: records disagree")
        ok = np.stack([(R[C]["grad_full"] <= R[C]["tol"]) & (R[C]["grad_val"] <= R[C]["tol"]) for C in C_GRID])
        share = float(1 - np.concatenate([np.stack([R[C]["grad_full"] <= R[C]["tol"] for C in C_GRID]).ravel(),
                                          np.stack([R[C]["grad_val"] <= R[C]["tol"] for C in C_GRID]).ravel()]).mean())
        V = np.stack([np.where(ok[i] & np.isfinite(R[C]["val_auroc"]), R[C]["val_auroc"], -np.inf) for i, C in enumerate(C_GRID)])
        pick, usable = V.argmax(0), np.isfinite(V.max(0))
        if share > MAX_FAIL or not usable.all():
            return {"invalid": f"unconverged share {share:.4f}; unusable fold-columns {int((~usable).sum())}"}
        complete = np.all(np.stack([np.isfinite(R[C]["on_score"]).all(1) for C in C_GRID]), 0)
        Z = np.full(r0["on_score"].shape, np.nan)
        for f in range(5):
            for col in range(pick.shape[1]):
                mine = (r0["on_fold"] == f) & complete
                Z[mine, col] = R[C_GRID[pick[f, col]]]["on_score"][mine, col]
        runs = r0["on_runs"].astype(str)
        task = np.array([u.split("|", 1)[1] for u in runs])
        m = r0["train"] & np.isin(task, sorted(tasks)) & np.isfinite(Z[:, 0])
        one = lambda c: auc(Z[m, c][r0["Y"][m, c] == 1], Z[m, c][r0["Y"][m, c] == 0])  # noqa: E731
        obs = one(0)
        null = [one(c) for c in range(1, Z.shape[1]) if np.isfinite(Z[m, c]).all()]
        return {"auroc": obs, "runs": int(m.sum()), "train_runs": int(r0["train"].sum()), "null_p95": float(np.percentile(null, 95)), "null_n": len(null),
                "p": (1 + sum(x >= obs for x in null)) / (1 + len(null)), "choices_C": [float(C_GRID[pick[f, 0]]) for f in range(5)], "unconverged_share": share}

    root = Path("/score")
    sol = {f"d{dr}_{cmp}_{k}": block(root / OLD / f"t2_sol_d{dr}_{cmp}_{k}", set(sol_tasks[str(dr)])) for dr in range(5) for cmp in ("real_vs_sim", "control") for k in ("k1", "k3", "kall")}
    res = {"sol": sol, "before": {}, "after": {}, "readings": {}, "gate": {}}
    for when, base in (("before", OLD), ("after", f"onpolreply_t1bcoh_{new_tag}")):
        for v in CACHE:
            for cmp in ("real_vs_sim", "control"):
                for k in ("k1", "k3", "kall"):
                    res[when][f"{v}_{cmp}_{k}"] = block(root / base / f"t1b_{v}_{cmp}_{k}", set(cohort))
    worst = 0.0
    for key, c in committed["t1b"].items():
        for f in ("auroc", "p", "null_p95"):
            worst = max(worst, abs(res["before"][key][f] - c[f]))
    for key, c in committed["t2_sol"].items():
        worst = max(worst, abs(sol[key]["auroc"] - c["auroc"]), abs(sol[key]["p"] - c["p"]))
    res["gate"] = {"max_abs_diff_vs_committed": worst, "passes": worst <= 1e-9}
    if not res["gate"]["passes"]:
        return res
    for when in ("before", "after"):
        for k in ("k1", "k3", "kall"):
            for v in CACHE:
                own = res[when][f"{v}_real_vs_sim_{k}"]["auroc"]
                diffs = [sol[f"d{dr}_real_vs_sim_{k}"]["auroc"] - own for dr in range(5)]
                rev = "reversed: own runs carry MORE origin information than Sol's in every draw" + (" (every difference <= -0.10)" if all(x <= -0.10 for x in diffs) else "")
                res["readings"][f"{when}_{v}_{k}"] = {"sol_minus_own": diffs, "reading": "supports H2 (origin information accessible to this classifier is lower on its own runs; cause not identified)"
                                                      if all(x >= 0.10 for x in diffs) else "against H2" if all(abs(x) < 0.10 for x in diffs) else rev if all(x < 0 for x in diffs)
                                                      else "mixed across draws"}
    res["changed_readings"] = [k for k in ("k1", "k3", "kall") for v in CACHE if res["readings"][f"before_{v}_{k}"]["reading"] != res["readings"][f"after_{v}_{k}"]["reading"]]
    return res


if __name__ == "__main__":
    lab = json.loads(Path(sys.argv[1]).read_text())["labels"]
    run_rec, committed = json.loads(Path(sys.argv[2]).read_text()), json.loads(Path(sys.argv[3]).read_text())
    att = sorted(k for k, v in lab.items() if v == "attempted")
    tasks = {k.split("|", 1)[1] for k in lab}
    cohort = sorted(t for t in tasks if all(lab.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim")))
    if len(cohort) != 183:
        raise SystemExit(f"cohort {len(cohort)} != 183")
    if sys.argv[4:5] == ["--analyze-only"]:  # the GPU jobs already ran (records under onpolreply_t1bcoh_<tag>); only the CPU analysis
        tag, jobs = sys.argv[5], None
    else:
        tag = hashlib.sha256((HERE / "run_t1b_cohort.py").read_bytes() + json.dumps(att).encode()).hexdigest()[:12]
        fn = modal.Function.from_name(APP, "run_jobs")
        calls = {f"{v}_{cmp}": fn.spawn(tag, v, 63, [(f"t1b_{v}_{cmp}_k{k or 'all'}", pos, "run1", k, 20261014) for k in KS], att, cohort)
                 for v in CACHE for cmp, pos in (("real_vs_sim", "sim"), ("control", "run2"))}
        print(json.dumps({"tag": tag, "containers": len(calls)}), flush=True)
        jobs = {k: c.get() for k, c in calls.items()}
    print(json.dumps(jobs), flush=True)
    sol_tasks = {str(dr): run_rec["containers"][f"sol_draw{dr}_L63"]["tasks"] for dr in range(5)}
    res = modal.Function.from_name(APP, "analyze").remote(tag, cohort, sol_tasks, {"t1b": committed["t1b"], "t2_sol": committed["t2_sol"]})
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dest = HERE / "results" / f"t1b_cohort_POSTHOC_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE_T1B_COHORT.md)", "tag": tag, "jobs": jobs, "result": res}, indent=1))
    print(dest.name, json.dumps({"gate": res["gate"], "changed_readings": res.get("changed_readings")}))
