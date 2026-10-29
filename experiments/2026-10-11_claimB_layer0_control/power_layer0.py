"""Power (a sensitivity analysis) for the layer-0 control (PLAN.md), $0, numbers only: how likely is a clear answer from
"S = AUROC(step A's probe) - AUROC(layer-0 probe); beyond the linear embedding baseline if S's lower one-sided 95% bound
> M; noninferior if its upper bound < M; else inconclusive", with M = 0.05, on the stored runs? Conditional on E's
plant-2 control passing (the "beyond" verdict also requires it; not simulated here).

Step A's side is real: its per-output test scores from the refit of its chosen probes
(claimB/level1/output_scores_A_e402bcc321a8.jsonl.gz on the activations Volume; sha256 checked below), averaged per run
(the run score step A used). The layer-0 side is simulated, since no layer-0 probe has been fit: per run,
  s = delta * y + rho * r_A + sqrt(1 - rho^2) * (sqrt(tau) * u_task + sqrt(1 - tau) * e_run),
where y is the arm (1 = fixed sim), r_A is A's within-arm residual (standardized), tau is A's share of residual variance
between main tasks (so the simulated probe's errors cluster by task as A's do), and delta is set by bisection so that
the layer-0 AUROC averages the target. Each replicate draws u and e, then bootstraps main tasks (paired: one resample
for both probes; 2,000 draws as planned) for S's one-sided 95% bounds. tau is the one-way variance-component (ANOVA,
unequal cluster sizes) share; the naive share of task-mean variance is reported beside it. Continuous, independent run
noise ignores E's token-level ties and shared weights, so the result is a sensitivity analysis, not a guarantee; each
probability carries Monte Carlo error (about +-5.5 points at 0.8 with 200 replicates).

Sets: the 5 test environments (946 runs, the 0.822's set) and all 10 environments (1,878 runs, out-of-fold everywhere).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      python experiments/2026-10-11_claimB_layer0_control/power_layer0.py
"""

import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SCORES = "claimB/level1/output_scores_A_e402bcc321a8.jsonl.gz"
SCORES_SHA = "ae246e08e5b0390920826b29a8488c4b6352601b17ccd283f1a4c799123f3de6"
TEST_ENVS = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
M, BOUND = 0.05, 0.95
TARGETS = [0.50, 0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82]
RHOS = [0.0, 0.3, 0.6]
REPS, BOOT, SEED = 200, 2000, 20261011


def auc_weighted(s, y, W):
    """AUROC of scores s for labels y under each row of bootstrap weights W (B x n); ties are measure-zero here."""
    o = np.argsort(s, kind="stable")
    y, W = y[o], W[:, o]
    wneg = W * (y == 0)
    below = np.cumsum(wneg, axis=1) - wneg  # negative weight strictly below each position
    num = (W * (y == 1) * below).sum(1)
    return num / ((W * (y == 1)).sum(1) * wneg.sum(1))


def auc(s, y):
    return float(auc_weighted(s, y, np.ones((1, len(y))))[0])


def boot_weights(task_idx, n_tasks, rng, B):
    counts = np.zeros((B, n_tasks))
    np.add.at(counts, (np.arange(B)[:, None], rng.integers(0, n_tasks, (B, n_tasks))), 1)
    return counts[:, task_idx]


def sim_scores(delta, rho, tau, y, r, task_idx, n_tasks, rng):
    u = rng.standard_normal(n_tasks)[task_idx]
    e = rng.standard_normal(len(y))
    return delta * y + rho * r + np.sqrt(1 - rho**2) * (np.sqrt(tau) * u + np.sqrt(1 - tau) * e)


def calibrate(target, rho, tau, y, r, task_idx, n_tasks, seed):
    lo, hi = -1.0, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rng = np.random.default_rng(seed)
        m = np.mean([auc(sim_scores(mid, rho, tau, y, r, task_idx, n_tasks, rng), y) for _ in range(30)])
        lo, hi = (mid, hi) if m < target else (lo, mid)
    return (lo + hi) / 2


def one_config(args):
    name, target, rho, tau, sA, y, r, task_idx, n_tasks, seed = args
    delta = calibrate(target, rho, tau, y, r, task_idx, n_tasks, seed)
    rng = np.random.default_rng(seed + 1)
    out = {"beyond": 0, "noninferior": 0, "inconclusive": 0}
    S_hat, aucs = [], []
    for _ in range(REPS):
        s0 = sim_scores(delta, rho, tau, y, r, task_idx, n_tasks, rng)
        W = boot_weights(task_idx, n_tasks, rng, BOOT)
        S = auc_weighted(sA, y, W) - auc_weighted(s0, y, W)
        lo, hi = np.quantile(S, 1 - BOUND), np.quantile(S, BOUND)
        out["beyond" if lo > M else "noninferior" if hi < M else "inconclusive"] += 1
        S_hat.append(auc(sA, y) - auc(s0, y))
        aucs.append(auc(s0, y))
    p = {k: v / REPS for k, v in out.items()}
    return {"set": name, "target_auroc": target, "rho": rho, "delta": round(delta, 4), "mean_layer0_auroc": round(float(np.mean(aucs)), 4),
            "mean_S": round(float(np.mean(S_hat)), 4), **{f"p_{k}": round(v, 3) for k, v in p.items()},
            "p_clear": round(p["beyond"] + p["noninferior"], 3)}


def main():
    import modal

    raw = b"".join(modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_file(SCORES))
    assert hashlib.sha256(raw).hexdigest() == SCORES_SHA, "scores file changed"
    runs = defaultdict(list)
    info = {}
    for line in gzip.decompress(raw).decode().splitlines():
        if line.strip():
            d = json.loads(line)
            runs[(d["arm"], d["key"])].append(d["score"])
            info[(d["arm"], d["key"])] = (d["environment"], d["main_task"])
    keys = sorted(runs)
    sets = {"test_envs": [k for k in keys if info[k][0] in TEST_ENVS], "all_envs": keys}
    jobs, meta = [], {}
    for si, (name, ks) in enumerate(sets.items()):
        y = np.array([1 if k[0] == "sim-attack-fixed" else 0 for k in ks])
        sA = np.array([np.mean(runs[k]) for k in ks])
        tasks = sorted({info[k][1] for k in ks})
        task_idx = np.array([tasks.index(info[k][1]) for k in ks])
        resid = sA - np.where(y == 1, sA[y == 1].mean(), sA[y == 0].mean())
        r = resid / resid.std()
        ni = np.bincount(task_idx).astype(float)
        tm = np.bincount(task_idx, weights=r) / ni
        k, N = len(ni), len(r)  # one-way ANOVA variance components with unequal cluster sizes
        msb = (ni * (tm - r.mean()) ** 2).sum() / (k - 1)
        msw = ((r - tm[task_idx]) ** 2).sum() / (N - k)
        n0 = (N - (ni**2).sum() / N) / (k - 1)
        sb2 = max((msb - msw) / n0, 0.0)
        tau = float(min(sb2 / (sb2 + msw), 0.95))
        tau_naive = float(np.var(tm[task_idx]) / np.var(r))
        meta[name] = {"runs": len(ks), "sim": int(y.sum()), "real_e1": int((y == 0).sum()), "main_tasks": len(tasks),
                      "auroc_A": round(auc(sA, y), 4), "tau_task_share": round(tau, 4), "tau_naive": round(tau_naive, 4)}
        for ti, t in enumerate(TARGETS):
            for ri, rho in enumerate(RHOS):
                jobs.append((name, t, rho, tau, sA, y, r, task_idx, len(tasks), SEED + 1000 * si + 10 * ti + ri))
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one_config, jobs))
    out = {"label": "power for the layer-0 control's rule (simulated layer-0 side; step A's side is its stored scores)",
           "rule": {"M": M, "one_sided_bound": BOUND, "beyond": "lower bound of S > M (and E's plant-2 control passes)",
                    "noninferior": "upper bound of S < M"},
           "reps": REPS, "boot": BOOT, "seed": SEED, "targets": TARGETS, "rhos": RHOS, "sets": meta, "rows": rows,
           "inputs": {SCORES: SCORES_SHA},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_layer0_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps(meta, indent=1))
    for row in rows:
        print(row)
    print("->", dst)


if __name__ == "__main__":
    main()
