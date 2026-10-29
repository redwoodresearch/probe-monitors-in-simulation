"""The label-shuffle null for honest realism by repair level (RULE.md Amendment 1; POST HOC). Modal CPU, one container per
(layer, level) and per layer for the A-vs-B control: 12 in all.

Per job: run_realism_by_repair.py's labels and nested models, fit at once on 101 label columns (column 0 real; 1-100 permuted
within each main task among the job's runs). Summary (ii) per column: each held-out run's score averaged over its 8 models,
one AUROC. Gate: column 0 equals the as-run summary (ii) within 0.005, else the job raises and nothing is reported.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy modal run --detach \\
      experiments/2026-10-13_realism_by_repair_POSTHOC/run_realism_null.py --runs <fix-full scores npz> --out <results json>
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
APP = "probe-shift_aswin-surya_sub-realism-null_cpu_kill-2026-11-13"
WORK, STAMP = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "2026-10-13_152618"
C, MAX_ITER, N_NULL = 0.01, 1000, 100
H = "real-honest"
LEVELS = {"as-is": "sim-honest", "fix939 (SH′)": "sim-honest-edited", "fix-full (SH″)": "sim-honest-fullfix", "newline-only": "sim-honest-newline",
          "full + F9": "sim-honest-f9"}
AS_RUN = {48: {"as-is": 0.941757, "fix939 (SH′)": 0.814501, "fix-full (SH″)": 0.807814, "newline-only": 0.936638, "full + F9": 0.80801, "control": 0.492752},
          63: {"as-is": 0.971457, "fix939 (SH′)": 0.889374, "fix-full (SH″)": 0.858328, "newline-only": 0.950738, "full + F9": 0.855199, "control": 0.540604}}
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
         .uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift"))


def labels(job: str, arm, task, key):
    if job == "control":  # A = first, B = second real honest run by key, on tasks with >= 2
        y = np.full(len(arm), np.nan)
        for t in np.unique(task[arm == H]):
            idx = np.flatnonzero((arm == H) & (task == t))
            if len(idx) >= 2:
                o = idx[np.argsort(key[idx], kind="mergesort")]
                y[o[0]], y[o[1]] = 0.0, 1.0
        return y
    sim_tasks = sorted(set(task[np.isin(arm, list(LEVELS.values()))]))
    return np.where(arm == LEVELS[job], 1.0, np.where((arm == H) & np.isin(task, sim_tasks), 0.0, np.nan))


@app.function(image=image, cpu=32, memory=32768, timeout=1500, retries=0, volumes={"/work": work})
def run(layer: int, job: str, ji: int, arm: list, fold: list, task: list, key: list) -> dict:
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import auc, pool, shuffle_within_task

    torch.set_num_threads(32)
    t0 = time.time()
    arm, fold, task, key = np.array(arm), np.array(fold), np.array(task), np.array(key)
    n = len(arm)
    y = labels(job, arm, task, key)
    use = ~np.isnan(y)
    rng = np.random.default_rng([20261016, ji, layer])
    Y = np.stack([y] + [shuffle_within_task(y, task, use, rng) for _ in range(N_NULL)], 1)
    d = Path("/work/sce_fixfull") / STAMP
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    rows = np.flatnonzero(use[r_run])
    X = torch.from_numpy(np.load(d / f"L{layer}.npy", mmap_mode="r")[rows].astype(np.float32))
    rr = r_run[rows]
    acc, cnt, evals = np.zeros((n, N_NULL + 1)), np.zeros(n), []
    for f, v in PAIRS:
        te = (fold == f) | (fold == v)
        tr, ter = np.flatnonzero((use & ~te)[rr]), np.flatnonzero((use & te)[rr])
        Xtr = X[tr]
        mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        W, b, ne = tp.fit((Xtr - mean) / std, torch.from_numpy(Y[rr[tr]]).float(), C, MAX_ITER)
        evals.append(int(ne))
        S = pool(rr[ter], tp.decision((X[ter] - mean) / std, W, b).double().numpy(), n, "mean")
        m = use & te & np.isfinite(S[:, 0])
        acc[m] += S[m]
        cnt[m] += 1
    pooled = acc / np.maximum(cnt, 1)[:, None]
    m = use & (cnt > 0)
    a = np.array([float(auc(pooled[m, c], Y[m, c], np.ones((1, m.sum())))[0]) for c in range(N_NULL + 1)])
    obs, null = a[0], a[1:]
    if abs(obs - AS_RUN[layer][job]) > 0.005:
        raise RuntimeError(f"gate failed: L{layer} {job}: {obs:.6f} vs as-run {AS_RUN[layer][job]}")
    return {"layer": layer, "job": job, "observed": obs, "as_run": AS_RUN[layer][job], "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "null_max": float(null.max()), "p": float((1 + (null >= obs).sum()) / (1 + N_NULL)), "runs": [int((Y[m, 0] == 1).sum()), int((Y[m, 0] == 0).sum())],
            "max_evals": max(evals), "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(runs: str, out: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    sha = hashlib.sha256(Path(runs).read_bytes()).hexdigest()
    if not sha.startswith("48fc5eb52546"):
        raise SystemExit("not d9's fix-full scores file")
    R = np.load(runs)
    cols = [R[f"runs__{k}"].tolist() for k in ("arm", "fold", "main_task", "key")]
    cols[1] = [int(x) for x in cols[1]]
    jobs = [(L, j, i) for L in (48, 63) for i, j in enumerate([*LEVELS, "control"])]
    res = list(run.starmap([(L, j, i, *cols) for L, j, i in jobs]))
    Path(out).open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 1)", "git": git("rev-parse", "HEAD"), "app": APP, "runs_file_sha256": sha, "jobs": res}, indent=1))
    for r in res:
        print(f"L{r['layer']} {r['job']:16s}: observed {r['observed']:.4f} (as run {r['as_run']:.4f}) | null mean {r['null_mean']:.4f}, 95th {r['null_p95']:.4f}, max {r['null_max']:.4f} | "
              f"p {r['p']:.4f} | runs {r['runs']} | {r['seconds']} s")
