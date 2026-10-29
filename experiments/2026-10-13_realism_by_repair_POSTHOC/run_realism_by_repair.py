"""Honest realism AUROC by repair level, and the real-vs-real control (RULE.md; POST HOC). One Modal CPU container per layer.

On d9's fix-full packed features (Qwen3-32B, output_end last token; run score = mean over outputs), for each nested model
(f, v) (training runs outside folds f and v): the C = 0.01 logistic (torch_probe.fit, CPU), real honest (0) vs simulated
honest at a level (1); held-out runs are those of folds f and v. Summary (i): the mean over models of the held-out run
AUROC (#3's definition). Summary (ii): each held-out run's score averaged over its 8 models, one AUROC, a 95% main-task
bootstrap (2,000 draws, seed 20261013). The control: real honest A vs B (the first two runs by key of tasks with >= 2).
The gate: fix-full's summary (i), with every real honest run as #3 had it, within 0.005 of #3's k = 0 realism.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy modal run --detach \\
      experiments/2026-10-13_realism_by_repair_POSTHOC/run_realism_by_repair.py --runs <fix-full scores npz> --out <results json>
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
APP = "probe-shift_aswin-surya_sub-realism-repair_cpu_kill-2026-11-13"
WORK, STAMP = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "2026-10-13_152618"
C, MAX_ITER, B, SEED = 0.01, 1000, 2000, 20261013
H = "real-honest"
LEVELS = {"as-is": "sim-honest", "fix939 (SH′)": "sim-honest-edited", "fix-full (SH″)": "sim-honest-fullfix", "newline-only": "sim-honest-newline",
          "full + F9": "sim-honest-f9"}
GATE = {48: 0.7841, 63: 0.8419}  # #3's k = 0 realism (aswin/sce-world-direction-removal, worlddir_scores_*.json)
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
         .uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift"))


@app.function(image=image, cpu=32, memory=65536, timeout=3600, retries=0, volumes={"/work": work})
def run(layer: int, arm: list, fold: list, task: list, key: list) -> dict:
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import auc, pool

    torch.set_num_threads(32)
    t0 = time.time()
    arm, fold, task, key = np.array(arm), np.array(fold), np.array(task), np.array(key)
    n = len(arm)
    d = Path("/work/sce_fixfull") / STAMP
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    if r_run.max() >= n or np.any(np.diff(r_run) < 0):
        raise RuntimeError("rows don't match the run table")
    use = np.isin(arm, [H, *LEVELS.values()])
    keep_rows = np.flatnonzero(use[r_run])
    X = torch.from_numpy(np.load(d / f"L{layer}.npy", mmap_mode="r")[keep_rows].astype(np.float32))
    rr = r_run[keep_rows]
    tasks, tix = np.unique(task, return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wt = np.vstack([np.ones(len(tasks))] + [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)])[:, tix]
    # the control: A = first, B = second real honest run by key, on tasks with >= 2
    ctrl = np.full(n, np.nan)
    for t in np.unique(task[arm == H]):
        idx = np.flatnonzero((arm == H) & (task == t))
        if len(idx) >= 2:
            o = idx[np.argsort(key[idx], kind="mergesort")]
            ctrl[o[0]], ctrl[o[1]] = 0.0, 1.0
    evals = []

    def nested(y):
        """y: [n] labels with NaN = not used. Returns (summary i, held-out run AUROCs per model, pooled out-of-fold scores [n])."""
        per, acc, cnt = [], np.zeros(n), np.zeros(n)
        for f, v in PAIRS:
            te = (fold == f) | (fold == v)
            tr_runs, te_runs = ~np.isnan(y) & ~te, ~np.isnan(y) & te
            tr, ter = np.flatnonzero(tr_runs[rr]), np.flatnonzero(te_runs[rr])
            Xtr = X[tr]
            mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            W, b, ne = tp.fit((Xtr - mean) / std, torch.from_numpy(y[rr[tr]]).float(), C, MAX_ITER)
            evals.append(int(ne))
            s = tp.decision((X[ter] - mean) / std, W, b).double().numpy()[:, 0]
            S = pool(rr[ter], s, n, "mean")
            m = te_runs & np.isfinite(S)
            per.append(float(auc(S[m], y[m], np.ones((1, m.sum())))[0]))
            acc[m] += S[m]
            cnt[m] += 1
        with np.errstate(invalid="ignore"):
            return float(np.mean(per)), per, acc / cnt

    def summary_ii(S, y):
        m = np.isfinite(S) & ~np.isnan(y)
        a = auc(S[m], y[m], Wt[:, m])
        return {"value": float(a[0]), "ci95": [float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))], "runs": [int((y[m] == 1).sum()), int((y[m] == 0).sum())]}

    sim_tasks = set(task[np.isin(arm, list(LEVELS.values()))])
    out = {"layer": layer, "levels": {}, "gate": {}}
    for name, a in LEVELS.items():
        y = np.where(arm == a, 1.0, np.where((arm == H) & np.isin(task, sorted(sim_tasks)), 0.0, np.nan))  # rule: real honest on tasks with a sim honest run
        i, per, S = nested(y)
        out["levels"][name] = {"summary_i": i, "per_model": per, "summary_ii": summary_ii(S, y)}
    yg = np.where(arm == LEVELS["fix-full (SH″)"], 1.0, np.where(arm == H, 0.0, np.nan))  # #3's population, for the gate
    gi, _, _ = nested(yg)
    out["gate"] = {"fixfull_summary_i_all_real_honest": gi, "worlddir_k0": GATE[layer], "passes": abs(gi - GATE[layer]) <= 0.005}
    ci, cper, cS = nested(ctrl)
    out["control"] = {"summary_i": ci, "per_model": cper, "summary_ii": summary_ii(cS, ctrl), "tasks": int(len(set(task[~np.isnan(ctrl)])))}
    out |= {"max_evals": max(evals), "fits_at_cap": sum(e >= MAX_ITER for e in evals), "n_fits": len(evals), "seconds": round(time.time() - t0)}
    return out


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
    res = list(run.starmap([(L, *cols) for L in (48, 63)]))
    rec = {"label": "POST HOC (RULE.md)", "git": git("rev-parse", "HEAD"), "app": APP, "runs_file_sha256": sha, "layers": res}
    Path(out).open("x").write(json.dumps(rec, indent=1))
    for r in res:
        print(f"L{r['layer']} gate {r['gate']} | fits {r['n_fits']} max_evals {r['max_evals']} at_cap {r['fits_at_cap']} | {r['seconds']} s")
        for name, x in r["levels"].items():
            s = x["summary_ii"]
            print(f"  {name:16s}: i {x['summary_i']:.3f} | ii {s['value']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] runs {s['runs']}")
        c = r["control"]["summary_ii"]
        print(f"  {'control A vs B':16s}: i {r['control']['summary_i']:.3f} | ii {c['value']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] runs {c['runs']} tasks {r['control']['tasks']}")
