"""Realism rows for F10 and F11 (realism RULE.md Amendment 3; POST HOC). Modal H100, one container per layer.

Real honest rows come from d9's fix-full pack (as for every level so far). The simulated honest arms are read by key from
q2's 7-layer extractions on probe-shift_aswin-surya_sub-la-activations (last-token output_end, kept rows):
- "full + F9 (q2)": fxavg's sim-honest-fx-fullF9, all 437 runs. The loader gate: its summary (ii) reproduces the as-run
  full + F9 value within 0.005. Nothing else is computed if it doesn't.
- "full + F9 + F10": fxavg's sim-honest-fx-fullF9F10 (36 changed runs), the other runs from "full + F9 (q2)".
- "full + F9 + F11": f11's sim-honest-fx-fullF9F11 (226 changed runs), the other runs from "full + F9 (q2)".
Per row: summary (ii) with the as-run 2,000-draw main-task bootstrap (seed 20261013), Amendment 1's label-shuffle null
(seed [20261018, row index, layer]), and for F10 and F11 the paired difference from "full + F9 (q2)" on the same draws.
Also: how many of the 437 runs' rows equal d9's full + F9 rows bit for bit. Each row's JSON is checkpointed.

  NULL_IN=<folder with d9's run table> PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy \\
      modal run --detach experiments/2026-10-13_realism_by_repair_POSTHOC/run_fx_realism.py --tag <checkpoint folder>
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-realism-null_fx_kill-2026-11-13"
WORK, STAMP = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "2026-10-13_152618"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
CKPT = "probe-shift_aswin-surya_sub-realism-null_ckpt_kill-2026-11-13"
FXAVG, F11 = "2026-10-13_q2_fxavg_qwen3-32b_l7_v1", "2026-10-13_q2_f11_qwen3-32b_l7_v1"
BASE = (FXAVG, "sim-honest-fx-fullF9", 437)
ROWS = {"full + F9 (q2)": None, "full + F9 + F10": (FXAVG, "sim-honest-fx-fullF9F10", 36), "full + F9 + F11": (F11, "sim-honest-fx-fullF9F11", 226)}
AS_RUN_F9 = {48: 0.80801, 63: 0.855199}
C, MAX_ITER, N_NULL, B, SEED = 0.01, 1000, 100, 2000, 20261013
H, SF9 = "real-honest", "sim-honest-f9"
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
IN = Path(os.environ.get("NULL_IN", "/nonexistent"))
app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2).read_only()
acts = modal.Volume.from_name(ACTS, version=2).read_only()
ckpt = modal.Volume.from_name(CKPT, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3").add_local_python_source("probe_shift")
if IN.exists():
    image = image.add_local_file(IN / "sce_fixfull_scores_2026-10-13_152618.npz", "/root/in/d9.npz")


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=1800, retries=0, volumes={"/work": work, "/acts": acts, "/ckpt": ckpt})
def run(layer: int, tag: str) -> list:
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import auc, pool, shuffle_within_task

    R = np.load("/root/in/d9.npz")
    arm, fold, task, key = (R[f"runs__{k}"] for k in ("arm", "fold", "main_task", "key"))
    fold, n0 = fold.astype(int), len(arm)
    d = Path("/work/sce_fixfull") / STAMP
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    if np.any(np.diff(r_run) < 0):
        raise RuntimeError("d9's rows are not grouped by run")
    Xd = np.load(d / f"L{layer}.npy", mmap_mode="r")
    f9 = np.flatnonzero(arm == SF9)  # every new row's runs: these keys, folds and tasks
    if len(f9) != 437 or len(set(key[f9])) != 437:
        raise RuntimeError("d9's full + F9 runs are not 437 distinct keys")

    def metas(run, marm, expect):
        out = {}
        for p in (Path("/acts") / run / "meta" / marm).glob("*.json"):
            m = json.loads(p.read_text())
            if m["key"] in out:
                raise RuntimeError(f"{marm}: a run extracted twice")
            out[m["key"]] = (run, marm, m)  # the arm is part of a reading's identity: F9 and F10 share an extraction run and keys
        if len(out) != expect or not set(out) <= set(key[f9]):
            raise RuntimeError(f"{marm}: {len(out)} runs ({expect} expected) or keys outside the full + F9 runs")
        return out

    base = metas(*BASE)
    if set(base) != set(key[f9]):
        raise RuntimeError("full + F9 (q2): its keys differ from d9's full + F9 runs")
    sources = {name: base | ({} if spec is None else metas(*spec)) for name, spec in ROWS.items()}
    groups, feats = {}, {}
    for (r, a, k), m in {(r, a, k): m for s in sources.values() for k, (r, a, m) in s.items()}.items():
        rows = [m["row_start"] + p["row"] for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
        groups.setdefault((r, m["chunk"]), []).append(((a, k), rows))

    def load(item):
        (run, ch), its = item
        block = np.load(Path("/acts") / run / "by_layer" / f"L{layer}" / f"{ch}.npy", mmap_mode="r")
        for (a, k), rows in its:
            feats[(run, a, k)] = np.asarray(block[rows], np.float32)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(load, groups.items()))
    if not all(np.isfinite(v).all() for v in feats.values()):
        raise RuntimeError("non-finite features")
    lo, hi = np.searchsorted(r_run, f9), np.searchsorted(r_run, f9, "right")
    same_count, exact, maxdiff = 0, 0, 0.0
    for i, a, b in zip(f9, lo, hi):
        q, dd = feats[(FXAVG, BASE[1], key[i])], np.asarray(Xd[a:b], np.float32)
        if q.shape == dd.shape:
            same_count += 1
            exact += bool(np.array_equal(q, dd))
            maxdiff = max(maxdiff, float(np.abs(q - dd).max()) if len(q) else 0.0)
    check = {"runs_with_same_row_count_as_d9": same_count, "runs_bit_exact_to_d9": exact, "max_abs_diff_where_counts_match": maxdiff}

    # runs: d9's table, then the 437 new runs (index n0 + j is f9[j]'s key)
    n = n0 + len(f9)
    fold_x, task_x = np.r_[fold, fold[f9]], np.r_[task, task[f9]]
    sim_tasks = sorted(set(task[f9]))
    y = np.r_[np.where((arm == H) & np.isin(task, sim_tasks), 0.0, np.nan), np.ones(len(f9))]
    use = ~np.isnan(y)
    real_rows = np.flatnonzero(use[r_run])
    Xr, rr_real = torch.from_numpy(np.asarray(Xd[real_rows], np.float32)).cuda(), r_run[real_rows]
    tasks, tix = np.unique(task, return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wt = np.vstack([np.ones(len(tasks))] + [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)])[:, np.r_[tix, tix[f9]]]
    out_dir = Path("/ckpt") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results, boots = [], {}
    for ri, (name, spec) in enumerate(ROWS.items()):
        tu = time.time()
        new = [sources[name][key[i]] for i in f9]
        Xn = [feats[(r, a, m["key"])] for r, a, m in new]
        X = torch.cat([Xr, torch.from_numpy(np.concatenate(Xn)).cuda()])
        r_all = np.r_[rr_real, np.repeat(n0 + np.arange(len(f9)), [len(x) for x in Xn])]
        rng_n = np.random.default_rng([20261018, ri, layer])
        Y = np.stack([y] + [shuffle_within_task(y, task_x, use, rng_n) for _ in range(N_NULL)], 1)
        acc, cnt = np.zeros((n, N_NULL + 1)), np.zeros(n)
        for f, v in PAIRS:
            te = (fold_x == f) | (fold_x == v)
            tr, ter = np.flatnonzero((use & ~te)[r_all]), np.flatnonzero((use & te)[r_all])
            Xtr = X[torch.from_numpy(tr).cuda()]
            mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            W, b, _ = tp.fit((Xtr - mean) / std, torch.from_numpy(Y[r_all[tr]]).cuda().float(), C, MAX_ITER)
            del Xtr
            S = pool(r_all[ter], tp.decision((X[torch.from_numpy(ter).cuda()] - mean) / std, W, b).double().cpu().numpy(), n, "mean")
            m = use & te & np.isfinite(S[:, 0])
            acc[m] += S[m]
            cnt[m] += 1
        del X
        m = use & (cnt > 0)
        P = acc / np.maximum(cnt, 1)[:, None]
        a = np.array([float(auc(P[m, c], Y[m, c], np.ones((1, m.sum())))[0]) for c in range(N_NULL + 1)])
        boot = auc(P[m, 0], y[m], Wt[:, m])
        boots[name] = boot
        null = a[1:]
        row = {"row": name, "layer": layer, "summary_ii": float(a[0]), "ci95": [float(np.nanpercentile(boot[1:], 2.5)), float(np.nanpercentile(boot[1:], 97.5))],
               "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)), "null_max": float(null.max()),
               "p": float((1 + (null >= a[0]).sum()) / (1 + N_NULL)), "runs": [int((y[m] == 1).sum()), int((y[m] == 0).sum())],
               "runs_swapped_in": 0 if spec is None else spec[2], "new_runs_without_rows": int(sum(len(x) == 0 for x in Xn)), "seconds": round(time.time() - tu)}
        if spec is None:
            row |= {"as_run": AS_RUN_F9[layer], "gate_passes": bool(abs(a[0] - AS_RUN_F9[layer]) <= 0.005), "loader_check": check}
            if not row["gate_passes"]:  # Amendment 3: a failed gate reports the observed value only
                row |= {k: None for k in ("ci95", "null_mean", "null_p95", "null_max", "p")}
        else:
            diff = boot - boots["full + F9 (q2)"]
            row |= {"minus_full_F9_q2": float(diff[0]), "minus_full_F9_q2_ci95": [float(np.nanpercentile(diff[1:], 2.5)), float(np.nanpercentile(diff[1:], 97.5))]}
        (out_dir / f"L{layer}__fx_{ri}.json").write_text(json.dumps(row, indent=1))
        ckpt.commit()
        results.append(row)
        if spec is None and not row["gate_passes"]:
            break
    return results


@app.local_entrypoint()
def main(tag: str = "", layers: str = "48,63"):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("refusing to run from a dirty tree")
    if not tag:
        raise SystemExit("--tag is required (the checkpoint folder)")
    if not hashlib.sha256((IN / "sce_fixfull_scores_2026-10-13_152618.npz").read_bytes()).hexdigest().startswith("48fc5eb52546"):
        raise SystemExit("not d9's fix-full run table")
    for res in run.starmap([(int(L), tag) for L in layers.split(",")]):
        for r in res:
            print(json.dumps(r))
