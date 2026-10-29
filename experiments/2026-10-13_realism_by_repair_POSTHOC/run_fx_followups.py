"""Where F11's realism move lives (realism RULE.md Amendment 4; POST HOC, descriptive). Modal H100, one container per layer.

Real honest rows are read from d9's extraction 2026-10-12_la_mean-pool36_qwen3-32b_l7_v1 (it records each row's tool);
simulated honest rows from q2's fxavg (full + F9) and f11 (full + F9 + F11, 226 runs swapped in by key), as in
run_fx_realism.py. Readings are keyed by (extraction, arm, key). Variants:
- A (the loader gate): last-token output_end, all rows, full + F9; summary (ii) within 0.005 of the as-run value. Also the
  number of real honest runs whose rows equal d9's pack bit for bit. If A fails, nothing else is computed.
- B: last-token output_end with every text_editor output row dropped (real and simulated alike), full + F9 and + F11.
- C: the averaged readout (output_end_mean), all rows, full + F9 and + F11.
Per row: summary (ii) with the as-run bootstrap draws (seed 20261013), the label-shuffle null (seed [20261020, row index,
layer]); per variant: + F11 minus full + F9 on the same draws. Each row's JSON is checkpointed.

  NULL_IN=<folder with d9's run table> PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy \\
      modal run --detach experiments/2026-10-13_realism_by_repair_POSTHOC/run_fx_followups.py --tag <checkpoint folder>
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
APP = "probe-shift_aswin-surya_sub-realism-null_fxfollow_kill-2026-11-13"
WORK, STAMP = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "2026-10-13_152618"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
CKPT = "probe-shift_aswin-surya_sub-realism-null_ckpt_kill-2026-11-13"
REAL = ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "real-honest")
F9 = ("2026-10-13_q2_fxavg_qwen3-32b_l7_v1", "sim-honest-fx-fullF9", 437)
F11 = ("2026-10-13_q2_f11_qwen3-32b_l7_v1", "sim-honest-fx-fullF9F11", 226)
VARIANTS = [("A", "output_end", False, ("full + F9",)), ("B", "output_end", True, ("full + F9", "+ F11")), ("C", "output_end_mean", False, ("full + F9", "+ F11"))]
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


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, timeout=1800, retries=0, volumes={"/work": work, "/acts": acts, "/ckpt": ckpt})
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
    Xd = np.load(d / f"L{layer}.npy", mmap_mode="r")
    f9 = np.flatnonzero(arm == SF9)
    sim_tasks = sorted(set(task[f9]))
    real = np.flatnonzero((arm == H) & np.isin(task, sim_tasks))  # the 885 real honest runs every row uses

    def metas(run, marm, keys, expect=None):
        out = {}
        for p in (Path("/acts") / run / "meta" / marm).glob("*.json"):
            m = json.loads(p.read_text())
            if m["key"] in out:
                raise RuntimeError(f"{marm}: a run extracted twice")
            out[m["key"]] = (run, marm, m)
        if expect is None:  # the real arm must hold every key a row needs
            bad = not set(keys) <= set(out)
        else:  # a simulated arm must hold exactly its expected runs, all among the full + F9 keys
            bad = len(out) != expect or not set(out) <= set(keys)
        if bad:
            raise RuntimeError(f"{marm}: {len(out)} runs; expected {expect}, or missing some of the {len(keys)} keys needed")
        return out

    mr = metas(*REAL, key[real])
    m9 = metas(*F9[:2], key[f9], F9[2])
    if set(m9) != set(key[f9]):
        raise RuntimeError("full + F9: keys differ from d9's full + F9 runs")
    m11 = m9 | metas(*F11[:2], key[f9], F11[2])
    src = {"full + F9": m9, "+ F11": m11}

    def rows_of(m, kind, drop):
        return [m["row_start"] + p["row"] for p in m["positions"] if p["kind"] == kind and p["kept"] and not (drop and p["tool"] == "text_editor")]

    def load(entries, kind, drop):
        """entries: [(run, arm, meta)] in run order -> list of float32 arrays (one per run)."""
        out, groups = [None] * len(entries), {}
        for i, (r, a, m) in enumerate(entries):
            groups.setdefault((r, m["chunk"]), []).append((i, rows_of(m, kind, drop)))

        def one(item):
            (r, ch), its = item
            block = np.load(Path("/acts") / r / "by_layer" / f"L{layer}" / f"{ch}.npy", mmap_mode="r")
            for i, rows in its:
                out[i] = np.asarray(block[rows], np.float32)

        with ThreadPoolExecutor(16) as ex:
            list(ex.map(one, groups.items()))
        if not all(np.isfinite(x).all() for x in out):
            raise RuntimeError("non-finite features")
        return out

    n = n0 + len(f9)
    fold_x, task_x = np.r_[fold, fold[f9]], np.r_[task, task[f9]]
    y = np.r_[np.where(np.isin(np.arange(n0), real), 0.0, np.nan), np.ones(len(f9))]
    use = ~np.isnan(y)
    tasks, tix = np.unique(task, return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wt = np.vstack([np.ones(len(tasks))] + [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)])[:, np.r_[tix, tix[f9]]]
    out_dir = Path("/ckpt") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results, ri = [], 0
    for vname, kind, drop, arms in VARIANTS:
        Xr_list = load([mr[k] for k in key[real]], kind, drop)
        Xr = torch.from_numpy(np.concatenate(Xr_list)).cuda()
        rr_real = np.repeat(real, [len(x) for x in Xr_list])
        check = None
        if vname == "A":  # real honest rows against d9's pack, run by run
            lo, hi = np.searchsorted(r_run, real), np.searchsorted(r_run, real, "right")
            same = [(x.shape == (b - a, x.shape[1])) and np.array_equal(x, np.asarray(Xd[a:b], np.float32)) for x, a, b in zip(Xr_list, lo, hi)]
            check = {"real_honest_runs_bit_exact_to_d9": int(sum(same)), "of": len(real)}
        boots = {}
        for name in arms:
            tu = time.time()
            Xn = load([src[name][k] for k in key[f9]], kind, drop)
            X = torch.cat([Xr, torch.from_numpy(np.concatenate(Xn)).cuda()])
            r_all = np.r_[rr_real, np.repeat(n0 + np.arange(len(f9)), [len(x) for x in Xn])]
            rng_n = np.random.default_rng([20261020, ri, layer])
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
                if not np.isfinite(S[np.isin(np.arange(n), r_all[ter])]).all():
                    raise RuntimeError("non-finite scores")
                m = use & te & np.isfinite(S[:, 0])
                acc[m] += S[m]
                cnt[m] += 1
            del X
            m = use & (cnt > 0)
            if np.any(cnt[m] != 8):
                raise RuntimeError("a scored run lacks one of its 8 nested predictions")
            P = acc / np.maximum(cnt, 1)[:, None]
            a = np.array([float(auc(P[m, c], Y[m, c], np.ones((1, m.sum())))[0]) for c in range(N_NULL + 1)])
            boot = boots[name] = auc(P[m, 0], y[m], Wt[:, m])
            null = a[1:]
            row = {"variant": vname, "readout": kind, "text_editor_rows_dropped": drop, "row": name, "layer": layer, "summary_ii": float(a[0]),
                   "runs": [int((y[m] == 1).sum()), int((y[m] == 0).sum())],
                   "runs_without_rows": {"simulated": int(sum(len(x) == 0 for x in Xn)), "real": int(sum(len(x) == 0 for x in Xr_list))}, "seconds": round(time.time() - tu)}
            gate_fail = vname == "A" and abs(a[0] - AS_RUN_F9[layer]) > 0.005
            if vname == "A":
                row |= {"as_run": AS_RUN_F9[layer], "gate_passes": not gate_fail, "loader_check": check}
            if not gate_fail:
                row |= {"ci95": [float(np.nanpercentile(boot[1:], 2.5)), float(np.nanpercentile(boot[1:], 97.5))], "null_mean": float(null.mean()),
                        "null_p95": float(np.percentile(null, 95)), "null_max": float(null.max()), "p": float((1 + (null >= a[0]).sum()) / (1 + N_NULL))}
            if name == "+ F11":
                diff = boot - boots["full + F9"]
                row |= {"minus_full_F9": float(diff[0]), "minus_full_F9_ci95": [float(np.nanpercentile(diff[1:], 2.5)), float(np.nanpercentile(diff[1:], 97.5))]}
            (out_dir / f"L{layer}__{vname}_{ri}.json").write_text(json.dumps(row, indent=1))
            ckpt.commit()
            results.append(row)
            ri += 1
            if gate_fail:
                return results
        del Xr
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
