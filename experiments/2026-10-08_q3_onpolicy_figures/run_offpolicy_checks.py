"""Why do on-policy scores sit below off-policy ones? Two checks, both on the off-policy refit's own pipeline and rows.
Modal, 6 T4 containers (5 size shards and the transfer), about $2 (0d relaying Aswin, 2026-10-09; prediction committed first: predictions_offpolicy_checks_2026-10-09.md).

Figure 3's real -> real (Qwen3.5 does the 20 tasks itself) is AUROC 0.975, standardized pAUC 0.801; figure 2's (Qwen3.5
reads Sol's runs, 522 tasks) is 0.995 and 0.969. The pipeline is probe_shift.refit_bootstrap / probes (16 layers, C grid,
earliest layer within 0.005, unpenalized intercept, float32 L-BFGS); the rows are the refit's own (Volume
`...q2-refit_boot...`, `2026-10-08_impactful_refit_bootstrap_v1/qwen3.5/{rows.json, L<l>.npy}`, read-only).

- size: 50 draws of 20 main tasks from those in both arms (seed [20261009, d]), all their real and simulated rows, the
  20 put 4 to a fold at random, one `replicate` (every task once). Real -> real AUROC and pAUC pooled over the 5 test
  folds, each scored by its own fold's probe (as figure 2 and figure 3).
- transfer: one real-trained probe on every off-policy real row except the 20 pilot main tasks (layer and C picked with
  `select`, folds_v3 fold 0 held out, inner training on folds 1-4; then refit on all of them), applied unchanged at its
  layer to (a) the 20 pilot tasks' off-policy rows (held out) and (b) figure 3's on-policy calls (`run_score_onpolicy.load`,
  September config: real and simulated, all scored calls and the matched subset). Intervals for (a) and (b): 2,000
  task resamples of the fixed scores.
- transfer at a fixed layer (`--fixed-layers 51,39`): check 2 with the layer fixed (C still picked on fold 0), one
  probe per layer; only the transfer runs (deep-layer rerun, 0d's go 2026-10-09; prediction
  predictions_transfer_deep_2026-10-09.md).

  PYTHONPATH=src uv run --no-project --with 'modal>=1.2' --with pyyaml --with numpy modal run --detach experiments/2026-10-08_q3_onpolicy_figures/run_offpolicy_checks.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in a Modal container the script sits at /root
APP = "probe-shift_aswin-surya_sub-q3onpol_offchecks_kill-2026-11-08"
REFIT = ("probe-shift_aswin-surya_sub-q2-refit_boot_kill-2026-11-08", "2026-10-08_impactful_refit_bootstrap_v1/qwen3.5")
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
CONFIG = "config/score_rerun_strict_september.yaml"
PILOT = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9.json"
LAYERS, GRID, TOL, MAX_ITER, FPR = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63], [0.001, 0.01, 0.1], 0.005, 500, 0.01
ARMS = {"real-attack-e1": 0, "sim-attack": 1}  # world 0 real, 1 simulated (refit.yaml)
N_DRAWS, N_TASKS, SEED, N_BOOT = 50, 20, 20261009, 2000

app = modal.App(APP)
refit_vol = modal.Volume.from_name(REFIT[0], version=2)
acts_vol = modal.Volume.from_name(ACTS, version=2)
save_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", version=2)  # each output, in case the client drops
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "pyyaml")
         .add_local_python_source("probe_shift")
         .add_local_file(HERE / "run_score_onpolicy.py", "/root/run_score_onpolicy.py")
         .add_local_file(HERE / CONFIG, f"/root/{CONFIG}")
         .add_local_file(REPO / PILOT, f"/root/{PILOT}"))


def pooled(y, s, w=None):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s, sample_weight=w)), float(roc_auc_score(y, s, sample_weight=w, max_fpr=0.05))


def boot(y, s, task, seed=0):
    """Observed AUROC and pAUC, and 95% intervals over task resamples of the fixed scores."""
    import numpy as np

    tasks, ti = np.unique(task, return_inverse=True)
    rng, vals = np.random.default_rng(seed), []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))[ti]
        if (w * y).sum() and (w * (1 - y)).sum():
            vals.append(pooled(y, s, w))
    v = np.array(vals)
    a, p = pooled(y, s)
    return {"auroc": round(a, 4), "auroc_ci95": [round(float(np.percentile(v[:, 0], q)), 4) for q in (2.5, 97.5)],
            "pauc": round(p, 4), "pauc_ci95": [round(float(np.percentile(v[:, 1], q)), 4) for q in (2.5, 97.5)],
            "calls": int(len(y)), "impactful": int(y.sum()), "read_only": int((1 - y).sum()), "tasks": int(len(tasks)),
            "failed_resamples": N_BOOT - len(v)}


def load_refit(layers=LAYERS):
    import numpy as np
    import torch

    d = Path("/refit") / REFIT[1]
    meta = json.loads((d / "rows.json").read_text())
    X = torch.from_numpy(np.stack([np.load(d / f"L{layer}.npy") for layer in layers])).cuda()  # [layers, rows, 5120] fp16
    return meta, X, np.array([m["y"] for m in meta]), np.array([ARMS[m["arm"]] for m in meta]), np.array([m["main_task"] for m in meta])


GPU = dict(image=image, gpu="T4", cpu=4, memory=32768, timeout=2 * 3600, retries=0,
           volumes={"/refit": refit_vol.read_only(), "/acts": acts_vol.read_only(), "/save": save_vol})


def save(stamp: str, name: str, obj) -> None:
    d = Path("/save") / f"offchecks_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj))
    save_vol.commit()


@app.function(**GPU)
def size_shard(draws: list, stamp: str) -> dict:
    """Check 1 for the given draws: 20 random main tasks, refit pipeline, held out within the 20."""
    import numpy as np

    from probe_shift.refit_bootstrap import replicate

    meta, X, y, world, main = load_refit()
    both = sorted(set(main[world == 0]) & set(main[world == 1]))
    out = {"rows": len(meta), "task_pool": len(both), "draws": []}
    for k in draws:
        rng = np.random.default_rng([SEED, k])
        tasks = [both[i] for i in rng.choice(len(both), N_TASKS, replace=False)]
        fold_of = {t: i // (N_TASKS // 5) for i, t in enumerate(rng.permutation(tasks))}
        idx = np.flatnonzero(np.isin(main, tasks))
        tix = np.searchsorted(np.array(sorted(tasks)), main[idx])
        r = replicate(X[:, idx], y[idx], world[idx], np.array([fold_of[t] for t in main[idx]]), tix,
                      np.ones(len(tasks), int), LAYERS, GRID, TOL, MAX_ITER, FPR)
        rec = {"draw": k, "tasks": sorted(tasks), "real_calls": int((world[idx] == 0).sum()),
               "real_read_only": int(((world[idx] == 0) & (y[idx] == 0)).sum())}
        if "undefined" in r:
            out["draws"].append(rec | {"undefined": r["undefined"]})
            continue
        fold = np.array([fold_of[t] for t in main[idx]])
        rr = [np.flatnonzero((world[idx] == 0) & (fold == f)) for f in range(5)]
        s = np.concatenate([r["scores"][f, 0, rr[f]] for f in range(5)])
        yy = np.concatenate([y[idx][rr[f]] for f in range(5)])
        a, p = pooled(yy, s)
        if abs(a - r["real->real"]["auroc"]) > 1e-6:
            raise ValueError(f"draw {k}: pooled AUROC {a} differs from replicate's {r['real->real']['auroc']}")
        rec |= {"auroc": round(a, 4), "pauc": round(p, 4), "real->sim_auroc": round(r["real->sim"]["auroc"], 4),
                "layers": [q["layer"] for q in r["picks"] if q["train"] == "real"], "C": [q["C"] for q in r["picks"] if q["train"] == "real"]}
        out["draws"].append(rec)
        print(f"size draw {k}: AUROC {rec['auroc']} pAUC {rec['pauc']} layers {rec['layers']}", flush=True)
    save(stamp, f"size_{draws[0]}", out)
    return out


@app.function(**GPU)
def transfer(stamp: str, layer: int | None = None) -> dict:
    """Check 2: one probe on every off-policy real row except the pilot main tasks, applied unchanged. layer: fix the
    layer (C is still picked on fold 0); None picks it with the refit's rule over all 16."""
    import sys

    import numpy as np
    import torch
    import yaml

    sys.path.insert(0, "/root")
    import run_score_onpolicy as rso
    from probe_shift.probes import fit_logreg, select

    layers = LAYERS if layer is None else [layer]
    meta, X, y, world, main = load_refit(layers)
    out = {"rows": len(meta), "per_world": {w: int((world == i).sum()) for w, i in (("real", 0), ("sim", 1))}}
    pilot = json.loads((Path("/root") / PILOT).read_text())
    pilot_main = {c["main"] for c in pilot["cells"]}
    missing = sorted(pilot_main - set(main))
    if missing:
        raise ValueError(f"pilot main tasks not in the refit rows: {missing}")
    fold3 = np.array([m["fold"] for m in meta])
    keep = (world == 0) & ~np.isin(main, list(pilot_main))
    train, val = np.flatnonzero(keep & (fold3 != 0)), np.flatnonzero(keep & (fold3 == 0))
    pick = select(X, y, train, val, layers, GRID, TOL, MAX_ITER)
    tr = np.flatnonzero(keep)
    Xl = X[pick["li"]].float()
    mu, sd = Xl[tr].mean(0), Xl[tr].std(0).clamp_min(1e-6)
    w8, g = fit_logreg(((Xl - mu) / sd)[tr], torch.tensor(y[tr], dtype=torch.float32, device="cuda"), pick["C"], MAX_ITER)
    w, b = w8[:-1], float(w8[-1])
    score = lambda Z: (((Z - mu) / sd) @ w + b).cpu().numpy()
    s_off = score(Xl)
    del Xl
    tr_auc = pooled(y[tr], s_off[tr])
    out["transfer"] = {"probe": {"layer": pick["layer"], "C": pick["C"], "val_auroc": round(pick["val_auroc"], 4), "grad_max": g,
                                 "train_rows": int(len(tr)), "train_tasks": int(len(set(main[tr]))), "train_auroc_in_sample": round(tr_auc[0], 4),
                                 "layer_curve": pick["layer_curve"], "excluded_pilot_mains": sorted(pilot_main)}}
    for name, wi in (("real", 0), ("sim", 1)):
        i = np.flatnonzero((world == wi) & np.isin(main, list(pilot_main)))
        out["transfer"][f"(a) off-policy pilot tasks, {name}"] = boot(y[i], s_off[i], main[i])
    # (b) figure 3's on-policy calls at the same layer
    cfg = yaml.safe_load((Path("/root") / CONFIG).read_text())
    folds = rso.frozen_folds(Path("/root") / PILOT)
    rows = rso.load(cfg | {"n_layers": 0}, folds)["_rows"]
    Xo = np.empty((len(rows), 5120), np.float16)
    groups = {}
    for i, (_, _, run, chunk, row, *_) in enumerate(rows):
        groups.setdefault((run, chunk), ([], []))
        groups[(run, chunk)][0].append(i)
        groups[(run, chunk)][1].append(row)
    for (run, chunk), (pos, sel) in groups.items():
        Xo[pos] = np.load(Path("/acts") / run / "by_layer" / f"L{pick['layer']}" / f"{chunk}.npy", mmap_mode="r")[sel]
    if not np.isfinite(Xo.astype(np.float32)).all():
        raise ValueError("non-finite on-policy activations")
    s_on = score(torch.from_numpy(Xo).float().cuda())
    yo = np.array([r[5] for r in rows])
    wo = np.array([r[0] for r in rows])
    to = np.array([r[1] for r in rows])
    mo = np.array([r[7] for r in rows])
    for name in ("real", "sim"):
        for sub, m in (("all scored calls", np.ones(len(rows), bool)), ("matched subset", mo)):
            i = np.flatnonzero((wo == name) & m)
            out["transfer"][f"(b) on-policy {name} ({cfg['worlds'][name]}), {sub}"] = boot(yo[i], s_on[i], to[i])
    out["transfer"]["on_policy_rows_sha256"] = rso.rows_sha(rows)
    res = out["transfer"] | {"refit_rows": out["rows"], "refit_per_world": out["per_world"]}
    save(stamp, "transfer" if layer is None else f"transfer_L{layer}", res)
    return res


@app.local_entrypoint()
def main(fixed_layers: str = ""):
    import numpy as np

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print("outputs also on the score Volume under", f"offchecks_{stamp}/", flush=True)
    meta = {"git": git, "dirty": dirty, "script_sha256": __import__("hashlib").sha256((HERE / Path(__file__).name).read_bytes()).hexdigest(),
            "refit": REFIT, "config": CONFIG, "C_grid": GRID}
    if fixed_layers:  # the deep-layer transfer rerun only
        ls = [int(x) for x in fixed_layers.split(",")]
        r = {"transfer_by_layer": {f"L{layer}": res for layer, res in zip(ls, transfer.starmap([(stamp, layer) for layer in ls]))},
             "meta": meta | {"fixed_layers": ls}}
        path = HERE / "results" / f"offpolicy_transfer_fixed_layers_{stamp}.json"
        with path.open("x") as fh:
            fh.write(json.dumps(r, indent=1))
        for k, v in r["transfer_by_layer"].items():
            print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "probe"} | {"probe": {kk: vv for kk, vv in v["probe"].items() if kk != "layer_curve"}}, indent=1))
        print("->", path)
        return
    tr = transfer.spawn(stamp)
    shards = list(size_shard.map([list(range(i, N_DRAWS, 5)) for i in range(5)], kwargs={"stamp": stamp}))
    r = {"size": {"task_pool": shards[0]["task_pool"], "draws": sorted((d for s in shards for d in s["draws"]), key=lambda d: d["draw"])},
         "transfer": tr.get()}
    if [d["draw"] for d in r["size"]["draws"]] != list(range(N_DRAWS)):
        raise ValueError("size draws missing or duplicated")
    ok = [d for d in r["size"]["draws"] if "auroc" in d]
    a, p = np.array([d["auroc"] for d in ok]), np.array([d["pauc"] for d in ok])
    r["size"]["summary"] = {"defined_draws": len(ok), "undefined_draws": N_DRAWS - len(ok),
                            "auroc_median": round(float(np.median(a)), 4), "auroc_5_95": [round(float(np.percentile(a, q)), 4) for q in (5, 95)],
                            "pauc_median": round(float(np.median(p)), 4), "pauc_5_95": [round(float(np.percentile(p, q)), 4) for q in (5, 95)],
                            "share_auroc_le_0.975": round(float((a <= 0.975).mean()), 3), "share_pauc_le_0.80": round(float((p <= 0.80).mean()), 3),
                            "real_calls_median": int(np.median([d["real_calls"] for d in ok])),
                            "real_read_only_median": int(np.median([d["real_read_only"] for d in ok]))}
    r["meta"] = {"git": git, "dirty": dirty, "script_sha256": __import__("hashlib").sha256((HERE / Path(__file__).name).read_bytes()).hexdigest(),
                 "refit": REFIT, "config": CONFIG, "layers": LAYERS, "C_grid": GRID, "seed": SEED, "draws": N_DRAWS, "tasks_per_draw": N_TASKS}
    path = HERE / "results" / f"offpolicy_checks_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(r, indent=1))
    print(json.dumps(r["size"]["summary"], indent=1))
    print(json.dumps({k: v for k, v in r["transfer"].items() if k != "probe"} | {"probe": {k: v for k, v in r["transfer"]["probe"].items() if k != "layer_curve"}}, indent=1))
    print("->", path)
