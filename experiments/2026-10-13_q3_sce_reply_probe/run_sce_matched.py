"""T3 (RULE.md Amendment 6): do Qwen3.5's SCE cells keep their gap when trained at the on-policy cohort's size? Cells
(a) e1 vs real honest and (c) sim attack vs real honest at L63, the Amendment 3 recipe, trained only on a subsample:
183 main tasks, one paired attack cell per task, 190 real honest runs (one per sampled task plus 7). There are 5 draws
per cell, seeded by sha256(cell|draw), and the planted cue is added in (c). Every run in folds f and v is scored, so the
analysis can evaluate on the subsample (primary) or on all runs (beside). Features: the Amendment 3 job's packed L63
matrix, checked by sha256. Modal H100, one container; no text leaves the Volume, and failures are re-raised by type
only (a Gate's own message is numbers only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \\
      experiments/2026-10-13_q3_sce_reply_probe/run_sce_matched.py
"""

import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q3sce_matched_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
SRC, LAYER, ROWS = "sce_q35/2026-10-13_130117", 63, 148224  # the Amendment 3 job's packed features (app ap-yX13GtXbTsuQGWL1nt7jrd)
FEATURES_SHA256 = "ed73041b69fde32f408a28602827f02615007d74f736b735efcd2b7c80a60b58"  # its pack record
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
H_, E1, E2, SA, SAF = "real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-attack-fixed"
CELLS = {"a": (E1, H_), "c": (SA, H_)}  # (1, 0)
N_TASKS, N_HONEST, DRAWS = 183, 190, 5  # the on-policy cohort (plan_attack_detector_retrain_2026-10-10.md @ 5858dfc6, line 892)
C, MAX_ITER, DELTA, SEED, H = 0.01, 1000, 0.5, 20261013, 5120
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
ROOT = Path("/work")

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12")  # #36's image, unchanged
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.16.2")
         .add_local_python_source("probe_shift"))


class Gate(Exception):
    """A failed check; its message holds numbers only."""


def subsample(cell_name, d, runs):
    """Boolean mask of the subsample's runs: 183 tasks; per task one paired attack cell (its e1, e2, sim attack and fixed
    sim attack runs) and one real honest run; then 7 more honest runs from the sampled tasks."""
    arm, task = np.array([r["arm"] for r in runs]), np.array([r["main_task"] for r in runs])
    cell = np.array([r["cell"] for r in runs])
    rng = np.random.default_rng(int(hashlib.sha256(f"{cell_name}|{d}".encode()).hexdigest()[:8], 16))
    tasks = np.array(sorted(set(task[arm == E1])))
    T = sorted(tasks[rng.choice(len(tasks), N_TASKS, replace=False)])
    chosen, honest, spare = [], [], []
    for t in T:
        cells = sorted(set(cell[(arm == E1) & (task == t)]))
        chosen.append(cells[rng.integers(len(cells))])
        idx = np.flatnonzero((arm == H_) & (task == t))
        first = idx[rng.integers(len(idx))]
        honest.append(first)
        spare += [i for i in idx if i != first]
    honest += list(rng.choice(np.array(spare), N_HONEST - N_TASKS, replace=False))
    mask = np.isin(arm, [E1, E2, SA, SAF]) & np.isin(cell, chosen)
    mask[np.array(honest)] = True
    if (mask & (arm == E1)).sum() != N_TASKS or (mask & (arm == H_)).sum() != N_HONEST:
        raise Gate(f"{cell_name} draw {d}: subsample counts off")
    return mask


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0,
              volumes={"/work": work_vol.read_only()})
def fit(runs: list[dict]) -> dict:
    try:
        return _fit(runs)
    except Gate as e:
        raise RuntimeError(f"fit gate failed: {e}") from None
    except Exception as e:  # noqa: BLE001 — re-raised by type only, never swallowed
        raise RuntimeError(f"fit failed: {type(e).__name__}") from None


def _fit(runs):
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool

    t0, d = time.time(), ROOT / SRC
    raw = np.load(d / f"L{LAYER}.npy")
    if raw.shape[0] != ROWS or hashlib.sha256(raw.tobytes()).hexdigest() != FEATURES_SHA256:
        raise Gate("packed features differ from the Amendment 3 job's")
    r_run = np.load(d / f"rows_L{LAYER}.npz")["run"].astype(np.int64)
    n = len(runs)
    if len(r_run) != ROWS or r_run.max() >= n:
        raise Gate(f"row table does not match {n} runs")
    X = torch.from_numpy(raw).cuda().float()
    del raw
    arm, fold = np.array([r["arm"] for r in runs]), np.array([r["fold"] for r in runs])
    honest = arm == H_
    planted_rows = torch.from_numpy(arm[r_run] == SA).cuda()
    u = np.random.default_rng(SEED).standard_normal(H)
    u = torch.from_numpy(u / np.linalg.norm(u)).cuda().float()
    sigma_u = float((X[torch.from_numpy(np.flatnonzero(honest[r_run])).cuda()] @ u).std(unbiased=False))  # Amendment 3's
    plant = DELTA * sigma_u * u
    out, evals, masks = {}, [], {}
    for c, (pos, neg) in CELLS.items():
        variants = ["real", "plant"] if c == "c" else ["real"]
        nested = {k: np.full((DRAWS, 5, 5, n), np.nan, np.float32) for k in variants}
        masks[c] = np.stack([subsample(c, dr, runs) for dr in range(DRAWS)])
        for dr in range(DRAWS):
            for f, v in PAIRS:
                score_runs = (fold == f) | (fold == v)
                train = masks[c][dr] & np.isin(arm, [pos, neg]) & ~score_runs
                tr, te = np.flatnonzero(train[r_run]), np.flatnonzero(score_runs[r_run])
                tr_t, te_t = torch.from_numpy(tr).cuda(), torch.from_numpy(te).cuda()
                y = torch.from_numpy((arm[r_run[tr]] == pos).astype(np.float32)).cuda()
                for k in variants:
                    Xtr = X[tr_t] + plant * planted_rows[tr_t, None] if k == "plant" else X[tr_t]
                    mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
                    std = std.where(std > 0, std.new_ones(()))
                    W, b, ne = tp.fit((Xtr - mean) / std, y, C, MAX_ITER)
                    del Xtr
                    Xte = X[te_t] + plant * planted_rows[te_t, None] if k == "plant" else X[te_t]
                    s = tp.decision((Xte - mean) / std, W, b).double().cpu().numpy()
                    if not np.isfinite(s).all():
                        raise Gate("non-finite output scores")
                    nested[k][dr, f, v] = np.where(score_runs, pool(r_run[te], s[:, 0], n, "mean"), np.nan)
                    evals.append(int(ne))
        out[c] = nested
    info = {"layer": LAYER, "sigma_u": sigma_u, "fits": len(evals), "max_evals": max(evals),
            "fits_at_cap": sum(e >= MAX_ITER for e in evals), "seconds": round(time.time() - t0),
            "subsample_runs": {c: [dict(Counter(arm[m].tolist())) for m in ms] for c, ms in masks.items()}}
    return {"info": info, "nested": out, "masks": {c: m.astype(np.uint8) for c, m in masks.items()}}


def build_runs() -> list[dict]:
    """run_sce_qwen35.build_runs, as at fdae8534."""
    keep = ("arm", "key", "cell", "main_task", "fold")
    base = [{k: r[k] for k in keep} | {"in_base": True} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} | {"in_base": False}
          for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == "sim-honest"]
    return base + sh + [r | {"arm": SAF, "in_base": False} for r in base if r["arm"] == SA]


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head, t0 = git("rev-parse", "HEAD"), time.time()
    runs = build_runs()
    if len(runs) != 5183 or len({(r["arm"], r["key"]) for r in runs}) != len(runs):
        raise SystemExit("unexpected runs")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"stamp {stamp}, git {head}", flush=True)
    res = fit.remote(runs)
    (HERE / "results").mkdir(exist_ok=True)
    npz = HERE / "results" / f"sce_matched_scores_{stamp}.npz"  # git-ignored (*.npz): archived to S3 with a manifest
    with open(npz, "xb") as fh:
        np.savez(fh, **{f"L{LAYER}__{c}__nested_{k}": v for c, d in res["nested"].items() for k, v in d.items()},
                 **{f"mask__{c}": m for c, m in res["masks"].items()},
                 **{f"runs__{k}": np.array([r[k] for r in runs]) for k in ("arm", "key", "cell", "main_task", "fold", "in_base")})
    rec = {"stamp": stamp, "git": head, "app": APP, "rule": "RULE.md Amendment 6", "features": SRC,
           "settings": {"layer": LAYER, "cells": CELLS, "tasks": N_TASKS, "honest": N_HONEST, "draws": DRAWS, "C": C,
                        "max_iter": MAX_ITER, "delta": DELTA, "seed": SEED},
           "fit": res["info"], "scores_file": npz.name, "scores_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
           "client_seconds": round(time.time() - t0)}
    with (HERE / "results" / f"sce_matched_run_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("stamp", "scores_file", "scores_sha256", "client_seconds")}), flush=True)
