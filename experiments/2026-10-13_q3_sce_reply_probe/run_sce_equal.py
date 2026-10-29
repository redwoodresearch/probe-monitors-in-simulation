"""SCE-built reply probe, equal-count check (RULE.md Amendment 2; added after launch, before any result was read).

The main job's cells and nested models, with every training set cut to equal numbers. In each nested model (f, v), both
classes of every cell keep m(f, v) whole runs, where m is the smallest class count over the five cells (runs with an
output, in the three training folds). Each class is sampled within main task proportionally (largest remainder,
random ties). There are 5 draws, seeded by sha256(cell|draw), and both layers get the same draws. Features are the main
job's packed matrices, checked by sha256. No null and no plant. Modal H100, one container per layer; no text leaves
the Volume, and failures are re-raised by type only (a Gate's own message is numbers only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \\
      experiments/2026-10-13_q3_sce_reply_probe/run_sce_equal.py
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
APP = "probe-shift_aswin-surya_sub-q3sce_equal_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
MAIN = "2026-10-13_121538"  # the main job's stamp (app ap-JXedkWSJhaFEow9OOu24pm): its packed features
FEATURES_SHA256 = {48: "084dcf1cf056ab6211bef45a13af99ada3ef7b51da577867f9b403e1cd741250",  # the main job's pack record
                   63: "42b3236e47fe59461f931cb927e66a741af94c878b679c35b9af8b767f9d3a0e"}
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
LAYERS, ROWS = (48, 63), 135963
CELLS = {"a": ("real-attack-e1", "real-honest"), "b": ("sim-attack", "sim-honest"), "c": ("sim-attack", "real-honest"),
         "b_fixed": ("sim-attack-fixed", "sim-honest-edited"), "c_fixed": ("sim-attack-fixed", "real-honest")}  # (1, 0)
C, MAX_ITER, DRAWS = 0.01, 1000, 5
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
ROOT = Path("/work")

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12")  # #36's image, unchanged
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.16.2")
         .add_local_python_source("probe_shift"))


class Gate(Exception):
    """A failed check; its message holds numbers only."""


def subsample(rng, idx, task, m):
    """m of the runs idx, within main task proportionally: quotas by largest remainder (random ties), runs at random."""
    tasks, inv = np.unique(task[idx], return_inverse=True)
    n_t = np.bincount(inv)
    exact = m * n_t / len(idx)
    q = np.floor(exact).astype(int)
    order = np.lexsort((rng.random(len(tasks)), -(exact - q)))  # largest remainder first, ties at random
    q[order[:m - q.sum()]] += 1
    return np.sort(np.concatenate([rng.permutation(idx[inv == t])[:q[t]] for t in range(len(tasks))]))


def equal_sets(runs, has_rows):
    """{(cell, draw): [20 arrays of training run indices]}, the same at both layers. Each array holds m(f, v) runs of
    each class."""
    arm, task = np.array([r["arm"] for r in runs]), np.array([r["main_task"] for r in runs])
    fold = np.array([r["fold"] for r in runs])
    m = {}
    for f, v in PAIRS:
        tr = has_rows & (fold != f) & (fold != v)
        m[f, v] = min(int((tr & (arm == a)).sum()) for pos, neg in CELLS.values() for a in (pos, neg))
    out = {}
    for c, (pos, neg) in CELLS.items():
        for d in range(DRAWS):
            rng = np.random.default_rng(int(hashlib.sha256(f"{c}|{d}".encode()).hexdigest()[:8], 16))
            sets = []
            for f, v in PAIRS:
                tr = has_rows & (fold != f) & (fold != v)
                parts = [subsample(rng, np.flatnonzero(tr & (arm == a)), task, m[f, v]) for a in (pos, neg)]
                if any(len(p) != m[f, v] for p in parts):
                    raise Gate(f"{c} draw {d} ({f}, {v}): a class did not get {m[f, v]} runs")
                sets.append(np.concatenate(parts))
            out[c, d] = sets
    return out, m


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0,
              volumes={"/work": work_vol.read_only()})
def fit(layer: int, runs: list[dict]) -> dict:
    try:
        return _fit(layer, runs)
    except Gate as e:
        raise RuntimeError(f"fit gate failed: {e}") from None
    except Exception as e:  # noqa: BLE001 — re-raised by type only, never swallowed
        raise RuntimeError(f"fit failed: {type(e).__name__}") from None


def _fit(layer, runs):
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool

    t0, d = time.time(), ROOT / "sce" / MAIN
    raw = np.load(d / f"L{layer}.npy")
    if raw.shape[0] != ROWS or hashlib.sha256(raw.tobytes()).hexdigest() != FEATURES_SHA256[layer]:
        raise Gate(f"L{layer}: packed features differ from the main job's")
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    n = len(runs)
    if len(r_run) != ROWS or r_run.max() >= n:
        raise Gate(f"L{layer}: row table does not match {n} runs")
    X = torch.from_numpy(raw).cuda().float()
    del raw
    arm, fold = np.array([r["arm"] for r in runs]), np.array([r["fold"] for r in runs])
    sets, m = equal_sets(runs, np.bincount(r_run, minlength=n) > 0)
    out, evals = {}, []
    for c, (pos, _) in CELLS.items():
        nested = np.full((DRAWS, 5, 5, n), np.nan, np.float32)
        for dr in range(DRAWS):
            for (f, v), train in zip(PAIRS, sets[c, dr]):
                score_runs = (fold == f) | (fold == v)
                tr, te = np.flatnonzero(np.isin(r_run, train)), np.flatnonzero(score_runs[r_run])
                Xtr = X[torch.from_numpy(tr).cuda()]
                mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
                std = std.where(std > 0, std.new_ones(()))
                y = torch.from_numpy((arm[r_run[tr]] == pos).astype(np.float32)).cuda()
                W, b, ne = tp.fit((Xtr - mean) / std, y, C, MAX_ITER)
                del Xtr
                s = tp.decision((X[torch.from_numpy(te).cuda()] - mean) / std, W, b).double().cpu().numpy()
                if not np.isfinite(s).all():
                    raise Gate(f"L{layer}: non-finite output scores")
                S = pool(r_run[te], s[:, 0], n, "mean")
                nested[dr, f, v] = np.where(score_runs, S, np.nan)
                evals.append(int(ne))
        out[c] = nested
    info = {"layer": layer, "m_by_pair": {f"{f}_{v}": k for (f, v), k in m.items()}, "fits": len(evals),
            "max_evals": max(evals), "fits_at_cap": sum(e >= MAX_ITER for e in evals), "seconds": round(time.time() - t0),
            "train_sets_sha256": hashlib.sha256(b"".join(np.asarray(s, np.int64).tobytes() for k in sorted(sets) for s in sets[k])).hexdigest()}
    return {"info": info, "nested": out}


def build_runs() -> list[dict]:
    """run_sce_probe.build_runs, as at f6d50a21: #36's 3,807 runs, the 437 sim honest runs, then the fixed and edited
    copies."""
    keep = ("arm", "key", "cell", "main_task", "fold")
    base = [{k: r[k] for k in keep} | {"in_base": True} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} | {"in_base": False}
          for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == "sim-honest"]
    copies = [r | {"arm": new, "in_base": False} for new, orig in (("sim-attack-fixed", "sim-attack"), ("sim-honest-edited", "sim-honest"))
              for r in base + sh if r["arm"] == orig]
    return base + sh + copies



@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head, t0 = git("rev-parse", "HEAD"), time.time()
    runs = build_runs()
    if len(runs) != 5620 or len({(r["arm"], r["key"]) for r in runs}) != len(runs):
        raise SystemExit("unexpected runs")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"stamp {stamp}, git {head}, runs {dict(Counter(r['arm'] for r in runs))}", flush=True)
    fits = list(fit.starmap([(l, runs) for l in LAYERS]))
    (HERE / "results").mkdir(exist_ok=True)
    npz = HERE / "results" / f"sce_equal_scores_{stamp}.npz"  # git-ignored (*.npz): archived to S3 with a manifest
    with open(npz, "xb") as fh:
        np.savez(fh, **{f"L{x['info']['layer']}__{c}__nested_equal": v for x in fits for c, v in x["nested"].items()},
                 **{f"runs__{k}": np.array([r[k] for r in runs]) for k in ("arm", "key", "cell", "main_task", "fold", "in_base")})
    rec = {"stamp": stamp, "git": head, "app": APP, "rule": "RULE.md Amendment 2", "main_job": MAIN,
           "settings": {"layers": LAYERS, "cells": CELLS, "C": C, "max_iter": MAX_ITER, "draws": DRAWS},
           "fit": [x["info"] for x in fits], "scores_file": npz.name,
           "scores_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(), "client_seconds": round(time.time() - t0)}
    with (HERE / "results" / f"sce_equal_run_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("stamp", "scores_file", "scores_sha256", "client_seconds")}), flush=True)
