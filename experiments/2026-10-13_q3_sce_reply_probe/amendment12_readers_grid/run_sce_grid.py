"""The 2x2 training grid's remaining cells for Olmo-3.1-32B or Llama-3.3-70B (RULE.md Amendment 12): (b) sim attack vs sim
honest, (b′) fixed sim attack vs edited sim honest, (d) real attack e1 vs sim honest. Cells (a), (c) and (c′) are
Amendments 7 and 8's. No new extraction and no re-pack: one H100 container per layer loads that layer's packed matrix,
written by Amendments 7/8's job and checked by its sha256, once, and fits every cell, every nested (f, v) model, the
100-shuffle null (batched as label columns) and the planted cue in cell (b) from that one load. nvidia-smi is sampled
every 0.5 s inside each container. The reader is chosen by SCE_READER (olmo or llama), which the image carries into the
containers. Modal; stored activations only; no model calls. No text leaves the Volumes, and failures are re-raised by type
only (a Gate's own message is numbers only).

The local entrypoint takes the runs table from Amendments 7/8's scores file (the packed rows index it), writes
results/sce_grid_<reader>_scores_<stamp>.npz (git-ignored, archived to S3) and results/sce_grid_<reader>_run_<stamp>.json.
Metrics: analyze_sce_grid.py.

  SCE_READER=olmo PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \\
      experiments/2026-10-13_q3_sce_reply_probe/run_sce_grid.py
"""

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
READERS = {  # Amendments 7/8's packed run (stamp), its scores file and run record, and each layer's packed features sha256
    "olmo": {"stamp": "2026-10-13_133322", "layers": (32, 40, 56), "H": 5120,
             "sha": {32: "3cef81650396", 40: "ff31cf552bd1", 56: "55ceda8a9671"}},
    "llama": {"stamp": "2026-10-13_133321", "layers": (60, 70), "H": 8192,
              "sha": {60: "a7358cc80ccd", 70: "79bebcdcc61f"}},
}
READER = os.environ["SCE_READER"]
CFG = READERS[READER]
APP = f"probe-shift_aswin-surya_sub-q3sce_grid_{READER}_kill-2026-11-13"
LAYERS, H = CFG["layers"], CFG["H"]
CELLS = {"b": ("sim-attack", "sim-honest"), "b_fixed": ("sim-attack-fixed", "sim-honest-edited"),
         "d": ("real-attack-e1", "sim-honest")}  # (1, 0)
PLANT_CELL = "b"  # the main rule's planted cell
C, MAX_ITER, N_NULL, ALPHAS, DELTA, SEED = 0.01, 1000, 100, (0.01, 0.05), 0.5, 20261013

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12")  # #36's image, unchanged, plus the reader's name
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.16.2")
         .env({"SCE_READER": READER})
         .add_local_python_source("probe_shift"))


class Gate(Exception):
    """A failed check; its message holds numbers only."""


def sanitized(fn):
    def run(*a):
        try:
            return fn(*a)
        except Gate as e:
            raise RuntimeError(f"{fn.__name__} gate failed: {e}") from None
        except Exception as e:  # noqa: BLE001 — re-raised by type only, never swallowed: a message could carry text
            raise RuntimeError(f"{fn.__name__} failed: {type(e).__name__}") from None
    return run


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(65536, 65536), timeout=1800, retries=0,
              volumes={"/work": work_vol.read_only()})
def fit(layer: int, runs: list[dict], features_sha256: str) -> dict:
    return sanitized(_fit)(layer, runs, features_sha256)


def start_smi(path):
    """nvidia-smi sampling to a file every 0.5 s. Telemetry only: every failure is recorded in the state, never raised."""
    st = {"proc": None, "fh": None, "errors": [], "stopped": False}
    try:
        st["fh"] = open(path, "w")
        st["proc"] = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                                       "-lms", "500"], stdout=st["fh"], stderr=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001 — recorded in the result, not swallowed
        st["errors"].append(f"start: {type(e).__name__}")
    return st


def stop_smi(st):
    """Idempotent; records an early exit of the sampler and any cleanup error."""
    if st["stopped"]:
        return
    st["stopped"] = True
    proc = st["proc"]
    try:
        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                st["errors"].append(f"exited early, rc {rc}")
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
    except Exception as e:  # noqa: BLE001 — recorded in the result, not swallowed
        st["errors"].append(f"stop: {type(e).__name__}")
    finally:
        try:
            if st["fh"] is not None:
                st["fh"].close()
        except Exception as e:  # noqa: BLE001
            st["errors"].append(f"close: {type(e).__name__}")


def _fit(layer, runs, features_sha256):
    t0 = time.time()
    smi_path = Path(f"/tmp/smi_L{layer}.csv")
    smi = start_smi(smi_path)
    try:
        return _fit_body(layer, runs, features_sha256, t0, smi_path, smi)
    finally:
        stop_smi(smi)


def _fit_body(layer, runs, features_sha256, t0, smi_path, smi):
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool, shuffle_within_task

    d = Path("/work") / f"sce_{READER}" / CFG["stamp"]
    Xh = np.load(d / f"L{layer}.npy")
    got = hashlib.sha256(Xh.tobytes()).hexdigest()
    if got != features_sha256 or not got.startswith(CFG["sha"][layer]) or Xh.shape[1] != H:
        raise Gate(f"L{layer}: packed features differ from Amendments 7/8's ({Xh.shape})")
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    X = torch.from_numpy(Xh).cuda().float()
    del Xh
    t_load = time.time() - t0
    n = len(runs)
    if int(r_run.max()) >= n:
        raise Gate("rows index runs beyond the table")
    arm, task = np.array([r["arm"] for r in runs]), np.array([r["main_task"] for r in runs])
    fold, honest = np.array([r["fold"] for r in runs]), np.array([r["arm"] == "real-honest" for r in runs])
    planted_rows = torch.from_numpy(arm[r_run] == "sim-attack").cuda()
    u = np.random.default_rng(SEED).standard_normal(H)
    u = torch.from_numpy(u / np.linalg.norm(u)).cuda().float()
    sigma_u = float((X[torch.from_numpy(np.flatnonzero(honest[r_run])).cuda()] @ u).std(unbiased=False))
    plant = DELTA * sigma_u * u
    pairs = [(f, v) for f in range(5) for v in range(5) if v != f]

    def fit_score(train_runs, score_runs, Y, planted):
        """#36's fit_score: fit on the outputs of train_runs, return run scores [n, k] (NaN outside score_runs)."""
        tr, te = np.flatnonzero(train_runs[r_run]), np.flatnonzero(score_runs[r_run])
        tr_t, te_t = torch.from_numpy(tr).cuda(), torch.from_numpy(te).cuda()
        Xtr = X[tr_t] + plant * planted_rows[tr_t, None] if planted else X[tr_t]
        mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        W, b, n_eval = tp.fit((Xtr - mean) / std, torch.from_numpy(Y[r_run[tr]]).cuda().float(), C, MAX_ITER)
        del Xtr
        Xte = X[te_t] + plant * planted_rows[te_t, None] if planted else X[te_t]
        s = tp.decision((Xte - mean) / std, W, b).double().cpu().numpy()
        if not np.isfinite(s).all():  # every variant and shuffle: a non-finite output score would void a threshold
            raise Gate(f"L{layer}: {int((~np.isfinite(s)).sum())} non-finite output scores")
        S = pool(r_run[te], s, n, "mean")
        S[~score_runs] = np.nan
        return S, int(n_eval)

    out, info = {}, {"layer": layer, "sigma_u": sigma_u, "load_seconds": round(t_load), "rows": int(len(r_run)), "cells": {}}
    for cell, (pos, neg) in CELLS.items():
        tc = time.time()
        y = np.where(arm == pos, 1.0, np.where(arm == neg, 0.0, np.nan))
        trainable = ~np.isnan(y)
        rng = np.random.default_rng(int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16))
        y_null = np.stack([shuffle_within_task(y, task, trainable, rng) for _ in range(N_NULL)], 1)
        variants = {"real": (y[:, None], False), "null": (y_null, False)} | ({"plant": (y[:, None], True)} if cell == PLANT_CELL else {})
        a_ = {f"nested_{k}": np.full((5, 5, n), np.nan, np.float32) for k in variants if k != "null"}
        a_ |= {f"null_caught_{a:g}": np.zeros((n, N_NULL), np.uint8) for a in ALPHAS}  # nested models (of 4) above threshold
        evals = {k: [] for k in variants}
        for f, v in pairs:
            te = (fold == f) | (fold == v)
            for k, (Y, planted) in variants.items():
                S, ne = fit_score(trainable & ~te, te, Y, planted)
                evals[k].append(ne)
                if k != "null":
                    a_[f"nested_{k}"][f, v] = S[:, 0]
                    continue
                hv = honest & (fold == v) & np.isfinite(S[:, 0])
                Sh = np.sort(S[hv], 0)[::-1]  # attack_eval.threshold per shuffle: s[floor(alpha n)] of the descending scores
                for a in ALPHAS:
                    a_[f"null_caught_{a:g}"][fold == f] += (S[fold == f] > Sh[int(np.floor(a * hv.sum()))]).astype(np.uint8)
        a_["y_null"] = np.nan_to_num(y_null, nan=-1).astype(np.int8)
        info["cells"][cell] = {"train_runs": {"1": int((y == 1).sum()), "0": int((y == 0).sum())},
                               "max_evals": {k: max(e) for k, e in evals.items()},
                               "fits_at_cap": {k: sum(x >= MAX_ITER for x in e) for k, e in evals.items()},
                               "label_columns_fitted": 20 * sum(Y.shape[1] for Y, _ in variants.values()),
                               "seconds": round(time.time() - tc)}
        out[cell] = a_
    stop_smi(smi)
    try:
        samples = [line.split(",") for line in smi_path.read_text().strip().splitlines() if line.count(",") == 1]
    except OSError as e:
        samples = []
        smi["errors"].append(f"read: {type(e).__name__}")
    util = np.array([float(s[0]) for s in samples if s[0].strip().isdigit()])
    mem = np.array([float(s[1]) for s in samples if s[1].strip().isdigit()])
    if not len(util):
        smi["errors"].append("no usable samples")
    info["gpu"] = {"telemetry_errors": smi["errors"], "samples": int(len(util)), "util_mean_pct": round(float(util.mean()), 1) if len(util) else None,
                   "util_median_pct": round(float(np.median(util)), 1) if len(util) else None,
                   "util_share_ge_50pct": round(float((util >= 50).mean()), 3) if len(util) else None,
                   "mem_max_mib": int(mem.max()) if len(mem) else None}
    info["seconds"] = round(time.time() - t0)
    return {"info": info, "arrays": out}


@app.local_entrypoint()
def main():
    import numpy as np

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head, t0 = git("rev-parse", "HEAD"), time.time()
    src = HERE / "results" / f"sce_{READER}_scores_{CFG['stamp']}.npz"  # Amendments 7/8's: its runs table indexes the packed rows
    rec7_path = HERE / "results" / f"sce_{READER}_run_{CFG['stamp']}.json"
    man = {f["file"]: f["sha256"] for f in json.loads((HERE / "results" / "manifest_sce_readers_results_v1.json").read_text())["files"]}
    if any(hashlib.sha256(q.read_bytes()).hexdigest() != man[q.name] for q in (src, rec7_path)):
        raise SystemExit("Amendments 7/8's scores file or run record differs from the committed manifest")
    rec7 = json.loads(rec7_path.read_text())
    if {p["layer"]: p["features_sha256"][:12] for p in rec7["pack"]} != CFG["sha"]:
        raise SystemExit("the pinned feature hashes differ from Amendments 7/8's run record")
    Z = np.load(src)
    runs = [{"arm": a, "key": k, "cell": c, "main_task": m, "fold": int(f)}
            for a, k, c, m, f in zip(*(Z[f"runs__{q}"].tolist() for q in ("arm", "key", "cell", "main_task", "fold")))]
    counts = Counter(r["arm"] for r in runs)
    want = {"real-honest": 990, "real-attack-e1": 939, "real-attack-e2": 939, "sim-attack": 939, "sim-honest": 437,
            "sim-attack-fixed": 939, "sim-honest-edited": 437}
    if dict(counts) != want:
        raise SystemExit(f"unexpected runs: {dict(counts)}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"reader {READER}, stamp {stamp}, git {head}, runs {dict(counts)}", flush=True)
    full = {p["layer"]: p["features_sha256"] for p in rec7["pack"]}
    fits = list(fit.starmap([(l, runs, full[l]) for l in LAYERS]))
    print(json.dumps([{k: v for k, v in x["info"].items() if k != "cells"} for x in fits]), flush=True)
    npz = HERE / "results" / f"sce_grid_{READER}_scores_{stamp}.npz"  # git-ignored (*.npz): archived to S3 with a manifest
    with open(npz, "xb") as fh:
        np.savez(fh, **{f"L{x['info']['layer']}__{c}__{k}": v for x in fits for c, arrs in x["arrays"].items() for k, v in arrs.items()},
                 **{f"runs__{q}": Z[f"runs__{q}"] for q in ("arm", "key", "cell", "main_task", "fold", "in_base", "repaired")})
    rec = {"stamp": stamp, "git": head, "app": APP, "reader": READER, "rule": "RULE.md Amendment 12", "runs": dict(counts),
           "settings": {"layers": LAYERS, "H": H, "cells": CELLS, "plant_cell": PLANT_CELL, "C": C, "max_iter": MAX_ITER, "n_null": N_NULL,
                        "alphas": ALPHAS, "delta": DELTA, "seed": SEED, "packed_from": f"{WORK}:sce_{READER}/{CFG['stamp']}",
                        "packed_features_sha256_prefix": CFG["sha"], "runs_table_from": src.name},
           "fit": [x["info"] for x in fits], "scores_file": npz.name,
           "scores_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(), "client_seconds": round(time.time() - t0)}
    with (HERE / "results" / f"sce_grid_{READER}_run_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("stamp", "scores_file", "scores_sha256", "client_seconds")}), flush=True)
