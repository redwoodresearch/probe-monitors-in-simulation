"""Retrain the Q3 detector on attack runs that attempted the side task vs attack runs that declined it, with no honest
runs, so the attack instruction is the same on both sides (Modal CPU). Does anything in the activations track the
attempt itself?

Features: the base run's packed level-0 features on the #36 work Volume (data 307cbfee925c): content-token calls at
layer 63 and all calls at layer 48, the readouts and layers #36 chose. Recipe as run_q3_attack.probe_layer: per call,
standardized on the training rows, L2 logistic regression (C = 0.01, L-BFGS, torch_probe.fit), run score = mean over
the run's calls. Labels: real attack e1 runs labelled attempted (1) or declined (0) in the shared attempt labels. Five
main-task folds: train on four folds' labelled e1 runs, score every run of the fifth (e1, e2, sim; out of fold). 20
label-shuffle columns are fitted alongside (labels permuted within main task among the training runs, seed 0) as a null
that repeats the training. Writes results/attempt_detector_<stamp>.npz and .json; analyze_attempt_detector.py reads them.

  uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run run_attempt_detector.py --labels <attempt labels>
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q3followups_attempt_kill-2026-11-06"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
DATA_ID = "307cbfee925c"
READOUTS = {"call_end_content": (63, [0]), "call_end": (48, None)}
CPU, MEMORY, SHUFFLES = 16, 65536, 20

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3")
         .add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/work": work_vol}, cpu=CPU, memory=MEMORY, timeout=3600)
def fit_all(y_run: list, fold: list, task: list) -> dict:
    import time

    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    torch.set_num_threads(CPU)
    t0 = time.time()
    d = Path("/work/data") / DATA_ID
    t = np.load(d / "tables.npz")
    y_run, fold, task = np.array(y_run, float), np.array(fold), np.array(task)
    n = len(y_run)
    rng = np.random.default_rng(0)
    trainable = ~np.isnan(y_run)
    # shuffled run labels: permuted within main task among labelled runs, one column per shuffle
    Y = np.repeat(y_run[:, None], 1 + SHUFFLES, axis=1)
    for k in range(1, 1 + SHUFFLES):
        for tk in np.unique(task[trainable]):
            ix = np.flatnonzero(trainable & (task == tk))
            Y[ix, k] = y_run[ix][rng.permutation(len(ix))]
    out = {}
    for ro, (layer, kinds) in READOUTS.items():
        X = np.load(d / "packed" / f"L{layer}_call_end.npy", mmap_mode="r")
        sel = np.arange(len(t["call_end__run"])) if kinds is None else np.flatnonzero(np.isin(t["call_end__kind"], kinds))
        r_run = t["call_end__run"][sel]
        scores = np.full((n, 1 + SHUFFLES), np.nan)
        fits = []
        for f in range(5):
            tr = np.flatnonzero(trainable[r_run] & (fold[r_run] != f))
            te = np.flatnonzero(fold[r_run] == f)
            Xtr = torch.from_numpy(np.asarray(X[sel[tr]], dtype=np.float32))
            mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            W, b, n_eval = tp.fit((Xtr - mean) / std, torch.from_numpy(Y[r_run[tr]]).float(), 0.01, 1000)
            Xte = (torch.from_numpy(np.asarray(X[sel[te]], dtype=np.float32)) - mean) / std
            s = tp.decision(Xte, W, b).double().numpy()  # [rows, 1 + SHUFFLES]
            runs_te = r_run[te]
            sums = np.zeros((n, s.shape[1])); cnt = np.bincount(runs_te, minlength=n).astype(float)
            np.add.at(sums, runs_te, s)
            m = cnt > 0
            scores[m] = sums[m] / cnt[m, None]
            fits.append({"fold": f, "train_rows": int(len(tr)), "scored_rows": int(len(te)), "n_eval": int(n_eval)})
            print(f"{ro} L{layer} fold {f}: {len(tr)} train rows, {len(te)} scored, {n_eval} evals, {time.time() - t0:.0f} s", flush=True)
        out[ro] = {"layer": layer, "scores": scores.tolist(), "fits": fits}
    out["seconds"] = round(time.time() - t0)
    return out


@app.local_entrypoint()
def main(labels: str):
    import hashlib

    import numpy as np

    runs = json.loads((REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json").read_text())
    lab = {(x["arm"], x["key"]): x["label"] for x in map(json.loads, Path(labels).read_text().splitlines())}
    y = [1.0 if r["arm"] == "real-attack-e1" and lab.get((r["arm"], r["key"])) == "attempted"
         else 0.0 if r["arm"] == "real-attack-e1" and lab.get((r["arm"], r["key"])) == "declined" else float("nan") for r in runs]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"{int(np.nansum(y))} attempted / {int(np.sum(np.array(y) == 0))} declined e1 runs; commit {head[:8]}{' (dirty)' if dirty else ''}", flush=True)
    out = fit_all.remote(y, [r["fold"] for r in runs], [r["main_task"] for r in runs])
    res = HERE / "results"
    np.savez(res / f"attempt_detector_{stamp}.npz", **{f"{ro}__scores": np.array(v["scores"]) for ro, v in out.items() if ro != "seconds"})
    rec = {"stamp": stamp, "git": head, "git_dirty": dirty, "app": APP, "cpu": CPU, "memory_mb": MEMORY, "data_id": DATA_ID,
           "work_volume": WORK, "readouts": {ro: {"layer": v["layer"], "fits": v["fits"]} for ro, v in out.items() if ro != "seconds"},
           "shuffles": SHUFFLES, "C": 0.01, "labels": {"file": labels, "sha256": hashlib.sha256(Path(labels).read_bytes()).hexdigest()},
           "runs_file": "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json", "seconds": out["seconds"]}
    with (res / f"attempt_detector_{stamp}.json").open("x") as f:
        f.write(json.dumps(rec, indent=1))
    print(f"-> results/attempt_detector_{stamp}.npz/.json ({out['seconds']} s)")
