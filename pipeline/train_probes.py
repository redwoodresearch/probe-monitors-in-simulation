# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.3.3", "safetensors==0.8.0", "torch==2.8.0"]
# ///
"""Train the probes we trained, from Qwen3.5-27B's activations, with the arithmetic that produced the post's probe weights.

Run it with:  uv run pipeline/train_probes.py on_policy --rows <folder> --out <folder>
              uv run pipeline/train_probes.py off_policy --rows <folder> --out <folder> [--select-layers]
              uv run pipeline/train_probes.py simulated_attack --rows <folder> --out <folder>

The activations ("training rows") are not in this repository (about 4.4 GB for the on-policy probe, about 4 GB for each of the other
two at their chosen layers, and about 34 GB more to rerun the off-policy probe's layer selection); see INPUT for the format.

WHAT A PROBE IS. A logistic regression on the activations of one layer of Qwen3.5-27B: the output of decoder layer L (counting from 0),
before the final norm, 5,120 numbers. It scores units. A unit is a tool output (its activations averaged over the output's content
tokens) or a command (the activation at the last content token of the call; commands whose last content token is "0" or "n" are left
out). Every unit of an attack run is a positive and every unit of an honest run a negative. Transcripts are used whole.

FOLDS. Each probe comes in five copies. The main tasks are split into five folds, and copy k trains on the runs whose main task is not
in fold k, so it can score the tasks of fold k out of sample. training_data/*_folds.csv gives each main task's fold; the folds of the
two kinds of probe agree on every main task they share.

WHAT EACH PROBE IS TRAINED ON (training_data/ lists every training run with its label and fold):
  On-policy probe (on_policy_probe): Qwen3.5-27B's own runs on real machines. Positives: its 838 attack runs (real run 1); negatives:
    its 837 honest runs; 839 tasks, each with its own main task. Layer 48. training_data/on_policy_probe_runs.csv.
  On-policy probe on fewer tasks (on_policy_probe_25pct_tasks, on_policy_probe_10pct_tasks): the same, but each copy trains only on a
    seeded random 25% (or 10%) of its training tasks (subsample_tasks below; the draw is the same for tool outputs and
    commands here, and is listed in training_data/on_policy_probe_task_subsamples.csv).
  Off-policy probe (off_policy_probe): runs by a different agent, GPT-5.6-Sol, on real machines, read by Qwen3.5-27B (we fed Sol's
    transcripts to Qwen3.5-27B and recorded its activations). Positives: Sol's 939 attack runs; negatives: its 990 honest runs, over
    536 main tasks. Tool outputs: each copy uses two layers, chosen by cross-validation inside its own training folds (step 3),
    and a unit's score is the average of the two layers' logits; commands: layer 63. training_data/off_policy_probe_runs.csv and
    off_policy_probe_layers.csv; the validation AUROC of every layer is in off_policy_probe_layer_selection.csv.
  Simulated-attack probe (extra_probe_simulated_attack_vs_real_honest, Appendix D): the off-policy probe's recipe with Sol's 939
    simulated attack runs as positives and the same 990 real honest runs as negatives; each copy uses the off-policy probe's layers.
    training_data/simulated_attack_probe_runs.csv.

ARITHMETIC (as in the original runs; the details matter only to reproduce the stored weights to the last bit).
  1. Standardize each feature by the mean and the population standard deviation (zero -> 1) of the copy's training units. On-policy:
     moments and standardization in float64. Off-policy and simulated-attack: moments and standardization in float32 (from the stored
     float16 activations), then the standardized units are cast to float64.
  2. Fit (fit_logistic): minimize G(w, b) = (sum_i logloss(x_i . w + b, y_i) + |w|^2 / (2C)) / n with C = 0.01, the objective of
     scikit-learn's LogisticRegression(C=0.01) divided by C n (same minimizer; the intercept is not penalized). PyTorch L-BFGS in
     float64 from zero: history 20, strong Wolfe line search, gradient tolerance 1e-8, no tolerance on the change, at most 5,000
     iterations, and a budget of 6,250 evaluations of G (checked after each line search, so a fit can end at 6,251 or 6,252). Fits on
     all training tasks usually end at that budget; every stored fit ends with max |grad G| <= 1e-6. (For the off-policy and
     simulated-attack probes the original first ran a quick float32 fit and refit as here only if that fit's max |grad G| exceeded
     1e-6; every one of their stored fits was refit, so this code always fits as here.)
  3. Layer selection (off-policy probe, tool outputs only; validation_auroc and top_two): for copy k and each layer, fit the four nested
     copies (k, v) that train on folds other than k and v, standardized in float32 as in step 1 and fit with the quicker float32 fit the
     selection used (fit_quick); score fold v's runs (a run's score is the mean of its units' logits) and take the AUROC of attack
     against honest runs; average over v. Copy k uses the two layers with the highest average (ties: the lower layer).
  The original fits ran on NVIDIA GPUs with torch 2.8.0: on an H100, and on an L4 for the off-policy and simulated-attack probes'
  commands. On the same GPU types this code reproduced the stored weights bit for bit (we refit 18 of the stored copies: 12 on-policy
  and 6 off-policy; the simulated-attack probe was not refit). The objective is convex and every fit converged, so other hardware
  may give slightly different weights (rounding, and where L-BFGS stops). On a CPU each fit takes much longer.

INPUT. <rows>/<probe>__<tool_outputs|commands>__L<layer>.npz, with <probe> one of on_policy, off_policy, simulated_attack:
  X    [units, 5120]  the activations: float32 for the on-policy probe, float16 for the other two (as stored)
  run  [units]        the unit's run: its row number in training_data/<probe>_probe_runs.csv (the column "run")
  A run's units are consecutive and in transcript order, and runs come in the order of the runs table. This is the order the stored
  weights were fit in; another order changes them only slightly.

OUTPUT, in the format of the stored weights: <out>/<probe name>/<tool_outputs|commands>__fold<k>.safetensors for the on-policy probes
(arrays W [5120, 1], b [1], mean [5120], std [5120], all float64), and <out>/<probe name>/<tool_outputs|commands>__L<layer>.npz for
the other two, holding all five copies at that layer (keys <p>__f<k>__W [5120] and <p>__f<k>__b [1] in float64, <p>__f<k>__mean and
<p>__f<k>__std [5120] in float32, with <p> = "a" for the off-policy probe and "c_asis" for the simulated-attack probe, as stored).
A unit's score (logit) is ((x - mean) / std) . W + b. Plus <out>/<probe>_fit_log.json: units, runs, evaluations of G and max |grad G|
per copy.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TABLES = Path(__file__).resolve().parent / "training_data"
C = 0.01  # inverse regularization strength
READOUTS = ("tool_outputs", "commands")
ON_POLICY = {"on_policy_probe": 1.0, "on_policy_probe_25pct_tasks": 0.25, "on_policy_probe_10pct_tasks": 0.10}  # name -> share of tasks
OTHERS = {"off_policy": ("off_policy_probe", "a"), "simulated_attack": ("extra_probe_simulated_attack_vs_real_honest", "c_asis")}


# ---------------------------------------------------------------- inputs


def read_table(name):
    """A CSV table from training_data/, as a list of dicts (integers converted)."""
    with open(TABLES / name, newline="") as fh:
        return [{k: int(v) if v.lstrip("-").isdigit() else v for k, v in row.items()} for row in csv.DictReader(fh)]


def load_training_rows(rows_dir, probe, readout, layer, n_runs):
    """One probe's units at one layer: X and run, checked against the format in INPUT."""
    z = np.load(Path(rows_dir) / f"{probe}__{readout}__L{layer}.npz")
    X, run = z["X"], z["run"]
    if X.dtype != (np.float32 if probe == "on_policy" else np.float16) or not np.issubdtype(run.dtype, np.integer):
        raise ValueError(f"{probe} {readout} L{layer}: X must be {'float32' if probe == 'on_policy' else 'float16'} and run integers")
    run = run.astype(np.int64)
    if X.shape != (len(run), 5120) or not np.isfinite(X).all() or run.min() < 0 or run.max() >= n_runs:
        raise ValueError(f"{probe} {readout} L{layer}: rows do not match the input format")
    if np.any(np.diff(run) < 0):
        raise ValueError(f"{probe} {readout} L{layer}: units must be grouped by run, in the runs table's order")
    return X, run


def training_units(run, runs, fold, tasks=None):
    """Positions of the units copy `fold` trains on: units of runs outside fold `fold` (and of the chosen tasks), in stored order."""
    keep = np.array([r["fold"] != fold and (tasks is None or r["task"] in tasks) for r in runs])
    return np.flatnonzero(keep[run])


def subsample_tasks(tasks, fold, fraction):
    """The seeded share of a copy's training tasks: sorted, shuffled with the seed string the original run used, first ceil(f n)."""
    # Original: run_own_all.py @ 63096838c, lines 203-210 (_subsample, run_famexp._train_tasks's rule).
    t = sorted(tasks)
    random.Random(f"famexp|own_a|f{fold}").shuffle(t)
    return set(t[: math.ceil(fraction * len(t))])


# ---------------------------------------------------------------- fitting


def fit_logistic(Xs, y):
    """Float64 L-BFGS on G (ARITHMETIC, step 2) from zero. Xs [n, d] standardized float64, y [n] in {0, 1}. Returns W [d, 1], b [1],
    and the number of evaluations of G."""
    # Original: torch_probe.py @ 341ee1585, 60-120 (run_own_all.py @ 63096838c, line 194); run_avg_transfer.py @ 199f50846, 1254-1274.
    n, d = Xs.shape
    Y = y.reshape(n, 1).to(Xs.dtype)
    W = torch.zeros(d, 1, device=Xs.device, dtype=Xs.dtype, requires_grad=True)
    b = torch.zeros(1, device=Xs.device, dtype=Xs.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=5000, max_eval=6250, tolerance_grad=1e-8, tolerance_change=0.0, history_size=20,
                            line_search_fn="strong_wolfe")
    evals = [0]

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(Xs @ W + b, Y, reduction="sum") + (W * W).sum() / (2 * C)) / n
        loss.backward()
        evals[0] += 1
        return loss

    opt.step(closure)
    return W.detach(), b.detach(), evals[0]


def max_gradient(Xs, y, W, b):
    """max |grad G| at (W, b) in float64: the convergence check (the stored fits all have <= 1e-6)."""
    # Original: src/probe_shift/torch_probe.py @ 341ee1585, lines 30-57 (certify).
    Xs, W, b = Xs.double(), W.double().reshape(-1, 1), b.double().reshape(-1)
    r = torch.sigmoid(Xs @ W + b) - y.double().reshape(-1, 1)
    return float(torch.cat([(Xs.T @ r + W / C) / len(Xs), r.sum(0, keepdim=True) / len(Xs)]).abs().max())


def fit_quick(Xs, y):
    """The layer selection's fit: float32 L-BFGS on (C sum_i logloss + |w|^2 / 2) / n (the same minimizer as G), from zero, history 20,
    strong Wolfe, gradient tolerance 1e-5, change tolerance 1e-12, at most 1,000 iterations. Returns W [d, 1], b [1]."""
    # Original: src/probe_shift/torch_probe.py @ 199f50846, lines 17-42 (fit), called at run_avg_transfer.py @ 199f50846, line 190.
    n, d = Xs.shape
    Y = y.reshape(n, 1).to(Xs.dtype)
    W = torch.zeros(d, 1, device=Xs.device, dtype=Xs.dtype, requires_grad=True)
    b = torch.zeros(1, device=Xs.device, dtype=Xs.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=1000, tolerance_grad=1e-5, tolerance_change=1e-12, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (C * F.binary_cross_entropy_with_logits(Xs @ W + b, Y, reduction="sum") + 0.5 * (W * W).sum()) / n
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def standardize(Xg, units, in_float64):
    """Mean and population std of the units (zero std -> 1), and the standardized units, all in float64 or all in float32."""
    # Original: run_own_all.py @ 63096838c, lines 189-193 (float64); run_avg_transfer.py @ 199f50846, lines 1342-1344 (float32).
    Xt = Xg[torch.as_tensor(units, device=Xg.device)]
    Xt = Xt.double() if in_float64 else Xt.float()
    mean, std = Xt.mean(0), Xt.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    return mean, std, (Xt - mean) / std


def fit_copy(Xg, run, runs, fold, in_float64, tasks=None):
    """One fold copy: its training units, standardization and fit (ARITHMETIC, steps 1 and 2). Returns (W, b, mean, std) and a log entry."""
    t0 = time.time()
    units = training_units(run, runs, fold, tasks)
    y = torch.as_tensor(np.array([r["label"] for r in runs], np.float64)[run[units]], device=Xg.device)
    mean, std, Z = standardize(Xg, units, in_float64)
    Xs = Z.double()  # float32 standardized units are cast here (run_avg_transfer.py @ 199f50846, line 1347)
    del Z
    W, b, evals = fit_logistic(Xs, y)
    log = {"fold": fold, "units": len(units), "runs": len(np.unique(run[units])), "evaluations": evals,
           "max_gradient": max_gradient(Xs, y, W, b), "seconds": round(time.time() - t0, 1)}
    return (W, b, mean, std), log


# ---------------------------------------------------------------- layer selection (off-policy probe, tool outputs)


def validation_auroc(Xg, run, runs, outer_fold):
    """Copy `outer_fold`'s selection score for one layer: the mean over the other folds v of the run-level validation AUROC of the
    nested copy trained on folds other than `outer_fold` and v (ARITHMETIC, step 3)."""
    # Original: run_avg_transfer.py @ 199f50846, lines 185-224 (_select: fit_on, run_scores, the nested loop), recipe a, nested.
    label, fold = np.array([r["label"] for r in runs], np.float64), np.array([r["fold"] for r in runs])
    va = np.full(5, np.nan)
    for v in range(5):
        if v == outer_fold:
            continue
        units = np.flatnonzero(np.isin(run, np.flatnonzero((fold != outer_fold) & (fold != v))))
        mean, std, Z = standardize(Xg, units, in_float64=False)
        W, b = fit_quick(Z, torch.as_tensor(label[run[units]], device=Xg.device, dtype=torch.float32))
        del Z
        val = np.flatnonzero(fold == v)
        sc = np.flatnonzero(np.isin(run, val))
        s = ((Xg[torch.as_tensor(sc, device=Xg.device)].float() - mean) / std @ W + b).double().cpu().numpy()  # [units, 1]
        starts = np.unique(run[sc], return_index=True)[1]
        per_run = np.full(len(runs), np.nan)  # a run's score: the mean of its units' logits (attack_eval.pool, "mean")
        per_run[np.unique(run[sc])] = (np.add.reduceat(s, starts, axis=0) / np.diff(np.r_[starts, len(sc)])[:, None])[:, 0]
        s_val, y_val = per_run[val], label[val]
        ok = np.isfinite(s_val)
        p, q = s_val[ok & (y_val == 1)], s_val[ok & (y_val == 0)]
        va[v] = float((p[:, None] > q[None, :]).mean() + 0.5 * (p[:, None] == q[None, :]).mean())
    return float(np.nanmean(va))


def top_two(auroc_by_layer):
    """The two layers with the highest selection score (ties: the lower layer)."""
    # Original: run_avg_transfer.py @ 199f50846, line 1953 (analyze's rule).
    return sorted(auroc_by_layer, key=lambda L: (-auroc_by_layer[L], L))[:2]


# ---------------------------------------------------------------- saving


def save_on_policy(path, weights):
    """W, b, mean, std as a safetensors file (the stored on-policy format)."""
    # Original: run_own_all.py @ 63096838c, lines 616-619.
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({n: np.ascontiguousarray(t.cpu().numpy()) for n, t in zip(("W", "b", "mean", "std"), weights, strict=True)}, str(path))


def save_other(path, prefix, copies):
    """All five copies at one layer as an .npz file (the stored off-policy format); copies maps fold -> (W, b, mean, std)."""
    # Original: run_avg_transfer.py @ 199f50846, lines 1387-1389.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{prefix}__f{k}__{p}": t.cpu().numpy().reshape(-1) for k, ws in sorted(copies.items())
                      for p, t in zip(("W", "b", "mean", "std"), ws, strict=True)})


# ---------------------------------------------------------------- the probes


def train_on_policy(rows_dir, out, device):
    """The on-policy probe and its 25% and 10% variants: 3 x 2 readouts x 5 folds, layer 48, float64 standardization."""
    runs = read_table("on_policy_probe_runs.csv")
    log = {}
    for readout in READOUTS:
        X, run = load_training_rows(rows_dir, "on_policy", readout, 48, len(runs))
        Xg = torch.from_numpy(X).to(device)
        for name, fraction in ON_POLICY.items():
            for k in range(5):
                tasks = None
                if fraction < 1:
                    tasks = subsample_tasks({runs[r]["task"] for r in np.unique(run) if runs[r]["fold"] != k}, k, fraction)
                weights, log[f"{name}|{readout}|fold{k}"] = fit_copy(Xg, run, runs, k, in_float64=True, tasks=tasks)
                save_on_policy(Path(out) / name / f"{readout}__fold{k}.safetensors", weights)
        del Xg
    (Path(out) / "on_policy_fit_log.json").write_text(json.dumps(log, indent=1))


def train_other(probe, rows_dir, out, device, select_layers):
    """The off-policy or simulated-attack probe: tool outputs at each copy's two layers, commands at layer 63; float32 standardization."""
    name, prefix = OTHERS[probe]
    runs = read_table(f"{'off_policy' if probe == 'off_policy' else 'simulated_attack'}_probe_runs.csv")
    layers = {r["fold"]: [r["tool_outputs_first_layer"], r["tool_outputs_second_layer"]] for r in read_table("off_policy_probe_layers.csv")}
    log = {}
    if select_layers:  # rerun the selection on all 64 layers (off-policy probe only; the simulated-attack probe inherits its layers)
        if probe != "off_policy":
            raise SystemExit("--select-layers applies to the off-policy probe only")
        auroc = {k: {} for k in range(5)}
        for L in range(64):
            X, run = load_training_rows(rows_dir, probe, "tool_outputs", L, len(runs))
            Xg = torch.from_numpy(X).to(device)
            for k in range(5):
                auroc[k][L] = validation_auroc(Xg, run, runs, k)
            del Xg
        layers = {k: top_two(auroc[k]) for k in range(5)}
        log["selection"] = {"layers": layers, "validation_auroc": auroc}
    jobs = [("tool_outputs", L) for L in sorted({L for ls in layers.values() for L in ls})] + [("commands", 63)]
    for readout, L in jobs:
        X, run = load_training_rows(rows_dir, probe, readout, L, len(runs))
        Xg = torch.from_numpy(X).to(device)
        copies = {}
        for k in range(5):
            copies[k], log[f"{name}|{readout}|L{L}|fold{k}"] = fit_copy(Xg, run, runs, k, in_float64=False)
        save_other(Path(out) / name / f"{readout}__L{L}.npz", prefix, copies)
        del Xg
    (Path(out) / f"{probe}_fit_log.json").write_text(json.dumps(log, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe", choices=["on_policy", "off_policy", "simulated_attack"])
    ap.add_argument("--rows", type=Path, required=True, help="folder with the training rows (see INPUT)")
    ap.add_argument("--out", type=Path, required=True, help="folder for the weights and the fit log")
    ap.add_argument("--select-layers", action="store_true", help="off_policy: rerun the layer selection on all 64 layers first")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.probe == "on_policy":
        train_on_policy(a.rows, a.out, a.device)
    else:
        train_other(a.probe, a.rows, a.out, a.device, a.select_layers)


if __name__ == "__main__":
    main()
