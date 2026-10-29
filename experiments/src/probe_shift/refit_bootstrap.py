"""One replicate of the off-policy 2x2 with every step refit: layer and C choice, fitting, task-disjoint calibration.

A replicate weights main tasks by their multiplicity (all ones for the point estimate, a task resample otherwise) by
repeating each task's rows; an unweighted fit on the repeated rows is the weighted fit. Per outer fold f (main-task
folds 0-4): test fold f, calibration fold f+1, validation fold f+2, inner training folds f+3 and f+4 (mod 5). For each
training world, `probes.select` fits every candidate layer and C on the inner training folds and picks on the
validation fold (earliest layer within tolerance of the best AUROC); the pick is refit on the inner training and
validation folds. Thresholds come from the calibration fold: at the target false-alarm rate on the test world's
read-only calls (primary), and on the training world's (secondary). Test-fold calls are scored by that fold's probe;
catch and false-alarm rates pool the test folds, and AUROC pools their scores.
"""
import numpy as np

WORLDS = ("real", "sim")


def task_counts(n_tasks: int, seed: int | None) -> np.ndarray:
    """Multiplicity per task: ones for the point estimate (seed None), else a resample with replacement."""
    if seed is None:
        return np.ones(n_tasks, int)
    rng = np.random.default_rng(seed)
    return np.bincount(rng.integers(0, n_tasks, n_tasks), minlength=n_tasks)


def shuffle_within(y: np.ndarray, group: np.ndarray, seed) -> np.ndarray:
    """A label-shuffle draw: y permuted within each group (transcripts, as in Q2's null; they sit inside main tasks, so
    every task and fold keeps its labels' counts)."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for g in np.unique(group):
        i = np.flatnonzero(group == g)
        out[i] = y[rng.permutation(i)]
    return out


def run_item(X, meta: list[dict], kind: str, i: int, cfg: dict) -> dict:
    """One unit of the run. kind "boot": replicate i, 0 the point estimate, else a main-task resample (seed cfg seed + i).
    kind "null": shuffle draw i, every task once and labels shuffled within transcripts (seed (cfg seed, i)), with the
    same layer and C choice, fitting and calibration as the point estimate."""
    y = np.array([m["y"] for m in meta])
    world = np.array([0 if m["arm"] == cfg["arms"]["real"] else 1 for m in meta])
    fold = np.array([m["fold"] for m in meta])
    tasks, task = np.unique([m["main_task"] for m in meta], return_inverse=True)
    if kind == "boot":
        counts = task_counts(len(tasks), None if i == 0 else cfg["seed"] + i)
    elif kind == "null":
        counts = task_counts(len(tasks), None)
        _, transcript = np.unique([f"{m['arm']}\0{m['key']}" for m in meta], return_inverse=True)
        y = shuffle_within(y, transcript, [cfg["seed"], i])
    else:
        raise ValueError(kind)
    out = replicate(X, y, world, fold, task, counts, cfg["layers"], cfg["C_grid"], cfg["layer_tolerance"], cfg["max_iter"], cfg["fpr"],
                    keep=kind == "boot" and i == 0)
    return out | {"task_counts": counts, "labels": y.astype(np.int8)}  # with "scores": enough to rescore any subset of runs


def replicate(X, y, world, fold, task, counts, layers, grid, tol, max_iter, fpr, keep=False) -> dict:
    """X: torch [layer, row, hidden] on the device. y (0/1), world (0 real, 1 sim), fold (0-4), task (index): numpy per
    row. counts: multiplicity per task. Returns per training world and test world: catch and false-alarm rate at the
    test-world calibrated threshold and at the training-world threshold, AUROC, the picks per fold, and under "scores"
    every row's score from each fold's probe ([fold, training world, row], float32; original row order), so a subset of
    runs can be rescored, and recalibrated on its calibration folds, without refitting. keep: also
    return each fold's probe under "probes" (score = ((x - mu) / sd) @ w + b on the float32 activations at its layer),
    with its thresholds."""
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift.probes import fit_logreg, select
    from probe_shift.transfer import threshold_at_fpr

    rep = np.repeat(np.arange(len(y)), counts[task])
    yr, wr, fr = y[rep], world[rep], fold[rep]
    acc = {(t, w): {"y": [], "s": [], "cal": [], "frozen": []} for t in (0, 1) for w in (0, 1)}
    picks, probes = [], []
    scores = np.full((5, 2, len(y)), np.nan, np.float32)
    for f in range(5):
        c, v, inner = (f + 1) % 5, (f + 2) % 5, [(f + 3) % 5, (f + 4) % 5]
        for t in (0, 1):
            tr_inner = rep[(wr == t) & np.isin(fr, inner)]
            val = rep[(wr == t) & (fr == v)]
            if len(set(y[tr_inner])) < 2 or len(set(y[val])) < 2:
                return {"undefined": f"fold {f}, world {WORLDS[t]}: a single class in inner training or validation"}
            pick = select(X, y, tr_inner, val, layers, grid, tol, max_iter)
            tr = rep[(wr == t) & np.isin(fr, inner + [v])]
            Xl = X[pick["li"]].float()
            mu, sd = Xl[tr].mean(0), Xl[tr].std(0).clamp_min(1e-6)
            Z = (Xl - mu) / sd
            w8, g = fit_logreg(Z[tr], torch.tensor(y[tr], dtype=torch.float32, device=X.device), pick["C"], max_iter)
            s = (Z @ w8[:-1] + w8[-1]).cpu().numpy()
            scores[f, t] = s
            del Xl, Z
            picks.append({"fold": f, "train": WORLDS[t], "layer": pick["layer"], "C": pick["C"], "val_auroc": round(pick["val_auroc"], 4),
                          "grad_max": g})
            if keep:
                probes.append({"w": w8[:-1].cpu().numpy(), "b": float(w8[-1]), "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(),
                               "fold": f, "train": WORLDS[t], "layer": pick["layer"], "C": pick["C"], "calibration_fold": c,
                               "validation_fold": v, "inner_folds": inner})
            neg_t = rep[(wr == t) & (fr == c) & (yr == 0)]
            frozen = threshold_at_fpr(s[neg_t], fpr) if len(neg_t) else np.nan
            for w in (0, 1):
                neg_w = rep[(wr == w) & (fr == c) & (yr == 0)]
                cal = threshold_at_fpr(s[neg_w], fpr) if len(neg_w) else np.nan
                test = rep[(wr == w) & (fr == f)]
                a = acc[(t, w)]
                a["y"].append(y[test]), a["s"].append(s[test])
                a["cal"].append(s[test] > cal), a["frozen"].append(s[test] > frozen)
                if keep:
                    probes[-1][f"threshold_{WORLDS[w]}"] = cal  # the training world's is the frozen threshold
    out = {"picks": picks, "scores": scores} | ({"probes": probes} if keep else {})
    for (t, w), a in acc.items():
        yy, ss = np.concatenate(a["y"]), np.concatenate(a["s"])
        cal, frz = np.concatenate(a["cal"]), np.concatenate(a["frozen"])
        pos, neg = yy == 1, yy == 0
        out[f"{WORLDS[t]}->{WORLDS[w]}"] = {
            "catch, calibrated in test world": float(cal[pos].mean()), "false-alarm rate, calibrated in test world": float(cal[neg].mean()),
            "catch, frozen training-world threshold": float(frz[pos].mean()), "false-alarm rate, frozen": float(frz[neg].mean()),
            "auroc": float(roc_auc_score(yy, ss)) if len(set(yy)) == 2 else float("nan"), "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
    return out
