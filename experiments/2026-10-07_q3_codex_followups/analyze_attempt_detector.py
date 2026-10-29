"""Analyze run_attempt_detector.py: does a probe trained on real e1 attempted vs declined attack runs pick out attempts?

Per readout and arm (e1 out of fold, e2 and simulated attack scored by the same fold models): run AUROC attempted
(positive) vs declined, on main tasks that have both kinds of run in that arm. Primary: the within-task AUROC (only
attempted-declined pairs from the same main task are compared, ties count 1/2), which ignores how often each task gets
attempted; pooled AUROC also reported. Both statistics get both nulls:
  refit:   the same AUROC for each of the 20 label-shuffle probes (labels permuted within main task before training).
           e1 trained the other folds' probes, so each e1 refit is scored against its own shuffled labels, rebuilt
           exactly as run_attempt_detector.fit_all drew them (same inputs, numpy Generator seed 0, same call order);
           e2 and simulated runs never trained anything and are scored against their real labels;
  permute: e2 and simulated attack only (labels never used in training): the real probe's scores with labels permuted
           within main task (1,000 draws, seed 0).
95% interval from 1,000 main-task resamples. Also the AUROC of each kind against real honest runs on the same tasks:
a probe for the attempt should put declined runs near honest ones.

Within a main task, attempted and declined runs usually differ in side task too, so the probe can win by reading the side
task (the within-task refit null for e2 and simulated runs is well above 1/2). The clean test holds the cell (main and
side task) fixed: for each pair of arms (e1-e2, e1-sim, e2-sim), the cells where one arm's run attempted and the other's
declined, and the share where the attempted run scores higher (ties 1/2). Both runs of a cell are in one fold, scored by
the same out-of-fold model. Nulls: the refit columns on the same pairs (true labels), and swapping the two runs' scores
within each pair (1,000 draws). The swap is valid for pairs with an e1 run too: the model scoring a pair never saw that
fold's labels, so under no signal the two runs are exchangeable unless the arms differ systematically. That is checked
directly: the mean score gap between the two arms over all their labelled cells, and an arm-balanced share (the mean
of the shares with the attempted run in either arm), which an arm offset cannot move. 95% interval from main-task
resamples.

  python analyze_attempt_detector.py --stamp <attempt_detector stamp> --labels <attempt labels>
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from probe_shift.attack_eval import E1, E2, HONEST, SIM, auc, boot_weights, summary  # noqa: E402


def within_task_auc(s, y, task, W=None):
    """Share of same-task (attempted, declined) pairs ranked correctly, ties 1/2; with W [B, runs] task-resample
    weights (a task's weight is its multiplicity), one value per resample."""
    conc, npairs, first = [], [], []
    for t in sorted(set(task)):
        i = np.flatnonzero(task == t)
        a, d = s[i][y[i] == 1], s[i][y[i] == 0]
        conc.append(float((a[:, None] > d[None, :]).sum() + 0.5 * (a[:, None] == d[None, :]).sum()))
        npairs.append(len(a) * len(d))
        first.append(i[0])
    conc, npairs = np.array(conc), np.array(npairs, float)
    if W is None:
        return conc.sum() / npairs.sum()
    w = W[:, first]
    return (w * conc).sum(1) / (w * npairs).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    rec = json.loads((HERE / "results" / f"attempt_detector_{args.stamp}.json").read_text())
    if rec["labels"]["sha256"] != hashlib.sha256(Path(args.labels).read_bytes()).hexdigest():
        raise ValueError("labels differ from the ones the probe was trained on")
    z = np.load(HERE / "results" / f"attempt_detector_{args.stamp}.npz")
    runs = json.loads((REPO / rec["runs_file"]).read_text())
    lab = {(x["arm"], x["key"]): x["label"] for x in map(json.loads, Path(args.labels).read_text().splitlines())}
    arm = np.array([r["arm"] for r in runs]); task = np.array([r["main_task"] for r in runs])
    label = np.array([lab.get((r["arm"], r["key"]), "") for r in runs])
    # the shuffled training labels, rebuilt as fit_all drew them
    y_run = np.array([1.0 if r["arm"] == E1 and lab.get((r["arm"], r["key"])) == "attempted"
                      else 0.0 if r["arm"] == E1 and lab.get((r["arm"], r["key"])) == "declined" else np.nan for r in runs])
    shuffles = rec["shuffles"]
    rng_fit = np.random.default_rng(0)
    trainable = ~np.isnan(y_run)
    Y = np.repeat(y_run[:, None], 1 + shuffles, axis=1)
    for k in range(1, 1 + shuffles):
        for tk in np.unique(task[trainable]):
            ix = np.flatnonzero(trainable & (task == tk))
            Y[ix, k] = y_run[ix][rng_fit.permutation(len(ix))]
    rng = np.random.default_rng(0)
    out = {"stamp": args.stamp, "labels": rec["labels"], "results": {}}
    for ro in rec["readouts"]:
        S = z[f"{ro}__scores"]  # [runs, 1 + shuffles]; column 0 = real labels
        for a in (E1, E2, SIM):
            m = (arm == a) & np.isin(label, ["attempted", "declined"]) & ~np.isnan(S[:, 0])
            y = (label == "attempted").astype(float)
            mixed = {t for t in set(task[m]) if 0 < y[m & (task == t)].mean() < 1}
            m &= np.isin(task, list(mixed))
            ix = np.flatnonzero(m)
            yy, s, tk = y[ix], S[ix, 0], task[ix]
            W = boot_weights(tk, 1000, 0)
            groups = [np.flatnonzero(tk == t) for t in sorted(mixed)]
            perms = []
            for k in range(1000 if a != E1 else 0):
                yp = yy.copy()
                for g in groups:
                    yp[g] = yy[g][rng.permutation(len(g))]
                perms.append(yp)
            stats = {}
            for stat, fn in (("within_task", lambda sc, lab: within_task_auc(sc, lab, tk)), ("pooled", roc_auc_score)):
                obs = fn(yy, s) if stat == "pooled" else fn(s, yy)
                f = (lambda sc, lab: roc_auc_score(lab, sc)) if stat == "pooled" else fn
                refit = np.array([f(S[ix, k], Y[ix, k] if a == E1 else yy) for k in range(1, S.shape[1])])
                perm = np.array([f(s, yp) for yp in perms])
                stats[stat] = {"value": summary(within_task_auc(s, yy, tk, W) if stat == "within_task" else auc(s, yy, W)),
                               "refit_null": {"mean": round(float(refit.mean()), 4), "max": round(float(refit.max()), 4),
                                              "p_one_sided": round((1 + int((refit >= obs).sum())) / (1 + len(refit)), 4)},
                               "permute_null": None if a == E1 else {"p2.5_p97.5": [round(float(v), 4) for v in np.percentile(perm, [2.5, 97.5])],
                                                                     "p_one_sided": round((1 + int((perm >= obs).sum())) / (1 + len(perm)), 4)}}
            # against real honest runs on the same main tasks
            h = np.flatnonzero((arm == HONEST) & np.isin(task, list(mixed)) & ~np.isnan(S[:, 0]))
            vs_honest = {kind: round(float(roc_auc_score(np.r_[np.ones((label[ix] == kind).sum()), np.zeros(len(h))],
                                                          np.r_[s[label[ix] == kind], S[h, 0]])), 4) for kind in ("attempted", "declined")}
            r = {"tasks": len(mixed), "attempted": int(yy.sum()), "declined": int((1 - yy).sum()), **stats,
                 "auroc_vs_real_honest": vs_honest | {"honest_runs": len(h)}}
            out["results"][f"{ro}|{a}"] = r
            fmt = lambda x: (f"{x['value']['value']:.3f} [{x['value']['ci95'][0]:.3f}, {x['value']['ci95'][1]:.3f}] refit {x['refit_null']['mean']:.3f}"
                             f"/max {x['refit_null']['max']:.3f} p {x['refit_null']['p_one_sided']:.3f}"
                             + (f" perm {x['permute_null']['p2.5_p97.5']} p {x['permute_null']['p_one_sided']:.3f}" if x["permute_null"] else ""))
            print(f"{ro:17s} {a:15s} tasks {len(mixed):3d} att/dec {r['attempted']}/{r['declined']} | within {fmt(stats['within_task'])} | "
                  f"pooled {fmt(stats['pooled'])} | vs honest att {vs_honest['attempted']:.3f} dec {vs_honest['declined']:.3f}")
    cell = np.array([r["cell"] for r in runs])
    W_all = boot_weights(task, 1000, 0)
    for ro in rec["readouts"]:
        S = z[f"{ro}__scores"]
        idx = {(arm[i], cell[i]): i for i in range(len(runs)) if label[i] in ("attempted", "declined") and not np.isnan(S[i, 0])}
        for a1, a2 in ((E1, E2), (E1, SIM), (E2, SIM)):
            pairs = []
            for c in sorted({c for (a, c) in idx if a == a1}):
                i, j = idx.get((a1, c)), idx.get((a2, c))
                if i is not None and j is not None and label[i] != label[j]:
                    pairs.append((i, j) if label[i] == "attempted" else (j, i))  # (attempted, declined)
            ia, idd = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            win = lambda col: (S[ia, col] > S[idd, col]) + 0.5 * (S[ia, col] == S[idd, col])
            w0 = win(0).astype(float)
            obs = w0.mean()
            refit = np.array([win(k).mean() for k in range(1, S.shape[1])])
            d = S[ia, 0] - S[idd, 0]
            flips = np.where(rng.random((1000, len(d))) < 0.5, -1.0, 1.0)
            swap = ((flips * d) > 0).mean(1) + 0.5 * ((flips * d) == 0).mean(1)
            wt = W_all[:, ia]  # both runs of a cell share its main task
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = (wt * w0).sum(1) / wt.sum(1)
            arm_att = {a1: int(sum(arm[i] == a1 for i in ia)), a2: int(sum(arm[i] == a2 for i in ia))}
            in_a1 = np.array([arm[i] == a1 for i in ia])
            balanced = (w0[in_a1].mean() + w0[~in_a1].mean()) / 2
            both = [(idx[(a1, c)], idx[(a2, c)]) for c in {c for (a, c) in idx if a == a1} if (a2, c) in idx]
            offset = float(np.mean([S[i, 0] - S[j, 0] for i, j in both]))
            r = {"cells": len(pairs), "attempted_run_in": arm_att, "attempted_scores_higher": summary(boot),
                 "attempted_higher_by_arm": {a1: round(float(w0[in_a1].mean()), 4), a2: round(float(w0[~in_a1].mean()), 4)},
                 "arm_balanced_share": round(float(balanced), 4), "arm_offset_mean_score": {"a1_minus_a2": round(offset, 4), "cells": len(both)},
                 "refit_null": {"mean": round(float(refit.mean()), 4), "max": round(float(refit.max()), 4),
                                "p_one_sided": round((1 + int((refit >= obs).sum())) / (1 + len(refit)), 4)},
                 "swap_null": {"p2.5_p97.5": [round(float(v), 4) for v in np.percentile(swap, [2.5, 97.5])],
                               "p_one_sided": round((1 + int((swap >= obs).sum())) / 1001, 4)},
                 "mean_score_gap": round(float(d.mean()), 4)}
            out["results"][f"{ro}|same_cell|{a1}~{a2}"] = r
            print(f"{ro:17s} same cell {a1}~{a2}: {len(pairs)} discordant cells (attempted in {arm_att}) | attempted higher "
                  f"{obs:.3f} [{r['attempted_scores_higher']['ci95'][0]:.3f}, {r['attempted_scores_higher']['ci95'][1]:.3f}] | refit "
                  f"{r['refit_null']['mean']:.3f}/max {r['refit_null']['max']:.3f} p {r['refit_null']['p_one_sided']:.3f} | swap "
                  f"{r['swap_null']['p2.5_p97.5']} p {r['swap_null']['p_one_sided']:.3f} | by arm {r['attempted_higher_by_arm']} balanced "
                  f"{balanced:.3f} | arm offset {offset:+.3f} over {len(both)} cells")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"attempt_detector_analysis_{args.stamp}__{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
