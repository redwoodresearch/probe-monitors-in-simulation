"""H3 planning (Aswin via 0d, 2026-10-11 about 10:00 and 10:05 PT): the smallest equivalence margin X that our test data
can test. Local and free: reads step A's stored grid cells (score arrays, no activations) from the Volume into memory,
as `2026-10-11_claimB_q11_posthoc/analyze_POSTHOC_stepA_catch_null.py` does. No new data, no fits.

The planned test: delta = AUROC(sim vs real e1) - AUROC(real e2 vs real e1) on the 5 test environments (473 runs per
arm). Its one-sided 95% upper bound is the 95th percentile of delta over 2,000 draws that resample main tasks, with one
draw serving both terms. Equivalence is claimed when the bound is below X. Power at X is the chance of that when the
simulator is truly as close to real e1 as real e2 is (true delta = 0). Two sets of such worlds:
- refit (H3's endpoint, probes refit on the fixed data): step A's 100 stored label shuffles. Each swaps the labels
  within every cell and refits the whole pipeline; step A's layer-and-C selection is redone here on every column.
  World k pairs fixed_vs_e1's shuffle k with e2_vs_e1's shuffle k.
  The stored shuffles flip the same cells in both comparisons (one generator, the same sorted cells), so e1 plays the
  same role in both terms, as it would in the real test.
- frozen (H1's endpoint, probes fixed): column 0's selected probes; 1,000 worlds swapping the two labels within every
  cell by coin flip, once with the same coins for both comparisons ("frozen_same_cells") and once with independent
  coins ("frozen_independent"). The first term flips sim against real e1, whose scores differ more than two real runs'.
All of these are conditional on these simulated worlds: a perfectly fixed simulator need not have exactly this spread.
Each world's bound is the planned test's: the 95th percentile of delta over 2,000 draws of main tasks.
X80 is the 80th percentile of the bound over worlds, with a 95% interval from 2,000 resamples of the worlds.
Detection (H4's question, Aswin 10:11): MDE80 is the smallest true delta whose one-sided 95% lower bound clears 0 in 80%
of worlds, from the same worlds, assuming a true delta shifts delta and its bounds by that amount. It is reported for
all test runs and for 100 cells (one simulated attack run per task, H4's size): each world draws its own 100 cells
(seeded; 10 subsamples per refit world) and the task draws then resample only those cells' main tasks. Probes stay the ones fitted on all data, so the
100-cell numbers are optimistic (a 100-task generation would also train on fewer runs).
Seeds: task draws default_rng(20261011) (all runs) and 20261013 (with the cell subsamples); flips 20261012; the
worlds' resampling for X80's interval 20261014.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy --with pyyaml \\
      python experiments/2026-10-11_claimB_h3_fix_to_plateau/analyze_power_margin.py
"""

import hashlib
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
CLAIMB = Path.home() / "probe-shift-claimb"  # the worktree at d9's pinned commit a1821be0
GRID_FILE = CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_grid.py"
PINNED, TAG, LEVEL, PACK = "a1821be0", "e402bcc321a8", "level1", "39f842231a46"
# step A's stored summary (the gate); untracked in this branch, read from the main checkout and hashed
SUMMARY = Path.home() / "probe-shift-sims/experiments/2026-10-09_fixed_sim/results/claimB_probe_summary_2026-10-10_021700.json"
sys.path[:0] = [str(CLAIMB / "src"), str(GRID_FILE.parent)]
import run_claimB_grid as G  # noqa: E402  (d9's module, unchanged)
from probe_shift import q1_eval  # noqa: E402  (a1821be0's)

COMPS = ("fixed_vs_e1", "e2_vs_e1")
KEEP = ("val_auroc", "test_run_score")
SEED, DRAWS, FROZEN_WORLDS, REFIT_SUBSAMPLES = 20261011, 2000, 1000, 10
X_GRID = np.round(np.arange(0.02, 0.1501, 0.0025), 4)
SHIFTS = (0.0, 0.01, 0.02)  # true delta: 0 (a perfect fix) and small residual gaps


def setup(cfg, runs, comparison):
    """G.setup's labels, shuffles and folds (copied from the catch-null script; runs from the Volume)."""
    q1_eval.DESIGNS["claimB"] = {"comparisons": G.COMPS}
    sc = q1_eval.Scorer(cfg | {"design": "claimB"}, runs, comparison)
    rng = np.random.default_rng(G.SEED + 1)
    cells = sorted({runs[i]["cell"] for i in np.flatnonzero(sc.runs_in)})
    Y = np.zeros((len(runs), 1 + G.N_NULL), np.float32)
    Y[:, 0] = sc.y_run
    for d in range(G.N_NULL):
        flip = dict(zip(cells, rng.random(len(cells)) < 0.5))
        Y[:, 1 + d] = [1 - v if (sc.runs_in[i] and flip[runs[i]["cell"]]) else v for i, v in enumerate(sc.y_run)]
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    return sc, Y, env


def read(vol, path):
    raw = b"".join(vol.read_file(path))
    z = np.load(io.BytesIO(raw))
    return {k: z[k] for k in KEEP} | {"folds": z["folds"], "sha256": hashlib.sha256(raw).hexdigest()}


def selected_scores(vol, cfg, runs, comp):
    """Step A's summarize selection (best layer and C on validation AUROC per outer fold) on every column."""
    sc, Y, env = setup(cfg, runs, comp)
    base = f"claimB/{LEVEL}/grid_{PACK}_{TAG}/{comp}"
    lc = [(l, C) for l in G.LAYERS for C in G.C_GRID]
    with ThreadPoolExecutor(16) as ex:
        R = dict(zip(lc, ex.map(lambda x: read(vol, f"{base}/L{x[0]}_C{x[1]:g}.npz"), lc)))
    folds = R[lc[0]]["folds"].tolist()
    V = np.stack([np.stack([R[(l, C)]["val_auroc"] for C in G.C_GRID]) for l in G.LAYERS])  # [layer, C, fold, column]
    c_ix, l_of = V.argmax(1), V.max(1).argmax(0)
    c_of = c_ix[l_of, np.arange(len(folds))[:, None], np.arange(Y.shape[1])[None, :]]
    S = np.full(Y.shape, np.nan)
    for i, f in enumerate(folds):
        m = env == f
        for k in range(Y.shape[1]):
            S[m, k] = R[(G.LAYERS[l_of[i, k]], G.C_GRID[c_of[i, k]])]["test_run_score"][m, k]
    r_in = np.flatnonzero(sc.runs_in)
    r = r_in[np.isin([runs[i]["environment"] for i in r_in], G.TEST_ENVS)]
    sha = hashlib.sha256("".join(v["sha256"] for _, v in sorted(R.items())).encode()).hexdigest()
    return sc, Y, S, r, sha


def wauc(y, s, W):
    """AUROC under each row of run weights W [draws, runs] (ties count half); y bool, s scores."""
    o = np.argsort(s, kind="stable")
    s, y, W = s[o], y[o], W[:, o]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pos, neg = np.add.reduceat(W * y, starts, axis=1), np.add.reduceat(W * ~y, starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def bound(y1, s1, W1, y2, s2, W2):
    """delta's one-sided 95% upper and lower bounds and its bootstrap sd; zero-weight runs are out of the sample."""
    d = wauc(y1, s1, W1) - wauc(y2, s2, W2)
    return float(np.percentile(d, 95)), float(np.percentile(d, 5)), float(d.std())


def q80(v, rng, groups):
    """80th percentile over worlds, with a 95% interval from resampling whole groups of worlds (a refit world's subsamples
    form one group; otherwise each world is its own group)."""
    v, groups = np.asarray(v), np.asarray(groups)
    ids = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in ids]
    b = [np.percentile(v[np.concatenate([members[j] for j in rng.integers(0, len(ids), len(ids))])], 80) for _ in range(2000)]
    return {"value": float(np.percentile(v, 80)), "mc_ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def summarize(deltas, ubs, lbs, ses, rng, groups, n_per_arm):
    deltas, ubs, lbs = np.array(deltas), np.array(ubs), np.array(lbs)
    x80 = q80(ubs, rng, groups)
    return {"worlds": len(ubs), "delta_mean": float(deltas.mean()), "delta_sd_across_worlds": float(deltas.std()),
            "bootstrap_sd_mean": float(np.mean(ses)), "power_at_X_true_delta_0": {f"{x:g}": float((ubs < x).mean()) for x in X_GRID},
            "X80": x80, "X80_if_true_delta": {f"{d:g}": x80["value"] + d for d in SHIFTS},  # assumes a true delta of d shifts the bound by d
            "MDE80": q80(-lbs, rng, groups), "independent_world_groups": int(len(np.unique(groups))), "runs_per_arm": n_per_arm,
            "runs_per_arm_for_X_0.03_assumed_sqrt_n": int(np.ceil(n_per_arm * (x80["value"] / 0.03) ** 2)),
            "world_bounds": {"upper": [round(x, 5) for x in ubs], "lower": [round(x, 5) for x in lbs]}}


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLAIMB, capture_output=True, text=True, check=True).stdout.strip()
    assert head.startswith(PINNED), f"the claimb worktree is at {head}, not {PINNED}"
    tag = hashlib.sha256(GRID_FILE.read_bytes() + (CLAIMB / "src/probe_shift/torch_probe.py").read_bytes()
                         + (CLAIMB / "src/probe_shift/q1_eval.py").read_bytes()).hexdigest()[:12]
    assert tag == TAG, f"grid code tag {tag}, expected {TAG}"
    folds = json.loads((CLAIMB / "splits/folds_v2.json").read_text())
    cfg = {"packs": {LEVEL: PACK}, "seed": G.SEED, "bootstrap": G.BOOT, "tag": TAG,
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    vol = modal.Volume.from_name(G.ACTS, version=2)
    runs_raw = b"".join(vol.read_file(f"claimB/{LEVEL}/packed_{PACK}/runs.L{G.LAYERS[0]}.json"))
    runs = json.loads(runs_raw)
    summary = json.loads(SUMMARY.read_text())["levels"][LEVEL]
    D = {}
    for comp in COMPS:
        sc, Y, S, r, sha = selected_scores(vol, cfg, runs, comp)
        if not np.isfinite(S[r]).all():
            raise SystemExit(f"{comp}: a compared run has no selected score in some column")
        ref = summary[comp]["best_layer_and_C"]["test_envs"]
        auc0, nul = roc_auc_score(Y[r, 0], S[r, 0]), G.null_against(G.col_auc(Y[r], S[r]))
        if abs(auc0 - ref["test"]["auroc"]) > 1e-4 or abs(nul["null_p95"] - ref["null"]["null_p95"]) > 1e-4:
            raise SystemExit(f"{comp}: AUROC {auc0} / null p95 {nul['null_p95']} do not reproduce step A's summary; stop")
        D[comp] = {"y": Y[r] > 0.5, "S": S[r], "task": sc.main[r], "cell": np.array([runs[i]["cell"] for i in r]),
                   "auroc": float(auc0), "null_p95": nul["null_p95"], "cells_sha256": sha}
        print(f"{comp}: gate OK, {len(r)} runs, AUROC {auc0:.4f}, null p95 {nul['null_p95']:.4f}", flush=True)
    tasks = np.unique(np.concatenate([D[c]["task"] for c in COMPS]))
    rng = np.random.default_rng(SEED)
    counts = np.zeros((DRAWS, len(tasks)))
    np.add.at(counts, (np.repeat(np.arange(DRAWS), len(tasks)), rng.integers(0, len(tasks), DRAWS * len(tasks))), 1)
    W = {c: counts[:, np.searchsorted(tasks, D[c]["task"])] for c in COMPS}
    a, b = D["fixed_vs_e1"], D["e2_vs_e1"]
    for c in COMPS:  # the weighted AUROC matches sklearn at unit weights
        assert abs(wauc(D[c]["y"][:, 0], D[c]["S"][:, 0], np.ones((1, len(D[c]["S"]))))[0] - D[c]["auroc"]) < 1e-9

    observed = bound(a["y"][:, 0], a["S"][:, 0], W["fixed_vs_e1"], b["y"][:, 0], b["S"][:, 0], W["e2_vs_e1"])
    out = {"label": "H3 planning: power simulation for the equivalence margin (no new data; preliminary until reviewed)",
           "test": "delta = AUROC(fixed sim vs e1) - AUROC(e2 vs e1), test 5, one-sided 95% bound = 95th percentile over main-task draws",
           "observed_delta_column0": {"delta": a["auroc"] - b["auroc"], "upper_bound_95": observed[0], "lower_bound_95": observed[1]},
           "runs": {c: int(len(D[c]["S"])) for c in COMPS}, "main_tasks": int(len(tasks)), "draws": DRAWS, "seed": SEED}

    cells = np.unique(np.concatenate([a["cell"], b["cell"]]))
    out["cells"] = {"fixed_vs_e1": int(len(np.unique(a["cell"]))), "e2_vs_e1": int(len(np.unique(b["cell"]))), "union": int(len(cells))}
    cell_task = {c: t for d in (a, b) for c, t in zip(d["cell"], d["task"])}
    srng, mrng = np.random.default_rng(SEED + 2), np.random.default_rng(SEED + 3)

    def weights(keep):
        """Task draws restricted to the kept cells' main tasks (all cells when keep is None), and the kept-run masks."""
        if keep is None:
            return W, {c: np.ones(len(D[c]["S"]), bool) for c in COMPS}
        t_keep = np.unique([cell_task[c] for c in keep])
        cnt = np.zeros((DRAWS, len(tasks)))
        np.add.at(cnt, (np.repeat(np.arange(DRAWS), len(t_keep)), np.searchsorted(tasks, t_keep[srng.integers(0, len(t_keep), DRAWS * len(t_keep))])), 1)
        mask = {c: np.isin(D[c]["cell"], keep) for c in COMPS}
        return {c: cnt[:, np.searchsorted(tasks, D[c]["task"])] * mask[c] for c in COMPS}, mask

    def worlds(label_pairs, n_cells, reps=1):
        deltas, ubs, lbs, ses = [], [], [], []
        groups = [g for g in range(len(label_pairs)) for _ in range(reps)]
        for ya, sa, yb, sb in [x for x in label_pairs for _ in range(reps)]:
            Wk, mask = weights(None if n_cells is None else srng.choice(cells, n_cells, replace=False))
            ub, lb, se = bound(ya, sa, Wk["fixed_vs_e1"], yb, sb, Wk["e2_vs_e1"])
            deltas.append(wauc(ya, sa, mask["fixed_vs_e1"][None] * 1.0)[0] - wauc(yb, sb, mask["e2_vs_e1"][None] * 1.0)[0])
            ubs.append(ub), lbs.append(lb), ses.append(se)
        n_per_arm = int(a["y"][:, 0].sum()) if n_cells is None else n_cells  # one simulated run per cell
        return summarize(deltas, ubs, lbs, ses, mrng, groups, n_per_arm)

    refit = [(a["y"][:, k], a["S"][:, k], b["y"][:, k], b["S"][:, k]) for k in range(1, 1 + G.N_NULL)]
    frng = np.random.default_rng(SEED + 1)
    coins = [dict(zip(cells, frng.random(len(cells)) < 0.5)) for _ in range(2 * FROZEN_WORLDS)]
    flip = lambda d, f: d["y"][:, 0] ^ np.array([f[c] for c in d["cell"]])  # noqa: E731
    same = [(flip(a, f), a["S"][:, 0], flip(b, f), b["S"][:, 0]) for f in coins[:FROZEN_WORLDS]]
    indep = [(flip(a, coins[i]), a["S"][:, 0], flip(b, coins[FROZEN_WORLDS + i]), b["S"][:, 0]) for i in range(FROZEN_WORLDS)]
    for n_cells in (None, 100):
        key = "all_test_runs" if n_cells is None else "100_cells"
        reps = 1 if n_cells is None else REFIT_SUBSAMPLES  # each refit world gets several cell subsamples at 100 cells
        out[key] = {"refit": worlds(refit, n_cells, reps), "frozen_same_cells": worlds(same, n_cells), "frozen_independent": worlds(indep, n_cells)}

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "grid": {"code": head, "tag": TAG, "pack": PACK},
            "inputs": {"runs_json": hashlib.sha256(runs_raw).hexdigest(), SUMMARY.name: hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
                       **{f"cells_{c}": D[c]["cells_sha256"] for c in COMPS}}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"power_margin_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(f"observed delta {out['observed_delta_column0']['delta']:.4f}, bound {observed[0]:.4f}")
    for key in ("all_test_runs", "100_cells"):
        for w, v in out[key].items():
            print(f"{key} {w}: delta mean {v['delta_mean']:+.4f}, sd across worlds {v['delta_sd_across_worlds']:.4f}, bootstrap sd "
                  f"{v['bootstrap_sd_mean']:.4f} | X80 {v['X80']['value']:.4f} {[round(x, 4) for x in v['X80']['mc_ci95']]} | MDE80 "
                  f"{v['MDE80']['value']:.4f} {[round(x, 4) for x in v['MDE80']['mc_ci95']]}")
    print("->", dst)


if __name__ == "__main__":
    main()
