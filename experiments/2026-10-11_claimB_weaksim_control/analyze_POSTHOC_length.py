"""POST HOC (PLAN.md amendment 1, rule committed at 06969719 before any number): does the frozen probe add anything beyond
run length on la18 R0 vs real? Stored scores only (`results/scores_2026-10-11_233726.json`); laptop CPU, $0, numbers only.

Readings, exactly as amendment 1 registers them: (1) the probe's AUROC within strata = environment x within-environment
tertile of the length measure (counted only with >= 2 runs of each class); (2) logistic regression of R0 = 1 on
z(log length) alone vs with z(probe), in-sample AUROCs, dAUC and the likelihood-ratio statistic, intervals from 2,000
main-task draws stratified by environment (both models refit per draw); (3) A0's estimand for the probe and for -log
length, and their difference with an interval from shared draws; (4) readings 2 and 3 for the 20 scrambled refits
(points). Rule: ADDS BEYOND LENGTH if dAUC's 2.5th percentile > 0 and dAUC > the scrambled dAUCs' 95th percentile;
NOT BEYOND ARBITRARY DIRECTIONS if dAUC's 97.5th percentile < that 95th percentile; else UNCLEAR. log n_tokens decides.

  uv run --no-project --python 3.12 --with numpy --with scikit-learn python \\
      experiments/2026-10-11_claimB_weaksim_control/analyze_POSTHOC_length.py results/scores_2026-10-11_233726.json
"""

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
REAL, R0 = "real-attack-la18", "sim-attack-la18-r0"
BOOT, SEED, C_BIG = 2000, 20261013, 1e6
MEASURES = ("n_tokens", "n_outputs")


def zscore_by_env(x, env):
    out = np.empty(len(x))
    for e in np.unique(env):
        k = env == e
        sd = x[k].std()
        out[k] = (x[k] - x[k].mean()) / (sd if sd > 0 else 1.0)
    return out


MAX_ITER = 10000


def fit_pair(y, xl, xp, w):
    """In-sample AUROCs of L (length) and LP (length + probe) from the linear scores, the LR statistic (stable log
    likelihood), and whether both fits converged; case weights w."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    res, ok = [], True
    for X in (xl[:, None], np.c_[xl, xp]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            m = LogisticRegression(C=C_BIG, max_iter=MAX_ITER).fit(X, y, sample_weight=w)
        ok &= bool(m.n_iter_[0] < MAX_ITER) and not any(issubclass(c.category, ConvergenceWarning) for c in caught)
        z = m.decision_function(X)
        ll = -float((w * (y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))).sum())
        res.append((roc_auc_score(y, z, sample_weight=w), ll))
    (a_l, ll_l), (a_lp, ll_lp) = res
    return a_l, a_lp, a_lp - a_l, 2 * (ll_lp - ll_l), ok


def a0(sim, real, w=None):
    """A0's estimand for one environment: AUROC over pairs of different main tasks, optionally with task weights w."""
    K = (sim[:, None] > real[None, :]) + 0.5 * (sim[:, None] == real[None, :])
    n = len(sim)
    if w is None:
        return (K.sum() - np.trace(K)) / (n * n - n)
    return (w @ K @ w - (w**2) @ np.diag(K)) / (n * n - (w**2).sum())


def main():
    path = HERE / sys.argv[1]
    raw = path.read_bytes()
    S = json.loads(raw)
    runs = S["la18"]
    by = defaultdict(dict)
    for i, r in enumerate(runs):
        by[(r["environment"], r["main_task"])][r["arm"]] = i
    cells = sorted(k for k, v in by.items() if REAL in v and R0 in v)
    if len(cells) != 198:
        raise ValueError(f"{len(cells)} eligible cells, expected the registered 198")
    idx = np.array([by[c][a] for c in cells for a in (REAL, R0)])
    env = np.array([runs[i]["environment"] for i in idx])
    task = np.array([f"{runs[i]['environment']}|{runs[i]['main_task']}" for i in idx])
    y = np.array([runs[i]["arm"] == R0 for i in idx], float)
    probe = np.array([runs[i]["score"] for i in idx])
    length = {m: np.log(np.array([float(runs[i][m]) for i in idx])) for m in MEASURES}
    zl = {m: zscore_by_env(v, env) for m, v in length.items()}  # amendment 1a: within environment, like the probe
    zp = zscore_by_env(probe, env)  # moments frozen at the full sample; draws reweight, they do not re-normalize
    envs = sorted(set(env))
    rng = np.random.default_rng(SEED)
    # task-clustered draws, stratified by environment; each draw's per-run weight = its task's count; a draw holding a
    # single task in an environment is redrawn (A0's rule)
    tasks_of = {e: sorted(set(task[env == e])) for e in envs}
    draws = []
    for _ in range(BOOT):
        w_task = {}
        for e in envs:
            t = tasks_of[e]
            c = rng.multinomial(len(t), np.full(len(t), 1 / len(t)))
            while (c > 0).sum() < 2:
                c = rng.multinomial(len(t), np.full(len(t), 1 / len(t)))
            w_task |= dict(zip(t, c))
        draws.append(np.array([w_task[t] for t in task], float))
    out = {"label": "POST HOC length-controlled reading (PLAN.md amendment 1)", "cells": len(cells), "runs": len(idx)}

    # (1) within length strata
    strata = {}
    for m in MEASURES:
        num = den = n_runs = 0
        for e in envs:
            k = env == e
            edges = np.quantile(length[m][k], [1 / 3, 2 / 3])
            s = np.searchsorted(edges, length[m][k], side="right")
            for b in range(3):
                kk = np.where(k)[0][s == b]
                sim, rea = kk[y[kk] == 1], kk[y[kk] == 0]
                if len(sim) < 2 or len(rea) < 2:
                    continue
                K = (probe[sim][:, None] > probe[rea][None, :]) + 0.5 * (probe[sim][:, None] == probe[rea][None, :])
                same = task[sim][:, None] == task[rea][None, :]
                num += K[~same].sum()
                den += (~same).sum()
                n_runs += len(kk)
        strata[m] = {"auroc_within_strata": float(num / den) if den else None, "pairs": int(den), "runs_in_counted_strata": n_runs}
    out["within_length_strata"] = strata

    # (2) added value over length, and (3) probe minus length on A0's estimand
    def a0_all(score, w=None):
        n_tot = len(cells)
        tot = 0.0
        for e in envs:
            ks, kr = (env == e) & (y == 1), (env == e) & (y == 0)
            # sim and real rows of one environment are both ordered by task (cells sorted), so weights align by task
            we = None if w is None else w[ks]
            tot += ks.sum() / n_tot * a0(score[ks], score[kr], we)
        return tot

    added, diff = {}, {}
    for m in MEASURES:
        pt = fit_pair(y, zl[m], zp, np.ones(len(y)))
        if not pt[4]:
            raise RuntimeError(f"{m}: the full-sample logistic fits did not converge")
        bs = np.array([fit_pair(y, zl[m], zp, w) for w in draws])
        n_bad = int((bs[:, 4] == 0).sum())
        bs = bs[bs[:, 4] == 1]  # unconverged draws are dropped and counted
        added[m] = {"auc_length": pt[0], "auc_length_plus_probe": pt[1], "dAUC": pt[2], "LR": pt[3], "draws_unconverged": n_bad,
                    "dAUC_ci95": [float(np.percentile(bs[:, 2], 2.5)), float(np.percentile(bs[:, 2], 97.5))],
                    "LR_ci95": [float(np.percentile(bs[:, 3], 2.5)), float(np.percentile(bs[:, 3], 97.5))]}
        a_p, a_l = a0_all(probe), a0_all(-length[m])
        bd = np.array([a0_all(probe, w) - a0_all(-length[m], w) for w in draws])
        diff[m] = {"A0_probe": a_p, "A0_minus_log_length": a_l, "probe_minus_length": a_p - a_l,
                   "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
    out["added_value_over_length"] = added
    out["probe_minus_length"] = diff

    # (4) the scrambled refits, points
    sc = S["scrambled"]
    bad = {int(k.split(":")[0].removeprefix("scramble")) for k in sc["unconverged"]}
    scr = {m: {"dAUC": [], "LR": [], "probe_minus_length": [], "unconverged": 0} for m in MEASURES}
    for d in [d for d in range(sc["per_probe"]) if d not in bad]:
        s_of = {}
        for draws_p in sc["draws"].values():
            s_of |= dict(zip(draws_p[d]["run_index"], draws_p[d]["score"]))
        sp = np.array([s_of[i] for i in idx])
        zsp = zscore_by_env(sp, env)
        for m in MEASURES:
            f = fit_pair(y, zl[m], zsp, np.ones(len(y)))
            if not f[4]:
                scr[m]["unconverged"] += 1
                continue
            scr[m]["dAUC"].append(f[2])
            scr[m]["LR"].append(f[3])
            scr[m]["probe_minus_length"].append(a0_all(sp) - a0_all(-length[m]))
    verdicts = {}
    for m in MEASURES:
        p95 = float(np.percentile(scr[m]["dAUC"], 95))
        lo, hi = added[m]["dAUC_ci95"]
        verdicts[m] = ("ADDS BEYOND LENGTH" if lo > 0 and added[m]["dAUC"] > p95 else
                       "NOT BEYOND ARBITRARY DIRECTIONS" if hi < p95 else "UNCLEAR")
        scr[m] |= {"draws": len(scr[m]["dAUC"]), "dAUC_p95": p95, "dAUC_max": float(max(scr[m]["dAUC"]))}
    out["scrambled"] = scr | {"upstream_unconverged_draws_dropped": sorted(bad)}
    out["verdict"] = {"deciding (log n_tokens)": verdicts["n_tokens"], "log n_outputs (reported)": verdicts["n_outputs"]}
    out["inputs"] = {str(path.relative_to(HERE)): hashlib.sha256(raw).hexdigest()}
    out["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"length_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    show = {k: out[k] for k in ("verdict", "within_length_strata", "added_value_over_length", "probe_minus_length")}
    show["scrambled"] = {m: {k: v for k, v in scr[m].items() if k not in ("dAUC", "LR", "probe_minus_length")} |
                         {"LR_max": float(max(scr[m]["LR"])), "probe_minus_length_max": float(max(scr[m]["probe_minus_length"]))}
                         for m in MEASURES}
    print(json.dumps(show, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
