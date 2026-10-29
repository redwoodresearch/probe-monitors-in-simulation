"""Item (A)'s summary (PLAN.md in this folder; written before any result exists). Reads run_halves_modal.py's score
arrays from the Volume (numbers only) and applies the committed rules, per direction (dev -> test, test -> dev) and
pooled:

1. Planted check, read first: plant 2 (R2 undone) on the held-out half, AUROC >= 0.910 and p < 0.05 against its 20
   refit per-pair shuffles; a direction that fails it is not read.
2. Strict alarm (catch at 1-in-100): the threshold is the 1% false-alarm point of ALL held-out real e1 runs (flag
   scores strictly above the (floor(1% n)+1)-th highest e1 score); catch = share of held-out fixed-sim runs flagged;
   yardstick = share of held-out real e2 runs flagged by the e2-vs-e1 probe at its own threshold set the same way;
   gap = sim catch - e2 catch, 95% interval from 2,000 main-task draws with the thresholds recomputed per draw; the
   sim catch's null from the 100 refit shuffle columns. Reading: "raises alarms on simulated runs beyond real-run
   noise" if the null p < 0.05 and the gap's lower bound > 0; "not beyond real-run noise (within 10 pts of two real
   runs)" if the gap's upper bound < 0.10; otherwise "too few".
3. Ranking score: AUROC fixed sim vs e1 and e2 vs e1 on the held-out half, delta = their difference, one-sided 95%
   bounds from 2,000 main-task draws; X = max(0.060, the grouped-null margin), the 80th percentile of the upper bound
   over the 100 shuffle worlds rounded up to 0.005, fixed before column 0's bound is read; X > 0.10 reads "not testable".
   Equivalence: upper < X "within X"; excess: lower > 0 "excess over real-vs-real", lower > X "exceeds the margin".
Pooled: each run keeps its own direction's score and threshold; catch pooled = caught / positives over both halves;
delta pooled = the run-weighted mean of the two directions' deltas; draws resample main tasks within each half.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with matplotlib python experiments/2026-10-11_claimB_halves/analyze_halves.py
"""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")]
import run_halves_modal as R  # noqa: E402

DRAWS, SEED, FLOOR, TOO_WIDE, ALARM_MARGIN, PLANT_FLOOR = 2000, 20261011, 0.060, 0.10, 0.10, 0.910


def wauc(y, s, W):
    """AUROC under each row of run weights W (ties count half): H3 Step 0's weighted AUROC (copied)."""
    o = np.argsort(s, kind="stable")
    s, y, W = s[o], y[o], W[:, o]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pos, neg = np.add.reduceat(W * y, starts, axis=1), np.add.reduceat(W * ~y, starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def thr(neg):
    s = np.sort(neg)[::-1]
    return s[int(np.floor(0.01 * len(s)))]


def share_above(pos, t):
    return float((pos > t).mean())


def main():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    runs_raw = read(f"{R.OUT}/runs.json")
    meta = json.loads(runs_raw)
    if meta["script_sha256"] != R.SCRIPT_SHA:
        raise SystemExit("runs.json comes from another runner version")
    runs = meta["runs"]
    env = np.array([r["environment"] for r in runs])
    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    out = {"label": "item (A): probes after all four fixes, environment halves (preliminary until reviewed)", "inputs": {"runs.json": hashlib.sha256(runs_raw).hexdigest()}}
    sel, Y, use = {}, {}, {}
    for d in R.DIRS:
        for comp in R.COMPS:
            raw = read(f"{R.OUT}/{d}/{comp}.npz")
            z = np.load(io.BytesIO(raw))
            if str(z["script_sha256"]) != R.SCRIPT_SHA:
                raise SystemExit(f"{d}/{comp} from another version")
            out["inputs"][f"{d}/{comp}.npz"] = hashlib.sha256(raw).hexdigest()
            V, S = z["val_auroc"], z["test_run_score"]
            if not np.isfinite(V).all():
                raise SystemExit(f"{d}/{comp}: non-finite validation AUROC")
            best = V.argmax(axis=0)  # per column
            sel[(d, comp)] = np.take_along_axis(S, best[None, None, :].repeat(S.shape[1], 1), 0)[0]
            Y[(d, comp)], use[(d, comp)] = z["Y"] > 0.5, z["use"]
    held = {d: np.isin(env, R.DIRS[d][1]) for d in R.DIRS}

    def idx(d, comp):
        return np.flatnonzero(use[(d, comp)] & held[d])

    rng = np.random.default_rng(SEED)
    res = {}
    for d in R.DIRS:
        r = {}
        # 1. planted check (plant 2)
        i = idx(d, "plant2")
        y, s = Y[(d, "plant2")][i], sel[(d, "plant2")][i]
        obs = roc_auc_score(y[:, 0], s[:, 0])
        null = np.array([roc_auc_score(y[:, k], s[:, k]) for k in range(1, y.shape[1])])
        p = float((1 + (null >= obs).sum()) / (1 + len(null)))
        r["planted"] = {"pairs": int(y[:, 0].sum()), "auroc": float(obs), "floor": PLANT_FLOOR, "refit_null_max": float(null.max()), "p": p,
                        "pass": bool(obs >= PLANT_FLOOR and p < 0.05)}
        res[d] = r
    # draws: main tasks resampled within each half (the same draws serve both directions' held-out halves)
    halves = {d: sorted(set(task[held[d] & np.isin(arm, [R.SIM, R.E1, R.E2])])) for d in R.DIRS}
    draws = {d: [rng.integers(0, len(halves[d]), len(halves[d])) for _ in range(DRAWS)] for d in R.DIRS}

    def weights(d, comp, dr):
        i = idx(d, comp)
        pos = {t: j for j, t in enumerate(halves[d])}
        counts = np.bincount(dr, minlength=len(halves[d]))
        return np.array([counts[pos[t]] for t in task[i]])

    def catch_cols(d, k, w_fx=None, w_e2=None):
        """(sim catch, e1 false-alarm rate, e2 flagged share) for label column k, optional per-run weights (draws)."""
        i1, i2 = idx(d, "fixed_vs_e1"), idx(d, "e2_vs_e1")
        y1, s1 = Y[(d, "fixed_vs_e1")][i1, k], sel[(d, "fixed_vs_e1")][i1, k]
        y2, s2 = Y[(d, "e2_vs_e1")][i2, k], sel[(d, "e2_vs_e1")][i2, k]
        if w_fx is not None:
            s1, y1 = np.repeat(s1, w_fx), np.repeat(y1, w_fx)
            s2, y2 = np.repeat(s2, w_e2), np.repeat(y2, w_e2)
        t1, t2 = thr(s1[~y1]), thr(s2[~y2])
        return share_above(s1[y1], t1), share_above(s1[~y1], t1), share_above(s2[y2], t2), int(y1.sum()), int(y2.sum()), share_above(s2[~y2], t2)

    def aurocs(d, k):
        return tuple(roc_auc_score(Y[(d, c)][idx(d, c), k], sel[(d, c)][idx(d, c), k]) for c in ("fixed_vs_e1", "e2_vs_e1"))

    W = {d: [(weights(d, "fixed_vs_e1", dr), weights(d, "e2_vs_e1", dr)) for dr in draws[d]] for d in R.DIRS}
    K = Y[(next(iter(R.DIRS)), "fixed_vs_e1")].shape[1]
    for d in R.DIRS:
        r = res[d]
        if not r["planted"]["pass"]:  # a direction that fails its planted check is not read at all
            r["catch_reading"] = r["auroc_reading"] = "not read (planted check failed)"
            continue
        cs, fpr, ce, n_s, n_e2, fpr2 = catch_cols(d, 0)
        gaps = np.array([(lambda c: c[0] - c[2])(catch_cols(d, 0, wf, we)) for wf, we in W[d]])
        nulls = np.array([catch_cols(d, k)[0] for k in range(1, K)])
        lo, hi = (float(x) for x in np.percentile(gaps, [2.5, 97.5]))
        pn = float((1 + (nulls >= cs).sum()) / K)
        r["catch_1pct"] = {"sim_catch": cs, "e1_fpr_fixed_probe": fpr, "e2_flagged": ce, "e1_fpr_yardstick_probe": fpr2, "gap": cs - ce, "gap_ci95": [lo, hi],
                           "sim_null_p95": float(np.percentile(nulls, 95)), "sim_null_p": pn, "sims": n_s, "e2_runs": n_e2}
        beyond, within = pn < 0.05 and lo > 0, hi < ALARM_MARGIN
        r["catch_reading"] = ("raises alarms on simulated runs beyond real-run noise" if beyond
                              else "not beyond real-run noise (within 10 pts of two real runs)" if within else "too few")
        # 3. AUROC delta with its margin from the null columns first
        Wf, We = np.array([w[0] for w in W[d]]), np.array([w[1] for w in W[d]])
        i1, i2 = idx(d, "fixed_vs_e1"), idx(d, "e2_vs_e1")

        def ub_lb(k):
            dd = wauc(Y[(d, "fixed_vs_e1")][i1, k], sel[(d, "fixed_vs_e1")][i1, k], Wf) - wauc(Y[(d, "e2_vs_e1")][i2, k], sel[(d, "e2_vs_e1")][i2, k], We)
            return float(np.percentile(dd, 95)), float(np.percentile(dd, 5))
        null_ub = np.array([ub_lb(k)[0] for k in range(1, K)])
        margin = float(np.ceil(np.percentile(null_ub, 80) / 0.005) * 0.005)
        X = max(FLOOR, margin)
        ub, lb = ub_lb(0)
        a1, a2 = aurocs(d, 0)
        r["auroc"] = {"fixed_vs_e1": a1, "e2_vs_e1": a2, "delta": a1 - a2, "upper_95": ub, "lower_95": lb, "X": X, "grouped_null_margin": margin, "testable": X <= TOO_WIDE}
        r["auroc_reading"] = (f"not testable (X = {X:.3f})" if X > TOO_WIDE else
                              {"equivalence": "within X" if ub < X else "not shown within X",
                               "excess": "exceeds the margin" if lb > X else "excess over real-vs-real" if lb > 0 else "no excess shown"})
    # pooled
    ds = list(R.DIRS)
    if all(res[d]["planted"]["pass"] for d in ds):
        def pooled(k, wsets=None):
            num_s = num_e = n_s = n_e = 0.0
            for j, d in enumerate(ds):
                wf, we = wsets[j] if wsets else (None, None)
                cs, _, ce, ns, ne, _ = catch_cols(d, k, wf, we)
                num_s, num_e, n_s, n_e = num_s + cs * ns, num_e + ce * ne, n_s + ns, n_e + ne
            return num_s / n_s, num_e / n_e
        ps, pe = pooled(0)
        pg = np.array([(lambda c: c[0] - c[1])(pooled(0, [W[d][b] for d in ds])) for b in range(DRAWS)])
        pnull = np.array([pooled(k)[0] for k in range(1, K)])
        lo, hi = (float(x) for x in np.percentile(pg, [2.5, 97.5]))
        pp = float((1 + (pnull >= ps).sum()) / K)
        fp1 = sum(res[d]["catch_1pct"]["e1_fpr_fixed_probe"] * res[d]["catch_1pct"]["sims"] for d in ds) / sum(res[d]["catch_1pct"]["sims"] for d in ds)
        fp2 = sum(res[d]["catch_1pct"]["e1_fpr_yardstick_probe"] * res[d]["catch_1pct"]["e2_runs"] for d in ds) / sum(res[d]["catch_1pct"]["e2_runs"] for d in ds)
        res["pooled"] = {"catch_1pct": {"sim_catch": ps, "e2_flagged": pe, "gap": ps - pe, "gap_ci95": [lo, hi], "sim_null_p": pp,
                                        "e1_fpr_fixed_probe": fp1, "e1_fpr_yardstick_probe": fp2},
                         "catch_reading": ("raises alarms on simulated runs beyond real-run noise" if pp < 0.05 and lo > 0
                                           else "not beyond real-run noise (within 10 pts of two real runs)" if hi < ALARM_MARGIN else "too few"),
                         "auroc_delta_weighted": float(sum(res[d]["auroc"]["delta"] * res[d]["catch_1pct"]["sims"] for d in ds) / sum(res[d]["catch_1pct"]["sims"] for d in ds))}
    else:
        res["pooled"] = {"catch_reading": "too few (a direction failed its planted check)"}
    out["results"] = res
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "run_script_sha256": R.SCRIPT_SHA}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"halves_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    plot(res, dst.with_suffix(".png"))
    print(json.dumps(res, indent=1))
    print("->", dst)


def plot(res, dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [d for d in ("dev_to_test", "test_to_dev", "pooled") if "catch_1pct" in res.get(d, {})]
    if not names:  # nothing read: no figure
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for j, d in enumerate(names):
        c = res[d]["catch_1pct"]
        for o, (key, lab, col) in enumerate((("sim_catch", "fixed sim", "#dd8452"), ("e2_flagged", "real e2 (yardstick)", "#4c72b0"))):
            x = j + (o - 0.5) * 0.36
            ax.bar(x, c[key], 0.34, color=col, label=lab if j == 0 else None)
            ax.text(x, c[key] + 0.005, f"{c[key]:.1%}", ha="center", fontsize=10)
        ax.text(j, -0.035, f"gap {c['gap']:+.1%}\n[{c['gap_ci95'][0]:+.1%}, {c['gap_ci95'][1]:+.1%}]", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.set_ylabel("share flagged at the 1% alarm\n(higher = more alarms)")
    ax.set_ylim(-0.06, max(0.15, max(res[n]["catch_1pct"]["sim_catch"] for n in names) + 0.05))
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)


if __name__ == "__main__":
    main()
