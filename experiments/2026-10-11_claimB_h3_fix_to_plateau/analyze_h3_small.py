"""The smaller H3's summary (PLAN.md amendment 1 on branch aswin/claimb-h-plans; written before any H3 result). Reads the
refit's saved arrays (numbers only) from the Volume and applies the frozen rules:

- Selection: step A's, the best (layer, C) per fold and label column on validation AUROC.
- delta = AUROC(fixed sim vs e1) - AUROC(e2 vs e1) on the 5 test environments, evened; the one-sided 95% upper and
  lower bounds over 2,000 main-task draws (H3 Step 0's bound).
- X = max(0.060, the grouped-null margin): the 80th percentile of the upper bound over the 100 shuffle worlds (column k
  of both real comparisons, same coins), rounded up to 0.005, computed from the null columns before column 0's bound
  is read. X > 0.10 means "not testable at this size".
- Equivalence: upper bound < X. Excess: lower bound > 0; lower bound > X means "exceeds the margin". Reported apart.
- Planted checks: AUROC >= 0.833 (check 1), >= 0.910 (check 2), and p < 0.05 against 20 refit per-pair shuffles (the
  same fitting and selection; p = (1 + exceedances) / 21, so one exceedance fails), else the reading is void (Aswin's pick
  about 13:56 PT, PR #722 review QRWy7). The classifier-fixed per-pair null (1,000 draws) is reported beside it.
- Catch at 1% (headline) and 5% with their shuffle nulls, its own lines; B-dev's frozen-probe delta, descriptive.
- H2 read "inconclusive" (h2_summary_2026-10-11_132926.json), so both pipelines are read, each by the rules above with
  its own planted checks and X: step A's refit grid and H2's M2 (read only if M2 reproduced H2 within 0.005).
  Equivalence needs both upper bounds below their own X; excess is reported per pipeline.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy --with pyyaml \\
      python experiments/2026-10-11_claimB_h3_fix_to_plateau/analyze_h3_small.py
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
TPR = HERE.parent / "2026-10-09_fixed_sim" / "analyze_tpr_at_fpr.py"  # this checkout's copy, hashed into the result
sys.path[:0] = [str(HERE), str(TPR.parent)]
import analyze_tpr_at_fpr as T  # noqa: E402  (two-way cross-fit catch)
import run_h3_small_modal as R  # noqa: E402  (OUT, constants)

COMPS = ("fixed_vs_e1", "e2_vs_e1", "plant1", "plant2")
PLANT_MIN = {"plant1": 0.833, "plant2": 0.910}
DRAWS, SEED, FLOOR, TOO_WIDE = 2000, 20261011, 0.060, 0.10
ALARM_MARGIN = 0.10  # strict-alarm answer: "within 10 pts of two real runs" (the project's catch margin, 10-09 evening)


def wauc(y, s, W):
    """AUROC under each row of run weights W (ties count half): H3 Step 0's weighted AUROC."""
    o = np.argsort(s, kind="stable")
    s, y, W = s[o], y[o], W[:, o]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pos, neg = np.add.reduceat(W * y, starts, axis=1), np.add.reduceat(W * ~y, starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def select(V, S, env, folds):
    if not np.isfinite(V).all():
        raise SystemExit("non-finite validation AUROCs")
    out = np.full(S.shape[1:], np.nan)
    best = V.argmax(axis=0)
    for i, f in enumerate(folds):
        fr = np.flatnonzero(env == f)
        out[fr] = np.take_along_axis(S[:, fr, :], best[i][None, None, :].repeat(len(fr), 1), 0)[0]
    return out


def alarm_gap(Y, sel, task, rng):
    """Strict-alarm answer (added about 14:5x PT 10-11, before any H3 result was read; Aswin via 0d): catch at the 1%
    target on fixed sim vs e1 minus the same on e2 vs e1 (two real runs, the yardstick). One split policy for the point
    estimate and the interval: the union of both comparisons' main tasks, two-way cross-fit halves over that union
    (observed: the 1% table's seeded permutation; draws: DRAWS resamples of the union's tasks, thresholds redone per
    draw). Returns (observed fixed share, observed e2 share or None, draws, skipped draws)."""
    f_, e_ = "fixed_vs_e1", "e2_vs_e1"
    tasks = np.unique(np.concatenate([task[f_], task[e_]]))
    mem = {c: [np.flatnonzero(task[c] == t) for t in tasks] for c in (f_, e_)}

    def fit(c, k, counts, perm):
        return T.crossfit(Y[c][:, k], sel[c][:, k], mem[c], counts, perm)

    def shares(counts, perm):
        if len(perm) < 2:
            return None
        r = [fit(c, 0, counts, perm) for c in (f_, e_)]
        return None if not all(r) else [x[1]["caught"] / x[1]["positives"] for x in r]
    ones, perm0 = np.ones(len(tasks), int), np.random.default_rng(T.SEED).permutation(len(tasks))
    r0 = [fit(c, 0, ones, perm0) for c in (f_, e_)]
    obs = None if not all(r0) else {"fixed": r0[0][1]["caught"] / r0[0][1]["positives"], "e2": r0[1][1]["caught"] / r0[1][1]["positives"],
                                     "fpr_fixed": r0[0][1]["false_alarms"] / r0[0][1]["negatives"], "fpr_e2": r0[1][1]["false_alarms"] / r0[1][1]["negatives"]}
    if obs is not None:  # the fixed comparison's shuffle null under the same split
        nl = [fit(f_, k, ones, perm0) for k in range(1, Y[f_].shape[1])]
        nl = np.array([x[1]["caught"] / x[1]["positives"] for x in nl if x])
        obs["null_n"] = int(len(nl))
        obs["null_p"] = float((1 + (nl >= obs["fixed"]).sum()) / (1 + len(nl))) if len(nl) else None
    d, skipped = [], 0
    for _ in range(DRAWS):
        counts = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))
        sh = shares(counts, rng.permutation(np.flatnonzero(counts)))
        if sh is None:
            skipped += 1
        else:
            d.append(sh[0] - sh[1])
    return obs, np.array(d), skipped


def catch(y, s, task):
    tasks, inv = np.unique(task, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
    return T.crossfit(y, s, members, np.ones(len(tasks), int), np.random.default_rng(T.SEED).permutation(len(tasks)))


def read_pipeline(name, fname, read, env, folds, pruns, inputs):
    """One pipeline's reading by the frozen rules: planted floors + classifier-fixed null, X from the null columns first,
    delta's bounds, the AUROC nulls, and catch at 1% and 5%."""
    out, sel, Y, test = {}, {}, {}, {}
    for comp in COMPS:
        raw = read(f"{R.OUT}/refit/{fname(comp)}.npz")
        z = np.load(io.BytesIO(raw))
        if str(z["script_sha256"]) != R.SCRIPT_SHA or list(z["folds"]) != folds:
            raise SystemExit(f"{fname(comp)}.npz comes from another version or folds")
        inputs[f"{fname(comp)}.npz"] = hashlib.sha256(raw).hexdigest()
        y = z["Y"] > 0.5
        in_test = np.isin(env, folds) & np.isfinite(z["test_run_score"][0, :, 0])
        sel[comp], Y[comp], test[comp] = select(z["val_auroc"], z["test_run_score"], env, folds)[in_test], y[in_test], in_test
    if name == "step A":
        out["unconverged_selected_fit_pairs_note"] = "per-column gradients saved; step A had no convergence filter"
    task = {c: np.array([pruns[i]["main_task"] for i in np.flatnonzero(test[c])]) for c in COMPS}
    cell = {c: np.array([pruns[i]["cell"] for i in np.flatnonzero(test[c])]) for c in COMPS}
    # planted checks: AUROC floors and the refit-shuffle null (binding); the classifier-fixed per-pair null beside it
    rng = np.random.default_rng(SEED)
    planted_ok = True
    for c in ("plant1", "plant2"):
        y, s = Y[c][:, 0], sel[c][:, 0]
        obs = roc_auc_score(y, s)
        pairs = np.unique(cell[c])
        null = []
        for _ in range(1000):
            flip = dict(zip(pairs, rng.random(len(pairs)) < 0.5))
            null.append(roc_auc_score(y ^ np.array([flip[x] for x in cell[c]]), s))
        p = (1 + sum(n >= obs for n in null)) / 1001
        refit = np.array([roc_auc_score(Y[c][:, k], sel[c][:, k]) for k in range(1, Y[c].shape[1])])
        if len(refit) != R.PLANT_SHUFFLES:
            raise SystemExit(f"{c}: {len(refit)} refit shuffles, expected {R.PLANT_SHUFFLES}")
        p_refit = float((1 + (refit >= obs).sum()) / (1 + len(refit)))
        out[c] = {"auroc": float(obs), "floor": PLANT_MIN[c], "refit_null_max": float(refit.max()), "refit_null_median": float(np.median(refit)),
                  "p_refit": p_refit, "classifier_fixed_null_p95": float(np.percentile(null, 95)), "p_classifier_fixed": float(p),
                  "pass": bool(obs >= PLANT_MIN[c] and p_refit < 0.05)}
        planted_ok &= out[c]["pass"]
    # delta, its bounds, and the margin from the null columns first
    f_, e_ = "fixed_vs_e1", "e2_vs_e1"
    tasks = np.unique(np.concatenate([task[f_], task[e_]]))
    cnt = np.zeros((DRAWS, len(tasks)))
    np.add.at(cnt, (np.repeat(np.arange(DRAWS), len(tasks)), np.random.default_rng(SEED).integers(0, len(tasks), DRAWS * len(tasks))), 1)
    Wf, We = cnt[:, np.searchsorted(tasks, task[f_])], cnt[:, np.searchsorted(tasks, task[e_])]

    def bounds(k):
        d = wauc(Y[f_][:, k], sel[f_][:, k], Wf) - wauc(Y[e_][:, k], sel[e_][:, k], We)
        return float(np.percentile(d, 95)), float(np.percentile(d, 5))
    null_ub = np.array([bounds(k)[0] for k in range(1, R.K)])
    margin_null = float(np.ceil(np.percentile(null_ub, 80) / 0.005) * 0.005)
    X = max(FLOOR, margin_null)
    out["margin"] = {"grouped_null_X80_rounded": margin_null, "X": X, "testable": X <= TOO_WIDE}
    ub, lb = bounds(0)  # column 0 read only after X is fixed
    a_f, a_e = roc_auc_score(Y[f_][:, 0], sel[f_][:, 0]), roc_auc_score(Y[e_][:, 0], sel[e_][:, 0])
    out["delta"] = {"auroc_fixed_vs_e1": float(a_f), "auroc_e2_vs_e1": float(a_e), "delta": float(a_f - a_e), "upper_bound_95": ub, "lower_bound_95": lb}
    nulls = {c: np.array([roc_auc_score(Y[c][:, k], sel[c][:, k]) for k in range(1, R.K)]) for c in (f_, e_)}
    out["auroc_nulls"] = {c: {"p95": float(np.percentile(v, 95)), "p": float((1 + (v >= (a_f if c == f_ else a_e)).sum()) / R.K)} for c, v in nulls.items()}
    out["valid"] = bool(planted_ok and out["margin"]["testable"])
    if not planted_ok:
        out["reading"] = "planted checks failed: nothing read"
    elif not out["margin"]["testable"]:
        out["reading"] = f"not testable at this size (X = {X:.3f} > {TOO_WIDE})"
    else:
        out["reading"] = {"equivalence": "within X" if ub < X else "not shown within X",
                          "excess": "exceeds the margin" if lb > X else "excess over real-vs-real" if lb > 0 else "no excess shown"}
    for pct in T.TARGETS:  # catch, its own lines
        for c in (f_, e_):
            cf = catch(Y[c][:, 0], sel[c][:, 0], task[c])
            nl = [catch(Y[c][:, k], sel[c][:, k], task[c]) for k in range(1, R.K)]
            share = lambda x: x[pct]["caught"] / x[pct]["positives"]  # noqa: E731
            if not cf or not any(nl):
                out[f"catch_{pct}pct_{c}"] = {"caught": None, "note": "cross-fit unavailable"}
                continue
            out[f"catch_{pct}pct_{c}"] = {"caught": share(cf), "fpr": cf[pct]["false_alarms"] / cf[pct]["negatives"],
                                          "null_p95": float(np.percentile([share(x) for x in nl if x], 95)),
                                          "p": float((1 + sum(share(x) >= share(cf) for x in nl if x)) / (1 + sum(1 for x in nl if x)))}
    # strict-alarm answer, per pipeline (added before any H3 result was read)
    obs, g, skipped = alarm_gap(Y, sel, task, np.random.default_rng(SEED + 7))
    enough = obs is not None and obs["null_p"] is not None and obs["null_n"] >= 0.95 * (R.K - 1) and len(g) >= 0.95 * DRAWS
    ans = {"draws": int(len(g)), "skipped_draws": int(skipped), "planted_ok": bool(planted_ok), "split": "union of both comparisons' main tasks"}
    if enough:
        ans |= {"fixed_null_p": obs["null_p"], "fixed_null_n": obs["null_n"], "fpr_fixed_vs_e1": obs["fpr_fixed"], "fpr_e2_vs_e1": obs["fpr_e2"],
                "fpr_off_target": bool(obs["fpr_fixed"] > 0.02 or obs["fpr_e2"] > 0.02)}
        lo, hi = (float(x) for x in np.percentile(g, [2.5, 97.5]))
        lo_b, hi_b = (float(x) for x in np.percentile(g, [1.25, 98.75]))  # Bonferroni over the two pipelines (overall answer)
        ans |= {"catch_1pct_fixed_vs_e1": obs["fixed"], "catch_1pct_e2_vs_e1": obs["e2"], "gap": obs["fixed"] - obs["e2"], "gap_ci95": [lo, hi], "gap_ci97_5": [lo_b, hi_b]}
        beyond, within = planted_ok and obs["null_p"] < 0.05 and lo > 0, planted_ok and hi < ALARM_MARGIN
        ans["reading"] = ("too few (planted checks failed)" if not planted_ok else "raises alarms on simulated runs beyond real-run noise" if beyond
                          else f"not beyond real-run noise (within {ALARM_MARGIN:.0%} of two real runs)" if within else "too few")
        ans["overall_pass"] = bool(planted_ok and obs["null_p"] < 0.025 and lo_b > 0)
        ans["overall_within"] = bool(planted_ok and hi_b < ALARM_MARGIN)
    else:
        ans |= {"reading": "too few (no estimate: observed cross-fit or its null unavailable, or under 95% of draws valid)", "overall_pass": False, "overall_within": False}
    out["alarm_answer"] = ans
    return out, test, Y


def main():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    folds_file = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())
    e2f = folds_file["environment_folds"]["environment_to_fold"]
    pruns = json.loads(read(f"{R.M.PACK}/runs.L63.json"))
    env = np.array([e2f[m["environment"]] for m in pruns])
    folds = sorted({e2f[e] for e in R.TEST})
    out = {"label": "smaller H3 summary (on previously inspected test environments; rules reproducible from dev counts alone)", "inputs": {}}
    # H2 read "inconclusive", so both pipelines are read (H3 PLAN's table): step A's and H2's winning configuration M2
    repro = json.loads(read(f"{R.OUT}/refit/m2_reproduction.json"))
    if not repro["pass"] or repro["script_sha256"] != R.SCRIPT_SHA:
        raise SystemExit("M2 did not reproduce H2 in this version: nothing read")
    out["m2_reproduction"] = repro
    out["pipelines"], tests, Ys = {}, {}, {}
    for name, fname in (("step A", lambda c: c), ("M2", lambda c: f"{c}_M2")):
        out["pipelines"][name], tests[name], Ys[name] = read_pipeline(name, fname, read, env, folds, pruns, out["inputs"])
    P = out["pipelines"]
    both = all(P[n]["valid"] for n in P)
    A = {n: P[n]["alarm_answer"] for n in P}
    out["alarm_answer"] = {"per_pipeline": {n: A[n]["reading"] for n in A},
                           "overall": ("raises alarms on simulated runs beyond real-run noise" if any(A[n]["overall_pass"] for n in A) else
                                       f"not beyond real-run noise (within {ALARM_MARGIN:.0%} of two real runs)" if all(A[n]["overall_within"] for n in A)
                                       else "too few"),
                           "rule": "pass: either pipeline's catch beats its shuffle null at p < 0.025 and its gap's 97.5% interval is above 0; "
                                   "fail: both pipelines' 97.5% upper bounds below 0.10; otherwise too few (a void pipeline can't fail)"}
    out["reading"] = {
        "equivalence": ("within each pipeline's X" if both and all(P[n]["delta"]["upper_bound_95"] < P[n]["margin"]["X"] for n in P)
                        else "not shown within X" if both else "not read (a pipeline failed its planted checks or is not testable)"),
        "excess_per_pipeline": {n: (P[n]["reading"]["excess"] if P[n]["valid"] else P[n]["reading"]) for n in P}}
    # B-dev's frozen probes on the evened test runs: descriptive (frozen margin 0.065)
    f_, e_ = "fixed_vs_e1", "e2_vs_e1"
    test = tests["step A"]
    fz = json.loads(read(f"{R.OUT}/refit/frozen_run_scores.json"))["scores"]
    fs = {c: np.array([fz[str(i)] for i in np.flatnonzero(test[c])]) for c in (f_, e_)}
    fa = float(roc_auc_score(Ys["step A"][f_][:, 0], fs[f_]))
    fe = float(roc_auc_score(Ys["step A"][e_][:, 0], fs[e_]))
    out["frozen_delta"] = {"auroc_fixed_vs_e1": fa, "auroc_e2_vs_e1": fe, "delta": fa - fe,
                           "note": "descriptive; any frozen 'within' statement uses 0.065"}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "run_script_sha256": R.SCRIPT_SHA,
            "tpr_script_sha256": hashlib.sha256(TPR.read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"h3_small_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({"alarm_answer": out["alarm_answer"], "reading": out["reading"], "m2_reproduction": repro, "frozen_delta": out["frozen_delta"]} |
                     {n: {k: P[n][k] for k in ("alarm_answer", "margin", "delta", "reading", "plant1", "plant2")} for n in P}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
