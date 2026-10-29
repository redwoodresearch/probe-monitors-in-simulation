"""POST HOC, price-only (SIZING_RULE.md): how much would more simulated honest calibration runs narrow the 5% planted
check of edited - real e2 at matched false alarms (../2026-10-12_q3_edit_matched_fa, full cohort)? $0, local, stored
run scores only; keys and numbers only.

Reuses analyze_matched_fa.py's inputs, gates, helpers and stratified draws. Gate: with no thinning it reproduces the
as-run 5% main point, interval and plant. Then, in draws 1..2000 only, a thinned world's calibration weight for task t
(bootstrap count M) becomes Binomial(M, f) (common random numbers across f), for f in {0.5, 0.625, 0.75, 0.875, 1} and
three series (sim, real, both). Fits hu_p(f)^2 = A + B/f, where hu_p = hi_plant - planted point; readable iff hu_p < s.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q3_e2_calsize/size_calibration.py \
      --base-scores <scores_L63.npz> --fix-runs <fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl> \
      --edit-scores <edithonest_scores_2026-10-12_022927.npz> --asrun <matched_fa_POSTHOC_2026-10-12_025014.json>
"""

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
spec = importlib.util.spec_from_file_location("amf", REPO / "experiments/2026-10-12_q3_edit_matched_fa/analyze_matched_fa.py")
spec.loader.exec_module(AMF := importlib.util.module_from_spec(spec))
AEH = AMF.AEH
ASRUN_SHA = "bcb2ab17a2cb5a3f"  # prefix of the as-run full-cohort result's sha256 (SIZING_RULE.md)
FS = (0.5, 0.625, 0.75, 0.875, 1.0)
SERIES = {"sim": (("edited",), 20261020), "real": (("real",), 20261021), "both": (("edited", "real"), 20261022)}
A_ = "0.05"


def main():
    ap = argparse.ArgumentParser()
    for a in ("--base-scores", "--fix-runs", "--edit-scores", "--asrun"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout.strip():
        raise SystemExit("refusing to run from a dirty tree")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    scoring = json.loads(AMF.EH_SCORING.read_text())
    f939_all = json.loads(AMF.FIX939.read_text())
    gates = {"base_scores_sha": AMF.sha(args.base_scores) == AMF.SHA["base"], "fix_runs_sha": AMF.sha(args.fix_runs) == AMF.SHA["fix"],
             "edit_scores_sha": AMF.sha(args.edit_scores) == scoring["scores_sha256"],
             "base_runs_json_sha": AMF.sha(AMF.BASE_RUNS) == f939_all["inputs"]["scores_dir"]["runs_json_sha256"],
             "honest_runs_json_sha": AMF.sha(AEH.RUNS) == scoring["runs_json_sha256"],
             "asrun_result_sha": AMF.sha(args.asrun).startswith(ASRUN_SHA)}
    if not all(gates.values()):
        raise SystemExit(f"input hashes differ: {gates}")
    # attack cells and honest worlds, exactly as analyze_matched_fa.py
    runs = json.loads(AMF.BASE_RUNS.read_text())
    arm, fold, rtask = (np.array([r[k] for r in runs]) for k in ("arm", "fold", "main_task"))
    Z = np.load(args.base_scores)
    nz, oz = Z[f"{AMF.RO}__mean__task_nested"], Z[f"{AMF.RO}__mean__task_outer"]
    valid = lambda i: np.isfinite(oz[i]) and np.isfinite(np.delete(nz[fold[i], :, i], fold[i])).all()
    arm_idx = {a: {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == a} for a in (AMF.SIM, AMF.E1, AMF.E2)}
    cells = sorted(c for c in set(arm_idx[AMF.SIM]) & set(arm_idx[AMF.E1]) & set(arm_idx[AMF.E2]) if all(valid(arm_idx[a][c]) for a in arm_idx))
    ii = {n: np.array([arm_idx[a][c] for c in cells]) for n, a in (("sim", AMF.SIM), ("e2", AMF.E2))}
    S = {n: np.stack([nz[fold[i], :, i] for i in v]) for n, v in ii.items()}
    F = {n: fold[v] for n, v in ii.items()}
    atask = rtask[ii["sim"]]
    fix = {}
    for line in Path(args.fix_runs).read_text().splitlines():
        r = json.loads(line)
        if r["label"] == "FIX":
            fix[r["key"]] = r
    S["fix"], F["fix"] = S["sim"].copy(), F["sim"]
    for j, i in enumerate(ii["sim"]):
        r = fix.get(runs[i]["key"])
        if r is not None:
            row = np.full(5, np.nan)
            for v_, s in r["nested_scores"][AMF.RO].items():
                row[int(v_)] = np.nan if s is None else s
            S["fix"][j] = row
    hruns = json.loads(AEH.RUNS.read_text())
    ez = np.load(args.edit_scores)
    rows = {k: len(hruns) + j for j, k in enumerate(scoring["edited_keys"])}
    o, ne, harm, meta, _ = AEH.assemble(hruns, ez["L63__outer"], ez["L63__nested"], rows)
    hfold, htask = meta["fold"].astype(int), meta["main_task"]
    M, _, _ = AEH.masks(o, harm, meta)
    H = {"real": AMF.cal_sets(ne, np.flatnonzero(M["i"]), hfold, htask), "edited": AMF.cal_sets(ne, np.flatnonzero(M["iii"]), hfold, htask)}
    cal_tasks = {w: sorted(set(htask[M[g]])) for w, g in (("real", "i"), ("edited", "iii"))}
    # the as-run stratified draws
    A, Hs = set(atask), set(htask[M["ii"]])
    strata = [sorted(A & Hs), sorted(A - Hs), sorted(Hs - A)]
    rng = np.random.default_rng(AMF.SEED)
    draws = [{t: 1.0 for st in strata for t in st}]
    for _ in range(AMF.B):
        cnt = {}
        for st in strata:
            for k in rng.integers(0, len(st), len(st)):
                cnt[st[k]] = cnt.get(st[k], 0.0) + 1.0
        draws.append(cnt)
    aw = np.array([[d.get(t, 0.0) for t in atask] for d in draws])
    al = float(A_)
    # the as-run plant: elig at draw-0 edited thresholds, first choice from PLANT_SEED (5% is the first rate)
    T0e = AMF.thresholds(H["edited"], al, draws[0])
    elig = np.flatnonzero(AMF.catch(S["fix"], F["fix"], T0e) == 1.0)
    X = float(np.ceil(max(abs(c) for c in f939_all["readouts"][AMF.RO][A_]["real_e2_minus_e1 (e1 was in training)"]["ci95"]) * 200) / 200)
    k = int(round(X * len(cells)))
    pick = np.sort(np.random.default_rng(AMF.PLANT_SEED).choice(elig, k, replace=False))
    Sp = S["fix"].copy()
    Sp[pick] = np.where(np.isnan(Sp[pick]), np.nan, -1e6)
    wmean = lambda cv: (aw * cv).sum(1) / aw.sum(1)

    def evaluate(wdraws):
        """wdraws[w][b]: calibration task weights of world w in draw b -> main and planted draws, thresholds"""
        Te = [AMF.thresholds(H["edited"], al, wdraws["edited"][b]) for b in range(AMF.B + 1)]
        Tr = [AMF.thresholds(H["real"], al, wdraws["real"][b]) for b in range(AMF.B + 1)]
        e2 = wmean(np.array([AMF.catch(S["e2"], F["e2"], Tr[b]) for b in range(AMF.B + 1)]))
        d = wmean(np.array([AMF.catch(S["fix"], F["fix"], Te[b]) for b in range(AMF.B + 1)])) - e2
        dp = wmean(np.array([AMF.catch(Sp, F["fix"], Te[b]) for b in range(AMF.B + 1)])) - e2
        return d, dp, {"edited": np.array(Te), "real": np.array(Tr)}

    def summary(d, dp, sl=slice(1, None)):
        lo, hi = AMF.ci(d[sl]); plo, phi = AMF.ci(dp[sl])
        shift = float(dp[0] - d[0])
        return {"main": float(d[0]), "main_ci95": [lo, hi], "planted_point": float(dp[0]), "planted_ci95": [plo, phi],
                "shift": shift, "hu_p": float(phi - dp[0]),
                "detected": bool(-1.5 * X <= shift <= -0.5 * X and not (plo <= d[0] <= phi))}

    base = {w: draws for w in ("edited", "real")}
    d1, dp1, Tb = evaluate(base)
    r0 = json.loads(Path(args.asrun).read_text())["results"][A_]
    s1 = summary(d1, dp1)
    pl = r0["plant"]
    gate_vals = {"main": [s1["main"], r0["edited_minus_real_e2 (main)"]["value"]],
                 "main_lo": [s1["main_ci95"][0], r0["edited_minus_real_e2 (main)"]["ci95"][0]],
                 "main_hi": [s1["main_ci95"][1], r0["edited_minus_real_e2 (main)"]["ci95"][1]],
                 "plant_shift": [s1["shift"], pl["shift"]], "planted_lo": [s1["planted_ci95"][0], pl["planted_ci95"][0]],
                 "planted_hi": [s1["planted_ci95"][1], pl["planted_ci95"][1]], "plant_k": [k, pl["k"]], "detected": [s1["detected"], pl["detected"]]}
    gates["asrun_reproduced"] = all(abs(float(x) - float(y)) <= 1e-12 for x, y in gate_vals.values())
    out = {"label": "POST HOC, price-only sizing (SIZING_RULE.md); from raw data", "git": head, "rule": "SIZING_RULE.md",
           "gates": gates | {"all": all(gates.values())}, "gate_values": gate_vals, "X": X, "k": k, "cells": len(cells),
           "s_points": -s1["shift"] * 100,
           "calibration_runs_per_fold": {w: [int(np.sum(hfold[M[g]] == v)) for v in range(5)] for w, g in (("real", "i"), ("edited", "iii"))}}
    if not out["gates"]["all"]:
        _write(out)
        raise SystemExit(f"a gate FAILED: {gates}")
    # binomial thinning with common random numbers across f: per series, world, draw, task, sorted uniforms
    fold_of = {w: {t: int(hfold[M[g]][list(htask[M[g]]).index(t)]) for t in cal_tasks[w]} for w, g in (("real", "i"), ("edited", "iii"))}
    maxcal = {w: {fv: float(np.max(s)) for fv, (s, _) in H[w].items()} for w in H}
    res = {}
    for name, (worlds, seed) in SERIES.items():
        g = np.random.default_rng(seed)
        U = {w: [None] + [{t: g.random(int(c)) for t, c in sorted(draws[b].items()) if t in fold_of[w]} for b in range(1, AMF.B + 1)]
             for w in worlds}
        per_f = {}
        for f in FS:
            wd = {w: draws for w in ("edited", "real")}
            for w in worlds:
                wd[w] = [draws[0]] + [{t: float((u < f).sum()) for t, u in U[w][b].items()} for b in range(1, AMF.B + 1)]
            d, dp, T = evaluate(wd)
            sm = summary(d, dp)
            blocks = [summary(d, dp, slice(1 + 500 * j, 1 + 500 * (j + 1)))["hu_p"] for j in range(4)]
            diag = {}
            for w in worlds:
                ret = np.array([[sum(1 for t, c in wd[w][b].items() if c > 0 and fold_of[w][t] == v) for v in range(5)] for b in range(1, AMF.B + 1)])
                wt = np.array([[sum(c for t, c in wd[w][b].items() if fold_of[w][t] == v) for v in range(5)] for b in range(1, AMF.B + 1)])
                Tw = T[w][1:]
                at_max = np.mean([[Tw[b][fv] == maxcal[w][fv] for fv in maxcal[w]] for b in range(AMF.B)])
                tv = np.nanvar(Tw, axis=0)  # per nested model (f, v); NaN on the diagonal
                diag[w] = {"retained_unique_tasks_per_fold_mean": ret.mean(0).tolist(), "total_weight_per_fold_mean": wt.mean(0).tolist(),
                           "threshold_var_mean_over_models": float(np.nanmean(tv)),
                           "threshold_var_per_model": [float(tv[a, b]) for a in range(5) for b in range(5) if a != b],
                           "threshold_mean_shift_vs_draw0": float(np.nanmean(np.nanmean(Tw, axis=0) - T[w][0])),
                           "share_draws_threshold_at_fold_max": float(at_max)}
            per_f[str(f)] = sm | {"hu_p_blocks": blocks, "diagnostics": diag}
        res[name] = {"per_f": per_f} | fit(per_f, worlds, s1)
    out["series"] = res
    out["prediction"] = "SIZING_RULE.md: reducible fraction sim 0.35+-0.2, real 0.35+-0.2, both 0.7+-0.2; sim-only not readable (p~0.8); both not readable (p~0.65); variance ratio 1.8-2.3"
    _write(out)


def fit(per_f, worlds, s1):
    s2 = (-s1["shift"]) ** 2
    f = np.array(FS)
    y = np.array([per_f[str(x)]["hu_p"] ** 2 for x in FS])
    yb = np.array([[per_f[str(x)]["hu_p_blocks"][j] ** 2 for x in FS] for j in range(4)])
    X_ = np.stack([np.ones_like(f), 1 / f], 1)

    def ab(yy, keep=slice(None)):
        return np.linalg.lstsq(X_[keep], yy[keep], rcond=None)[0]

    def crossing(a, b):
        return float(b / (s2 - a)) if (b > 0 and a < s2) else None

    a, b = ab(y)
    blk = np.array([ab(yb[j]) for j in range(4)])
    a_sd, b_sd = blk.std(0, ddof=1) / np.sqrt(4)  # SD of the full-data estimate, from 4 blocks of 500 draws
    resid = (X_ @ np.array([a, b]) - y) / y
    a2, b2 = ab(y, slice(1, None))  # without f = 0.5
    m, m2 = crossing(a, b), crossing(a2, b2)
    # SIZING_RULE.md diagnostic 2: the mean over the 20 nested models of each model's variance ratio (amendment 1)
    ratio = {w: float(np.mean(np.array(per_f["0.5"]["diagnostics"][w]["threshold_var_per_model"]) /
                              np.array(per_f["1.0"]["diagnostics"][w]["threshold_var_per_model"]))) for w in worlds}
    ratio_of_means = {w: per_f["0.5"]["diagnostics"][w]["threshold_var_mean_over_models"] / per_f["1.0"]["diagnostics"][w]["threshold_var_mean_over_models"] for w in worlds}
    checks = {"A_nonneg_B_pos": bool(a > -2 * a_sd and b > 2 * b_sd), "threshold_var_ratio_in_1.5_2.7": all(1.5 <= v <= 2.7 for v in ratio.values()),
              "max_rel_residual_le_0.15": bool(np.max(np.abs(resid)) <= 0.15),
              "crossing_stable_without_f0.5": bool((m is None and m2 is None) or (m is not None and m2 is not None and 0.5 <= m2 / m <= 2))}
    rt = lambda v: float(np.sqrt(v)) * 100 if v >= 0 else None  # None: A + B/m < 0, no valid width (amendment 1)
    pred = {str(mm): rt(a + b / mm) for mm in (1.5, 2, 3, 4, 8, 16)} | {"inf": rt(a)}
    mb = [crossing(*blk[j]) for j in range(4)]
    return {"fit_hu_p2_points2": {"A": a * 1e4, "B": b * 1e4, "A_sd": a_sd * 1e4, "B_sd": b_sd * 1e4},
            "reducible_fraction_at_f1": float(b / (a + b)), "rel_residuals": resid.tolist(), "fit_without_f0.5": {"A": a2 * 1e4, "B": b2 * 1e4},
            "threshold_var_ratio_f0.5_over_f1": ratio, "threshold_var_ratio_of_means_f0.5_over_f1": ratio_of_means,
            "checks": checks | {"all": all(checks.values())},
            "hu_p_points_predicted_at_m": pred, "s_points": np.sqrt(s2) * 100,
            "m_star": m, "m_star_without_f0.5": m2, "m_star_blocks": mb,
            "reading": ("no reliable extrapolation" if not all(checks.values()) else
                        "no finite crossing (A >= s^2)" if m is None else f"m* = {m:.2f}")}


def _write(out):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"calsize_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "series"}, indent=1, default=float))
    for n, r in out.get("series", {}).items():
        print(n, json.dumps({k: r[k] for k in r if k != "per_f"}, default=float))
    print(path.name)


if __name__ == "__main__":
    main()
