"""09's separate-threshold LORO (25h amendment item 8 and fix D), exactly per the frozen manifest
experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest.json @ 9b8692ac1 (sha256 c79d816e...) and the LORO RULE
@ 8d59e2989's estimators. `cell` below is experiments/2026-10-15_loro_09/run_loro.py's estimator (@ 96b6acb10) with only the member
value lookup changed (each row's per-draw array comes from the manifest's npz field instead of the cross-recipe comparator axis).

Input: 2e's per-draw npz e2e_2026-10-15_130337 (sha256 b6e5fa00...), 183 T196 cells (S_old only), 10,000 shared draws + draw 0.
Per readout x registered per-action point (unit0.01 leads; unit0.001, unit0.05) x input (Delta_full joint = primary; its
fixed-threshold version; Delta_rr with honest C, joint and fixed-threshold, beside): (a), tau^2, (b) ranges for a new recipe's
MEAN Delta_full, leave-one-out, stress positions; members and per-member readability (reported, never used for membership).
Refits nothing. POST HOC (the 183 e2e values had been read by 2e), PRELIMINARY. One Modal CPU container.

  modal run --detach experiments/2026-10-15_loro_sep_09/run_loro_sep.py
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-loro_sep183_kill-2026-10-22"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
OUT_VOL = "probe-shift_aswin-surya_sub-loro_results_kill-2026-11-14"  # existing LORO results Volume; not created here
NPZ = "xfer/e2e_sim/e2e_2026-10-15_130337/e2e_2026-10-15_130337_perdraw.npz"
NPZ_SHA = "b6e5fa005deb429387c8607113afc483ce741655483cddbeff436105277772e7"
REC_COMMIT, REC_PATH = "46bd0dce3", "experiments/2026-10-15_e2e_sim_pipeline/results/e2e_2026-10-15_130337.json"
REC_SHA = "c74d8c33e721ad3589711f08e7fad106100f60757acbe4588533eba8fd6634f9"
MAN_COMMIT, MAN_PATH = "9b8692ac1", "experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest.json"
MAN_SHA = "c79d816eb858568540aaa8d793f9b85bb48daf9f5a567adae46c2c797d2f232d"
# draw-identity check (manifest: reported, not a stop): the cross-recipe npz that LORO @ 8d59e2989 used
XR_NPZ = "xfer/fprsweep_unit/crossrecipe_2026-10-15_103328/crossrecipe_2026-10-15_103328_perdraw.npz"
XR_SHA = "9a7b16be0398c7ed2ab4b59acddf46ff528484bc00b60b5cee179457fba0d32e"
MIN_DRAWS = 9500
INPUTS = ("primary", "fixed_threshold", "yardstick", "yardstick_fixed_threshold")
ARMS = ["sim_at_tau_sim_M", "run1_at_tau_B_M", "run2_at_tau_B_M", "run2_at_tau_C_M"]
CATCH_KEYS = ("full", "run1", "run2", "run2_at_C")  # record keys of the four arms, in ARMS order

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3", "scipy==1.14.1")
in_vol = modal.Volume.from_name(IN_VOL).read_only()
out_vol = modal.Volume.from_name(OUT_VOL)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def gate_diff(x, ref, what, np):
    """|x - ref| for a finite pair; NaN must meet null/NaN; infinities and finite-vs-missing mismatches stop the job."""
    x = float(x)
    if np.isinf(x):
        raise SystemExit(f"draw-0 gate: infinite npz value at {what}")
    if ref is None:
        if not np.isnan(x):
            raise SystemExit(f"draw-0 gate: record null vs finite npz value at {what}")
        return 0.0
    r = float(ref)
    if np.isinf(r):
        raise SystemExit(f"draw-0 gate: infinite record value at {what}")
    if np.isnan(r) or np.isnan(x):
        if not (np.isnan(r) and np.isnan(x)):
            raise SystemExit(f"draw-0 gate: NaN vs finite at {what}")
        return 0.0
    return abs(x - r)


def cell(rows, comparator, np, stats):
    """One readout x point x input cell, per LORO RULE.md @ 8d59e2989. rows: [{label, group, values (10001,) in points}]."""
    members, excluded = {}, []
    stress = []
    for r in rows:
        cname = comparator
        v = r["values"]
        n_bad = int((~np.isfinite(v[1:])).sum())
        ok = bool(np.isfinite(v[0])) and (v.size - 1 - n_bad) >= MIN_DRAWS
        rec = {"label": r["label"], "comparator": cname, "draw0": float(v[0]) if np.isfinite(v[0]) else None, "nonfinite_draws": n_bad}
        if not ok:
            excluded.append(rec | {"reason": "draw 0 non-finite" if not np.isfinite(v[0]) else f"only {v.size - 1 - n_bad} finite draws (< {MIN_DRAWS})"})
            continue
        if r["group"] == "STRESS":
            stress.append(rec | {"values": v})
        else:
            members.setdefault(r["group"], []).append(rec | {"values": v})
    groups = sorted(members)
    out = {
        "comparator": comparator,
        "groups": {g: [{"label": m["label"], "comparator": m["comparator"], "nonfinite_draws": m["nonfinite_draws"]} for m in members[g]] for g in groups},
        "excluded_members": excluded,
        "k": len(groups),
        "stress": [{"label": x["label"], "comparator": x["comparator"], "gap": x["draw0"], "nonfinite_draws": x["nonfinite_draws"], "position_vs_b_observed": "unavailable (no range)",
                    "position_vs_b_true": "unavailable (no range)"} for x in stress],
    }
    if not groups:
        return out | {"status": "unavailable: k = 0 < 3"}
    G = np.stack([np.mean(np.stack([m["values"] for m in members[g]]), axis=0) for g in groups], axis=1)  # (10001, k); NaN if any member NaN
    invalid = {g: int((~np.isfinite(G[1:, j])).sum()) for j, g in enumerate(groups)}
    Dmask = np.all(np.isfinite(G[1:]), axis=1)
    nD = int(Dmask.sum())
    out |= {"invalid_draws_by_group": invalid, "D": nD,
            "group_gaps_draw0": {g: (float(G[0, j]) if np.isfinite(G[0, j]) else None) for j, g in enumerate(groups)}}
    if len(groups) < 3:
        return out | {"status": f"unavailable: k = {len(groups)} < 3"}
    if not np.all(np.isfinite(G[0])):
        return out | {"status": "unavailable: a group's draw 0 is non-finite"}
    if nD < MIN_DRAWS:
        return out | {"status": f"unavailable: |D| = {nD} < {MIN_DRAWS}"}
    k = len(groups)
    g0 = G[0]
    Gd = G[1:][Dmask]
    Sigma = np.cov(Gd, rowvar=False, ddof=1)
    one = np.ones(k)
    m0 = float(g0.mean())
    m_d = Gd.mean(axis=1)
    abs0 = float(np.abs(g0).mean())
    abs_d = np.abs(Gd).mean(axis=1)
    s2 = float(np.var(g0, ddof=1))
    corr = float((np.trace(Sigma) - one @ Sigma @ one / k) / (k - 1))
    tau2_raw = s2 - corr
    tau2 = max(0.0, tau2_raw)
    Vm = tau2 / k + float(one @ Sigma @ one) / k**2
    v_new = float(np.median(np.diag(Sigma)))
    t = float(stats.t.ppf(0.975, k - 1))
    half_obs = t * (tau2 + Vm + v_new) ** 0.5
    half_true = t * (tau2 + Vm) ** 0.5
    loo = []
    for h in range(k):
        idx = [j for j in range(k) if j != h]
        n = k - 1
        mh = float(g0[idx].mean())
        Sh = Sigma[np.ix_(idx, idx)]
        s2h = float(np.var(g0[idx], ddof=1))
        corr_h = float((np.trace(Sh) - np.ones(n) @ Sh @ np.ones(n) / n) / (n - 1))
        tau2h_raw = s2h - corr_h
        tau2h = max(0.0, tau2h_raw)
        a = np.full(k, -1.0 / n)
        a[h] = 1.0
        samp = float(a @ Sigma @ a)
        if samp < -1e-9:
            raise SystemExit(f"a'Sigma a materially negative ({samp}) for held-out {groups[h]}")
        samp = max(samp, 0.0)
        var = tau2h * (1 + 1 / n) + samp
        th = float(stats.t.ppf(0.975, n - 1))
        lo, hi = mh - th * var**0.5, mh + th * var**0.5
        gh = float(g0[h])
        loo.append({
            "held_out": groups[h], "gap": gh, "m_minus_h": mh, "tau2_minus_h": tau2h, "tau2_minus_h_untruncated": tau2h_raw,
            "tau2_minus_h_truncated": tau2h_raw < 0, "sampling_term_aSa": samp, "t_df": n - 1, "t": th,
            "range": [lo, hi], "width": hi - lo, "residual": gh - mh,
            "position": "below" if gh < lo else ("above" if gh > hi else "inside"),
        })
    rng_obs = [m0 - half_obs, m0 + half_obs]
    rng_true = [m0 - half_true, m0 + half_true]

    def pos(x, r):
        return "below" if x < r[0] else ("above" if x > r[1] else "inside")

    return out | {
        "status": "ok",
        "a_mean_signed_gap": m0,
        "a_mean_signed_gap_ci95": [float(x) for x in np.percentile(m_d, [2.5, 97.5])],
        "a_mean_abs_gap": abs0,
        "a_mean_abs_gap_spread95_descriptive": [float(x) for x in np.percentile(abs_d, [2.5, 97.5])],
        "Sigma": Sigma.tolist(),
        "s2_draw0": s2,
        "noise_correction": corr,
        "tau2": tau2,
        "tau2_untruncated": tau2_raw,
        "tau2_truncated": tau2_raw < 0,
        "V_m": Vm,
        "v_new_working": v_new,
        "t_df": k - 1,
        "t": t,
        "b_range_new_recipe_mean_observed_independent_cohort": rng_obs,
        "b_range_new_recipe_mean_true_gap": rng_true,
        "loo": loo,
        "loo_inside": f"{sum(x['position'] == 'inside' for x in loo)} of {k}",
        "stress": [
            {"label": s["label"], "comparator": s["comparator"], "gap": s["draw0"], "nonfinite_draws": s["nonfinite_draws"],
             "position_vs_b_observed": pos(s["draw0"], rng_obs), "position_vs_b_true": pos(s["draw0"], rng_true)}
            for s in stress
        ],
    }


def readability(m, inp):
    """Per-member draw-0 readability, REPORTED ONLY (manifest: never changes membership). m = record rows[label].unit[pt].matched."""
    if inp in ("primary", "fixed_threshold"):
        return {"readable_contrast_e2e": bool(m["readable_contrast_e2e"]), "readable": {"real_B": bool(m["readable"]["real"]), "sim": bool(m["readable"]["sim"])},
                "n_units": {"real_B": m["n_units"]["real"], "sim": m["n_units"]["sim"]},
                "units_above_T_draw0": {"real_B": m["units_above_draw0"]["real"], "sim": m["units_above_draw0"]["sim"]}}
    return {"readable_contrast_yardstick": bool(m["readable_contrast_yardstick"]), "readable": {"real_B": bool(m["readable"]["real"]), "real_C": bool(m["readable"]["real_C"])},
            "n_units": {"real_B": m["n_units"]["real"], "real_C": "unavailable in this pinned input"},
            "units_above_T_draw0": {"real_B": m["units_above_draw0"]["real"], "real_C": "unavailable in this pinned input"}}


@app.function(image=image, volumes={"/in": in_vol, "/out": out_vol}, cpu=2, memory=8192, timeout=900, retries=0)
def run(record_bytes: bytes, manifest_bytes: bytes, git: str, stamp: str) -> dict:
    import numpy as np
    import scipy
    from scipy import stats

    if sha(record_bytes) != REC_SHA:
        raise SystemExit("record sha256 differs")
    if sha(manifest_bytes) != MAN_SHA:
        raise SystemExit("manifest sha256 differs")
    raw = (Path("/in") / NPZ).read_bytes()
    if sha(raw) != NPZ_SHA:
        raise SystemExit("npz sha256 differs")
    rec, man = json.loads(record_bytes), json.loads(manifest_bytes)
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    # gate: every array the manifest uses, with its shape
    used = man["input_provenance"]["npz"]["arrays_used"]
    A = {}
    for name, shp in used.items():
        if name not in z.files:
            raise SystemExit(f"npz lacks {name}")
        A[name] = z[name]
        if list(A[name].shape) != shp:
            raise SystemExit(f"{name} shape {A[name].shape} != manifest {shp}")
    # gate: row metadata, exact string equality in row_index order
    rows = man["rows"]
    if [r["row_index"] for r in rows] != list(range(len(rows))) or len(rows) != 20:
        raise SystemExit("manifest rows are not in row_index order 0..19")
    for fld, key in (("row_labels", "npz_row_label"), ("row_recipe", "npz_row_recipe"), ("row_family", "npz_row_family"),
                     ("row_readout", "npz_row_readout"), ("row_registered", "npz_row_registered")):
        if [str(x) for x in A[fld]] != [r[key] for r in rows]:
            raise SystemExit(f"npz {fld} differs from the manifest's {key}")
    if any(r["inventory_key"] != r["npz_row_label"] for r in rows):
        raise SystemExit("inventory_key != npz_row_label")
    # gate: point and arm labels
    pts = [str(x) for x in A["points"]]
    REG = [q["label"] for q in man["points"]["registered"]]
    if REG != ["unit0.01", "unit0.001", "unit0.05"] or any(pts.count(q) != 1 for q in REG):
        raise SystemExit(f"registered points {REG} not each exactly once in the npz points")
    if [str(x) for x in A["arms_e2e"]] != ARMS:
        raise SystemExit("arms_e2e differ")
    # gate: cohort
    coh, uni = [str(x) for x in A["cohort_tasks"]], [str(x) for x in A["universe_tasks"]]
    if len(coh) != 183 or len(set(coh)) != 183 or not set(coh) <= set(uni) or len(set(uni)) != 195:
        raise SystemExit("cohort_tasks / universe_tasks differ from the manifest")
    # gate: comparator fields are the manifest's
    fields = {inp: man["comparators"][inp]["npz_field"] for inp in INPUTS}
    if fields != {"primary": "e2e_matched_joint", "fixed_threshold": "e2e_matched_fixedtau", "yardstick": "yardstick_e2e_matched_joint",
                  "yardstick_fixed_threshold": "yardstick_e2e_matched_fixedtau"} or any(r["comparators"] != fields for r in rows):
        raise SystemExit("comparator fields differ from the manifest")
    # gate: draw 0 vs the record at every row x registered point (values compared by script, not printed)
    worst, n_checked, n_exact, n_read = 0.0, 0, 0, 0
    for r in rows:
        i, lab = r["row_index"], r["inventory_key"]
        for pt in REG:
            p = pts.index(pt)
            m = rec["rows"][lab]["unit"][pt]["matched"]
            worst = max(worst, gate_diff(A["e2e_matched_joint"][i, p, 0], m["gap"]["e2e_vs_real_mean"]["point"], f"{lab} {pt} e2e", np))
            worst = max(worst, gate_diff(A["yardstick_e2e_matched_joint"][i, p, 0], m["gap"]["yardstick_run2atC_minus_run1atB"]["point"], f"{lab} {pt} yardstick", np))
            n_checked += 2
            for j_, f_ in (("e2e_matched_joint", "e2e_matched_fixedtau"), ("yardstick_e2e_matched_joint", "yardstick_e2e_matched_fixedtau")):
                a_, b_ = float(A[j_][i, p, 0]), float(A[f_][i, p, 0])
                if not (a_ == b_ or (np.isnan(a_) and np.isnan(b_))):
                    raise SystemExit(f"draw-0 gate: {f_} != {j_} at {lab} {pt}")
                n_exact += 1
            for ai, ck in enumerate(CATCH_KEYS):
                worst = max(worst, gate_diff(A["catches_e2e"][i, p, ai, 0], m["catch"][ck]["point"], f"{lab} {pt} catch {ck}", np))
                n_checked += 1
            if bool(A["readable_e2e_both_sides"][i, p]) != bool(m["readable_contrast_e2e"]):
                raise SystemExit(f"readable_e2e_both_sides != readable_contrast_e2e at {lab} {pt}")
            n_read += 1
    if (n_checked, n_exact, n_read) != (20 * 3 * 6, 20 * 3 * 2, 20 * 3):
        raise SystemExit(f"draw-0 gate coverage {n_checked}/{n_exact}/{n_read}")
    if worst > 1e-9:
        raise SystemExit(f"draw-0 gate failed: max |diff| {worst}")
    # units (LORO RULE): catch fractions -> x100 to percentage points
    fin = A["catches_e2e"][..., 0][np.isfinite(A["catches_e2e"][..., 0])]
    if fin.size == 0 or fin.min() < -1e-12 or fin.max() > 100 + 1e-9:
        raise SystemExit("catch range outside [0, 100]")
    scale = 100.0 if fin.max() <= 1.0 + 1e-12 else 1.0
    # draw identity (reported, not a stop): this npz's multiplicities on the cohort columns vs the cross-recipe npz's
    xr_path = Path("/in") / XR_NPZ
    xr_raw = xr_path.read_bytes() if xr_path.exists() else None
    if xr_raw is None:
        draw_identity = {"status": "cross-recipe npz not found; check unavailable"}
    elif sha(xr_raw) != XR_SHA:
        draw_identity = {"status": "cross-recipe npz sha256 differs; check not run"}
    else:
        zx = np.load(io.BytesIO(xr_raw), allow_pickle=False)
        xc = [str(x) for x in zx["cohort_tasks"]]
        Wx = zx["task_multiplicities"]
        same_set = set(xc) == set(coh)
        ui = {t: j for j, t in enumerate(uni)}
        if same_set:
            We = A["task_multiplicities"][:, [ui[t] for t in xc]]
            draw_identity = {"status": "run", "same_cohort_set": True, "shapes": [list(We.shape), list(Wx.shape)],
                             "equal": bool(We.shape == Wx.shape and np.array_equal(We, Wx)),
                             "max_abs_diff": (float(np.abs(We - Wx).max()) if We.shape == Wx.shape else None)}
        else:
            draw_identity = {"status": "run", "same_cohort_set": False, "n_e2e_only": len(set(coh) - set(xc)), "n_crossrecipe_only": len(set(xc) - set(coh))}
    # cells: readout x registered point x input
    cells = {}
    for ro in ("R", "C"):
        rws = [r for r in rows if r["npz_row_readout"] == ro]
        for pt in REG:
            p = pts.index(pt)
            for inp in INPUTS:
                fld = fields[inp]
                members = [{"label": r["inventory_key"], "group": r["loro_group"], "values": A[fld][r["row_index"], p, :] * scale} for r in rws]
                c = cell(members, fld, np, stats)
                cells[f"{ro}|{pt}|{inp}"] = {
                    "readout": ro, "point": pt, "input": inp, "npz_field": fld, "name": man["comparators"][inp]["name"], "input_label": man["comparators"][inp]["label"],
                    "role": ("PRIMARY (lead)" if (inp == "primary" and pt == "unit0.01") else ("primary input, beside point" if inp == "primary" else "beside, labelled")),
                    "readability_by_member_reported_only": {r["inventory_key"]: readability(rec["rows"][r["inventory_key"]]["unit"][pt]["matched"], inp) for r in rws},
                } | c
    HK = ("status", "k", "D", "a_mean_signed_gap", "a_mean_signed_gap_ci95", "tau2", "tau2_untruncated", "t", "b_range_new_recipe_mean_observed_independent_cohort",
          "b_range_new_recipe_mean_true_gap", "loo_inside")
    headline = {key: {k_: cells[key].get(k_) for k_ in HK} for key in (f"{ro}|{pt}|{inp}" for pt in REG for ro in ("R", "C") for inp in ("primary", "yardstick"))}
    result = {
        "label": "Separate-threshold LORO (09; 25h amendment item 8, fix D), S_old 183 T196 cells. POST HOC (the 183 e2e values had been read by 2e), PRELIMINARY. "
                 "Target = a new recipe's MEAN Delta_full (per action, separate thresholds), not an individual probe",
        "scope": "S_old (T196, 183 attempted cells, both v6 halves) only; S_new, S_newmains and the two-origin pooled row are unavailable from this input",
        "provenance": "manifest @ 9b8692ac1 (frozen before any gap was inspected by 09); estimators = LORO RULE @ 8d59e2989 (cell() from run_loro.py @ 96b6acb10, member lookup only changed)",
        "git": git,
        "stamp": stamp,
        "inputs": {"npz": {"volume": IN_VOL, "path": NPZ, "sha256": NPZ_SHA},
                   "record": {"path": REC_PATH, "commit": REC_COMMIT, "sha256": REC_SHA},
                   "manifest": {"path": MAN_PATH, "commit": MAN_COMMIT, "sha256": MAN_SHA}},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "gates": {"npz_sha256": True, "record_sha256": True, "manifest_sha256": True, "shapes": True, "row_metadata_exact": True, "points_and_arms": True, "cohort": True,
                  "draw0_values_checked": n_checked, "draw0_fixed_equals_joint": n_exact, "draw0_readable_mask_checked": n_read, "draw0_max_abs_diff": worst},
        "units": {"scale_applied": scale, "rule": "catch fractions stored -> x100 to percentage points" if scale == 100.0 else "stored in percentage points"},
        "draw_identity_vs_crossrecipe_reported_not_stop": draw_identity,
        "membership": "LORO RULE finite-draw eligibility only; readability reported per member, never used for membership (manifest)",
        "wording": {"joint": "thresholds re-estimated per draw on each pool", "fixed": "FIXED-THRESHOLD INPUT: thresholds and fold moments fixed at draw 0; only attack tasks resampled",
                    "yardstick": "Delta_rr with honest C (25h item 3 names H_A, absent from this input)"},
        "useful_probe_sensitivity": "unavailable (as LORO @ 8d59e2989): no estimation-half per-draw arrays",
        "headline": headline,
        "cells": cells,
        "caveats": [
            "model: independent, approximately Gaussian recipe effects with common mean mu and variance tau^2, plus mean-zero sampling errors with covariance Sigma independent of those effects",
            "complete-case handling of draws (D) does not establish that missingness is ignorable",
            "mean absolute gap keeps sampling-noise bias, especially near 0; its spread is descriptive, not a calibrated interval",
            "tau^2 is truncated at 0, which biases it upward; the untruncated value and the flag are saved",
            "prediction target: a new recipe's mean Delta_full (members averaged by the same rule); within-recipe variation is not modelled; ranges do not cover individual probes",
            "nominal pointwise 95% model-based ranges, t_{n-1}; coverage approximate and not guaranteed conservative; tau^2 from at most 6 recipes is very imprecise",
            "the recipes are not a random sample of future probes; one cohort (183 S_old cells), one reader (Qwen3.5-27B), one simulator",
            "joint inputs: thresholds re-estimated per draw on each pool; fixed-threshold inputs condition on draw 0's thresholds; (a) is task-sampling uncertainty conditional on the included recipes",
            "the leave-one-out check evaluates same-cohort prediction and does not validate the independent-cohort range; j-of-k is descriptive",
            "stress-test positions are descriptive; under separate thresholds c_asis is not a positive control (25h item 12)",
            "own-real was trained on run 1, so the real-run mean includes that launch (LORO @ 8d59e2989 used sim - run2 for own-real); own-real has 4 members here, not 10",
            "matched design: calibration pools restricted to M (148 tasks); Delta_rr uses honest C as the alternate pool, not 25h's H_A",
            "not like-for-like with LORO @ 8d59e2989's results (different estimand: separate thresholds per action vs real-calibrated per run)",
            "(b) assumes evaluation on an independently sampled cohort of the same size and design; v_new (median over groups of Sigma_gg) is a working value for a new recipe mean's "
            "sampling variance, not an estimate for an individual probe; for evaluation on the same cohort the prediction-error variance is tau^2 + V_m + v_new - 2 c_new, and c_new = 0 is not assumed there",
            "the range for the new recipe mean's TRUE gap omits v_new",
            "coverage wording: nominal pointwise 95% ranges with t_{n-1}, n = k for (b) and k - 1 for leave-one-out; at k = 3 leave-one-out has one degree of freedom (t about 12.706); "
            "small k gives a large multiplier but guarantees neither a wide range nor adequate coverage",
            "held-out indicators are dependent; no binomial calibration test or simultaneous coverage claim is made; with about six groups coverage counts carry very little evidence about calibration",
            "the intervals condition on the fitted probes",
        ],
        "manifest_caveats_verbatim": man["caveats"],
    }
    dest = Path("/out") / "loro_sep_09" / f"loro_sep_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as fh:
        fh.write(json.dumps(result, indent=1))
    out_vol.commit()
    return result


@app.local_entrypoint()
def main():
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    record_bytes = g("show", f"{REC_COMMIT}:{REC_PATH}")
    manifest_bytes = g("show", f"{MAN_COMMIT}:{MAN_PATH}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(record_bytes, manifest_bytes, head, stamp)
    dest = HERE / "results" / f"loro_sep_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps({"gates": res["gates"], "units": res["units"], "draw_identity": res["draw_identity_vs_crossrecipe_reported_not_stop"]}, indent=1))
    print("wrote", dest.name)
