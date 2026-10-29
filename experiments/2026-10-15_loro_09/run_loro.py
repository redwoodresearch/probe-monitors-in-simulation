"""09's leave-one-recipe-out (LORO) post-processor (RULE.md in this folder: grouping and estimators fixed and Codex-reviewed before
09's disclosed 10:41 exposure to some table values; table identifier and mapping committed after it @ 8d59e2989).
Reads 2e's per-draw npz (183 T196 tasks, 10,000 shared draws) from its Volume, gates its sha256 and draw 0 against the
cross-recipe record, averages each training-world group's fixed eligible members draw by draw, and per readout x run-level
point x comparator reports (a) the typical error across the included recipes, tau^2, (b) the prediction range for a new
recipe's mean gap (independent cohort, and its true gap), the leave-one-recipe-out check and the stress-test positions.
Refits nothing. One Modal CPU container. PRELIMINARY.

  $MODAL run --detach experiments/2026-10-15_loro_09/run_loro.py [--cohort 183|455]

--cohort 455 (RUN_455.md): the same method on 2e's 455-task npz, draw 0 gated against run455's record (blocks.all).
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
APP = "probe-shift_aswin-surya_sub-loro_postproc_kill-2026-11-14"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
OUT_VOL = "probe-shift_aswin-surya_sub-loro_results_kill-2026-11-14"
COHORTS = {
    "183": {"npz": "xfer/fprsweep_unit/crossrecipe_2026-10-15_103328/crossrecipe_2026-10-15_103328_perdraw.npz",
            "npz_sha": "9a7b16be0398c7ed2ab4b59acddf46ff528484bc00b60b5cee179457fba0d32e",
            "rec_path": "experiments/2026-10-14_fpr_sweep_unit/results/crossrecipe_2026-10-15_103328.json", "rec_commit": "bfe8867e6",
            "rec_sha": "84dac40da0958aa5bdf7383ec339a0736f73aaa2cb299d4011a9911b03455ec4", "gate": "crossrecipe", "n_tasks": 183},
    "455": {"npz": "xfer/fpr507prep/perdraw455_2026-10-15_112352/perdraw455_2026-10-15_112352_perdraw.npz",
            "npz_sha": "fef54633f19de9a1980051dde665b96788d9c6fffd1d0344ab3e90133b0e2c1d",
            "rec_path": "experiments/2026-10-15_fpr_sweep_507_prep/results/common_run455_2026-10-15_101451.json", "rec_commit": "492b31c9e",
            "rec_sha": "c7b348575e5e1a2c2a4c97e192659d3a432d7c99721b0ab996c92e36f306d810", "gate": "run455", "n_tasks": 455},
}
MIN_DRAWS = 9500
POINTS = ("run0.01", "run0.02", "run0.05", "run0.1")
COMPARATORS = ("registered", "sim_minus_run1", "sim_minus_run2", "sim_minus_real_mean")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3", "scipy==1.14.1")
in_vol = modal.Volume.from_name(IN_VOL).read_only()
out_vol = modal.Volume.from_name(OUT_VOL, create_if_missing=True)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def gate_diff(x, ref, what, np):
    """|x - ref| for a finite pair; NaN must meet null/NaN; infinities and finite-vs-missing mismatches stop the job."""
    x = float(x)
    if np.isinf(x):
        raise SystemExit(f"draw-0 gate: infinite npz value at {what}")
    if ref is None:
        if not np.isnan(x):
            raise SystemExit(f"draw-0 gate: record null vs npz {x} at {what}")
        return 0.0
    r = float(ref)
    if np.isinf(r):
        raise SystemExit(f"draw-0 gate: infinite record value at {what}")
    if np.isnan(r) or np.isnan(x):
        if not (np.isnan(r) and np.isnan(x)):
            raise SystemExit(f"draw-0 gate: NaN vs finite at {what}")
        return 0.0
    return abs(x - r)


def cell(gaps, scale, rows, comp_index, p, comparator, np, stats):
    """One readout x point x comparator cell, per RULE.md. gaps: (rows, points, comparators, 10001)."""
    members, excluded = {}, []
    stress = []
    for r in rows:
        cname = r["registered_comparator"] if comparator == "registered" else comparator
        v = gaps[r["i"], p, comp_index[cname], :] * scale
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


def draw0_reference(rec, kind, lab, pt):
    """The table's draw-0 entry for one row x point as {"gap": {cn: point}, "catch": {an: point}}."""
    if kind == "crossrecipe":
        e = rec["rows"][lab]["points"][pt]
        return {"gap": {k: v["point"] for k, v in e["gap"].items()}, "catch": {k: v["point"] for k, v in e["catch"].items()}}
    pr = rec["probes"][lab]
    if pt.startswith("run"):
        hits = [x for x in pr["run"]["points"] if abs(x["alpha"] - float(pt[3:])) < 1e-12]
    else:
        hits = [x for x in pr["unit"] if abs(x["fpr_target"] / float(pt[4:]) - 1) < 1e-2]  # npz names round to 3 significant digits; grid ratio 10^(1/8)
    if len(hits) != 1:
        raise SystemExit(f"run455 gate: {len(hits)} matching points for {lab} {pt}")
    b = hits[0]["blocks"]["all" if pt.startswith("run") else "combined"]  # run455 names the whole-cohort block "all" (run) and "combined" (unit)
    return {"gap": {k: v["point"] for k, v in b["diff"].items()}, "catch": {k: v["point"] for k, v in b["tpr"].items()}}


@app.function(image=image, volumes={"/in": in_vol, "/out": out_vol}, cpu=2, memory=8192, timeout=900, retries=0)
def run(record_bytes: bytes, mapping_bytes: bytes, git: str, stamp: str, cohort: str = "183") -> dict:
    import numpy as np
    import scipy
    from scipy import stats

    C = COHORTS[cohort]
    NPZ, NPZ_SHA, REC_PATH, REC_COMMIT, REC_SHA = C["npz"], C["npz_sha"], C["rec_path"], C["rec_commit"], C["rec_sha"]
    if sha(record_bytes) != REC_SHA:
        raise SystemExit("record sha256 differs")
    raw = (Path("/in") / NPZ).read_bytes()
    if sha(raw) != NPZ_SHA:
        raise SystemExit("npz sha256 differs")
    rec, mp = json.loads(record_bytes), json.loads(mapping_bytes)
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    gaps, catches = z["gaps"], z["catches"]
    labels = [str(x) for x in z["row_labels"]]
    points, comps, arms = [str(x) for x in z["points"]], [str(x) for x in z["comparators"]], [str(x) for x in z["arms"]]
    # metadata gate: the npz rows are the mapping's rows, same recipe/family/readout
    by_label = {r["label"]: r for r in mp["rows"]}
    if len(set(labels)) != len(labels) or len(by_label) != len(mp["rows"]) or sorted(labels) != sorted(by_label):
        raise SystemExit("npz row labels differ from the mapping (or are not unique)")
    if gaps.shape != (32, 7, 4, 10001) or catches.shape != (32, 7, 4, 10001) or z["task_multiplicities"].shape != (10001, C["n_tasks"]):
        raise SystemExit(f"array shapes differ from the spec: {gaps.shape} {catches.shape} {z['task_multiplicities'].shape}")
    row_registered = sorted({str(x) for x in z["row_registered"]})
    for i, lab in enumerate(labels):
        r = by_label[lab]
        if (str(z["row_recipe"][i]), str(z["row_family"][i]), str(z["row_readout"][i])) != (r["recipe"], r["family"], r["readout"]):
            raise SystemExit(f"row metadata differs for {lab}")
        reg = str(z["row_registered"][i])
        if reg not in comps or reg != r["registered_comparator"]:
            raise SystemExit(f"row_registered {reg} differs from the mapping's comparator for {lab}")
        r["i"] = i
    # draw-0 gate: every gap and catch equals the record's table point (<= 1e-9; NaN must match NaN)
    worst, n_checked = 0.0, 0
    for lab in labels:
        i = by_label[lab]["i"]
        for p, pt in enumerate(points):
            entry = draw0_reference(rec, C["gate"], lab, pt)
            for c, cn in enumerate(comps):
                worst = max(worst, gate_diff(gaps[i, p, c, 0], entry["gap"][cn], f"{lab} {pt} gap {cn}", np))
                n_checked += 1
            for a, an in enumerate(arms):
                worst = max(worst, gate_diff(catches[i, p, a, 0], entry["catch"][an], f"{lab} {pt} catch {an}", np))
                n_checked += 1
    if n_checked != len(labels) * len(points) * (len(comps) + len(arms)):
        raise SystemExit(f"draw-0 gate covered {n_checked} values, not every row x point x (comparator + arm)")
    if worst > 1e-9:
        raise SystemExit(f"draw-0 gate failed: max |diff| {worst}")
    fin = catches[..., 0][np.isfinite(catches[..., 0])]
    if fin.size == 0 or fin.min() < -1e-12 or fin.max() > 100 + 1e-9:
        raise SystemExit("catch range outside [0, 100]")
    scale = 100.0 if fin.max() <= 1.0 + 1e-12 else 1.0
    comp_index = {cn: c for c, cn in enumerate(comps)}
    cells = {}
    for readout in ("R", "C"):
        rows = [r for r in mp["rows"] if r["readout"] == readout]
        for pt in POINTS:
            p = points.index(pt)
            for comparator in COMPARATORS:
                cells[f"{readout}|{pt}|{comparator}"] = cell(gaps, scale, rows, comp_index, p, comparator, np, stats)
    # descriptive: each row's draw-0 real-mean and sim catch at the run points (percentage points)
    catch0 = {
        lab: {pt: {an: (float(catches[by_label[lab]["i"], points.index(pt), arms.index(an), 0]) * scale
                        if np.isfinite(catches[by_label[lab]["i"], points.index(pt), arms.index(an), 0]) else None)
                   for an in ("real_mean", "sim")} for pt in POINTS}
        for lab in labels
    }
    headline = {
        key: {k_: cells[key].get(k_) for k_ in ("status", "k", "D", "a_mean_signed_gap", "a_mean_signed_gap_ci95", "tau2", "tau2_untruncated", "t",
                                                "b_range_new_recipe_mean_observed_independent_cohort", "b_range_new_recipe_mean_true_gap", "loo_inside")}
        for key in ("R|run0.05|registered", "R|run0.01|registered", "C|run0.05|registered", "C|run0.01|registered")
    }
    result = {
        "label": f"LORO post-processor (09), PRELIMINARY; cohort {cohort} tasks; target = a new recipe's MEAN gap, not an individual probe",
        "cohort": cohort,
        "task_origin_counts": ({str(k): int(v) for k, v in zip(*np.unique(z["task_origin"], return_counts=True))} if "task_origin" in z.files else None),
        "per_origin_sensitivity": ("unavailable: the npz stores whole-cohort gaps per draw; S_old-only and S_new-only per-draw gaps would need per-origin arrays (or per-task hits) from the producer"
                                   if cohort == "455" else None),
        "mapping_equals_183_groups_row_by_row": True,
        "provenance": "grouping rule and every estimator drafted 09:38 PT and Codex-reviewed (r1 09:43, r2 10:24; draft last changed 10:23:49) BEFORE 09 saw some table values at about 10:41 PT (disclosed verbatim in RULE.md); the table identifier and the row-to-group-to-comparator mapping (from row metadata only) were committed afterwards @ 8d59e2989; this post-processor was written and reviewed after that",
        "git": git,
        "stamp": stamp,
        "inputs": {"npz": {"volume": IN_VOL, "path": NPZ, "sha256": NPZ_SHA, "shape_gaps": list(gaps.shape)},
                   "record": {"path": REC_PATH, "commit": REC_COMMIT, "sha256": REC_SHA},
                   "mapping_sha256": sha(mapping_bytes)},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "gates": {"npz_sha256": True, "record_sha256": True, "row_metadata": True, "row_registered_values": row_registered, "draw0_values_checked": n_checked, "draw0_max_abs_diff": worst},
        "units": {"scale_applied": scale, "rule": "catch fractions stored -> x100 to percentage points" if scale == 100.0 else "stored in percentage points"},
        "useful_probe_sensitivity": "unavailable: the npz holds only the whole-cohort draws, not the v6 block's estimation-half draws (RULE.md)",
        "headline": headline,
        "cells": cells,
        "catch_draw0_descriptive": catch0,
        "caveats": [
            "model: independent, approximately Gaussian recipe effects with common mean mu and variance tau^2, plus mean-zero sampling errors with covariance Sigma independent of those effects",
            "complete-case handling of draws (D) does not establish that missingness is ignorable",
            "mean absolute gap keeps sampling-noise bias, especially near 0; its spread is descriptive, not a calibrated interval",
            "tau^2 is truncated at 0, which biases it upward; the untruncated value and the flag are saved",
            "prediction target: a new recipe's mean gap (members averaged by the same rule); within-recipe variation is not modelled; ranges do not cover individual probes",
            "nominal pointwise 95% model-based ranges, t_{n-1}; coverage approximate and not guaranteed conservative; tau^2 from about 6 recipes is very imprecise",
            f"the recipes are not a random sample of future probes; one cohort ({cohort} tasks), one reader (Qwen3.5-27B), one simulator",
            "intervals condition on the fitted probes and on the calibration; (a) is task-sampling uncertainty conditional on the included recipes",
            "the leave-one-out check evaluates same-cohort prediction and does not validate the independent-cohort range; j-of-k is descriptive",
            "stress-test positions are descriptive; 'expected outside' is a hypothesis, not a guaranteed positive control",
        ],
    }
    dest = Path("/out") / "loro_09" / f"loro{'' if cohort == '183' else '455'}_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as fh:
        fh.write(json.dumps(result, indent=1))
    out_vol.commit()
    return result


@app.local_entrypoint()
def main(cohort: str = "183"):
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    C = COHORTS[cohort]
    record_bytes = g("show", f"{C['rec_commit']}:{C['rec_path']}")
    mapping_bytes = (HERE / "config" / "loro_mapping.json").read_bytes()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(record_bytes, mapping_bytes, head, stamp, cohort)
    dest = HERE / "results" / f"loro{'' if cohort == '183' else '455'}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps({"gates": res["gates"], "units": res["units"], "headline": res["headline"]}, indent=1))
    print("wrote", dest.name)
