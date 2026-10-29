"""Length-matched separate-threshold LORO (POST HOC, PRELIMINARY) on the v3 (Q2-FINAL pooled v2) inputs, per AMENDMENT 4 @ 6187764ad
(experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_amend4.json), with the LORO RULE @ 8d59e2989's estimators.
Matching rule: 2e's RULE @ b8db23ea7, Design item 2. Variants: cap30 (LEADS), captaskmin (registered sensitivity), cappaired (labelled
POST HOC sensitivity row, Delta_full only, never a lead). `sha`, `gate_diff` and `cell` are copied verbatim from run_loro_sep.py @ 6b4cd3a3e.
Cells per variant: +311 (S_new per-set), pooled 455, and S_old per-set beside; both readouts. Gates (fail closed): npz re-hash vs AMENDMENT 4;
for cappaired, every K = inf base array equals the v3 npz array of the same key on every draw; draw 0 vs the variant records; pooled
consistency on every draw. One Modal CPU container per variant.

  modal run --detach experiments/2026-10-15_loro_sep_09/run_loro_sep_runlen.py --variant cap30
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
APP = "probe-shift_aswin-surya_sub-loro_runlen_kill-2026-10-22"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
OUT_VOL = "probe-shift_aswin-surya_sub-loro_results_kill-2026-11-14"  # existing; not created here
RP = "experiments/2026-10-15_runlength_check/results"
VARIANTS = {  # variant: readout: (npz path, npz sha256, record path, record sha256); records @ 30c7b1b24
    "cap30": {"C": ("xfer/runlen/runlen_v2_C_cap30_2026-10-15_164057_perdraw.npz", "86adab795de8a8e4efe89a21d0aabf79734d17b99bf07f7263f62af7ceb98d12",
                    f"{RP}/runlen_v2_C_cap30_2026-10-15_164057.json", "3383433f29536115afb36cbbc8079db1586cdd06ea1b4fd5f421060e11a1ccb8"),
              "R": ("xfer/runlen/runlen_v2_R_cap30_2026-10-15_164100_perdraw.npz", "92bcf8ba157ad71b752f0edd78e1032d79ab19f2aef9e0af118af0b2fee7f2f0",
                    f"{RP}/runlen_v2_R_cap30_2026-10-15_164100.json", "46421a1385f46ac68ff549bcb00b9bec9928da0540c7c8b29cffe61245d4c8c1")},
    "captaskmin": {"C": ("xfer/runlen/runlen_v2_C_captaskmin_2026-10-15_164104_perdraw.npz", "a9df0dced15435f706276cbbae9cb19fcfc4745aabe92b5e8eb798dfdf46566c",
                         f"{RP}/runlen_v2_C_captaskmin_2026-10-15_164104.json", "d9b1e9992a636da0340ffa0285609798cb4b971bb874efa6eb00d51af9451f14"),
                   "R": ("xfer/runlen/runlen_v2_R_captaskmin_2026-10-15_164109_perdraw.npz", "f20ad6bba070d98508c1174e3f9756be9ad04828c3c3a2c062e01b59bbb63e2a",
                         f"{RP}/runlen_v2_R_captaskmin_2026-10-15_164109.json", "3f0c898654b2403ce984cb9c6ef8cf3b16703d491d474896e3bb9d714b192c68")},
    "cappaired": {"C": ("xfer/runlen/runlen_v2_C_cappaired_2026-10-15_164049_perdraw.npz", "931a347de3745f8c767e5158483172737c5dd708d15d88cab192ab06fab5c6ec",
                        f"{RP}/runlen_v2_C_cappaired_2026-10-15_164049.json", "2af64eba880e557c3dc2fb878551bec9ef12e682683e92f1e30d0145ae46817d"),
                  "R": ("xfer/runlen/runlen_v2_R_cappaired_2026-10-15_164052_perdraw.npz", "aa40f7020c0b7113251a5d8b6369f1d1667a81ffe78292ecffb5668b93488cad",
                        f"{RP}/runlen_v2_R_cappaired_2026-10-15_164052.json", "96769a73faf3edc3b14c17e18bdfa89ec9a9b1b51abe6c0da55196e63f4d324e")},
}
REC_COMMIT = "30c7b1b24"
V3_NPZ = {"C": ("xfer/fpr507prep/pooled455v2_C_2026-10-15_160734_perdraw.npz", "89b0fb5d1b25e6ab0e60b35ea31600f6943619ced99b0085d6ba56dffef6dd2e"),
          "R": ("xfer/fpr507prep/pooled455v2_R_2026-10-15_160736_perdraw.npz", "4589a3e9b215a7d9c882573cebdc5b809d1530a328f6c6344673fb796beb0d3a")}
MD = "experiments/2026-10-14_priority1_spec_09/loro_sep"
MANS = {"311": ("cd1afb33e", f"{MD}/loro_sep_manifest_311.json", "2dd2d5397b8ccb9f070e4301a41112740259560f9387692c1f6335ad7c1cfa73"),
        "amend4": ("6187764ad", f"{MD}/loro_sep_manifest_amend4.json", "76b97fde6c95bfb0551492a37667c18c4ae75725d356043d4fda944c5a205c23")}
ROLE = {"cap30": "LEADS (registered; RULE @ b8db23ea7 Design item 2)", "captaskmin": "registered sensitivity of K = 30",
        "cappaired": "POST HOC sensitivity row, Delta_full only, never a lead"}
MIN_DRAWS = 9500
B1 = 10001
MODE_OF = {"primary": ("joint", "Δ_full"), "fixed_threshold": ("fixed", "Δ_full"), "yardstick": ("joint", "Δ_rr"), "yardstick_fixed_threshold": ("fixed", "Δ_rr")}
PTS = ("0.01", "0.001", "0.05")
NAMES = {"primary": "Δ_full (joint)", "fixed_threshold": "Δ_full (fixed threshold)", "yardstick": "Δ_rr (joint), beside", "yardstick_fixed_threshold": "Δ_rr (fixed threshold), beside"}
LABELS = {"primary": "thresholds re-estimated per draw on each pool", "fixed_threshold": "FIXED-THRESHOLD INPUT: pool moments and thresholds at draw 0; attack tasks resampled",
          "yardstick": "thresholds re-estimated per draw on each pool", "yardstick_fixed_threshold": "FIXED-THRESHOLD INPUT"}

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



LORO_CAVEATS = [
    "model: independent, approximately Gaussian recipe effects with common mean mu and variance tau^2, plus mean-zero sampling errors with covariance Sigma independent of those effects",
    "complete-case handling of draws (D) does not establish that missingness is ignorable",
    "mean absolute gap keeps sampling-noise bias, especially near 0; its spread is descriptive, not a calibrated interval",
    "tau^2 is truncated at 0, which biases it upward; the untruncated value and the flag are saved",
    "prediction target: a new recipe's mean gap (members averaged by the same rule); within-recipe variation is not modelled; ranges do not cover individual probes",
    "(b) assumes evaluation on an independently sampled cohort of the same size and design; v_new (median over groups of Sigma_gg) is a working value for a new recipe mean's "
    "sampling variance, not an estimate for an individual probe; for evaluation on the same cohort the prediction-error variance is tau^2 + V_m + v_new - 2 c_new, and c_new = 0 is not assumed there",
    "the range for the new recipe mean's TRUE gap omits v_new",
    "coverage wording: nominal pointwise 95% ranges with t_{n-1}, n = k for (b) and k - 1 for leave-one-out; at k = 3 leave-one-out has one degree of freedom (t about 12.706); "
    "small k gives a large multiplier but guarantees neither a wide range nor adequate coverage",
    "held-out indicators are dependent; no binomial calibration test or simultaneous coverage claim is made; with about six groups coverage counts carry very little evidence about calibration",
    "the recipes are not a random sample of future probes; one reader (Qwen3.5-27B), one simulator; the intervals condition on the fitted probes",
    "joint inputs: thresholds re-estimated per draw on each pool; fixed-threshold inputs condition on draw 0's thresholds",
    "the leave-one-out check evaluates same-cohort prediction and does not validate the independent-cohort range; j-of-k is descriptive",
    "stress-test positions are descriptive; under separate thresholds c_asis is not a positive control (25h item 12)",
]


def readability(rec_row, f, inp, origin):
    """Per-member draw-0 readability for one origin, REPORTED ONLY (never changes membership). rec_row = a per-origin row of the record.
    S_old's yardstick uses run 1 at honest B alone (B2); its readability is k >= 1 and >= 10 units above T from pools_at_draw0 (not in .readable)."""
    rd, pl = rec_row["readable"][f], rec_row["pools_at_draw0"][f]
    sides = ("S", "B") if inp in ("primary", "fixed_threshold") else (("A", "B2") if origin == "S_old" else ("A", "B"))
    rdx = {s_: (bool(rd[s_]) if s_ in rd else bool(pl[s_]["k"] >= 1 and pl[s_]["units_above"] >= 10)) for s_ in sides}
    return {"readable": rdx, "readable_contrast": bool(all(rdx.values())),
            "n_units": {s_: pl[s_]["n_units"] for s_ in sides}, "units_above_T_draw0": {s_: pl[s_]["units_above"] for s_ in sides}}



HK = ("status", "k", "D", "a_mean_signed_gap", "a_mean_signed_gap_ci95", "tau2", "tau2_untruncated", "t", "b_range_new_recipe_mean_observed_independent_cohort",
      "b_range_new_recipe_mean_true_gap", "loo_inside")


@app.function(image=image, volumes={"/in": in_vol, "/out": out_vol}, cpu=2, memory=8192, timeout=900, retries=0)
def run(variant: str, readouts: str, rec_bytes: dict, man_bytes: dict, git: str, stamp: str) -> dict:
    import numpy as np
    import scipy
    from scipy import stats

    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant}")
    RO = tuple(readouts)
    if not RO or any(ro not in ("C", "R") for ro in RO) or len(set(RO)) != len(RO):
        raise SystemExit(f"bad readouts {readouts!r}")
    paired = variant == "cappaired"
    V = VARIANTS[variant]
    for nm, (_, _, s_) in MANS.items():
        if sha(man_bytes[nm]) != s_:
            raise SystemExit(f"manifest {nm} sha256 differs")
    man = {nm: json.loads(b) for nm, b in man_bytes.items()}
    am = man["amend4"]["inputs"][variant]
    for ro in RO:  # pins equal AMENDMENT 4
        p_, s_, rp_, rs_ = V[ro]
        if (am[ro]["npz"]["path"], am[ro]["npz"]["sha256"], am[ro]["record"]["path"], am[ro]["record"]["sha256"]) != (p_, s_, rp_, rs_):
            raise SystemExit(f"{variant} {ro}: pins differ from AMENDMENT 4")
        if not am[ro]["record"]["commit"].startswith(REC_COMMIT[:9]) or sha(rec_bytes[ro]) != rs_:
            raise SystemExit(f"{variant} {ro}: record commit or sha256 differs")
    rec = {ro: json.loads(rec_bytes[ro]) for ro in RO}
    Z, gate = {}, {"variant": variant, "npz_rehash_ok": [], "kinf_arrays_equal": 0, "draw0_checked": 0, "draw0_max_abs_diff": 0.0, "fixed_equals_joint": 0,
                   "pooled_consistency_draws_checked": 0, "pooled_consistency_max_abs_diff": 0.0}
    for ro in RO:  # re-hash on the Volume, fail closed
        raw = (Path("/in") / V[ro][0]).read_bytes()
        if sha(raw) != V[ro][1]:
            raise SystemExit(f"npz {variant} {ro} sha256 differs on the Volume")
        Z[ro] = np.load(io.BytesIO(raw), allow_pickle=False)
        gate["npz_rehash_ok"].append(ro)
    if paired:  # K = inf gate: every base array equals the v3 npz array of the same key, every draw
        for ro in RO:
            raw3 = (Path("/in") / V3_NPZ[ro][0]).read_bytes()
            if sha(raw3) != V3_NPZ[ro][1]:
                raise SystemExit(f"v3 npz {ro} sha256 differs on the Volume")
            z3 = np.load(io.BytesIO(raw3), allow_pickle=False)
            if not set(z3.files) <= set(Z[ro].files):
                raise SystemExit(f"K = inf gate: paired npz {ro} lacks v3 keys")
            for k in z3.files:
                if not np.array_equal(np.asarray(z3[k]), np.asarray(Z[ro][k]), equal_nan=True):
                    raise SystemExit(f"K = inf gate failed at {ro} {k}")
                gate["kinf_arrays_equal"] += 1

    def arr(ro, key):
        if key not in Z[ro].files:
            raise SystemExit(f"npz {variant} {ro} lacks {key}")
        a = np.asarray(Z[ro][key], dtype=float)
        if a.shape != (B1,) or np.isinf(a).any():
            raise SystemExit(f"{key}: shape {a.shape} or infinite values")
        return a

    def g0(x, ref, what):
        gate["draw0_max_abs_diff"] = max(gate["draw0_max_abs_diff"], gate_diff(x, ref, what, np))
        gate["draw0_checked"] += 1

    catch_seen = []
    rows = [r for r in man["311"]["rows"] if r["readout"] in RO]
    if len(rows) != 9 * len(RO):
        raise SystemExit("manifest rows for these readouts are not 9 per readout")
    GAPS = ("Δ_full",) if paired else ("Δ_full", "Δ_rr")
    RKEY = {"Δ_full": "delta_full_paired" if paired else "delta_full", "Δ_rr": "delta_rr"}

    def per_origin(ro, k, m, f, o):
        """Per-set gaps for one origin from its stored catches (the producer's definitions)."""
        pre = f"{k}__{m}__{f}__{o}__"
        if paired:
            s1, r1, s2, r2 = (arr(ro, pre + c) for c in ("sim_at_S__p1", "run1_at_B__p1", "sim_at_S__p2", "run2_at_B__p2"))
            catch_seen.extend([s1[0], r1[0], s2[0], r2[0]])
            return {"Δ_full": ((s1 - r1) + (s2 - r2)) / 2}
        sim, rm, r2A = arr(ro, pre + "sim_at_S__primary"), arr(ro, pre + "real_mean_at_B"), arr(ro, pre + "run2_at_A")
        y1 = arr(ro, pre + ("run1_at_B2" if o == "S_old" else "run1_at_B"))
        catch_seen.extend([sim[0], rm[0], r2A[0], y1[0]])
        return {"Δ_full": sim - rm, "Δ_rr": r2A - y1}

    def pooled_recomputed(ro, k, m, f):
        """The producer's pool_reduce arithmetic on the per-origin arrays (equal-main, D-weighted)."""
        o_ = {c: arr(ro, f"{k}__{m}__{f}__S_old__{c}") for c in ("D",)}
        n_ = {c: arr(ro, f"{k}__{m}__{f}__S_new__{c}") for c in ("D",)}
        Dd = o_["D"] + n_["D"]

        def comb(c_old, c_new=None):
            return (arr(ro, f"{k}__{m}__{f}__S_old__{c_old}") * o_["D"] + arr(ro, f"{k}__{m}__{f}__S_new__{c_new or c_old}") * n_["D"]) / Dd
        if paired:
            return {"Δ_full": ((comb("sim_at_S__p1") - comb("run1_at_B__p1")) + (comb("sim_at_S__p2") - comb("run2_at_B__p2"))) / 2}
        return {"Δ_full": comb("sim_at_S__primary") - comb("real_mean_at_B"), "Δ_rr": comb("run2_at_A") - comb("run1_at_B2", "run1_at_B")}

    V311, V455, VOLD = {}, {}, {}
    for r in rows:
        lab, ro, k = r["inventory_key"], r["readout"], r["npz_label"]
        rr = rec[ro]["rows"][lab]
        pr = rec[ro]["pooled"][lab]["pooled"]
        for f in PTS:
            for m in ("joint", "fixed"):
                new_, old_ = per_origin(ro, k, m, f, "S_new"), per_origin(ro, k, m, f, "S_old")
                rc = pooled_recomputed(ro, k, m, f)
                for g_ in GAPS:
                    v = arr(ro, f"{k}__{m}__{f}__{RKEY[g_]}")
                    x = rc[g_]
                    if np.isinf(x).any() or not np.array_equal(np.isnan(v), np.isnan(x)):
                        raise SystemExit(f"pooled consistency: NaN/inf pattern differs at {lab} {f} {m} {g_}")
                    fin = np.isfinite(v)
                    d_ = float(np.abs(v[fin] - x[fin]).max()) if fin.any() else 0.0
                    if d_ > 1e-12:
                        raise SystemExit(f"pooled consistency: max |diff| {d_} at {lab} {f} {m} {g_}")
                    gate["pooled_consistency_max_abs_diff"] = max(gate["pooled_consistency_max_abs_diff"], d_)
                    gate["pooled_consistency_draws_checked"] += int(fin.sum())
                    V455[(lab, f, m, g_)], V311[(lab, f, m, g_)], VOLD[(lab, f, m, g_)] = v, new_[g_], old_[g_]
                    g0(v[0], pr[m][f]["gap"][RKEY[g_]]["point"], f"pooled {lab} {f} {m} {g_}")
                    g0(new_[g_][0], rr["S_new"]["points"][m][f]["gap"][RKEY[g_]]["point"], f"S_new {lab} {f} {m} {g_}")
                    g0(old_[g_][0], rr["S_old"]["points"][m][f]["gap"][RKEY[g_]]["point"], f"S_old {lab} {f} {m} {g_}")
            for g_ in GAPS:
                for Vx, tag in ((V311, "S_new"), (V455, "pooled"), (VOLD, "S_old")):
                    a_, b_ = float(Vx[(lab, f, "joint", g_)][0]), float(Vx[(lab, f, "fixed", g_)][0])
                    if not ((np.isnan(a_) and np.isnan(b_)) or abs(a_ - b_) <= 1e-12):
                        raise SystemExit(f"fixed draw 0 != joint draw 0 at {tag} {lab} {f} {g_}")
                    gate["fixed_equals_joint"] += 1
    if gate["draw0_max_abs_diff"] > 1e-9:
        raise SystemExit(f"draw-0 gate failed: max |diff| {gate['draw0_max_abs_diff']}")
    cs = np.array(catch_seen, dtype=float)
    cs = cs[np.isfinite(cs)]
    if cs.size == 0 or cs.min() < -1e-12 or cs.max() > 1 + 1e-12:
        raise SystemExit("finite catch entries at draw 0 are not fractions in [0, 1]")
    scale = 100.0
    INPUTS = ("primary", "fixed_threshold") if paired else ("primary", "fixed_threshold", "yardstick", "yardstick_fixed_threshold")

    def role_of(entry, inp, f):
        if variant == "cappaired":
            return "POST HOC sensitivity row (paired cut), never a lead"
        if variant == "captaskmin":
            return "registered sensitivity of K = 30 (per-task min)"
        if entry == "S_old_per_set":
            return "beside (S_old per-set), K = 30"
        return "PRIMARY (lead) of the length-matched LORO, K = 30" if (inp == "primary" and f == "0.01") else ("K = 30, primary input, beside point" if inp == "primary" else "K = 30, beside, labelled")

    def cells_for(Vx, origins, prefix, entry):
        out = {}
        for ro in RO:
            rws = [r for r in rows if r["readout"] == ro]
            for f in PTS:
                for inp in INPUTS:
                    m, g_ = MODE_OF[inp]
                    members = [{"label": r["inventory_key"], "group": r["loro_group"], "values": Vx[(r["inventory_key"], f, m, g_)] * scale} for r in rws]
                    c = cell(members, f"{g_} ({m})", np, stats)
                    out[f"{ro}|{f}|{inp}"] = {
                        "readout": ro, "point": f, "input": inp, "variant": variant, "name": f"{prefix}{NAMES[inp]} [{variant}]", "input_label": f"{LABELS[inp]}; matching {variant}: {ROLE[variant]}",
                        "role": role_of(entry, inp, f),
                        "readability_by_member_reported_only": {r["inventory_key"]: {o: readability(rec[ro]["rows"][r["inventory_key"]][o], f, inp, o) for o in origins} for r in rws},
                    } | c
        return out

    c311, c455, cold = cells_for(V311, ("S_new",), "+311 ", "311"), cells_for(V455, ("S_old", "S_new"), "pooled ", "455"), cells_for(VOLD, ("S_old",), "S_old per-set ", "S_old_per_set")

    def headline(cells):
        return {key: {"label": cells[key]["input_label"], "variant": variant, "role": cells[key]["role"]} | {k_: cells[key].get(k_) for k_ in HK}
                for key in (f"{ro}|{f}|{inp}" for f in PTS for ro in RO for inp in ("primary", "yardstick") if inp in INPUTS)}

    res = {"label": f"Length-matched separate-threshold LORO, variant {variant} ({ROLE[variant]}). POST HOC; PRELIMINARY. v3 inputs: Q1–Q4 FINAL with replacements pending "
                    "(09's corrected label @ 60d13fd01). Target = a new recipe's MEAN Delta_full under this matching; ranges do not cover individual probes",
           "variant": variant, "role": ROLE[variant], "readouts_run": list(RO), "git": git, "stamp": stamp,
           "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
           "inputs": {"npz": {ro: {"volume": IN_VOL, "path": V[ro][0], "sha256": V[ro][1]} for ro in RO},
                      "records": {ro: {"commit": REC_COMMIT, "path": V[ro][2], "sha256": V[ro][3]} for ro in RO},
                      "manifests": {nm: {"commit": c_, "path": p_, "sha256": s_} for nm, (c_, p_, s_) in MANS.items()},
                      "kinf_baseline_npz": ({ro: {"path": V3_NPZ[ro][0], "sha256": V3_NPZ[ro][1]} for ro in RO} if paired else None)},
           "matching_rule": man["amend4"]["matching_rule"][variant], "gates": gate,
           "units": {"scale_applied": scale, "rule": "producer stores fractions; x100 to percentage points"},
           "membership": "LORO RULE finite-draw eligibility only; readability reported per member, never used for membership",
           "estimators": "LORO RULE @ 8d59e2989 (cell() verbatim from run_loro_sep.py @ 6b4cd3a3e)",
           "unmatched_baseline": "loro_sep311v3_CR / loro_sep455v3_CR _2026-10-15_162243 (aswin/loro-sep-09)",
           "entries": {"311": {"label": "+311 (S_new per-set)", "headline": headline(c311), "cells": c311},
                       "455": {"label": "pooled 455 (S_old + S_new, equal-main); origin strata not shown equivalent", "headline": headline(c455), "cells": c455},
                       "S_old_per_set": {"label": "S_old per-set (the pooled job's S_old design), beside", "headline": headline(cold), "cells": cold}},
           "caveats": LORO_CAVEATS + ["length matching changes the units each probe sees, so thresholds and catches are re-estimated on capped units; compare with the v3 unmatched records by key"]}
    dest = Path("/out") / "loro_sep_09" / f"loro_runlen_{variant}_{readouts}_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    out_vol.commit()
    return res


@app.local_entrypoint()
def main(variant: str, readouts: str = "CR"):
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant}")
    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    rec_bytes = {ro: g("show", f"{REC_COMMIT}:{VARIANTS[variant][ro][2]}") for ro in readouts}
    man_bytes = {nm: g("show", f"{c_}:{p_}") for nm, (c_, p_, _) in MANS.items()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(variant, readouts, rec_bytes, man_bytes, head, stamp)
    dest = HERE / "results" / f"loro_runlen_{variant}_{readouts}_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps({"gates": res["gates"]}, indent=1))
    print("wrote", dest.name)
