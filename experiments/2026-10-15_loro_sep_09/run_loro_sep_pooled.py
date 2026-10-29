"""09's separate-threshold LORO on +311 (S_new) and on the pooled 455, both readouts, exactly per the frozen manifests
experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_311.json and loro_sep_manifest_455.json @ cd1afb33e, with the
LORO RULE @ 8d59e2989's estimators. `sha`, `gate_diff` and `cell` are copied verbatim from run_loro_sep.py @ 6b4cd3a3e (the 183 run).

Inputs: 2e's pooled-455 per-draw npzs (command 84112f50..., reply bd5875fb...). +311 = each npz's S_new per-origin catches
(Delta_full = sim@S|primary - real_mean@B; Delta_rr = run2@A - run1@B), gated at draw 0 against the hot-deck records @ c838127d3 and
the pooled records' per-set rows. Pooled = the stored pooled gaps, checked on every draw against the equal-main combination of the
per-origin arrays; S_old per-set cells beside (Delta_full = sim@S|primary - real_mean@B; Delta_rr = run2@A - run1@B2).
Writes two records: loro_sep311_<stamp>.json and loro_sep455_<stamp>.json. Refits nothing. POST HOC, PRELIMINARY. One Modal CPU container.

  modal run --detach experiments/2026-10-15_loro_sep_09/run_loro_sep_pooled.py
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
APP = "probe-shift_aswin-surya_sub-loro_sep311pooled_kill-2026-10-22"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
OUT_VOL = "probe-shift_aswin-surya_sub-loro_results_kill-2026-11-14"  # existing LORO results Volume; not created here
NPZ = {"C": ("xfer/fpr507prep/pooled455_2026-10-15_141921_perdraw.npz", "84112f5061455f9b1970ac90ff963e31d38f8c75b26b3cf28510e2c06476dbc8"),
       "R": ("xfer/fpr507prep/pooled455_2026-10-15_142548_perdraw.npz", "bd5875fbd09fc9177783f7bd84f6b862dae1d2eb8818f2d25b1461b1bd165a11")}
RD = "experiments/2026-10-15_fpr_sweep_507_prep/results"
RECS = {  # name: (commit, path, sha256)
    "pooled_C": ("6d2a2fbc5", f"{RD}/pooled455_C_2026-10-15_141921.json", "587739d0b7d9a7fc7e10a292927fc06dc5d460fe91e606b3ae306c94c508b3c4"),
    "pooled_R": ("45db6b238", f"{RD}/pooled455_R_2026-10-15_142548.json", "c794e175a324b17e2eac45cac06824463acc0eff655def7d07efc6d371214bbf"),
    "hd_C": ("c838127d3", f"{RD}/peraction311_all_hotdeck_C_2026-10-15_140206.json", "8a00f2060ef6aec70822a6e6af3f8169540b2e513641a82a51749ff6bdc69c41"),
    "hd_R": ("c838127d3", f"{RD}/peraction311_all_hotdeck_R_2026-10-15_140206.json", "a1a150fb20c54c519d5ab127ca675d610efea41e54ce824d3eb6cb577fcfce78"),
    "hd_O": ("c838127d3", f"{RD}/peraction311_all_hotdeck_O_2026-10-15_140209.json", "bcd97eb54097bedb94e43a7672ab190cdb4fa24bc3db89307f6ac98b60a4cf0a"),
    "hd_X": ("c838127d3", f"{RD}/peraction311_all_hotdeck_X_2026-10-15_140208.json", "c232bbb6d7d3692aa8d5e9239bde4c9f0be96e0487228583c138eed7d0f8a6d5"),
}
MD = "experiments/2026-10-14_priority1_spec_09/loro_sep"
MANS = {"311": ("cd1afb33e", f"{MD}/loro_sep_manifest_311.json", "2dd2d5397b8ccb9f070e4301a41112740259560f9387692c1f6335ad7c1cfa73"),
        "455": ("cd1afb33e", f"{MD}/loro_sep_manifest_455.json", "f83c8880298ffe1a28fe8a761e9c024b0709e17cb7b9ba734e30a36cfffa0688")}
MIN_DRAWS = 9500
B1 = 10001
INPUTS = ("primary", "fixed_threshold", "yardstick", "yardstick_fixed_threshold")
MODE_OF = {"primary": ("joint", "Δ_full"), "fixed_threshold": ("fixed", "Δ_full"), "yardstick": ("joint", "Δ_rr"), "yardstick_fixed_threshold": ("fixed", "Δ_rr")}
REC_GAP = {"Δ_full": "delta_full", "Δ_rr": "delta_rr"}
PTS = ("0.01", "0.001", "0.05")

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
HK = ("status", "k", "D", "a_mean_signed_gap", "a_mean_signed_gap_ci95", "tau2", "tau2_untruncated", "t", "b_range_new_recipe_mean_observed_independent_cohort",
      "b_range_new_recipe_mean_true_gap", "loo_inside")


def readability(rec_row, f, inp):
    """Per-member draw-0 readability for one origin, REPORTED ONLY (never changes membership). rec_row = a per-origin row of the pooled record."""
    rd, pl = rec_row["readable"][f], rec_row["pools_at_draw0"][f]
    sides = ("S", "B") if inp in ("primary", "fixed_threshold") else ("A", "B")
    return {"readable": {s: bool(rd[s]) for s in sides}, "readable_contrast": bool(all(rd[s] for s in sides)),
            "n_units": {s: pl[s]["n_units"] for s in sides}, "units_above_T_draw0": {s: pl[s]["units_above"] for s in sides}}


@app.function(image=image, volumes={"/in": in_vol, "/out": out_vol}, cpu=2, memory=8192, timeout=900, retries=0)
def run(rec_bytes: dict, man_bytes: dict, git: str, stamp: str) -> dict:
    import numpy as np
    import scipy
    from scipy import stats

    for nm, (_, _, s_) in RECS.items():
        if sha(rec_bytes[nm]) != s_:
            raise SystemExit(f"record {nm} sha256 differs")
    for nm, (_, _, s_) in MANS.items():
        if sha(man_bytes[nm]) != s_:
            raise SystemExit(f"manifest {nm} sha256 differs")
    rec = {nm: json.loads(b) for nm, b in rec_bytes.items()}
    man = {nm: json.loads(b) for nm, b in man_bytes.items()}
    Z = {}
    for ro, (p_, s_) in NPZ.items():
        raw = (Path("/in") / p_).read_bytes()
        if sha(raw) != s_:
            raise SystemExit(f"npz {ro} sha256 differs")
        Z[ro] = np.load(io.BytesIO(raw), allow_pickle=False)
    # every record pin must equal the governing manifests' provenance (commit prefix, path, sha256)
    pins = {f"hd_{g_}": v for g_, v in man["311"]["input_provenance"]["records"]["hotdeck"].items()}
    for nm in ("311", "455"):
        pins |= {f"pooled_{ro}": v for ro, v in man[nm]["input_provenance"]["records"]["pooled_455"].items()}
    if sorted(pins) != sorted(RECS) or any(not (pins[nm]["commit"].startswith(c_) or c_.startswith(pins[nm]["commit"])) or pins[nm]["path"] != p_ or pins[nm]["sha256"] != s_
                                           for nm, (c_, p_, s_) in RECS.items()):
        raise SystemExit("record pins differ from the manifests' provenance")
    for nm in ("311", "455"):
        mp = man[nm]["input_provenance"]["npz"]
        if sorted(mp) != sorted(NPZ) or any((mp[ro]["path"], mp[ro]["sha256"]) != NPZ[ro] for ro in NPZ):
            raise SystemExit(f"manifest {nm} npz pointer differs from the pinned npz")
    cache = {}

    def arr(ro, key):
        """One per-draw array; it must exist with shape (10001,)."""
        if (ro, key) not in cache:
            if key not in Z[ro].files:
                raise SystemExit(f"npz {ro} lacks {key}")
            a = np.asarray(Z[ro][key], dtype=float)
            if a.shape != (B1,):
                raise SystemExit(f"{key} shape {a.shape}")
            cache[(ro, key)] = a
        return cache[(ro, key)]

    def diff(ro, pair):
        a_, op, b_ = pair
        if op != "minus":
            raise SystemExit("unexpected key-pair operator")
        return arr(ro, a_) - arr(ro, b_)

    catch_seen = []
    gate = {"draw0_checked": 0, "draw0_max_abs_diff": 0.0, "fixed_equals_joint": 0, "pooled_consistency_draws_checked": 0, "pooled_consistency_max_abs_diff": 0.0}

    def g0(x, ref, what):
        gate["draw0_max_abs_diff"] = max(gate["draw0_max_abs_diff"], gate_diff(x, ref, what, np))
        gate["draw0_checked"] += 1

    def fixed_eq(xj, xf, what):
        a_, b_ = float(xj[0]), float(xf[0])
        if not ((np.isnan(a_) and np.isnan(b_)) or abs(a_ - b_) <= 1e-12):
            raise SystemExit(f"fixed draw 0 != joint draw 0 at {what}")
        gate["fixed_equals_joint"] += 1

    def short(key, prefix):
        if not key.startswith(prefix):
            raise SystemExit(f"key {key} lacks prefix {prefix}")
        return key[len(prefix):]

    def hd_name(lab):
        fam, ro = lab[:-2], lab[-1]
        return "hd_X" if fam == "c_asis" else ("hd_O" if fam.startswith("own_") else f"hd_{ro}")

    # ---------------- +311 (S_new) entry ----------------
    V311 = {}
    for r in man["311"]["rows"]:
        lab, ro = r["inventory_key"], r["readout"]
        prow = rec[f"pooled_{ro}"]["rows"][lab]["S_new"]
        hrow = rec[hd_name(lab)]["rows"][lab]
        for f in PTS:
            for m in ("joint", "fixed"):
                ks = r["npz_keys"][m][f]
                for nm_ in ("Δ_full", "Δ_rr"):
                    v = diff(ro, ks[nm_])
                    V311[(lab, f, m, nm_)] = v
                    g0(v[0], hrow["points"][m][f]["gap"][REC_GAP[nm_]]["point"], f"311 {lab} {f} {m} {nm_} vs hot-deck")
                    g0(v[0], prow["points"][m][f]["gap"][REC_GAP[nm_]]["point"], f"311 {lab} {f} {m} {nm_} vs pooled per-set")
                for k_ in (ks["Δ_full"][0], ks["Δ_full"][2], ks["Δ_rr"][0], ks["Δ_rr"][2]):
                    catch_seen.append(arr(ro, k_)[0])
            for nm_ in ("Δ_full", "Δ_rr"):
                fixed_eq(V311[(lab, f, "joint", nm_)], V311[(lab, f, "fixed", nm_)], f"311 {lab} {f} {nm_}")

    # ---------------- pooled 455 entry (+ S_old per-set) ----------------
    V455, VOLD = {}, {}
    for r in man["455"]["rows"]:
        lab, ro = r["inventory_key"], r["readout"]
        pr = rec[f"pooled_{ro}"]
        for f in PTS:
            for m in ("joint", "fixed"):
                ks = r["npz_keys"][m][f]
                po = ks["per_origin_consistency"]
                o_, n_ = ({short(k, f"{r['npz_label']}__{m}__{f}__{o}__"): arr(ro, k) for k in po[o]} for o in ("S_old", "S_new"))
                Dd = o_["D"] + n_["D"]
                comb = lambda k1, k2=None: (o_[k1] * o_["D"] + n_[k2 or k1] * n_["D"]) / Dd  # noqa: E731  (the producer's pool_reduce arithmetic)
                recomputed = {"Δ_full": comb("sim_at_S__primary") - comb("real_mean_at_B"), "Δ_rr": comb("run2_at_A") - comb("run1_at_B2", "run1_at_B")}
                for nm_ in ("Δ_full", "Δ_rr"):
                    v = arr(ro, ks[nm_])
                    x = recomputed[nm_]
                    if np.isinf(v).any() or np.isinf(x).any():
                        raise SystemExit(f"pooled consistency: infinite value at {lab} {f} {m} {nm_}")
                    if not np.array_equal(np.isnan(v), np.isnan(x)):
                        raise SystemExit(f"pooled consistency: NaN pattern differs at {lab} {f} {m} {nm_}")
                    fin = np.isfinite(v)
                    d_ = float(np.abs(v[fin] - x[fin]).max()) if fin.any() else 0.0
                    if d_ > 1e-12:
                        raise SystemExit(f"pooled consistency: max |diff| {d_} at {lab} {f} {m} {nm_}")
                    gate["pooled_consistency_max_abs_diff"] = max(gate["pooled_consistency_max_abs_diff"], d_)
                    gate["pooled_consistency_draws_checked"] += int(fin.sum())
                    V455[(lab, f, m, nm_)] = v
                    g0(v[0], pr["pooled"][lab]["pooled"][m][f]["gap"][REC_GAP[nm_]]["point"], f"455 {lab} {f} {m} {nm_} vs pooled record")
                    w = diff(ro, r["S_old_per_set_npz_keys"][m][f][nm_])
                    VOLD[(lab, f, m, nm_)] = w
                    g0(w[0], pr["rows"][lab]["S_old"]["points"][m][f]["gap"][REC_GAP[nm_]]["point"], f"S_old per-set {lab} {f} {m} {nm_} vs pooled record")
                for o in ("S_old", "S_new"):  # every exported per-origin catch array (not D, not gaps)
                    for c_ in ("sim_at_S__primary", "run1_at_B", "run2_at_B", "real_mean_at_B", "run2_at_A", "run1_at_A") + (("run1_at_B2",) if o == "S_old" else ()):
                        catch_seen.append(arr(ro, f"{r['npz_label']}__{m}__{f}__{o}__{c_}")[0])
            for nm_ in ("Δ_full", "Δ_rr"):
                fixed_eq(V455[(lab, f, "joint", nm_)], V455[(lab, f, "fixed", nm_)], f"455 {lab} {f} {nm_}")
                fixed_eq(VOLD[(lab, f, "joint", nm_)], VOLD[(lab, f, "fixed", nm_)], f"S_old {lab} {f} {nm_}")
    if gate["draw0_max_abs_diff"] > 1e-9:
        raise SystemExit(f"draw-0 gate failed: max |diff| {gate['draw0_max_abs_diff']}")
    cs = np.array(catch_seen, dtype=float)
    cs = cs[np.isfinite(cs)]
    if cs.size == 0 or cs.min() < -1e-12 or cs.max() > 1 + 1e-12:
        raise SystemExit("finite catch entries at draw 0 are not fractions in [0, 1]")
    scale = 100.0
    gate["catch_draw0_entries_checked"] = int(cs.size)

    def cells_for(entry, V, origin_of_readability, name_prefix=""):
        out = {}
        rows = man[entry]["rows"]
        for ro in ("C", "R"):
            rws = [r for r in rows if r["readout"] == ro]
            for f in PTS:
                for inp in INPUTS:
                    m, nm_ = MODE_OF[inp]
                    members = [{"label": r["inventory_key"], "group": r["loro_group"], "values": V[(r["inventory_key"], f, m, nm_)] * scale} for r in rws]
                    c = cell(members, f"{nm_} ({m})", np, stats)
                    comp = man[entry]["comparators"][inp]
                    out[f"{ro}|{f}|{inp}"] = {
                        "readout": ro, "point": f, "input": inp, "name": name_prefix + comp["name"], "input_label": comp.get("label"),
                        "role": ("PRIMARY (lead)" if (inp == "primary" and f == "0.01") else ("primary input, beside point" if inp == "primary" else "beside, labelled")),
                        "readability_by_member_reported_only": {r["inventory_key"]: {o: readability(rec[f"pooled_{ro}"]["rows"][r["inventory_key"]][o], f, inp) for o in origin_of_readability}
                                                                for r in rws},
                    } | c
        return out

    c311 = cells_for("311", V311, ("S_new",))
    c455 = cells_for("455", V455, ("S_old", "S_new"))
    cold = cells_for("455", VOLD, ("S_old",), "S_old per-set ")
    common = {"git": git, "stamp": stamp, "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
              "inputs": {"npz": {ro: {"volume": IN_VOL, "path": p_, "sha256": s_} for ro, (p_, s_) in NPZ.items()},
                         "records": {nm: {"commit": c_, "path": p_, "sha256": s_} for nm, (c_, p_, s_) in RECS.items()},
                         "manifests": {nm: {"commit": c_, "path": p_, "sha256": s_} for nm, (c_, p_, s_) in MANS.items()}},
              "gates": {"npz_sha256": True, "records_sha256": True, "manifests_sha256": True, "keys_and_shapes": True, **gate},
              "units": {"scale_applied": scale, "rule": "producer stores fractions; x100 to percentage points"},
              "membership": "LORO RULE finite-draw eligibility only; readability reported per member, never used for membership (manifest)",
              "estimators": "LORO RULE @ 8d59e2989 (cell() verbatim from run_loro_sep.py @ 6b4cd3a3e)"}

    def headline(cells):
        return {key: {k_: cells[key].get(k_) for k_ in HK} for key in (f"{ro}|{f}|{inp}" for f in PTS for ro in ("C", "R") for inp in ("primary", "yardstick"))}

    r311 = {"label": "Separate-threshold LORO, +311 (S_new) only, both readouts. POST HOC (+311 values were read by 2e), PRELIMINARY. 'Q4 FINAL; Q1, Q3 interim (routed roots so far)'. "
                     "Target = a new recipe's MEAN Delta_full (per action, separate thresholds), not an individual probe",
            **common, "draws": man["311"]["draws"], "cohort": man["311"]["cohort"], "own_a": man["311"]["own_a"],
            "headline": headline(c311), "cells": c311, "caveats": LORO_CAVEATS, "manifest_caveats_verbatim": man["311"]["caveats"]}
    r455 = {"label": "Separate-threshold LORO, POOLED 455 (S_old + S_new, equal-main), both readouts, with S_old per-set cells beside. POST HOC; not an authorized registered row; "
                     "origin strata not shown equivalent (21j: inconclusive at this n); PRELIMINARY. Target = a new recipe's MEAN pooled Delta_full",
            **common, "draws": man["455"]["draws"], "cohort": man["455"]["cohort"], "own_a": man["455"]["own_a"],
            "headline": headline(c455), "cells": c455,
            "S_old_per_set": {"label": "S_old per-set, the pooled job's S_old design (tau_real on B u C; yardstick honest C at tau_C - honest-B-alone run 1 at tau_B); NOT the 183 LORO's matched design @ b5b3796e7. "
                                       "Yardstick readability: the record reports S/B/A (B = B u C, A = honest C); honest-B-alone (B2) readability is not in the record",
                              "headline": headline(cold), "cells": cold},
            "S_new_per_set": "the +311 record of this run", "caveats": LORO_CAVEATS, "manifest_caveats_verbatim": man["455"]["caveats"]}
    outs = {}
    for nm, res in (("311", r311), ("455", r455)):
        dest = Path("/out") / "loro_sep_09" / f"loro_sep{nm}_{stamp}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x") as fh:
            fh.write(json.dumps(res, indent=1))
        outs[nm] = res
    out_vol.commit()
    return outs


@app.local_entrypoint()
def main():
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    rec_bytes = {nm: g("show", f"{c_}:{p_}") for nm, (c_, p_, _) in RECS.items()}
    man_bytes = {nm: g("show", f"{c_}:{p_}") for nm, (c_, p_, _) in MANS.items()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(rec_bytes, man_bytes, head, stamp)
    (HERE / "results").mkdir(exist_ok=True)
    for nm, r in res.items():
        dest = HERE / "results" / f"loro_sep{nm}_{stamp}.json"
        dest.open("x").write(json.dumps(r, indent=1))
        print("wrote", dest.name)
    print(json.dumps({"gates": res["311"]["gates"], "units": res["311"]["units"]}, indent=1))
