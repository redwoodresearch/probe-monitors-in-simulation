"""Independent check of the U_rt real record against v6 (scores-only JSON; no transcript text).
Usage: python3 -I check_thr.py <real.json> <xerr_v6.json>"""

import hashlib
import json
import sys

rp, vp = sys.argv[1], sys.argv[2]
print("sha256 real", hashlib.sha256(open(rp, "rb").read()).hexdigest()[:16])
print("sha256 v6  ", hashlib.sha256(open(vp, "rb").read()).hexdigest()[:16])
R = json.load(open(rp))
V = json.load(open(vp))["result"]
r = R["result"]
print("record git", R["git"][:9], "stamp", R["stamp"], "mode", R["mode"])
print("gates (record flags):", json.dumps(r["gates"]))
print("test_checks pass:", r["test_checks"]["pass"], "draws:", r["draws"], "universes:", r["calibration_universes"])

ROWS = list(r["rows"])
COMPS = ("sim_minus_mean_real", "reexecution_run2_minus_run1")
SCOPES = ("pooled", "with_strategy", "without_strategy")

# ---- Gate A: record's OFF blocks vs v6 registered values (independent re-comparison)
n_fields, max_diff, worst = 0, 0.0, None
for rk in ROWS:
    elig = r["rows"][rk]["eligible_frozen"]
    assert elig == V["results"][rk]["eligibility_append13_4"]["eligible"], rk
    for cn in COMPS:
        for sc in SCOPES:
            o = r["rows"][rk]["comparisons"][cn][sc]["fixed_threshold_reproduced"]
            v6 = V["results"][rk]["primary_eligible_estimation_half"][cn][sc]
            pairs = []
            for f in ("X_err", "U_approx_upper_95", "bias_mean", "spread"):
                assert (o.get(f) is None) == (v6.get(f) is None), (rk, cn, sc, f)
                if o.get(f) is not None:
                    pairs.append((f, o[f], v6[f]))
            if o.get("bias_ci95") is not None:
                pairs += [("bias_ci95", a, b) for a, b in zip(o["bias_ci95"], v6["bias_ci95"], strict=True)]
            for p in elig:
                q, q6 = o["per_probe"][p], v6["per_probe"][p]
                pairs += [(p + ".delta", q["delta"], q6["delta"]), (p + ".U_abs", q["U_abs_approx"], q6["U_abs_approx"])]
                pairs += [(p + ".ci95", a, b) for a, b in zip(q["ci95"], q6["ci95"], strict=True)]
            for f, a, b in pairs:
                n_fields += 1
                d = abs(a - b)
                if d > max_diff or worst is None:
                    max_diff, worst = max(d, max_diff), (rk, cn, sc, f)
print(f"GATE A: OFF vs v6 fields={n_fields} max_abs_diff={max_diff}")

# ---- Gate B: T0 vs v6 thresholds, all 5 probes x 4 rows
td = [abs(r["rows"][rk]["per_probe_threshold_info"][p]["threshold_T0"] - V["results"][rk]["thresholds"][p]["threshold"])
      for rk in ROWS for p in r["rows"][rk]["per_probe_threshold_info"]]
print(f"GATE B: T0 vs v6 thresholds n={len(td)} max_abs_diff={max(td)}")
print("GATE C: Sol moments are NOT stored in the record (only the record flag sol_moments_equal_v6 is available)")

# ---- Table
f2 = lambda x: "None" if x is None else f"{x:.2f}"
print("\nTABLE  row | comp | scope | n | U | U_rt_joint | U_rt_indep | rt-U joint | indep-joint | X_err | X_err_rt | complete j/i")
gaps = []
for rk in ROWS:
    for cn in COMPS:
        for sc in SCOPES:
            b = r["rows"][rk]["comparisons"][cn][sc]
            o, j, i = b["fixed_threshold_reproduced"], b["rt_joint_U_rt"], b["rt_independent_sensitivity"]
            U, Uj, Ui = o.get("U_approx_upper_95"), j.get("U_approx_upper_95"), i.get("U_approx_upper_95")
            dj = None if None in (U, Uj) else Uj - U
            dij = None if None in (Uj, Ui) else Ui - Uj
            if dij is not None:
                gaps.append((cn, sc, rk, dij))
            if b.get("U_rt_minus_U") and dj is not None:
                assert abs(b["U_rt_minus_U"]["joint"] - dj) < 1e-12
            print(f"{rk.split('|')[0][:10]} a={rk.split('=')[1]} | {cn[:6]} | {sc[:7]} | {b['n_tasks']} | {f2(U)} | {f2(Uj)} | {f2(Ui)} | "
                  f"{f2(dj)} | {f2(dij)} | {f2(o.get('X_err'))} | {f2(j.get('X_err'))} | {j.get('complete_draws')}/{i.get('complete_draws')}")
            if cn == "sim_minus_mean_real" and "catch_percent" in b:
                cp = b["catch_percent"]
                elig = r["rows"][rk]["eligible_frozen"]
                fam = cp["family_mean"]
                for w in ("run1", "run2", "real_mean", "sim"):  # family = equal-weight mean over eligible probes
                    m = sum(cp[p][w]["point"] for p in elig) / len(elig)
                    assert abs(m - fam[w]["point"]) < 1e-9, (rk, sc, w)
                    if w == "real_mean":
                        assert abs(fam[w]["point"] - (fam["run1"]["point"] + fam["run2"]["point"]) / 2) < 1e-9
                print(f"      catch% run1 {fam['run1']['point']:.2f} run2 {fam['run2']['point']:.2f} real {fam['real_mean']['point']:.2f} "
                      f"sim {fam['sim']['point']:.2f} | real ci fixed {[round(x, 1) for x in fam['real_mean']['ci95_fixed_threshold']]} "
                      f"rt {[round(x, 1) for x in fam['real_mean']['ci95_threshold_resampled_joint']]}")

head = [g for g in gaps if g[0] == "sim_minus_mean_real" and g[1] in ("pooled", "with_strategy")]
print("\nindep - joint, sim_minus pooled/with_strategy:", [(g[2].split('|')[0][:3] + g[2][-4:], g[1][:4], round(g[3], 4)) for g in head])
print("max |indep-joint| headline cells:", round(max(abs(g[3]) for g in head), 4))
print("max |indep-joint| all cells:", round(max(abs(g[3]) for g in gaps), 4), [g for g in gaps if abs(g[3]) == max(abs(x[3]) for x in gaps)])

# ---- Eligibility-stability shares and threshold SDs
print("\nELIG  row | probe | eligible_frozen | v6 real TPR elig half | share joint | share indep | T0 | sd joint | invalid j/i")
for rk in ROWS:
    tpr = V["results"][rk]["eligibility_append13_4"]["real_tpr_eligibility_half"]
    for p, inf in r["rows"][rk]["per_probe_threshold_info"].items():
        s = inf["eligibility_decision_unchanged_share"]
        print(f"{rk.split('|')[0][:10]} a={rk.split('=')[1]} | {p} | {inf['eligible_frozen']} | {tpr[p]:.4f} | {s['joint']:.4f} | "
              f"{s['independent']:.4f} | {inf['threshold_T0']:.4f} | {inf['threshold_sd_over_draws']['joint']:.4f} | "
              f"{inf['invalid_draws_by_reason']['joint']['union']}/{inf['invalid_draws_by_reason']['independent']['union']}")

# ---- prediction
b = r["rows"]["R_reply_meanpooled|mechanical|alpha=0.05"]["comparisons"]["sim_minus_mean_real"]["with_strategy"]
print("\nPREDICTION reply 5% with-strategy joint U_rt-U =", round(b["U_rt_minus_U"]["joint"], 4), "(predicted +1.5 +/- 1.5)")
