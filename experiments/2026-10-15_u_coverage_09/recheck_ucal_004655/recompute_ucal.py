"""Independent recheck of ucal_2026-10-15_004655.json (Amendment 1, POST-HOC) from its per-dataset rows.

Definitions are taken from RULE.md Amendment 1 @ e4a2fd6ca, not from run_ucal.py:
  d = theta(T-hat) - X-hat; q = max(0, max over patterns of the inverted_cdf 95th pct of d)
  r = theta/U (U>0); 0 if U=theta=0; +inf if U=0<theta; c = max over patterns of inverted_cdf 95th pct of r
  coverage uses (theta - X) <= q and r <= c.
Real X-hat and U are read from the v6 primary record (xerr_v6_2026-10-14_161122.json), not the ucal record's G0 copy.
Usage: python3 -I recompute_ucal.py UCAL_JSON V6_JSON
"""

import json
import sys

import numpy as np

ucal = json.load(open(sys.argv[1]))
v6 = json.load(open(sys.argv[2]))["result"]["results"]
R = ucal["result"]
ORIG = [
    "a_null", "b_plus5", "b_minus5", "b_plus2p5", "b_minus2p5",
    "c_hetero_spread5", "c_hetero_spread2p5", "d_single_plus10", "e_sparse_single_plus1p5",
]
ROWS = ["R_reply_meanpooled|alpha=0.01", "R_reply_meanpooled|alpha=0.05", "C_anycommand_max|alpha=0.05"]


def fam(p):
    if p == "a_null":
        return "null"
    if p.startswith(("b_", "h_")):
        return "common"
    if p.startswith(("c_", "i_", "j_", "k_")):
        return "heterogeneous"
    if p.startswith(("d_", "e_", "l_")):
        return "single"
    if p.startswith("m_"):
        return "mixed"
    raise ValueError(p)


def q95(v):
    return float(np.quantile(np.asarray(v, float), 0.95, method="inverted_cdf"))


def ratio(th, U):
    out = np.empty_like(th)
    for k, (t, u) in enumerate(zip(th, U)):
        out[k] = t / u if u > 0 else (0.0 if t == 0 else np.inf)
    return out


def fit(per, pats, sel=None):
    q = max(q95((per[p]["th"] - per[p]["X"])[sel(p) if sel else slice(None)]) for p in pats)
    c = max(q95(ratio(per[p]["th"], per[p]["U"])[sel(p) if sel else slice(None)]) for p in pats)
    return max(0.0, q), c


def cov(per, p, q, c, m=slice(None)):
    th, X, U = per[p]["th"][m], per[p]["X"][m], per[p]["U"][m]
    return float(((th - X) <= q).mean()), float((ratio(th, U) <= c).mean()), len(th)


errs = []
check_counts = 0
total = 0
for rk in ROWS:
    cal = R["setup"]["calibration"][rk]
    pats = list(cal["patterns"])
    el = cal["eligible"]
    P = len(el)
    print(f"\n######## {rk}  eligible={el}  P={P}  real_pool_n={cal['real_pool_n']}  n_patterns={len(pats)}")
    # grid membership
    added = [p for p in pats if p not in ORIG]
    print("  original grid present:", [p for p in ORIG if p in pats] == ORIG, "| added:", added)
    # 5. realised theta per pattern + saturation, recomputed from achieved points
    print("  pattern | target | achieved@T* | theta*(recomp) vs rec | theta*_target(recomp) | sat(recomp) vs rec | room up/down")
    for p in pats:
        cp_ = cal["patterns"][p]
        tv, ach = np.array(cp_["target_pts"]), np.array(cp_["achieved_pts_at_Tstar"])
        th = float(np.sqrt(ach.mean() ** 2 + ach.var(ddof=1))) if P > 1 else abs(ach[0])
        tht = float(np.sqrt(tv.mean() ** 2 + tv.var(ddof=1)))
        sat = [el[j] for j in range(P) if abs(ach[j] - tv[j]) > 0.5]
        room = {k: (round(v["up"], 1), round(v["down"], 1)) for k, v in cp_["catch_room_pts"].items()}
        ok = abs(th - cp_["theta_Tstar"]) < 1e-9 and abs(tht - cp_["theta_Tstar_target"]) < 1e-9 and sat == cp_["saturated_probes"]
        if not ok:
            errs.append(f"{rk} {p} theta/sat mismatch")
        check_counts += 1
        print(f"  {p:30s} tgt={np.round(tv,2).tolist()} ach={np.round(ach,2).tolist()} th*={th:.2f}"
              f" (ws {cp_['theta_Tstar_with_strategy']:.2f}) tgtθ={tht:.2f} sat={sat} {'OK' if ok else 'MISMATCH'}")
    print("  catch room (up, down) pts:", room)
    # pool sizes
    pn = np.array([r["pool_n"] for p in pats for r in R["datasets"][f"{rk}|{p}"]])
    print("  pool_n min/max per probe:", pn.min(0).tolist(), pn.max(0).tolist(), "== real n:",
          bool((pn == np.array(cal["real_pool_n"])).all()))
    if not (pn == np.array(cal["real_pool_n"])).all():
        errs.append(f"{rk} pool n drift")
    for st in ("pooled", "with_strategy"):
        per = {}
        for p in pats:
            rows = R["datasets"][f"{rk}|{p}"]
            idx = [r["i"] for r in rows]
            assert sorted(idx) == list(range(1000)), (rk, p)
            assert all(r[st]["X"] is not None and r[st]["U"] is not None for r in rows)
            total += len(rows) if st == "pooled" else 0
            per[p] = {
                "X": np.array([r[st]["X"] for r in rows]),
                "U": np.array([r[st]["U"] for r in rows]),
                "th": np.array([r[st]["theta"] for r in rows]),
                "even": np.array([r["i"] % 2 == 0 for r in rows]),
            }
        rec = R["calibrated"][f"{rk}|{st}"]
        q_o, c_o = fit(per, ORIG)
        q_e, c_e = fit(per, pats)
        v = v6[rk.replace("|alpha", "|mechanical|alpha")]["primary_eligible_estimation_half"]["sim_minus_mean_real"][st]
        x, u = v["X_err"], v["U_approx_upper_95"]
        print(f"\n  == {st}: q orig {q_o:.4f} ext {q_e:.4f} | c orig {c_o:.4f} ext {c_e:.4f}"
              f"  [rec q {rec['q_original']:.4f}/{rec['q_extended']:.4f} c {rec['c_original']:.4f}/{rec['c_extended']:.4f}]")
        for a, b, nm in ((q_o, rec["q_original"], "q_o"), (q_e, rec["q_extended"], "q_e"), (c_o, rec["c_original"], "c_o"), (c_e, rec["c_extended"], "c_e")):
            if abs(a - b) > 1e-12:
                errs.append(f"{rk}|{st} {nm} {a} vs {b}")
        print(f"     REAL v6: X_hat {x:.4f} U {u:.4f} | U_cal ext {x+q_e:.4f} orig {x+q_o:.4f} | U_s ext {c_e*u:.4f} orig {c_o*u:.4f}"
              f"  [rec U_cal {rec['real_v6']['U_cal_extended']:.4f}/{rec['real_v6']['U_cal_original']:.4f} U_s {rec['real_v6']['U_s_extended']:.4f}/{rec['real_v6']['U_s_original']:.4f}]")
        top = sorted(((q95(per[p]['th'] - per[p]['X']), p) for p in pats), reverse=True)[:3]
        print("     binding (q95 of theta-X):", [(p, round(s, 3)) for s, p in top])
        print("     binding (q95 of ratio):", [(p, round(s, 3)) for s, p in sorted(((q95(ratio(per[p]['th'], per[p]['U'])), p) for p in pats), reverse=True)[:2]])
        print("     U=0 datasets:", sum(int((per[p]["U"] == 0).sum()) for p in pats))
        for gname, grid in (("ext", pats), ("orig", ORIG)):
            qh, ch = fit(per, grid, sel=lambda p: per[p]["even"])
            sp = {p: cov(per, p, qh, ch, ~per[p]["even"]) for p in grid}
            mn_cal = min(sp.items(), key=lambda kv: kv[1][0])
            mn_s = min(sp.items(), key=lambda kv: kv[1][1])
            lopo = {}
            for j in grid:
                qj, cj = fit(per, [p for p in grid if p != j])
                lopo[j] = cov(per, j, qj, cj) + (qj,)
            w_cal = min(lopo.items(), key=lambda kv: kv[1][0])
            w_s = min(lopo.items(), key=lambda kv: kv[1][1])
            lofo = {}
            for f in ("null", "common", "heterogeneous", "single", "mixed"):
                held = [p for p in grid if fam(p) == f]
                rest = [p for p in grid if fam(p) != f]
                if held and rest:
                    qf, cf = fit(per, rest)
                    for p in held:
                        lofo[p] = cov(per, p, qf, cf)
            w_f = min(lofo.items(), key=lambda kv: kv[1][0])
            rv = rec[f"validation_{'extended' if gname == 'ext' else 'original'}_grid"]
            rl = rv["leave_one_pattern_out"]["worst"]
            print(f"     [{gname}] split-half q_even {qh:.3f} c_even {ch:.3f}: min odd U_cal {mn_cal[1][0]:.3f} ({mn_cal[0]}) U_s {mn_s[1][1]:.3f} ({mn_s[0]})"
                  f" [rec {rv['split_half']['min_odd']['U_cal']:.3f}/{rv['split_half']['min_odd']['U_s']:.3f}]")
            print(f"       LOPO worst U_cal {w_cal[1][0]:.3f} ({w_cal[0]}, k={round(w_cal[1][0]*1000)}/1000, q_without {w_cal[1][3]:.3f})"
                  f" U_s {w_s[1][1]:.3f} ({w_s[0]}) [rec {rl['U_cal']:.3f}/{rl['U_s']:.3f}]  LOFO worst U_cal {w_f[1][0]:.3f} ({w_f[0]})")
            if abs(mn_cal[1][0] - rv['split_half']['min_odd']['U_cal']) > 1e-12 or abs(w_cal[1][0] - rl['U_cal']) > 1e-12 or abs(w_s[1][1] - rl['U_s']) > 1e-12:
                errs.append(f"{rk}|{st}|{gname} validation mismatch")
        # 4. X_hat = 0 conditional coverage (theta <= q_e among X == 0)
        rows0 = []
        for p in pats:
            m = per[p]["X"] == 0
            n0 = int(m.sum())
            k0 = int((per[p]["th"][m] <= q_e).sum())
            rows0.append((p, n0, k0, float(per[p]["th"].mean())))
            rz = rec["X_hat_zero_diagnostic"][p]
            if rz["n_X_zero"] != n0 or (n0 and abs(rz["coverage_U_cal_given_X_zero"] - k0 / n0) > 1e-12):
                errs.append(f"{rk}|{st} X0 {p}")
        big = [r for r in rows0 if r[1] and r[2] / r[1] < 0.5]
        print("     X_hat=0: share of datasets with X=0 per pattern / covered:",
              ", ".join(f"{p} {k0}/{n0}" for p, n0, k0, _ in rows0 if n0))
        print("     X_hat=0 patterns with conditional coverage < 0.5:", [(p, f"{k0}/{n0}", f"{k0/n0:.3f}") for p, n0, k0, _ in big])

print("\nTOTAL datasets:", total, "| checks:", check_counts, "| MISMATCHES:", errs if errs else "none")
