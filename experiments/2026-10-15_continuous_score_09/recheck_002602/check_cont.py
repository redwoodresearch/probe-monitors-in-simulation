"""Independent checker for experiments/2026-10-15_continuous_score_09 (branch aswin/continuous-score-09 @ 835acb130).
Reads records only (scores/summaries; no transcript text). Pure Python, no bootstrap.
usage: python3 -I check_cont.py <dir with src/ copies> <v6 ref record> <v7 phase2 record> <v7 phase1 record>"""

import hashlib
import json
import math
import subprocess
import sys

D, V6REF, P2, P1 = sys.argv[1:5]
REPO = "/Users/aswinsurya/probe-distribution-shift-from-sims"
EXP = "experiments/2026-10-15_continuous_score_09"
CONT6 = f"{D}/src/xerr_cont_v6_2026-10-15_002602.json"
CONT7 = f"{D}/src/v7_cont_2026-10-15_002602.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def ok(b):
    return "MATCH" if b else "MISMATCH"


print("=== 0. hashes")
for p in (CONT6, CONT7, V6REF, P2, P1):
    print(p.split("/")[-1], sha(p)[:16])

print("\n=== 1. RULE timing")
for c in ("30704b56e", "18301a844", "835acb130"):
    print(c, git("show", "-s", "--format=%cd %s", "--date=iso", c)[:110])
print("RULE.md commits:", git("log", "--format=%h %cd", "--date=iso", "835acb130", "--", f"{EXP}/RULE.md"))
print("RULE.md diff 30704b56e..835acb130 empty:", git("diff", "30704b56e", "835acb130", "--", f"{EXP}/RULE.md") == "")
for p in ("results/xerr_cont_v6_2026-10-15_002602.json", "results/v7_cont_2026-10-15_002602.json"):
    print(p, "first commit:", git("log", "--diff-filter=A", "--format=%h %cd", "--date=iso", "835acb130", "--", f"{EXP}/{p}"))
print("RULE commit is ancestor of record code commit 18301a844:",
      subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", "30704b56e", "18301a844"]).returncode == 0)
c6, c7 = json.load(open(CONT6)), json.load(open(CONT7))
print("record git/stamp:", c6["git"][:9], c6["stamp"], "|", c7["git"][:9], c7["stamp"])
rule = git("show", f"835acb130:{EXP}/RULE.md")
for term in ["σ_pool", "D_p = the weighted mean", "Append 19's formula on the D_p matrix", "f_real(T') · D_p", "0.9 · min(SD, IQR/1.34) · n^(−1/5)",
             "held fixed across draws", "M_p,d = 100 · f̂_d(T') · D_p,d", "Mapping-discrepancy diagnostic", "first-order KDE mapping disagrees",
             "Width ratios", "own_frac diagnostic reading", "concordant point-estimate ordering", "pool B while own_frac100 uses B∪C",
             "Codex round 1", "round 2", "within 1e-12", "0.4 ± 0.2"]:
    print(f"  RULE contains {term!r}: {term in rule}")

print("\n=== 2a. gate 1: v6 mechanical rows vs v6 reference (all fields except the added ones)")
ref = json.load(open(V6REF))["result"]["results"]
rows6 = c6["result"]["results"]
maxdiff = [0.0]
mism = []


def cmp(a, b, path):
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            if math.isnan(a) and math.isnan(b):
                return
            d = abs(a - b)
            maxdiff[0] = max(maxdiff[0], d)
            if d > 1e-12:
                mism.append((path, a, b))
            return
        mism.append((path, a, b)); return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            mism.append((path, "keys", sorted(set(a) ^ set(b))))
        for k in set(a) & set(b):
            cmp(a[k], b[k], path + "/" + k)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            mism.append((path, "len", len(a), len(b))); return
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, f"{path}[{i}]")
        return
    if a != b:
        mism.append((path, a, b))


MECH = ["R_reply_meanpooled|mechanical|alpha=0.01", "R_reply_meanpooled|mechanical|alpha=0.05",
        "C_anycommand_max|mechanical|alpha=0.01", "C_anycommand_max|mechanical|alpha=0.05"]
print("rows in cont record == the four mechanical rows:", sorted(rows6) == sorted(MECH))
for r in MECH:
    a = {k: v for k, v in rows6[r].items() if k not in ("continuous", "gate1_reproduces_v6_primary_row")}
    cmp(a, ref[r], r)
print(f"gate 1 field mismatches: {len(mism)}; max |float diff| {maxdiff[0]:.3g}; record says {c6['result']['gates']['gate1_max_abs_float_diff']:.3g}")
for m in mism[:5]:
    print("  ", m)

print("\n=== 2b. binary family + per-probe Δ in the continuous block vs v6 sim_minus_mean_real")
FROZEN = {"R_reply_meanpooled|mechanical|alpha=0.01": ["sol_a", "fibs", "harmfulness"],
          "R_reply_meanpooled|mechanical|alpha=0.05": ["sol_a", "fibs", "harmfulness", "refusal"],
          "C_anycommand_max|mechanical|alpha=0.01": ["fibs"], "C_anycommand_max|mechanical|alpha=0.05": ["sol_a", "fibs"]}
bad = 0
for r in MECH:
    for st in ("pooled", "with_strategy", "without_strategy"):
        b = rows6[r]["continuous"][st]
        v = ref[r]["primary_eligible_estimation_half"]["sim_minus_mean_real"][st]
        if sorted(b["per_probe"]) != sorted(FROZEN[r]):
            print("  eligible set differs", r, st, list(b["per_probe"]))
            bad += 1
        fb = b["family_binary_points"]
        for k in ("X_err", "U_approx_upper_95", "bias_mean"):
            if fb.get(k) != v.get(k):
                print("  family mismatch", r, st, k, fb.get(k), v.get(k)); bad += 1
        for p in FROZEN[r]:
            bd, vp = b["per_probe"][p]["binary_delta_points"], v["per_probe"][p]
            if bd["point"] != vp["delta"] or bd["ci95"] != vp["ci95"]:
                print("  Δ mismatch", r, st, p, bd, vp["delta"], vp["ci95"]); bad += 1
        if r.startswith("R") and r.endswith("0.05") and st == "with_strategy":
            print(f"  R5% with_strategy: v6 X_err {v['X_err']:.4f} U {v['U_approx_upper_95']:.4f} | cont X_err {fb['X_err']:.4f} U {fb['U_approx_upper_95']:.4f}")
print("binary family/Δ/eligibility mismatches:", bad, "| gate2 flags:",
      {f"{r.split('|')[0][0]}{r[-4:]}|{st}": rows6[r]["continuous"][st]["gate2_binary_family_reproduced"] for r in MECH for st in ("pooled", "with_strategy")})

print("\n=== 2c. v7 gates")
g7 = c7["result"]["gates"]
print("V7 self-report:", g7["V7_all_cohort_catches_equal_v7_phase2"])
print("O self-report:", g7["O_own_frac100_L16_R5_estimation_catches"])
cat = c7["result"]["rows"]["own_frac100_L16|R|alpha=0.05"]["pooled"]["catches_pct"]
print("O from row, rounded:", {k: round(v, 1) for k, v in cat.items()}, ok({k: round(v, 1) for k, v in cat.items()} == {"run1": 80.0, "run2": 84.4, "sim": 70.0}))
print("fixtures:", all(g7["selftest_fixtures"].values()), len(g7["selftest_fixtures"]))
# all-cohort reference catch count in phase 2
p2 = json.load(open(P2))
n = 0
for rk in ("R|0.01", "R|0.05", "C|0.01", "C|0.05"):
    d = p2["result"]["results"][rk]["descriptive_all_variants_all_cohort"]
    n += sum(3 for k, v in d.items() if k.startswith("own_frac") and v.get("status") == "available")
print("phase-2 own_frac all-cohort catches available:", n)
# independent: v7 cont binary Δ point+CI vs phase-2 primary per-probe (eligible variants), pooled stratum
nm = nc = 0
for rk, alpha in (("R|0.01", "0.01"), ("R|0.05", "0.05"), ("C|0.01", "0.01"), ("C|0.05", "0.05")):
    pp = p2["result"]["results"][rk]["sensitivity_b_each_variant_own_member"]["sim_minus_mean_real"]["pooled"]["per_probe"]
    ro = rk[0]
    for var, vv in pp.items():
        if not var.startswith("own_frac"):
            continue
        row = c7["result"]["rows"].get(f"{var}|{ro}|alpha={alpha}")
        bd = row["pooled"]["binary_delta_points"]
        nc += 1
        if abs(bd["point"] - vv["delta"]) > 1e-9 or any(abs(x - y) > 1e-9 for x, y in zip(bd["ci95"], vv["ci95"])):
            nm += 1; print("  v7 Δ mismatch", var, rk, bd, vv)
print(f"v7 binary Δ (point+CI) vs phase-2 sensitivity_b per-variant (eligible own_frac variants, pooled, sim-minus-mean-real): {nc} compared, {nm} mismatches")
spread = p2["result"]["results"]["R|0.05"]["within_recipe_spread"]["own_a_L16"]
print("phase-2 own_a_L16 R5% Δ:", dict(zip(spread["variants"], [round(x, 2) for x in spread["delta"]])))

print("\n=== 3. recompute width ratios / U ratio / bias mean / M=100 f D from the v6 continuous record")
W = lambda ci: ci[1] - ci[0]
for r in MECH:
    for st in ("pooled", "with_strategy", "without_strategy"):
        b = rows6[r]["continuous"][st]
        ratios = []
        for p, s in b["per_probe"].items():
            wb = W(s["binary_delta_points"]["ci95"])
            wm = W(s["mapped_points"]["ci95"]) if s["mapped_points"]["ci95"] else None
            rr = (wm / wb) if (wm is not None and wb > 0) else None
            ratios.append(rr)
            rec = s["width_ratio_mapped_over_binary"]
            if (rr is None) != (rec is None) or (rr is not None and abs(rr - rec) > 1e-12):
                print("  per-probe ratio mismatch", r, st, p, rr, rec)
            Mchk = 100 * s["density_at_T_std"]["point"] * s["D_std_units"]["point"]
            if abs(Mchk - s["mapped_points"]["point"]) > 1e-9:
                print("  M != 100 f D", r, st, p, Mchk, s["mapped_points"]["point"])
        prim = None if any(x is None for x in ratios) else sum(ratios) / len(ratios)
        rec = b["width_ratio_primary_mean_over_eligible"]
        fb, fm = b["family_binary_points"], b["family_mapped_points"]
        Ur = (fm["U_approx_upper_95"] / fb["U_approx_upper_95"]) if fb.get("U_approx_upper_95") else None
        bm = sum(s["mapped_points"]["point"] for s in b["per_probe"].values()) / len(b["per_probe"])
        bmok = fm.get("bias_mean") is None or abs(bm - fm["bias_mean"]) < 1e-9
        tag = f"{r.split('_')[0]} {r[-4:]} {st:16}"
        print(f"{tag} width ratio mine {prim if prim is None else round(prim, 4)} rec {rec if rec is None else round(rec, 4)} {ok((prim is None and rec is None) or (prim is not None and rec is not None and abs(prim - rec) < 1e-12))}"
              f" | bin X_err {fb.get('X_err') if fb.get('X_err') is None else round(fb['X_err'], 2)} U {fb.get('U_approx_upper_95') if fb.get('U_approx_upper_95') is None else round(fb['U_approx_upper_95'], 2)}"
              f" | map X_err {fm.get('X_err') if fm.get('X_err') is None else round(fm['X_err'], 2)} U {fm.get('U_approx_upper_95') if fm.get('U_approx_upper_95') is None else round(fm['U_approx_upper_95'], 2)}"
              f" | U ratio mine {Ur if Ur is None else round(Ur, 3)} rec {b['U_ratio_mapped_over_binary'] if b['U_ratio_mapped_over_binary'] is None else round(b['U_ratio_mapped_over_binary'], 3)} | map bias_mean = mean(M) {bmok}")
        if prim is None:
            print("     undefined constituents:", {p: (s["width_ratio_mapped_over_binary"], s["width_ratio_note"]) for p, s in b["per_probe"].items()})
s = rows6["R_reply_meanpooled|mechanical|alpha=0.05"]["continuous"]
for st in ("with_strategy", "pooled"):
    q = s[st]["per_probe"]["sol_a"]
    print(f"Sol (a) R5% {st}: D {q['D_std_units']['point']:+.3f} [{q['D_std_units']['ci95'][0]:+.3f}, {q['D_std_units']['ci95'][1]:+.3f}] "
          f"M {q['mapped_points']['point']:+.1f} [{q['mapped_points']['ci95'][0]:+.1f}, {q['mapped_points']['ci95'][1]:+.1f}] "
          f"binary {q['binary_delta_points']['point']:+.1f} [{q['binary_delta_points']['ci95'][0]:+.1f}, {q['binary_delta_points']['ci95'][1]:+.1f}] "
          f"f(T') {q['density_at_T_std']['point']:.4f} h {q['descriptives']['bandwidth']:.4f} F_real(T') {q['descriptives']['F_real_at_T']:.3f}"
          f" | binary CI unrounded {q['binary_delta_points']['ci95']}")

print("\n=== 4. v7 own_frac primary cell")
dg = c7["result"]["own_frac_diagnostic"]["primary"]
print("cell", dg["cell"], "| reading:", dg["reading"])
rowsD = {f: c7["result"]["rows"][f"own_frac{f}_L16|R|alpha=0.05"] for f in ("10", "25", "100")}
for f, rv in rowsD.items():
    pp = rv["pooled"]
    print(f"frac{f}: pool {rv['pool']} | D {pp['D_std_units']['point']:+.3f} {[round(x, 3) for x in pp['D_std_units']['ci95']]} | Δ {pp['binary_delta_points']['point']:+.1f} "
          f"{[round(x, 1) for x in pp['binary_delta_points']['ci95']]} | catches {[round(pp['catches_pct'][a], 1) for a in ('run1', 'run2', 'sim')]} "
          f"| diag D {dg['D_points'][f]:+.3f} Δ {dg['binary_points'][f]:+.2f} pool {dg['pools'][f]} | consistent {abs(dg['D_points'][f] - pp['D_std_units']['point']) < 1e-12 and abs(dg['binary_points'][f] - pp['binary_delta_points']['point']) < 1e-12}")
for pk, pv in dg["pairs"].items():
    Dx = pv["D_diff_ci95"][0] > 0 or pv["D_diff_ci95"][1] < 0
    Bx = pv["binary_diff_ci95_points"][0] > 0 or pv["binary_diff_ci95_points"][1] < 0
    print(f"  {pk}: D diff {[round(x, 3) for x in pv['D_diff_ci95']]} excl0 mine {Dx} rec {pv['D_diff_excludes_0']} | Δ diff {[round(x, 2) for x in pv['binary_diff_ci95_points']]} excl0 mine {Bx} rec {pv['binary_diff_excludes_0']} | n {pv['D_jointly_finite_draws']}/{pv['binary_jointly_finite_draws']}")
# reimplement the rule's reading
Dp, Bp = dg["D_points"], dg["binary_points"]
anyD = any(pv["D_diff_ci95"][0] > 0 or pv["D_diff_ci95"][1] < 0 for pv in dg["pairs"].values())
anyB = any(pv["binary_diff_ci95_points"][0] > 0 or pv["binary_diff_ci95_points"][1] < 0 for pv in dg["pairs"].values())
ties = len(set(Dp.values())) < 3 or len(set(Bp.values())) < 3
same_order = (not ties) and sorted(Dp, key=Dp.get) == sorted(Bp, key=Bp.get)
if not anyD and anyB:
    mine = "binary-gap differences detected; standardized-mean differences not detected"
elif anyD and same_order:
    mine = "standardized-mean differences detected, with concordant point-estimate ordering"
else:
    mine = "unresolved"
print("reading mine:", mine, "|", ok(mine == dg["reading"]), "| D order", sorted(Dp, key=Dp.get), "Δ order", sorted(Bp, key=Bp.get))
# descriptive readings recomputed the same way
mis = 0
for ck, cv in c7["result"]["own_frac_diagnostic"]["descriptive"].items():
    if cv.get("reading") == "unavailable":
        continue
    Dp2, Bp2 = cv["D_points"], cv["binary_points"]
    aD = any(p["D_diff_ci95"][0] > 0 or p["D_diff_ci95"][1] < 0 for p in cv["pairs"].values())
    aB = any(p["binary_diff_ci95_points"][0] > 0 or p["binary_diff_ci95_points"][1] < 0 for p in cv["pairs"].values())
    t2 = len(set(Dp2.values())) < 3 or len(set(Bp2.values())) < 3
    so = (not t2) and sorted(Dp2, key=Dp2.get) == sorted(Bp2, key=Bp2.get)
    m2 = ("binary-gap differences detected; standardized-mean differences not detected" if (not aD and aB)
          else "standardized-mean differences detected, with concordant point-estimate ordering" if (aD and so) else "unresolved")
    mis += m2 != cv["reading"]
print("descriptive cells:", len(c7["result"]["own_frac_diagnostic"]["descriptive"]), "reading mismatches:", mis)
# pools from phase-1 calibration
p1 = json.load(open(P1))["result"]["phase1_eligibility_record"]["decisions"]
for f in ("10", "25", "100"):
    dd = p1[f"own_frac{f}_L16|R"]
    sds = dd["moments"]["sd"]
    print(f"phase1 own_frac{f}_L16|R pool {dd['pool']} | C-sameness AUROC {dd['sameness']['auroc']:.3f} {[round(x, 3) for x in dd['sameness']['ci95']]} C_pooled {dd['sameness']['C_pooled']} "
          f"| 5% thr {dd['0.05']['threshold']:.4f} n {dd['0.05']['n']} k {dd['0.05']['k']} thin {dd['0.05']['thin_pool_k_lt_10']} | per-fold n {list(dd['moments']['n'].values())}")
print("v7 cont T_std vs phase1 5% threshold (frac100):", rowsD["100"]["T_std"], p1["own_frac100_L16|R"]["0.05"]["threshold"])

print("\n=== 5. mapping-discrepancy (shift-model) flags")
EXC = lambda ci: ci is not None and (ci[0] > 0 or ci[1] < 0)
FLAG = "first-order KDE mapping disagrees with the binary estimate"
flags, n_cells, incons = [], 0, 0
for r in MECH:
    for st in ("pooled", "with_strategy", "without_strategy"):
        for p, s in rows6[r]["continuous"][st]["per_probe"].items():
            for kind in ("mapping_discrepancy", "reexecution_mapping_discrepancy"):
                md = s[kind]; n_cells += 1
                mine = EXC(md["ci95_points"])
                rec = md["flag"] == FLAG
                incons += mine != rec
                if mine or rec:
                    flags.append(("v6", r.split("|")[0][:1] + r[-4:], st, p, kind[:4], round(md["draw0_points"], 2), [round(x, 2) for x in md["ci95_points"]], md["flag"]))
for k, rv in c7["result"]["rows"].items():
    for st in ("pooled", "with_strategy", "without_strategy"):
        s = rv.get(st)
        if not s or "mapping_discrepancy" not in s:
            continue
        for kind in ("mapping_discrepancy", "reexecution_mapping_discrepancy"):
            md = s[kind]; n_cells += 1
            mine = EXC(md["ci95_points"])
            rec = md["flag"] == FLAG
            incons += mine != rec
            if mine or rec:
                flags.append(("v7", k, st, "-", kind[:4], round(md["draw0_points"], 2), [round(x, 2) for x in md["ci95_points"]], md["flag"]))
print(f"cells checked {n_cells}; flagged {len(flags)}; my-flag vs recorded-flag inconsistencies {incons}")
other = sorted({md for md in [] })
for fl in flags:
    print("  ", fl)
allflags = set()
for r in MECH:
    for st in ("pooled", "with_strategy", "without_strategy"):
        for s in rows6[r]["continuous"][st]["per_probe"].values():
            allflags.add(s["mapping_discrepancy"]["flag"]); allflags.add(s["reexecution_mapping_discrepancy"]["flag"])
for rv in c7["result"]["rows"].values():
    for st in ("pooled", "with_strategy", "without_strategy"):
        if rv.get(st) and "mapping_discrepancy" in rv[st]:
            allflags.add(rv[st]["mapping_discrepancy"]["flag"]); allflags.add(rv[st]["reexecution_mapping_discrepancy"]["flag"])
print("distinct flag strings:", allflags)
