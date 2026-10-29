"""Tables from the continuous-score records (scores only): per row x stratum, binary vs mapped per probe, width ratios,
diagnostics; v7 own_frac rows with catches and the diagnostic reading. Prints values from the records, nothing recomputed."""

import json
import sys


def f(x, k=1.0, nd=2):
    return "None" if x is None else f"{k * x:+.{nd}f}"


def c(ci, nd=2):
    return "None" if not ci else f"[{ci[0]:+.{nd}f},{ci[1]:+.{nd}f}]"


v6 = json.load(open(sys.argv[1]))["result"]["results"]
for row, rv in v6.items():
    cont = rv["continuous"]
    if "pooled" not in cont:
        print(row, cont)
        continue
    for st in ("pooled", "with_strategy", "without_strategy"):
        b = cont[st]
        print(f"== v6 {row} | {st} (n={b['n_tasks']}) primary width ratio {f(b['width_ratio_primary_mean_over_eligible'])} "
              f"bias-ratio {f(b['bias_ci_width_ratio'])} U-ratio {f(b['U_ratio_mapped_over_binary'])}")
        fb, fm, fc = b["family_binary_points"], b["family_mapped_points"], b["family_continuous_std_units"]
        print(f"   family binary X_err {f(fb.get('X_err'))} U {f(fb.get('U_approx_upper_95'))} | mapped X_err {f(fm.get('X_err'))} "
              f"U {f(fm.get('U_approx_upper_95'))} | continuous(SD) X_err {f(fc.get('X_err'),nd=3)} U {f(fc.get('U_approx_upper_95'),nd=3)}")
        for p, s in b["per_probe"].items():
            cat = b["catches_pct_per_arm"][p]
            md = s["mapping_discrepancy"]
            print(f"   {p:12} catch r1/r2/sim {cat['run1']:.1f}/{cat['run2']:.1f}/{cat['sim']:.1f} | Δ {f(s['binary_delta_points']['point'])}"
                  f"{c(s['binary_delta_points']['ci95'])} | D {f(s['D_std_units']['point'],nd=3)}{c(s['D_std_units']['ci95'],3)} "
                  f"| f(T') {f(s['density_at_T_std']['point'],nd=3)} | M {f(s['mapped_points']['point'])}{c(s['mapped_points']['ci95'])} "
                  f"| N {f(s['nonlinear_mapped_points']['point'])} | wr {f(s['width_ratio_mapped_over_binary'])} | disc {md['flag']} "
                  f"{c(md['ci95_points'])} | F(T') {f(s['descriptives']['F_real_at_T'],nd=2)}")
if len(sys.argv) > 2:
    v7 = json.load(open(sys.argv[2]))["result"]
    print("\n== v7 own_frac (pooled stratum)")
    for k, rv in v7["rows"].items():
        if "pooled" not in rv:
            print(k, rv)
            continue
        s = rv["pooled"]
        cat = s["catches_pct"]
        print(f"   {k:28} pool {rv['pool']:3} catch {cat['run1']:.1f}/{cat['run2']:.1f}/{cat['sim']:.1f} | Δ {f(s['binary_delta_points']['point'])}"
              f"{c(s['binary_delta_points']['ci95'])} | D {f(s['D_std_units']['point'],nd=3)}{c(s['D_std_units']['ci95'],3)} "
              f"| M {f(s['mapped_points']['point'])}{c(s['mapped_points']['ci95'])} | wr {f(s['width_ratio_mapped_over_binary'])} "
              f"| disc {s['mapping_discrepancy']['flag']}")
    d = v7["own_frac_diagnostic"]["primary"]
    print("\n== own_frac diagnostic, primary cell", d.get("cell"), "->", d["reading"])
    print("   D points", {k: round(v, 3) for k, v in (d.get("D_points") or {}).items()},
          "binary points", {k: round(v, 2) for k, v in (d.get("binary_points") or {}).items()}, "pools", d.get("pools"))
    for pk, pv in (d.get("pairs") or {}).items():
        print("  ", pk, "D diff", c(pv["D_diff_ci95"], 3), pv["D_diff_excludes_0"], "| Δ diff", c(pv["binary_diff_ci95_points"]),
              pv["binary_diff_excludes_0"])
    desc = v7["own_frac_diagnostic"]["descriptive"]
    print("   descriptive readings:", {k: v["reading"] for k, v in desc.items()})
