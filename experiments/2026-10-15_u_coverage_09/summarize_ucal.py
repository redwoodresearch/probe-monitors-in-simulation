"""Print the U_cal tables from a ucal record (formatting only; every number was computed in the Modal job)."""

import json
import sys

r = json.load(open(sys.argv[1]))["result"]
g = r["setup"]["G0"]
print(
    f"G0 pass {g['pass']} max|diff| {g['max_abs_diff']}; CP gate {g['cp_gate_max_abs_diff']:.2e}; wall {r['wall_seconds']:.0f} s"
)
nd = sum(len(v) for v in r["datasets"].values())
print(f"datasets {nd}")


def f(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)


for k, v in r["calibrated"].items():
    re_ = v["real_v6"]
    ve, vo = v["validation_extended_grid"], v["validation_original_grid"]
    print(f"\n== {k}  ({len(v['patterns'])} patterns)")
    print(
        f"  q orig {f(v['q_original'])}  q ext {f(v['q_extended'])} | c orig {f(v['c_original'])}  c ext {f(v['c_extended'])}"
    )
    if re_:
        print(
            f"  REAL v6: X_hat {f(re_['X_hat'])}  U {f(re_['U'])} [{re_['frame_U']}] | U_cal ext {f(re_['U_cal_extended'])} "
            f"[{re_['frame_U_cal_extended']}]  U_cal orig {f(re_['U_cal_original'])} | U_s ext {f(re_['U_s_extended'])} "
            f"[{re_['frame_U_s_extended']}]  U_s orig {f(re_['U_s_original'])}"
        )
    for nm, vv in (("ext", ve), ("orig", vo)):
        sh, lo = vv["split_half"], vv["leave_one_pattern_out"]
        fam = vv["leave_one_family_out"]
        wf = min(
            (pp["U_cal"], fn, pn)
            for fn, fv in fam.items()
            for pn, pp in fv["per_pattern"].items()
            if pp["U_cal"] is not None
        )
        wlo = min((pp["U_cal"], pn) for pn, pp in lo["per_pattern"].items() if pp["U_cal"] is not None)
        print(
            f"  [{nm}] split-half: q_even {f(sh['q_even'])}, min odd cov U_cal {f(sh['min_odd']['U_cal'])} U_s {f(sh['min_odd']['U_s'])}"
            f" | LOPO worst U_cal {f(lo['worst']['U_cal'])} ({wlo[1]}) U_s {f(lo['worst']['U_s'])} | LOFO worst U_cal {f(wf[0])}"
            f" ({wf[1]}: {wf[2]}) | {lo['diagnostic'][:40]}"
        )
    rt = v["realized_truth"]
    sat = {p: x["saturated_probes"] for p, x in rt.items() if x["saturated_probes"]}
    top = sorted(v["per_pattern_q95_theta_minus_X"].items(), key=lambda kv: -kv[1])[:3]
    print(
        "  binding patterns (q95 of theta - X):",
        ", ".join(f"{p} {x:.2f} (theta* {rt[p]['theta_Tstar']:.2f})" for p, x in top),
    )
    print("  saturated:", sat if sat else "none")
    z = v["X_hat_zero_diagnostic"]
    zz = [(p, d["n_X_zero"], d["coverage_U_cal_given_X_zero"]) for p, d in z.items() if d["n_X_zero"]]
    worstz = min((c, p, n) for p, n, c in zz if c is not None) if zz else None
    print(f"  X_hat=0 diagnostic: worst coverage given X_hat=0 {worstz}")
    print("  pool n drift:", {p: (d["real_n"], d["min"], d["max"]) for p, d in v["pool_n_drift"].items()})
    print("  U=0 datasets:", sum(v["U_zero_datasets"].values()))
