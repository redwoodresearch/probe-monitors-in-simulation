"""Print the coverage table from a ucov record (by script; numbers only)."""

import json
import sys

r = json.load(open(sys.argv[1]))["result"]
nd = sum(len(v) for v in r["datasets"].values())
nn = sum(1 for v in r["datasets"].values() for x in v if x["U"] is None)
print(f"datasets {nd}, undefined {nn}, wall {r['wall_seconds']:.1f} s, G0 max diff {r['setup']['G0']['max_abs_diff']}")
for k, s in r["summary"].items():
    p, q = s["coverage_primary_theta_That"], s["coverage_secondary_theta_Tstar"]
    x, u = s["X_err"], s["U"]
    th = s["theta_That"]
    rd = s["reading"] if isinstance(s["reading"], str) else s["reading"]["coverage_primary_theta_That"]
    print(
        f"{k}: theta*={s['theta_Tstar']:.2f} thetaHat_mean={th['mean']:.2f} | cov(thetaHat) {p['rate']:.3f} "
        f"[{p['cp95'][0]:.3f},{p['cp95'][1]:.3f}] | cov(theta*) {q['rate']:.3f} [{q['cp95'][0]:.3f},{q['cp95'][1]:.3f}] | "
        f"Xerr mean {x['mean']:.2f} med {x['median']:.2f} zero {x['share_zero']:.2f} bias(That) {x['mean_bias_vs_theta_That']:+.2f} | "
        f"U mean {u['mean']:.2f} p95 {u['p95']:.2f} | {rd}"
    )
