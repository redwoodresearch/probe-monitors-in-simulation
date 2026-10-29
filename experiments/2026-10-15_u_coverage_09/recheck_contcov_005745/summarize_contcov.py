"""Print the Amendment 2 tables from a contcov record (formatting only; every number was computed in the Modal job)."""

import json
import sys

r = json.load(open(sys.argv[1]))["result"]
s = r["setup"]
print(
    f"G0 {s['G0']['pass']} max {s['G0']['max_abs_diff']}; G0c {s['G0c']['pass']} max {s['G0c']['max_abs_diff']:.2e}; CP gate {s['cp_gate_max_abs_diff']:.1e}; wall {r['wall_seconds']:.0f} s; datasets {sum(len(v) for v in r['datasets'].values())}"
)
c = r["coverage"]
print("OVERALL:", c["overall"])


def cv(x):
    if x.get("unavailable"):
        return "UNAVAILABLE"
    k = x["coverage"]
    return f"{k['rate']:.3f}[{k['cp95'][0]:.3f}]"


for k, v in c.items():
    if k == "overall":
        continue
    print(f"\n== {k}")
    for est in ("mapped", "binary"):
        sc = v[f"screen_{est}"]
        print(
            f"  {est}: {sc['verdict']} | fail cov {sc.get('patterns_failing_coverage')} | fail cond {sc.get('patterns_failing_conditional')} | "
            f"undercov evidence {sc.get('patterns_with_evidence_of_undercoverage')} | worst loc {v[f'worst_{est}_location']} | worst nonloc {v[f'worst_{est}_non-location']}"
        )
    print(
        "  pattern | theta* | mean theta(That) | share theta=0 | MAPPED cov[CPlo] meanU meanX shareX0 cond(n0) | BINARY cov[CPlo] meanU cond(n0) | sat"
    )
    for p, e in v["per_pattern"].items():
        m, b = e["mapped"], e["binary"]
        mc = m.get("cond_X0", {})
        bc = b.get("cond_X0", {})
        mcs = (
            f"{mc.get('rate', float('nan')):.2f}({mc.get('n0')})"
            if mc.get("rate") is not None
            else f"-({mc.get('n0')})"
        )
        bcs = (
            f"{bc.get('rate', float('nan')):.2f}({bc.get('n0')})"
            if bc.get("rate") is not None
            else f"-({bc.get('n0')})"
        )
        print(
            f"  {p:30s} {e['theta_Tstar']:5.2f} {e['mean_theta_That']:5.2f} {e['share_theta_That_zero']:.2f} | "
            f"{cv(m)} {m.get('mean_U', float('nan')):5.2f} {m.get('mean_X', float('nan')):5.2f} {m.get('share_X0', float('nan')):.2f} {mcs} | "
            f"{cv(b)} {b.get('mean_U', float('nan')):5.2f} {bcs} | {e['saturated_probes'] or ''}"
        )
