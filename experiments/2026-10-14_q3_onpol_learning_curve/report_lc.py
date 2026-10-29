"""Print the learning curve's stored fields (no recomputation): python3 -I report_lc.py <lc_*.json>"""
import json
import sys

d = json.load(open(sys.argv[1]))
pc = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}"  # noqa: E731
iv = lambda i: f"[{pc(i['ci95'][0]).strip()}, {pc(i['ci95'][1]).strip()}]" + (" INVALID" if i.get("invalid") else "")  # noqa: E731
print(f"git {d['git']} stamp {d['stamp']} layer {d['layer']}")
print("gate (frozen catches, on-policy convention):", json.dumps({r: {k: v["abs_diff"] for k, v in g.items()} for r, g in d["gate_frozen_catch_reproduced"].items()}))
for f in d["fits"]:
    print(f"fit {f['reader']}/{f['base']}: rows {f['rows']} sol {f['sol_runs']} fits {f['fits']} (>=1,000 evals {f['fits with >= 1,000 evaluations']}) "
          f"dtypes {f.get('feature_dtypes')} diag {f['diagnostic']} {f['seconds']} s")
for s in d["stats"]:
    c = s["curve"]
    print(f"\n== {s['reader']} base {s['base']} (cohort {c['cohort_tasks']} tasks; catch = second real run; threshold = pooled honest A)")
    for fp in ("0.05", "0.01"):
        for g in ("k0h", "k10", "k30", "k100", "k100noh", "S_k10", "S_k30", "S_k100", "S_k100noh"):
            o = c[g][f"at_{fp}"]
            line = (f"{fp:>4} {g:10} catch {pc(o['catch_mean_over_draws'])} (k0 {pc(o['catch_k0'])}) minus k0 {pc(o['minus_k0'])} {iv(o['minus_k0_interval'])}")
            if "share_of_gap_closed" in o:
                line += f" | on-policy {pc(o['catch_onp'])} share closed {o['share_of_gap_closed']:.2f} [{o['share_of_gap_closed_interval']['ci95'][0]:.2f}, {o['share_of_gap_closed_interval']['ci95'][1]:.2f}]"
            if "S_minus_R" in o:
                line += f" | S - R {pc(o['S_minus_R'])} {iv(o['S_minus_R_interval'])}"
            print(line)
        for k, v in c.get("k0_vs_frozen", {}).items():
            if k.endswith(f"at_{fp}"):
                print(f"{fp:>4} k0 vs frozen ensemble (same convention): {json.dumps({a: round(100 * b, 2) for a, b in v.items()})}")
    fa = s["false_alarms_honest_b_per_column"]
    print("honest-B false alarms per column at 5% / 1%: " + "; ".join(f"{cn} {pc(v['at_0.05']['false_alarm_honest_b']).strip()}/{pc(v['at_0.01']['false_alarm_honest_b']).strip()}" for cn, v in fa.items()))
