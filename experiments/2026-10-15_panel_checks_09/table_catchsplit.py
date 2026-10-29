"""Print the panel check (b) table from results/catchsplit_v6_*.json (by script): primary (mechanical) rows only."""

import json
import sys

r = json.load(open(sys.argv[1]))["result"]  # noqa: SIM115
print("gate v6_primary_equal:", r["gates"]["v6_primary_equal"])
f = lambda x: "None" if x is None else f"{x:.2f}"  # noqa: E731
for rk, row in r["results"].items():
    if "|mechanical|" not in rk:
        continue
    sp = row["catch_level_split_posthoc"]
    print(f"== {rk}")
    for gname, g in sp["groups"].items():
        if "note" in g:
            print(f"  {gname}: {g['note']}")
            continue
        cats = "; ".join(
            f"{p} real {c['run1']:.1f}/{c['run2']:.1f}/{c['real_mean']:.1f} sim {c['sim']:.1f}"
            for p, c in g["catch_percent_estimation_half"].items()
        )
        print(f"  {gname}: members {g['members']} | {cats}")
        for st in ("pooled", "with_strategy"):
            b = g["sim_minus_mean_real"][st]
            rb = g["reexecution_run2_minus_run1"][st]
            if b.get("X_err") is not None:
                print(f"    {st}: X_err {f(b['X_err'])} U {f(b['U_approx_upper_95'])} n={b['n_tasks']} | rerun U {f(rb.get('U_approx_upper_95'))}")
            else:
                pp = b.get("per_probe", {})
                for p, v in (pp.items() if isinstance(pp, dict) else []):
                    print(f"    {st} {p}: delta {f(v['delta'])} ci95 [{f(v['ci95'][0])},{f(v['ci95'][1])}] U_abs {f(v['U_abs_approx'])} ({b.get('note','')})")
        print(f"    strategy pooling (post hoc): {g['strategy_pooling_test_tost5_posthoc'].get('pooled_label')}")
