"""Table for the attempt-filter sensitivity record (by script; reads the committed JSON only)."""

import json
import sys

r = json.load(open(sys.argv[1]))["result"]
V = {"base": "attempted", "(a)": "a_unclear_not_caught", "(b)": "b_all_as_scored"}
f1 = lambda x: "  -  " if x is None else f"{x:5.2f}"  # noqa: E731
for name, v in V.items():
    ac = r[v]["added_counts"]
    print(
        f"{name}: tasks est {ac['tasks_estimation']} (with {ac['strata_estimation']['with_strategy']}/without "
        f"{ac['strata_estimation']['without_strategy']}), all {ac['tasks_all']}; added attack runs est "
        f"{ac['added_attack_runs_estimation']} / all {ac['added_attack_runs_all']}; unclear runs est "
        f"{ac['unclear_runs_estimation']} / all {ac['unclear_runs_all']}"
    )
print("missing-score exclusions:", sorted(r["attempted"]["cohort"]["missing_score_exclusions"]))
for row in ("R_reply_meanpooled", "C_anycommand_max"):
    for a in ("0.01", "0.05"):
        key = f"{row}|mechanical|alpha={a}"
        print(f"\n== {key}")
        for st in ("with_strategy", "without_strategy", "pooled"):
            line = f"  {st:17}"
            for name, v in V.items():
                b = r[v]["results"][key]["primary_eligible_estimation_half"]["sim_minus_mean_real"].get(st) or {}
                line += f" | {name} n={b.get('n_tasks')} X_err {f1(b.get('X_err'))} U {f1(b.get('U_approx_upper_95'))}"
            print(line)
        elig = r["attempted"]["results"][key]["eligibility_append13_4"]["eligible"]
        for p in elig:
            for st in ("with_strategy", "pooled"):
                line = f"   {p:11} {st:13}"
                for name, v in V.items():
                    res = r[v]["results"][key]
                    pp = (
                        (res["primary_eligible_estimation_half"]["sim_minus_mean_real"].get(st) or {})
                        .get("per_probe", {})
                        .get(p, {})
                    )
                    cs = res["catches_by_stratum_percent"][p][f"estimation_half|{'all' if st == 'pooled' else st}"]
                    line += (
                        f" | {name} Δ {f1(pp.get('delta'))} U_abs {f1(pp.get('U_abs_approx'))}"
                        f" real {f1(cs['mean_real'])} (r1 {f1(cs['run1'])} r2 {f1(cs['run2'])}) sim {f1(cs['sim'])}"
                    )
                print(line)
