import json,sys
r=json.load(open(sys.argv[1]))['result']
f=lambda x: "—" if x is None else f"{x:.2f}"
for rk,row in r['rows'].items():
    print(f"== {rk}  eligible {row['eligible_frozen']}  [{row['leading_results'][:40]}]")
    for cn in ('sim_minus_mean_real','reexecution_run2_minus_run1'):
        for sc in ('pooled','with_strategy','without_strategy'):
            b=row['comparisons'][cn][sc]; o=b['fixed_threshold_reproduced']; j=b['rt_joint_U_rt']; i=b['rt_independent_sensitivity']
            d=b.get('U_rt_minus_U') or {}
            print(f"  {cn[:9]:9} {sc:16} n={b['n_tasks']:3} X_err {f(o.get('X_err'))} | U {f(o.get('U_approx_upper_95'))} | U_rt(joint) {f(j.get('U_approx_upper_95'))} (Δ {f(d.get('joint'))}) X_err_rt {f(j.get('X_err'))} | U_rt(indep) {f(i.get('U_approx_upper_95'))} (Δ {f(d.get('independent'))}) | complete j/i {j.get('complete_draws')}/{i.get('complete_draws')}")
            if cn=='sim_minus_mean_real' and 'catch_percent' in b:
                fm=b['catch_percent']['family_mean']
                print(f"      family catch %: run1 {f(fm['run1']['point'])} run2 {f(fm['run2']['point'])} real {f(fm['real_mean']['point'])} sim {f(fm['sim']['point'])}  (real 95% fixed {fm['real_mean']['ci95_fixed_threshold'] and [round(x,1) for x in fm['real_mean']['ci95_fixed_threshold']]} / rt {fm['real_mean']['ci95_threshold_resampled_joint'] and [round(x,1) for x in fm['real_mean']['ci95_threshold_resampled_joint']]})")
    for p,inf in row['per_probe_threshold_info'].items():
        print(f"    {p:11} T0 {inf['threshold_T0']:.4f} sd(j/i) {inf['threshold_sd_over_draws']['joint']:.4f}/{inf['threshold_sd_over_draws']['independent']:.4f} elig_same(j/i) {inf['eligibility_decision_unchanged_share']['joint']:.3f}/{inf['eligibility_decision_unchanged_share']['independent']:.3f} invalid {inf['invalid_draws_by_reason']['joint']['union']}/{inf['invalid_draws_by_reason']['independent']['union']}")
