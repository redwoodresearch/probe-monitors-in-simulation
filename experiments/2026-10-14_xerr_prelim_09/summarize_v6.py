import json, sys
d = json.load(open(sys.argv[1])); r = d["result"]
print("git", d["git"][:12], "| gates pass", r["gates"]["pass"], "| G1 max", max(r["gates"]["G1"].values()))
u = r["universe_and_split"]
print("universe", u["universe_n"], "| cohort", r["cohort"]["n"], "| cohort elig half", u["cohort_in_eligibility_half"], "| est half", u["cohort_in_estimation_half"], "| strata", u["strata_cohort_counts"])
print("hs label:", r["probe_labels"]["highstakes"][:90])
for rn, v in r["c_sameness_append5"].items():
    print("sameness", rn, {p: (round(x["auroc"],3), [round(y,3) for y in x["ci95"]], "C" if x["C_pooled"] else "B") for p, x in v.items()})
def f(x): return "NA" if x is None else f"{x:.2f}"
for key, row in r["results"].items():
    ro, cfg, a = key.split("|")
    if cfg not in ("mechanical", "mechanical_sol_raw", "B", "BuC", "AuB", "mechanical_sol_permodel_raw"): continue
    e = row["eligibility_append13_4"]
    pm = row["primary_eligible_estimation_half"]
    line = f"{ro[:7]} {cfg:28s} {a:10s} elig={','.join(e['eligible']) or '-'}"
    for cn in ("sim_minus_mean_real", "sim_minus_run1", "sim_minus_run2", "reexecution_run2_minus_run1"):
        b = pm.get(cn, {}); p = b.get("pooled", {}) if isinstance(b, dict) else {}
        line += f" | {cn.replace('_minus_','-').replace('reexecution_run2-run1','rerun')}: X={f(p.get('X_err'))} U={f(p.get('U_approx_upper_95'))}"
    print(line)
    if cfg == "mechanical":
        p = pm["sim_minus_mean_real"]["pooled"]
        print("     primary sim-mean: bias", f(p.get("bias_mean")), p.get("bias_ci95") and [round(x,2) for x in p["bias_ci95"]], "spread", f(p.get("spread")), "plain", f(p.get("X_err_plugin_no_deconvolution")), "| floored b/s", p.get("bias2_corrected_floored"), p.get("spread_floored"), "| complete", p.get("complete_draws"), "| n", p.get("n_tasks"), "|", p.get("note",""))
        print("     strata label:", p.get("strata_label"))
        for st in ("with_strategy", "without_strategy"):
            q = pm["sim_minus_mean_real"][st]; print(f"     {st}: n={q['n_tasks']} X={f(q.get('X_err'))} U={f(q.get('U_approx_upper_95'))} {q.get('stratum_label') or ''} {q.get('note','')}")
        reg = row["registered_strata_pooling_test_tost5"]["primary_eligible_estimation_half"]
        if reg: print("     TOST:", {k: (v["ci90"] and [round(x,1) for x in v["ci90"]]) for k, v in reg["per_probe"].items()}, "|", reg["difference_reading"])
        print("     fold vs env widths differ:", pm["sim_minus_mean_real"]["fold_vs_env_95pct_widths_differ_gt10pct"], "U env/fold", [f(x) for x in pm["sim_minus_mean_real"]["U_env_vs_fold"]])
        print("     per-probe delta [ci] Uabs:", {k: (round(v["delta"],2), [round(x,1) for x in v["ci95"]], round(v["U_abs_approx"],1)) for k, v in p["per_probe"].items()} if isinstance(p.get("per_probe"), dict) else p.get("per_probe"))
        s = row["sensitivity_eligibility_not_applied"]["sim_minus_mean_real"]["pooled"]
        print("     sens (no eligibility, all 5, all cohort): X", f(s.get("X_err")), "U", f(s.get("U_approx_upper_95")), "| real TPR elig-half", {k: round(100*v,1) for k, v in e["real_tpr_eligibility_half"].items()})
        print("     pools", row["pools"], "| thin", {k: v.get("thin_pool_k_lt_10") for k, v in row["thresholds"].items()})
