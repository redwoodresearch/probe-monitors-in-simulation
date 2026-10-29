import json, sys
d = json.load(open(sys.argv[1]))["result"]
for key, row in d["rows"].items():
    if "projections" not in row: print(key, row.get("note")); continue
    print(f"== {key}: probes {row['probes']} | rho raw {row['rho_raw_backed_out']:.2f} used {row['rho_used_all_scenarios']:.2f} | v6 observed U {row['v6_observed_U']:.2f} (n {row['v6_n_estimation']})")
    for sc, cells in row["projections"].items():
        c0 = cells["repeat_of_v6_design_n90"]
        print(f"  {sc}: repeat-of-v6-design U {c0['U_median']:.1f} [{c0['U_p10_p90'][0]:.1f}, {c0['U_p10_p90'][1]:.1f}]")
        for coh in ("current_183", "507cells_507mains_attempt93", "507cells_507mains_attempt50", "1000cells_by542mains_attempt93", "1000cells_by542mains_attempt50", "1000cells_by1000cells_attempt93_upper"):
            parts = []
            for rule, short in (("single_split", "single"), ("crossfit_A7_mean_of_folds", "xfitA7"), ("frozen_eligibility_from_v6_half", "frozen")):
                c = cells[f"{coh}|{rule}"]
                parts.append(f"{short} n{c['n_estimation_tasks']} U {c['U_median']:.1f}")
            print(f"    {coh:40s} " + " | ".join(parts))
