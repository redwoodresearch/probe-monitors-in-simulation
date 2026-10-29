import json, sys
d = json.load(open(sys.argv[1]))["result"]
for key, row in d["rows"].items():
    if "projections" not in row:
        print(key, row.get("note")); continue
    print(f"== {key}: {row['probes']} | between share {[round(x, 2) for x in row['between_share_of_current_variance']]} | floored {row['between_floored_at_0']} | rho {row['rho_used']:.2f}")
    for sc, cells in row["projections"].items():
        for coh in ("current_183_posthoc", "507cells_attempt93", "507cells_attempt50"):
            print(f"  {sc:14s} {coh:20s} " + " | ".join(f"{r}: {cells[f'{coh}|{r}']['U_median']:.1f}" for r in ("sim1_real2", "sim2_real2", "sim2_real4", "sim3_real3", "sim3_real6")))
