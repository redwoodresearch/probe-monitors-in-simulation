# 09 S1b-7 PSR, step 5: the pooled builds' companion set (the catalogue manifest's labels x the 4 "(all-origins draws)" member origins)
# must be present and NOT withheld in the per-set record. Keys / booleans / counts only. $labels: the manifest label list (JSON array).
["S_old (all-origins draws)", "S_new (all-origins draws)", "S_newmains (all-origins draws)", "S_paper493 (all-origins draws)"] as $mem
| [$labels[] as $l | $mem[] as $o | {l: $l, o: $o, row: (.rows[$l][$o])}] as $cells
| {n_cells: ($cells | length),
   n_absent: ([$cells[] | select(.row == null)] | length),
   n_withheld: ([$cells[] | select(.row != null and (.row | type == "object") and (.row | has("withheld")))] | length),
   absent: ([$cells[] | select(.row == null) | "\(.l) | \(.o)"] | .[:10]),
   withheld: ([$cells[] | select(.row != null and (.row | type == "object") and (.row | has("withheld"))) | "\(.l) | \(.o)"] | .[:10])}
