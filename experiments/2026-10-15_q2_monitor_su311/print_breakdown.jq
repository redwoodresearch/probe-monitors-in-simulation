def p: if . == null then "–" else (. * 1000 | round / 10 | tostring) end;
def iv: if . == null then "" else "[\(.[0]|p), \(.[1]|p)]" end;
def c($t): "n \($t.n) | 5% \($t["0.05"].catch|p) \($t["0.05"].ci95|iv) | 1% \($t["0.01"].catch|p) \($t["0.01"].ci95|iv) | mean susp \($t.mean_suspicion*100|round/100)";
"cohorts: \(.cohorts|tostring)",
"strategy share in cohort: \(.strategy_coverage_in_cohort|tostring)",
"attempt labels in cohort: \(.label_coverage_in_cohort|tostring)",
"honest A∪B quantiles: 196 \(.honest_quantiles["196"].A_union_B|tostring)",
"                      S_new \(.honest_quantiles.S_new.A_union_B|tostring)",
"batch C (196, 10-15): \(.honest_quantiles["196"].C_1015|tostring)",
"OVERALL (own threshold; other stratum's threshold):",
(.overall | to_entries[] | .key as $s | .value | to_entries[] | "  \($s) \(.key): own \(c(.value.own_threshold)) || other-threshold 5% \(.value.other_stratum_threshold["0.05"].catch|p) 1% \(.value.other_stratum_threshold["0.01"].catch|p)"),
"BY STRATEGY (own threshold):",
(.by_strategy | to_entries[] | .key as $s | .value | to_entries[] | .key as $l | .value | to_entries[] | "  \($s) \($l) strategy=\(.key): \(c(.value))"),
"BY ATTEMPT LABEL (own threshold):",
(.by_attempt_label | to_entries[] | .key as $s | .value | to_entries[] | .key as $l | .value | to_entries[] | "  \($s) \($l) \(.key): \(c(.value))"),
"BY ENVIRONMENT, real_run1 (own threshold):",
(.by_environment | to_entries[] | .key as $s | .value.real_run1 | to_entries[] | "  \($s) \(.key): \(c(.value))"),
"REWEIGHTED S_new (per line):",
(["reweighted_env", "reweighted_env_x_strategy", "reweighted_env_x_label"][] as $r | .[$r] | to_entries[] | "  \($r) \(.key): 196 own 5% \(.value["196_own"]["0.05"].catch|p) 1% \(.value["196_own"]["0.01"].catch|p) | S_new raw 5% \(.value.S_new_unweighted_own["0.05"].catch|p) 1% \(.value.S_new_unweighted_own["0.01"].catch|p) | reweighted own-thr 5% \(.value.S_new_reweighted_own_threshold["0.05"].catch|p) \(.value.S_new_reweighted_own_threshold["0.05"].ci95|iv) 1% \(.value.S_new_reweighted_own_threshold["0.01"].catch|p) | reweighted 196-thr 5% \(.value.S_new_reweighted_196_threshold["0.05"].catch|p) 1% \(.value.S_new_reweighted_196_threshold["0.01"].catch|p) | 196 share outside support \(.value["196_weight_share_absent_from_S_new"]|p) | S_new weight-0 cells \(.value.S_new_cells_weight_0)"),
"line counts: \(.line_counts|tostring)",
"label rows (196 files): \(.label_rows_196_by_line_and_label|tostring)"
