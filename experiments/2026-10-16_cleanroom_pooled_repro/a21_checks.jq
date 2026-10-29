# Addendum 21 (PLANT tier 1): the laptop shape gate on cleanroom_plant_<stamp>.json (booleans and counts only; PASS reported, not gated).
# Use: jq -e -f a21_checks.jq results/cleanroom_plant_<stamp>.json
{shape: {
   no_stops: (.n_stops == 0),
   controls_named: (([.controls[].name | select(startswith("C5_") | not)] | sort) == ["C0_reproduces_A18_values", "C1_plant0_equals_A_draws",
                     "C2_catch_shift_is_gap_shift", "C3_member_unplanted", "C4_points_scale", "C6_origin_catch0", "C7_solve_scale", "C8_achieved_scale"]),
   families: ((.by_family | keys) - ["aggregate_excess_move", "aggregate_excess_r1"] | sort == ["aggregate", "counts", "plant_rows", "plant_solve"]),
   counts_rows_304: (.by_family.counts.rows == 304),
   plant_rows_192_verdicts: (.n_plant_rows_verdict == 192),
   c5_eight: ([.controls[].name | select(startswith("C5_"))] | length == 8),
   printed_58: (.printed.n == 58 and .printed.n_present == 58)},
 result: {PASS, n_verdict, n_verdict_match, n_unresolved, n_failing_verdict, printed: {n_match: .printed.n_match, failing: .printed.failing},
          controls: [.controls[] | {name, pass, n, n_true, idx_rule, method, scale}]}}
| .ok = ([.shape[]] | all)
| ., (if .ok then empty else error("addendum 21 shape check failed") end)
