# Addendum 20d step 2: the rerun must reproduce the 20b record (2a675acd6) on every family except group_catch's interval rows.
# Use: jq -n --slurpfile old results/cleanroom_downstream_2026-10-20_121301.json --slurpfile new <rerun result> -f a20d_regress.jq
#      (exit 0 only when .EQUAL is true)
($old[0]) as $o | ($new[0]) as $n
| {families_equal: (["group_excess", "group_excess_17g", "group_catch_rc", "pw_shares"] | map(. as $f | $o.by_family[$f] == $n.by_family[$f]) | all),
   pw_diagnostic_equal: ($o.pw_diagnostic == $n.pw_diagnostic),
   other_controls_equal: ([$o.controls[] | select(.name != "member_catch")] == [$n.controls[] | select(.name != "member_catch")]),
   no_stops: ($n.n_stops == 0),
   n_verdict_114: ($n.n_verdict == 114),
   group_catch_exact: (($n.by_family.group_catch | [.verdict_match, .unresolved]) as $v | $v == [48, 0] or $v == [16, 32]),
   no_non_interval_catch_failure: ([$n.failing_verdict_keys[] | select(startswith("groups.")) | select(test("\\.(ci95\\[[01]\\]|draws|finite_draws)$") | not)] | length == 0)}
| .EQUAL = ([.[]] | all)
| ., (if .EQUAL then empty else error("addendum 20d regression vs 20b failed") end)
