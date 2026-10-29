# Addendum 20: the laptop gate on cleanroom_downstream_<stamp>.json (booleans and counts only). Pre-stated counts:
# verdict 102 (group_excess 18, group_catch 48, pw_shares S 36), diagnostic 144 (pw_shares), record consistency 11 (17g 3, group_catch 8),
# controls 5 (member_gap, N2_index_rule_detected, six_group_excess, N1_membership_detected, member_catch).
# Use: jq -e -f a20_checks.jq results/cleanroom_downstream_<stamp>.json  (exit 0 only when every shape check holds; PASS is reported, not gated)
def fam(f): .by_family[f] // {};
{shape: {
   no_stops: (.n_stops == 0),
   verdict_102: (.n_verdict == 102),
   group_excess: ((fam("group_excess") | [.rows, .verdict, .rc]) == [18, 18, 0]),
   group_catch: ((fam("group_catch") | [.rows, .verdict, .rc]) == [48, 48, 0]),
   pw_shares: ((fam("pw_shares") | [.rows, .verdict, .rc]) == [180, 36, 0]),
   rc_17g: ((fam("group_excess_17g") | [.rows, .rc]) == [3, 3]),
   rc_group_catch: ((fam("group_catch_rc") | [.rows, .rc]) == [8, 8]),
   controls_named: ([.controls[].name] | sort == ["N1_membership_detected", "N2_index_rule_detected", "member_catch", "member_gap", "six_group_excess"]),
   diag_readings: ((.pw_diagnostic | keys) == ["B|l3002", "B|record_convention", "S_minus_B|l3002", "S_minus_B|record_convention"]
                   and ([.pw_diagnostic[].n] | unique) == [36])},
 result: {PASS, n_verdict_match, n_unresolved, n_failing_verdict, rc_all_true,
          controls: [.controls[] | {name, pass, method, idx_rule, how, n, n_true}],
          pw_diagnostic}}
| .ok = ([.shape[]] | all)
| ., (if .ok then empty else error("addendum 20 shape check failed") end)
