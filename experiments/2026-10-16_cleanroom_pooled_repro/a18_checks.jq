# Addendum 18b / 18e: a compute record against its committed expected counts (config/add18_expect.json, config/add18e_<tag>_expect.json).
# Keys, booleans and counts only. With verdict_counts.pass true: every verdict row, threshold, cell, control and group passes, and PASS.
# With pass false (18e, R37S18: RULE 37's withheld groups, addendum 16): exactly expected_failing_rows failing rows and
# expected_group_differs differing groups, PASS false; addendum16_R37_v2.jq then decides PASS16.
# Use (with set -o pipefail):
#   jq --slurpfile e <expect json> -f a18_checks.jq <compute record> | tee <summary> | jq -e -s 'length == 1 and .[0].OK == true' > /dev/null
($e[0]) as $x | .summary.verdict as $v | $x.verdict_counts as $w
| {rows: $v.rows.compared, rows_expected_from_A: $v.rows.expected_from_A, rows_passed: $v.rows.passed,
   record_consistency: $v.rows.record_consistency,
   thresholds: $v.thresholds.compared, thresholds_passed: $v.thresholds.passed, cells: $v.cells.compared, cells_passed: $v.cells.passed,
   controls: $v.controls.registered, controls_passed: $v.controls.passed,
   groups: $v.group_membership.checked, groups_equal: $v.group_membership.equal, ge_control: $v.ge_control, PASS: $v.PASS}
| .OK = (.rows == $w.rows and .rows_expected_from_A == $w.rows and .record_consistency == $x.record_consistency_rows
         and .thresholds == $w.thresholds and .thresholds_passed == .thresholds and .cells == $w.cells and .cells_passed == .cells
         and .controls == $w.controls and .controls_passed == .controls and .groups == $w.groups and .ge_control.ok == true
         and (if $w.pass then (.rows_passed == .rows and .groups_equal == .groups and .PASS == true)
              else (.rows_passed == .rows - $w.expected_failing_rows and .groups_equal == .groups - $w.expected_group_differs
                    and .PASS == false) end))
