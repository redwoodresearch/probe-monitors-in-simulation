# Addendum 17 (R2's note): the pre-stated verdict counts, enforced in code. Keys and counts only.
# Use (with set -o pipefail):
#   jq --argjson want '{"rows": 736, "thresholds": 72, "cells": 36, "controls": 3, "groups": 40}' -f verdict_counts.jq <compute record> \
#      | tee <summary> | jq -e -s 'length == 1 and .[0].COUNTS_OK == true' > /dev/null
# COUNTS_OK needs each count exactly as pre-stated (compared and expected_from_A) and, with "pass": true, every part passing.
.summary.verdict as $v
| {rows: $v.rows.compared, rows_expected_from_A: $v.rows.expected_from_A, rows_passed: $v.rows.passed,
   thresholds: $v.thresholds.compared, thresholds_passed: $v.thresholds.passed,
   cells: $v.cells.compared, cells_passed: $v.cells.passed,
   controls: $v.controls.registered, controls_passed: $v.controls.passed,
   groups: $v.group_membership.checked, groups_equal: $v.group_membership.equal, PASS: $v.PASS}
| .COUNTS_OK = (.rows == $want.rows and .rows_expected_from_A == $want.rows and .thresholds == $want.thresholds
                and .cells == $want.cells and .controls == $want.controls and .groups == $want.groups
                and (if $want.pass then (.PASS == true and .rows_passed == .rows and .thresholds_passed == .thresholds
                                         and .cells_passed == .cells and .controls_passed == .controls and .groups_equal == .groups)
                     else true end))
