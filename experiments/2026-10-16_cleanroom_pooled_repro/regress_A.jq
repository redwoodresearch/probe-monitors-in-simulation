# A regression after the addendum 14 generalization (09's condition, a1 2026-10-20). Keys, counts, booleans and buckets only.
# Use (with set -o pipefail):
#   jq -n --slurpfile old results/cleanroom_compute_2026-10-20_001845.json --slurpfile new results/cleanroom_A_compute_<stamp>.json \
#      -f regress_A.jq | tee <summary> | jq -e -s 'length == 1 and .[0].EQUAL == true' > /dev/null
# rc 0 only if every pre-stated field is identical between A's committed compute record and the rerun under the new code and config.
def vsum: .summary.verdict | {PASS, rows: (.rows | {expected_from_A, compared, passed, missing_from_mine, not_in_A, not_passing}),
                              thresholds, cells, controls, group_membership, min_cell_weight: (.min_cell_weight.rows_below),
                              registered_absent_from_A, n_exempt: (.registered_absent_exempt | length), not_compared};
def buckets: [.comparison[] | select(.verdict)] | group_by(.delta_bucket) | map({b: .[0].delta_bucket, n: length});
def rowflags: [.comparison[] | {path: (.path | tostring), what, verdict, delta_bucket, match_flip_rule}] | sort_by(.path, .what);
def ctlkeys: .controls | map_values({label, key, pass});
def thr: [.thresholds[] | {label, level, side, match, counts_ok, n_swap_candidates}] | sort_by(.label, .level, .side);
def cel: .cells | map_values({pass, compared, equal, n_flips, n_fails, flips, fails});
($old[0]) as $o | ($new[0]) as $n
| {verdict_summary_equal: (($o | vsum) == ($n | vsum)),
   buckets_equal: (($o | buckets) == ($n | buckets)), buckets: ($n | buckets),
   row_flags_equal: (($o | rowflags) == ($n | rowflags)), n_rows: ($n.comparison | length),
   controls_equal: (($o | ctlkeys) == ($n | ctlkeys)),
   thresholds_equal: (($o | thr) == ($n | thr)),
   cells_equal: (($o | cel) == ($n | cel)),
   rel_x1_bitwise_equal_info: ([$o.comparison[] | .rel_x1] == [$n.comparison[] | .rel_x1]),
   new_negative_mode: $n.controls.negative.negative_mode, new_control_level: $n.controls.positive.control_level,
   new_PASS: $n.summary.verdict.PASS}
| .EQUAL = (.verdict_summary_equal and .buckets_equal and .row_flags_equal and .controls_equal and .thresholds_equal and .cells_equal
            and .new_PASS == true and .new_control_level == "0.01" and .new_negative_mode == "raise")
