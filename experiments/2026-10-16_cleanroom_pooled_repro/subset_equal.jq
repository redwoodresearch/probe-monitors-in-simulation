# Addendum 17 (09's condition): adding the sensitivity labels changes nothing else. A run with them must equal the earlier record
# of the same target, key by key, on everything the earlier record had; every NEW row must belong to a sensitivity label.
# R1 (P2): it FAILs unless the sensitivity labels were actually added: $sens non-empty, rows added, every $sens label among the
# added rows, and exactly |sens| x levels x 2 thresholds added.
# Keys, booleans and counts only.
# Use (with set -o pipefail):
#   jq -n --slurpfile old <earlier compute record> --slurpfile new <+S compute record> --argjson sens '["c_asis_uncut C", ...]' \
#      -f subset_equal.jq | tee <summary> | jq -e -s 'length == 1 and .[0].EQUAL == true' > /dev/null
def rk: "\(.path | tostring)|\(.what)";
def rmap: reduce .[] as $r ({}; .[$r | rk] = ($r | {verdict, delta_bucket, match_flip_rule}));
def tmap: reduce .[] as $t ({}; .["\($t.label)|\($t.level)|\($t.side)"] = ($t | {match, counts_ok, n_swap_candidates}));
($old[0]) as $o | ($new[0]) as $n
| ($o.comparison | rmap) as $om | ($n.comparison | rmap) as $nm
| ($o.thresholds | tmap) as $ot | ($n.thresholds | tmap) as $nt
| ([$om | keys[]] - [$nm | keys[]]) as $lost
| ([$nm | keys[]] - [$om | keys[]]) as $added
| [$n.comparison[] | select((rk) as $k | $added | index($k))] as $addrows
| {rows_old: ($om | length), rows_new: ($nm | length), rows_lost: ($lost | length),
   rows_equal_on_old: ([$om | keys[] | . as $k | $om[$k] == $nm[$k]] | all),
   rows_added: ($added | length),
   added_all_sensitivity: ([$addrows[] | .path[0] == "pooled" and (.path[1] as $l | $sens | index($l) != null)] | all),
   thresholds_equal_on_old: ([$ot | keys[] | . as $k | $ot[$k] == $nt[$k]] | all),
   thresholds_added_all_sensitivity: ([($nt | keys) - ($ot | keys) | .[] | split("|") as $p | ($p[:-2] | join("|")) as $l | $sens | index($l) != null] | all),
   added_labels: ([$addrows[] | .path[1]] | unique),
   n_levels: ([$n.thresholds[] | .level] | unique | length),
   thresholds_added: (($nt | keys) - ($ot | keys) | length),
   cells_equal_on_old: ([$o.cells | keys[] | . as $k | ($o.cells[$k] | {pass, compared, equal, n_flips, n_fails, flips, fails}) == ($n.cells[$k] | {pass, compared, equal, n_flips, n_fails, flips, fails})] | all),
   controls_equal: (($o.controls | map_values({label, key, pass})) == ($n.controls | map_values({label, key, pass}))),
   group_membership_equal: ($o.summary.verdict.group_membership == $n.summary.verdict.group_membership)}
| .EQUAL = (($sens | length) > 0 and .rows_added > 0 and ((.added_labels | sort) == ($sens | sort))
            and .thresholds_added == (($sens | length) * .n_levels * 2) and .rows_lost == 0 and .rows_equal_on_old and .added_all_sensitivity and .thresholds_equal_on_old
            and .thresholds_added_all_sensitivity and .cells_equal_on_old and .controls_equal and .group_membership_equal)
