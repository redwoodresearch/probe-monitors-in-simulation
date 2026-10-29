# Addendum 18 / 18c: adding the A18 rows changes nothing else. The A18 record must equal the AS record (same labels, same inputs) key
# by key on everything AS had, add no thresholds or cells, and the NEW rows must be exactly the pre-stated count per family, with exactly
# the pre-stated number of DIAGNOSTIC (verdict false) rows per family (18c: B's share_wnu, mean, median); all other added rows verdict.
# Keys, booleans and counts only. $want = {"all": {family: n}, "diagnostic": {family: n}} (config/add18_expect.json .subset_add18).
# Use (with set -o pipefail):
#   jq -n --slurpfile old <AS compute record> --slurpfile new <A18 compute record> --argjson want "$(jq -c .subset_add18 config/add18_expect.json)" \
#      -f subset_add18.jq | tee <summary> | jq -e -s 'length == 1 and .[0].EQUAL == true' > /dev/null
def rk: "\(.path | tostring)|\(.what)";
def rmap: reduce .[] as $r ({}; .[$r | rk] = ($r | {verdict, delta_bucket, match_flip_rule}));
def tmap: reduce .[] as $t ({}; .["\($t.label)|\($t.level)|\($t.side)"] = ($t | {match, counts_ok, n_swap_candidates}));
def byfam: group_by(.family) | map({key: .[0].family, value: length}) | from_entries;
($old[0]) as $o | ($new[0]) as $n
| ($o.comparison | rmap) as $om | ($n.comparison | rmap) as $nm
| ($o.thresholds | tmap) as $ot | ($n.thresholds | tmap) as $nt
| ([$om | keys[]] - [$nm | keys[]]) as $lost
| ([$nm | keys[]] - [$om | keys[]]) as $added
| [$n.comparison[] | select((rk) as $k | $added | index($k))] as $addrows
| {rows_old: ($om | length), rows_new: ($nm | length), rows_lost: ($lost | length),
   keys_unique: (($om | length) == ($o.comparison | length) and ($nm | length) == ($n.comparison | length)),
   rows_equal_on_old: ([$om | keys[] | . as $k | $om[$k] == $nm[$k]] | all),
   rows_added: ($added | length), added_by_family: ($addrows | byfam),
   added_diagnostic_by_family: ([$addrows[] | select(.verdict != true)] | byfam),
   thresholds_equal: ($ot == $nt),
   cells_equal: (($o.cells | map_values({pass, compared, equal, n_flips, n_fails, flips, fails}))
                 == ($n.cells | map_values({pass, compared, equal, n_flips, n_fails, flips, fails}))),
   controls_equal: (($o.controls | map_values({label, key, pass})) == ($n.controls | map_values({label, key, pass}))),
   group_membership_equal: ($o.summary.verdict.group_membership == $n.summary.verdict.group_membership)}
| .EQUAL = (($want.all | length) > 0 and .keys_unique and .rows_lost == 0 and .rows_equal_on_old
            and .added_by_family == $want.all and .rows_added == ([$want.all[]] | add)
            and .added_diagnostic_by_family == $want.diagnostic and .thresholds_equal and .cells_equal and .controls_equal
            and .group_membership_equal)
