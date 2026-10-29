# Route C, 09's ruling (ii) ("as for PLANT"; via 2e, 10-18): Stage D's and kappa's OWN section 4 pre-flight records must agree with A's
# FINAL section 4 record by key, restricted to SD's (or KAP's) own jobs and files. A may hold more files and drops than SD; SD may hold
# none A lacks. Checked:
#   file set                  every pilot file in SD's zero_unit_by_file (the own + non-own union) is in A's;
#   zero_unit_by_file         for each such file, SD's zero-unit list equals A's, by key;
#   zero_unit_by_file_nonown  for each file SD lists as non-own, its zero_unit equals A's union list for that file;
#   gate_view.drops           every SD run-level drop (run, filter, citation, conditional_on_21b) is one of A's (label_combos excluded:
#                             they name each record's own label x combos);
#   both records CLEAR, with the fields present.
# The records are K by their writer, preflight_routec.py @ 37224810 (main L240-250: the record dict L240, json.dumps L247, the exclusive
# "xb" write L249-250); analyze (L49-183) emits run keys, shas, labels, unit counts, fold integers and booleans, never a measured value.
# Any difference STOPs and goes to 09.
#
#   jq -n -c --slurpfile s <SD or KAP record> --slurpfile a <A's final record> -f sd_preflight_consistent.jq
#
# Prints {"consistent": <bool>, "differing": [<field names>]}: names only, never a value.
def gdrop: {run, filter, citation, conditional_on_21b};
($s[0]) as $d | ($a[0]) as $x
| [ (if ($d | type) != "object" or ($x | type) != "object" then "not_two_records" else empty end),
    (if [$d, $x] | any(.zero_unit_by_file == null or .jobs == null or .gate_view == null) then "missing_fields" else empty end),
    (if [$d, $x] | any(.status != "CLEAR") then "status_not_both_CLEAR" else empty end),
    (($d.zero_unit_by_file // {}) as $dz | ($x.zero_unit_by_file // {}) as $xz
     | (if (($dz | keys) - ($xz | keys)) != [] or ((($d.zero_unit_by_file_nonown // {}) | keys) - ($xz | keys)) != [] then "file_set" else empty end),
       (if [$dz | to_entries[] | .key as $k | select(($xz | has($k)) and $xz[$k] != .value)] != [] then "zero_unit_by_file" else empty end),
       (if [($d.zero_unit_by_file_nonown // {}) | to_entries[] | .key as $k | select(($xz | has($k)) and $xz[$k] != .value.zero_unit)] != []
        then "zero_unit_by_file_nonown" else empty end)),
    (if (([($d.gate_view.drops // [])[] | gdrop] | unique) - ([($x.gate_view.drops // [])[] | gdrop] | unique)) != [] then "gate_view.drops" else empty end) ]
| {consistent: (length == 0), differing: .}
