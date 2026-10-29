# Route C, GATES Amendment 8 (09 @ c1b4a9bdd; 0d's GO, via 2e, 10-18): the FINAL section 4 pre-flight record of a job must equal its
# PROVISIONAL one (built on the 10-16 pooled pins) by key on zero_unit_by_file, zero_unit_by_file_nonown, gate_view.drops, jobs and
# the file set; both must be CLEAR. Any difference STOPs and goes to 09. The records are K by their writer, preflight_routec.py @ 37224810
# (main L240-250: the record dict L240, json.dumps L247, the exclusive "xb" write L249-250); analyze (L49-183) emits run keys, shas,
# labels, unit counts, fold integers and booleans, never a measured value.
#
#   jq -n -c --slurpfile p <provisional record> --slurpfile f <final record> -f prov_final_preflight_equal.jq
#
# Prints {"equal": <bool>, "differing": [<field names>]}: names only, never a value.
def fileset(r): (((r.zero_unit_by_file // {}) | keys) + ((r.zero_unit_by_file_nonown // {}) | keys)) | unique;
($p[0]) as $a | ($f[0]) as $b
| [ (if ($a | type) != "object" or ($b | type) != "object" then "not_two_records" else empty end),
    (if [$a, $b] | any(.zero_unit_by_file == null or .jobs == null or .gate_view == null) then "missing_fields" else empty end),
    (if [$a, $b] | any(.status != "CLEAR") then "status_not_both_CLEAR" else empty end),
    (if $a.zero_unit_by_file != $b.zero_unit_by_file then "zero_unit_by_file" else empty end),
    (if $a.zero_unit_by_file_nonown != $b.zero_unit_by_file_nonown then "zero_unit_by_file_nonown" else empty end),
    (if $a.gate_view.drops != $b.gate_view.drops then "gate_view.drops" else empty end),
    (if $a.jobs != $b.jobs then "jobs" else empty end),
    (if fileset($a) != fileset($b) then "file_set" else empty end) ]
| {equal: (length == 0), differing: .}
