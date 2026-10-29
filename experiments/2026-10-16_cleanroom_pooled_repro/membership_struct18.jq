# Addendum 18b/18c: the A18 membership record's zero-row runs (derived from my inputs) against the pre-stated expectation in
# config/add18_expect.json (membership_structural; NOT blind: reviewers saw the record's structural list and the input-side exclusions).
# Per label: gate pass; exactly per_side structural runs on B and on S, all of the registered cell; the known exclusions exactly as
# pre-registered (by key); no unexpected zero-row run. Keys and counts only.
# Use (with set -o pipefail):
#   jq --slurpfile e config/add18_expect.json -f membership_struct18.jq <A18 membership record> | tee <summary> | jq -e -s 'length == 1 and .[0].STRUCT_OK == true' > /dev/null
($e[0].membership_structural) as $w
| [.labels | to_entries[] | .key as $l | .value as $r
   | {label: $l, gate: $r.gate, B: ($r.structural_no_row.B | length), S: ($r.structural_no_row.S | length),
      cells: ([$r.structural_no_row.B[], $r.structural_no_row.S[]] | map(split("|")[1:] | join("|")) | unique),
      excluded_known: $r.zero_row_excluded_known, unexpected: $r.zero_row_unexpected,
      unresolved: ($r.zero_row_world_unresolved | length)}] as $rows
| {n_labels: ($rows | length), labels: $rows,
   STRUCT_OK: (($rows | length) == $w.n_labels
               and ([$rows[] | .gate == "pass" and .B == $w.per_side and .S == $w.per_side and .cells == [$w.cell]
                     and (.excluded_known.B | sort) == ($w.excluded_known.B | sort) and (.excluded_known.S | sort) == ($w.excluded_known.S | sort)
                     and (.unexpected.B | length) == 0 and (.unexpected.S | length) == 0] | all))}
