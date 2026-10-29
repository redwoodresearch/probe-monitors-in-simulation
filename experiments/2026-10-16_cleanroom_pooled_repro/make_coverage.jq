# Clean-room coverage list, machine-readable, for the FINAL-release builder (a1, 2026-10-20). Keys, booleans and counts only.
# Use: jq --arg src <record path> --arg sha <record sha256> --argjson q '<[{label, quantity}] pairs>' -f make_coverage.jq <compute record>
# quantity is "<group>.<key>" under pooled[label][pooled_all_origins][mode][level], e.g. "catch.run1", "gap.delta_full", "gap.real_line".
# A pair is COVERED iff the record has at least one VERDICT row for it, at every mode x level it has, and every one of them passes
# (match_flip_rule true). Anything else, including a quantity compared only as a diagnostic, is NOT covered.
. as $c
| [$q[] | . as $p | ($p.quantity | split(".")) as [$g, $k]
   | [$c.comparison[] | select(.path[0] == "pooled" and .path[1] == $p.label and .path[5] == $g and .path[6] == $k)] as $rows
   | [$rows[] | select(.verdict)] as $vr
   | {label: $p.label, quantity: $p.quantity, verdict_rows: ($vr | length), passing: ([$vr[] | select(.match_flip_rule)] | length),
      diagnostic_rows: ([$rows[] | select(.verdict | not)] | length),
      modes_levels: ([$vr[] | "\(.path[3])|\(.path[4])"] | unique),
      covered: (($vr | length) > 0 and ([$vr[] | .match_flip_rule] | all))}] as $slots
| {source: $src, source_sha256: $sha, verdict_PASS: $c.summary.verdict.PASS, slots: $slots,
   n_covered: ([$slots[] | select(.covered)] | length), n_not_covered: ([$slots[] | select(.covered | not)] | length)}
