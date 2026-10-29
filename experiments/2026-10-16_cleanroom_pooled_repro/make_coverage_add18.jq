# Addendum 18 coverage, machine-readable, for the FINAL-release builder (a1, 2026-10-20). Keys, booleans and counts only.
# Use: jq --arg src <A18 compute record path> --arg sha <its sha256> --argjson per '{"real_line": 4, ...}' -f make_coverage_add18.jq <record>
# One slot per (label, addendum 18 family), labels = the record's threshold labels. A row belongs to label l iff its .what starts with "l|" (labels may contain "|", and end in " C" or " R").
# A slot is COVERED iff it has exactly the pre-stated number of VERDICT rows ($per[family]) and every one passes (match_flip_rule).
. as $c
| ([$c.thresholds[] | .label] | unique) as $labels
| [$labels[] as $l | $per | keys[] as $f
   | [$c.comparison[] | select(.family == $f and (.what | startswith($l + "|")))] as $rows
   | [$rows[] | select(.verdict)] as $vr
   | {label: $l, family: $f, verdict_rows: ($vr | length), expected_rows: $per[$f],
      passing: ([$vr[] | select(.match_flip_rule)] | length), diagnostic_rows: ([$rows[] | select(.verdict | not)] | length),
      covered: (($vr | length) == $per[$f] and ([$vr[] | .match_flip_rule] | all))}] as $slots
| {source: $src, source_sha256: $sha, verdict_PASS: $c.summary.verdict.PASS, n_labels: ($labels | length),
   note: "A slot with diagnostic_rows > 0 is covered by its verdict rows only: on B, share_wnu, units_per_run mean / median (18c, RULE l.3002) and #identity (18d) are diagnostic, so those slots are covered by the S side",
   slots: $slots,
   n_covered: ([$slots[] | select(.covered)] | length), n_not_covered: ([$slots[] | select(.covered | not)] | length)}
