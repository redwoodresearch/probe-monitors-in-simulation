# Clean-room coverage over EVERY compared quantity of one compute record (a1, 2026-10-20; 2e's request for B and R37). Keys, booleans
# and counts only. Same slot schema as coverage_add17.json, plus kind / suffix / withheld fields.
# Use: jq --arg src <record path> --arg sha <record sha256> [--slurpfile a16 <addendum 16 summary>] -f make_coverage_all.jq <compute record>
#      (without an addendum 16 summary: jq ... --argjson a16 '[]' ...)
# A slot is one (label, quantity) of one row kind; quantity = the row path's suffix after mode and level, joined with ".":
#   pooled:            path = ["pooled", label, COMBO, mode, level] + suffix
#   coheadline_signed: path = ["coheadline_signed", COMBO, mode, level, "group|readout"] + suffix   (label = "group|readout")
#   panel_groups:      path = ["panel_groups", COMBO, "pooled", mode, level, "group|readout"] + suffix (label = "group|readout")
# covered (as registered): at least one VERDICT row, at every mode x level it has, and every one passes (match_flip_rule).
# With an addendum 16 summary (R37), a failing verdict row is marked withheld_rule37 only when that summary proves the failing set is
# exactly RULE 37's expected-withheld group set (PASS16 true, no group fails, every non-group row passes, expected withheld == failing);
# covered_pass16 = at least one verdict row, and every one passes or is withheld. Otherwise the script stops with an error.
def slot_of: if .path[0] == "pooled" then {kind: "pooled", label: .path[1], mode: .path[3], lvl: .path[4], suffix: .path[5:]}
             elif .path[0] == "coheadline_signed" then {kind: .path[0], label: .path[4], mode: .path[2], lvl: .path[3], suffix: .path[5:]}
             elif .path[0] == "panel_groups" then {kind: .path[0], label: .path[5], mode: .path[3], lvl: .path[4], suffix: .path[6:]}
             else error("unknown row kind \(.path[0])") end;
. as $c
| (if ($a16 | length) == 0 then null
   elif ($a16 | length) == 1 and $a16[0].PASS16 == true and ($a16[0].group_fail16 | length) == 0
        and $a16[0].nongroup_pass == $a16[0].nongroup_rows and $a16[0].group_expected_withheld == $a16[0].orig_failing_rows
        and $a16[0].n_verdict_rows == ([$c.comparison[] | select(.verdict)] | length)
   then $a16[0] else error("addendum 16 summary does not prove the failing set is RULE 37's withheld set") end) as $w
| ([$c.comparison[] | select(.verdict and (.match_flip_rule | not))] | length) as $nfail
| (if $w != null and $nfail != $w.group_expected_withheld then error("failing rows \($nfail) != expected withheld") else . end)
| [$c.comparison[] | (slot_of) as $s | . + {_s: $s}]
| group_by([._s.kind, ._s.label, ._s.suffix])
| map(. as $rows | $rows[0]._s as $s | [$rows[] | select(.verdict)] as $vr
      | [$vr[] | select((.match_flip_rule | not) and $w != null and (.path[0] == "coheadline_signed" or .path[0] == "panel_groups"))] as $wh
      | {kind: $s.kind, label: $s.label, quantity: ($s.suffix | map(tostring) | join(".")), suffix: $s.suffix,
         verdict_rows: ($vr | length), passing: ([$vr[] | select(.match_flip_rule)] | length),
         diagnostic_rows: ([$rows[] | select(.verdict | not)] | length),
         modes_levels: ([$vr[] | ._s | "\(.mode)|\(.lvl)"] | unique),
         covered: (($vr | length) > 0 and ([$vr[] | .match_flip_rule] | all))}
        + (if $w == null then {} else
             {withheld_rule37: ($wh | length), withheld_keys: [$wh[] | .what],
              covered_pass16: (($vr | length) > 0 and ([$vr[] | .match_flip_rule or ((.path[0] == "coheadline_signed" or .path[0] == "panel_groups")
                                                                                      and .delta_bucket == "ref absent")] | all))} end))
| . as $slots
| {source: $src, source_sha256: $sha, verdict_PASS: $c.summary.verdict.PASS, PASS16: ($w | if . == null then null else .PASS16 end),
   n_slots: ($slots | length), n_covered: ([$slots[] | select(.covered)] | length),
   n_verdict_slots: ([$slots[] | select(.verdict_rows > 0)] | length),
   n_diagnostic_only: ([$slots[] | select(.verdict_rows == 0)] | length),
   n_failing_verdict_rows: $nfail,
   n_withheld_rule37: (if $w == null then null else ([$slots[] | .withheld_rule37] | add) end),
   n_covered_pass16: (if $w == null then null else ([$slots[] | select(.covered_pass16)] | length) end),
   slots: $slots}
