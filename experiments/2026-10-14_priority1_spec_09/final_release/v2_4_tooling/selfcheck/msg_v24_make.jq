# msg_v24_make.jq: the v2.4 release commit message, generated from the record by key (counts, ids, shas, names and labels only).
#   jq -n -r -f msg_v24_make.jq --slurpfile REC <record> --arg RECSHA <sha256> --argjson NTOOL <number of files in v2_4_tooling/>
# The Claude-Session trailer is NOT in the message: the commit RUN_ME appends it from its first argument.
def bn: split("/") | last;
$REC[0] as $r
| ([$r.printed_items.all_rows[] | .status | if startswith("FINAL") then "F" elif startswith("PRELIMINARY") then "P" elif startswith("not route C") then "N" else "O" end]) as $s
| $r.row_rulings_v2_4.r_node_switch as $sw
| ([$r.printed_items.all_rows[] | select(.row == 122)][0].status | startswith("FINAL")) as $f122
| $r.row_rulings_v2_4.row122_recheck as $rc
| ([($r.element_tables // [])[] | select(.kind == "figs78_item")] | length > 0) as $has78
| ([$r.sensitivity_rows.per_output[]?.entries[]?]) as $sa
| ([$r.disclosures[] | select(.id | startswith("v24."))]) as $d24
| [ "Route C FINAL-release record, version 2.4 (v2.3 + the v2.4 rulings, per row)",
    "",
    "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.4_2026-10-21.json (sha256 \($RECSHA)) was built by script from git, by key (v2_4_tooling/build/build_v2_4.zsh + build_v2_4.jq), and self-checked: SC4 (the v2.4 invariants, re-derived from constants) with its negatives, the v2.4 variant of the v2.3 self-check, the message checks, and the tooling gate (keys, shas, counts and class words only).",
    "It supersedes routec_FINAL_release_v2.3_2026-10-20.json @ 80a77ecdc (ebb47e28).",
    "Provenance: provenance_v84_final18_0141.jsonl @ 58bf4a3ac (13d29fb9); equals final17 @ 98b5054fb except rows 119-122.",
    "GATES: GATES_routec_2026-10-17.md @ \([$d24[].cite.commit] | unique | map(.[0:9]) | join(", ")) (09's rulings; every v2.4 disclosure is a cited GATES line or one of its quoted strings, copied from git).",
    "Printed rows: \($s | length); FINAL \([$s[] | select(. == "F")] | length); PRELIMINARY \([$s[] | select(. == "P")] | length); not route C \([$s[] | select(. == "N")] | length).",
    "Interval rule (09, R_component): the \($sw.rows_new_interval | length) interval rows are \(if $sw.state == "OFF" then "PRELIMINARY" else "FINAL" end); v2.3 rows FINAL -> \(if $sw.state == "OFF" then "PRELIMINARY" else "FINAL" end): \($sw.rows_v2_3_corrections | map(tostring) | join(", ")); Figs 1/6 and Fig 4 \(if $sw.state == "OFF" then "PRELIMINARY" else "FINAL" end). R_node switch: \($sw.state).",
    (if $f122 then "Rows 119-122: row 122 FINAL (v18 pinned; fresh recheck \($rc.sha256[0:8]) \($rc.verdict) of the verifier results commit \($rc.of.commit[0:9]), \($rc.label)); rows 119-121 PRELIMINARY."
     else "Rows 119-122: PRELIMINARY (no confirmed recheck before the freeze)." end),
    (if $has78 then "Figs 7/8 and slot 24 (rows 69-74, 241-248): FINAL (the figs78 item, verified by 09)."
     else "Figs 7/8 and slot 24 (rows 69-74, 241-248): PRELIMINARY (the figs78 item has not run)." end),
    "",
    "In-scope records: \([$r.in_scope_records[].name] | join(", ")). FINAL outputs: \([$r.in_scope_outputs[].name] | join(", ")).",
    "PRELIMINARY outputs / printed items: \([$r.preliminary_outputs[].name] | join(", ")).",
    "Out of scope: \([$r.out_of_scope_records[] | "\(.name) (\(.status))"] | join("; ")).",
    "Exclusions inside FINAL outputs (Correction 3 to A21, amended): \([$r.exclusions[] | "\(.id) \(.entries_n - .flipped_n) PRELIMINARY of \(.entries_n)"] | join("; ")).",
    "Figure labels: \([$r.figure_labels[] | "\(.file | bn | sub("_source\\.json$"; "")) \(.label)"] | join("; ")).",
    # b8 (09, fresh review N2): the slot-level and the printed-row counts, both true
    (([$r.row_rulings_v2_4.items[] | .disclosure_rows // {} | .["v24.v23_interval_corrections.INT-CORR.2"] // empty] | add // []) as $sr
     | ([$r.printed_items.all_rows[] | select(.row as $n | $sr | index([$n]) != null) | select(.status | startswith("PRELIMINARY"))] | length) as $srp
     | if ($sa | length) > 0 and ([$sa[] | select(.status == "FINAL")] | length) == ($sa | length) then
         "Sensitivity: in sensitivity_rows the \($sa | length) rc1a sensitivity slots are FINAL (their sensitivity_halves, unchanged from v2.3); the \($srp) printed rows that rest on them (\($sr | map(tostring) | join(", "))) are PRELIMINARY by [INT-CORR] (interval ends)."
       else "Sensitivity: \(($sa | length) - ([$sa[] | select(.status == "FINAL")] | length)) of \($sa | length) rc1a sensitivity slots stay PRELIMINARY; the \($srp) printed rows that rest on them (\($sr | map(tostring) | join(", "))) are PRELIMINARY by [INT-CORR]." end),
    "counts_check: the App A printed counts (rows 250/251) are FINAL with disclosure; printed_numbers rows \([$r.counts_check.printed_numbers[] | select(.status | startswith("FINAL")) | .row] | sort | map(tostring) | join(", ")) are FINAL; the rule text says so.",
    "Tooling: experiments/2026-10-14_priority1_spec_09/final_release/v2_4_tooling/ (\($NTOOL) files: build/, selfcheck/, q1/; keys, shas, counts and class words only).",
    "Disclosures (v2.4): \([$d24[].id] | join(", ")).",
    "",
    "Labels change only after 09 verifies this record by key.",
    "",
    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" ] | join("\n")
