# msg_v25_make.jq: the v2.5 release commit message, generated from the record by key (counts, ids, shas, names and labels only).
#   jq -n -r -f msg_v25_make.jq --slurpfile REC <record> --arg RECSHA <sha256> --argjson NTOOL <number of files in v2_5_tooling/>
# The Claude-Session trailer is NOT in the message: the commit RUN_ME appends it from its first argument. The block from "In-scope records"
# to the counts_check line is the v2.4 generator's, copied by line range (the inherited message checks read it).
# c5 (09 via 2e): the erratum line and the axis "now applies" note, copied by script from build/c5_strings_09_ruling.json (668b7c06…)
def C5_ERRATUM: "v2.4 erratum: the Figs 1/6 downstream tag and label notes said 'FINAL per element' while Figs 1/6 is PRELIMINARY (interval ends); corrected in v2.5; no status changes";
def C5_AXIS_NOW: "(now applies: Figs 7/8 are in this version)";
def bn: split("/") | last;
$REC[0] as $r
| ([$r.printed_items.all_rows[] | .status | if startswith("FINAL") then "F" elif startswith("PRELIMINARY") then "P" elif startswith("not route C") then "N" else "O" end]) as $s
| $r.row_rulings_v2_4.r_node_switch as $sw
| $r.row_rulings_v2_5.draft_scope as $ds
| ($r.row_rulings_v2_5.figs78.included == true) as $has78
| ([$r.sensitivity_rows.per_output[]?.entries[]?]) as $sa
| ([$r.disclosures[] | select(.id | startswith("v25."))]) as $d25
| $r.row_rulings_v2_5.figs78 as $f78
| [ (if $has78 then "Route C FINAL-release record, version 2.5 (v2.4 + rows 119-121 FINAL for draft 0029's visible text; Figs 7/8 and slot 24 FINAL)"
     else "Route C FINAL-release record, version 2.5 (v2.4 + rows 119-121 FINAL for draft 0029's visible text)" end),
    "",
    "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.5_2026-10-21.json (sha256 \($RECSHA)) was built by script from git, by key (v2_5_tooling/build/build_v2_5.zsh + build_v2_5.jq), and self-checked: SC5 (the v2.5 invariants against v2.4, re-derived from constants) with its negatives, the ported self-check (sc_rest_v25: rows 119-121 against v19), the message checks, and the tooling gate (keys, shas, counts and class words only).",
    "It supersedes routec_FINAL_release_v2.4_2026-10-21.json @ 6a46541d5 (f14031d6).",
    "Provenance: printed_items.provenance stays v18 (provenance_v84_final18_0141.jsonl @ 58bf4a3ac, 13d29fb9) for every row; rows 119-121's printed token and anchor come from v19 (provenance_v84_final19_X.jsonl @ \($ds.printed_and_anchor_source.commit[0:9]), \($ds.printed_and_anchor_source.sha256[0:8])), A2.",
    "GATES: GATES_routec_2026-10-17.md @ \([$d25[].cite.commit] | unique | map(.[0:9]) | join(", ")) (09's rulings; the v2.5 disclosures are the quoted strings of GATES L\([$d25[].cite.line] | map(tostring) | join(" and L")), copied from git).",
    "Printed rows: \($s | length); FINAL \([$s[] | select(. == "F")] | length); PRELIMINARY \([$s[] | select(. == "P")] | length); not route C \([$s[] | select(. == "N")] | length).",
    "Rows 119-121: FINAL for \($ds.visible_text_of)'s visible text (re-verify \($ds.recheck.of.commit[0:9]); fresh recheck \($ds.recheck.sha256[0:8]) \($ds.recheck.verdict), \($ds.recheck.label)); v2.4 had them PRELIMINARY.",
    (if $has78 then "Figs 7/8 and slot 24 (rows 69-74, 241-248): FINAL (figs78 branch (i): results commit \($f78.record.commit[0:9]) (record \($f78.record.sha256[0:8])), verified by 09 at GATES L\($f78.verified_cite.line); fresh recheck \($f78.recheck.sha256[0:8]) \($f78.recheck.verdict), \($f78.recheck.label)); v2.4 had them PRELIMINARY."
     else "Figs 7/8 and slot 24 (rows 69-74, 241-248): PRELIMINARY (figs78 branch (ii): no 09 verification line at freeze time; v2.6)." end),
    "Interval rule (09, R_component): unchanged from v2.4; R_node switch: \($sw.state).",
    C5_ERRATUM,
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
    "Tooling: experiments/2026-10-14_priority1_spec_09/final_release/v2_5_tooling/ (\($NTOOL) files: build/, selfcheck/; keys, shas, counts and class words only).",
    "Disclosures (v2.5): \(([$d25[].id] + (if $has78 then [$r.row_rulings_v2_5.items[] | select(.id == "figs78") | .disclosures[] | select(startswith("v24.")) | . + " " + C5_AXIS_NOW] else [] end)) | join(", ")).",
    "",
    "Labels change only after 09 verifies this record by key.",
    "",
    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" ] | join("\n")
