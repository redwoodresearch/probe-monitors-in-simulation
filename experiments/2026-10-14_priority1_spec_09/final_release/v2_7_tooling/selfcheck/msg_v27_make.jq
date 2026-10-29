# msg_v27_make.jq: the v2.7 release commit message, generated from the record by key (counts, ids, shas, names and labels only).
#   jq -n -r -f msg_v27_make.jq --slurpfile REC <record> --arg RECSHA <sha256> --argjson NTOOL <number of files in v2_7_tooling/>
# The Claude-Session trailer is NOT in the message: the commit RUN_ME appends it. Every GATES cite carries its commit (V26 follow-up 71).
# The block from "In-scope records" to the counts_check line is the v2.4 - v2.6 generator's.
def bn: split("/") | last;
def pf: if startswith("FINAL") then "F" elif startswith("PRELIMINARY") then "P" elif startswith("not route C") then "N" else "O" end;
def gl($c): "L\($c.line) @ \($c.commit[0:9])";
$REC[0] as $r
| ([$r.printed_items.all_rows[] | .status | pf]) as $s
| ([$r.sensitivity_rows.per_output[]?.entries[]?]) as $sa
| ($r.row_rulings_v2_7.items | map({(.id): .}) | add) as $I
| ($r.disclosures | map({(.id): .}) | add) as $D
| ([$r.figure_labels[] | select(.output == "Fig 4")][0]) as $l4
| ([$r.element_tables[] | select(.file == ([$r.figure_labels[] | select(.output == "Fig 5")][0].file))][0]) as $t5
| ([$r.in_scope_outputs[] | select(.name == "rc1b")][0]) as $rc
| [ "Route C FINAL-release record, version 2.7 (v2.6 + Fig 4, Fig 5 and rc1b FINAL)",
    "",
    "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.7_2026-10-21.json (sha256 \($RECSHA)) was built by script from git, by key (v2_7_tooling/build/build_v2_7.zsh + build_v2_7.jq), and self-checked: SC7 (the v2.7 invariants against v2.6, re-derived from constants and git) with its negatives, the ported self-check (sc_rest_v27) with its negatives, the message checks, and the tooling gate (keys, shas, counts and class words only).",
    "It supersedes routec_FINAL_release_v2.6_2026-10-21.json @ \($r.supersedes.commit[0:9]) (\($r.supersedes.sha256[0:8])).",
    "GATES: GATES_routec_2026-10-17.md (09's v2.7 rulings; the cited lines, by script from git): \([$I.fig4_record_b_exact.cites[0], $D["v27.fig4_record_b.L3171.1"].cite, $I.fig5_option_a.cites[0], $D["v27.fig5_option_a.L3074.1"].cite, $rc.ruling_cite] | map(gl(.)) | join(", ")).",
    "Printed rows: \($s | length); FINAL \([$s[] | select(. == "F")] | length); PRELIMINARY \([$s[] | select(. == "P")] | length); not route C \([$s[] | select(. == "N")] | length).",
    "Fig 4: \($l4.label) by record B (GATES \(gl($I.fig4_record_b_exact.cites[0]))); label \($l4.values_n) values, \($l4.preliminary_elements_n) PRELIMINARY, \($l4.final_elements_n_in_exclusions) FINAL in exclusions; disclosure \($l4.disclosures[0]) (GATES \(gl($D[$l4.disclosures[0]].cite))); the output moved to the FINAL outputs.",
    "Fig 5: \([$r.figure_labels[] | select(.output == "Fig 5")][0].label) by Aswin's option A (GATES \(gl($I.fig5_option_a.cites[0]))); \([$t5.elements[] | select(.status == "FINAL")] | length) FINAL elements + \([$t5.elements[] | select(.status | startswith("caption-only"))] | length) caption-only, not plotted; disclosure \($I.fig5_option_a.disclosures[0]) (GATES \(gl($D[$I.fig5_option_a.disclosures[0]].cite))); the output moved to the FINAL outputs.",
    "rc1b: \($rc.status) (GATES \(gl($rc.ruling_cite))); out of scope: \($rc.out_of_scope_keys | join(", ")).",
    "Disclosures: \($r.disclosures | length) (v2.6's 64 + 2). status_superseded_by (keys only): 2 new (v26.fig4_record_b.L3001.1, saturated_undefined), 3 extended (INT-FIG.1 / .2, C3_downstream_judgment); \([$r.disclosures[] | select(has("status_superseded_by"))] | length) disclosures carry it.",
    "",
    "In-scope records: \([$r.in_scope_records[].name] | join(", ")). FINAL outputs: \([$r.in_scope_outputs[].name] | join(", ")).",
    "PRELIMINARY outputs / printed items: \([$r.preliminary_outputs[].name] | join(", ")).",
    "Out of scope: \([$r.out_of_scope_records[] | "\(.name) (\(.status))"] | join("; ")).",
    "Exclusions inside FINAL outputs (Correction 3 to A21, amended): \([$r.exclusions[] | "\(.id) \(.entries_n - .flipped_n) PRELIMINARY of \(.entries_n)"] | join("; ")).",
    "Figure labels: \([$r.figure_labels[] | "\(.file | bn | sub("_source\\.json$"; "")) \(.label)"] | join("; ")).",
    (([$r.row_rulings_v2_4.items[] | .disclosure_rows // {} | .["v24.v23_interval_corrections.INT-CORR.2"] // empty] | add // []) as $sr
     | ([$r.printed_items.all_rows[] | select(.row as $n | $sr | index([$n]) != null)]) as $srr
     | ([$srr[] | select(.status | startswith("PRELIMINARY"))] | length) as $srp
     | ([$srr[] | select(.status | startswith("FINAL"))] | length) as $srf
     | if ($sa | length) > 0 and ([$sa[] | select(.status == "FINAL")] | length) == ($sa | length) and $srp == 0 then
         "Sensitivity: in sensitivity_rows the \($sa | length) rc1a sensitivity slots are FINAL (their sensitivity_halves, unchanged from v2.3); the \($srf) printed rows that rest on them (\($sr | map(tostring) | join(", "))) are FINAL by RULING A (interval ends)."
       else "Sensitivity: \(($sa | length) - ([$sa[] | select(.status == "FINAL")] | length)) of \($sa | length) rc1a sensitivity slots stay PRELIMINARY; \($srp) of the \($sr | length) printed rows that rest on them are PRELIMINARY." end),
    "counts_check: the App A printed counts (rows 250/251) are FINAL with disclosure; printed_numbers rows \([$r.counts_check.printed_numbers[] | select(.status | startswith("FINAL")) | .row] | sort | map(tostring) | join(", ")) are FINAL; the rule text says so.",
    "Tooling: experiments/2026-10-14_priority1_spec_09/final_release/v2_7_tooling/ (\($NTOOL) files: build/, selfcheck/; keys, shas, counts and class words only).",
    "",
    "Labels change only after 09 verifies this record by key.",
    "",
    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" ] | join("\n")
