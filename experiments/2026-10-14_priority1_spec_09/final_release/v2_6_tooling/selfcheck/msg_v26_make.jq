# msg_v26_make.jq: the v2.6 release commit message, generated from the record by key (counts, ids, shas, names and labels only).
#   jq -n -r -f msg_v26_make.jq --slurpfile REC <record> --arg RECSHA <sha256> --argjson NTOOL <number of files in v2_6_tooling/>
# The Claude-Session trailer is NOT in the message: the commit RUN_ME appends it from its first argument. The block from "In-scope records"
# to the counts_check line is the v2.4 / v2.5 generator's; its sensitivity line states the printed rows on the sensitivity slots as they are
# now (FINAL in v2.6). The v2.5-news lines are replaced by the v2.6 lines.
def bn: split("/") | last;
def pf: if startswith("FINAL") then "F" elif startswith("PRELIMINARY") then "P" elif startswith("not route C") then "N" else "O" end;
$REC[0] as $r
| ([$r.printed_items.all_rows[] | .status | pf]) as $s
| $r.row_rulings_v2_4.r_node_switch as $sw
| ([$r.sensitivity_rows.per_output[]?.entries[]?]) as $sa
| ([$r.row_rulings_v2_6.items[] | select(.id == "intervals_option3")][0]) as $it
| ([$r.disclosures[] | select(.id | startswith("v26."))]) as $d26
| ([$it.rows[] | . as $n | select($sw.rows_v2_3_corrections | index([$n]) != null)] | length) as $ncorr
| ($it.cites[0]) as $rc
| ([$r.row_rulings_v2_6.items[] | select(.id == "fig4_record_b")][0].cites[0]) as $rb
| ([$r.element_tables[] | select(.output == "Figs 1/6")]) as $t16
| [ "Route C FINAL-release record, version 2.6 (v2.5 + the \($it.rows | length) printed interval rows and Figs 1/6 FINAL; Fig 4 stays PRELIMINARY)",
    "",
    "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.6_2026-10-21.json (sha256 \($RECSHA)) was built by script from git, by key (v2_6_tooling/build/build_v2_6.zsh + build_v2_6.jq), and self-checked: SC6 (the v2.6 invariants against v2.5, re-derived from constants and git) with its negatives, the ported self-check (sc_rest_v26) with its negatives, the message checks, and the tooling gate (keys, shas, counts and class words only).",
    "It supersedes routec_FINAL_release_v2.5_2026-10-21.json @ \($r.supersedes.commit[0:9]) (\($r.supersedes.sha256[0:8])).",
    "GATES: GATES_routec_2026-10-17.md @ \($rc.commit[0:9]) (09's option (3) ruling L\($rc.line), RULING A L\($it.cites[1].line), RULING B L\($rb.line); the v2.6 disclosures are the quoted strings of \([$d26[] | "L\(.cite.line) (@ \(.cite.commit[0:9]))"] | join(" and ")), copied from git).",
    "Printed rows: \($s | length); FINAL \([$s[] | select(. == "F")] | length); PRELIMINARY \([$s[] | select(. == "P")] | length); not route C \([$s[] | select(. == "N")] | length).",
    "Interval rows: the \($it.rows | length) printed interval rows (\($ncorr) v2.4 corrections + \(($it.rows | length) - $ncorr) interval-only) are FINAL by RULING A (results commit \($it.evidence[0].commit[0:9]), summary \($it.evidence[0].sha256[0:8]); recheck (a) \($it.rechecks[0].report.sha256[0:8]) and recheck (b) \($it.rechecks[1].committed.sha256[0:8]) @ \($it.rechecks[1].committed.commit[0:9])); v2.5 had them PRELIMINARY.",
    "Figs 1/6: every plotted element is FINAL (\([$t16[] | .elements | length] | map(tostring) | join(" + ")) elements; \([$t16[] | [.elements[] | select(.status == "FINAL")] | length] | add) FINAL); both labels FINAL; the output moved to the FINAL outputs.",
    "Fig 4: stays PRELIMINARY (RULING B, GATES L\($rb.line)); its label reason and both interval rules carry 09's text.",
    "Disclosures (v2.6): \([$d26[] | .id + (if (.id | startswith("v26.option3_intervals.")) then " (the \($it.rows | length) rows and Figs 1/6)" elif (.id | startswith("v26.fig4_record_b.")) then " (Fig 4: RULING B, GATES L\(.cite.line) @ \(.cite.commit[0:9]))" else "" end)] | join(", ")). status_superseded_by (keys only): \([$r.disclosures[] | select(has("status_superseded_by"))] | length) disclosures and \([$r.row_rulings_v2_4.items[] | select(has("status_superseded_by"))] | length) v2.4 items. R_node switch: \($sw.state), superseded_by_ruling GATES L\($sw.superseded_by_ruling.line).",
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
       elif ($sa | length) > 0 and ([$sa[] | select(.status == "FINAL")] | length) == ($sa | length) then
         "Sensitivity: in sensitivity_rows the \($sa | length) rc1a sensitivity slots are FINAL (their sensitivity_halves, unchanged from v2.3); \($srp) of the \($sr | length) printed rows that rest on them (\($sr | map(tostring) | join(", "))) are PRELIMINARY."
       else "Sensitivity: \(($sa | length) - ([$sa[] | select(.status == "FINAL")] | length)) of \($sa | length) rc1a sensitivity slots stay PRELIMINARY; \($srp) of the \($sr | length) printed rows that rest on them are PRELIMINARY." end),
    "counts_check: the App A printed counts (rows 250/251) are FINAL with disclosure; printed_numbers rows \([$r.counts_check.printed_numbers[] | select(.status | startswith("FINAL")) | .row] | sort | map(tostring) | join(", ")) are FINAL; the rule text says so.",
    "Tooling: experiments/2026-10-14_priority1_spec_09/final_release/v2_6_tooling/ (\($NTOOL) files: build/, selfcheck/; keys, shas, counts and class words only).",
    "",
    "Labels change only after 09 verifies this record by key.",
    "",
    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" ] | join("\n")
