# msg_v26_check.jq: the v2.6-specific checks of the release commit message (the v2.3 message checks run in sc_rest_v26.jq via sc_core.zsh).
# Independent of msg_v26_make.jq: its own constants (09's v2.6 rulings; pre-statement 33a40662…), counts re-derived from the record.
# Output {n, bad: [check names]}.
#   jq -n -f msg_v26_check.jq --slurpfile REC <record> --rawfile MSG <message> --arg RECSHA <sha256> --argjson NTOOL <files in v2_6_tooling/>
def RECPATH: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.6_2026-10-21.json";
def COAUTH: "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>";
def pfx: if startswith("FINAL") then "F" elif startswith("PRELIMINARY") then "P" elif startswith("not route C") then "N" else "O" end;
$REC[0] as $r | ($MSG | split("\n") | if .[-1] == "" then .[0:-1] else . end) as $L
| [$r.printed_items.all_rows[] | .status | pfx] as $s
| ([$r.element_tables[] | select(.output == "Figs 1/6")]) as $t16
| [ {k: "subject", ok: ($L[0] == "Route C FINAL-release record, version 2.6 (v2.5 + the 65 printed interval rows and Figs 1/6 FINAL; Fig 4 stays PRELIMINARY)" and $L[1] == "")},
    {k: "record path and sha", ok: ($MSG | contains(RECPATH + " (sha256 " + $RECSHA + ")"))},
    {k: "record sha is 64 hex", ok: ($RECSHA | test("^[0-9a-f]{64}$"))},
    {k: "supersedes v2.5", ok: ($MSG | contains("It supersedes routec_FINAL_release_v2.5_2026-10-21.json @ 4df86c4b9 (0d16330e)."))},
    {k: "GATES commit and lines", ok: ($MSG | contains("GATES: GATES_routec_2026-10-17.md @ 4d43ba0a4 (09's option (3) ruling L2869, RULING A L2876, RULING B L2877; the v2.6 disclosures are the quoted strings of L2878 (@ 4d43ba0a4) and L3001 (@ 89435ec9a), copied from git)."))},
    {k: "row counts", ok: (($MSG | contains("Printed rows: \($s | length); FINAL \([$s[] | select(. == "F")] | length); PRELIMINARY \([$s[] | select(. == "P")] | length); not route C \([$s[] | select(. == "N")] | length)."))
                          and ([$s[] | select(. == "F")] | length) == 237 and ([$s[] | select(. == "P")] | length) == 2)},
    {k: "interval rows", ok: ($MSG | contains("Interval rows: the 65 printed interval rows (12 v2.4 corrections + 53 interval-only) are FINAL by RULING A (results commit 7d787550b, summary 8283c81a; recheck (a) a2603440 and recheck (b) cb53eef3 @ d1d74901a); v2.5 had them PRELIMINARY."))},
    {k: "Figs 1/6", ok: (($MSG | contains("Figs 1/6: every plotted element is FINAL (96 + 96 elements; 192 FINAL); both labels FINAL; the output moved to the FINAL outputs."))
                        and all($t16[] | .elements[]; .status == "FINAL") and all($r.figure_labels[] | select(.output == "Figs 1/6"); .label == "FINAL"))},
    {k: "Fig 4", ok: (($MSG | contains("Fig 4: stays PRELIMINARY (RULING B, GATES L2877); its label reason and both interval rules carry 09's text."))
                     and all($r.figure_labels[] | select(.output == "Fig 4"); .label == "PRELIMINARY"))},
    {k: "disclosures and keys", ok: ($MSG | contains("Disclosures (v2.6): v26.option3_intervals.L2878.1 (the 65 rows and Figs 1/6), v26.fig4_record_b.L3001.1 (Fig 4: RULING B, GATES L3001 @ 89435ec9a). status_superseded_by (keys only): 8 disclosures and 3 v2.4 items. R_node switch: OFF, superseded_by_ruling GATES L2869."))},
    {k: "v26 disclosure cites (09 ruling: L2878 @ 4d43ba0a4, L3001 @ 89435ec9a)", ok: ([$r.disclosures[] | select(.id | startswith("v26.")) | {id, l: .cite.line, c: (.cite.commit | .[0:9])}]
                                == [{id: "v26.option3_intervals.L2878.1", l: 2878, c: "4d43ba0a4"}, {id: "v26.fig4_record_b.L3001.1", l: 3001, c: "89435ec9a"}])},
    {k: "sensitivity rows FINAL", ok: (($MSG | contains("printed rows that rest on them")) and ($MSG | contains("are FINAL by RULING A (interval ends).")) and ($MSG | contains("are PRELIMINARY by [INT-CORR]") | not))},
    {k: "tooling path and count", ok: ($MSG | contains("final_release/v2_6_tooling/ (\($NTOOL) files"))},
    {k: "no v2.5-news line", ok: ([$L[] | select(startswith("Rows 119-121:") or startswith("Figs 7/8 and slot 24") or startswith("v2.4 erratum:") or startswith("Provenance:") or startswith("Interval rule (09, R_component)"))] | length == 0)},
    {k: "no PENDING", ok: ($MSG | contains("PENDING") | not)},
    {k: "no Claude-Session line", ok: ([$L[] | select(startswith("Claude-Session:"))] | length == 0)},
    {k: "exactly one Co-Authored-By, as the last line", ok: ([$L[] | select(startswith("Co-Authored-By:"))] == [COAUTH] and $L[-1] == COAUTH)},
    {k: "line count", ok: (($L | length) == 23)},
    {k: "no percent token", ok: ($MSG | test("[0-9]%") | not)},
    {k: "no decimal token other than the versions (outside the trailer)", ok: ([$L[] | select(. != COAUTH)] | join("\n") | [match("(?<![0-9A-Za-z._])[0-9]+\\.[0-9]+(?![0-9])"; "g") | .string] - ["2.6", "2.5", "2.4", "2.3"] == [])} ] as $c
| {n: ($c | length), bad: [$c[] | select(.ok != true) | "message: " + .k]}
