# msg_v24_check.jq: the v2.4-specific checks of the release commit message (the v2.3 message checks run in sc_rest_v24.jq via sc_core.zsh).
# Independent of msg_v24_make.jq: its own constants (09's rulings), counts re-derived from the record. Output {n, bad: [check names]}.
#   jq -n -f msg_v24_check.jq --slurpfile REC <record> --rawfile MSG <message> --arg RECSHA <sha256> --argjson NTOOL <files in v2_4_tooling/>
def CORR12: [4, 5, 9, 10, 14, 15, 19, 20, 25, 27, 28, 29];
def RECPATH: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.4_2026-10-21.json";
def COAUTH: "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>";
def pfx: if startswith("FINAL") then "F" elif startswith("PRELIMINARY") then "P" elif startswith("not route C") then "N" else "O" end;
$REC[0] as $r | ($MSG | split("\n") | if .[-1] == "" then .[0:-1] else . end) as $L
| [$r.printed_items.all_rows[] | .status | pfx] as $s
| ($r.row_rulings_v2_4.r_node_switch.state == "OFF") as $off
| ([$r.printed_items.all_rows[] | select(.row == 122)][0].status | startswith("FINAL")) as $f122
| ([($r.element_tables // [])[] | select(.kind == "figs78_item")] | length > 0) as $has78
| [ {k: "subject", ok: ($L[0] == "Route C FINAL-release record, version 2.4 (v2.3 + the v2.4 rulings, per row)" and $L[1] == "")},
    {k: "record path and sha", ok: ($MSG | contains(RECPATH + " (sha256 " + $RECSHA + ")"))},
    {k: "record sha is 64 hex", ok: ($RECSHA | test("^[0-9a-f]{64}$"))},
    {k: "supersedes v2.3", ok: ($MSG | contains("routec_FINAL_release_v2.3_2026-10-20.json @ 80a77ecdc (ebb47e28)"))},
    {k: "v18 pin", ok: ($MSG | contains("provenance_v84_final18_0141.jsonl @ 58bf4a3ac (13d29fb9)"))},
    {k: "GATES commit = the disclosures' cite commit", ok: ($MSG | contains("GATES_routec_2026-10-17.md @ " + ([$r.disclosures[] | select(.id | startswith("v24.")) | .cite.commit[0:9]] | unique | join(", ")) + " ("))},
    {k: "row counts", ok: ($MSG | contains("Printed rows: \($s | length); FINAL \([$s[] | select(. == "F")] | length); PRELIMINARY \([$s[] | select(. == "P")] | length); not route C \([$s[] | select(. == "N")] | length)."))},
    {k: "the 12 corrected rows", ok: ($MSG | contains("v2.3 rows FINAL -> " + (if $off then "PRELIMINARY" else "FINAL" end) + ": " + (CORR12 | map(tostring) | join(", ")) + ";"))},
    {k: "the 53 interval rows", ok: ($MSG | contains("the 53 interval rows are " + (if $off then "PRELIMINARY" else "FINAL" end)))},
    {k: "switch state", ok: ($MSG | contains("R_node switch: " + (if $off then "OFF" else "ON" end) + "."))},
    {k: "row 122", ok: (if $f122 then ($MSG | contains("row 122 FINAL (v18 pinned; fresh recheck 870ed0de CONFIRMED of the verifier results commit fa4a475c9, from agent summary); rows 119-121 PRELIMINARY."))
                         else ($MSG | contains("Rows 119-122: PRELIMINARY")) end)},
    {k: "figs78", ok: (if $has78 then ($MSG | contains("Figs 7/8 and slot 24 (rows 69-74, 241-248): FINAL")) else ($MSG | contains("Figs 7/8 and slot 24 (rows 69-74, 241-248): PRELIMINARY")) end)},
    {k: "tooling path and count", ok: ($MSG | contains("final_release/v2_4_tooling/ (\($NTOOL) files"))},
    {k: "every v2.4 disclosure id listed", ok: ($MSG | contains("Disclosures (v2.4): " + ([$r.disclosures[] | select(.id | startswith("v24.")) | .id] | join(", ")) + "."))},
    {k: "no PENDING", ok: ($MSG | contains("PENDING") | not)},
    # b8 (09; fresh review N2): the sensitivity line carries both true counts: slot entries FINAL, and the 4 printed rows PRELIMINARY
    {k: "sensitivity counts", ok: (([$r.sensitivity_rows.per_output[]?.entries[]?]) as $sa | ([$r.printed_items.all_rows[] | select(.row as $n | [25, 27, 28, 29] | index([$n]) != null) | .status | pfx] | unique) as $sp
        | $sp == ["P"] and ($MSG | contains("in sensitivity_rows the \($sa | length) rc1a sensitivity slots are FINAL (their sensitivity_halves, unchanged from v2.3); the 4 printed rows that rest on them (25, 27, 28, 29) are PRELIMINARY by [INT-CORR]"))
        and ([$sa[] | select(.status == "FINAL")] | length) == ($sa | length))},
    {k: "counts_check line", ok: ($MSG | contains("counts_check: the App A printed counts (rows 250/251) are FINAL with disclosure; printed_numbers rows 75, 76, 77, 78, 92, 95, 96, 97, 98 are FINAL"))},
    {k: "line count", ok: (($L | length) == 24)},
    {k: "no Claude-Session line", ok: ([$L[] | select(startswith("Claude-Session:"))] | length == 0)},
    {k: "exactly one Co-Authored-By, as the last line", ok: ([$L[] | select(startswith("Co-Authored-By:"))] == [COAUTH] and $L[-1] == COAUTH)},
    {k: "no percent token", ok: ($MSG | test("[0-9]%") | not)},
    {k: "no decimal token other than the versions (outside the trailer)", ok: ([$L[] | select(. != COAUTH)] | join("\n") | [match("(?<![0-9A-Za-z._])[0-9]+\\.[0-9]+(?![0-9.])"; "g") | .string] - ["2.4", "2.3"] == [])} ] as $c
| {n: ($c | length), bad: [$c[] | select(.ok != true) | "message: " + .k]}
