# msg_v25_check.jq: the v2.5-specific checks of the release commit message (the v2.3 message checks run in sc_rest_v25.jq via sc_core.zsh).
# Independent of msg_v25_make.jq: its own constants (09's v2.5 rulings), counts re-derived from the record. Output {n, bad: [check names]}.
#   jq -n -f msg_v25_check.jq --slurpfile REC <record> --rawfile MSG <message> --arg RECSHA <sha256> --argjson NTOOL <files in v2_5_tooling/>
# c5 (09 via 2e): the erratum line and the axis "now applies" note, copied by script from build/c5_strings_09_ruling.json (668b7c06…)
def C5_ERRATUM: "v2.4 erratum: the Figs 1/6 downstream tag and label notes said 'FINAL per element' while Figs 1/6 is PRELIMINARY (interval ends); corrected in v2.5; no status changes";
def C5_AXIS_NOW: "(now applies: Figs 7/8 are in this version)";
def RECPATH: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.5_2026-10-21.json";
def COAUTH: "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>";
def pfx: if startswith("FINAL") then "F" elif startswith("PRELIMINARY") then "P" elif startswith("not route C") then "N" else "O" end;
$REC[0] as $r | ($MSG | split("\n") | if .[-1] == "" then .[0:-1] else . end) as $L
| [$r.printed_items.all_rows[] | .status | pfx] as $s
| ($r.row_rulings_v2_5.figs78.included == true) as $has78
| [ {k: "subject", ok: ($L[1] == "" and $L[0] == (if $has78 then "Route C FINAL-release record, version 2.5 (v2.4 + rows 119-121 FINAL for draft 0029's visible text; Figs 7/8 and slot 24 FINAL)"
                                                   else "Route C FINAL-release record, version 2.5 (v2.4 + rows 119-121 FINAL for draft 0029's visible text)" end))},
    {k: "record path and sha", ok: ($MSG | contains(RECPATH + " (sha256 " + $RECSHA + ")"))},
    {k: "record sha is 64 hex", ok: ($RECSHA | test("^[0-9a-f]{64}$"))},
    {k: "supersedes v2.4", ok: ($MSG | contains("routec_FINAL_release_v2.4_2026-10-21.json @ 6a46541d5 (f14031d6)"))},
    {k: "A2 provenance line", ok: ($MSG | contains("printed_items.provenance stays v18 (provenance_v84_final18_0141.jsonl @ 58bf4a3ac, 13d29fb9) for every row; rows 119-121's printed token and anchor come from v19 (provenance_v84_final19_X.jsonl @ c20699a5d, a8bf8fce), A2."))},
    {k: "GATES commit and lines", ok: ($MSG | contains(if $has78 then "GATES_routec_2026-10-17.md @ ae77708bf (09's rulings; the v2.5 disclosures are the quoted strings of GATES L1974 and L2267 and L2438, copied from git)."
                                                        else "GATES_routec_2026-10-17.md @ 16cd28aed (09's rulings; the v2.5 disclosures are the quoted strings of GATES L1974 and L2267, copied from git)." end))},
    {k: "row counts", ok: ($MSG | contains("Printed rows: \($s | length); FINAL \([$s[] | select(. == "F")] | length); PRELIMINARY \([$s[] | select(. == "P")] | length); not route C \([$s[] | select(. == "N")] | length)."))},
    {k: "rows 119-121", ok: (($MSG | contains("Rows 119-121: FINAL for draft 0029's visible text (re-verify afe11c809; fresh recheck 3f7770c1 CONFIRMED, from agent summary); v2.4 had them PRELIMINARY."))
                            and all($r.printed_items.all_rows[] | select(.row >= 119 and .row <= 121); (.status | pfx) == "F"))},
    {k: "figs78 branch", ok: (if $has78 then (($MSG | contains("Figs 7/8 and slot 24 (rows 69-74, 241-248): FINAL (figs78 branch (i): results commit 854f901b6 (record ee65f33f), verified by 09 at GATES L2430; fresh recheck a0492000 CONFIRMED, from agent summary); v2.4 had them PRELIMINARY."))
                                               and all($r.printed_items.all_rows[] | select((.row >= 69 and .row <= 74) or (.row >= 241 and .row <= 248)); (.status | pfx) == "F"))
                             else ($MSG | contains("Figs 7/8 and slot 24 (rows 69-74, 241-248): PRELIMINARY (figs78 branch (ii): no 09 verification line at freeze time; v2.6).")) end)},
    {k: "switch", ok: (($MSG | contains("R_node switch: OFF.")) and $r.row_rulings_v2_4.r_node_switch.state == "OFF")},
    {k: "tooling path and count", ok: ($MSG | contains("final_release/v2_5_tooling/ (\($NTOOL) files"))},
    {k: "v2.5 disclosure ids", ok: ($MSG | contains(if $has78 then "Disclosures (v2.5): v25.rows_119_121.B1X.1, v25.rows_119_121.rkg.1, v25.figs78.pins.1, v24.figs78_axis_presentation.axis.1 " + C5_AXIS_NOW + "."
                                                     else "Disclosures (v2.5): v25.rows_119_121.B1X.1, v25.rows_119_121.rkg.1." end))},
    # c5 (09 via 2e): the v2.4 erratum line, exactly once, right after the interval-rule line
    {k: "v2.4 erratum line", ok: (([$L[] | select(. == C5_ERRATUM)] | length) == 1
                                  and ([$L | to_entries[] | select(.value == C5_ERRATUM) | .key][0] as $i | $L[$i - 1] | startswith("Interval rule (09, R_component):")))},
    {k: "no PENDING", ok: ($MSG | contains("PENDING") | not)},
    {k: "no Claude-Session line", ok: ([$L[] | select(startswith("Claude-Session:"))] | length == 0)},
    {k: "exactly one Co-Authored-By, as the last line", ok: ([$L[] | select(startswith("Co-Authored-By:"))] == [COAUTH] and $L[-1] == COAUTH)},
    {k: "line count", ok: (($L | length) == 25)},
    {k: "no percent token", ok: ($MSG | test("[0-9]%") | not)},
    {k: "no decimal token other than the versions (outside the trailer)", ok: ([$L[] | select(. != COAUTH)] | join("\n") | [match("(?<![0-9A-Za-z._])[0-9]+\\.[0-9]+(?![0-9])"; "g") | .string] - ["2.5", "2.4", "2.3"] == [])} ] as $c
| {n: ($c | length), bad: [$c[] | select(.ok != true) | "message: " + .k]}
