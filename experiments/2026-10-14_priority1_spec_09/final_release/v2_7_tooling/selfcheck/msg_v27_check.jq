# msg_v27_check.jq: the v2.7-specific checks of the release commit message (the v2.3 message checks run in sc_rest_v27.jq via sc_core.zsh).
# Independent of msg_v27_make.jq: its own constants (09's v2.7 rulings; pre-statement fd03b8c0…), counts re-derived from the record, full
# 40-character commit compares of the record's v2.7 cites (V26 follow-up 72). Output {n, bad: [check names]}.
#   jq -n -f msg_v27_check.jq --slurpfile REC <record> --rawfile MSG <message> --arg RECSHA <sha256> --argjson NTOOL <files in v2_7_tooling/>
def RECPATH: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.7_2026-10-21.json";
def COAUTH: "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>";
def G80: "f80f3cfe6f1543318f933045c2559cb1cab17441"; def G0A: "0a719afaa0a74a8d666b87f1591725bfdd371f4e"; def G97: "97b6e634da7f42e351ff0bbfbedc1f72e33efca0";
def pfx: if startswith("FINAL") then "F" elif startswith("PRELIMINARY") then "P" elif startswith("not route C") then "N" else "O" end;
$REC[0] as $r | ($MSG | split("\n") | if .[-1] == "" then .[0:-1] else . end) as $L
| [$r.printed_items.all_rows[] | .status | pfx] as $s
| ($r.disclosures | map({(.id): .}) | add) as $D
| [ {k: "subject", ok: ($L[0] == "Route C FINAL-release record, version 2.7 (v2.6 + Fig 4, Fig 5 and rc1b FINAL)" and $L[1] == "")},
    {k: "record path and sha", ok: ($MSG | contains(RECPATH + " (sha256 " + $RECSHA + ")"))},
    {k: "record sha is 64 hex", ok: ($RECSHA | test("^[0-9a-f]{64}$"))},
    {k: "supersedes v2.6", ok: ($MSG | contains("It supersedes routec_FINAL_release_v2.6_2026-10-21.json @ 0db7dc714 (2db9cdee)."))},
    {k: "GATES cites", ok: ($MSG | contains("GATES: GATES_routec_2026-10-17.md (09's v2.7 rulings; the cited lines, by script from git): L3160 @ f80f3cfe6, L3171 @ f80f3cfe6, L3060 @ 0a719afaa, L3074 @ 0a719afaa, L2982 @ 97b6e634d."))},
    {k: "record cites carry the full ruled commits", ok: ($D["v27.fig4_record_b.L3171.1"].cite.commit == G80 and $D["v27.fig5_option_a.L3074.1"].cite.commit == G0A
          and ([$r.in_scope_outputs[] | select(.name == "rc1b") | .ruling_cite.commit] == [G97])
          and ([$r.row_rulings_v2_7.items[] | .cites[0].commit] == [G80, G0A, G97]))},
    {k: "row counts", ok: (($MSG | contains("Printed rows: \($s | length); FINAL \([$s[] | select(. == "F")] | length); PRELIMINARY \([$s[] | select(. == "P")] | length); not route C \([$s[] | select(. == "N")] | length)."))
                           and ([$s[] | select(. == "F")] | length) == 237 and ([$s[] | select(. == "P")] | length) == 2)},
    {k: "Fig 4", ok: (($MSG | contains("Fig 4: FINAL by record B (GATES L3160 @ f80f3cfe6); label 264 values, 0 PRELIMINARY, 264 FINAL in exclusions; disclosure v27.fig4_record_b.L3171.1 (GATES L3171 @ f80f3cfe6); the output moved to the FINAL outputs."))
                      and all($r.figure_labels[] | select(.output == "Fig 4"); .label == "FINAL"))},
    {k: "Fig 5", ok: (($MSG | contains("Fig 5: FINAL by Aswin's option A (GATES L3060 @ 0a719afaa); 68 FINAL elements + 6 caption-only, not plotted; disclosure v27.fig5_option_a.L3074.1 (GATES L3074 @ 0a719afaa); the output moved to the FINAL outputs."))
                      and all($r.figure_labels[] | select(.output == "Fig 5"); .label == "FINAL"))},
    {k: "rc1b", ok: ($MSG | contains("rc1b: FINAL (printed items only) (GATES L2982 @ 97b6e634d); out of scope: DETDROP, DETDROP_label, PLANT, PLANT_label, LABELS_171."))},
    {k: "disclosures", ok: (($MSG | contains("Disclosures: 66 (v2.6's 64 + 2). status_superseded_by (keys only): 2 new (v26.fig4_record_b.L3001.1, saturated_undefined), 3 extended (INT-FIG.1 / .2, C3_downstream_judgment); 10 disclosures carry it."))
                            and ($r.disclosures | length) == 66)},
    {k: "every GATES cite carries its commit", ok: all($L[] | select(contains("GATES")) | . as $line | [match("(?<![A-Za-z0-9._])L[0-9]{4}(-[0-9]{4})?"; "g")][] | $line[(.offset + .length):(.offset + .length + 13)]; test("^ @ [0-9a-f]{9}"))},
    {k: "sensitivity rows FINAL", ok: (($MSG | contains("printed rows that rest on them")) and ($MSG | contains("are FINAL by RULING A (interval ends).")))},
    {k: "tooling path and count", ok: ($MSG | contains("final_release/v2_7_tooling/ (\($NTOOL) files"))},
    {k: "no PENDING", ok: ($MSG | contains("PENDING") | not)},
    {k: "no Claude-Session line", ok: ([$L[] | select(startswith("Claude-Session:"))] | length == 0)},
    {k: "exactly one Co-Authored-By, as the last line", ok: ([$L[] | select(startswith("Co-Authored-By:"))] == [COAUTH] and $L[-1] == COAUTH)},
    {k: "line count", ok: (($L | length) == 23)},
    {k: "no percent token", ok: ($MSG | test("[0-9]%") | not)},
    {k: "no decimal token other than the versions (outside the trailer)", ok: ([$L[] | select(. != COAUTH)] | join("\n") | [match("(?<![0-9A-Za-z._])[0-9]+\\.[0-9]+(?![0-9])"; "g") | .string] - ["2.7", "2.6", "2.5", "2.4", "2.3"] == [])} ] as $c
| {n: ($c | length), bad: [$c[] | select(.ok != true) | "message: " + .k]}
