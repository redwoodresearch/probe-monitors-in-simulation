# SC5: the v2.5 invariants against the committed v2.4 record, re-derived independently of build_v2_5.jq (its own constants, typed from 09's
# rulings via 2e and GATES @ ae77708bf; it never reads rulings_v2_5.json). Keys, counts and booleans only. Output {n: {...}, bad: [...]}.
# Inputs: --slurpfile REC draft  --slurpfile V24 v2.4 record  --slurpfile P18 / P19 provenance rows  --slurpfile LN {k: "commit:file:line", v}
#         --argjson A2OK (a2_check.jq)  --argjson RVOK (the driver's re-check of the RV report and verifier blob shas)
#         --argjson F78LINE (the driver: 09's "figs78 results commit <sha> verified" GATES line present once, at L2430 @ ae77708bf)
#         --slurpfile F78R (the figs78 record read from git at 854f901b6)  --argjson F78ROK (its sha is the pinned one)
#         --argjson F78RCOK (the driver's re-check of the figs78 recheck report: sha as recorded, CONFIRMED line present)
def W($c; $w): if $c then empty else $w end;
def pfx: if startswith("FINAL") then "FINAL" elif startswith("PRELIMINARY") then "PRELIMINARY" elif startswith("not route C") then "not route C" else "OTHER" end;
def quotes: [match("\"([^\"]+)\""; "g") | .captures[0].string];
# ---- constants (09 via 2e; GATES @ ae77708bf, whose L1-2400 are 16cd28aed's) ----
def R3: [119, 120, 121];
def S24: [69, 70, 71, 72, 73, 74] + [range(241; 249)];
def RECP: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.5_2026-10-21.json";
def V24P: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.4_2026-10-21.json";
def V24C: "6a46541d52b58f9626e14e7cc96339bb5d36ee3d";
def V24SHA: "f14031d65dfe090f53d9e28045a200cff4b875bb41e60f8f7ebdaeba989c2c91";
def GATES_C: "ae77708bf8fa980ec83acc0c456b89a365b95066";   # c2: GATES re-pinned to the commit holding 09's figs78 lines (L1-2400 = 16cd28aed)
def V19: {path: "experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final19_X.jsonl", commit: "c20699a5d4d4aed5344650fec54d9c166f421296",
          sha256: "a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda"};
def RV: {sha256: "3f7770c15a2f44ec216d9a2a929c0117241a45d7ab239505e836b78d4706ed1b", label: "from agent summary", verdict: "CONFIRMED",
         of: {commit: "afe11c8096efb4acfdf9eca1efa94959c432b41a", branch: "aswin/routec-numbers-verify", parent: "f11453af93dee8a32bd7e4befb90d8a590341033",
              path: "experiments/2026-10-20_routec_numbers_verify/results/numbers_verify_2026-10-21_025253.json",
              sha256: "d59138c43422e2e50766611e1781737075bc9aaf9d10af1211a30c7deef7e618"}};
# the two v2.5 disclosures: id -> [GATES line, quoted-string index]; attached to rows 119-121 only (ITEM_DISC25 / ITEM_ROWS25, #9)
def DISC25: {"v25.rows_119_121.B1X.1": [1974, 1], "v25.rows_119_121.rkg.1": [2267, 1], "v25.figs78.pins.1": [2438, 1]};
def ITEM_DISC25: {"rows_119_121": ["v25.rows_119_121.B1X.1", "v25.rows_119_121.rkg.1"], "figs78": ["v25.figs78.pins.1"]};
def ITEM_ROWS25: {"rows_119_121": [119, 120, 121], "figs78": [69, 70, 71, 72, 73, 74, 241, 242, 243, 244, 245, 246, 247, 248]};
# figs78 branch (i): the FILL constants are replaced only after 09's "figs78 results commit <sha> verified" line and a CONFIRMED recheck
def F78: {results_commit: "854f901b606cd8b44f30f87c263b441e7101a11f", record_sha256: "ee65f33f9a7856e1be6a6c9043345019785e0c395e1ccd0e0ac8e7f9677e35d1",
          record_path: "experiments/2026-10-20_routec_rerun_v24figs78/results/figs78_map_2026-10-21_033233.json", gates_line_must: "results commit 854f901b6 verified** (commit by key",
          recheck_sha256: "a0492000d52c225dffbe3a79e9fbb591346a5cbf64fdc2c01d276ad2be239835"};
def GATES_F: "experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md";
# branch (i) GATES lines @ GATES_C: the reason (L2426), the recheck (L2427), 09's verified line (L2430); the reason string itself
def F78L: {reason: 2426, recheck: 2427, verified: 2430};
def F78_REASON: "figs78 results commit " + F78.results_commit[0:9] + " verified (fresh recheck CONFIRMED); slot 24 + Figs 7/8 FINAL-eligible for v2.5 branch (i)";
# the fields of each figs78 record element table copied into the release (diffed against build_v2_5.jq's copy list)
def F78_FIELDS: ["file", "commit", "sha256", "figure_status", "inventory_complete", "elements_all_final", "classes", "metadata_not_drawn", "elements"];
def MOVED_IN: ["Figs 7/8", "rc1b printed items, slot 24 (App H / Fig 7-8 captions)"];
def MOVED_PRE: ["Figs 7/8", "rc1b printed items (App H / Fig 7-8 captions)"];
def ALLOWED: ["builder", "date_pt", "disclosures", "figure_labels", "flip_groups", "in_scope_outputs", "preliminary_outputs", "printed_items", "repo_path",
              "row_rulings_v2_5", "superseded_fields", "supersedes", "version"];
def ALLOWED_I: ["element_tables", "figure_labels", "pending_downstream"];
# c4 (09 via 2e): the branch (i) reading, copied by script from build/reading_v25_branch_i_09_ruling.txt (f61ac716…); the v2.4
# figs78_axis_presentation disclosure id, copied by script from the v2.4 record (attached figure-level to the figs78 item, #2)
def READING_I: "v2.5 = v2.4 + rows 119-121 FINAL for draft 0029's visible text (09, GATES L2364) + slot-24 rows 69-74 / 241-248 and Figs 7/8 FINAL (figs78 branch (i); 09, GATES L2430); everything else as v2.4";
def AXIS24: ["v24.figs78_axis_presentation.axis.1"];
def ITEM_DISC25_I: ITEM_DISC25 | .figs78 += AXIS24;
def FIGLEVEL25: {"figs78": AXIS24};
# c4 (09 via 2e): the Figs 7/8 downstream_tag.text = the quoted string of build_v2_4.jq L103 @ 6a46541d5 (read by the driver from git) + this suffix
def TAG_SUFFIX: "; results commit " + F78.results_commit[0:9] + " verified, GATES L" + (F78L.verified | tostring) + " @ " + GATES_C;
# c5 (09 via 2e): the joiner, copied by script from build/c5_strings_09_ruling.json (668b7c06…); the Figs 1/6 leaves left out of the v2.4
# comparisons (S6 / S9) because S12 asserts them exactly
def C5_JOINER: "; point elements: ";
def NOTE16: if .output == "Figs 1/6" then del(.note) else . end;
def TAG16: if .name == "Figs 1/6" then del(.downstream_tag.text) else . end;
def NAMES_IN: ["Tables 3/4","rc1a","rc2","Table 6 plant_rows","honest_all_routec printed items","routec_simpool_numbers printed items (FINAL rows)","rc1b printed items, slots 21/22","attack_sol_share printed item (row 106)","cunits_by_tool printed items (Appendix A)","rc1b_detl48 printed items, slot 23 (rows 119-122)"];
def NAMES_PRE: ["rc1b","Fig 5","Figs 7/8","routec_simpool_numbers printed items","rc1b printed items (App H / Fig 7-8 captions)","Fig 4","Figs 1/6"];
# branch (i) (c2) output names, derived from the c2 draft
def NAMES_IN_I: ["Tables 3/4","rc1a","rc2","Table 6 plant_rows","honest_all_routec printed items","routec_simpool_numbers printed items (FINAL rows)","rc1b printed items, slots 21/22","attack_sol_share printed item (row 106)","cunits_by_tool printed items (Appendix A)","rc1b_detl48 printed items, slot 23 (rows 119-122)","Figs 7/8","rc1b printed items, slot 24 (App H / Fig 7-8 captions)"];
def NAMES_PRE_I: ["rc1b","Fig 5","routec_simpool_numbers printed items","Fig 4","Figs 1/6"];
$REC[0] as $r | $V24[0] as $v
| ($LN | map({(.k): .v}) | add // {}) as $L
| ($P18 | map({(.row | tostring): .}) | add) as $p18 | ($P19 | map({(.row | tostring): .}) | add) as $p19
| ($r.row_rulings_v2_5.figs78.included == true) as $has78
| (if $has78 then DISC25 else (DISC25 | del(.["v25.figs78.pins.1"])) end) as $disc25
| (if $has78 then ITEM_DISC25_I else (ITEM_DISC25 | del(.figs78)) end) as $idisc
| (if $has78 then ITEM_ROWS25 else (ITEM_ROWS25 | del(.figs78)) end) as $irows
| ($L["\(GATES_C):REASON:\(F78L.reason)"] // "") as $rline
| ("FINAL (" + F78_REASON + ")") as $FR
| ({path: F78.record_path, commit: F78.results_commit, sha256: F78.record_sha256}) as $src
| ([$r.figure_labels[] | select(.output == "Figs 7/8")]) as $rl78 | ([$v.figure_labels[] | select(.output == "Figs 7/8")]) as $vl78
| ($v.element_tables | length) as $net
| ([$v.printed_items.all_rows[] | {(.row | tostring): .status}] | add) as $vs
| ([$r.disclosures[] | select(.id | startswith("v25."))]) as $d25
| ([$r.in_scope_outputs[] | select(.name == "rc1b_detl48 printed items, slot 23 (rows 119-122)")]) as $g
| ($r.row_rulings_v2_5.draft_scope) as $ds
| [
  # S1 A2 and the pins
  W($A2OK == true; "S1 A2: v19 vs v18 field differences are not the pre-stated set"),
  W($r.printed_items.provenance == $v.printed_items.provenance; "S1 printed_items.provenance is not v2.4's (v18) pin"),
  # S2 rows: same rows and records as v2.4; statuses change exactly on rows 119-121 (and the slot-24 rows only in branch (i))
  W(([$r.printed_items.all_rows[] | {row, record}]) == ([$v.printed_items.all_rows[] | {row, record}]); "S2 all_rows rows / records differ from v2.4"),
  W(([$r.printed_items.all_rows[] | select(.status != $vs[.row | tostring]) | .row] | sort) == ((R3 + (if $has78 then S24 else [] end)) | sort); "S2 a row's status changed outside rows 119-121"),
  W(all($r.printed_items.all_rows[] | select(.row as $n | R3 | index([$n]) != null); (.status | pfx) == "FINAL"); "S2 rows 119-121 are not FINAL"),
  # S3 the rows 119-122 group: items 119-121 from v19 (A2), 122 from v2.4; no other group holds rows 119-121
  W(($g | length) == 1 and $g[0].rows == [119, 120, 121, 122] and $g[0].status == "FINAL" and $g[0].items_n == 4
    and ($g[0].items[0:3] == [R3[] | tostring as $n | {row: $p19[$n].row, item: $p19[$n].item, anchor: $p19[$n].anchor, key_path: $p19[$n].key_path, record: $p19[$n].record, record_sha256: $p19[$n].record_sha256}])
    and ($g[0].items[3:] == [$v.in_scope_outputs[] | select(.name == "rc1b_detl48 printed item, slot 23 (row 122)") | .items[] | select(.row == 122)]);
    "S3 the rows 119-122 group's items are not v19 (119-121) + v2.4 (122)"),
  W(([($r.in_scope_outputs + $r.preliminary_outputs)[] | select(.kind == "printed_items") | select(any(.rows[]; . as $n | R3 | index([$n]) != null)) | .name] | length) == 1; "S3 another group holds rows 119-121"),
  # S4 disclosures: v2.4's 59 kept; exactly the 2 new ones, each its GATES line's quoted string at the pinned GATES commit
  W($r.disclosures[0:($v.disclosures | length)] == $v.disclosures and ($r.disclosures | length) == ($v.disclosures | length) + (if $has78 then 3 else 2 end);
    "S4 v2.4 disclosures not kept, or not exactly 2 new (3 in branch (i))"),
  W(([$d25[].id] | sort) == ($disc25 | keys); "S4 the v2.5 disclosure ids"),
  ($d25[] | . as $d | ($disc25[$d.id] // null) as $e | ($L["\($d.cite.commit):\($d.cite.file):\($d.cite.line)"] // null) as $t
   | W($e != null and $t != null and $d.cite.commit == GATES_C and $d.cite.line == $e[0] and (($t | quotes)[$e[1] - 1] == $d.text) and (($t | quotes) | length) == 1;
       "S4 disclosure \($d.id) is not its GATES line's quoted string")),
  # S4b (#9) the v2.5 ruling item, its rows and attachments: the pre-stated tables
  W(([$r.row_rulings_v2_5.items[] | {(.id): .disclosures}] | add) == $idisc and ([$r.row_rulings_v2_5.items[] | {(.id): .rows}] | add) == $irows
    and all($r.row_rulings_v2_5.items[]; . as $it | ($it.disclosure_rows | keys) == (($it.disclosures - (FIGLEVEL25[$it.id] // [])) | sort) and all($it.disclosure_rows[]; . == ITEM_ROWS25[$it.id]));
    "S4b the v2.5 ruling item, its rows or its attachments are not the pre-stated table"),
  # S5 draft_scope: the pre-stated block
  W($ds.rows == R3 and $ds.visible_text_of == "draft 0029" and ($ds.verifier_input | test("^X = draft 0141 with draft 0029's [0-9]+ App C digits$"))
    and ([$L[] | select(contains($ds.verifier_input))] | length) >= 1
    and ($ds.printed_and_anchor_source | {path, commit, sha256}) == V19 and $ds.printed_and_anchor_source.rows == R3
    and $ds.printed_and_anchor_source.fields == ["anchor", "printed"] and $ds.printed_and_anchor_source.equal_to_v18_except == ["anchor", "printed", "v84_sha256"]
    and ($ds.recheck | {sha256, label, verdict, of}) == RV and $RVOK == true
    and ([$ds.evidence[] | {path, commit}] | sort) == ([{path: RV.of.path, commit: RV.of.commit}, {path: V19.path, commit: V19.commit}] | sort);
    "S5 draft_scope is not the pre-stated block"),
  # S6 unchanged: every top-level key outside ALLOWED; printed_items beyond the 3 statuses and the counts; the other groups; flip groups 1-3; history
  ((($v | keys) - ALLOWED - (if $has78 then ALLOWED_I else [] end))[] as $k | W($r[$k] == $v[$k]; "S6 top-level key \($k) changed")),
  W((($r | keys) - ($v | keys)) == ["row_rulings_v2_5"]; "S6 new top-level keys other than row_rulings_v2_5"),
  W(($r.printed_items | del(.all_rows, .all_rows_status_counts, .groups_n, .items_n)) == ($v.printed_items | del(.all_rows, .all_rows_status_counts, .groups_n, .items_n));
    "S6 printed_items changed beyond rows 119-121's status, the counts and the group counts"),
  ((if $has78 then MOVED_IN else [] end) as $mi | (if $has78 then MOVED_PRE else [] end) as $mp
   | W(([$r.in_scope_outputs[] | select(.name != "rc1b_detl48 printed items, slot 23 (rows 119-122)") | select(.name as $n | $mi | index([$n]) == null)])
         == ([$v.in_scope_outputs[] | select(.name != "rc1b_detl48 printed item, slot 23 (row 122)")])
       and ([$r.preliminary_outputs[] | TAG16]) == ([$v.preliminary_outputs[] | select(.name != "rc1b_detl48 printed items, slot 23") | select(.name as $n | $mp | index([$n]) == null) | TAG16]);
       "S6 another output changed")),
  W(($r.flip_groups | del(.group_4)) == $v.flip_groups; "S6 flip groups 1-3 changed"),
  W($r.superseded_fields.v2_4_superseded_fields == $v.superseded_fields; "S6 v2.4 superseded_fields not kept in full"),
  # S7 identity
  W($r.version == "2.5" and $r.repo_path == RECP and $r.supersedes.path == V24P and $r.supersedes.commit == V24C and $r.supersedes.sha256 == V24SHA;
    "S7 version / repo_path / supersedes"),
  # S8 (#10) the stored status-count map equals the one recomputed from all_rows
  W($r.printed_items.all_rows_status_counts == ($r.printed_items.all_rows | group_by(.status) | map({(.[0].status): length}) | add);
    "S8 all_rows_status_counts is not the map recomputed from all_rows"),
  # S9 figs78: branch (i) only with 09's verification line and filled constants; branch (ii) = v2.4's Figs 7/8 and slot-24 rows
  (if $has78 then
     W($F78LINE == true and all(F78[]; . != "FILL"); "S9 figs78 branch (i) is not open (no 09 verification line; constants unfilled)"),
     # S9i-0 the reason string occurs exactly once on its GATES line (L2426 @ GATES_C, read by the driver from git)
     W(($rline | split(F78_REASON) | length) == 2; "S9i branch (i): the reason string is not on its GATES line"),
     # S9i-a the 14 slot-24 rows: FINAL with the GATES reason
     W(all($r.printed_items.all_rows[] | select(.row as $n | S24 | index([$n]) != null); .status == $FR)
       and ([$r.printed_items.all_rows[] | select(.row as $n | S24 | index([$n]) != null)] | length) == 14;
       "S9i branch (i): the 14 slot-24 rows are not FINAL with the GATES reason"),
     # S9i-b the 2 Figs 7/8 labels: FINAL, no reason, the GATES-reason note, the pins disclosure, otherwise v2.4's; the other labels unchanged
     W(($rl78 | length) == 2 and ($vl78 | length) == 2 and ([$r.figure_labels[].output] == [$v.figure_labels[].output])
       and all(range(0; 2) as $i | {a: $rl78[$i], b: $vl78[$i]};
               .a.label == "FINAL" and (.a | has("reason") | not) and .a.note == $FR and .a.disclosures == ["v25.figs78.pins.1"]
               and (.a | del(.label, .note, .disclosures)) == (.b | del(.label, .note, .reason)))
       and ([$r.figure_labels[] | select(.output != "Figs 7/8") | NOTE16] == [$v.figure_labels[] | select(.output != "Figs 7/8") | NOTE16])
       and ([$r.row_rulings_v2_5.items[] | select(.id == "figs78") | .label_disclosures] == [[$rl78[].file]]);
       "S9i branch (i): the Figs 7/8 labels are not FINAL with the GATES reason and the pins disclosure"),
     # S9i-c element tables: v2.4's kept; then exactly the record's tables (F78_FIELDS copied) with the pinned source_record
     W($F78ROK == true and $r.element_tables[0:$net] == $v.element_tables
       and ($r.element_tables | length) == $net + ($F78R[0].element_tables | length)
       and all($F78R[0].element_tables[]; . as $t | all(F78_FIELDS[]; . as $k | $t | has($k)))
       and ([$r.element_tables[$net:][] | del(.source_record)]
            == [$F78R[0].element_tables[] | {output: "Figs 7/8", kind: "figs78_item"} + with_entries(select(.key as $k | F78_FIELDS | index([$k]) != null))])
       and all($r.element_tables[$net:][]; (.source_record | {path, commit, sha256}) == $src and (.source_record.origin_branches | type) == "array")
       and all($r.element_tables[$net:][]; .figure_status == "FINAL" and .elements_all_final == true and .inventory_complete == true);
       "S9i branch (i): the figs78 element tables are not the 854f901b6 record's"),
     # S9i-d pending_downstream: v2.4's with Figs 7/8 out of outputs / figure_files / waits and recomputed from the record
     W($v.pending_downstream.outputs == ["Figs 7/8"] and (($v.pending_downstream.figure_files | sort) == ([$vl78[].file] | sort))
       and ($r.pending_downstream | del(.recomputed["Figs 7/8"])) == ($v.pending_downstream | .outputs = [] | .figure_files = [] | .waits |= del(.["Figs 7/8"]))
       and ($v.pending_downstream.recomputed | has("Figs 7/8") | not) and ($r.pending_downstream.recomputed["Figs 7/8"] | keys) == ["figs78_record"]
       and ($r.pending_downstream.recomputed["Figs 7/8"].figs78_record | {path, commit, sha256}) == $src;
       "S9i branch (i): pending_downstream is not v2.4's with Figs 7/8 recomputed from the record"),
     # S9i-e the figs78 ruling block and item: record, recheck (driver-rechecked) and GATES cites are the pinned ones
     ($r.row_rulings_v2_5.figs78 as $f | [$r.row_rulings_v2_5.items[] | select(.id == "figs78")] as $it
      | W($F78RCOK == true and $f.branch == "i" and ($f.record | {path, commit, sha256}) == $src
          and $f.recheck.sha256 == F78.recheck_sha256 and $f.recheck.verdict == "CONFIRMED"
          and ($f.reason_cite | {commit, file, line}) == {commit: GATES_C, file: GATES_F, line: F78L.reason}
          and ($f.verified_cite | {commit, file, line}) == {commit: GATES_C, file: GATES_F, line: F78L.verified}
          and ($it | length) == 1 and $it[0].status == $FR
          and ([$it[0].cites[] | {commit, file, line}] == [{commit: GATES_C, file: GATES_F, line: F78L.verified}])
          and ([$it[0].rechecks[] | {line: .cite.line, commit: .cite.commit, sha256: .report.sha256}] == [{line: F78L.recheck, commit: GATES_C, sha256: F78.recheck_sha256}]);
          "S9i branch (i): the figs78 ruling block (record, recheck, cites) is not the pinned one")),
     # S9i-f the moved outputs: v2.4's preliminary entries with only status / name / reason / ruling / moved_from changed, plus (09, finding 2 (a))
     # the Figs 7/8 downstream_tag.recomputed_output, which EQUALS pending_downstream.recomputed["Figs 7/8"].figs78_record and holds the record pin
     ([$r.in_scope_outputs[] | select(.name == "Figs 7/8")] as $a | [$v.preliminary_outputs[] | select(.name == "Figs 7/8")] as $b
      | [$r.in_scope_outputs[] | select(.name == MOVED_IN[1])] as $c | [$v.preliminary_outputs[] | select(.name == MOVED_PRE[1])] as $d
      | W(($a | length) == 1 and ($b | length) == 1 and ($c | length) == 1 and ($d | length) == 1
          and $a[0].status == "FINAL" and ($a[0] | del(.status, .ruling, .moved_from, .downstream_tag.recomputed_output, .downstream_tag.text)) == ($b[0] | del(.status, .reason, .pending_downstream_label, .downstream_tag.text))
          and ($b[0].downstream_tag | has("recomputed_output") | not)
          and ($a[0].downstream_tag.recomputed_output | type) == "object"
          and $a[0].downstream_tag.recomputed_output == $r.pending_downstream.recomputed["Figs 7/8"].figs78_record
          and ($a[0].downstream_tag.recomputed_output | {path, commit, sha256}) == $src
          and $a[0].ruling == "figs78 (v2.5 branch (i))" and $a[0].moved_from == "preliminary_outputs (v2.4)" and ($b[0] | has("ruling") | not)
          and $c[0].status == "FINAL" and ($c[0] | del(.name, .status, .ruling)) == ($d[0] | del(.name, .status, .reason, .ruling))
          and $c[0].ruling == "figs78 (v2.5 branch (i))";
          "S9i branch (i): the moved Figs 7/8 output or slot-24 group differs from v2.4's beyond status / name / ruling"))
     ,
     # S9i-g (c4) the Figs 7/8 downstream_tag.text: EXACTLY build_v2_4.jq L103's quoted string @ 6a46541d5 + TAG_SUFFIX
     ((($L["\(V24C):BUILD24:103"] // "") | [capture("\\.downstream_tag\\.text = \"(?<t>[^\"]*)\""; "g") | .t]) as $t24
     | W(($t24 | length) == 1 and ([$r.in_scope_outputs[] | select(.name == "Figs 7/8") | .downstream_tag.text] == [$t24[0] + TAG_SUFFIX]);
         "S9i branch (i): the Figs 7/8 downstream_tag.text is not the v2.4 L103 string + the GATES L2430 suffix")),
     # S9i-h (c4, #1) the reading: EXACTLY 09's ruled string, byte for byte
     W($r.row_rulings_v2_5.reading == READING_I; "S9i branch (i): row_rulings_v2_5.reading is not 09's ruled string"),
     # S9i-i (c4, #2) the axis disclosure's pre-stated condition: v2.4's item figs78_axis_presentation applies when the figs78 item is in the version
     W(([$v.row_rulings_v2_4.items[] | select(.id == "figs78_axis_presentation") | select(.status == "disclosure (applies only when the figs78 item is in the version)") | .disclosures]
         == [AXIS24]) and ([$r.disclosures[] | select(.id == AXIS24[0])] == [$v.disclosures[] | select(.id == AXIS24[0])]) and (AXIS24 | length) == 1;
       "S9i branch (i): the v2.4 axis disclosure's pre-stated condition or object is not as in v2.4")
   else
     W(([$r.figure_labels[] | NOTE16] == [$v.figure_labels[] | NOTE16]) and ($r.element_tables == $v.element_tables)
       and all($r.printed_items.all_rows[] | select(.row as $n | S24 | index([$n]) != null); .status == $vs[.row | tostring]);
       "S9 branch (ii): Figs 7/8 and slot-24 rows are not as in v2.4")
   end),
  # S10 (c4, both branches) no in_scope output's downstream_tag.text starts with "PRELIMINARY"
  W(all($r.in_scope_outputs[] | select(.downstream_tag | type == "object"); (.downstream_tag.text // "") | startswith("PRELIMINARY") | not);
    "S10 an in_scope output's downstream_tag.text starts with PRELIMINARY"),
  # S10b (c5, both branches; closes the S10 hole) every in_scope object downstream_tag has a string text
  W(all($r.in_scope_outputs[] | select(.downstream_tag | type == "object"); .downstream_tag.text | type == "string");
    "S10b an in_scope output's object downstream_tag has no string text"),
  # S12 (c5, both branches) the 3 Figs 1/6 leaves EXACTLY: P (the v2.4 Figs 1/6 labels' .reason, both equal) + C5_JOINER + v2.4's text
  (([$v.figure_labels[] | select(.output == "Figs 1/6")]) as $vl16 | ([$r.figure_labels[] | select(.output == "Figs 1/6")]) as $rl16
  | ([$v.preliminary_outputs[] | select(.name == "Figs 1/6")]) as $vo16 | ([$r.preliminary_outputs[] | select(.name == "Figs 1/6")]) as $ro16
  | W(($vl16 | length) == 2 and ($rl16 | length) == 2 and $vl16[0].reason == $vl16[1].reason and ($vl16[0].reason | type) == "string"
      and ($vo16 | length) == 1 and ($ro16 | length) == 1
      and $ro16[0].downstream_tag.text == ($vl16[0].reason + C5_JOINER + $vo16[0].downstream_tag.text)
      and all(range(0; 2); $rl16[.].note == ($vl16[0].reason + C5_JOINER + $vl16[.].note));
      "S12 the Figs 1/6 downstream_tag.text or label notes are not P + the joiner + v2.4's text")),
  # S13 (c5, both branches; the mirror of S10) preliminary outputs and PRELIMINARY figure labels do not read FINAL
  W(all($r.preliminary_outputs[] | select(.downstream_tag | type == "object"); .downstream_tag.text | type == "string");
    "S13 a preliminary output's object downstream_tag has no string text"),
  W(all($r.preliminary_outputs[] | select(.downstream_tag | type == "object"); (.downstream_tag.text | strings | startswith("FINAL")) | not);
    "S13 a preliminary output's downstream_tag.text starts with FINAL"),
  W(all($r.figure_labels[] | select(.label == "PRELIMINARY"); ([.note, .reason] | map(strings | startswith("FINAL")) | any) | not);
    "S13 a PRELIMINARY figure label's note or reason starts with FINAL"),
  # S11 names
  W(if $has78 then [$r.in_scope_outputs[].name] == NAMES_IN_I and [$r.preliminary_outputs[].name] == NAMES_PRE_I
    else [$r.in_scope_outputs[].name] == NAMES_IN and [$r.preliminary_outputs[].name] == NAMES_PRE end; "S11 output names are not the pre-stated set")
  ] as $bad
| {n: {rows: ($r.printed_items.all_rows | length), FINAL: ([$r.printed_items.all_rows[] | select(.status | startswith("FINAL"))] | length),
       PRELIMINARY: ([$r.printed_items.all_rows[] | select(.status | startswith("PRELIMINARY"))] | length),
       not_route_c: ([$r.printed_items.all_rows[] | select(.status | startswith("not route C"))] | length),
       changed_rows: ([$r.printed_items.all_rows[] | select(.status != $vs[.row | tostring])] | length), v25_disclosures: ($d25 | length),
       a2: $A2OK, rv_ok: $RVOK, figs78_included: $has78, f78_line: $F78LINE, f78_record_ok: $F78ROK, f78_recheck_ok: $F78RCOK},
   bad: $bad}
