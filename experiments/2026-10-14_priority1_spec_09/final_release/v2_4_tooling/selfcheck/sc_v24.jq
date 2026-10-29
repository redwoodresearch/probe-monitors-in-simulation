# SC4: the v2.4 invariants, re-derived independently of build_v2_4.jq (its own constants, typed from 09's GATES rulings; it never reads
# rulings_v2_4.json). Keys, counts and booleans only. Output: {n: {...}, bad: [...]}.
# Inputs: --slurpfile REC draft  --slurpfile V23 v2.3  --slurpfile P17 final17 rows  --slurpfile P18 v18 rows  --slurpfile Q1 Q1 table
#         --slurpfile LN line texts {k: "commit:file:line", v}  --argjson IDX {fig source path: [value indices with ci95 and input "record"]}
#         --argjson R122OK true / false / null (the driver's re-check of the row-122 recheck report and verifier result shas)
#         --argjson POWEROK true / false (the driver's grep: every power number of disclosure v24.slots_21_22.1 is on PRESTATED_W2 L26-27)
# NOTE (09, disclosed): for rows 107-251 outside the ruled sets, S3's expectation is the Q1 table's class, the same table the builder reads;
# the independent part there is the fresh reviewer's seeded sample (14/14 + 15/15) and 09's own sample, not SC4.
def W($c; $w): if $c then empty else $w end;
def pfx: if startswith("FINAL") then "FINAL" elif startswith("PRELIMINARY") then "PRELIMINARY" elif startswith("not route C") then "not route C" else "OTHER" end;
def strip_bullet: sub("^\\s*-\\s+"; "");
def quotes: [match("\"([^\"]+)\""; "g") | .captures[0].string];
# ---- constants from GATES (09) ----
def PLANT: [range(47; 63), range(79; 86)];
def CORR12: [4, 5, 9, 10, 14, 15, 19, 20, 25, 27, 28, 29];
def RULED_FINAL_1_106: PLANT + [75, 76, 77, 78, 92, 95, 96, 97, 98, 106];
def S24: [69, 70, 71, 72, 73, 74] + [range(241; 249)];
def V18_COMMIT: "58bf4a3ac170aefc41a9692108316a49d2abe083";
def ALLOWED_CHANGED: ["builder", "counts_check", "date_pt", "disclosures", "element_tables", "extra_final_bars", "figure_labels", "flip_groups", "in_scope_outputs",
  "in_scope_records", "in_scope_set", "not_reproduced", "out_of_scope_records", "pending_downstream", "preliminary_outputs", "printed_items", "repo_path", "row_rulings_v2_4",
  "sensitivity_rows", "superseded_fields", "supersedes", "v2_conditions", "version"];
# b8 constants (fresh review R1/R2/N1/N3, Codex v24 r1 items 3/5/9, 09's rulings via 2e 01:5x)
def RECP: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.4_2026-10-21.json";
def PN_FINAL: [75, 76, 77, 78, 92, 95, 96, 97, 98];
def RULE_NEW($d): "in v2.4 the printed_numbers rows 75-78, 92 and 95-98 and the App A printed counts (rows 250/251, with disclosure) are FINAL per row_rulings_v2_4 (09); everything else here stays PRELIMINARY (Aswin's decision \($d))";
def V24_IDS: ["v24.PLANT.1", "v24.PLANT.2", "v24.PLANT.3", "v24.PLANT.4", "v24.PLANT.5", "v24.PLANT.6", "v24.PLANT.7", "v24.PLANT.8", "v24.PLANT.9", "v24.PLANT.10",
  "v24.step_D.1", "v24.step_D.2", "v24.rows_75_78.1", "v24.rows_95_98.1", "v24.row_106.1", "v24.row_92.1", "v24.app_A.1", "v24.app_A.2", "v24.app_A.3", "v24.app_A.4",
  "v24.slots_21_22.1", "v24.slots_21_22.2", "v24.slots_21_22.3", "v24.v23_interval_corrections.INT-CORR.1", "v24.v23_interval_corrections.INT-CORR.2",
  "v24.interval_value_half_only.INT-ROW.1", "v24.rows_119_122.R122.1", "v24.rows_119_122.R122.2", "v24.rows_119_122.R122.3", "v24.rows_119_122.rows119_121.1",
  "v24.figs_interval_rule.INT-FIG.1", "v24.figs_interval_rule.INT-FIG.2", "v24.figs78_axis_presentation.axis.1", "v24.repin_final17"];
# the canonical texts: id -> [GATES line, quoted string index (1-based)] (09, GATES 00:34; rows119_121 at L1837)
def CANON: {"v24.interval_value_half_only.INT-ROW.1": [1868, 1], "v24.v23_interval_corrections.INT-CORR.1": [1869, 1], "v24.v23_interval_corrections.INT-CORR.2": [1869, 2],
  "v24.figs_interval_rule.INT-FIG.1": [1870, 1], "v24.figs_interval_rule.INT-FIG.2": [1870, 2], "v24.rows_119_122.R122.1": [1871, 1], "v24.rows_119_122.R122.2": [1871, 2],
  "v24.rows_119_122.R122.3": [1871, 3], "v24.rows_119_122.rows119_121.1": [1837, 1]};
def SENS4: [25, 27, 28, 29];
def R122_DESC: {sha256: "870ed0de76ac62452711d35434a782dac3875abf69c838cd40b350b8e8963300", label: "from agent summary", verdict: "CONFIRMED",
  of: {commit: "fa4a475c9069b9c5edc23d3da4f791e1d8c837b0", branch: "aswin/routec-numbers-verify", parent: "e14aa51f98830e1e3024d7cb65543e2975167368",
       path: "experiments/2026-10-20_routec_numbers_verify/results/numbers_verify_2026-10-21_002558.json", sha256: "45bfd0a3b8a0b3e3c4229e0d4d941c54f76c84bc2f8439f0f7bff4906f7bbafe"}};
def POWER_SRC: {path: "experiments/2026-10-20_routec_reruns/rc1b/PRESTATED_W2_2026-10-20_145024.md", commit: "0b6bdd4e7748ae0f0696a725ff051230f042288a", lines: [26, 27]};
def FIGN: {"fig1_main_figs156_routecA_source.json": 60, "fig1_main_R_figs156_routecA_source.json": 60, "fig4_pooled_fig4_routecB_source.json": 240};
# b9-tooling (Codex v24 r2 items 3, 14; 09): every v2.4 disclosure's GATES line and text form (0 = the whole line minus its bullet, k = the k-th
# quoted string), and the output / record names, pre-stated (derived by script from b9 and reviewed in 09's key check).
def DISC_LINES: {"v24.PLANT.1":[889,0],"v24.PLANT.2":[890,0],"v24.PLANT.3":[891,0],"v24.PLANT.4":[892,0],"v24.PLANT.5":[893,0],"v24.PLANT.6":[913,0],"v24.PLANT.7":[917,0],"v24.PLANT.8":[599,0],"v24.PLANT.9":[596,0],"v24.PLANT.10":[686,0],"v24.step_D.1":[765,0],"v24.step_D.2":[810,0],"v24.rows_75_78.1":[1267,0],"v24.rows_95_98.1":[1601,0],"v24.row_106.1":[1340,0],"v24.row_92.1":[1357,0],"v24.app_A.1":[1359,0],"v24.app_A.2":[1360,0],"v24.app_A.3":[1361,0],"v24.app_A.4":[1680,0],"v24.slots_21_22.1":[1657,0],"v24.slots_21_22.2":[1658,0],"v24.slots_21_22.3":[1659,0],"v24.v23_interval_corrections.INT-CORR.1":[1869,1],"v24.v23_interval_corrections.INT-CORR.2":[1869,2],"v24.interval_value_half_only.INT-ROW.1":[1868,1],"v24.rows_119_122.R122.1":[1871,1],"v24.rows_119_122.R122.2":[1871,2],"v24.rows_119_122.R122.3":[1871,3],"v24.rows_119_122.rows119_121.1":[1837,1],"v24.figs_interval_rule.INT-FIG.1":[1870,1],"v24.figs_interval_rule.INT-FIG.2":[1870,2],"v24.figs78_axis_presentation.axis.1":[1856,1],"v24.repin_final17":[1675,0]};
def NAMES: {"in_scope_outputs":["Tables 3/4","rc1a","rc2","Table 6 plant_rows","honest_all_routec printed items","routec_simpool_numbers printed items (FINAL rows)","rc1b printed items, slots 21/22","attack_sol_share printed item (row 106)","cunits_by_tool printed items (Appendix A)","rc1b_detl48 printed item, slot 23 (row 122)"],"preliminary_outputs":["rc1b","Fig 5","Figs 7/8","routec_simpool_numbers printed items","rc1b_detl48 printed items, slot 23","rc1b printed items (App H / Fig 7-8 captions)","Fig 4","Figs 1/6"],"in_scope_records":["A","stage_d","kappa","kappa_extra","group_excess","group_catch","pw_shares","anchor","R37","B","PLANT"],"out_of_scope_records":["detector_vs_drop","honest_all_routec","routec_simpool_numbers"]};
def NARROW: {"v24.v23_interval_corrections.INT-CORR.2": [25, 27, 28, 29], "v24.rows_119_122.R122.1": [122], "v24.rows_119_122.R122.2": [122],
  "v24.rows_119_122.R122.3": [122], "v24.rows_119_122.rows119_121.1": [119, 120, 121]};
# b9-2 (Codex v24 r3 items 7, 8; 09): the item -> disclosure-id attachments and the figure_labels {file, output, label}, pre-stated
# (derived by script from b9, frozen in the tooling; the record is checked against these, not against its own lists).
def ITEM_DISC: {"PLANT":["v24.PLANT.1","v24.PLANT.2","v24.PLANT.3","v24.PLANT.4","v24.PLANT.5","v24.PLANT.6","v24.PLANT.7","v24.PLANT.8","v24.PLANT.9","v24.PLANT.10"],"step_D":["v24.step_D.1","v24.step_D.2"],"rows_75_78":["v24.rows_75_78.1","v24.repin_final17"],"rows_95_98":["v24.rows_95_98.1"],"row_106":["v24.row_106.1"],"row_92":["v24.row_92.1"],"failed_footnote":[],"app_A":["v24.app_A.1","v24.app_A.2","v24.app_A.3","v24.app_A.4"],"slots_21_22":["v24.slots_21_22.1","v24.slots_21_22.2","v24.slots_21_22.3"],"slot_24_figs78":[],"repin_final17":[],"record_level_text":[],"v23_interval_corrections":["v24.v23_interval_corrections.INT-CORR.1","v24.v23_interval_corrections.INT-CORR.2"],"interval_value_half_only":["v24.interval_value_half_only.INT-ROW.1"],"rows_119_122":["v24.rows_119_122.R122.1","v24.rows_119_122.R122.2","v24.rows_119_122.R122.3","v24.rows_119_122.rows119_121.1"],"figs_interval_rule":["v24.figs_interval_rule.INT-FIG.1","v24.figs_interval_rule.INT-FIG.2"],"figs78_axis_presentation":["v24.figs78_axis_presentation.axis.1"]};
def FIGLABELS: [{"file":"experiments/2026-10-15_fpr_sweep_507_prep/results/fig4_pooled_fig4_routecB_source.json","output":"Fig 4","label":"PRELIMINARY"},{"file":"experiments/2026-10-15_fpr_sweep_507_prep/results/fig1_main_R_figs156_routecA_source.json","output":"Figs 1/6","label":"PRELIMINARY"},{"file":"experiments/2026-10-15_fpr_sweep_507_prep/results/fig1_main_figs156_routecA_source.json","output":"Figs 1/6","label":"PRELIMINARY"},{"file":"experiments/2026-10-15_fpr_sweep_507_prep/results/fig5_excess_C_figs156_routecA_source.json","output":"Fig 5","label":"PRELIMINARY"},{"file":"experiments/2026-10-15_fpr_sweep_507_prep/results/fig_pertask_onpolicy_figs78_routecA_source.json","output":"Figs 7/8","label":"PRELIMINARY"},{"file":"experiments/2026-10-15_fpr_sweep_507_prep/results/fig_pertask_summary_figs78_routecA_source.json","output":"Figs 7/8","label":"PRELIMINARY"}];
$REC[0] as $r | $V23[0] as $v
| ($LN | map({(.k): .v}) | add // {}) as $L
| ($P17 | map({(.row | tostring): .}) | add) as $p17 | ($P18 | map({(.row | tostring): .}) | add) as $p18
| ([($r.element_tables // [])[] | select(.kind == "figs78_item")] | length > 0) as $has78
| ($r.row_rulings_v2_4.r_node_switch.state == "OFF") as $off
| ($r.row_rulings_v2_4.row122_recheck | type == "object" and has("file") and has("sha256")) as $r122
# the 53 interval rows, derived from v18 key paths (not from the Q1 table)
| ([range(107; 252) as $n | $p18[$n | tostring] | select((.record | test("slots_routeC_writing_rc1a\\.json@")) and (.key_path[0] | type) == "string")
    | select(((.key_path[1] | test("^1[01]\\.(delta_full|run2_minus_run1_at_tauB)\\.")) and ((.predicate.component // "") | test("^interval_(low|high)$")))
             or (.key_path[1] | test("^10\\.halfwidth_C\\."))) | .row]) as $IV
| def expected($n):
    if ([CORR12[] | select(. == $n)] | length) > 0 then (if $off then "PRELIMINARY" else "FINAL" end)
    elif ([RULED_FINAL_1_106[] | select(. == $n)] | length) > 0 then "FINAL"
    elif $n == 93 or $n == 94 then "PRELIMINARY"
    elif ([S24[] | select(. == $n)] | length) > 0 then (if $has78 then "FINAL" else "PRELIMINARY" end)
    elif $n <= 106 then ([$v.printed_items.all_rows[] | select(.row == $n)][0].status | pfx)
    elif $n >= 231 and $n <= 240 then "FINAL"
    elif $n == 250 or $n == 251 then "FINAL"
    elif $n >= 119 and $n <= 121 then "PRELIMINARY"
    elif $n == 122 then (if $r122 then "FINAL" else "PRELIMINARY" end)
    elif ([$IV[] | select(. == $n)] | length) > 0 then (if $off then "PRELIMINARY" else "FINAL" end)
    else ([$Q1[0].rows[] | select(.row == $n)][0].class_R_component) end;
  ([$r.printed_items.all_rows[] | {row, s: (.status | pfx)}]) as $act
| ([$v.printed_items.all_rows[] | {(.row | tostring): .status}] | add) as $v23s
| ([$r.printed_items.all_rows[] | select(.row <= 106) | {row, old: $v23s[.row | tostring], new: .status}]) as $r106
| ([$r.disclosures[] | select(.id | startswith("v24."))]) as $d24
| [
  # S1 provenance
  W($r.printed_items.provenance.commit == V18_COMMIT and ($r.printed_items.provenance.sha256 | startswith("13d29fb90cac3e3e")) and ($r.printed_items.provenance.path | endswith("provenance_v84_final18_0141.jsonl")); "S1 provenance pin is not v18 @ A"),
  W(([range(1; 252) | select(. < 119 or . > 122)] | all(. as $n | $p17[$n | tostring] == $p18[$n | tostring])); "S1 v18 != final17 outside rows 119-122"),
  W([range(119; 123) | $p18[tostring].record] | all(test("slots_routeC_writing_rc1b_detl48\\.json@6d77ea369")); "S1 v18 rows 119-122 do not read rc1b_detl48 @ 6d77ea369"),
  # S2 rows
  W(([$r.printed_items.all_rows[].row] == [range(1; 252)]) and all($r.printed_items.all_rows[]; .record == $p18[.row | tostring].record); "S2 all_rows != v18 rows / records"),
  # S3 status class per row, re-derived
  ($act[] | . as $a | expected($a.row) as $e | W($a.s == $e; "S3 row \($a.row): \($a.s) != expected \($e)")),
  # S4 rows 1-106 versus v2.3: exactly the 12 corrected rows go FINAL -> PRELIMINARY; the ruled flips go to FINAL; nothing else changes
  W(([$r106[] | select((.old | pfx) == "FINAL" and (.new | pfx) == "PRELIMINARY") | .row] | sort) == (if $off then CORR12 else [] end); "S4 FINAL->PRELIMINARY on rows 1-106 is not exactly the 12"),
  W(([$r106[] | select((.old | pfx) != "FINAL" and (.new | pfx) == "FINAL") | .row] | sort) == ((RULED_FINAL_1_106 + (if $has78 then [69, 70, 71, 72, 73, 74] else [] end)) | sort); "S4 the flips to FINAL on rows 1-106"),
  W(([$r106[] | select(.old != .new) | .row] - (CORR12 + RULED_FINAL_1_106 + [93, 94] + (if $has78 then [69, 70, 71, 72, 73, 74] else [] end))) == []; "S4 another row's status changed on rows 1-106"),
  # S5 the 53 interval rows
  W(($IV | length) == 53; "S5 the interval-row set is not 53"),
  W(([$IV[] | . as $n | $act[] | select(.row == $n) | .s] | unique) == [if $off then "PRELIMINARY" else "FINAL" end]; "S5 interval rows' class"),
  # S6 disclosures: every v24 disclosure is the cited GATES line (minus its bullet) or one of that line's quoted strings, from git
  ($d24[] | . as $d | ($L["\($d.cite.commit):\($d.cite.file):\($d.cite.line)"] // null) as $t
   | W($t != null and ($d.text == ($t | strip_bullet) or (($t | quotes) | index([$d.text]) != null)); "S6 disclosure \($d.id) is not its GATES line text")),
  # S7 the canonical texts are present (09, GATES 00:34), each at its own GATES line and quoted string (b8: text/cite pairs cannot be swapped)
  ((CANON | keys[]) as $id
   | W(([$d24[] | select(.id == $id)] | length) == 1; "S7 missing disclosure \($id)"),
     ([$d24[] | select(.id == $id)][0] as $d | ($L["\($d.cite.commit // ""):\($d.cite.file // ""):\($d.cite.line // "")"] // null) as $t
      | W($d != null and $d.cite.line == CANON[$id][0] and $t != null and (($t | quotes)[CANON[$id][1] - 1] == $d.text); "S7 disclosure \($id) is not its canonical line / quoted string"))),
  # S7b the disclosure inventory: v2.3's 25 kept byte for byte, exactly the 34 v2.4 ids, one GATES commit for all v2.4 cites (Codex v24 r1 item 3)
  W($r.disclosures[0:($v.disclosures | length)] == $v.disclosures; "S7b v2.3 disclosures not kept"),
  W(([$d24[].id] | sort) == (V24_IDS | sort) and ($r.disclosures | length) == (($v.disclosures | length) + (V24_IDS | length)); "S7b the v2.4 disclosure inventory"),
  W(([$d24[].cite.commit] | unique | length) == 1; "S7b v2.4 disclosures cite more than one GATES commit"),
  # S7e b9-tooling (Codex v24 r2 item 3): EVERY v2.4 disclosure is bound to its pre-stated GATES line and text form, so no text/cite pair can be swapped
  ($d24[] | . as $d | (DISC_LINES[$d.id] // null) as $e | ($L["\($d.cite.commit):\($d.cite.file):\($d.cite.line)"] // null) as $t
   | W($e != null and $t != null and $d.cite.line == $e[0]
       and (if $e[1] == 0 then $d.text == ($t | strip_bullet) else (($t | quotes)[$e[1] - 1] == $d.text) end); "S7e disclosure \($d.id) is not bound to its pre-stated line / form")),
  W((DISC_LINES | keys) == (V24_IDS | sort); "S7e the pre-stated disclosure table is not the v2.4 inventory"),
  # S7c per-row attachment (09: INT-CORR.2 only on 25/27/28/29; R122.* only on 122; rows119_121.1 on 119-121; INT-CORR.1 on the 12; INT-ROW.1 on the 53)
  (([$r.row_rulings_v2_4.items[] | .disclosure_rows // {} | to_entries[]] | from_entries) as $DR
  | W($DR["v24.v23_interval_corrections.INT-CORR.2"] == SENS4 and ($DR["v24.v23_interval_corrections.INT-CORR.1"] | sort) == CORR12
      and $DR["v24.rows_119_122.R122.1"] == [122] and $DR["v24.rows_119_122.R122.2"] == [122] and $DR["v24.rows_119_122.R122.3"] == [122]
      and $DR["v24.rows_119_122.rows119_121.1"] == [119, 120, 121] and $DR["v24.repin_final17"] == [75, 76, 77, 78] and ($DR["v24.interval_value_half_only.INT-ROW.1"] | sort) == ($IV | sort); "S7c per-row disclosure attachment")),
  W(all($r.row_rulings_v2_4.items[]; . as $it | ($it.disclosure_rows // {}) as $m
        | if ($it.rows | length) > 0 then (($m | keys) == ($it.disclosures | sort)) and all($m[]; (. - $it.rows) == []) else $m == {} end); "S7c disclosure_rows keys or rows outside the item"),
  # S7c b9-tooling (Codex v24 r2 item 13): complete, not just a subset: each attachment is exactly the item's rows, unless 09 narrowed it (NARROW)
  W(all($r.row_rulings_v2_4.items[] | select((.rows | length) > 0); . as $it
        | all(($it.disclosure_rows // {}) | to_entries[]; (.value | sort) == ((NARROW[.key] // $it.rows) | sort))); "S7c a disclosure_rows list is not the item's full rows (or 09's narrowing)"),
  W(all($r.printed_items.all_rows[] | select(.row as $n | SENS4 | index([$n]) != null); .status | endswith(" " + ([$d24[] | select(.id == "v24.v23_interval_corrections.INT-CORR.2")][0].text)))
    and all($r.printed_items.all_rows[] | select(.row as $n | (CORR12 - SENS4) | index([$n]) != null); .status | contains("Sensitivity rows") | not); "S7c the INT-CORR suffix is verbatim on 25/27/28/29 only"),
  # S7d R4a: the power numbers' source cite, verbatim, and the pinned file (the driver re-checks the numbers on its lines: POWEROK)
  (([$d24[] | select(.id == "v24.slots_21_22.1")][0]) as $pw
  | W($pw.source_cite == "(power as pre-stated in PRESTATED_W2 L26-27)" and $pw.source.path == POWER_SRC.path and $pw.source.commit == POWER_SRC.commit
      and $pw.source.lines == POWER_SRC.lines and $POWEROK == true; "S7d the power source cite or its numbers")),
  # S8 Figs 1/6 and Fig 4 under the interval rule
  (if $off then
     W(all($r.figure_labels[] | select(.file | test("fig1_main(_R)?_figs156_routecA_source\\.json$|fig4_pooled_fig4_routecB_source\\.json$")); .label == "PRELIMINARY"); "S8 a Figs 1/6 / Fig 4 label is not PRELIMINARY"),
     W(([$r.in_scope_outputs[] | select(.name == "Figs 1/6" or .name == "Fig 4")] | length) == 0 and ([$r.preliminary_outputs[] | select((.name == "Figs 1/6" or .name == "Fig 4") and .interval_rule != null)] | length) == 2; "S8 Figs 1/6 / Fig 4 outputs"),
     ($r.element_tables[] | .file as $f0 | select($IDX | has($f0)) | . as $t
      | W(([$t.elements[] | select(.status == "PRELIMINARY") | .value_index] | sort) == ($IDX[$t.file] | sort) and $t.figure_status == "PRELIMINARY"; "S8 element table \($t.file | split("/") | last): PRELIMINARY elements != the record ci95 entries")),
     # b8 (Codex v24 r1 item 9; 09 [INT-FIG]): the inventory is present, and the label counts are 60 / 60 / 240
     W(([$r.element_tables[] | .file | select(. as $f | $IDX | has($f))] | length) == 2 and ([$r.element_tables[] | .file | select(. as $f | $IDX | has($f))] | unique | length) == 2
       and ($IDX | length) == 2; "S8 the two Figs 1/6 element tables"),
     W(([$r.figure_labels[] | (.file | split("/") | last) as $b | select(FIGN | has($b)) | {($b): .preliminary_elements_n}] | add) == FIGN; "S8 figure_labels preliminary_elements_n (60 / 60 / 240)")
   else empty end),
  # S9 switch
  W($off and ([$r.printed_items.all_rows[] | select(.status | contains("R_node extension"))] | length) == 0; "S9 the R_node switch is not OFF or a status uses it"),
  # S10 unchanged keys
  ((($v | keys) - ALLOWED_CHANGED)[] as $k | W($r[$k] == $v[$k]; "S10 top-level key \($k) changed")),
  W((($v | keys) - ALLOWED_CHANGED | length) >= 1; "S10 no unchanged top-level keys to compare"),
  W((($r | keys) - ($v | keys)) == ["row_rulings_v2_4"]; "S10 new top-level keys other than row_rulings_v2_4"),
  W(($r.counts_check | del(.printed_numbers, .other_printed_counts, .rule)) == ($v.counts_check | del(.printed_numbers, .other_printed_counts, .rule)); "S10 counts_check outside printed_numbers / other_printed_counts / rule changed"),
  # b8 (fresh review R2, 09): the App A printed counts FINAL with disclosure; the rule text; the FINAL printed_numbers rows
  W(($r.counts_check.other_printed_counts | length) == ($v.counts_check.other_printed_counts | length)
    and all(range(0; ($v.counts_check.other_printed_counts | length)) as $i | ($v.counts_check.other_printed_counts[$i]) as $o | ($r.counts_check.other_printed_counts[$i]) as $n
        | if ($o.name | test("App A")) then ($n | del(.status, .status_cite, .label)) == ($o | del(.status, .label)) and $n.label == ($o.label + " (superseded: FINAL with disclosure per Aswin 21:23 / GATES L1352)") and ($n.status | startswith("FINAL with disclosure (rows 250/251;"))
            and $n.status_cite.line == 1352 and ($n.status | contains("v24.app_A.1, v24.app_A.2, v24.app_A.3, v24.app_A.4"))
          else $n == $o end; .); "S10 counts_check.other_printed_counts (App A FINAL with disclosure; others unchanged)"),
  W(([$v.counts_check.rule | match("nothing here is FINAL \\(Aswin's decision ([0-9]+)\\)$") | .captures[0].string][0]) as $d
    | $d != null and $r.counts_check.rule == ($v.counts_check.rule | sub("nothing here is FINAL \\(Aswin's decision [0-9]+\\)$"; "") + RULE_NEW($d)); "S10 counts_check.rule text"),
  W(([$r.counts_check.printed_numbers[] | select(.status | startswith("FINAL")) | .row] | sort) == PN_FINAL; "S10 counts_check.printed_numbers FINAL rows"),
  W($r.repo_path == RECP and $r.version == "2.4" and $v.repo_path != RECP; "S1b repo_path / version"),
  # Codex v24 r1 item 4: a printed-items group's placement agrees with its rows (in_scope: all FINAL; preliminary: none FINAL)
  (([$r.printed_items.all_rows[] | {(.row | tostring): (.status | pfx)}] | add) as $RS
  | W(all($r.in_scope_outputs[] | select(.kind == "printed_items"); all(.rows[]; $RS[tostring] == "FINAL"))
      and all($r.preliminary_outputs[] | select(.kind == "printed_items"); all(.rows[]; $RS[tostring] != "FINAL"))
      and ([$r.in_scope_outputs[], $r.preliminary_outputs[] | select(.kind == "printed_items")] | length) >= 1; "S15 a printed-items group's placement disagrees with its rows")),
  # Codex v24 r1 item 2: no ruling item other than rows_119_122 claims a status for rows 119-122
  W(all($r.row_rulings_v2_4.items[] | select(.id != "rows_119_122"); .status | test("119-12[12] stay PRELIMINARY|row 122 (stays )?PRELIMINARY") | not); "S16 a ruling status other than rows_119_122 claims rows 119-122 PRELIMINARY"),
  W(($r.sensitivity_rows | del(.per_output)) == ($v.sensitivity_rows | del(.per_output)) and ($r.sensitivity_rows.per_output[0:($v.sensitivity_rows.per_output | length)] == $v.sensitivity_rows.per_output); "S10 sensitivity_rows changed beyond the appended Table 6 entry"),
  W(($r.flip_groups | del(.group_3)) == $v.flip_groups; "S10 flip groups 1-2 changed"),
  W($r.superseded_fields.v2_3_superseded_fields == $v.superseded_fields; "S10 v2.3 superseded_fields not kept in full"),
  W($r.exclusions == $v.exclusions; "S10 exclusions changed"),
  # S12 row 122
  W((([$act[] | select(.row == 122)][0].s == "FINAL") == $r122); "S12 row 122 FINAL without the recheck, or the reverse"),
  W((if $r122 then $R122OK == true else true end); "S12 the row-122 recheck report or its verifier result does not match its sha, or is not CONFIRMED"),
  W((if $r122 then ($r.row_rulings_v2_4.row122_recheck | {sha256, label, verdict, of}) == R122_DESC else true end); "S12 the row-122 descriptor is not the relayed recheck (constants)"),
  # S13 PLANT
  W(([$r.in_scope_records[] | select(.name == "PLANT")] | length) == 1 and ([$r.out_of_scope_records[] | select(.name == "PLANT")] | length) == 0
    and ([$r.extra_final_bars[] | select(.record == "PLANT")] | length) == 1; "S13 PLANT records"),
  # S14 record-level text (Q9)
  W(all($r.out_of_scope_records[] | select(.name == "honest_all_routec" or .name == "routec_simpool_numbers"); .status == "PRELIMINARY (record not reproduced as a whole); rows per row_rulings_v2_4"); "S14 Q9 text"),
  # S17 b9-tooling (Codex v24 r2 item 14): the output, printed-items group and record names are the pre-stated set (names reach the message)
  W([$r.in_scope_outputs[].name] == NAMES.in_scope_outputs and [$r.preliminary_outputs[].name] == NAMES.preliminary_outputs
    and [$r.in_scope_records[].name] == NAMES.in_scope_records and [$r.out_of_scope_records[].name] == NAMES.out_of_scope_records; "S17 output / group / record names are not the pre-stated set"),
  # S7f b9-2 (Codex v24 r3 item 7): each ruling item's disclosures (and so its disclosure_rows keys, S7c) equal the pre-stated table
  W(([$r.row_rulings_v2_4.items[] | {(.id): .disclosures}] | add) == ITEM_DISC; "S7f the item -> disclosure attachments are not the pre-stated table"),
  # S18 b9-2 (Codex v24 r3 item 8): the figure_labels {file, output, label} are exactly the pre-stated six (labels reach the message)
  W([$r.figure_labels[] | {file, output, label}] == FIGLABELS; "S18 figure_labels are not the pre-stated set")
  ] as $bad
| {n: {rows: ($act | length), FINAL: ([$act[] | select(.s == "FINAL")] | length), PRELIMINARY: ([$act[] | select(.s == "PRELIMINARY")] | length),
       not_route_c: ([$act[] | select(.s == "not route C")] | length), interval_rows: ($IV | length), corrected_rows: ([$r106[] | select((.old | pfx) == "FINAL" and (.new | pfx) == "PRELIMINARY")] | length),
       v24_disclosures: ($d24 | length), pending_fill_strings: ([$r | .. | strings | select(startswith("PENDING"))] | length), figs78_included: $has78, row122_recheck: $r122, r_node_switch_off: $off},
   bad: $bad}
