# build_v2_4.jq: route C FINAL-release record v2.4 = v2.3 + the v2.4 rulings, by key. Never prints; the wrapper writes the output.
# Inputs (all keys / shas / labels; the GATES line texts come from git via resolve_rulings.zsh):
#   --slurpfile V23 the v2.3 record          --slurpfile P17 final17 rows (jsonl)       --slurpfile Q1 the Q1 per-row table
#   --slurpfile RUL rulings_v2_4.json        --slurpfile RES resolved rulings (anchors / evidence / rechecks; keys and shas only)
#   --slurpfile LN  {k: "commit:file:line", v: line text} for every resolved anchor, read from git by the wrapper (line sha re-checked)
#   --slurpfile COV {output name: marker covered list} for every output with an S2 marker
#   --slurpfile F78 the figs78 item's keys-only record, or [] when the item is not included (then Figs 7/8 + slot 24 stay PRELIMINARY)
#   --argjson META {date_pt, builder: {...}, final17: {path, commit, sha256, kind, origin_branches}, v23: {path, commit, sha256, origin_branches},
#                   q1_table: {path, sha256, note}, figs78: null | {record: {path, commit, sha256}, verified_by_09: <anchor or null>}, reading: "R_component" | "R_node"}
def truthy: . != null and . != false and . != "" and . != 0 and . != [] and . != {};
def ce: "\(.path)@\(if has("decompressed_sha256") then .decompressed_sha256 else .sha256 end)";
def strip_bullet: sub("^\\s*-\\s+"; "");
$V23[0] as $v | $RUL[0] as $R | $RES[0] as $X | ($Q1[0].rows | map({(.row | tostring): .}) | add) as $QR
| ($LN | map({(.k): .v}) | add // {}) as $LT
| ($P17 | map({(.row | tostring): .}) | add) as $P
| ($F78 | length > 0) as $HAS78
| ($META.r_node_switch.on == true) as $SW
| ($META.row122_recheck) as $R122
| def item($id): [$R.items[] | select(.id == $id)][0];
  def anchor($must): ($X.anchors[$must] // error("NO ANCHOR " + $must)).anchor;
  def ev($e): ($X.evidence[$e.path + "@" + $e.commit] // error("NO EVIDENCE " + $e.path)) + {role: $e.role};
  def disc_id($it; $i): "v24." + $it.id + "." + ($i | tostring);
  # the cited GATES line's text and its quoted strings, read from git by the wrapper (--slurpfile LN); never stored in the resolved file
  def atext($must): anchor($must) as $a | ($LT["\($a.commit):\($a.file):\($a.line)"] // error("NO LINE TEXT " + $must));
  def aquotes($must): [atext($must) | match("\"([^\"]+)\""; "g") | .captures[0].string];
  def item_disclosures($it): [($it.disclosure_lines // []) | to_entries[] | {id: disc_id($it; .key + 1), text: (atext(.value) | strip_bullet), cite: anchor(.value)}]
                             + [($it.quote_lines // [])[] | . as $ql
                                | aquotes($ql.must) | to_entries[] | {id: ("v24." + $it.id + "." + $ql.role + "." + ((.key + 1) | tostring)), text: .value, cite: anchor($ql.must)}];
  def item_rechecks($it): [($it.rechecks // [])[] | {label: .label, what: (.what // null), cite: anchor(.gates_must)}
                           + (if .file then {report: ($X.rechecks[.file] | {file, sha256, in_git: false})} else {} end)];
  # b8 (fresh review N3 / 09): each disclosure's rows, explicitly: the item's rows unless the rulings narrow it (INT-CORR.2 -> 25/27/28/29,
  # R122.* -> 122, rows119_121.1 -> 119-121); a narrowed list must be a subset of the item's rows
  def drows($it; $rows): if ($rows | length) == 0 then {} else
      ([item_disclosures($it)[].id | . as $id | ($id | ltrimstr("v24." + $it.id + ".")) as $short
        | (($it.disclosure_rows // {})[$short] // $rows) as $dr
        | if ($dr - $rows) != [] then error("disclosure_rows not a subset: " + $id) else {($id): $dr} end] | add // {}) end;
  def ruling($it): {id: $it.id, rows: $it.rows, status: $it.status, outputs: ($it.outputs // []),
                    evidence: [($it.evidence // [])[] | ev(.)], fills: ($it.fills // []), rechecks: item_rechecks($it), cites: [$it.cites[] | anchor(.)],
                    disclosures: [item_disclosures($it)[].id], disclosure_rows: drows($it; $it.rows)};
  def rows_of($id): item($id).rows;
  def status_of($id): item($id).status;
  def prov_item($n): $P[$n | tostring] | {row, item, anchor, key_path, record, record_sha256};
  def group($name; $status; $rows; $why): {name: $name, kind: "printed_items", status: $status, rows: $rows, items_n: ($rows | length),
                                            items: [$rows[] | prov_item(.)], ruling: $why};
  def isin($n; $rows): ([$rows[] | select(. == $n)] | length) > 0;

# ---------------- row statuses (09: R_component, GATES 00:21 / 00:27; the R_node switch is pre-stated and OFF) ----------------
  ([$Q1[0].rows[] | select(.class_R_component == "PRELIMINARY" and ((.component_kind // "") | test("interval")) and .class_R_node == "FINAL") | .row]) as $IV53
| (if $HAS78 then status_of("slot_24_figs78") | sub("^FINAL only if.*"; "FINAL (slot 24: a1 addendum 24 anchor package; figs78 item verified; independent recompute from the producer's stated definition, POST HOC)")
   else "PRELIMINARY (slot 24 / Figs 7/8: the figs78 item is not in this version; v2.5)" end) as $S24
| ($SW | not) as $FR
| def figrule($name): {rule: item("figs_interval_rule").status, switch: "R_node OFF (applies only on Aswin's option (2), verbatim with its time)",
      cites: [item("figs_interval_rule").cites[] | anchor(.)],
      disclosures: (["v24.figs_interval_rule.INT-FIG.1"] + (if $name == "Figs 1/6" then ["v24.figs_interval_rule.INT-FIG.2"] else [] end))};
  ("FINAL under the R_node extension (Aswin option (2): " + ($META.r_node_switch.aswin_quote // "NONE") + " @ " + ($META.r_node_switch.time // "NONE") + ")") as $SWL
| def rowstatus($n):
    if isin($n; item("PLANT").rows) then status_of("PLANT")
    elif isin($n; item("rows_75_78").rows) then status_of("rows_75_78")
    elif isin($n; item("rows_95_98").rows) then status_of("rows_95_98")
    elif $n == 106 then status_of("row_106")
    elif $n == 92 then status_of("row_92")
    elif $n == 93 or $n == 94 then status_of("failed_footnote")
    elif $n == 250 or $n == 251 then status_of("app_A")
    elif isin($n; item("slots_21_22").rows) then status_of("slots_21_22")
    elif isin($n; item("slot_24_figs78").rows) then (if $HAS78 or $n > 106 then $S24 else ([$v.printed_items.all_rows[] | select(.row == $n)][0].status) end)
    elif isin($n; item("v23_interval_corrections").rows) then
      (if $SW then $SWL
       elif isin($n; item("v23_interval_corrections").sensitivity_rows) then status_of("v23_interval_corrections") + " " + aquotes(item("v23_interval_corrections").quote_lines[0].must)[1]   # [INT-CORR]'s suffix, verbatim from GATES
       else status_of("v23_interval_corrections") end)
    elif $n == 122 then (if ($R122 | type) == "object" then "FINAL (row 122: re-pointed to the L48 readout; numbers verifier slot_component_eq OK @ 45bfd0a3; step D reproduces the L48 record)"
                         else "PRELIMINARY (row 122: FINAL pending the fresh recheck of the verifier results commit; else v2.5)" end)
    elif $n >= 119 and $n <= 121 then "PRELIMINARY (rows 119-121: the printed value is the earlier L63 number; the L48 record value differs; numbers verifier DATA_FALSE @ 45bfd0a3; text update pending (Aswin))"
    elif $n <= 106 then ([$v.printed_items.all_rows[] | select(.row == $n)][0].status)
    else ($QR[$n | tostring]) as $q
      | if $q.class_R_component == "FINAL" then "FINAL (per quantity, Q1 table: G7b ledger MATCH + a1 verdict row)"
        elif isin($n; $IV53) then (if $SW then $SWL else status_of("interval_value_half_only") end)
        else "PRELIMINARY (per quantity, Q1 table: a half does not list this key)" end
    end;
[range(1; 252) | . as $n | {record: $P[$n | tostring].record, row: $n, status: rowstatus($n)}] as $all_rows
| ($all_rows | group_by(.status) | map({(.[0].status): length}) | add) as $counts

# ---------------- records ----------------
| ([$v.out_of_scope_records[] | select(.name == "PLANT")][0]) as $plant0
| ($plant0 | del(.status, .why, .cite) + {kind: "pooled_record", item16: null,
     scope_note: "FINAL per quantity for Table 6 and its 23 printed rows only (09, GATES 15:22; 22:46 Q8); PLANT_NOT_COMPARED keys are not reproduced (not_reproduced)",
     moved_from: "out_of_scope_records (v2.3)", cites: [item("PLANT").cites[] | anchor(.)]}) as $plant
| ($v.in_scope_records + [$plant]) as $in_recs
| ([$in_recs[] | ce, (select(.sidecar | truthy) | .sidecar | ce)] | unique) as $in_set
| ($in_set | map({(.): true}) | add) as $INS
| ([($v.gate_inputs_no_values // [])[] | ce] | map({(.): true}) | add // {}) as $GS
| def outside($name): [$COV[0][$name][] | select(($INS[.] or $GS[.]) | not)] | sort;
  def recount($o): $o + {covered_in_scope_n: ([$COV[0][$o.name][] | select($INS[.])] | length)};
  def q9: {status: item("record_level_text").status, status_cite: anchor(item("record_level_text").cites[0])};

# ---------------- outputs ----------------
  ([$v.preliminary_outputs[] | select(.name == "Table 6 plant_rows")][0]) as $t6
| ([$v.preliminary_outputs[] | select(.name == "Figs 7/8")][0]) as $f78o
| ([$v.preliminary_outputs[] | select(.name == "rc1b")][0]) as $rc1b
| (recount($t6) | .status = "FINAL" | del(.out_of_scope_inputs) | . + {ruling: "PLANT", moved_from: "preliminary_outputs (v2.3)"}) as $t6new
| (recount($rc1b) | .out_of_scope_inputs = [.out_of_scope_inputs[] | select(.input as $i | outside("rc1b") | index([$i]) != null)]) as $rc1bnew
| (if $HAS78 then
     ($f78o | .status = "FINAL" | del(.reason, .pending_downstream_label)
      | .downstream_tag.text = "FINAL per element after the figs78 item (every drawn element mapped to a1's verdict rows; GATES 14:05, 22:22, 22:46 Q4-Q6)"
      | .downstream_tag.recomputed_output = {a1: [item("slot_24_figs78").evidence[] | ev(.)], figs78_record: $META.figs78.record,
                                             verified_by_09: $META.figs78.verified_by_09,
                                             element_tables: [$F78[0].element_tables[] | .file]}
      | . + {ruling: "slot_24_figs78", moved_from: "preliminary_outputs (v2.3)"})
   else null end) as $f78new
| ([ group("honest_all_routec printed items"; "FINAL"; rows_of("rows_75_78"); "rows_75_78"),
    group("routec_simpool_numbers printed items (FINAL rows)"; "FINAL"; (rows_of("row_92") + rows_of("rows_95_98")); "row_92 + rows_95_98"),
    group("rc1b printed items, slots 21/22"; "FINAL"; rows_of("slots_21_22"); "slots_21_22"),
    group("attack_sol_share printed item (row 106)"; "FINAL"; rows_of("row_106"); "row_106"),
    group("cunits_by_tool printed items (Appendix A)"; "FINAL"; rows_of("app_A"); "app_A") ]
  + (if ($R122 | type) == "object" then [group("rc1b_detl48 printed item, slot 23 (row 122)"; "FINAL"; [122]; "rows_119_122")] else [] end)
  + (if $HAS78 then [group("rc1b printed items, slot 24 (App H / Fig 7-8 captions)"; "FINAL"; rows_of("slot_24_figs78"); "slot_24_figs78")] else [] end)) as $final_groups
| ([ group("routec_simpool_numbers printed items"; "PRELIMINARY"; rows_of("failed_footnote"); "failed_footnote") + {reason: status_of("failed_footnote")},
    group("rc1b_detl48 printed items, slot 23"; "PRELIMINARY"; (if ($R122 | type) == "object" then [119, 120, 121] else [119, 120, 121, 122] end); "rows_119_122") + {reason: rowstatus(119)} ]
  + (if $HAS78 then [] else [group("rc1b printed items (App H / Fig 7-8 captions)"; "PRELIMINARY"; rows_of("slot_24_figs78"); "slot_24_figs78") + {reason: $S24}] end)) as $prelim_groups

# ---------------- the record ----------------
| $v
| .version = "2.4"
| .repo_path = $META.repo_path
| .date_pt = $META.date_pt
| .supersedes = ($META.v23 + {version: "2.3"})
| .builder = $META.builder
| .in_scope_records = $in_recs
| .in_scope_set = $in_set
| .out_of_scope_records = [$v.out_of_scope_records[] | select(.name != "PLANT")
                           | if .name == "honest_all_routec" or .name == "routec_simpool_numbers" then . + q9 else . end]
| .extra_final_bars += [{record: "PLANT",
                          g7b_value: ev(item("PLANT").evidence[0]), cleanroom: ev(item("PLANT").evidence[3]), derivation: null,
                          rechecks: item_rechecks(item("PLANT")), cites: [item("PLANT").cites[] | anchor(.)]}]
| .not_reproduced += [
    {record: "PLANT", item: "planted sim2 path: NOT REPRODUCED by V3 (PLANT_NOT_COMPARED); no printed row reads it", cite: anchor("planted-sim2 NOT REPRODUCED (build omission; not printed);")},
    {record: "PLANT", item: "planted RMS b-hat-squared check: outside V3's scope; its own status (rms_full @ af1a12e32)", cite: anchor("planted RMS outside V3's scope")},
    {record: "PLANT", item: "unconditional pooled categories: in PLANT_NOT_COMPARED under the G1 POST HOC loosening", cite: anchor("G1 POST HOC loosening;")}]
| .in_scope_outputs = (($v.in_scope_outputs | if $FR then map(select(.name != "Figs 1/6" and .name != "Fig 4")) else . end) + [$t6new] + (if $HAS78 then [$f78new] else [] end) + $final_groups)
| .preliminary_outputs = [ $v.preliminary_outputs[]
      | select(.name != "Table 6 plant_rows" and .name != "honest_all_routec printed items" and .name != "routec_simpool_numbers printed items"
               and .name != "rc1b printed items (App H / Fig 7-8 captions)" and (($HAS78 | not) or .name != "Figs 7/8"))
      | if .name == "rc1b" then $rc1bnew else . end ] + $prelim_groups
      + (if $FR then [$v.in_scope_outputs[] | select(.name == "Figs 1/6" or .name == "Fig 4") | .name as $nm
                      | .status = "PRELIMINARY" | . + {interval_rule: figrule($nm), moved_from: "in_scope_outputs (v2.3)"}] else [] end)
| .figure_labels = [ $v.figure_labels[] | if ($FR and ((.file | test("fig1_main(_R)?_figs156_routecA_source.json$")) or (.file | test("fig4_pooled_fig4_routecB_source.json$")))) then
      (.output as $o | .file as $lf | .label = "PRELIMINARY" | .reason = item("figs_interval_rule").status | .interval_rule = figrule(if ($o | test("1/6")) then "Figs 1/6" else "Fig 4" end)
       # b8 (09, [INT-FIG]): the PRELIMINARY interval elements: 60 per Figs 1/6 file, 240 for Fig 4 (counted from the committed sources)
       | .preliminary_elements_n = (if ($lf | test("fig4_pooled")) then $META.fig4_interval_n else ($META.figs16_interval_idx[$lf] | length) end))
    elif ($HAS78 and (.file | test("figs78"))) then
      (.label = "FINAL" | del(.reason) | .note = "FINAL: every drawn element mapped to a1's verdict rows (figs78 item); captions: rc1b slot 24 rows, FINAL") else . end ]
| .element_tables |= map(.file as $f0 | if $FR and ($META.figs16_interval_idx | has($f0)) then (.file as $f
      | .elements |= map(.value_index as $vi | if ($META.figs16_interval_idx[$f] | index([$vi])) != null then .status = "PRELIMINARY" | .interval_note = "plotted interval ends: value half only (interval rule, GATES 00:34)" else . end)
      | .figure_status = "PRELIMINARY" | .interval_rule = figrule("Figs 1/6")) else . end)
| .v2_conditions.fig4_elements |= (if $FR then .label = "PRELIMINARY" | .note = "v2.4: Fig 4 PRELIMINARY by the interval rule (GATES 00:34); its 264 point entries stay FINAL per quantity" else . end)
| .element_tables += (if $HAS78 then [ $F78[0].element_tables[] | {output: "Figs 7/8", kind: "figs78_item", file, commit, sha256, figure_status,
      inventory_complete, elements_all_final, classes, metadata_not_drawn, elements, source_record: $META.figs78.record} ] else [] end)
| .pending_downstream = (if $HAS78 then (.pending_downstream | .outputs = [] | .figure_files = [] | .recomputed["Figs 7/8"] = {figs78_record: $META.figs78.record}
                           | .waits = (.waits | del(.["Figs 7/8"]))) else .pending_downstream end)
| .sensitivity_rows.per_output += [{output: "Table 6 plant_rows", commit: $t6new.commit, prints_sensitivity_labels: false,
     per_label: ($v.sensitivity_rows.labels | map({(.): 0}) | add), entries: [],
     files: [$t6new.files[] | {path, kind} + $META.sens_scan_table6[.path]],
     note: "added in v2.4: Table 6 became a FINAL output; its record has no slots dict and no sensitivity-label mention outside metadata blocks (sens_scan.jq)"}]
| .counts_check.printed_numbers = [ .counts_check.printed_numbers[] | . as $pn
      | if ([92, 75, 76, 77, 78, 95, 96, 97, 98] | index([$pn.row])) != null then .status = rowstatus($pn.row)
        elif $pn.row == 93 or $pn.row == 94 then .status = (rowstatus($pn.row) + "; internal consistency of the record")
        else . end ]
# b8 (09, fresh review R2): the App A printed counts are rows 250/251, FINAL with disclosure; the rule text follows the v2.4 FINAL rows
| (anchor(item("app_A").counts_status_cite)) as $appa
| if ([.counts_check.other_printed_counts[] | select(.name | test("App A"))] | length) != 1 then error("counts_check: not exactly one App A entry") else . end
| .counts_check.other_printed_counts |= map(if (.name | test("App A")) then
      .status = ("FINAL with disclosure (rows 250/251; ruling app_A per Aswin 21:23 + ~21:30 and 0d's 18:21 item 8; GATES L\($appa.line) @ \($appa.commit[0:9]); disclosures "
                 + ([item_disclosures(item("app_A"))[].id] | join(", ")) + ")")
      | .status_cite = $appa
      # b9 (09, NB3): the v2.3 label is the decision-10 bar verdict; one clause, verbatim, says it is superseded
      | (if $appa.line != 1352 then error("App A cite is not GATES L1352") else . end)
      | .label = (.label + " (superseded: FINAL with disclosure per Aswin 21:23 / GATES L1352)") else . end)
| .counts_check.rule |= (if test("nothing here is FINAL \\(Aswin's decision [0-9]+\\)$") | not then error("counts_check.rule: not the v2.3 text")
      else sub("nothing here is FINAL \\((?<d>Aswin's decision [0-9]+)\\)$"; "in v2.4 the printed_numbers rows 75-78, 92 and 95-98 and the App A printed counts (rows 250/251, with disclosure) are FINAL per row_rulings_v2_4 (09); everything else here stays PRELIMINARY (\(.d))") end)
| .printed_items.provenance = $META.provenance
| .printed_items.approval = anchor(item("repin_final17").cites[0])
| .printed_items.outside_lists.rows = [$all_rows[] | select(.status | startswith("not route C")) | .row]
| .printed_items.outside_lists.rows_n = (.printed_items.outside_lists.rows | length)
| .printed_items.all_rows = $all_rows
| .printed_items.all_rows_status_counts = $counts
| .printed_items.groups_n = ([(.in_scope_outputs + .preliminary_outputs)[] | select(.kind == "printed_items")] | length)
| .printed_items.items_n = ([(.in_scope_outputs + .preliminary_outputs)[] | select(.kind == "printed_items") | .items_n] | add)
| .disclosures += ([$R.items[] | item_disclosures(.)[]] + [{id: "v24.repin_final17", text: (atext(item("repin_final17").cites[0]) | strip_bullet),
                     cite: anchor(item("repin_final17").cites[0])}])
# b8 (09, R4a): the power numbers in a slots 21/22 disclosure are pre-registered design parameters; their source cite, verbatim, plus the pinned file
| (item("slots_21_22")) as $s2122
| .disclosures |= map(if .id == disc_id($s2122; $s2122.power_source.disclosure) then
      . + {source_cite: $s2122.power_source.text, source: (ev([$s2122.evidence[] | select(.role == $s2122.power_source.evidence_role)][0]) + {lines: $s2122.power_source.lines})}
    else . end)
| .row_rulings_v2_4 = {reading: "R_component (09, GATES 00:21 PDT 10-21: FINAL needs a clean-room verdict per quantity)", q1_table: $META.q1_table,
     r_node_switch: {state: (if $SW then "ON" else "OFF" end), applies_only_on: "Aswin's option (2), quoted verbatim with its time (GATES 00:21)",
                     aswin_quote: ($META.r_node_switch.aswin_quote // null), time: ($META.r_node_switch.time // null),
                     rows_v2_3_corrections: item("v23_interval_corrections").rows, rows_new_interval: $IV53, rows_total: ((item("v23_interval_corrections").rows + $IV53) | length)},
     row122_recheck: $R122,
     items: [$R.items[] | ruling(.) + (if .id == "interval_value_half_only" then {rows: $IV53, disclosure_rows: drows(.; $IV53)} else {} end) +(if .provenance_condition then {provenance_condition: (.provenance_condition + {pinned: $META.provenance.path, pinned_commit: $META.provenance.commit, holds_by: "printed_items.provenance is v18 (= final17 on every row except 119-122, by cmp)"})} else {} end)],
     slot24_quantities: (if $HAS78 then $F78[0].slot24_quantities else null end)}
# b9 (09, NB6): attach disclosures a ruling names (v24.repin_final17 -> rows 75-78: the provenance-pin condition of their predicate)
| .row_rulings_v2_4.items |= map(. as $it | ([$R.items[] | select(.id == $it.id)][0].attach_disclosures // []) as $ad
      | if ($ad | length) == 0 then .
        elif ([$ad[] | select(. as $a | [$v.disclosures[].id, ([$R.items[] | item_disclosures(.)[].id][]), "v24.repin_final17"] | index([$a]) == null)] | length) > 0 then error("attach_disclosures: unknown id")
        else .disclosures += $ad | .disclosure_rows += ([$ad[] | {(.): $it.rows}] | add) end)
| .flip_groups.group_3 = {what: "v2.4 rulings: PLANT (Table 6 + 23 rows), rows 75-78, 92, 95-98, 106, 231-240, 250-251; rc1a rows 107-249 per quantity (Q1); slot 24 + Figs 7/8 only with the figs78 item",
                          evidence: [$R.items[] | .id],
                          reversals: ["Figs 1/6 and Fig 4: FINAL in v2.3 (group 2 / in_scope_outputs) -> PRELIMINARY in v2.4 by the interval rule ([INT-FIG]; R_node switch OFF)",
                                      "rows 4, 5, 9, 10, 14, 15, 19, 20, 25, 27, 28, 29: FINAL in v2.3 -> PRELIMINARY in v2.4 ([INT-CORR])"]}
| .superseded_fields = {scope: "v2.4 = v2.3 + the v2.4 rulings, by key (09, GATES @ 1363d6bdc); everything not listed is byte-identical to v2.3",
     v2_3: $META.v23,
     fields: ["version", "repo_path", "date_pt", "supersedes", "builder", "in_scope_records[+PLANT]","in_scope_set", "out_of_scope_records[-PLANT; honest_all_routec, routec_simpool_numbers status]",
              "extra_final_bars[+PLANT]", "not_reproduced[+3 PLANT items]", "in_scope_outputs[+Table 6, +printed-items groups" + (if $HAS78 then ", +Figs 7/8, +slot 24" else "" end) + "]",
              "preliminary_outputs[-Table 6, -honest_all group, simpool group -> rows 93-94, +rc1b slot 23 group, rc1b out_of_scope_inputs]",
              "preliminary_outputs[+Figs 1/6, +Fig 4 (interval rule)]", "in_scope_outputs[-Figs 1/6, -Fig 4]",
              "figure_labels[Figs 1/6, Fig 4: PRELIMINARY, interval_rule, preliminary_elements_n]" + (if $HAS78 then "[figs78 FINAL]" else "" end),
              "element_tables[Figs 1/6: interval elements PRELIMINARY]" + (if $HAS78 then "[+2 figs78]" else "" end), "v2_conditions.fig4_elements.label",
              "pending_downstream" + (if $HAS78 then "[Figs 7/8 recomputed]" else "(unchanged)" end), "counts_check.printed_numbers[].status (rows 75-78, 92-98)",
              "counts_check.other_printed_counts[App A].status (+status_cite)", "counts_check.rule", "sensitivity_rows.per_output[+Table 6]",
              "printed_items.{provenance, approval, outside_lists, all_rows, all_rows_status_counts, groups_n, items_n}", "disclosures[+v24.*]", "row_rulings_v2_4", "flip_groups.group_3"],
     v2_3_superseded_fields: $v.superseded_fields}
