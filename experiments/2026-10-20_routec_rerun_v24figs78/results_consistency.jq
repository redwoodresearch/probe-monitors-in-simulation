# results_consistency.jq (AMEND11; Codex fix r1 #3, fresh review R9-N3): is a figs78 record internally consistent, by key? Run as
#   /usr/bin/jq -c -f results_consistency.jq <record>   ->   {five booleans}; the results RUN_ME requires all true.
# - FINAL elements and FINAL slot-24 rows carry the main gate's FINAL evidence (gate_figs78_out.jq L164-165; row 72's half-width follow-up);
# - each table gate = inventory_complete and elements_all_final; elements_all_final = drawn elements non-empty and all FINAL;
# - the slot-24 gate EQUALS (14 rows, all FINAL, and slot24_quantities == the 10 printed keys), in both directions (the mapper's condition).
# PRINTED10 below is gate_figs78_out.jq's definition, copied byte for byte (lines 17-23; diff-checked).
def NOTDRAWN: "disclosure only (not drawn)";
def PRINTED10: ["24.n_cells.lead.own_all_rs_uncut_L48 C", "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",
  "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]", "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]",
  "24.r2.lead.own_all_rs_uncut_L48 C.main", "24.r2.lead.own_all_rs_uncut_L48 C.yardstick",
  "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",
  "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]",
  "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]",
  "24.r2.yardstick_halfwidth.lead.own_all_rs_uncut_L48 C"] | sort;
def ok_el: .a1_rows == 1 and .a1_verdict == true and .a1_match == true and .a1_kind_ok == true and .value_identity == true;
def ok_s24: .provenance_row_unique == true and .record_is_rc1b_600f63614 == true and .record_sha_is_a1_pin == true and .in_a1_printed10 == true
            and .a1_value_rows == 1 and .a1_value_match == true and .a1_display_rows == 1 and .a1_display_match == true
            and (if .row == 72 then (.halfwidth_followup | .identified_readings_is_half_range == true and .ident_match == true and .all_pairs_match == true) else true end);
. as $r | ([$r.gates[] | {(.name): .ok}] | add) as $G
| {final_elements_have_evidence: all($r.element_tables[].elements[] | select(.status == "FINAL"); ok_el),
   final_slot24_rows_have_evidence: all($r.slot24_rows[] | select(.status == "FINAL"); ok_s24),
   table_gates_match_tables: all($r.element_tables[]; $G["element_table_complete_and_final:" + (.file | split("/") | last)] == (.inventory_complete and .elements_all_final)),
   all_final_flags_match_elements: all($r.element_tables[]; .elements_all_final == ([.elements[] | select(.status != NOTDRAWN)] | length > 0 and all(.[]; .status == "FINAL"))),
   slot24_gate_matches_rows: ($G["slot24_14_rows_10_quantities_all_final"] == (($r.slot24_rows | length) == 14 and all($r.slot24_rows[]; .status == "FINAL")
                                                                              and (($r.slot24_quantities // []) | sort) == PRINTED10))}
