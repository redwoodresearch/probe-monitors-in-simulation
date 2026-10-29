#!/bin/zsh -f
# Toy test r3 of gate_figs78_prelim.jq (r3) and guard_one_object.jq against the predictions frozen in
# PREDICTIONS_gate_figs78_prelim_r2_2026-10-21_0055.md (C1-C11, G1-G3) and _r3_2026-10-21_0121.md (C12-C15). jq only; ids and outcomes only.
set -e; setopt pipefail; export LC_ALL=C
H=${0:A:h}; C=$H/gate_figs78_prelim.jq; GU=$H/guard_one_object.jq
TOY=$(jq -n -c -f $H/toy_gate_record.jq)
# C2 base: 3 scatter elements PRELIMINARY (ids of elements 20, 21, 22 of the Fig 7 table), histogram updated, BOTH figures PRELIMINARY, Fig 7's gate false
C2='.element_tables[0].elements |= (to_entries | map(if .key >= 20 and .key <= 22 then .value |= (.status = "PRELIMINARY" | .a1_match = false | .reason = "r") else . end) | map(.value))
    | .element_tables[0].classes["Fig 7 scatter coordinate (Stage D max_minus_tau) -> a1 stage_d_pt"] |= (.FINAL -= 3 | .PRELIMINARY += 3)
    | .element_tables[].figure_status = "PRELIMINARY" | .element_tables[0].elements_all_final = false
    | .gates |= map(if .name == "element_table_complete_and_final:fig_pertask_onpolicy_figs78_routecA_source.json" then .ok = false else . end)
    | .exit = "STOP_TO_09"'
IDS=$(print -r -- "$TOY" | jq -c '[.element_tables[0].elements[20, 21, 22].id] | sort')
integer ok=0 n=0
check() {  # id, jq transform, jq predicate on the companion's output
  local id=$1 tr=$2 pred=$3 got
  (( ++n ))
  got=$(print -r -- "$TOY" | jq -c "$tr" | jq -c -f $C | jq -e --argjson ids "$IDS" "$pred" > /dev/null && print true || print false)
  print -r -- "TOYC $id as_predicted $got"
  [[ $got == true ]] && (( ++ok ))
  return 0
}
guard() {  # id, shell producer of the input, predicted boolean
  local id=$1 prod=$2 want=$3 got
  (( ++n ))
  got=$(eval "$prod" | jq -e -s -f $GU > /dev/null 2>&1 && print true || print false)
  print -r -- "TOYG $id guard $got predicted $want"
  [[ $got == $want ]] && (( ++ok ))
  return 0
}
check C1 '.' '.valid == false and .kind == "not STOP_TO_09"'
check C2 "$C2" '.valid == true and .kind == "STOP_TO_09" and .n_prelim == 3 and (.prelim["fig_pertask_onpolicy_figs78_routecA_source.json"] | sort) == $ids
               and .gates_false == ["element_table_complete_and_final:fig_pertask_onpolicy_figs78_routecA_source.json"]'
check C3 "$C2 | .element_tables[0].elements[5] |= del(.id)" '.valid == false'
check C4 "$C2 | .element_tables[0].elements[30].cell = \"12345\"" '.valid == false'
check C5 '{stamp: .stamp, app: .app, inputs: .inputs, a1_values_file: .a1_values_file, gates: (.gates | map(.ok = false)), exit: "PLUMBING"}' '.valid == false and .kind == "PLUMBING"'
check C6 "$C2 | .slot24_rows |= map(if .row == 244 then .status = \"PRELIMINARY\" else . end) | .gates |= map(if .name == \"slot24_14_rows_10_quantities_all_final\" then .ok = false else . end)" \
           '.valid == true and .slot24_not_final == [244]'
check C7 "$C2 | .element_tables = [] | .slot24_rows = []" '.valid == false'
check C8 "$C2 | .slot24_rows |= map(if .row == 72 then .halfwidth_followup.x = 0.5 else . end)" '.valid == false'
check C9 "$C2 | .element_tables[0].elements[21].reason = 0.5" '.valid == false'
check C10 "$C2 | .element_tables[1].figure_status = \"FINAL\"" '.valid == false'
check C11 "$C2 | .gates |= map(select(.name != \"negative_controls_fail_as_predicted\"))" '.valid == false'
check C12 "$C2 | .element_tables[0].expected_class_counts.payload = 42" '.valid == false'
check C13 "$C2 | .gates |= map(if .name == \"inputs_sha\" then .ok = false else . end)" '.valid == false'
check C14 "$C2 | .gates |= map(if .name == \"real_printed10_equal\" then .ok = false else . end)" '.valid == false'
check C15 "$C2 | .a1_values_file.n = 3" '.valid == false'
C2REC=$(print -r -- "$TOY" | jq -c "$C2")
guard G1 'print -r -- "$C2REC"; print -r -- "$C2REC"' false
guard G2 'print -r -- "$C2REC"' true
guard G3 'print -r -- "[$C2REC]"' false
if (( ok == n )); then print -r -- "TOY_PRELIM_ALL_AS_PREDICTED $ok/$n"; else print -r -- "TOY_PRELIM_NOT_AS_PREDICTED $ok/$n"; exit 1; fi
