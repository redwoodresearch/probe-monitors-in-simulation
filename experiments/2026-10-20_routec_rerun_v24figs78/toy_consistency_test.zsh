#!/bin/zsh -f
# Toy test of results_consistency.jq against the predictions frozen in PRESTATED_v24figs78_AMEND11_2026-10-21_0300.md (K1-K5).
# jq only, on the synthetic toy record (toy_gate_record.jq). Prints ids, predictions and outcomes only.
set -e; setopt pipefail; export LC_ALL=C
H=${0:A:h}; C=$H/results_consistency.jq; JQ=/usr/bin/jq
TOY=$($JQ -n -c -f $H/toy_gate_record.jq)
integer ok=0 n=0
check() {  # id, jq transform, jq predicate on the consistency output
  local id=$1 tr=$2 pred=$3 got
  (( ++n ))
  got=$(print -r -- "$TOY" | $JQ -c "$tr" | $JQ -c -f $C | $JQ -e "$pred" > /dev/null && print true || print false)
  print -r -- "TOYK $id as_predicted $got"
  [[ $got == true ]] && (( ++ok ))
  return 0
}
check K1 '.' '[.[]] | all'
check K2 '.gates |= map(if .name == "slot24_14_rows_10_quantities_all_final" then .ok = false else . end)' '.slot24_gate_matches_rows == false'
check K3 '.slot24_quantities |= .[1:]' '.slot24_gate_matches_rows == false'
check K4 '.element_tables[0].elements |= (to_entries | map(if .key == 20 then .value.a1_match = false else . end) | map(.value))' '.final_elements_have_evidence == false'
check K5 '.gates |= map(if .name == "element_table_complete_and_final:fig_pertask_onpolicy_figs78_routecA_source.json" then .ok = false else . end)' '.table_gates_match_tables == false'
if (( ok == n )); then print -r -- "TOY_CONSISTENCY_ALL_AS_PREDICTED $ok/$n"; else print -r -- "TOY_CONSISTENCY_NOT_AS_PREDICTED $ok/$n"; exit 1; fi
