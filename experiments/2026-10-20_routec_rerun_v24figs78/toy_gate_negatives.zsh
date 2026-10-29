#!/bin/zsh -f
# Toy test of gate_figs78_out.jq (synthetic, keys-only; jq only, no Python). Predicted: the unmodified toy -> true (exit 0);
# every mutation below -> false (exit 1). Prints ids, predictions and outcomes only.
set -e; setopt pipefail; export LC_ALL=C
H=${0:A:h}; GATE=$H/gate_figs78_out.jq
TOY=$(jq -n -c -f $H/toy_gate_record.jq)
integer ok=0 n=0
check() {  # id, jq mutation, predicted exit
  local id=$1 mut=$2 want=$3 got
  (( ++n ))
  if print -r -- "$TOY" | jq -c "$mut" | jq -e -f $GATE > /dev/null; then got=0; else got=$?; fi
  print -r -- "TOY $id predicted_exit $want got_exit $got $([[ $got == $want ]] && print AS_PREDICTED || print NOT_AS_PREDICTED)"
  [[ $got == $want ]] && (( ++ok ))
  return 0
}
check P0  '.' 0
check T1  '.element_tables[1].elements = []' 1
check T2  '.element_tables[0].n_drawn = null' 1
check T3  '.extra = {n: 0}' 1
check T4  '.element_tables[0].elements[10].run1 = 0.5' 1
check T5  '.slot24_rows[3].halfwidth_followup.ident_match = false' 1
check T6  '.slot24_rows[0].a1_display_match = false' 1
check T7  '.element_tables[0].elements[5].cell = "12345"' 1
check T8  '.gates[0].ok = false' 1
check T9  '.element_tables[0].metadata_not_drawn = []' 1
check T10 '.element_tables[0].elements[100].value_identity = false' 1
check T11 '.element_tables[1].classes |= with_entries(.value.n += 1)' 1
check T12 '.exit = "STOP_TO_09"' 1
check T13 '.element_tables[0].elements[20] |= del(.a1_family, .a1_key)' 1
check T14 '.element_tables[0].classes |= map_values(.PRELIMINARY += 1)' 1
check T15 '.a1_duplicate_keys_by_family.extra = 0' 1
check T16 '.element_tables[0].elements[30].a1_key = "lead|x|y|z"' 1
check T17 '.element_tables[0].elements[40].origin = "S_bogus"' 1
check T18 '.slot24_rows[] |= del(.slot, .component)' 1
check T19 '.element_tables[0].elements[2].class as $a | .element_tables[0].elements[3].class as $b | .element_tables[0].elements[2].class = $b | .element_tables[0].elements[3].class = $a' 1
check T20 '.slot24_rows[8].component = "interval_low"' 1
check T21 '.slot24_rows[9] |= (.slot = .a1_key | .component = "point")' 1
if (( ok == n )); then print -r -- "TOY_GATE_ALL_AS_PREDICTED $ok/$n"; else print -r -- "TOY_GATE_NOT_AS_PREDICTED $ok/$n"; exit 1; fi
