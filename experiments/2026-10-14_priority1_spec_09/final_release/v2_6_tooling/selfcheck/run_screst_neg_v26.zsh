#!/bin/zsh -f
# Ported self-check negatives (sc_core.zsh with sc_rest_v26.jq and the message) on the real v2.6 record: the v2.5 figs78-table plants
# (S1, N-b1a, N-b1b, N-b1c) and N-corr1 (09 Q4: r_node_switch.superseded_by_ruling removed). Predictions commit/PREDICTIONS_v26_msg_gate_
# screst_20261021_1503.md (876afe6b…). Each plant names its exact path; leafdiff.jq must show exactly that change (C21; shapes as in
# run_sc6_neg.zsh); a plant that changes nothing STOPs. A plant changes the record's sha, so the message's record-sha check also fails; only
# the named lines ('|'-separated; "re:" = an extended regex) are asserted present.   zsh -f run_screst_neg_v26.zsh <record.json> <message.txt>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; REC=${1:A}; MSG=${2:A}
[[ -f $REC && -f $MSG ]] || { echo "STOP usage"; exit 2; }
NEG=$HERE/negtest; mkdir -p $NEG $HERE/_runs; STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
typeset -a IDS LOCS OPS KINDS WANTS
add() { IDS+=($1); LOCS+=($2); OPS+=($3); KINDS+=($4); WANTS+=($5); }
T0='(.element_tables | map(.kind == "figs78_item") | index(true)) as $i'
DS='MISMATCH FINAL output Figs 7/8 reads a downstream record without a recorded recompute and full FINAL element tables'
add S1     "$T0 | [\"element_tables\", \$i, \"elements\", (.element_tables[\$i].elements | map(.status == \"FINAL\") | index(true)), \"status\"]" 'setpath($p; "PRELIMINARY")' changed 'MISMATCH figure status|MISMATCH figs78 element table class counts'
add N-b1a  "$T0 | [\"element_tables\", \$i, \"elements\", (.element_tables[\$i].elements | map(.status == \"FINAL\") | index(true)), \"status\"]" 'setpath($p; "disclosure only (not drawn)")' changed "MISMATCH figs78 element table (b1) pin|$DS"
add N-b1b  "$T0 | [\"element_tables\", \$i, \"elements\", (.element_tables[\$i].elements | map(.status == \"disclosure only (not drawn)\") | index(true)), \"class\"]" 'setpath($p; getpath($p) + " x")' changed "MISMATCH figs78 element table (b1) pin|$DS"
add N-b1c  '["in_scope_outputs", (.in_scope_outputs | map(.name) | index("Figs 7/8")), "downstream_tag", "recomputed_output"]' 'delpaths([$p])' removed "$DS"
add N-corr1 '["row_rulings_v2_4", "r_node_switch", "superseded_by_ruling"]' 'delpaths([$p])' removed 're:^MISMATCH printed row [0-9]+ on sensitivity slot .*: status vs sensitivity_halves$'
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/screst26_${IDS[$i]}_$STAMP.json
  P=$(jq -c "${LOCS[$i]}" $REC)
  jq --argjson p "$P" "${OPS[$i]}" $REC > $F
  LD=$(jq -n -c -f $HERE/leafdiff.jq --slurpfile a $REC --slurpfile b $F)
  shape=$(jq -n -c --argjson d "$LD" --argjson p "$P" --arg k "${KINDS[$i]}" 'def under: (.[0:($p | length)] == $p);
    if ($d.removed + $d.added + $d.changed | length) == 0 then "noop"
    elif $k == "changed" then (if $d.changed == [$p] and $d.added == [] and $d.removed == [] then "ok" else "bad" end)
    elif $k == "removed" then (if $d.changed == [] and $d.added == [] and ($d.removed | length) >= 1 and all($d.removed[]; under) then "ok" else "bad" end)
    else "bad" end' | tr -d '"')
  [[ $shape != noop ]] || { print -r -- "STOP: negative ${IDS[$i]} plant changed nothing"; exit 3; }
  O=$HERE/_runs/screstneg26_${STAMP}_${IDS[$i]}.out
  set +e; SC_REST=$HERE/sc_rest_v26.jq zsh -f $HERE/sc_core.zsh $F $MSG > $O 2>&1; rc=$?; set -e
  hit=true; for w in ${(s:|:)WANTS[$i]}; do if [[ $w == re:* ]]; then grep -qE -- "${w#re:}" $O || hit=false; else grep -qxF -- "$w" $O || hit=false; fi; done
  print -r -- "NEG ${IDS[$i]} kind ${KINDS[$i]} leaf_shape $shape exit $rc predicted_lines_present $hit mismatch_lines $(grep -c '^MISMATCH ' $O || true)"
  [[ $shape == ok && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "SCREST_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "SCREST_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
