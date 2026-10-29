#!/bin/zsh -f
# Ported self-check negatives for the figs78_item tables and 09's finding-2 ruling ((a) + (b1) pinned): plants of the real v2.5 record, run
# through sc_core.zsh with sc_rest_v25.jq and the message. Predictions frozen first: commit/PREDICTIONS_v25_branch_i_selfcheck_figs78_tables_
# 20261021_0412.md (S1, 1ef8a829…) and commit/PREDICTIONS_v25_c3_finding2_negatives_20261021_0422.md (N-b1a, N-b1b, N-b1c, 1ee84cc2…).
# A plant changes the record's sha, so the message's record-sha check also fails; only the named lines are asserted present. S1, N-b1a and
# N-b1b change exactly 1 leaf; N-b1c removes an object (asserted != 1). Prints fixture ids, leaf-diff counts, exit codes and booleans only.
#   zsh -f run_screst_neg_v25.zsh <record.json> <message.txt>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; REC=${1:A}; MSG=${2:A}
[[ -f $REC && -f $MSG ]] || { echo "STOP usage"; exit 2; }
NEG=$HERE/negtest; mkdir -p $NEG $HERE/_runs
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
leafdiff() { jq -n --slurpfile a $1 --slurpfile b $2 '($a[0] | [paths(type | . != "object" and . != "array")]) as $pa | ($b[0] | [paths(type | . != "object" and . != "array")]) as $pb
  | if ($pa | sort) != ($pb | sort) then -1 else [$pa[] as $p | select(($a[0] | getpath($p)) != ($b[0] | getpath($p)))] | length end'; }
T0='(.element_tables | map(.kind == "figs78_item") | index(true)) as $i'
DRAWN='([.element_tables[$i].elements | to_entries[] | select(.value.status == "FINAL") | .key][0]) as $j'
NOTD='([.element_tables[$i].elements | to_entries[] | select(.value.status == "disclosure only (not drawn)") | .key][0]) as $j'
DS='MISMATCH FINAL output Figs 7/8 reads a downstream record without a recorded recompute and full FINAL element tables'
typeset -a IDS EDITS W1 W2
IDS+=(S1);    EDITS+=("$T0 | $DRAWN | .element_tables[\$i].elements[\$j].status = \"PRELIMINARY\"");                        W1+=('MISMATCH figure status'); W2+=('MISMATCH figs78 element table class counts')
IDS+=(N-b1a); EDITS+=("$T0 | $DRAWN | .element_tables[\$i].elements[\$j].status = \"disclosure only (not drawn)\"");        W1+=('MISMATCH figs78 element table (b1) pin'); W2+=($DS)
IDS+=(N-b1b); EDITS+=("$T0 | $NOTD | .element_tables[\$i].elements[\$j].class |= (. + \" x\")");                          W1+=('MISMATCH figs78 element table (b1) pin'); W2+=($DS)
IDS+=(N-b1c); EDITS+=('.in_scope_outputs |= map(if .name == "Figs 7/8" then del(.downstream_tag.recomputed_output) else . end)'); W1+=($DS); W2+=($DS)
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/screst_${IDS[$i]}_$STAMP.json
  jq "${EDITS[$i]}" $REC > $F
  nl=$(leafdiff $REC $F)
  O=$HERE/_runs/screstneg_${STAMP}_${IDS[$i]}.out
  set +e; SC_REST=$HERE/sc_rest_v25.jq zsh -f $HERE/sc_core.zsh $F $MSG > $O 2>&1; rc=$?; set -e
  h1=false; grep -qxF -- "${W1[$i]}" $O && h1=true; h2=false; grep -qxF -- "${W2[$i]}" $O && h2=true
  print -r -- "NEG ${IDS[$i]} leaf_diff $nl exit $rc predicted_lines_present $h1 $h2 mismatch_lines $(grep -c '^MISMATCH ' $O || true)"
  if [[ ${IDS[$i]} == N-b1c ]]; then nok=$([[ $nl != 1 ]] && print true || print false); else nok=$([[ $nl == 1 ]] && print true || print false); fi
  [[ $nok == true && $rc == 1 && $h1 == true && $h2 == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "SCREST_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "SCREST_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
