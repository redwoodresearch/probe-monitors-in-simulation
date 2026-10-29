#!/bin/zsh -f
# SC0 negatives: the port must FAIL on single-leaf plants of the real v2.3 record (each fixture diff-counted to exactly 1 changed leaf
# before it runs). Predicted for every fixture: port exit 1 (SELFCHECK_FAIL) and the named MISMATCH line present.
#   zsh -f run_sc0_neg.zsh
# Prints fixture ids, leaf-diff counts, exit codes and booleans only.
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}   # v2.5: run from this folder (was a fixed v2_4r path)
REC=$D/inputs/v2.3_record_80a77ecdc.json
MSG=$D/inputs/msg_final_release_v2.3.txt
[[ $(shasum -a 256 $REC | cut -c1-64) == ebb47e28f00596535f4fbaa5bdadb2b93b413bb768fb05e99b1abd36cac7de4f ]] || { echo "STOP record sha"; exit 2; }
NEG=$HERE/negtest; mkdir -p $NEG $HERE/_runs
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
# id | jq edit (keys only) | the MISMATCH line it must produce
typeset -a IDS EDITS WANTS
IDS+=(N1); EDITS+=('(.exclusions[] | select(.id == "C3a_g7b_and_cleanroom_gap") | .entries[0].status) = "PRELIMINARY"'); WANTS+=('MISMATCH exclusion C3a_g7b_and_cleanroom_gap flip by intersection')
IDS+=(N2); EDITS+=('.final_bar.g7b_value.sha256 |= (if .[0:1] == "0" then "1" else "0" end) + .[1:]'); WANTS+=('MISMATCH $.final_bar.g7b_value sha256')
IDS+=(N3); EDITS+=('.release_wording.source.line += 1'); WANTS+=('MISMATCH $.release_wording.source anchor line sha')
IDS+=(N4); EDITS+=('.printed_items.all_rows |= map(if .row == 3 then .status = "PRELIMINARY (planted)" else . end)'); WANTS+=('MISMATCH printed row 3 on 09.catch_avg.six|C.sim should be FINAL')
leafdiff() { jq -n --slurpfile a $1 --slurpfile b $2 '($a[0] | [paths(type | . != "object" and . != "array")]) as $pa | ($b[0] | [paths(type | . != "object" and . != "array")]) as $pb
  | if ($pa | sort) != ($pb | sort) then -1 else [$pa[] as $p | select(($a[0] | getpath($p)) != ($b[0] | getpath($p)))] | length end'; }
typeset -a PIDS
for i in {1..${#IDS}}; do
  F=$NEG/sc0_${IDS[$i]}.json
  jq "${EDITS[$i]}" $REC > $F
  n=$(leafdiff $REC $F)
  print -r -- "NEG ${IDS[$i]} leaf_diff $n"
  [[ $n == 1 ]] || { echo "STOP fixture ${IDS[$i]} does not differ by exactly 1 leaf"; exit 2; }
  ( set +e; zsh -f $HERE/sc_core.zsh $F $MSG > $HERE/_runs/sc0neg_${STAMP}_${IDS[$i]}.out; print $? > $HERE/_runs/sc0neg_${STAMP}_${IDS[$i]}.rc ) &
  PIDS+=($!)
done
wait
integer ok=0
for i in {1..${#IDS}}; do
  O=$HERE/_runs/sc0neg_${STAMP}_${IDS[$i]}.out; rc=$(cat $HERE/_runs/sc0neg_${STAMP}_${IDS[$i]}.rc)
  hit=false; grep -qxF -- "${WANTS[$i]}" $O && hit=true
  last=$(tail -1 $O)
  print -r -- "NEG ${IDS[$i]} exit $rc last_line $last predicted_mismatch_present $hit mismatch_lines $(grep -c '^MISMATCH ' $O || true)"
  [[ $rc == 1 && $hit == true && $last == SELFCHECK_FAIL ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "SC0_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "SC0_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
