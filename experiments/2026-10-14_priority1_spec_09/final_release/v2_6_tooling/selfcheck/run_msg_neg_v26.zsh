#!/bin/zsh -f
# Message-check negatives on the real v2.6 message (predictions commit/PREDICTIONS_v26_msg_gate_screst_20261021_1503.md, 876afe6b…): copies
# with ONE line edited, added or removed (diff-counted first; a plant that changes nothing STOPs). Predicted for every fixture: msg_v26.zsh
# exits 1 and prints the named MISMATCH. Prints fixture ids, counts, exit codes and booleans only.
#   zsh -f run_msg_neg_v26.zsh <record.json> <message.txt> <number of files in v2_6_tooling/> [selftest]
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; REC=${1:A}; MSG=${2:A}; NT=${3:-}
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S); NEG=$HERE/negtest/msg_$STAMP; mkdir -p $NEG
typeset -a IDS SEDS WANTS
IDS+=(M1); SEDS+=('s/^Printed rows: 251; FINAL /Printed rows: 251; FINAL 1/');                 WANTS+=('MISMATCH message: row counts')
IDS+=(M3); SEDS+=('s/^Co-Authored-By: \(.*\)$/Co-Authored-By: \1\nClaude-Session: https:\/\/claude.ai\/code\/session_x/'); WANTS+=('MISMATCH message: no Claude-Session line')
IDS+=(M5); SEDS+=('s/(sha256 \([0-9a-f]\{63\}\)[0-9a-f])/(sha256 \1z)/');                      WANTS+=('MISMATCH message: record path and sha')
IDS+=(M7); SEDS+=('s/^Labels change only after/Planted extra line 4096.\nLabels change only after/'); WANTS+=('MISMATCH message: not the generated message')
IDS+=(V1); SEDS+=('s/(12 v2\.4 corrections/(13 v2.4 corrections/');                            WANTS+=('MISMATCH message: interval rows')
IDS+=(V2); SEDS+=('s/both labels FINAL/both labels PRELIMINARY/');                              WANTS+=('MISMATCH message: Figs 1/6')
IDS+=(V3); SEDS+=('s/^Fig 4: stays PRELIMINARY/Fig 4: is FINAL/');                              WANTS+=('MISMATCH message: Fig 4')
IDS+=(V4); SEDS+=('s/: 8 disclosures and/: 7 disclosures and/');                                WANTS+=('MISMATCH message: disclosures and keys')
IDS+=(V5); SEDS+=('s/are FINAL by RULING A (interval ends)\./are PRELIMINARY by [INT-CORR] (interval ends)./'); WANTS+=('MISMATCH message: sensitivity rows FINAL')
IDS+=(V6); SEDS+=('s/@ 4d43ba0a4 (09/@ 4d43ba0a5 (09/');                                         WANTS+=('MISMATCH message: GATES commit and lines')
IDS+=(V7); SEDS+=('s/(0d16330e)\./(0d16330f)./');                                                WANTS+=('MISMATCH message: supersedes v2.5')
# c3 (predictions commit/PREDICTIONS_v26_c3_msg_20261021_1727.md, 7fa8b64c…)
IDS+=(V8); SEDS+=('s/(Fig 4: RULING B, GATES L3001 @ 89435ec9a)/(Fig 4: RULING B, GATES L3001)/');            WANTS+=('MISMATCH message: disclosures and keys')
# 2e's standing rule (c5): a negative's plant must change the message. "selftest" as the 4th argument runs only a deliberate no-op plant,
# which must trip the guard (predictions commit/PREDICTIONS_v25_c5_m10c_guard_20261021_0610.md, 2a7bdc8d…)
if [[ ${4:-} == selftest ]]; then IDS=(NOOP); SEDS=('s/__no_such_text_in_the_message__/x/'); WANTS=('-'); fi
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/${IDS[$i]}.txt
  sed "${SEDS[$i]}" $MSG > $F
  n=$( (diff $MSG $F || true) | grep -c '^[<>]' || true)
  (( n >= 1 )) || { print -r -- "STOP: negative ${IDS[$i]} plant changed nothing"; exit 3; }
  O=$HERE/_runs/msgneg_${STAMP}_${IDS[$i]}.out
  set +e; zsh -f $HERE/msg_v26.zsh $REC $F $NT > $O 2>&1; rc=$?; set -e
  hit=false; grep -qxF -- "${WANTS[$i]}" $O && hit=true
  print -r -- "NEG ${IDS[$i]} diff_lines $n exit $rc predicted_mismatch_present $hit mismatches $(grep -c '^MISMATCH ' $O || true)"
  [[ $n -ge 1 && $n -le 3 && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "MSG_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "MSG_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
