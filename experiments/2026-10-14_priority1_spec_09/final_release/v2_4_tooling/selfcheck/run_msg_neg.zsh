#!/bin/zsh -f
# Message-check negatives: copies of the real v2.4 message, each with ONE line changed (diff-counted first). Predicted for every fixture:
# msg_v24.zsh exits 1 and prints the named MISMATCH. Prints fixture ids, counts, exit codes and booleans only.
#   zsh -f run_msg_neg.zsh <record.json> <message.txt> <number of files in v2_4_tooling/>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; REC=${1:A}; MSG=${2:A}; NT=${3:-}
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S); NEG=$HERE/negtest/msg_$STAMP; mkdir -p $NEG
typeset -a IDS SEDS WANTS
IDS+=(M1); SEDS+=('s/^Printed rows: 251; FINAL /Printed rows: 251; FINAL 1/');                 WANTS+=('MISMATCH message: row counts')
IDS+=(M2); SEDS+=('s/: 4, 5, 9, 10, 14, 15, 19, 20, 25, 27, 28, 29;/: 4, 5, 9, 10, 14, 15, 19, 20, 25, 27, 28;/'); WANTS+=('MISMATCH message: the 12 corrected rows')
IDS+=(M3); SEDS+=('s/^Co-Authored-By: \(.*\)$/Co-Authored-By: \1\nClaude-Session: https:\/\/claude.ai\/code\/session_x/'); WANTS+=('MISMATCH message: no Claude-Session line')
IDS+=(M4); SEDS+=('s/R_node switch: OFF\./R_node switch: ON./');                                WANTS+=('MISMATCH message: switch state')
IDS+=(M5); SEDS+=('s/(sha256 \([0-9a-f]\{63\}\)[0-9a-f])/(sha256 \1z)/');                      WANTS+=('MISMATCH message: record path and sha')
IDS+=(M6); SEDS+=('s/^Rows 119-122: row 122 FINAL/Rows 119-122: row 122 PRELIMINARY/');        WANTS+=('MISMATCH message: row 122')
# b8 (Codex v24 r1 item 6; 09 sensitivity counts)
IDS+=(M7); SEDS+=('s/^Labels change only after/Planted extra line 4096.\nLabels change only after/'); WANTS+=('MISMATCH message: not the generated message')
IDS+=(M8); SEDS+=('s/the 4 printed rows that rest on them/the 3 printed rows that rest on them/');  WANTS+=('MISMATCH message: sensitivity counts')
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/${IDS[$i]}.txt
  sed "${SEDS[$i]}" $MSG > $F
  n=$( (diff $MSG $F || true) | grep -c '^[<>]' || true)
  O=$HERE/_runs/msgneg_${STAMP}_${IDS[$i]}.out
  set +e; zsh -f $HERE/msg_v24.zsh $REC $F $NT > $O 2>&1; rc=$?; set -e
  hit=false; grep -qxF -- "${WANTS[$i]}" $O && hit=true
  print -r -- "NEG ${IDS[$i]} diff_lines $n exit $rc predicted_mismatch_present $hit mismatches $(grep -c '^MISMATCH ' $O || true)"
  [[ $n -ge 1 && $n -le 3 && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "MSG_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "MSG_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
