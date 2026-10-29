#!/bin/zsh -f
# Message-check negatives: copies of the real v2.5 message (c2: branch (i); M6 -> M6i, + M9-M11; c5: + N-msg1, N-msg2, M10 -> M10c, a no-op guard), each with ONE line changed (diff-counted first). Predicted for every fixture:
# msg_v25.zsh exits 1 and prints the named MISMATCH. Prints fixture ids, counts, exit codes and booleans only.
#   zsh -f run_msg_neg.zsh <record.json> <message.txt> <number of files in v2_4_tooling/>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; REC=${1:A}; MSG=${2:A}; NT=${3:-}
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S); NEG=$HERE/negtest/msg_$STAMP; mkdir -p $NEG
typeset -a IDS SEDS WANTS
IDS+=(M1); SEDS+=('s/^Printed rows: 251; FINAL /Printed rows: 251; FINAL 1/');                 WANTS+=('MISMATCH message: row counts')
IDS+=(M2); SEDS+=('s/(re-verify afe11c809;/(re-verify afe11c80x;/');                         WANTS+=('MISMATCH message: rows 119-121')
IDS+=(M3); SEDS+=('s/^Co-Authored-By: \(.*\)$/Co-Authored-By: \1\nClaude-Session: https:\/\/claude.ai\/code\/session_x/'); WANTS+=('MISMATCH message: no Claude-Session line')
IDS+=(M4); SEDS+=('s/R_node switch: OFF\./R_node switch: ON./');                                WANTS+=('MISMATCH message: switch')
IDS+=(M5); SEDS+=('s/(sha256 \([0-9a-f]\{63\}\)[0-9a-f])/(sha256 \1z)/');                      WANTS+=('MISMATCH message: record path and sha')
IDS+=(M6i); SEDS+=('s/(figs78 branch (i):/(figs78 branch (ii):/');                         WANTS+=('MISMATCH message: figs78 branch')
# b8 (Codex v24 r1 item 6; 09 sensitivity counts)
IDS+=(M7); SEDS+=('s/^Labels change only after/Planted extra line 4096.\nLabels change only after/'); WANTS+=('MISMATCH message: not the generated message')
IDS+=(M8); SEDS+=('s/c20699a5d, a8bf8fce), A2\./c20699a5d, a8bf8fcf), A2./');                WANTS+=('MISMATCH message: A2 provenance line')
# branch (i) (c2; predictions commit/PREDICTIONS_v25_branch_i_msg_negatives_20261021_0404.md, 98402305…)
IDS+=(M9);  SEDS+=('s/ and L2438, copied/, copied/');                                         WANTS+=('MISMATCH message: GATES commit and lines')
IDS+=(M10c); SEDS+=('s/, v25\.figs78\.pins\.1, v24/, v24/');                                    WANTS+=('MISMATCH message: v2.5 disclosure ids')
IDS+=(M11); SEDS+=('s/a0492000 CONFIRMED/a0492001 CONFIRMED/');                               WANTS+=('MISMATCH message: figs78 branch')
# c5 (predictions commit/PREDICTIONS_v25_c5_negatives_20261021_0601.md, c848a4cb…)
IDS+=(N-msg1); SEDS+=('/^v2\.4 erratum: /d');                                                  WANTS+=('MISMATCH message: v2.4 erratum line')
IDS+=(N-msg2); SEDS+=('s/ (now applies: Figs 7\/8 are in this version)\.$/./');                 WANTS+=('MISMATCH message: v2.5 disclosure ids')
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
  set +e; zsh -f $HERE/msg_v25.zsh $REC $F $NT > $O 2>&1; rc=$?; set -e
  hit=false; grep -qxF -- "${WANTS[$i]}" $O && hit=true
  print -r -- "NEG ${IDS[$i]} diff_lines $n exit $rc predicted_mismatch_present $hit mismatches $(grep -c '^MISMATCH ' $O || true)"
  [[ $n -ge 1 && $n -le 3 && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "MSG_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "MSG_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
