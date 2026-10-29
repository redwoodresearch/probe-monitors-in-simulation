#!/bin/zsh -f
# Message-check negatives on the real v2.7 message (predictions commit/PREDICTIONS_v27_addendum_20261021_1830.md, dea34b67…): copies
# with ONE line edited, added or removed (diff-counted first; a plant that changes nothing STOPs). Predicted for every fixture: msg_v27.zsh
# exits 1 and prints the named MISMATCH. Prints fixture ids, counts, exit codes and booleans only.
#   zsh -f run_msg_neg_v27.zsh <record.json> <message.txt> <number of files in v2_7_tooling/> [selftest]
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; REC=${1:A}; MSG=${2:A}; NT=${3:-}
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S); NEG=$HERE/negtest/msg_$STAMP; mkdir -p $NEG
typeset -a IDS SEDS WANTS
# v2.7 (predictions commit/PREDICTIONS_v27_addendum_20261021_1830.md, dea34b67…)
IDS+=(M1); SEDS+=('s/^Printed rows: 251; FINAL /Printed rows: 251; FINAL 1/');                 WANTS+=('MISMATCH message: row counts')
IDS+=(M3); SEDS+=('s/^Co-Authored-By: \(.*\)$/Co-Authored-By: \1\nClaude-Session: https:\/\/claude.ai\/code\/session_x/'); WANTS+=('MISMATCH message: no Claude-Session line')
IDS+=(M5); SEDS+=('s/(sha256 \([0-9a-f]\{63\}\)[0-9a-f])/(sha256 \1z)/');                      WANTS+=('MISMATCH message: record path and sha')
IDS+=(M7); SEDS+=('s/^Labels change only after/Planted extra line 4096.\nLabels change only after/'); WANTS+=('MISMATCH message: not the generated message')
IDS+=(W1); SEDS+=('s/^Fig 4: FINAL by record B (GATES L3160 @ f80f3cfe6)/Fig 4: FINAL by record B (GATES L3160)/'); WANTS+=('MISMATCH message: Fig 4|MISMATCH message: every GATES cite carries its commit')
IDS+=(W2); SEDS+=('s/^Fig 5: FINAL by/Fig 5: PRELIMINARY by/');                                  WANTS+=('MISMATCH message: Fig 5')
IDS+=(W3); SEDS+=('s/^rc1b: FINAL (printed items only) (GATES/rc1b: FINAL (GATES/');               WANTS+=('MISMATCH message: rc1b')
IDS+=(W4); SEDS+=('s/^Disclosures: 66 (/Disclosures: 65 (/');                                    WANTS+=('MISMATCH message: disclosures')
IDS+=(W5); SEDS+=('s/@ 0db7dc714 (2db9cdee)\./@ 0db7dc714 (2db9cdef)./');                        WANTS+=('MISMATCH message: supersedes v2.6')
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
  set +e; zsh -f $HERE/msg_v27.zsh $REC $F $NT > $O 2>&1; rc=$?; set -e
  hit=true; for w in ${(s:|:)WANTS[$i]}; do grep -qxF -- "$w" $O || hit=false; done
  print -r -- "NEG ${IDS[$i]} diff_lines $n exit $rc predicted_mismatch_present $hit mismatches $(grep -c '^MISMATCH ' $O || true)"
  [[ $n -ge 1 && $n -le 3 && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "MSG_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "MSG_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
