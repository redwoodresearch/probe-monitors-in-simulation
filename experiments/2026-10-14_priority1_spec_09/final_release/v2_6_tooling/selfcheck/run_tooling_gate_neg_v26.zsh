#!/bin/zsh -f
# Tooling-gate negatives for v2_6_tooling (the v2.5 plants retargeted; predictions commit/PREDICTIONS_v26_msg_gate_screst_20261021_1503.md, 876afe6b…):
# copies of a real staged v2_6_tooling folder, each with ONE file changed by one plant (the changed-file count and the difference are checked
# first). Predicted: tooling_gate.zsh exits 1 and prints the named FLAG. Plants are built at run time (no value-like literal in this file);
# v18 strings are taken from the pinned v18 file and never printed. Prints fixture ids, counts, exit codes and booleans only.
#   zsh -f run_tooling_gate_neg_v26.zsh <staged v2_6_tooling folder>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; T=${1:A}
P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl
[[ $(shasum -a 256 $P18 | cut -c1-64) == 13d29fb90cac3e3e54145b5d9f41319614a0ff63ebf33b5b4de3d251419741b0 ]] || { echo "STOP v18 sha"; exit 2; }
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S); NEG=$HERE/negtest/tooling_$STAMP; mkdir -p $NEG
DEC="0$(print -n .)$(print -n 1)23"; PCT="3$(print -n 7)%"
P251=$(jq -r 'select(.row == 251) | [.printed | .. | strings | [match("[0-9][0-9,]*"; "g").string | gsub(","; "")]] | flatten | .[0]' $P18)
FI='integers in key-ish code strings equal a printed token'
typeset -a IDS FILES WANTS
IDS+=(T1);  FILES+=(build/v26_strings.json);  WANTS+=('FLAG build/v26_strings.json: decimal token outside a key or version')
IDS+=(T2);  FILES+=(build/evidence_v26.json);    WANTS+=('FLAG build/evidence_v26.json: holds a provenance item / anchor string')
IDS+=(T3);  FILES+=(build/evidence_v26.json);    WANTS+=('FLAG build/evidence_v26.json: 1 provenance / value keys')
IDS+=(T4);  FILES+=(build/evidence_v26.json);    WANTS+=('FLAG build/evidence_v26.json: 1 non-integer numbers')
IDS+=(T5);  FILES+=(selfcheck/sc_v26.zsh);     WANTS+=('FLAG selfcheck/sc_v26.zsh: percent token other than the 95% interval class word')
IDS+=(T6);  FILES+=(selfcheck/sc_v26.jq);      WANTS+=('FLAG selfcheck/sc_v26.jq: holds a printed token')
IDS+=(T7);  FILES+=(selfcheck/sc_v26.zsh);     WANTS+=('FLAG selfcheck/sc_v26.zsh: decimal token outside a key or version')
IDS+=(T8);  FILES+=(build/evidence_v26.json);    WANTS+=('FLAG build/evidence_v26.json: decimal token outside a key or version')
IDS+=(T9);  FILES+=(selfcheck/sc_v26.jq);      WANTS+=('FLAG selfcheck/sc_v26.jq: decimal token outside a key or version')
IDS+=(T10); FILES+=(selfcheck/sc_v26.jq);      WANTS+=('FLAG selfcheck/sc_v26.jq: percent as a word')
IDS+=(T11); FILES+=(build/evidence_v26.json);    WANTS+=('FLAG build/evidence_v26.json: 1 integers in key-ish fields equal a printed token')
IDS+=(T12); FILES+=(build/build_v2_6.jq);      WANTS+=("FLAG build/build_v2_6.jq: 1 $FI")
IDS+=(T13); FILES+=(build/build_v2_6.jq);      WANTS+=("FLAG build/build_v2_6.jq: 1 $FI")
IDS+=(T14); FILES+=(build/build_v2_6.jq);      WANTS+=("FLAG build/build_v2_6.jq: 1 $FI")
IDS+=(T15); FILES+=(build/build_v2_6.jq);      WANTS+=("FLAG build/build_v2_6.jq: 1 $FI")
IDS+=(T16); FILES+=(build/build_v2_6.jq);      WANTS+=("FLAG build/build_v2_6.jq: 1 $FI")
IDS+=(T17); FILES+=(selfcheck/sc_v26.jq);      WANTS+=('FLAG selfcheck/sc_v26.jq: decimal token outside a key or version')
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/${IDS[$i]}; mkdir -p $F; cp -R $T/. $F/; chmod -R u+w $F; tgt=$F/${FILES[$i]}
  case ${IDS[$i]} in
    T1) jq -c --arg d "$DEC" '.reading += (" " + $d)' $T/${FILES[$i]} > $tgt ;;
    T2) jq -c --slurpfile P $P18 '.note = ([$P[] | .item | strings | select(length >= 12)][0])' $T/${FILES[$i]} > $tgt ;;
    T3) jq -c '.predicate = {}' $T/${FILES[$i]} > $tgt ;;
    T4) jq -c '.lines = [1, (5 / 2)]' $T/${FILES[$i]} > $tgt ;;
    T5) print -r -- "# plant $PCT" >> $tgt ;;
    T6) jq -r --slurpfile P $P18 -n '[$P[] | .printed | .. | strings | select(test("[0-9]") and test("[.]"))][0] | "# plant " + .' >> $tgt ;;
    T7) print -r -- "# plant ends with a decimal $DEC." >> $tgt ;;
    T8) jq -c --arg d "$DEC" '.note = $d' $T/${FILES[$i]} > $tgt ;;
    T9) print -r -- "# plant $DEC|x" >> $tgt ;;
    T10) print -r -- "# plant 3$(print -n 7) percent" >> $tgt ;;
    T11) jq -c --arg n "$P251" '.note_key = ("(" + $n + " counted)")' $T/${FILES[$i]} > $tgt ;;
    T12) print -r -- "def plant12: {half2_key: \"($P251 counted)\"};" >> $tgt ;;
    T13) print -r -- "def plant13: {half2_key: (if . then \"a\" else \"($P251 counted)\" end)};" >> $tgt ;;
    T14) { print -r -- "def plant14: {half2_key: (if . then \"a\" else"; print -r -- "  \"($P251 counted)\" end)};"; } >> $tgt ;;
    T15) print -r -- "def plant15: {\"half2_key\": \"($P251 counted)\"};" >> $tgt ;;
    T16) print -r -- "def plant16: .half2_key = \"($P251 counted)\";" >> $tgt ;;
    T17) print -r -- "# plant L2878$(print -n .)1 x" >> $tgt ;;   # v2.6: a decimal not inside a release id (the idmask extension)
  esac
  nchanged=$( (diff -rq $T $F || true) | wc -l | tr -d ' ')
  differs=$(cmp -s $T/${FILES[$i]} $tgt && print false || print true)
  O=$HERE/_runs/toolneg26_${STAMP}_${IDS[$i]}.out
  set +e; zsh -f $HERE/tooling_gate.zsh $F > $O 2>&1; rc=$?; set -e
  hit=false; grep -E '^FLAG ' $O | sed -E 's/ \(line [0-9]+\)$//' | grep -qxF -- "${WANTS[$i]}" && hit=true
  print -r -- "NEG ${IDS[$i]} files_changed $nchanged target_differs $differs exit $rc predicted_flag_present $hit flags $(grep -c '^FLAG ' $O || true)"
  if [[ $nchanged == 1 && $differs == true && $rc == 1 && $hit == true ]]; then (( ++ok )); fi
done
if (( ok == ${#IDS} )); then print -r -- "TOOLING_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "TOOLING_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
