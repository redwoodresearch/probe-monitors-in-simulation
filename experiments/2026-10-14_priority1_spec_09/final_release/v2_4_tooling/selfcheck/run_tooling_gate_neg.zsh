#!/bin/zsh -f
# Tooling-gate negatives: copies of a real staged v2_4_tooling/ folder, each with ONE file changed by one plant (the changed-file count
# and the leaf / line difference are checked first). Predicted for every fixture: tooling_gate.zsh exits 1 and prints the named FLAG.
# Plants are built at run time (no value-like literal in this file); the v18 strings are taken from the pinned v18 file, never printed.
# Prints fixture ids, counts, exit codes and booleans only.   zsh -f run_tooling_gate_neg.zsh <staged v2_4_tooling folder>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; T=${1:A}
P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl
[[ $(shasum -a 256 $P18 | cut -c1-16) == 13d29fb90cac3e3e ]] || { echo "STOP v18 sha"; exit 2; }
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S); NEG=$HERE/negtest/tooling_$STAMP; mkdir -p $NEG
DEC="0$(print -n .)$(print -n 1)23"; PCT="3$(print -n 7)%"
typeset -a IDS FILES WANTS
IDS+=(T1); FILES+=(build/rulings_v2_4.json);  WANTS+=('FLAG build/rulings_v2_4.json: decimal token outside a key or version')
IDS+=(T2); FILES+=(build/r122_recheck.json);  WANTS+=('FLAG build/r122_recheck.json: holds a provenance item / anchor string')
IDS+=(T3); FILES+=(build/r122_recheck.json);  WANTS+=('FLAG build/r122_recheck.json: 1 provenance / value keys')
IDS+=(T4); FILES+=(q1/q1_rowtable_new145_20261020_232829_r251cite.json); WANTS+=('FLAG q1/q1_rowtable_new145_20261020_232829_r251cite.json: 1 non-integer numbers')
IDS+=(T5); FILES+=(selfcheck/sc_v24.zsh);     WANTS+=('FLAG selfcheck/sc_v24.zsh: percent token other than the 95% interval class word')
IDS+=(T6); FILES+=(selfcheck/sc_v24.jq);      WANTS+=('FLAG selfcheck/sc_v24.jq: holds a printed token')
# b8 (fresh review N4-N7; 09: integers in key-ish fields). The row-251 printed integer is read from v18 at run time and never printed.
P251=$(jq -r 'select(.row == 251) | [.printed | .. | strings | [match("[0-9][0-9,]*"; "g").string | gsub(","; "")]] | flatten | .[0]' $P18)
IDS+=(T7);  FILES+=(selfcheck/sc_v24.zsh);    WANTS+=('FLAG selfcheck/sc_v24.zsh: decimal token outside a key or version')
IDS+=(T8);  FILES+=(build/r122_recheck.json); WANTS+=('FLAG build/r122_recheck.json: decimal token outside a key or version')
IDS+=(T9);  FILES+=(selfcheck/sc_v24.jq);     WANTS+=('FLAG selfcheck/sc_v24.jq: decimal token outside a key or version')
IDS+=(T10); FILES+=(selfcheck/sc_v24.jq);     WANTS+=('FLAG selfcheck/sc_v24.jq: percent as a word')
IDS+=(T11); FILES+=(q1/q1_rowtable_new145_20261020_232829_r251cite.json); WANTS+=('FLAG q1/q1_rowtable_new145_20261020_232829_r251cite.json: 1 integers in key-ish fields equal a printed token')
IDS+=(T12); FILES+=(q1/q1_rowtable.jq);       WANTS+=('FLAG q1/q1_rowtable.jq: 1 integers in key-ish code strings equal a printed token')
# b9-tooling (Codex v24 r2 item 12): the original b7 R3 site form, a printed integer inside the conditional half2_key expression
IDS+=(T13); FILES+=(q1/q1_rowtable.jq);       WANTS+=('FLAG q1/q1_rowtable.jq: 1 integers in key-ish code strings equal a printed token')
# b9-2 (Codex v24 r3 item 6; predictions frozen in commit/PREDICTIONS_b9_2_negatives_20261021_0258.md): the else-string moved onto the
# next line of the conditional (T14), and a quoted key (T15)
IDS+=(T14); FILES+=(q1/q1_rowtable.jq);       WANTS+=('FLAG q1/q1_rowtable.jq: 1 integers in key-ish code strings equal a printed token')
IDS+=(T15); FILES+=(selfcheck/sc_v24.jq);     WANTS+=('FLAG selfcheck/sc_v24.jq: 1 integers in key-ish code strings equal a printed token')
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/${IDS[$i]}; mkdir -p $F; cp -R $T/. $F/; chmod -R u+w $F; tgt=$F/${FILES[$i]}
  case ${IDS[$i]} in
    T1) jq -c --arg d "$DEC" '.items[0].status += (" " + $d)' $T/${FILES[$i]} > $tgt ;;
    T2) jq -c --slurpfile P $P18 '.note = ([$P[] | .item | strings | select(length >= 12)][0])' $T/${FILES[$i]} > $tgt ;;
    T3) jq -c '.predicate = {}' $T/${FILES[$i]} > $tgt ;;
    T4) jq -c 'first(paths(type == "number")) as $p | setpath($p; getpath($p) + (1 / 2))' $T/${FILES[$i]} > $tgt ;;
    T5) print -r -- "# plant $PCT" >> $tgt ;;
    T6) jq -r --slurpfile P $P18 -n '[$P[] | .printed | .. | strings | select(test("[0-9]") and test("[.]"))][0] | "# plant " + .' >> $tgt ;;
    T7) print -r -- "# plant ends with a decimal $DEC." >> $tgt ;;
    T8) jq -c --arg d "$DEC" '.note = $d' $T/${FILES[$i]} > $tgt ;;
    T9) print -r -- "# plant $DEC|x" >> $tgt ;;
    T10) print -r -- "# plant 3$(print -n 7) percent" >> $tgt ;;
    T11) jq -c --arg n "$P251" '(.rows[] | select(.row == 251) | .half2_key) |= (. + " (" + $n + " counted)")' $T/${FILES[$i]} > $tgt ;;
    T12) print -r -- "def plant: {half2_key: \"($P251 counted)\"};" >> $tgt ;;
    T13) sed "s/(the count under every convention: GATES L1362 @ d4466dbe5; ruling GATES /($P251 under every convention, GATES /" $T/${FILES[$i]} > $tgt ;;
    T14) awk -v n="$P251" '{ s = "else \"same convention-independent count (the count under every convention: GATES L1362 @ d4466dbe5; ruling GATES "; i = index($0, s)
           if (i > 0) { print substr($0, 1, i + 4); print "           \"same convention-independent count (" n " under every convention, GATES " substr($0, i + length(s)) } else print }' $T/${FILES[$i]} > $tgt ;;
    T15) print -r -- "def plant2: {\"half2_key\": \"($P251 counted)\"};" >> $tgt ;;
  esac
  nchanged=$( (diff -rq $T $F || true) | wc -l | tr -d ' ')
  differs=$(cmp -s $T/${FILES[$i]} $tgt && print false || print true)
  O=$HERE/_runs/toolneg_${STAMP}_${IDS[$i]}.out
  set +e; zsh -f $HERE/tooling_gate.zsh $F > $O 2>&1; rc=$?; set -e
  hit=false; grep -E '^FLAG ' $O | sed -E 's/ \(line [0-9]+\)$//' | grep -qxF -- "${WANTS[$i]}" && hit=true
  print -r -- "NEG ${IDS[$i]} files_changed $nchanged target_differs $differs exit $rc predicted_flag_present $hit flags $(grep -c '^FLAG ' $O || true)"
  [[ $nchanged == 1 && $differs == true && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "TOOLING_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "TOOLING_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
