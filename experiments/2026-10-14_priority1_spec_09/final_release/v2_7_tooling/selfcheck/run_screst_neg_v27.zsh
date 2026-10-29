#!/bin/zsh -f
# Ported self-check negatives (sc_core.zsh with sc_rest_v27.jq and the message) on the real v2.7 record: the v2.5 figs78-table plants
# (S1, N-b1a, N-b1b, N-b1c) and N-corr1 (09 Q4: r_node_switch.superseded_by_ruling removed), plus v2.7's (A)-(D) negatives (09, GATES L3266-3284
# @ 1bed0178a): A1-A3, B1-B4, C1, D1, D2. Predictions: v2.6 commit/PREDICTIONS_v26_msg_gate_screst_20261021_1503.md (876afe6b…); v2.7
# PREDICTIONS_v27_addendum_20261021_1830.md (dea34b67…) and PREDICTIONS_v27_addendum2_20261021_1917.md (b07169d7…). First C0: every v2.7 constant in
# sc_rest_v27.jq against its source (a STOP on any difference). Each plant names its exact path; leafdiff.jq must show exactly that change (C21;
# shapes as in run_sc7_neg.zsh); a plant that changes nothing STOPs. A plant changes the record's sha, so the message's record-sha check also
# fails; only the named lines ('|'-separated; "re:" = an extended regex) are asserted present.   zsh -f run_screst_neg_v27.zsh <record.json> <message.txt>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; REC=${1:A}; MSG=${2:A}
[[ -f $REC && -f $MSG ]] || { echo "STOP usage"; exit 2; }
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
G() { git -C $REPO "$@"; }
V26=$D/inputs/v2.6_record_0db7dc714.json
[[ $(shasum -a 256 $V26 | cut -c1-64) == 2db9cdee658f30637f76d7ce507789f72b4b5854879a9ce6bcdcfee3605d2b96 ]] || { echo "STOP v2.6 sha"; exit 2; }
SR=$HERE/sc_rest_v27.jq

# ---------------- C0: the v2.7 constants against their sources ----------------
GF=experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md
GC=$(G rev-parse 1bed0178a); [[ $GC == 1bed0178aa75035cfce3083fcaeb68e926f39ab5 ]] || { echo "STOP GATES commit"; exit 2; }
G branch -r --contains $GC | tr -d ' ' | grep -qx origin/aswin/convergence-audit-09 || { echo "STOP GATES commit not on origin/aswin/convergence-audit-09"; exit 2; }
[[ $(G cat-file blob "${GC}:${GF}" | shasum -a 256 | cut -c1-64) == 8d0b1ef44368d325b116dd0579e712a36d99808812e9eff45223deead4d9fb4e ]] || { echo "STOP GATES file sha"; exit 2; }
GL() { G cat-file blob "${GC}:${GF}" | sed -n "${1}p" | tr -d '\n'; }
typeset -A LSHA=(3270 356c02ce162cfa3ca919dbebad522eff9925c9745a546c08c963d31470e61318 3272 9e6180f841499f1b9adb78cc1897f39e861343242aaa2aeaf71d2efd275a28cc
                 3280 f686619f48db981e2babcfdd537401bdfa9477ba6e87b6e6ceb7abd0194f75e3)
for n in ${(k)LSHA}; do [[ $(GL $n | shasum -a 256 | cut -c1-64) == ${LSHA[$n]} ]] || { echo "STOP GATES L$n line sha"; exit 2; }; done
Q() { GL $1 | jq -R -s -c '[match("\"([^\"]*)\""; "g") | .captures[0].string]'; }
DEF() { local n; n=$(grep -c -E "^def $2: " $1 || true); [[ $n == 1 ]] || { echo "STOP def $2 found $n times in ${1:t}" >&2; return 2; }; grep -E "^def $2: " $1 | sed -E "s/^def $2: (.*);\$/\\1/"; }
CKDEF='def V27_CK($b; $m): reduce (tojson | explode[]) as $c (0; ((. * $b) + $c) as $x | $x - ($m * (($x / $m) | floor)));'
typeset -a CN CV
cst() { CN+=($1); CV+=($2); print -r -- "CONST $1 $2"; }
cst K1_allow_eq_L3270 $(jq -n --argjson a "$(DEF $SR V27_ALLOW)" --argjson b "$(Q 3270)" '$a == $b and ($b | length) == 2')
cst K2_pio_eq_L3272 $(jq -n --argjson a "$(DEF $SR V27_PIO)" --argjson b "$(Q 3272)" '($b | length) == 1 and $a == $b[0]')
cst K3_d_eq_L3280 $(jq -n --argjson a "$(DEF $SR V27_D)" --argjson b "$(Q 3280)" '$a == $b and ($b | length) == 2')
L74=$(GL 3274)
cst K4_cite $(jq -n --argjson a "$(DEF $SR V27_CITE)" --arg c "$(G rev-parse 97b6e634d)" --arg l "$L74" \
  '$a == {line: 2982, commit: $c} and ($c | test("^[0-9a-f]{40}$")) and ($l | contains("ruling_cite.line == 2982")) and ($l | contains("97b6e634d"))')
k5() { [[ $(grep -cF -- "$1" $HERE/sc_v27.jq || true) == 1 && $(grep -cF -- "$1" $SR || true) == 1 ]] && print true || print false; }
cst K5_slot_re_eq_sc7_T6 $(k5 '"^(23|25)\\."')
cst K5_str_re_eq_sc7_T6 $(k5 '"detector_vs_drop|panel_plant_rows|slots_routeC_writing_rc1b"')
cst K6_p18_sha $(jq -n --argjson a "$(DEF $SR V27_P18_SHA)" --arg b "$(sed -n 16p $HERE/sc_v27.zsh | grep -o '[0-9a-f]\{64\}')" '$a == $b')
cst K6_p19_sha $(jq -n --argjson a "$(DEF $SR V27_P19_SHA)" --arg b "$(sed -n 17p $HERE/sc_v27.zsh | grep -o '[0-9a-f]\{64\}')" '$a == $b')
cst K7_ck_def_text $( [[ $(grep -cxF -- "$CKDEF" $SR || true) == 1 ]] && print true || print false )
K7=$(jq -c "$CKDEF"' [.exclusions[] | select(.id == "C3a_g7b_and_cleanroom_gap" or .id == "C3b_not_cleanroom_verdict")
  | ([.entries[] | [.output, .file, (if has("slot") then "slot" else "value" end), (if has("slot") then .slot else .value_index end)] | tojson] | sort) as $k
  | {(.id): {n: ($k | length), h1: ($k | V27_CK(131; 2147483647)), h2: ($k | V27_CK(257; 1000000007))}}] | add' $V26)
cst K7_v26_c3 $(jq -n --argjson a "$(DEF $SR V26_C3)" --argjson b "$K7" '$a == $b and ($b | keys) == ["C3a_g7b_and_cleanroom_gap", "C3b_not_cleanroom_verdict"]')
cst rc1b_path_eq_sc7_RC1B_REC $(jq -c --argjson rr "$(DEF $HERE/sc_v27.jq RC1B_REC)" '[.in_scope_outputs[] | select(.name == "rc1b") | [.files[] | select(.path | endswith(".json"))]] as $f
  | ($f | length) == 1 and ($f[0] | length) == 1 and ($f[0][0].path + "@") == $rr' $REC)
PV=$(jq -r '.printed_items.provenance.commit + ":" + .printed_items.provenance.path, .row_rulings_v2_5.draft_scope.printed_and_anchor_source.commit + ":" + .row_rulings_v2_5.draft_scope.printed_and_anchor_source.path' $REC)
cst record_v18_blob_sha $( [[ $(G cat-file blob "${${(f)PV}[1]}" | shasum -a 256 | cut -c1-64) == $(DEF $SR V27_P18_SHA | tr -d '"') ]] && print true || print false )
cst record_v19_blob_sha $( [[ $(G cat-file blob "${${(f)PV}[2]}" | shasum -a 256 | cut -c1-64) == $(DEF $SR V27_P19_SHA | tr -d '"') ]] && print true || print false )
# K8: 09's correction (GATES L3291 @ eca0ca69c): the (i-c) gate key = the token after "key == " on L3291
GC2=$(G rev-parse eca0ca69c); [[ $GC2 == eca0ca69c1ab2153dd7f9a2e9caef2cec1876f44 ]] || { echo "STOP GATES correction commit"; exit 2; }
G branch -r --contains $GC2 | tr -d ' ' | grep -qx origin/aswin/convergence-audit-09 || { echo "STOP GATES correction commit not on origin/aswin/convergence-audit-09"; exit 2; }
[[ $(G cat-file blob "${GC2}:${GF}" | shasum -a 256 | cut -c1-64) == dec234ebfe9a06ad51123faa971997414b4d4a1520f13698956e9c0b300de6e3 ]] || { echo "STOP GATES correction file sha"; exit 2; }
L91=$(G cat-file blob "${GC2}:${GF}" | sed -n 3291p | tr -d '\n')
[[ $(print -rn -- "$L91" | shasum -a 256 | cut -c1-64) == e08b7693c5045df4aedc1b880d9a95dffabcab26a4a2aa453b2c87e12e584f4b ]] || { echo "STOP GATES L3291 line sha"; exit 2; }
cst K8_gate_key_eq_L3291 $(jq -n --argjson a "$(DEF $SR V27_GATE_KEY)" --arg l "$L91" '([$l | match("key == ([A-Za-z0-9_]+)\\."; "g") | .captures[0].string]) as $t | ($t | length) == 1 and $a == $t[0]')
for i in {1..${#CN}}; do [[ ${CV[$i]} == true ]] || { print -r -- "STOP C0 ${CN[$i]}"; exit 2; }; done
print -r -- "CONST_OK ${#CN}"

# ---------------- negatives ----------------
NEG=$HERE/negtest; mkdir -p $NEG $HERE/_runs; STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
RC1B=$(jq -c '[.in_scope_outputs[] | select(.name == "rc1b")][0] | {c: .files[0].commit, p: .files[0].path, k: .out_of_scope_keys}' $REC)
SWAPK=$(G cat-file blob "$(jq -r '.c + ":" + .p' <<< $RC1B)" | jq -r --argjson o "$RC1B" '[.inputs | keys[] | select(. as $k | $o.k | index([$k]) == null)] | sort | .[0] // ""')
[[ -n $SWAPK ]] || { echo "STOP no swap key"; exit 2; }
typeset -a IDS LOCS OPS KINDS WANTS
add() { IDS+=($1); LOCS+=($2); OPS+=($3); KINDS+=($4); WANTS+=($5); }
T0='(.element_tables | map(.kind == "figs78_item") | index(true)) as $i'
DS='MISMATCH FINAL output Figs 7/8 reads a downstream record without a recorded recompute and full FINAL element tables'
add S1     "$T0 | [\"element_tables\", \$i, \"elements\", (.element_tables[\$i].elements | map(.status == \"FINAL\") | index(true)), \"status\"]" 'setpath($p; "PRELIMINARY")' changed 'MISMATCH figure status|MISMATCH figs78 element table class counts'
add N-b1a  "$T0 | [\"element_tables\", \$i, \"elements\", (.element_tables[\$i].elements | map(.status == \"FINAL\") | index(true)), \"status\"]" 'setpath($p; "disclosure only (not drawn)")' changed "MISMATCH figs78 element table (b1) pin|$DS"
add N-b1b  "$T0 | [\"element_tables\", \$i, \"elements\", (.element_tables[\$i].elements | map(.status == \"disclosure only (not drawn)\") | index(true)), \"class\"]" 'setpath($p; getpath($p) + " x")' changed "MISMATCH figs78 element table (b1) pin|$DS"
add N-b1c  '["in_scope_outputs", (.in_scope_outputs | map(.name) | index("Figs 7/8")), "downstream_tag", "recomputed_output"]' 'delpaths([$p])' removed "$DS"
add N-corr1 '["row_rulings_v2_4", "r_node_switch", "superseded_by_ruling"]' 'delpaths([$p])' removed 're:^MISMATCH printed row [0-9]+ on sensitivity slot .*: status vs sensitivity_halves$'
# v2.7 (A)-(D) (PREDICTIONS_v27_addendum2 P7)
F4='(.in_scope_outputs | map(.name) | index("Fig 4")) as $j'
RB='(.in_scope_outputs | map(.name) | index("rc1b")) as $j'
SX='(.exclusions | map(.id) | index("C3_saturated_undefined")) as $j'
WA='MISMATCH Fig 4 status (v2.7: (A) interval-rule allow-list; (B) printed items only)'
WBS='MISMATCH rc1b status (v2.7: (A) interval-rule allow-list; (B) printed items only)'
WSE='MISMATCH sensitivity rows cover the FINAL outputs (before figure splits)'
WC2='MISMATCH exclusion C3a_g7b_and_cleanroom_gap entries|MISMATCH exclusion C3b_not_cleanroom_verdict entries'
WCO="MISMATCH exclusion C3a_g7b_and_cleanroom_gap entries differ from v2.6's (count + key set; v2.7 (C))|MISMATCH exclusion C3b_not_cleanroom_verdict entries differ from v2.6's (count + key set; v2.7 (C))"
WBA='MISMATCH rc1b FINAL (printed items only): (B)(i-a) the covered inputs outside in_scope_records (gate inputs included), mapped to keys, differ from out_of_scope_keys'
WBC='MISMATCH rc1b FINAL (printed items only): (B)(i-c) the gate input'"'"'s key is not LABELS_171'
WB3='MISMATCH rc1b FINAL (printed items only): (B)(iii) T6_zero_printed_dependence is false'
VB() { print -r -- "V27B rc1b: i_a $1 i_b true i_c $2 ii_cite true iii_T6_zero_printed_dependence $3"; }
WD='MISMATCH C3_saturated_undefined: no halves, and an entry status neither starts with PRELIMINARY nor is exactly caption-only, not plotted (v2.7 (D))'
add A1 "$F4 | [\"in_scope_outputs\", \$j, \"interval_rule\", \"rule\"]" 'setpath($p; ([$V26[0].preliminary_outputs[] | select(.name == "Fig 4")][0].interval_rule.rule))' changed "$WA"
add A2 "$F4 | [\"in_scope_outputs\", \$j, \"interval_rule\", \"rule\"]" 'setpath($p; "FINAL (other)")' changed "$WA"
add A3 "$F4 | [\"in_scope_outputs\", \$j, \"interval_rule\", \"disclosures\", 0]" 'setpath($p; "v26.fig4_record_b.L3001.1")' changed "$WA"
# B1-B4 re-pointed by 09's correction (GATES L3293 @ eca0ca69c): a dropped key, a swapped key, the gate key renamed, T6 false
add B1 "$RB | [\"in_scope_outputs\", \$j, \"out_of_scope_keys\", 0]" 'delpaths([$p])' arrdel "$(VB false true true)|$WBA|$WBS|$WSE|$WC2|$WCO"
add B2 "$RB | [\"in_scope_outputs\", \$j, \"out_of_scope_keys\", 0]" 'setpath($p; $K)' changed "$(VB false false true)|$WBA|$WBC|$WBS|$WSE|$WC2|$WCO"
add B3 "$RB | [\"in_scope_outputs\", \$j, \"out_of_scope_keys\", (.in_scope_outputs[\$j].out_of_scope_keys | index(\"LABELS_171\"))]" 'setpath($p; "LABELS_171_renamed")' changed "$(VB false false true)|$WBA|$WBC|$WBS|$WSE|$WC2|$WCO"
add B4 '["element_tables", 0, "elements", (.element_tables[0].elements | map(has("reason")) | index(true)), "reason"]' 'setpath($p; getpath($p) + " slots_routeC_writing_rc1b")' changed "$(VB true true false)|$WB3|$WBS|$WSE|$WC2|$WCO"
add C1 "$RB | [\"in_scope_outputs\", \$j, \"status\"]" 'setpath($p; "FINAL")' changed "$WC2|$WCO|$WBS|$WSE"
add D1 "$SX | [\"exclusions\", \$j, \"entries\", 0, \"status\"]" 'setpath($p; "FINAL")' changed "$WD|MISMATCH exclusion C3_saturated_undefined flip by intersection"
add D2 "$SX | [\"exclusions\", \$j, \"entries\", 0, \"status\"]" 'setpath($p; "caption-only")' changed "$WD"
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/screst27_${IDS[$i]}_$STAMP.json
  P=$(jq -c "${LOCS[$i]}" $REC)
  jq --argjson p "$P" --slurpfile V26 $V26 --arg K "$SWAPK" "${OPS[$i]}" $REC > $F
  LD=$(jq -n -c -f $HERE/leafdiff.jq --slurpfile a $REC --slurpfile b $F)
  shape=$(jq -n -c --argjson d "$LD" --argjson p "$P" --arg k "${KINDS[$i]}" 'def under: (.[0:($p | length)] == $p);
    if ($d.removed + $d.added + $d.changed | length) == 0 then "noop"
    elif $k == "changed" then (if $d.changed == [$p] and $d.added == [] and $d.removed == [] then "ok" else "bad" end)
    elif $k == "removed" then (if $d.changed == [] and $d.added == [] and ($d.removed | length) >= 1 and all($d.removed[]; under) then "ok" else "bad" end)
    elif $k == "arrdel" then (($p | length) - 1) as $n
      | (if $d.added == [] and ($d.removed | length) == 1 and all(($d.removed + $d.changed)[]; .[0:$n] == $p[0:$n]) then "ok" else "bad" end)
    else "bad" end' | tr -d '"')
  [[ $shape != noop ]] || { print -r -- "STOP: negative ${IDS[$i]} plant changed nothing"; exit 3; }
  O=$HERE/_runs/screstneg27_${STAMP}_${IDS[$i]}.out
  set +e; SC_REST=$SR zsh -f $HERE/sc_core.zsh $F $MSG > $O 2>&1; rc=$?; set -e
  hit=true; for w in ${(s:|:)WANTS[$i]}; do if [[ $w == re:* ]]; then grep -qE -- "${w#re:}" $O || hit=false; else grep -qxF -- "$w" $O || hit=false; fi; done
  print -r -- "NEG ${IDS[$i]} kind ${KINDS[$i]} leaf_shape $shape exit $rc predicted_lines_present $hit mismatch_lines $(grep -c '^MISMATCH ' $O || true)"
  [[ $shape == ok && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "SCREST_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "SCREST_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
