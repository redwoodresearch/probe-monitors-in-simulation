#!/bin/zsh -f
# SC7 negatives (predictions frozen in commit/PRESTATEMENT_v27.md, fd03b8c0…, and N15-N18 in PREDICTIONS_v27_addendum_20261021_1830.md, dea34b67…): plants of the real v2.7 draft (N7: of a v19 copy, run through the
# driver's negatives-only override, which forces a failure on its own). Each plant names its exact path (a jq locator); leafdiff.jq must show
# exactly that change (C21: "changed" = one leaf at the path; "added" / "removed" = leaves only under the path). A plant that changes nothing
# STOPs (exit 3). Each is predicted to fail SC7 (exit 1) with every named line ('|'-separated) present. Prints ids, shapes, exit codes, booleans.
#   zsh -f run_sc7_neg.zsh <draft.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; REC=${1:A}; [[ -f $REC ]] || { echo "STOP usage"; exit 2; }
V26=$D/inputs/v2.6_record_0db7dc714.json; P19=$D/inputs/provenance_final19_X_c20699a5d.jsonl
NEG=$HERE/negtest; mkdir -p $NEG $HERE/_runs; STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
typeset -a IDS LOCS OPS KINDS WANTS
add() { IDS+=($1); LOCS+=($2); OPS+=($3); KINDS+=($4); WANTS+=($5); }
F4L='(.figure_labels | map(.output) | index("Fig 4"))'; F5L='(.figure_labels | map(.output) | index("Fig 5"))'
DI() { print -r -- "(.disclosures | map(.id) | index(\"$1\"))"; }
T1='MISMATCH T1 the Fig 4 label is not FINAL with 09'"'"'s fields (264 / 0, no reason, the L3171 disclosure, the L3195 note, the L3160 rule and cite)'
T7P='MISMATCH T7 an old Fig 4 / Fig 5 phrase sits in a current, non-exempt string'
add N1  "[\"figure_labels\", $F4L, \"interval_rule\", \"disclosures\", 1]" 'setpath($p; "v26.fig4_record_b.L3001.1")' added "$T1|MISMATCH T7 a listed disclosure id does not resolve, or a FINAL item lists a superseded one"
add N2  "[\"figure_labels\", $F5L, \"caption_only_elements_n\"]" 'setpath($p; 5)' changed "MISMATCH T2 the Fig 5 label is not FINAL with 09's fields (0 preliminary, 6 caption-only, the L3200 note, the L3074 disclosure)"
add N3  "[\"figure_labels\", $F4L, \"reason\"]" 'setpath($p; ($V[0].figure_labels[] | select(.output == "Fig 4") | .reason))' added "$T1|MISMATCH T7 a FINAL figure label has a reason or PRELIMINARY wording in its note / rule"
add N4  "[\"disclosures\", $(DI v27.fig4_record_b.L3171.1), \"text\"]" 'setpath($p; getpath($p) + " ")' changed "MISMATCH T4 the Fig 4 disclosure is not GATES L3171's quoted string with the full cite"
add N5  "[\"disclosures\", $(DI v27.fig5_option_a.L3074.1), \"text\"]" 'setpath($p; getpath($p) + " ")' changed "MISMATCH T4 the Fig 5 disclosure is not GATES L3074's quoted string with the full cite"
add N6  '["in_scope_outputs", (.in_scope_outputs | map(.name) | index("rc1b")), "ruling_text"]' 'setpath($p; getpath($p) + " ")' changed "MISMATCH T3 rc1b is not moved with the ruled status / ruling_text (L2982's 2nd quoted string) / cite / out-of-scope keys"
add N7  'P19' 'P19' p19 'MISMATCH T6 a printed row reads rc1b slots 23.* / 25.*, an element-table string references an out-of-scope input, or the 5 keys differ'
add N8  "[\"figure_labels\", $F4L, \"note\"]" 'setpath($p; getpath($p) + " stream gate")' changed "$T1|$T7P"
add N9  '["g7b_cross_check", "rule"]' 'setpath($p; getpath($p) + " Fig 4 stream gate")' changed "$T7P"
add N10 '["exclusions", (.exclusions | map(.id) | index("C3_saturated_undefined")), "entries", 0, "status"]' 'setpath($p; "PRELIMINARY")' changed 'MISMATCH T2 exclusion C3_saturated_undefined is not flipped in place as ruled'
add N11 "[\"disclosures\", $(DI v26.fig4_record_b.L3001.1), \"status_superseded_by\", \"disclosure_id\"]" 'setpath($p; "v27.no_such_id")' changed 'MISMATCH T5 a status_superseded_by target does not resolve, or its GATES line does not exist|MISMATCH T5 a status_superseded_by is not the ruled target'
add N12 '["disclosures", 0, "text"]' 'setpath($p; getpath($p) + " ")' changed 'MISMATCH T5 a v2.6 disclosure changed beyond the ruled status_superseded_by keys, or the count is not 66'
add N13 '["sensitivity_rows", "status"]' 'setpath($p; $V[0].sensitivity_rows.status)' changed 'MISMATCH T8 sensitivity_rows.status is not the Q5 text (L3230, <N> from v2.6)'
add N14 '["preliminary_outputs", (.preliminary_outputs | length)]' 'setpath($p; ($V[0].preliminary_outputs[] | select(.name == "Fig 5")))' added 'MISMATCH T8 the outputs are not v2.6'"'"'s with Fig 4 / Fig 5 / rc1b moved (preliminary_outputs == [routec_simpool_numbers printed items])'
# 09 L3243 condition (2) (predictions commit/PREDICTIONS_v27_addendum_20261021_1830.md, dea34b67…)
add N15 "[\"figure_labels\", $F5L, \"note\"]" 'setpath($p; getpath($p) + " saturated_undefined")' changed "$T7P|MISMATCH T2 the Fig 5 label is not FINAL with 09's fields (0 preliminary, 6 caption-only, the L3200 note, the L3074 disclosure)"
add N16 '["exclusions", (.exclusions | map(.id) | index("C3_saturated_undefined")), "entries", 0, "entry"]' 'setpath($p; getpath($p) + " saturated_undefined")' changed "$T7P|MISMATCH T2 exclusion C3_saturated_undefined is not flipped in place as ruled"
add N17 "[\"figure_labels\", $F4L, \"disclosures\", 1]" 'setpath($p; "v26.fig4_record_b.L3001.1")' added "MISMATCH T9 a current .disclosures list names a superseded disclosure|$T1"
add N18 "[\"figure_labels\", $F4L, \"note\"]" 'setpath($p; getpath($p) + " stays PRELIMINARY")' changed "$T7P|$T1|MISMATCH T7 a FINAL figure label has a reason or PRELIMINARY wording in its note / rule"
integer ok=0
for i in {1..${#IDS}}; do
  O=$HERE/_runs/sc7neg_${STAMP}_${IDS[$i]}.out
  if [[ ${KINDS[$i]} == p19 ]]; then
    # a fake printed row: the first rc1b row on slot 24.* moved to slot 23.x (one row changed in a v19 copy)
    F19=$NEG/sc7_${IDS[$i]}_v19_$STAMP.jsonl
    jq -c -s --arg r "experiments/2026-10-17_routec_slot_readout/results/slots_routeC_writing_rc1b.json@" '(map((.record | startswith($r)) and ((.key_path[1] // "") | tostring | startswith("24."))) | index(true)) as $k
      | to_entries | map(if .key == $k then (.value.key_path[1] = ("23.x" + (.value.key_path[1] | tostring | .[2:]))) | .value else .value end) | .[]' $P19 > $F19
    nd=$(jq -n --slurpfile a $P19 --slurpfile b $F19 '[range(0; $a | length) as $k | select($a[$k] != $b[$k])] | length')
    [[ $nd == 1 ]] && shape=ok || shape=bad; [[ $nd == 0 ]] && { print -r -- "STOP: negative ${IDS[$i]} plant changed nothing"; exit 3; }
    set +e; zsh -f $HERE/sc_v27.zsh $REC $F19 > $O 2>&1; rc=$?; set -e
  else
    F=$NEG/sc7_${IDS[$i]}_$STAMP.json
    P=$(jq -c "${LOCS[$i]}" $REC)
    jq --argjson p "$P" --slurpfile V $V26 "${OPS[$i]}" $REC > $F
    LD=$(jq -n -c -f $HERE/leafdiff.jq --slurpfile a $REC --slurpfile b $F)
    shape=$(jq -n -c --argjson d "$LD" --argjson p "$P" --arg k "${KINDS[$i]}" 'def under: (.[0:($p | length)] == $p);
      if ($d.removed + $d.added + $d.changed | length) == 0 then "noop"
      elif $k == "changed" then (if $d.changed == [$p] and $d.added == [] and $d.removed == [] then "ok" else "bad" end)
      elif $k == "added" then (if $d.changed == [] and $d.removed == [] and ($d.added | length) >= 1 and all($d.added[]; under) then "ok" else "bad" end)
      elif $k == "removed" then (if $d.changed == [] and $d.added == [] and ($d.removed | length) >= 1 and all($d.removed[]; under) then "ok" else "bad" end)
      else "bad" end' | tr -d '"')
    [[ $shape != noop ]] || { print -r -- "STOP: negative ${IDS[$i]} plant changed nothing"; exit 3; }
    set +e; zsh -f $HERE/sc_v27.zsh $F > $O 2>&1; rc=$?; set -e
  fi
  hit=true; for w in ${(s:|:)WANTS[$i]}; do grep -qxF -- "$w" $O || hit=false; done
  print -r -- "NEG ${IDS[$i]} kind ${KINDS[$i]} leaf_shape $shape exit $rc predicted_lines_present $hit mismatch_lines $(grep -c '^MISMATCH ' $O || true)"
  [[ $shape == ok && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "SC7_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "SC7_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
