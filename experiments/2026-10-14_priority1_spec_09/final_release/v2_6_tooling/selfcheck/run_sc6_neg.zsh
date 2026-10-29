#!/bin/zsh -f
# SC6 negatives (predictions frozen in commit/PRESTATEMENT_v26.md, 33a40662…, and for N23-N26 commit/PRESTATEMENT_v26_c2.md, 5e37095d…): plants of the real v2.6 draft. Each plant names its exact
# path (a jq locator on the draft); leafdiff.jq must show exactly that change (C21): "changed" = one leaf at the path; "added" = new leaves
# only under the path; "removed" = removed leaves only under the path; "replaced" = leaves removed under the path and one leaf added at it.
# A plant that changes nothing STOPs (exit 3). Each is predicted to fail SC6 (exit 1) with every named line ('|'-separated) present.
# Prints fixture ids, leaf-diff kinds, exit codes and booleans only.   zsh -f run_sc6_neg.zsh <draft.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; REC=${1:A}; [[ -f $REC ]] || { echo "STOP usage"; exit 2; }
V25=$D/inputs/v2.5_record_4df86c4b9.json
NEG=$HERE/negtest; mkdir -p $NEG $HERE/_runs; STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
typeset -a IDS LOCS OPS KINDS WANTS
add() { IDS+=($1); LOCS+=($2); OPS+=($3); KINDS+=($4); WANTS+=($5); }
ROWI='(.printed_items.all_rows | map(.row) | index($n))'
add N1  "4 as \$n | [\"printed_items\", \"all_rows\", $ROWI, \"status\"]" 'setpath($p; ($V[0].printed_items.all_rows[] | select(.row == 4) | .status))' changed 'MISMATCH S2 the status-changed rows are not exactly the 65 scope rows'
add N2  "93 as \$n | [\"printed_items\", \"all_rows\", $ROWI, \"status\"]" 'setpath($p; getpath($p) + " ")' changed 'MISMATCH S2 the status-changed rows are not exactly the 65 scope rows'
add N3  '["row_rulings_v2_6", "items", (.row_rulings_v2_6.items | map(.id) | index("intervals_option3")), "status"]' 'setpath($p; getpath($p) | sub("\\)$"; "]"))' changed "MISMATCH S3 a flipped row's status or the item status is not the composed FINAL string"
add N4  '["printed_items", "all_rows_status_counts", (.printed_items.all_rows_status_counts | keys | map(select(startswith("FINAL (INTERVALS"))) | .[0])]' 'setpath($p; getpath($p) + 1)' changed 'MISMATCH S4 all_rows_status_counts is not the recomputed map'
add N5  '["row_rulings_v2_6", "items", (.row_rulings_v2_6.items | map(.id) | index("intervals_option3")), "cites", 1]' 'delpaths([$p])' removed 'MISMATCH S5 the intervals_option3 item is not the pre-stated one'
add N6  '["disclosures", (.disclosures | map(.id) | index("v26.option3_intervals.L2878.1")), "text"]' 'setpath($p; getpath($p) + " ")' changed "MISMATCH S6 the v2.6 disclosure is not L2878's quoted string at the GATES commit"
add N7  '["disclosures", (.disclosures | map(.id) | index("row92_label")), "status_superseded_by", "disclosure_id"]' 'setpath($p; "v24.row_92.9")' changed 'MISMATCH S7 a status_superseded_by target does not resolve, or its GATES line does not exist|MISMATCH S7 row92_label: status_superseded_by is not the ruled target'
add N8  '["disclosures", (.disclosures | map(.id) | index("C3_downstream_judgment")), "status_superseded_by", "gates_cite", "line"]' 'setpath($p; 99999)' changed 'MISMATCH S7 a status_superseded_by target does not resolve, or its GATES line does not exist'
add N9  '["disclosures", 0, "text"]' 'setpath($p; getpath($p) + " ")' changed 'MISMATCH S7 a v2.5 disclosure changed beyond the added status_superseded_by key'
add N10 '["row_rulings_v2_4", "items", (.row_rulings_v2_4.items | map(.id) | index("interval_value_half_only")), "status_superseded_by"]' 'delpaths([$p])' removed "MISMATCH S8 the v2.4 items' status_superseded_by is not the ruled set, or row_rulings_v2_4 changed otherwise"
add N11 '["row_rulings_v2_4", "r_node_switch", "superseded_by_ruling", "line"]' 'setpath($p; 2870)' changed "MISMATCH S9 r_node_switch.superseded_by_ruling is not L2869's cite"
add N12 '["element_tables", 1, "elements", (.element_tables[1].elements | map(has("interval_note")) | index(true)), "status"]' 'setpath($p; "PRELIMINARY")' changed 'MISMATCH S10 the Figs 1/6 element tables are not v2.5'"'"'s with the interval elements FINAL and the composed note / rule'
add N13 '["figure_labels", 1, "reason"]' 'setpath($p; $V[0].figure_labels[1].reason)' added 'MISMATCH S14 a FINAL label has a .reason'
add N14 '["figure_labels", 1, "preliminary_elements_n"]' 'setpath($p; 60)' changed 'MISMATCH S11 the Figs 1/6 labels are not FINAL with the composed note / rule, no reason, the recomputed count and the v2.6 disclosure'
add N15 '["figure_labels", (.figure_labels | map(.output) | index("Fig 4")), "reason"]' 'setpath($p; getpath($p) | sub("y$"; "Y"))' changed "MISMATCH S13 Fig 4's three texts are not 09's string, or Fig 4 changed otherwise"
add N16 '["preliminary_outputs", (.preliminary_outputs | map(.name) | index("Fig 4")), "interval_rule", "rule"]' 'setpath($p; ($V[0].preliminary_outputs[] | select(.name == "Fig 4") | .interval_rule.rule))' changed "MISMATCH S13 Fig 4's three texts are not 09's string, or Fig 4 changed otherwise"
add N17 '["in_scope_outputs", (.in_scope_outputs | map(.name) | index("Figs 1/6")), "downstream_tag", "text"]' 'setpath($p; ($V[0].preliminary_outputs[] | select(.name == "Figs 1/6") | .downstream_tag.text))' changed 'MISMATCH S12 the Figs 1/6 output is not v2.5'"'"'s moved to in_scope with the ruled fields|MISMATCH S14 a FINAL row / item / output / label / table carries PRELIMINARY-prefixed status text'
add N18 '["in_scope_outputs", (.in_scope_outputs | map(.name) | index("rc1a")), "downstream_tag"]' 'setpath($p; "FINAL x")' replaced 'MISMATCH S14 a downstream_tag is not null or an object with a string text'
add N19 '["row_rulings_v2_6", "reading"]' 'setpath($p; getpath($p) | sub("v2\\.5$"; "v2.4"))' changed 'MISMATCH S18 row_rulings_v2_6.reading is not the pre-stated string'
add N20 '["row_rulings_v2_5", "items", (.row_rulings_v2_5.items | map(.id) | index("figs78")), "disclosures", ((.row_rulings_v2_5.items[] | select(.id == "figs78") | .disclosures) | index("v24.figs78_axis_presentation.axis.1"))]' 'delpaths([$p])' removed 'MISMATCH S15 a conditional disclosure whose condition holds is not attached to its current item'
add N21 '["supersedes", "sha256"]' 'setpath($p; getpath($p) | .[0:63] + (if .[63:64] == "0" then "1" else "0" end))' changed 'MISMATCH S17 version / repo_path / supersedes'
add N22 '["printed_items", "groups_n"]' 'setpath($p; getpath($p) + 1)' changed 'MISMATCH S16 groups_n / items_n are not the recomputed totals'
# c2 (09 via 2e; predictions commit/PRESTATEMENT_v26_c2.md, 5e37095d…)
add N23 '["figure_labels", 1, "interval_rule", "disclosures", 1]' 'setpath($p; "v24.figs_interval_rule.INT-FIG.1")' added 'MISMATCH S21 a FINAL output / table / label / item lists a disclosure whose text starts with PRELIMINARY|MISMATCH S11 the Figs 1/6 labels are not FINAL with the composed note / rule, no reason, the recomputed count and the v2.6 disclosure'
add N24 '["disclosures", (.disclosures | map(.id) | index("v26.fig4_record_b.L3001.1")), "text"]' 'setpath($p; getpath($p) + " ")' changed "MISMATCH S22 the Fig 4 disclosure is not GATES L3001's quoted string"
add N25 '["disclosures", (.disclosures | map(.id) | index("v26.fig4_record_b.L3001.1")), "text"]' 'setpath($p; "FINAL " + getpath($p))' changed "MISMATCH S21 a PRELIMINARY output / table / label / item lists a disclosure whose text starts with FINAL|MISMATCH S22 the Fig 4 disclosure is not GATES L3001's quoted string"
add N26 '["in_scope_outputs", (.in_scope_outputs | map(.name) | index("Figs 1/6")), "interval_rule", "disclosures", 0]' 'setpath($p; "v26.no_such_id")' changed "MISMATCH S21 a listed disclosure id does not resolve|MISMATCH S12 the Figs 1/6 output is not v2.5's moved to in_scope with the ruled fields"
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/sc6_${IDS[$i]}_$STAMP.json
  P=$(jq -c "${LOCS[$i]}" $REC)
  jq --argjson p "$P" --slurpfile V $V25 "${OPS[$i]}" $REC > $F
  LD=$(jq -n -c -f $HERE/leafdiff.jq --slurpfile a $REC --slurpfile b $F)
  shape=$(jq -n -c --argjson d "$LD" --argjson p "$P" --arg k "${KINDS[$i]}" 'def under: (.[0:($p | length)] == $p);
    if ($d.removed + $d.added + $d.changed | length) == 0 then "noop"
    elif $k == "changed" then (if $d.changed == [$p] and $d.added == [] and $d.removed == [] then "ok" else "bad" end)
    elif $k == "added" then (if $d.changed == [] and $d.removed == [] and ($d.added | length) >= 1 and all($d.added[]; under) then "ok" else "bad" end)
    elif $k == "removed" then (if $d.changed == [] and $d.added == [] and ($d.removed | length) >= 1 and all($d.removed[]; under) then "ok" else "bad" end)
    elif $k == "replaced" then (if $d.changed == [] and $d.added == [$p] and ($d.removed | length) >= 1 and all($d.removed[]; under) then "ok" else "bad" end)
    else "bad" end' | tr -d '"')
  [[ $shape != noop ]] || { print -r -- "STOP: negative ${IDS[$i]} plant changed nothing"; exit 3; }
  O=$HERE/_runs/sc6neg_${STAMP}_${IDS[$i]}.out
  set +e; zsh -f $HERE/sc_v26.zsh $F > $O 2>&1; rc=$?; set -e
  hit=true; for w in ${(s:|:)WANTS[$i]}; do grep -qxF -- "$w" $O || hit=false; done
  print -r -- "NEG ${IDS[$i]} kind ${KINDS[$i]} leaf_shape $shape exit $rc predicted_lines_present $hit mismatch_lines $(grep -c '^MISMATCH ' $O || true)"
  [[ $shape == ok && $rc == 1 && $hit == true ]] && (( ++ok ))
done
if (( ok == ${#IDS} )); then print -r -- "SC6_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "SC6_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
