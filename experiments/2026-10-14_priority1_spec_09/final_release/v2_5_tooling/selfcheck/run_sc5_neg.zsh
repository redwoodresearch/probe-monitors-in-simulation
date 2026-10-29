#!/bin/zsh -f
# SC5 negatives on the branch (i) draft c2 (predictions frozen in commit/PREDICTIONS_v25_negatives_20261021_0337.md, 550a6fc7…, and, for the
# branch (i) set, commit/PREDICTIONS_v25_branch_i_negatives_20261021_0351.md, 93a6505e…): plants of the real v2.5 draft (or, for G1, of a v19
# copy). Single-leaf plants are diff-counted to exactly 1 leaf; G7, G8 and B5 change 2+ leaves by construction (asserted != 1). G9 and G10
# (branch (ii)) are replaced by G9i and B1-B6; c3 adds R1 and A1; c4 adds N-tag1/2, N-read1/2, N-axis1 (multi-leaf); c5 adds N-m1, N-m2, N-m3 / N-m4a / N-m4b (multi-leaf). Each is predicted to fail with the named line. Prints fixture ids, leaf-diff counts, exit codes
# and booleans only.
#   zsh -f run_sc5_neg.zsh <draft.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; REC=${1:A}
P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl; P19=$D/inputs/provenance_final19_X_c20699a5d.jsonl
V24=$D/inputs/v2.4_record_6a46541d5.json
NEG=$HERE/negtest; mkdir -p $NEG $HERE/_runs
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
leafdiff() { jq -n --slurpfile a $1 --slurpfile b $2 '($a[0] | [paths(type | . != "object" and . != "array")]) as $pa | ($b[0] | [paths(type | . != "object" and . != "array")]) as $pb
  | if ($pa | sort) != ($pb | sort) then -1 else [$pa[] as $p | select(($a[0] | getpath($p)) != ($b[0] | getpath($p)))] | length end'; }
integer ok=0 n=0
# G1 (A2): a v19 copy with row 50's anchor edited -> a2_check.jq false (the copy differs from v19 in exactly 1 row)
(( ++n ))
jq -c 'if .row == 50 then .anchor = (.anchor + " x") else . end' $P19 > $NEG/v19_G1.jsonl
nd=$(jq -n --slurpfile a $P19 --slurpfile b $NEG/v19_G1.jsonl '[range(0; $a | length) as $i | select($a[$i] != $b[$i])] | length')
g1=$(jq -n -f $HERE/a2_check.jq --slurpfile a $P18 --slurpfile b $NEG/v19_G1.jsonl)
print -r -- "NEG G1 rows_changed $nd a2_check $g1"
[[ $nd == 1 && $g1 == false ]] && (( ++ok ))
typeset -a IDS EDITS WANTS
IDS+=(G2);  EDITS+=('.printed_items.all_rows |= map(if .row == 118 then .status = (.status + " ") else . end)'); WANTS+=('MISMATCH S2 a row'"'"'s status changed outside rows 119-121')
IDS+=(G3);  EDITS+=('(.in_scope_outputs | map(.name == "rc1b_detl48 printed items, slot 23 (rows 119-122)") | index(true)) as $i | .in_scope_outputs[$i].items[1].anchor = (.in_scope_outputs[$i].items[1].anchor + " ")'); WANTS+=('MISMATCH S3 the rows 119-122 group'"'"'s items are not v19 (119-121) + v2.4 (122)')
IDS+=(G4);  EDITS+=('.disclosures |= map(if .id == "v25.rows_119_121.B1X.1" then .text = (.text + " ") else . end)'); WANTS+=('MISMATCH S4 disclosure v25.rows_119_121.B1X.1 is not its GATES line'"'"'s quoted string')
IDS+=(G5);  EDITS+=('.row_rulings_v2_5.draft_scope.recheck.sha256 |= (.[0:63] + (if .[63:64] == "0" then "1" else "0" end))'); WANTS+=('MISMATCH S5 draft_scope is not the pre-stated block')
IDS+=(G6);  EDITS+=('.printed_items.provenance.commit = "98b5054fbf8306e54711b1a7bacd72b5f5e0af2a"'); WANTS+=('MISMATCH S6 printed_items changed beyond rows 119-121'"'"'s status, the counts and the group counts')
IDS+=(G7);  EDITS+=('(.printed_items.all_rows_status_counts | keys_unsorted) as $k | .printed_items.all_rows_status_counts[$k[0]] += 1 | .printed_items.all_rows_status_counts[$k[1]] -= 1'); WANTS+=('MISMATCH S8 all_rows_status_counts is not the map recomputed from all_rows')
IDS+=(G8);  EDITS+=('.row_rulings_v2_5.items |= map(if .id == "rows_119_121" then .rows = [119, 120] else . end)'); WANTS+=('MISMATCH S4b the v2.5 ruling item, its rows or its attachments are not the pre-stated table')
IDS+=(G9i); EDITS+=('.row_rulings_v2_5.figs78.included = false'); WANTS+=('MISMATCH S2 a row'"'"'s status changed outside rows 119-121')
IDS+=(B1);  EDITS+=('([$V[0].printed_items.all_rows[] | select(.row == 241) | .status][0]) as $s | .printed_items.all_rows |= map(if .row == 241 then .status = $s else . end)'); WANTS+=('MISMATCH S9i branch (i): the 14 slot-24 rows are not FINAL with the GATES reason')
IDS+=(B2);  EDITS+=('.figure_labels |= map(if (.file | test("fig_pertask_onpolicy_figs78")) then .label = "PRELIMINARY" else . end)'); WANTS+=('MISMATCH S9i branch (i): the Figs 7/8 labels are not FINAL with the GATES reason and the pins disclosure')
IDS+=(B3);  EDITS+=('(.element_tables | map(.kind == "figs78_item") | index(true)) as $i | .element_tables[$i].elements[0].status |= (tostring + " x")'); WANTS+=('MISMATCH S9i branch (i): the figs78 element tables are not the 854f901b6 record'"'"'s')
IDS+=(B4);  EDITS+=('.disclosures |= map(if .id == "v25.figs78.pins.1" then .text = (.text + " ") else . end)'); WANTS+=('MISMATCH S4 disclosure v25.figs78.pins.1 is not its GATES line'"'"'s quoted string')
IDS+=(B5);  EDITS+=('.row_rulings_v2_5.items |= map(if .id == "figs78" then .disclosure_rows["v25.figs78.pins.1"] |= .[1:] else . end)'); WANTS+=('MISMATCH S4b the v2.5 ruling item, its rows or its attachments are not the pre-stated table')
IDS+=(B6);  EDITS+=('(.element_tables | map(.kind == "figs78_item") | index(true)) as $i | .element_tables[$i].source_record.sha256 |= (.[0:63] + (if .[63:64] == "0" then "1" else "0" end))'); WANTS+=('MISMATCH S9i branch (i): the figs78 element tables are not the 854f901b6 record'"'"'s')
IDS+=(G11); EDITS+=('.in_scope_outputs |= map(if .name == "rc1b_detl48 printed items, slot 23 (rows 119-122)" then .name = (.name + " x") else . end)'); WANTS+=('MISMATCH S11 output names are not the pre-stated set')
# c3 (09's finding-2 ruling and item 2; predictions commit/PREDICTIONS_v25_c3_finding2_negatives_20261021_0422.md, 1ee84cc2…)
IDS+=(R1);  EDITS+=('.in_scope_outputs |= map(if .name == "rc1b printed items, slot 24 (App H / Fig 7-8 captions)" then .ruling = "figs78 (v2.5 branch (ii))" else . end)'); WANTS+=('MISMATCH S9i branch (i): the moved Figs 7/8 output or slot-24 group differs from v2.4'"'"'s beyond status / name / ruling')
IDS+=(A1);  EDITS+=('.in_scope_outputs |= map(if .name == "Figs 7/8" then .downstream_tag.recomputed_output.sha256 |= (.[0:63] + (if .[63:64] == "0" then "1" else "0" end)) else . end)'); WANTS+=('MISMATCH S9i branch (i): the moved Figs 7/8 output or slot-24 group differs from v2.4'"'"'s beyond status / name / ruling')
# c4 (09's c4 rulings; predictions commit/PREDICTIONS_v25_c4_all_negatives_20261021_0515.md, 478f49b0…)
IDS+=(N-tag1); EDITS+=('([$V[0].preliminary_outputs[] | select(.name == "Figs 7/8") | .downstream_tag.text][0]) as $t | .in_scope_outputs |= map(if .name == "Figs 7/8" then .downstream_tag.text = $t else . end)'); WANTS+=('MISMATCH S9i branch (i): the Figs 7/8 downstream_tag.text is not the v2.4 L103 string + the GATES L2430 suffix')
IDS+=(N-tag2); EDITS+=('.in_scope_outputs |= map(if .name == "rc1a" then .downstream_tag.text = ("PRELIMINARY " + .downstream_tag.text) else . end)'); WANTS+=('MISMATCH S10 an in_scope output'"'"'s downstream_tag.text starts with PRELIMINARY')
IDS+=(N-read1); EDITS+=('.row_rulings_v2_5.reading |= sub(" \\+ slot-24 rows 69-74 / 241-248 and Figs 7/8 FINAL \\(figs78 branch \\(i\\); 09, GATES L2430\\)"; "")'); WANTS+=('MISMATCH S9i branch (i): row_rulings_v2_5.reading is not 09'"'"'s ruled string')
IDS+=(N-read2); EDITS+=('.row_rulings_v2_5.reading |= sub("241-248"; "241-247")'); WANTS+=('MISMATCH S9i branch (i): row_rulings_v2_5.reading is not 09'"'"'s ruled string')
IDS+=(N-axis1); EDITS+=('.row_rulings_v2_5.items |= map(if .id == "figs78" then .disclosures -= ["v24.figs78_axis_presentation.axis.1"] else . end)'); WANTS+=('MISMATCH S4b the v2.5 ruling item, its rows or its attachments are not the pre-stated table')
# c5 (09's c5 ruling; predictions commit/PREDICTIONS_v25_c5_negatives_20261021_0601.md, c848a4cb…)
IDS+=(N-m1);  EDITS+=('([$V[0].preliminary_outputs[] | select(.name == "Figs 1/6") | .downstream_tag.text][0]) as $t | .preliminary_outputs |= map(if .name == "Figs 1/6" then .downstream_tag.text = $t else . end)'); WANTS+=('MISMATCH S12 the Figs 1/6 downstream_tag.text or label notes are not P + the joiner + v2.4'"'"'s text')
IDS+=(N-m2);  EDITS+=('.figure_labels[1].note = $V[0].figure_labels[1].note'); WANTS+=('MISMATCH S12 the Figs 1/6 downstream_tag.text or label notes are not P + the joiner + v2.4'"'"'s text')
IDS+=(N-m3);  EDITS+=('.figure_labels |= map(if .output == "Fig 4" then .note = "FINAL (planted)" else . end)'); WANTS+=('MISMATCH S13 a PRELIMINARY figure label'"'"'s note or reason starts with FINAL')
IDS+=(N-m4a); EDITS+=('.in_scope_outputs |= map(if .name == "rc1a" then del(.downstream_tag.text) else . end)'); WANTS+=('MISMATCH S10b an in_scope output'"'"'s object downstream_tag has no string text')
IDS+=(N-m4b); EDITS+=('.preliminary_outputs |= map(if .name == "Figs 1/6" then del(.downstream_tag.text) else . end)'); WANTS+=('MISMATCH S13 a preliminary output'"'"'s object downstream_tag has no string text')
for i in {1..${#IDS}}; do
  (( ++n ))
  F=$NEG/sc5_${IDS[$i]}.json
  jq --slurpfile V $V24 "${EDITS[$i]}" $REC > $F
  nl=$(leafdiff $REC $F)
  O=$HERE/_runs/sc5neg_${STAMP}_${IDS[$i]}.out
  set +e; zsh -f $HERE/sc_v25.zsh $F > $O 2>&1; rc=$?; set -e
  hit=false; grep -qxF -- "${WANTS[$i]}" $O && hit=true
  case ${IDS[$i]} in
    N-tag1) w2='MISMATCH S10 an in_scope output'"'"'s downstream_tag.text starts with PRELIMINARY' ;;
    N-m1)   w2='MISMATCH S13 a preliminary output'"'"'s downstream_tag.text starts with FINAL' ;;
    N-m2)   w2='MISMATCH S13 a PRELIMINARY figure label'"'"'s note or reason starts with FINAL' ;;
    *)      w2='' ;;
  esac
  [[ -z $w2 ]] || grep -qxF -- "$w2" $O || hit=false
  print -r -- "NEG ${IDS[$i]} leaf_diff $nl exit $rc predicted_mismatch_present $hit mismatch_lines $(grep -c '^MISMATCH ' $O || true)"
  if [[ ${IDS[$i]} == G7 || ${IDS[$i]} == G8 || ${IDS[$i]} == B5 || ${IDS[$i]} == N-axis1 || ${IDS[$i]} == N-m3 || ${IDS[$i]} == N-m4a || ${IDS[$i]} == N-m4b ]]; then nok=$([[ $nl != 1 ]] && print true || print false); else nok=$([[ $nl == 1 ]] && print true || print false); fi
  if [[ $nok == true && $rc == 1 && $hit == true ]]; then (( ++ok )); fi
done
if (( ok == n )); then print -r -- "SC5_NEG_ALL_FAIL_AS_PREDICTED $ok/$n"; else print -r -- "SC5_NEG_NOT_AS_PREDICTED $ok/$n"; exit 1; fi
