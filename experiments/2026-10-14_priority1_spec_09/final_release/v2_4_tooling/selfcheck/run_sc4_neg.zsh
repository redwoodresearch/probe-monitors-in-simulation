#!/bin/zsh -f
# SC4 negatives: single-leaf plants of a real v2.4 draft (each diff-counted to exactly 1 changed leaf first). Predicted for every fixture:
# sc_v24.zsh exits 1 (SC4_FAIL) and prints the named MISMATCH line. Prints fixture ids, leaf-diff counts, exit codes and booleans only.
#   zsh -f run_sc4_neg.zsh <draft.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; REC=${1:A}
NEG=$HERE/negtest; mkdir -p $NEG $HERE/_runs
STAMP=$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)
typeset -a IDS EDITS WANTS
IDS+=(F1); EDITS+=('.printed_items.all_rows |= map(if .row == 4 then .status = "FINAL" else . end)'); WANTS+=('MISMATCH S3 row 4: FINAL != expected PRELIMINARY')
IDS+=(F2); EDITS+=('.printed_items.all_rows |= map(if .row == 30 then .status = (.status + " ") else . end)'); WANTS+=('MISMATCH S4 another row'"'"'s status changed on rows 1-106')
IDS+=(F3); EDITS+=('.disclosures |= map(if .id == "v24.interval_value_half_only.INT-ROW.1" then .text = (.text + " ") else . end)'); WANTS+=('MISMATCH S6 disclosure v24.interval_value_half_only.INT-ROW.1 is not its GATES line text')
IDS+=(F4); EDITS+=('.printed_items.provenance.commit = "98b5054fbf8306e54711b1a7bacd72b5f5e0af2a"'); WANTS+=('MISMATCH S1 provenance pin is not v18 @ A')
IDS+=(F5); EDITS+=('.figure_labels |= map(if (.file | test("fig4_pooled")) then .label = "FINAL" else . end)'); WANTS+=('MISMATCH S8 a Figs 1/6 / Fig 4 label is not PRELIMINARY')
IDS+=(F6); EDITS+=('(.element_tables[] | select(.file | test("fig1_main_figs156")) | .elements[0].status) |= (if . == "PRELIMINARY" then "FINAL" else "PRELIMINARY" end)'); WANTS+=('MISMATCH S8 element table fig1_main_figs156_routecA_source.json: PRELIMINARY elements != the record ci95 entries')
IDS+=(F7); EDITS+=('.release_wording.addendum = (.release_wording.addendum + " ")'); WANTS+=('MISMATCH S10 top-level key release_wording changed')
IDS+=(F8); EDITS+=('.printed_items.all_rows |= map(if .row == 117 then .status = "FINAL (planted)" else . end)'); WANTS+=('MISMATCH S5 interval rows'"'"' class')
IDS+=(F9); EDITS+=('.printed_items.all_rows |= map(if .row == 122 then .status = (if (.status | startswith("FINAL")) then "PRELIMINARY (planted)" else "FINAL (planted)" end) else . end)'); WANTS+=('MISMATCH S12 row 122 FINAL without the recheck, or the reverse')
IDS+=(F11); EDITS+=('.row_rulings_v2_4.row122_recheck.sha256 |= (if . == null then null else (.[0:63] + (if .[63:64] == "0" then "1" else "0" end)) end)'); WANTS+=('MISMATCH S12 the row-122 recheck report or its verifier result does not match its sha, or is not CONFIRMED')
IDS+=(F10); EDITS+=('.disclosures |= map(if .id == "v24.figs_interval_rule.INT-FIG.2" then .id = "v24.figs_interval_rule.INT-FIG.9" else . end)'); WANTS+=('MISMATCH S7 missing disclosure v24.figs_interval_rule.INT-FIG.2')
# b8 negatives (fresh review R1/R2/N1/N3; Codex v24 r1 items 2-5, 9; 09 R4a)
IDS+=(F12); EDITS+=('.repo_path = .supersedes.path'); WANTS+=('MISMATCH S1b repo_path / version')
IDS+=(F13); EDITS+=('.counts_check.other_printed_counts[0].status |= sub("^FINAL"; "PRELIMINARY")'); WANTS+=('MISMATCH S10 counts_check.other_printed_counts (App A FINAL with disclosure; others unchanged)')
IDS+=(F14); EDITS+=('.counts_check.rule += " "'); WANTS+=('MISMATCH S10 counts_check.rule text')
IDS+=(F15); EDITS+=('.disclosures |= map(if .id == "v24.figs_interval_rule.INT-FIG.1" then .cite.line = 1871 else . end)'); WANTS+=('MISMATCH S7 disclosure v24.figs_interval_rule.INT-FIG.1 is not its canonical line / quoted string')
IDS+=(F16); EDITS+=('.row_rulings_v2_4.items |= map(if .id == "v23_interval_corrections" then .disclosure_rows["v24.v23_interval_corrections.INT-CORR.2"][0] = 4 else . end)'); WANTS+=('MISMATCH S7c per-row disclosure attachment')
IDS+=(F17); EDITS+=('.figure_labels |= map(if (.file | test("fig4_pooled")) then .preliminary_elements_n = 0 else . end)'); WANTS+=('MISMATCH S8 figure_labels preliminary_elements_n (60 / 60 / 240)')
IDS+=(F18); EDITS+=('.in_scope_outputs |= map(if .name == "honest_all_routec printed items" then .rows[0] = 93 else . end)'); WANTS+=('MISMATCH S15 a printed-items group'"'"'s placement disagrees with its rows')
IDS+=(F19); EDITS+=('.row_rulings_v2_4.items |= map(if .id == "step_D" then .status += "; rows 119-122 stay PRELIMINARY" else . end)'); WANTS+=('MISMATCH S16 a ruling status other than rows_119_122 claims rows 119-122 PRELIMINARY')
IDS+=(F20); EDITS+=('.disclosures |= map(if .id == "v24.slots_21_22.1" then .source_cite += " " else . end)'); WANTS+=('MISMATCH S7d the power source cite or its numbers')
IDS+=(F21); EDITS+=('.disclosures[0].text += " "'); WANTS+=('MISMATCH S7b v2.3 disclosures not kept')
# b9 negatives (09: NB3, NB6)
IDS+=(F22); EDITS+=('.counts_check.other_printed_counts[0].label |= sub(" \\(superseded: [^)]*\\)$"; " (superseded)")'); WANTS+=('MISMATCH S10 counts_check.other_printed_counts (App A FINAL with disclosure; others unchanged)')
IDS+=(F23); EDITS+=('.row_rulings_v2_4.items |= map(if .id == "rows_75_78" then .disclosure_rows["v24.repin_final17"][0] = 92 else . end)'); WANTS+=('MISMATCH S7c per-row disclosure attachment')
# b9-tooling negatives (Codex v24 r2 items 3, 13, 14, 9). F24 swaps two whole disclosures' text and cite (a swapped pair is several leaves by
# construction; it is the one multi-leaf fixture, and its leaf count is asserted > 1 instead of == 1).
IDS+=(F24); EDITS+=('(.disclosures | map(.id) | index("v24.rows_75_78.1")) as $a | (.disclosures | map(.id) | index("v24.rows_95_98.1")) as $b | .disclosures[$a] as $x | .disclosures[$b] as $y | .disclosures[$a] = ($y | .id = $x.id) | .disclosures[$b] = ($x | .id = $y.id)'); WANTS+=('MISMATCH S7e disclosure v24.rows_75_78.1 is not bound to its pre-stated line / form')
IDS+=(F25); EDITS+=('.row_rulings_v2_4.items |= map(if .id == "rows_75_78" then .disclosure_rows["v24.rows_75_78.1"][0] = 76 else . end)'); WANTS+=('MISMATCH S7c a disclosure_rows list is not the item'"'"'s full rows (or 09'"'"'s narrowing)')
IDS+=(F26); EDITS+=('.in_scope_outputs[0].name += " x"'); WANTS+=('MISMATCH S17 output / group / record names are not the pre-stated set')
IDS+=(F27); EDITS+=('(.element_tables | map(.file | test("fig1_main_R_figs156")) | index(true)) as $i | (.element_tables | map(.file | test("fig1_main_figs156")) | index(true)) as $j | .element_tables[$i].file = .element_tables[$j].file'); WANTS+=('MISMATCH S8 the two Figs 1/6 element tables')
# b9-2 negatives (Codex v24 r3 items 7, 8; predictions frozen in commit/PREDICTIONS_b9_2_negatives_20261021_0258.md). Both change the
# record's shape (a removed list entry and key; an added entry), so their leaf diff is asserted != 1 (the path sets differ) instead of == 1.
IDS+=(F28); EDITS+=('.row_rulings_v2_4.items |= map(if .id == "rows_75_78" then (.disclosures -= ["v24.rows_75_78.1"]) | (.disclosure_rows |= del(.["v24.rows_75_78.1"])) else . end)'); WANTS+=('MISMATCH S7f the item -> disclosure attachments are not the pre-stated table')
IDS+=(F29); EDITS+=('.figure_labels += [{file: "experiments/x/fig_extra_source.json", output: "Fig 9", label: ("FINAL " + ("4" + "096"))}]'); WANTS+=('MISMATCH S18 figure_labels are not the pre-stated set')
leafdiff() { jq -n --slurpfile a $1 --slurpfile b $2 '($a[0] | [paths(type | . != "object" and . != "array")]) as $pa | ($b[0] | [paths(type | . != "object" and . != "array")]) as $pb
  | if ($pa | sort) != ($pb | sort) then -1 else [$pa[] as $p | select(($a[0] | getpath($p)) != ($b[0] | getpath($p)))] | length end'; }
integer ok=0
for i in {1..${#IDS}}; do
  F=$NEG/sc4_${IDS[$i]}.json
  jq "${EDITS[$i]}" $REC > $F
  n=$(leafdiff $REC $F)
  O=$HERE/_runs/sc4neg_${STAMP}_${IDS[$i]}.out
  set +e; zsh -f $HERE/sc_v24.zsh $F > $O 2>&1; rc=$?; set -e
  hit=false; grep -qxF -- "${WANTS[$i]}" $O && hit=true
  print -r -- "NEG ${IDS[$i]} leaf_diff $n exit $rc predicted_mismatch_present $hit mismatch_lines $(grep -c '^MISMATCH ' $O || true)"
  if [[ ${IDS[$i]} == F24 ]]; then nok=$([[ $n -gt 1 ]] && print true || print false)
  elif [[ ${IDS[$i]} == F28 || ${IDS[$i]} == F29 ]]; then nok=$([[ $n != 1 ]] && print true || print false)
  else nok=$([[ $n == 1 ]] && print true || print false); fi
  if [[ $nok == true && $rc == 1 && $hit == true ]]; then (( ++ok )); fi
done
if (( ok == ${#IDS} )); then print -r -- "SC4_NEG_ALL_FAIL_AS_PREDICTED $ok/${#IDS}"; else print -r -- "SC4_NEG_NOT_AS_PREDICTED $ok/${#IDS}"; exit 1; fi
