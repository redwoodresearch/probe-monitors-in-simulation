# SC7: the v2.7 invariants against the committed v2.6 record, re-derived independently of build_v2_7.jq (its own constants from 09's rulings
# and the frozen pre-statement fd03b8c0…; GATES lines and cites come from the driver, which reads them from git by number). Keys, counts and
# booleans only. Output {n: {...}, exempt: {path class: count}, bad: [...]}.
# Inputs: --slurpfile REC draft  --slurpfile V26 the v2.6 record  --slurpfile P18 / P19 provenance rows  --slurpfile LN {k, v}  --slurpfile CI {k, v}
#         --slurpfile LX {k: "commit:file:line", v} (every gates_cite line in the draft)  --argjson PINOK (the driver's pin re-checks)
def W($c; $w): if $c then empty else $w end;
def pfx: if startswith("FINAL") then "FINAL" elif startswith("PRELIMINARY") then "PRELIMINARY" elif startswith("not route C") then "not route C" else "OTHER" end;
def qs($s): [$s | match("\"([^\"]*)\""; "g") | .captures[0].string];
def gc3: {commit, file, line};
def fex($r; $file): [$r.exclusions[].entries[] | select(.file == $file and .status == "FINAL")] | length;
# ---- constants (09 via 2e; pre-statement fd03b8c0…) ----
def D4: "v27.fig4_record_b.L3171.1";
def D5: "v27.fig5_option_a.L3074.1";
def RECP: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.7_2026-10-21.json";
def V26P: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.6_2026-10-21.json";
def OOS: ["DETDROP", "DETDROP_label", "PLANT", "PLANT_label", "LABELS_171"];
def RC1B_REC: "experiments/2026-10-17_routec_slot_readout/results/slots_routeC_writing_rc1b.json@";
def P4: ["stays PRELIMINARY", "stream gate", "UNRESOLVED", "diagnostic only"];
def P5: ["saturated_undefined", "A/B on the saturated", "saturated markers", "saturated undefined"];
def READING: "v2.7 = v2.6 + Fig 4 FINAL (record B INTERVALS REPRODUCED (EXACT); 09, GATES L3160), Fig 5 FINAL (Aswin's option A; 09, GATES L3060) and rc1b FINAL (printed items only) (09, GATES L2969 as amended at L2978); everything else as v2.6";
def MOVED: ["Fig 4", "Fig 5", "rc1b"];
def SSB_IDS: ["v26.fig4_record_b.L3001.1", "saturated_undefined", "v24.figs_interval_rule.INT-FIG.1", "v24.figs_interval_rule.INT-FIG.2", "C3_downstream_judgment"];
def ALLOWED: ["builder", "date_pt", "disclosures", "element_tables", "exclusions", "figure_labels", "flip_groups", "in_scope_outputs", "pending_downstream",
              "preliminary_outputs", "repo_path", "sensitivity_rows", "superseded_fields", "supersedes", "v2_conditions", "version"];
def EXEMPT_RE: "^(superseded_fields|row_rulings_v2_4|row_rulings_v2_5|row_rulings_v2_6|flip_groups|supersedes)(\\.|$)|^exclusions\\.[0-9]+\\.(id|entry|cite\\.anchor)$|^element_tables\\.[0-9]+\\.classes\\.[0-9]+\\.from_exclusion$";   # Q8 (pending 09): + the kept id references
$REC[0] as $r | $V26[0] as $v
| ($LN | map({(.k): .v}) | add) as $L | ($CI | map({(.k): .v}) | add) as $C | ($LX | map({(.k): .v}) | add // {}) as $LXM
| (qs($L.g97_3195)[0]) as $N4 | (qs($L.g97_3200)[0]) as $N5 | (qs($L.g0a_3072)[0]) as $CLS | (qs($L.g50_3232)[0]) as $CAP
| (qs($L.g50_3234)[1]) as $XENT | (qs($L.g50_3235)[0]) as $FR27 | (qs($L.g97_2974)[1]) as $RS | (qs($L.g97_2982)[1]) as $RT
| ("FINAL (" + ([$L.g80_3160 | match("INTERVALS REPRODUCED \\(EXACT\\)"; "g") | .string][0] // "MISSING") + "; GATES L3160 @ " + $C.g80_3160.commit[0:9] + ")") as $F4R
| ($L.g50_3230 | split("\"")) as $p30
| ((($p30[3] // "") | rtrimstr("\\")) + "\"" + (($p30[4] // "") | rtrimstr("\\")) + "\"" + ($p30[5] // "")
   | sub("<N>"; ($v.sensitivity_rows.status | capture("^FINAL for the (?<n>[0-9]+) rc1a slots").n))) as $Q5
| ([$r.figure_labels[] | select(.output == "Fig 4")]) as $a4 | ([$v.figure_labels[] | select(.output == "Fig 4")]) as $b4
| ([$r.figure_labels[] | select(.output == "Fig 5")]) as $a5 | ([$v.figure_labels[] | select(.output == "Fig 5")]) as $b5
| ([$r.in_scope_outputs[] | select(.name == "Fig 4")]) as $o4 | ([$v.preliminary_outputs[] | select(.name == "Fig 4")]) as $p4
| ([$r.in_scope_outputs[] | select(.name == "Fig 5")]) as $o5 | ([$v.preliminary_outputs[] | select(.name == "Fig 5")]) as $p5
| ([$r.in_scope_outputs[] | select(.name == "rc1b")]) as $orc | ([$v.preliminary_outputs[] | select(.name == "rc1b")]) as $prc
| ($r.disclosures | map({(.id): .}) | add) as $RD | ($v.disclosures | map({(.id): .}) | add) as $VD
| ([$v.element_tables | to_entries[] | select(.value.file == ($b5[0].file // "")) | .key][0]) as $t5
| ([$v.element_tables[$t5].elements[] | select(.key_path | tostring | test("/panel/pooled/saturated$")) | .class][0]) as $oldcls
| ([$r | paths(strings) as $p | getpath($p) as $s | ($p | map(tostring) | join(".")) as $ps
    | ([range(1; ($p | length)) as $i | $r | getpath($p[0:$i]) | select(type == "object") | has("status_superseded_by")] | any) as $ssbobj   # an ssb-marked ancestor (Q8, pending 09)
    | ([P4[] | . as $x | select($s | contains($x))] | length > 0) as $h4 | ([P5[] | . as $x | select($s | contains($x))] | length > 0) as $h5
    | (($s | test("Fig 4|fig4")) or ($ps | test("^v2_conditions\\.fig4_elements"))
       or ($p[0] == "figure_labels" and (($r.figure_labels[$p[1]] | .output) == "Fig 4"))
       or (($p[0] == "in_scope_outputs" or $p[0] == "preliminary_outputs") and (($r | getpath($p[0:2]) | .name) == "Fig 4"))
       or ($p[0] == "disclosures" and (($r.disclosures[$p[1]] | .id) | test("^v2[0-9]\\.fig4")))) as $ctx4
    | select(($h4 and $ctx4) or $h5)
    | {ps: $ps, exempt: (($ps | test(EXEMPT_RE)) or $ssbobj)}]) as $PH
| [
  W($PINOK == true and ([$L.g80_3171, $L.g0a_3074, $L.g0a_3072, $L.g97_3195, $L.g97_3200, $L.g50_3232, $L.g50_3235, $L.g50_3241] | all(qs(.) | length == 1))
    and (qs($L.g97_2982) | length) == 2 and (qs($L.g97_2974) | length) == 3 and ($p30 | length) == 9;
    "T0 a pin re-check or a GATES line's quote shape failed"),
  # T1 Fig 4: label (09's asserts + the ruled fields) and the moved output
  W(($a4 | length) == 1 and ($b4 | length) == 1 and $a4[0].label == "FINAL" and $a4[0].values_n == 264 and $a4[0].preliminary_elements_n == 0
    and ($a4[0] | has("reason") | not) and $a4[0].disclosures == [D4] and $a4[0].interval_rule.disclosures == [D4]
    and $a4[0].note == $N4 and $a4[0].interval_rule.rule == $F4R and $F4R == (qs($L.g50_3241)[0])
    and $a4[0].final_elements_n_in_exclusions == fex($r; $a4[0].file) and $a4[0].final_elements_n_in_exclusions == 264
    and $a4[0].interval_rule.cites == ($b4[0].interval_rule.cites + [$C.g80_3160])
    and ($a4[0] | keys) == (["file", "output", "label", "values_n", "preliminary_elements_n", "final_elements_n_in_exclusions", "rule", "interval_rule", "note", "disclosures"] | sort)
    and ($a4[0] | del(.label, .note, .preliminary_elements_n, .final_elements_n_in_exclusions, .disclosures, .interval_rule.disclosures, .interval_rule.rule, .interval_rule.cites))
        == ($b4[0] | del(.label, .reason, .preliminary_elements_n, .final_elements_n_in_exclusions, .interval_rule.disclosures, .interval_rule.rule, .interval_rule.cites));
    "T1 the Fig 4 label is not FINAL with 09's fields (264 / 0, no reason, the L3171 disclosure, the L3195 note, the L3160 rule and cite)"),
  W(($o4 | length) == 1 and ($p4 | length) == 1 and $o4[0].status == "FINAL" and $o4[0].ruling == "fig4_record_b_exact (v2.7: record B, GATES L3160)"
    and $o4[0].moved_from == "preliminary_outputs (v2.6)" and $o4[0].interval_rule.rule == $F4R and $o4[0].interval_rule.disclosures == [D4]
    and $o4[0].interval_rule.cites == ($p4[0].interval_rule.cites + [$C.g80_3160])
    and ($o4[0] | del(.status, .ruling, .moved_from, .interval_rule.rule, .interval_rule.disclosures, .interval_rule.cites))
        == ($p4[0] | del(.status, .moved_from, .interval_rule.rule, .interval_rule.disclosures, .interval_rule.cites));
    "T1 the Fig 4 output is not v2.6's moved to in_scope with the ruled fields"),
  # T2 Fig 5: label, table, exclusion, output
  W(($a5 | length) == 1 and ($b5 | length) == 1 and $a5[0].label == "FINAL" and $a5[0].values_n == 74 and $a5[0].preliminary_elements_n == 0
    and $a5[0].caption_only_elements_n == 6 and ($a5[0] | has("reason") | not) and ($a5[0] | has("interval_rule") | not)
    and $a5[0].note == $N5 and $a5[0].disclosures == [D5] and $a5[0].final_elements_n_in_exclusions == fex($r; $a5[0].file)
    and ($a5[0] | del(.label, .note, .preliminary_elements_n, .caption_only_elements_n, .final_elements_n_in_exclusions, .disclosures))
        == ($b5[0] | del(.label, .preliminary_elements_n, .final_elements_n_in_exclusions));
    "T2 the Fig 5 label is not FINAL with 09's fields (0 preliminary, 6 caption-only, the L3200 note, the L3074 disclosure)"),
  W(([range(0; $v.element_tables[$t5].elements | length) as $j | $v.element_tables[$t5].elements[$j] as $e | $r.element_tables[$t5].elements[$j] as $f
       | if ($e.key_path | tostring | test("/panel/pooled/saturated$")) then $f == ($e | .status = $CAP | .class = $CLS) else $f == $e end] | all)
    and ([$r.element_tables[$t5].elements[] | select(.status == "FINAL")] | length) == 68
    and ([$r.element_tables[$t5].elements[] | select(.status == $CAP)] | length) == 6 and ($CAP | pfx) == "OTHER"
    and $r.element_tables[$t5].figure_status == "FINAL"
    and ($r.element_tables[$t5].classes == ($v.element_tables[$t5].classes | map(if .class == $oldcls then .class = $CLS else . end)))
    and ($r.element_tables[$t5] | del(.elements, .classes, .figure_status)) == ($v.element_tables[$t5] | del(.elements, .classes, .figure_status));
    "T2 the Fig 5 table is not v2.6's with the 6 saturated elements caption-only (68 FINAL + 6)"),
  ([$r.exclusions[] | select(.id == "C3_saturated_undefined")] as $x | [$v.exclusions[] | select(.id == "C3_saturated_undefined")] as $y
   | W(($x | length) == 1 and ($y | length) == 1 and ($x[0].entries == ($y[0].entries | map(.status = $CAP | .entry = $XENT)))
       and $x[0].flipped_n == 6 and $x[0].flip_rule_v27 == $FR27 and $x[0].entry == $y[0].entry and $x[0].id == $y[0].id
       and ($x[0] | del(.entries, .flipped_n, .flip_rule_v27)) == ($y[0] | del(.entries, .flipped_n));
       "T2 exclusion C3_saturated_undefined is not flipped in place as ruled")),
  W(($o5 | length) == 1 and ($p5 | length) == 1 and $o5[0].status == "FINAL" and ($o5[0] | has("reason") | not)
    and $o5[0].ruling == "fig5_option_a (v2.7: Aswin's option A, GATES L3060)" and $o5[0].moved_from == "preliminary_outputs (v2.6)"
    and ($o5[0] | del(.status, .ruling, .moved_from)) == ($p5[0] | del(.status, .reason));
    "T2 the Fig 5 output is not v2.6's moved to in_scope with the ruled fields"),
  # T3 rc1b
  W(($orc | length) == 1 and ($prc | length) == 1 and $orc[0].status == $RS and $RS == "FINAL (printed items only)" and $orc[0].ruling_text == $RT
    and $orc[0].ruling == "rc1b_printed_items_only (v2.7: GATES L2969 as amended at L2978)" and $orc[0].ruling_cite == $C.g97_2982
    and $orc[0].out_of_scope_keys == OOS and $orc[0].moved_from == "preliminary_outputs (v2.6)"
    and ($orc[0] | del(.status, .ruling, .ruling_text, .ruling_cite, .out_of_scope_keys, .moved_from)) == ($prc[0] | del(.status));
    "T3 rc1b is not moved with the ruled status / ruling_text (L2982's 2nd quoted string) / cite / out-of-scope keys"),
  # T4 the 2 new disclosures: one object each; text == the GATES quoted string; cite == all 5 fields
  W(([$r.disclosures[] | select(.id == D4)] | length) == 1 and $RD[D4].text == (qs($L.g80_3171)[0]) and $RD[D4].cite == $C.g80_3171;
    "T4 the Fig 4 disclosure is not GATES L3171's quoted string with the full cite"),
  W(([$r.disclosures[] | select(.id == D5)] | length) == 1 and $RD[D5].text == (qs($L.g0a_3074)[0]) and $RD[D5].cite == $C.g0a_3074;
    "T4 the Fig 5 disclosure is not GATES L3074's quoted string with the full cite"),
  # T5 ssb targets; resolve; lines exist; every other disclosure byte-identical (no id skip)
  W($RD["v26.fig4_record_b.L3001.1"].status_superseded_by == {disclosure_id: D4, gates_cite: ($C.g80_3160 | gc3)}
    and $RD["saturated_undefined"].status_superseded_by == {disclosure_id: D5, gates_cite: ($C.g0a_3060 | gc3)}
    and $RD["v24.figs_interval_rule.INT-FIG.1"].status_superseded_by == ($VD["v24.figs_interval_rule.INT-FIG.1"].status_superseded_by + [{disclosure_id: D4, gates_cite: ($C.g80_3160 | gc3)}])
    and $RD["v24.figs_interval_rule.INT-FIG.2"].status_superseded_by == ($VD["v24.figs_interval_rule.INT-FIG.2"].status_superseded_by + [{disclosure_id: D4, gates_cite: ($C.g80_3160 | gc3)}])
    and $RD["C3_downstream_judgment"].status_superseded_by == [$VD["C3_downstream_judgment"].status_superseded_by, {disclosure_id: D5, gates_cite: ($C.g0a_3074 | gc3)}];
    "T5 a status_superseded_by is not the ruled target"),
  W(all($r.disclosures[] | (.status_superseded_by // empty) | if type == "array" then .[] else . end;
        (.disclosure_id == null or $RD[.disclosure_id] != null) and ($LXM["\(.gates_cite.commit):\(.gates_cite.file):\(.gates_cite.line)"] != null)
        and (.gates_cite.commit | test("^[0-9a-f]{40}$")));
    "T5 a status_superseded_by target does not resolve, or its GATES line does not exist"),
  W(([$r.disclosures[] | select(.id != D4 and .id != D5) | del(.status_superseded_by)]) == ([$v.disclosures[] | del(.status_superseded_by)])
    and ([$r.disclosures[] | select(.id != D4 and .id != D5 and (.id | IN(SSB_IDS[]) | not))] == [$v.disclosures[] | select(.id | IN(SSB_IDS[]) | not)])
    and ($r.disclosures | length) == 66;
    "T5 a v2.6 disclosure changed beyond the ruled status_superseded_by keys, or the count is not 66"),
  # T6 the 0-printed-dependence re-run (v18 and v19), the element-table strings, the 5 keys
  W(([$P18[], $P19[] | select((.record | startswith(RC1B_REC)) and ((.key_path | type) == "array") and ((.key_path[1] // "") | tostring | test("^(23|25)\\.")))] | length) == 0
    and ([$P18[], $P19[] | select(.record | startswith(RC1B_REC))] | length) >= 1
    and ([$r.element_tables[] | .. | strings | select(test("detector_vs_drop|panel_plant_rows|slots_routeC_writing_rc1b"))] | length) == 0
    and (($orc[0] // {}).out_of_scope_keys // []) == OOS;
    "T6 a printed row reads rc1b slots 23.* / 25.*, an element-table string references an out-of-scope input, or the 5 keys differ"),
  # T7 the sweep (09 Q2): (i) status fields both ways, incl. element .label / class status / PRELIMINARY wording in FINAL figure statements
  ([ ($r.printed_items.all_rows[] | {s: (.status | pfx), t: [.status]}),
     (($r.row_rulings_v2_7.items // [])[] | {s: (.status | pfx), t: [.status]}),
     ($r.in_scope_outputs[] | {s: (.status | pfx), t: [.status, .reason, .pending_downstream_label, (.downstream_tag | objects | .text), (.interval_rule | objects | .rule)]}),
     ($r.preliminary_outputs[] | {s: (.status | pfx), t: [.status, .reason, .pending_downstream_label, (.downstream_tag | objects | .text), (.interval_rule | objects | .rule)]}),
     ($r.figure_labels[] | {s: .label, t: [.note, .reason, (.interval_rule | objects | .rule)]}),
     ($r.element_tables[] | {s: .figure_status, t: [(.interval_rule | objects | .rule)]}),
     ($r.element_tables[] | .elements[] | {s: (.status | pfx), t: [.interval_note, .reason, .label]}),
     ($r.element_tables[] | .classes | if type == "array" then (.[] | select(.status != null) | {s: (.status | pfx), t: [.class]}) else empty end) ]
   | map(.t |= map(strings))) as $SW
| W(all($SW[] | select(.s == "FINAL"); all(.t[]; startswith("PRELIMINARY") | not)); "T7 a FINAL row / item / output / label / table / element / class carries PRELIMINARY-prefixed text"),
  W(all($SW[] | select(.s == "PRELIMINARY"); all(.t[]; startswith("FINAL") | not)); "T7 a PRELIMINARY row / item / output / label / table / element / class carries FINAL-prefixed text"),
  W(all($r.figure_labels[] | select(.label == "FINAL"); (has("reason") | not) and ([.note, (.interval_rule | objects | .rule)] | map(strings) | all(contains("PRELIMINARY") | not)));
    "T7 a FINAL figure label has a reason or PRELIMINARY wording in its note / rule"),
  W(all($r.in_scope_outputs[] | select(.name | IN("Fig 4", "Fig 5", "rc1b", "Figs 1/6", "Figs 7/8")); [(.downstream_tag | objects | .text), (.interval_rule | objects | .rule)] | map(strings) | all(contains("PRELIMINARY") | not));
    "T7 a FINAL figure output's tag or rule carries PRELIMINARY wording"),
  W(all(($r.in_scope_outputs + $r.preliminary_outputs)[] | select(has("downstream_tag")) | .downstream_tag; . == null or ((type == "object") and ((.text | type) == "string")));
    "T7 a downstream_tag is not null or an object with a string text"),
  # (ii) the old phrases: Fig 4's in Fig 4 contexts and on co-occurrence; Fig 5's everywhere current; exempt paths counted
  W(all($PH[]; .exempt); "T7 an old Fig 4 / Fig 5 phrase sits in a current, non-exempt string"),
  W([($a4[0].note // ""), ($a5[0].note // ""), ($a4[0].interval_rule.rule // "")] | all(. as $n | [P4[], P5[]] | all(. as $x | $n | contains($x) | not));
    "T7 a Fig 4 / Fig 5 note or rule carries an old phrase"),
  # (iii) the disclosure-text sweep (v2.6 S21) with the v2.7 attachments
  ([ ($r.in_scope_outputs[] | {s: (.status | pfx), d: ((.interval_rule | objects | .disclosures) // [])}),
     ($r.preliminary_outputs[] | {s: (.status | pfx), d: ((.interval_rule | objects | .disclosures) // [])}),
     ($r.element_tables[] | {s: .figure_status, d: ((.interval_rule | objects | .disclosures) // [])}),
     ($r.figure_labels[] | {s: .label, d: (((.interval_rule | objects | .disclosures) // []) + (.disclosures // []))}),
     (($r.row_rulings_v2_5.items // [])[], ($r.row_rulings_v2_6.items // [])[], ($r.row_rulings_v2_7.items // [])[] | {s: (.status | pfx), d: (.disclosures // [])}) ]) as $DL
| W(all($DL[] | select(.s == "FINAL") | .d[]; ($RD[.].text // "") | startswith("PRELIMINARY") | not); "T7 a FINAL item lists a disclosure whose text starts with PRELIMINARY"),
  W(all($DL[] | select(.s == "PRELIMINARY") | .d[]; ($RD[.].text // "") | startswith("FINAL") | not); "T7 a PRELIMINARY item lists a disclosure whose text starts with FINAL"),
  W(all($DL[] | .d[]; $RD[.] != null) and ([$DL[] | select(.s == "FINAL") | .d[] | select(. == "v26.fig4_record_b.L3001.1" or . == "saturated_undefined")] | length) == 0;
    "T7 a listed disclosure id does not resolve, or a FINAL item lists a superseded one"),
  # T9 (09 L3243 condition 1): no current .disclosures list (outside superseded_fields and objects carrying status_superseded_by) names an id
  # whose disclosure object carries status_superseded_by
  ([$r.disclosures[] | select(has("status_superseded_by")) | .id]) as $SSBD
  | ([$r | paths(arrays) as $p | select($p[-1] == "disclosures" and ($p[0] != "superseded_fields"))
      | select(([range(1; ($p | length)) as $i | $r | getpath($p[0:$i]) | select(type == "object") | has("status_superseded_by")] | any) | not)
      | getpath($p)[] | strings | . as $id | select($SSBD | index([$id]) != null)] | length) as $t9
  | W($t9 == 0; "T9 a current .disclosures list names a superseded disclosure"),
  # T8 unchanged elsewhere; counts; identity; bookkeeping
  ((($v | keys) - ALLOWED)[] as $k | W($r[$k] == $v[$k]; "T8 top-level key \($k) changed")),
  W((($r | keys) - ($v | keys)) == ["row_rulings_v2_7"]; "T8 new top-level keys other than row_rulings_v2_7"),
  W(([$r.in_scope_outputs[] | select(.name | IN(MOVED[]) | not)]) == $v.in_scope_outputs and ([$r.in_scope_outputs[-3:][].name] == MOVED)
    and [$r.preliminary_outputs[].name] == ["routec_simpool_numbers printed items"]
    and ($r.preliminary_outputs == [$v.preliminary_outputs[] | select(.name | IN(MOVED[]) | not)]);
    "T8 the outputs are not v2.6's with Fig 4 / Fig 5 / rc1b moved (preliminary_outputs == [routec_simpool_numbers printed items])"),
  W(([$r.element_tables[] | select(.file != ($b5[0].file // ""))]) == ([$v.element_tables[] | select(.file != ($b5[0].file // ""))])
    and ([$r.figure_labels[] | select(.output != "Fig 4" and .output != "Fig 5")]) == ([$v.figure_labels[] | select(.output != "Fig 4" and .output != "Fig 5")])
    and [$r.figure_labels[].output] == [$v.figure_labels[].output]
    and ([$r.exclusions[] | select(.id != "C3_saturated_undefined")]) == ([$v.exclusions[] | select(.id != "C3_saturated_undefined")]);
    "T8 another element table, label or exclusion changed"),
  W($r.printed_items == $v.printed_items and ($r.printed_items.all_rows | map(.status | pfx) | group_by(.) | map({(.[0]): length}) | add) == {"FINAL": 237, "PRELIMINARY": 2, "not route C": 12};
    "T8 printed_items changed, or the counts are not 237 / 2 / 12"),
  W($r.sensitivity_rows.status == $Q5 and ($Q5 | startswith("FINAL for the ")) and ($Q5 | contains("<N>") | not) and ($r.sensitivity_rows | del(.status)) == ($v.sensitivity_rows | del(.status));
    "T8 sensitivity_rows.status is not the Q5 text (L3230, <N> from v2.6)"),
  W($r.pending_downstream == ($v.pending_downstream | .waits |= del(.["Fig 5 as a figure"])) and ($v.pending_downstream.waits | has("Fig 5 as a figure"));
    "T8 pending_downstream is not v2.6's minus the Fig 5 wait"),
  W($r.v2_conditions == ($v.v2_conditions | .fig4_elements.label = "FINAL" | .fig4_elements.status_superseded_by = {disclosure_id: D4, gates_cite: ($C.g80_3160 | gc3)});
    "T8 v2_conditions is not v2.6's with fig4_elements FINAL and its ssb"),
  W($r.row_rulings_v2_4 == $v.row_rulings_v2_4 and $r.row_rulings_v2_5 == $v.row_rulings_v2_5 and $r.row_rulings_v2_6 == $v.row_rulings_v2_6
    and ($r.flip_groups | del(.group_6)) == $v.flip_groups and $r.flip_groups.group_6.evidence == ["fig4_record_b_exact", "fig5_option_a", "rc1b_printed_items_only"];
    "T8 dated history changed, or flip group 6 is not the v2.7 one"),
  W($r.row_rulings_v2_7.reading == READING and ([$r.row_rulings_v2_7.items[].id] == ["fig4_record_b_exact", "fig5_option_a", "rc1b_printed_items_only"])
    and ([$r.row_rulings_v2_7.items[] | {status, cites: (.cites | map(gc3))}] == [{status: $F4R, cites: [$C.g80_3160 | gc3]}, {status: $N5, cites: [$C.g0a_3060 | gc3]}, {status: $RS, cites: [$C.g97_2982 | gc3]}]);
    "T8 row_rulings_v2_7 is not the pre-stated block"),
  W($r.version == "2.7" and $r.repo_path == RECP and ($r.supersedes | {path, commit, sha256}) == {path: V26P, commit: $C.base.commit, sha256: $C.base.sha256}
    and $r.supersedes.version == "2.6" and $r.superseded_fields.v2_6_superseded_fields == $v.superseded_fields
    and ($r.superseded_fields.prior_values | keys | length) == 10;
    "T8 version / repo_path / supersedes / superseded_fields")
  ] as $bad
| {n: {rows: ($r.printed_items.all_rows | length), FINAL: ([$r.printed_items.all_rows[] | select(.status | startswith("FINAL"))] | length),
       PRELIMINARY: ([$r.printed_items.all_rows[] | select(.status | startswith("PRELIMINARY"))] | length),
       not_route_c: ([$r.printed_items.all_rows[] | select(.status | startswith("not route C"))] | length), disclosures: ($r.disclosures | length),
       preliminary_outputs: ($r.preliminary_outputs | length), fig5_caption_only: ([$r.element_tables[] | .elements[] | select(.status == $CAP)] | length),
       old_phrase_hits: ($PH | length), old_phrase_exempt: ([$PH[] | select(.exempt)] | length), pin_ok: $PINOK},
   exempt: ([$PH[] | select(.exempt) | .ps | split(".") | map(if test("^[0-9]+$") then "#" else . end) | .[0:3] | join(".")] | group_by(.) | map({(.[0]): length}) | add // {}),
   bad: $bad}
