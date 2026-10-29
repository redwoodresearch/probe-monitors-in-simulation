# build_v2_7.jq: route C FINAL-release record v2.7 = v2.6 (0db7dc714) + the v2.7 rulings, by key. Never prints; the wrapper writes the output.
# 09 (via 2e): record B EXACT (GATES L3160 @ f80f3cfe6) -> Fig 4 FINAL; Aswin's option A (L3060 @ 0a719afaa) -> Fig 5 FINAL, the 6 saturated flags
# caption-only, not plotted; rc1b -> "FINAL (printed items only)" (L2969 as amended at L2978 @ 97b6e634d); 09's Q1-Q7 rulings (L3212-3241 @
# 5046d443f). Every text comes from a GATES line read by the wrapper. Pre-statement commit/PRESTATEMENT_v27.md (fd03b8c0…).
# Inputs: --slurpfile V26 the v2.6 record  --slurpfile LN {k, v: GATES line text}  --slurpfile CI {k, v: cite}  --argjson META {date_pt, repo_path, v26, builder}
def qs($s): [$s | match("\"([^\"]*)\""; "g") | .captures[0].string];
def one($a; $n; $i; $e): if ($a | length) == $n then $a[$i] else error($e) end;
def gc($c): $c | {commit, file, line};
def tgt($id; $cite): {disclosure_id: $id, gates_cite: gc($cite)};
def fex($r; $file): [$r.exclusions[].entries[] | select(.file == $file and .status == "FINAL")] | length;
$V26[0] as $v
| ($LN | map({(.k): .v}) | add) as $L | ($CI | map({(.k): .v}) | add) as $C
| one(qs($L.g80_3171); 1; 0; "L3171") as $T4 | one(qs($L.g0a_3074); 1; 0; "L3074") as $T5 | one(qs($L.g0a_3072); 1; 0; "L3072") as $CLS
| one(qs($L.g97_2974); 3; 1; "L2974") as $RS | one(qs($L.g97_2982); 2; 1; "L2982") as $RT
| one(qs($L.g97_3195); 1; 0; "L3195") as $N4 | one(qs($L.g97_3200); 1; 0; "L3200") as $N5
| one(qs($L.g50_3232); 1; 0; "L3232") as $CAP | one(qs($L.g50_3234); 2; 1; "L3234") as $XENT | one(qs($L.g50_3235); 1; 0; "L3235") as $FR27
| one(qs($L.g50_3241); 1; 0; "L3241") as $F4R09
| ($L.g50_3230 | split("\"") | if length == 9 and (.[3] | endswith("\\")) and (.[4] | endswith("\\"))
     then ((.[3] | rtrimstr("\\")) + "\"" + (.[4] | rtrimstr("\\")) + "\"" + .[5]) else error("L3230") end) as $Q5RAW
| ($v.sensitivity_rows.status | capture("^FINAL for the (?<n>[0-9]+) rc1a slots via sensitivity_halves").n) as $NS
| (if ($Q5RAW | startswith("FINAL for the <N> rc1a slots")) then ($Q5RAW | sub("<N>"; $NS)) else error("Q5 text") end) as $Q5
| one([$L.g80_3160 | match("INTERVALS REPRODUCED \\(EXACT\\)"; "g") | .string]; 1; 0; "L3160 phrase") as $PA
| ("FINAL (" + $PA + "; GATES L" + ($C.g80_3160.line | tostring) + " @ " + $C.g80_3160.commit[0:9] + ")") as $F4R
| (if $F4R != $F4R09 then error("the composed Fig 4 rule is not 09's L3241 string") else . end)
| "v27.fig4_record_b.L3171.1" as $D4 | "v27.fig5_option_a.L3074.1" as $D5
# shape guards
| ([$v.figure_labels[] | select(.output == "Fig 4")] | one(.; 1; 0; "the Fig 4 label")) as $l4
| ([$v.figure_labels[] | select(.output == "Fig 5")] | one(.; 1; 0; "the Fig 5 label")) as $l5
| ([$v.element_tables | to_entries[] | select(.value.file == $l5.file) | .key] | one(.; 1; 0; "the Fig 5 table")) as $t5i
| ([$v.element_tables[$t5i].elements[] | select(.key_path | tostring | test("/panel/pooled/saturated$"))]) as $sat
| (if ($sat | length) != 6 or ([$sat[] | {status, class}] | unique | length) != 1 then error("the 6 saturated elements") else . end)
| ([$v.element_tables[$t5i].classes[] | select(.class == $sat[0].class)] | one(.; 1; 0; "the saturated class")) as $scls
| ([$v.exclusions[] | select(.id == "C3_saturated_undefined")] | one(.; 1; 0; "the saturated exclusion")) as $xs
| (if ($xs.entries | length) != 6 or ([$xs.entries[] | {status, entry}] | unique | length) != 1 then error("the 6 exclusion entries") else . end)
| ([$v.preliminary_outputs[] | select(.name == "Fig 4")] | one(.; 1; 0; "Fig 4 output")) as $o4
| ([$v.preliminary_outputs[] | select(.name == "Fig 5")] | one(.; 1; 0; "Fig 5 output")) as $o5
| ([$v.preliminary_outputs[] | select(.name == "rc1b")] | one(.; 1; 0; "rc1b output")) as $orc
| ($v.disclosures | map({(.id): .}) | add) as $DI
| (if ($DI["v26.fig4_record_b.L3001.1"] == null) or ($DI["saturated_undefined"] == null) or ($DI["C3_downstream_judgment"].status_superseded_by | type) != "object"
      or ([$DI["v24.figs_interval_rule.INT-FIG.1"], $DI["v24.figs_interval_rule.INT-FIG.2"]] | map(.status_superseded_by | type == "array" and length == 2) | all | not)
   then error("the ssb targets") else . end)
| {"figure_labels[Fig 4]": ($l4 | {label, reason, preliminary_elements_n, final_elements_n_in_exclusions, interval_rule_rule: .interval_rule.rule, interval_rule_disclosures: .interval_rule.disclosures}),
   "preliminary_outputs[Fig 4]": ($o4 | {status, moved_from, interval_rule_rule: .interval_rule.rule, interval_rule_disclosures: .interval_rule.disclosures}),
   "figure_labels[Fig 5]": ($l5 | {label, preliminary_elements_n, final_elements_n_in_exclusions}),
   "preliminary_outputs[Fig 5]": ($o5 | {status, reason}), "preliminary_outputs[rc1b]": ($orc | {status}),
   "element_tables[Fig 5]": {figure_status: $v.element_tables[$t5i].figure_status, saturated_elements_x6: ($sat[0] | {status, class}), saturated_class: $scls.class},
   "exclusions[C3_saturated_undefined]": {entries_x6: ($xs.entries[0] | {status, entry}), flipped_n: $xs.flipped_n},
   "sensitivity_rows.status": $v.sensitivity_rows.status, "pending_downstream.waits[Fig 5 as a figure]": $v.pending_downstream.waits["Fig 5 as a figure"],
   "v2_conditions.fig4_elements.label": $v.v2_conditions.fig4_elements.label} as $PRIOR
# ---- the record ----
| $v
| .version = "2.7" | .repo_path = $META.repo_path | .date_pt = $META.date_pt
| .supersedes = ($META.v26 + {kind: $v.supersedes.kind, version: "2.6"})
| .builder = $META.builder
# 2. Fig 5 table and exclusion (the exclusion first, so the label counts below are derived from the flipped state)
| .exclusions |= map(if .id == "C3_saturated_undefined" then (.entries |= map(.status = $CAP | .entry = $XENT) | .flipped_n = 6 | .flip_rule_v27 = $FR27) else . end)
| .element_tables[$t5i] |= (.elements |= map(if (.key_path | tostring | test("/panel/pooled/saturated$")) then (.status = $CAP | .class = $CLS) else . end)
                            | .classes |= map(if .class == $scls.class then .class = $CLS else . end)
                            | .figure_status = (if any(.elements[]; .status == "PRELIMINARY") then "PRELIMINARY" else "FINAL" end))
| . as $rr
# 1-2. the labels
| .figure_labels |= map(
    if .output == "Fig 4" then (.label = "FINAL" | del(.reason) | .note = $N4 | .preliminary_elements_n = 0 | .final_elements_n_in_exclusions = fex($rr; .file)
                                | .disclosures = [$D4] | .interval_rule.disclosures = [$D4] | .interval_rule.rule = $F4R | .interval_rule.cites += [$C.g80_3160])
    elif .output == "Fig 5" then (.label = "FINAL" | .note = $N5 | .preliminary_elements_n = 0 | .caption_only_elements_n = 6
                                  | .final_elements_n_in_exclusions = fex($rr; .file) | .disclosures = [$D5])
    else . end)
| (if ([.figure_labels[] | select(.output == "Fig 4") | .final_elements_n_in_exclusions] != [264]) or ([.figure_labels[] | select(.output == "Fig 5") | .final_elements_n_in_exclusions] != [36])
   then error("final_elements_n_in_exclusions") else . end)
# 1-3. the outputs
| .preliminary_outputs |= map(select(.name != "Fig 4" and .name != "Fig 5" and .name != "rc1b"))
| .in_scope_outputs += [
    ($o4 | .status = "FINAL" | .ruling = "fig4_record_b_exact (v2.7: record B, GATES L3160)" | .moved_from = "preliminary_outputs (v2.6)"
         | .interval_rule.rule = $F4R | .interval_rule.disclosures = [$D4] | .interval_rule.cites += [$C.g80_3160]),
    ($o5 | .status = "FINAL" | del(.reason) | .ruling = "fig5_option_a (v2.7: Aswin's option A, GATES L3060)" | .moved_from = "preliminary_outputs (v2.6)"),
    ($orc | .status = $RS | .ruling = "rc1b_printed_items_only (v2.7: GATES L2969 as amended at L2978)" | .ruling_text = $RT | .ruling_cite = $C.g97_2982
          | .out_of_scope_keys = ["DETDROP", "DETDROP_label", "PLANT", "PLANT_label", "LABELS_171"] | .moved_from = "preliminary_outputs (v2.6)")]
# 1-2. disclosures (supersede in place, keys only) + the 2 new ones
| .disclosures |= map(
    if .id == "v26.fig4_record_b.L3001.1" then . + {status_superseded_by: tgt($D4; $C.g80_3160)}
    elif .id == "saturated_undefined" then . + {status_superseded_by: tgt($D5; $C.g0a_3060)}
    elif .id == "v24.figs_interval_rule.INT-FIG.1" or .id == "v24.figs_interval_rule.INT-FIG.2" then .status_superseded_by += [tgt($D4; $C.g80_3160)]
    elif .id == "C3_downstream_judgment" then .status_superseded_by = [.status_superseded_by, tgt($D5; $C.g0a_3074)]
    else . end)
| .disclosures += [{id: $D4, text: $T4, cite: $C.g80_3171}, {id: $D5, text: $T5, cite: $C.g0a_3074}]
# conditions, sensitivity, pending
| .v2_conditions.fig4_elements.label = "FINAL" | .v2_conditions.fig4_elements.status_superseded_by = tgt($D4; $C.g80_3160)
| .sensitivity_rows.status = $Q5
| .pending_downstream.waits |= del(.["Fig 5 as a figure"])
# 5. bookkeeping
| .row_rulings_v2_7 = {reading: "v2.7 = v2.6 + Fig 4 FINAL (record B INTERVALS REPRODUCED (EXACT); 09, GATES L3160), Fig 5 FINAL (Aswin's option A; 09, GATES L3060) and rc1b FINAL (printed items only) (09, GATES L2969 as amended at L2978); everything else as v2.6",
    items: [{id: "fig4_record_b_exact", rows: [], outputs: ["Fig 4"], status: $F4R, cites: [$C.g80_3160], disclosures: [$D4], disclosure_rows: {}},
            {id: "fig5_option_a", rows: [], outputs: ["Fig 5"], status: $N5, cites: [$C.g0a_3060], disclosures: [$D5], disclosure_rows: {}},
            {id: "rc1b_printed_items_only", rows: [], outputs: ["rc1b"], status: $RS, cites: [$C.g97_2982], disclosures: [], disclosure_rows: {}}]}
| .flip_groups.group_6 = {what: "v2.7 rulings: Fig 4 FINAL (record B, GATES L3160), Fig 5 FINAL (Aswin's option A, GATES L3060), rc1b FINAL (printed items only) (GATES L2969 / L2978)",
                          evidence: ["fig4_record_b_exact", "fig5_option_a", "rc1b_printed_items_only"]}
| .superseded_fields = {scope: "v2.7 = v2.6 + the v2.7 rulings, by key; everything not listed is byte-identical to v2.6",
     v2_6: ($META.v26 + {kind: $v.supersedes.kind}),
     fields: ["version", "repo_path", "date_pt", "supersedes", "builder",
              "figure_labels[Fig 4]: label, -reason, +note, preliminary_elements_n, +disclosures, interval_rule.rule, interval_rule.disclosures, interval_rule.cites[+L3160]",
              "figure_labels[Fig 5]: label, +note, preliminary_elements_n, +caption_only_elements_n, +disclosures",
              "element_tables[Fig 5]: the 6 saturated elements' status / class, the saturated class text, figure_status",
              "exclusions[C3_saturated_undefined]: entries' status / entry, flipped_n, +flip_rule_v27",
              "preliminary_outputs[-Fig 4, -Fig 5, -rc1b]",
              "in_scope_outputs[+Fig 4: status, ruling, moved_from, interval_rule.rule / .disclosures / .cites]",
              "in_scope_outputs[+Fig 5: status, -reason, ruling, moved_from]",
              "in_scope_outputs[+rc1b: status, ruling, ruling_text, ruling_cite, out_of_scope_keys, moved_from]",
              "disclosures[+v27.fig4_record_b.L3171.1, +v27.fig5_option_a.L3074.1]",
              "disclosures[status_superseded_by: v26.fig4_record_b.L3001.1, saturated_undefined (new); INT-FIG.1 / .2, C3_downstream_judgment (extended)]",
              "v2_conditions.fig4_elements: label, +status_superseded_by", "sensitivity_rows.status", "pending_downstream.waits[-Fig 5 as a figure]",
              "row_rulings_v2_7", "flip_groups.group_6"],
     prior_values: $PRIOR,
     v2_6_superseded_fields: $v.superseded_fields}
