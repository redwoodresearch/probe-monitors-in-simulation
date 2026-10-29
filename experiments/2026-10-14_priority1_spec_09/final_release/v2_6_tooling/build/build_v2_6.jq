# build_v2_6.jq: route C FINAL-release record v2.6 = v2.5 (4df86c4b9) + the v2.6 rulings, by key. Never prints; the wrapper writes the output.
# 09 (via 2e): option (3) RULING A (GATES L2869 / L2876 @ 4d43ba0a4): the 65 printed interval rows and the Figs 1/6 interval elements FINAL;
# RULING B (L2877): Fig 4 stays PRELIMINARY with 09's verbatim reason; the L2878 disclosure (copied by script); status_superseded_by (keys
# only) on the ruled disclosures and v2.4 items; r_node_switch.superseded_by_ruling. Pre-statement commit/PRESTATEMENT_v26.md (33a40662…).
# Inputs: --slurpfile V25 the v2.5 record  --slurpfile SCOPE the scope file (jsonl)  --slurpfile STR v26_strings.json
#         --slurpfile LN {k, v: GATES line text}  --slurpfile CI {k, v: cite}  (k in ruling, ruling_a, ruling_b, disclosure, figs78_verified)
#         --argjson META {date_pt, repo_path, v25, results, recheck_b, recheck_a, builder}   (c2: LN / CI also hold fig4_disclosure = GATES L3001 @ 89435ec9a)
def isin($n; $rows): ([$rows[] | select(. == $n)] | length) > 0;
def gc($c): $c | {commit, file, line};
def tgt($id; $cite): {disclosure_id: $id, gates_cite: gc($cite)};
def counts: group_by(.status) | map({(.[0].status): length}) | add;
$V25[0] as $v | $STR[0] as $S
| ($LN | map({(.k): .v}) | add) as $L | ($CI | map({(.k): .v}) | add) as $C
| ([$SCOPE[].row] | unique) as $R65
| (if ($R65 | length) != 65 then error("scope rows") else . end)
| ([$v.printed_items.all_rows[] | select((.status | startswith("PRELIMINARY (v2.4 correction")) or (.status | startswith("PRELIMINARY (value half only for the interval"))) | .row] | unique) as $P65
| (if $P65 != $R65 then error("the scope rows are not v2.5's interval-rule rows") else . end)
| ([$L.ruling_a | match("INTERVALS REPRODUCED \\(EXACT\\)"; "g") | .string]) as $PA
| (if ($PA | length) != 1 then error("RULING A phrase") else . end)
| ("FINAL (" + $PA[0] + "; GATES L" + ($C.ruling.line | tostring) + " @ " + $C.ruling.commit[0:9] + ")") as $F16
| (if $F16 != $S.f16_ruled then error("the composed FINAL string is not 09's") else . end)
| $S.fig4_reason as $FIG4
| ([$L.disclosure | match("\"([^\"]+)\""; "g") | .captures[0].string]) as $DT
| (if ($DT | length) != 1 then error("L2878 quoted string") else . end)
| "v26.option3_intervals.L2878.1" as $D26
| ([$v.preliminary_outputs[] | select(.name == "Figs 1/6")] | if length == 1 then .[0] else error("Figs 1/6 output") end) as $o16
| (if ([$v.figure_labels[] | select(.output == "Figs 1/6")] | length) != 2 or ([$v.element_tables[] | select(.output == "Figs 1/6")] | length) != 2
      or ([$v.figure_labels[] | select(.output == "Fig 4")] | length) != 1 or ([$v.preliminary_outputs[] | select(.name == "Fig 4")] | length) != 1
   then error("Figs 1/6 / Fig 4 shapes") else . end)
# status_superseded_by (09 Q5): targets and cites copied from the record / GATES
| ($v.disclosures | map({(.id): .}) | add) as $DI
| (if ([$v.disclosures[3].id, $v.disclosures[17].id, $v.disclosures[18].id] != ["C3_downstream_judgment", "row92_label", "appA_recount_label"]) then error("the 3 inventory disclosures") else . end)
| (if ($DI["v25.figs78.pins.1"] == null) or ($DI["v24.row_92.1"] == null) or ($DI["v24.app_A.1"] == null) then error("a target disclosure id") else . end)
| {"C3_downstream_judgment": tgt("v25.figs78.pins.1"; $C.figs78_verified),
   "row92_label": tgt("v24.row_92.1"; $DI["v24.row_92.1"].cite),
   "appA_recount_label": tgt("v24.app_A.1"; $DI["v24.app_A.1"].cite),
   "v24.v23_interval_corrections.INT-CORR.1": tgt($D26; $C.disclosure),
   "v24.v23_interval_corrections.INT-CORR.2": tgt($D26; $C.disclosure),
   "v24.interval_value_half_only.INT-ROW.1": tgt($D26; $C.disclosure),
   "v24.figs_interval_rule.INT-FIG.1": [tgt($D26; $C.disclosure), tgt(null; $C.ruling_b)],
   "v24.figs_interval_rule.INT-FIG.2": [tgt($D26; $C.disclosure), tgt(null; $C.ruling_b)]} as $SSB
| (if ([$SSB | keys[] | . as $k | select($DI[$k] == null)] | length) > 0 then error("a status_superseded_by disclosure id") else . end)
# ---- the record ----
| $v
| .version = "2.6"
| .repo_path = $META.repo_path
| .date_pt = $META.date_pt
| .supersedes = ($META.v25 + {kind: $v.supersedes.kind, version: "2.5"})
| .builder = $META.builder
# 1. the 65 rows
| .printed_items.all_rows |= map(if isin(.row; $R65) then .status = $F16 else . end)
| .printed_items.all_rows_status_counts = (.printed_items.all_rows | counts)
# 7. Figs 1/6 tables: the PRELIMINARY interval elements (each with an interval_note) -> FINAL
| (if [.element_tables[] | select(.output == "Figs 1/6") | [.elements[] | select(.status == "PRELIMINARY" and has("interval_note"))] | length] != [60, 60]
   then error("Figs 1/6: not 60 + 60 interval elements") else . end)
| .element_tables |= map(if .output == "Figs 1/6" then
      (.elements |= map(if .status == "PRELIMINARY" and has("interval_note") then (.status = "FINAL" | .interval_note = $F16) else . end))
      | .figure_status = "FINAL" | .interval_rule.rule = $F16
    else . end)
| (.element_tables | map({(.file): {all_final: all(.elements[]; .status == "FINAL"), prelim_n: ([.elements[] | select(.status == "PRELIMINARY")] | length)}}) | add) as $TS
| .figure_labels |= map(if .output == "Figs 1/6" then
      (if $TS[.file].all_final != true then error("Figs 1/6: a plotted element is not FINAL") else . end)
      | .label = "FINAL" | del(.reason) | .note = $F16 | .interval_rule.rule = $F16 | .preliminary_elements_n = $TS[.file].prelim_n | .disclosures = [$D26]
    elif .output == "Fig 4" then (.reason = $FIG4 | .interval_rule.rule = $FIG4)
    else . end)
| .preliminary_outputs |= map(if .name == "Fig 4" then .interval_rule.rule = $FIG4 else . end)
| .preliminary_outputs |= map(select(.name != "Figs 1/6"))
| .in_scope_outputs += [($o16 | .status = "FINAL" | .ruling = "intervals_option3 (v2.6: option (3), RULING A)" | .moved_from = "preliminary_outputs (v2.5)"
                         | .downstream_tag.text = $F16 | .interval_rule.rule = $F16)]
# 3-4. disclosures
| .disclosures |= map(if $SSB[.id] != null then . + {status_superseded_by: $SSB[.id]} else . end)
| .disclosures += [{id: $D26, text: $DT[0], cite: $C.disclosure}]
# 5-6. row_rulings_v2_4: keys only
| .row_rulings_v2_4.items |= map(
      if .id == "v23_interval_corrections" or .id == "interval_value_half_only" then . + {status_superseded_by: {item_id: "intervals_option3", gates_cite: gc($C.ruling)}}
      elif .id == "figs_interval_rule" then . + {status_superseded_by: [{item_id: "intervals_option3", gates_cite: gc($C.ruling)}, {item_id: "fig4_record_b", gates_cite: gc($C.ruling_b)}]}
      else . end)
| .row_rulings_v2_4.r_node_switch.superseded_by_ruling = gc($C.ruling)
# 2. the v2.6 ruling block
| .row_rulings_v2_6 = {reading: $S.reading,
    items: [{id: "intervals_option3", rows: $R65, outputs: ["Figs 1/6"], status: $F16, cites: [$C.ruling, $C.ruling_a],
             evidence: [$META.results, $META.recheck_b],
             rechecks: [{label: $META.recheck_a.label, report: {file: $META.recheck_a.file, sha256: $META.recheck_a.sha256, in_git: false},
                         verdict_lines: $META.recheck_a.verdict_lines},
                        {label: "recheck (b)", committed: ($META.recheck_b | {path, commit, sha256})}],
             disclosures: [$D26], disclosure_rows: {($D26): $R65}},
            {id: "fig4_record_b", rows: [], outputs: ["Fig 4"], status: $FIG4, cites: [$C.ruling_b], disclosures: [], disclosure_rows: {}}]}
# 9. bookkeeping
| .flip_groups.group_5 = {what: "v2.6 ruling (option (3), RULING A, GATES L2869): the 65 printed interval rows and the Figs 1/6 interval elements FINAL",
                          evidence: ["intervals_option3"]}
| .superseded_fields = {scope: "v2.6 = v2.5 + the v2.6 rulings, by key; everything not listed is byte-identical to v2.5",
     v2_5: ($META.v25 + {kind: $v.supersedes.kind}),
     fields: ["version", "repo_path", "date_pt", "supersedes", "builder", "printed_items.all_rows[the 65 interval rows].status",
              "printed_items.all_rows_status_counts", "element_tables[Figs 1/6 x2].elements[the 60 interval elements].status / .interval_note",
              "element_tables[Figs 1/6 x2].figure_status", "element_tables[Figs 1/6 x2].interval_rule.rule",
              "figure_labels[Figs 1/6 x2]: label, -reason, note, interval_rule.rule, preliminary_elements_n, +disclosures",
              "figure_labels[Fig 4]: reason, interval_rule.rule", "preliminary_outputs[-Figs 1/6]", "preliminary_outputs[Fig 4].interval_rule.rule",
              "in_scope_outputs[+Figs 1/6: status, ruling, moved_from, downstream_tag.text, interval_rule.rule]",
              "disclosures[+v26.option3_intervals.L2878.1]", "disclosures[8: +status_superseded_by]",
              "row_rulings_v2_4.items[v23_interval_corrections, interval_value_half_only, figs_interval_rule]: +status_superseded_by",
              "row_rulings_v2_4.r_node_switch: +superseded_by_ruling", "row_rulings_v2_6", "flip_groups.group_5"],
     v2_5_superseded_fields: $v.superseded_fields}
# ---- c2 (09 via 2e, after the c1 reviews): the interval_rule blocks' disclosure lists. The 5 Figs 1/6 blocks list only the v2.6 disclosure;
# the 2 Fig 4 blocks list only the new RULING B disclosure (the quoted string of GATES L3001 @ 89435ec9a, cited there). Prior lists -> superseded_fields.
# Appended after every c1 stage, so the c1 superseded_fields indices do not move.
| "v26.fig4_record_b.L3001.1" as $D4
| ([$L.fig4_disclosure | match("\"([^\"]+)\""; "g") | .captures[0].string]) as $T4
| (if ($T4 | length) != 1 then error("c2: L3001 quoted string") else . end)
| {"in_scope_outputs[Figs 1/6].interval_rule": ([.in_scope_outputs[] | select(.name == "Figs 1/6") | .interval_rule.disclosures][0]),
   "element_tables[1].interval_rule": .element_tables[1].interval_rule.disclosures, "element_tables[2].interval_rule": .element_tables[2].interval_rule.disclosures,
   "figure_labels[1].interval_rule": .figure_labels[1].interval_rule.disclosures, "figure_labels[2].interval_rule": .figure_labels[2].interval_rule.disclosures,
   "preliminary_outputs[Fig 4].interval_rule": ([.preliminary_outputs[] | select(.name == "Fig 4") | .interval_rule.disclosures][0]),
   "figure_labels[0].interval_rule": .figure_labels[0].interval_rule.disclosures} as $PRIOR
| (if ([$PRIOR[] | select(type != "array")] | length) > 0 or .element_tables[1].output != "Figs 1/6" or .element_tables[2].output != "Figs 1/6"
      or .figure_labels[0].output != "Fig 4" or .figure_labels[1].output != "Figs 1/6" or .figure_labels[2].output != "Figs 1/6"
   then error("c2: the 7 interval_rule blocks") else . end)
| .in_scope_outputs |= map(if .name == "Figs 1/6" then .interval_rule.disclosures = [$D26] else . end)
| .element_tables |= map(if .output == "Figs 1/6" then .interval_rule.disclosures = [$D26] else . end)
| .figure_labels |= map(if .output == "Figs 1/6" then .interval_rule.disclosures = [$D26] elif .output == "Fig 4" then .interval_rule.disclosures = [$D4] else . end)
| .preliminary_outputs |= map(if .name == "Fig 4" then .interval_rule.disclosures = [$D4] else . end)
| .disclosures += [{id: $D4, text: $T4[0], cite: $C.fig4_disclosure}]
| .superseded_fields.fields += ["interval_rule.disclosures of Figs 1/6 (output, tables x2, labels x2): -> [v26.option3_intervals.L2878.1]",
                                "interval_rule.disclosures of Fig 4 (output, label): -> [v26.fig4_record_b.L3001.1]", "disclosures[+v26.fig4_record_b.L3001.1]"]
| .superseded_fields.prior_interval_rule_disclosures = $PRIOR
