# SC6: the v2.6 invariants against the committed v2.5 record, re-derived independently of build_v2_6.jq (its own constants from 09's rulings
# via 2e and the frozen pre-statement 33a40662…; GATES lines, the brief's Fig 4 text and the evidence pins come from the driver, which reads
# them from git / pinned files). Keys, counts and booleans only. Output {n: {...}, bad: [...]}.
# Inputs: --slurpfile REC draft  --slurpfile V25 the v2.5 record  --slurpfile SCOPE scope rows (jsonl)  --slurpfile LN {k: "commit:file:line", v}
#         --argjson PIN {gc, gc78, gf, results: {path, commit, sha256}, recheck_b: {...}, recheck_a: {file, sha256}, base: {path, commit, sha256}}
#         --arg FIG4 (09's Fig 4 text, from the sha-pinned brief)  --arg READING (v26_strings.reading)  --argjson PINOK (the driver's re-checks)
def W($c; $w): if $c then empty else $w end;
def pfx: if startswith("FINAL") then "FINAL" elif startswith("PRELIMINARY") then "PRELIMINARY" elif startswith("not route C") then "not route C" else "OTHER" end;
def counts: group_by(.status) | map({(.[0].status): length}) | add;
def gc3: {commit, file, line};
def isin($n; $xs): ([$xs[] | select(. == $n)] | length) > 0;
# ---- constants (09 via 2e; pre-statement 33a40662…) ----
def D26: "v26.option3_intervals.L2878.1";
def D4: "v26.fig4_record_b.L3001.1";   # c2 (09 via 2e): the RULING B disclosure on the 2 Fig 4 blocks (GATES L3001 @ 89435ec9a)
def RECP: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.6_2026-10-21.json";
def L: {ruling: 2869, ruling_a: 2876, ruling_b: 2877, disclosure: 2878, figs78_verified: 2430};
def NAMES_IN: ["Tables 3/4","rc1a","rc2","Table 6 plant_rows","honest_all_routec printed items","routec_simpool_numbers printed items (FINAL rows)","rc1b printed items, slots 21/22","attack_sol_share printed item (row 106)","cunits_by_tool printed items (Appendix A)","rc1b_detl48 printed items, slot 23 (rows 119-122)","Figs 7/8","rc1b printed items, slot 24 (App H / Fig 7-8 captions)","Figs 1/6"];
def NAMES_PRE: ["rc1b","Fig 5","routec_simpool_numbers printed items","Fig 4"];
def ALLOWED: ["builder", "date_pt", "disclosures", "element_tables", "figure_labels", "flip_groups", "in_scope_outputs", "preliminary_outputs",
              "printed_items", "repo_path", "row_rulings_v2_4", "superseded_fields", "supersedes", "version"];
def SF_FIELDS: ["version", "repo_path", "date_pt", "supersedes", "builder", "printed_items.all_rows[the 65 interval rows].status",
              "printed_items.all_rows_status_counts", "element_tables[Figs 1/6 x2].elements[the 60 interval elements].status / .interval_note",
              "element_tables[Figs 1/6 x2].figure_status", "element_tables[Figs 1/6 x2].interval_rule.rule",
              "figure_labels[Figs 1/6 x2]: label, -reason, note, interval_rule.rule, preliminary_elements_n, +disclosures",
              "figure_labels[Fig 4]: reason, interval_rule.rule", "preliminary_outputs[-Figs 1/6]", "preliminary_outputs[Fig 4].interval_rule.rule",
              "in_scope_outputs[+Figs 1/6: status, ruling, moved_from, downstream_tag.text, interval_rule.rule]",
              "disclosures[+v26.option3_intervals.L2878.1]", "disclosures[8: +status_superseded_by]",
              "row_rulings_v2_4.items[v23_interval_corrections, interval_value_half_only, figs_interval_rule]: +status_superseded_by",
              "row_rulings_v2_4.r_node_switch: +superseded_by_ruling", "row_rulings_v2_6", "flip_groups.group_5",
              "interval_rule.disclosures of Figs 1/6 (output, tables x2, labels x2): -> [v26.option3_intervals.L2878.1]",
              "interval_rule.disclosures of Fig 4 (output, label): -> [v26.fig4_record_b.L3001.1]", "disclosures[+v26.fig4_record_b.L3001.1]"];
# 09 Q5: disclosure id -> [[target disclosure id, cite source]]; a 1-entry list is stored as one object, a 2-entry list as a list
def SSB_RULE: {"C3_downstream_judgment": [["v25.figs78.pins.1", "figs78_verified"]], "row92_label": [["v24.row_92.1", "own"]],
               "appA_recount_label": [["v24.app_A.1", "own"]],
               "v24.v23_interval_corrections.INT-CORR.1": [[D26, "disclosure"]], "v24.v23_interval_corrections.INT-CORR.2": [[D26, "disclosure"]],
               "v24.interval_value_half_only.INT-ROW.1": [[D26, "disclosure"]],
               "v24.figs_interval_rule.INT-FIG.1": [[D26, "disclosure"], [null, "ruling_b"]], "v24.figs_interval_rule.INT-FIG.2": [[D26, "disclosure"], [null, "ruling_b"]]};
# 09 Q1: v2.4 item id -> [[v2.6 item id, cite source]]
def ISSB_RULE: {"v23_interval_corrections": [["intervals_option3", "ruling"]], "interval_value_half_only": [["intervals_option3", "ruling"]],
                "figs_interval_rule": [["intervals_option3", "ruling"], ["fig4_record_b", "ruling_b"]]};
$REC[0] as $r | $V25[0] as $v
| ($LN | map({(.k): .v}) | add // {}) as $LT
| def lt($c; $n): $LT["\($c):\($PIN.gf):\($n)"];
  def cite($k): (if $k == "figs78_verified" then $PIN.gc78 else $PIN.gc end) as $c | {commit: $c, file: $PIN.gf, line: L[$k]};
  def one($x): if ($x | length) == 1 then $x[0] else $x end;
  ([$SCOPE[].row] | unique) as $R65
| ([lt($PIN.gc; L.ruling_a) // "" | match("INTERVALS REPRODUCED \\(EXACT\\)"; "g") | .string]) as $PA
| ("FINAL (" + ($PA[0] // "MISSING") + "; GATES L" + (L.ruling | tostring) + " @ " + $PIN.gc[0:9] + ")") as $F16
| ([lt($PIN.gc; L.disclosure) // "" | match("\"([^\"]+)\""; "g") | .captures[0].string]) as $DT
| ([$v.printed_items.all_rows[] | {(.row | tostring): .}] | add) as $vrow
| ([$r.row_rulings_v2_6.items[]? | select(.id == "intervals_option3")]) as $it
| ([$r.row_rulings_v2_6.items[]? | select(.id == "fig4_record_b")]) as $it4
| ($v.disclosures | map({(.id): .}) | add) as $VD
| ($r.disclosures | map({(.id): .}) | add) as $RD
| ([$r.figure_labels[] | select(.output == "Figs 1/6")]) as $rl16 | ([$v.figure_labels[] | select(.output == "Figs 1/6")]) as $vl16
| ([$r.element_tables[] | select(.output == "Figs 1/6")]) as $rt16 | ([$v.element_tables[] | select(.output == "Figs 1/6")]) as $vt16
| ([$r.in_scope_outputs[] | select(.name == "Figs 1/6")]) as $ro16 | ([$v.preliminary_outputs[] | select(.name == "Figs 1/6")]) as $vo16
| [
  # S0 the driver's pin re-checks (base record = git, scope file, GATES lines present, evidence blobs, recheck (a) report and its verdict lines)
  W($PINOK == true and ($PA | length) == 1 and ($DT | length) == 1 and ($R65 | length) == 65; "S0 a pin re-check failed (driver, RULING A phrase, L2878 quote or scope rows)"),
  # S1 (B12) rows: the same rows / records / every non-status field as v2.5
  W(([$r.printed_items.all_rows[] | del(.status)]) == ([$v.printed_items.all_rows[] | del(.status)]); "S1 all_rows differ from v2.5 beyond status"),
  # S2 status changes exactly the 65 scope rows (and the scope rows are v2.5's interval-rule PRELIMINARY rows)
  W(([$r.printed_items.all_rows[] | select(.status != $vrow[.row | tostring].status) | .row] | sort) == $R65
    and ([$v.printed_items.all_rows[] | select((.status | startswith("PRELIMINARY (v2.4 correction")) or (.status | startswith("PRELIMINARY (value half only for the interval"))) | .row] | sort) == $R65;
    "S2 the status-changed rows are not exactly the 65 scope rows"),
  # S3 (B9) each flipped row's status == the item status == F16 (composed here from L2876 and the GATES commit)
  W(($it | length) == 1 and $it[0].status == $F16 and all($r.printed_items.all_rows[] | select(isin(.row; $R65)); .status == $F16);
    "S3 a flipped row's status or the item status is not the composed FINAL string"),
  # S4 counts and the stored map
  W(($r.printed_items.all_rows | map(.status | pfx) | group_by(.) | map({(.[0]): length}) | add) == {"FINAL": 237, "PRELIMINARY": 2, "not route C": 12};
    "S4 the counts are not FINAL 237 / PRELIMINARY 2 / not route C 12"),
  W($r.printed_items.all_rows_status_counts == ($r.printed_items.all_rows | counts); "S4 all_rows_status_counts is not the recomputed map"),
  # S5 (B10) the intervals_option3 item
  W(($it | length) == 1 and $it[0].rows == $R65 and $it[0].outputs == ["Figs 1/6"]
    and ([$it[0].cites[] | gc3] == [cite("ruling"), cite("ruling_a")])
    and ([$it[0].evidence[] | {path, commit, sha256}] == [$PIN.results, $PIN.recheck_b])
    and ([$it[0].rechecks[] | .report? | select(. != null) | {file, sha256}] == [$PIN.recheck_a])
    and ([$it[0].rechecks[] | .committed? | select(. != null)] == [$PIN.recheck_b])
    and $it[0].disclosures == [D26] and $it[0].disclosure_rows == {(D26): $R65};
    "S5 the intervals_option3 item is not the pre-stated one"),
  W(($it4 | length) == 1 and $it4[0].rows == [] and $it4[0].outputs == ["Fig 4"] and $it4[0].status == $FIG4 and ([$it4[0].cites[] | gc3] == [cite("ruling_b")]);
    "S5b the fig4_record_b item is not the pre-stated one"),
  # S6 (B13) the v2.6 disclosure: one matching object; text = L2878's quoted string; cite = L2878
  W(([$r.disclosures[] | select(.id == D26)] | length) == 1 and $RD[D26].text == $DT[0] and ($RD[D26].cite | gc3) == cite("disclosure")
    and ([$r.disclosures[] | select(.id | startswith("v26."))] | length) == 2;
    "S6 the v2.6 disclosure is not L2878's quoted string at the GATES commit"),
  # S7 status_superseded_by: exactly the ruled 8 disclosures; targets resolve; cites exist at their commits; nothing else changed
  W(([$r.disclosures[] | select(has("status_superseded_by")) | .id] | sort) == (SSB_RULE | keys); "S7 status_superseded_by is not on exactly the 8 ruled disclosures"),
  ($r.disclosures[] | select(has("status_superseded_by")) | . as $d | (SSB_RULE[$d.id] // []) as $rule
   | one([$rule[] | {disclosure_id: .[0], gates_cite: (if .[1] == "own" then ($VD[.[0]].cite | gc3) else cite(.[1]) end)}]) as $want
   | W($d.status_superseded_by == $want; "S7 \($d.id): status_superseded_by is not the ruled target")),
  W(all($r.disclosures[] | select(has("status_superseded_by")) | (.status_superseded_by | if type == "array" then .[] else . end);
        (.disclosure_id == null or ($RD[.disclosure_id] != null)) and (lt(.gates_cite.commit; .gates_cite.line) != null) and .gates_cite.file == $PIN.gf);
    "S7 a status_superseded_by target does not resolve, or its GATES line does not exist"),
  W(([$r.disclosures[] | select(.id | startswith("v26.") | not) | del(.status_superseded_by)]) == $v.disclosures; "S7 a v2.5 disclosure changed beyond the added status_superseded_by key"),
  # S8 the v2.4 items' status_superseded_by (09 Q1); row_rulings_v2_4 otherwise unchanged (except S9's key)
  W(([$r.row_rulings_v2_4.items[] | select(has("status_superseded_by")) | .id] | sort) == (ISSB_RULE | keys)
    and all($r.row_rulings_v2_4.items[] | select(has("status_superseded_by")); .status_superseded_by == one([ISSB_RULE[.id][] | {item_id: .[0], gates_cite: cite(.[1])}]))
    and (($r.row_rulings_v2_4 | .items |= map(del(.status_superseded_by)) | del(.r_node_switch.superseded_by_ruling)) == $v.row_rulings_v2_4)
    and all($r.row_rulings_v2_4.items[] | select(has("status_superseded_by")) | (.status_superseded_by | if type == "array" then .[] else . end);
            . as $s | [($r.row_rulings_v2_6.items // [])[].id] | index([$s.item_id]) != null);
    "S8 the v2.4 items' status_superseded_by is not the ruled set, or row_rulings_v2_4 changed otherwise"),
  # S9 r_node_switch.superseded_by_ruling == L2869's {commit, file, line}
  W($r.row_rulings_v2_4.r_node_switch.superseded_by_ruling == cite("ruling") and $r.row_rulings_v2_4.r_node_switch.state == "OFF";
    "S9 r_node_switch.superseded_by_ruling is not L2869's cite"),
  # S10 the Figs 1/6 tables: only the 60 + 60 interval elements' status / interval_note, figure_status and interval_rule.rule changed
  W(($rt16 | length) == 2 and ($vt16 | length) == 2
    and all(range(0; 2); . as $i | ($rt16[$i] | del(.elements, .figure_status, .interval_rule.rule, .interval_rule.disclosures)) == ($vt16[$i] | del(.elements, .figure_status, .interval_rule.rule, .interval_rule.disclosures)) and $rt16[$i].interval_rule.disclosures == [D26]
          and $rt16[$i].figure_status == "FINAL" and $rt16[$i].interval_rule.rule == $F16
          and ([$vt16[$i].elements[] | select(.status == "PRELIMINARY" and has("interval_note"))] | length) == 60
          and ([range(0; $vt16[$i].elements | length) as $j | $vt16[$i].elements[$j] as $a | $rt16[$i].elements[$j] as $b
                | if ($a.status == "PRELIMINARY" and ($a | has("interval_note"))) then ($b == ($a | .status = "FINAL" | .interval_note = $F16)) else ($b == $a) end] | all)
          and all($rt16[$i].elements[]; .status == "FINAL"));
    "S10 the Figs 1/6 element tables are not v2.5's with the interval elements FINAL and the composed note / rule"),
  # S11 the Figs 1/6 labels
  W(($rl16 | length) == 2 and all(range(0; 2); . as $i | $rl16[$i] as $a | $vl16[$i] as $b
          | ([$r.element_tables[] | select(.file == $a.file)]) as $ts | ($ts[0] // {elements: []}) as $t
          | ($ts | length) == 1
            and (($a.label == "FINAL") == all($t.elements[]; .status == "FINAL")) and $a.label == "FINAL" and ($a | has("reason") | not)
            and $a.note == $F16 and $a.interval_rule.rule == $F16 and $a.disclosures == [D26]
            and $a.preliminary_elements_n == ([$t.elements[] | select(.status == "PRELIMINARY")] | length)
            and ($a | del(.label, .note, .interval_rule.rule, .interval_rule.disclosures, .preliminary_elements_n, .disclosures)) == ($b | del(.label, .note, .reason, .interval_rule.rule, .interval_rule.disclosures, .preliminary_elements_n)) and $a.interval_rule.disclosures == [D26]);
    "S11 the Figs 1/6 labels are not FINAL with the composed note / rule, no reason, the recomputed count and the v2.6 disclosure"),
  # S12 the Figs 1/6 output: v2.5's preliminary entry moved to in_scope, only status / ruling / moved_from / tag text / rule changed
  W(($ro16 | length) == 1 and ($vo16 | length) == 1 and ($v.in_scope_outputs | map(.name) | index("Figs 1/6")) == null
    and $ro16[0].status == "FINAL" and $ro16[0].ruling == "intervals_option3 (v2.6: option (3), RULING A)" and $ro16[0].moved_from == "preliminary_outputs (v2.5)"
    and $ro16[0].downstream_tag.text == $F16 and $ro16[0].interval_rule.rule == $F16
    and ($ro16[0] | del(.status, .ruling, .moved_from, .downstream_tag.text, .interval_rule.rule, .interval_rule.disclosures)) == ($vo16[0] | del(.status, .moved_from, .downstream_tag.text, .interval_rule.rule, .interval_rule.disclosures)) and $ro16[0].interval_rule.disclosures == [D26];
    "S12 the Figs 1/6 output is not v2.5's moved to in_scope with the ruled fields"),
  # S13 Fig 4: its 3 texts == 09's string byte for byte; PRELIMINARY; otherwise unchanged
  ([$r.figure_labels[] | select(.output == "Fig 4")] as $a | [$v.figure_labels[] | select(.output == "Fig 4")] as $b
   | [$r.preliminary_outputs[] | select(.name == "Fig 4")] as $c | [$v.preliminary_outputs[] | select(.name == "Fig 4")] as $d
   | W(($a | length) == 1 and ($b | length) == 1 and ($c | length) == 1 and ($d | length) == 1
       and $a[0].reason == $FIG4 and $a[0].interval_rule.rule == $FIG4 and $c[0].interval_rule.rule == $FIG4 and $a[0].label == "PRELIMINARY"
       and ($a[0] | del(.reason, .interval_rule.rule, .interval_rule.disclosures)) == ($b[0] | del(.reason, .interval_rule.rule, .interval_rule.disclosures)) and ($c[0] | del(.interval_rule.rule, .interval_rule.disclosures)) == ($d[0] | del(.interval_rule.rule, .interval_rule.disclosures))
       and $a[0].interval_rule.disclosures == [D4] and $c[0].interval_rule.disclosures == [D4];
       "S13 Fig 4's three texts are not 09's string, or Fig 4 changed otherwise")),
  # S14 the two-way status sweep (addition 3, B5, B6) over current status-bearing fields; dated history is not swept (09 Q1)
  ([ ($r.printed_items.all_rows[] | {s: (.status | pfx), t: [.status]}),
     (($r.row_rulings_v2_6.items // [])[] | {s: (.status | pfx), t: [.status]}),
     (($r.in_scope_outputs[] | {s: "FINAL", t: [.status, .reason, .pending_downstream_label, (.downstream_tag | objects | .text), (.interval_rule | objects | .rule)]}),
      ($r.preliminary_outputs[] | {s: (.status | pfx), t: [.status, .reason, .pending_downstream_label, (.downstream_tag | objects | .text), (.interval_rule | objects | .rule)]})),
     ($r.figure_labels[] | {s: .label, t: [.note, .reason, (.interval_rule | objects | .rule)]}),
     ($r.element_tables[] | {s: .figure_status, t: [(.interval_rule | objects | .rule)]}),
     ($r.element_tables[] | .elements[] | {s: .status, t: [.interval_note, .reason]}) ]
   | map(.t |= map(strings))) as $SW
| W(all($SW[] | select(.s == "FINAL"); all(.t[]; startswith("PRELIMINARY") | not)); "S14 a FINAL row / item / output / label / table carries PRELIMINARY-prefixed status text"),
  W(all($SW[] | select(.s == "PRELIMINARY"); all(.t[]; startswith("FINAL") | not)); "S14 a PRELIMINARY row / item / output / label / table carries FINAL-prefixed status text"),
  W(all($r.figure_labels[] | select(.label == "FINAL"); has("reason") | not); "S14 a FINAL label has a .reason"),
  W(all(($r.in_scope_outputs + $r.preliminary_outputs)[] | select(has("downstream_tag")) | .downstream_tag; . == null or ((type == "object") and ((.text | type) == "string")));
    "S14 a downstream_tag is not null or an object with a string text"),
  W(all($r.in_scope_outputs[]; .status == "FINAL") and all($r.preliminary_outputs[]; .status == "PRELIMINARY"); "S14 an output's status is not its list's"),
  # S15 (B7) a conditional ("applies only when the X item is in the version") item whose X item is in this version has its disclosures on a current item
  ([$r.row_rulings_v2_4.items[] | select(.status | test("^disclosure \\(applies only when the [^ ]+ item is in the version\\)"))
    | {x: (.status | capture("applies only when the (?<x>[^ ]+) item is in the version").x), d: .disclosures}]) as $cond
| W(all($cond[]; . as $c | if ([($r.row_rulings_v2_5.items[]?, $r.row_rulings_v2_6.items[]?) | select(.id == $c.x)] | length) > 0
                         then all($c.d[]; . as $id | [($r.row_rulings_v2_5.items[]?, $r.row_rulings_v2_6.items[]?) | select(.id == $c.x) | .disclosures[]] | index([$id]) != null)
                         else true end) and ($cond | length) >= 1;
    "S15 a conditional disclosure whose condition holds is not attached to its current item"),
  # S16 unchanged elsewhere
  ((($v | keys) - ALLOWED)[] as $k | W($r[$k] == $v[$k]; "S16 top-level key \($k) changed")),
  W((($r | keys) - ($v | keys)) == ["row_rulings_v2_6"]; "S16 new top-level keys other than row_rulings_v2_6"),
  W(([$r.in_scope_outputs[] | select(.name != "Figs 1/6")]) == $v.in_scope_outputs
    and ([$r.preliminary_outputs[] | select(.name != "Fig 4")]) == ([$v.preliminary_outputs[] | select(.name != "Fig 4" and .name != "Figs 1/6")]);
    "S16 another output changed"),
  W(([$r.element_tables[] | select(.output != "Figs 1/6")]) == ([$v.element_tables[] | select(.output != "Figs 1/6")]); "S16 an element table other than Figs 1/6 changed"),
  W(([$r.figure_labels[] | select(.output != "Figs 1/6" and .output != "Fig 4")]) == ([$v.figure_labels[] | select(.output != "Figs 1/6" and .output != "Fig 4")])
    and ([$r.figure_labels[].output] == [$v.figure_labels[].output]); "S16 a figure label other than Figs 1/6 / Fig 4 changed"),
  W(($r.printed_items | del(.all_rows, .all_rows_status_counts)) == ($v.printed_items | del(.all_rows, .all_rows_status_counts)); "S16 printed_items changed beyond statuses and counts"),
  W($r.printed_items.groups_n == ([($r.in_scope_outputs + $r.preliminary_outputs)[] | select(.kind == "printed_items")] | length)
    and $r.printed_items.items_n == ([($r.in_scope_outputs + $r.preliminary_outputs)[] | select(.kind == "printed_items") | .items_n] | add);
    "S16 groups_n / items_n are not the recomputed totals"),
  W(($r.flip_groups | del(.group_5)) == $v.flip_groups and ($r.flip_groups.group_5.evidence == ["intervals_option3"]); "S16 flip groups 1-4 changed, or group 5 is not the v2.6 one"),
  W($r.row_rulings_v2_5 == $v.row_rulings_v2_5; "S16 row_rulings_v2_5 changed"),
  W($r.superseded_fields.v2_5_superseded_fields == $v.superseded_fields; "S16 v2.5's superseded_fields not kept in full"),
  # S17 identity
  W($r.version == "2.6" and $r.repo_path == RECP and ($r.supersedes | {path, commit, sha256}) == $PIN.base and $r.supersedes.version == "2.5";
    "S17 version / repo_path / supersedes"),
  # S18 the reading
  W($r.row_rulings_v2_6.reading == $READING; "S18 row_rulings_v2_6.reading is not the pre-stated string"),
  # S19 output names
  W([$r.in_scope_outputs[].name] == NAMES_IN and [$r.preliminary_outputs[].name] == NAMES_PRE; "S19 output names are not the pre-stated lists"),
  # S20 superseded_fields.fields
  W($r.superseded_fields.fields == SF_FIELDS and ($r.superseded_fields.v2_5 | {path, commit, sha256}) == $PIN.base
    and $r.superseded_fields.prior_interval_rule_disclosures
        == {"in_scope_outputs[Figs 1/6].interval_rule": ([$v.preliminary_outputs[] | select(.name == "Figs 1/6") | .interval_rule.disclosures][0]),
            "element_tables[1].interval_rule": $v.element_tables[1].interval_rule.disclosures, "element_tables[2].interval_rule": $v.element_tables[2].interval_rule.disclosures,
            "figure_labels[1].interval_rule": $v.figure_labels[1].interval_rule.disclosures, "figure_labels[2].interval_rule": $v.figure_labels[2].interval_rule.disclosures,
            "preliminary_outputs[Fig 4].interval_rule": ([$v.preliminary_outputs[] | select(.name == "Fig 4") | .interval_rule.disclosures][0]),
            "figure_labels[0].interval_rule": $v.figure_labels[0].interval_rule.disclosures};
    "S20 superseded_fields.fields / v2_5 pin / prior interval_rule disclosure lists are not the pre-stated ones"),
  # S21 (c2, 09) the disclosure-text sweep: what a FINAL / PRELIMINARY output, table, label or ruling item lists (interval_rule blocks,
  # label disclosures, v2.5 / v2.6 items) never carries the opposite status prefix; every listed id resolves
  ([ (($r.in_scope_outputs[] | {s: "FINAL", d: ((.interval_rule | objects | .disclosures) // [])}),
      ($r.preliminary_outputs[] | {s: (.status | pfx), d: ((.interval_rule | objects | .disclosures) // [])})),
     ($r.element_tables[] | {s: .figure_status, d: ((.interval_rule | objects | .disclosures) // [])}),
     ($r.figure_labels[] | {s: .label, d: (((.interval_rule | objects | .disclosures) // []) + (.disclosures // []))}),
     (($r.row_rulings_v2_5.items // [])[], ($r.row_rulings_v2_6.items // [])[] | {s: (.status | pfx), d: (.disclosures // [])}) ]) as $DL
| W(all($DL[] | select(.s == "FINAL") | .d[]; ($RD[.].text // "") | startswith("PRELIMINARY") | not);
    "S21 a FINAL output / table / label / item lists a disclosure whose text starts with PRELIMINARY"),
  W(all($DL[] | select(.s == "PRELIMINARY") | .d[]; ($RD[.].text // "") | startswith("FINAL") | not);
    "S21 a PRELIMINARY output / table / label / item lists a disclosure whose text starts with FINAL"),
  W(all($DL[] | .d[]; $RD[.] != null); "S21 a listed disclosure id does not resolve"),
  # S22 (c2, 09) the Fig 4 disclosure: one object; text == GATES L3001's one quoted string @ 89435ec9a; cite == L3001
  ([lt($PIN.gc2; 3001) // "" | match("\"([^\"]+)\""; "g") | .captures[0].string]) as $Q4
| W(([$r.disclosures[] | select(.id == D4)] | length) == 1 and ($Q4 | length) == 1 and $RD[D4].text == $Q4[0]
    and ($RD[D4].cite | gc3) == {commit: $PIN.gc2, file: $PIN.gf, line: 3001};
    "S22 the Fig 4 disclosure is not GATES L3001's quoted string")
  ] as $bad
| {n: {rows: ($r.printed_items.all_rows | length), FINAL: ([$r.printed_items.all_rows[] | select(.status | startswith("FINAL"))] | length),
       PRELIMINARY: ([$r.printed_items.all_rows[] | select(.status | startswith("PRELIMINARY"))] | length),
       not_route_c: ([$r.printed_items.all_rows[] | select(.status | startswith("not route C"))] | length),
       changed_rows: ([$r.printed_items.all_rows[] | select(.status != $vrow[.row | tostring].status)] | length),
       ssb_disclosures: ([$r.disclosures[] | select(has("status_superseded_by"))] | length), v26_disclosures: ([$r.disclosures[] | select(.id | startswith("v26."))] | length),
       figs16_label_final: ([$rl16[] | select(.label == "FINAL")] | length), pin_ok: $PINOK},
   bad: $bad}
