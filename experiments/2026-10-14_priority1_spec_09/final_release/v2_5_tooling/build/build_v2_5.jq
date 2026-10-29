# build_v2_5.jq: route C FINAL-release record v2.5 = v2.4 (6a46541d5) + the v2.5 rulings, by key. Never prints; the wrapper writes the output.
# 09 (via 2e): rows 119-121 FINAL for draft 0029's visible text, with A2: printed_items.provenance stays v18 for every row, but rows
# 119-121's printed token and anchor come from v19 (X), copied by script; the B1-X (GATES L1974) and result_keys_gate (L2267) disclosures,
# attached to rows 119-121 only; the figs78 item is a pre-stated branch (only with 09's verification line and a CONFIRMED recheck).
# Inputs: --slurpfile V24 the v2.4 record   --slurpfile P18 / P19 provenance v18 / v19 rows (jsonl)   --slurpfile RUL rulings_v2_5.json
#         --slurpfile RES resolved rulings (anchors / evidence / rechecks; keys and shas only)
#         --slurpfile LN {k: "commit:file:line", v: line text} for every resolved anchor (read from git by the wrapper, line sha re-checked)
#         --argjson META {date_pt, repo_path, builder, v24, v19, rv_recheck, figs78: null}
def strip_bullet: sub("^\\s*-\\s+"; "");
# 09 (c4, via 2e): the Figs 7/8 downstream_tag.text base = the quoted string of build_v2_4.jq L103 at 6a46541d5 (experiments/2026-10-14_priority1_spec_09/final_release/v2_4_tooling/build/build_v2_4.jq),
# copied here by script from git show (the v2.4 builder's $HAS78 branch, which v2.5 does not run; see the DELTA).
def TAG24_L103: "FINAL per element after the figs78 item (every drawn element mapped to a1's verdict rows; GATES 14:05, 22:22, 22:46 Q4-Q6)";
# 09 (c4, via 2e): row_rulings_v2_5.reading in branch (i), copied here by script from build/reading_v25_branch_i_09_ruling.txt (f61ac716…)
def READING_I: "v2.5 = v2.4 + rows 119-121 FINAL for draft 0029's visible text (09, GATES L2364) + slot-24 rows 69-74 / 241-248 and Figs 7/8 FINAL (figs78 branch (i); 09, GATES L2430); everything else as v2.4";
# 09 (c5, via 2e): the joiner, copied here by script from build/c5_strings_09_ruling.json (668b7c06…)
def C5_JOINER: "; point elements: ";
$V24[0] as $v | $RUL[0] as $R | $RES[0] as $X
| ($LN | map({(.k): .v}) | add // {}) as $LT
| ($P18 | map({(.row | tostring): .}) | add) as $p18 | ($P19 | map({(.row | tostring): .}) | add) as $p19
| def item($id): [$R.items[] | select(.id == $id)][0];
  def anchor($must): ($X.anchors[$must] // error("NO ANCHOR " + $must)).anchor;
  def atext($must): anchor($must) as $a | ($LT["\($a.commit):\($a.file):\($a.line)"] // error("NO LINE TEXT " + $must));
  def aquotes($must): [atext($must) | match("\"([^\"]+)\""; "g") | .captures[0].string];
  def ev($e): ($X.evidence[$e.path + "@" + $e.commit] // error("NO EVIDENCE " + $e.path)) + {role: $e.role};
  def item_disclosures($it): [($it.quote_lines // [])[] | . as $ql | aquotes($ql.must) | to_entries[]
                              | {id: ("v25." + $it.id + "." + $ql.role + "." + ((.key + 1) | tostring)), text: .value, cite: anchor($ql.must)}];
  def prov_item($p): {row: $p.row, item: $p.item, anchor: $p.anchor, key_path: $p.key_path, record: $p.record, record_sha256: $p.record_sha256};
  def isin($n; $rows): ([$rows[] | select(. == $n)] | length) > 0;
# ---- rows 119-121 ----
  item("rows_119_121") as $it
| $it.rows as $R3
| (if ($R3 | sort) != [119, 120, 121] then error("rows_119_121 rows") else . end)
| ([$R3[] | tostring as $n | ($p18[$n] | del(.v84_sha256, .anchor, .printed)) == ($p19[$n] | del(.v84_sha256, .anchor, .printed))] | all) as $A2same
| (if $A2same | not then error("A2: v19 differs from v18 on rows 119-121 outside anchor / printed / v84_sha256") else . end)
| (item_disclosures($it)) as $D25
| ([atext($it.quote_lines[0].must) | match("X = draft [0-9]+ with draft [0-9]+'s [0-9]+ App C digits") | .string][0]
   // error("the verifier-input phrase is not on the B1-X line")) as $XIN
| ([$v.in_scope_outputs[] | select(.name == "rc1b_detl48 printed item, slot 23 (row 122)")][0] // error("no v2.4 row-122 group")) as $g122
| ([$v.preliminary_outputs[] | select(.name == "rc1b_detl48 printed items, slot 23")] | length) as $n_pre
| (if $n_pre != 1 then error("no single v2.4 slot-23 preliminary group") else . end)
# ---- the record ----
| $v
| .version = "2.5"
| .repo_path = $META.repo_path
| .date_pt = $META.date_pt
| .supersedes = ($META.v24 + {version: "2.4"})
| .builder = $META.builder
| .printed_items.all_rows |= map(if isin(.row; $R3) then .status = $it.status else . end)
| .printed_items.all_rows_status_counts = (.printed_items.all_rows | group_by(.status) | map({(.[0].status): length}) | add)
| .in_scope_outputs |= map(if .name == $g122.name then
      {name: "rc1b_detl48 printed items, slot 23 (rows 119-122)", kind: "printed_items", status: "FINAL", rows: [119, 120, 121, 122], items_n: 4,
       items: ([$R3[] | prov_item($p19[tostring])] + [$g122.items[] | select(.row == 122)]),
       ruling: "rows_119_121 (v2.5: draft 0029's visible text; printed token and anchor from v19, A2) + rows_119_122 (v2.4: row 122)"}
    else . end)
| .preliminary_outputs |= map(select(.name != "rc1b_detl48 printed items, slot 23"))
| .printed_items.groups_n = ([(.in_scope_outputs + .preliminary_outputs)[] | select(.kind == "printed_items")] | length)
| .printed_items.items_n = ([(.in_scope_outputs + .preliminary_outputs)[] | select(.kind == "printed_items") | .items_n] | add)
| .disclosures += $D25
| .row_rulings_v2_5 = {
     reading: "v2.5 = v2.4 + rows 119-121 FINAL for draft 0029's visible text (09, GATES L2364); everything else as v2.4",
     draft_scope: {rows: $R3, visible_text_of: "draft 0029", verifier_input: $XIN,
                   provenance_pin: "printed_items.provenance stays v18 for every row (A2)",
                   printed_and_anchor_source: ($META.v19 + {rows: $R3, fields: ["anchor", "printed"],
                                               equal_to_v18_except: ["anchor", "printed", "v84_sha256"]}),
                   evidence: [$it.evidence[] | ev(.)], recheck: $META.rv_recheck, recheck_cite: anchor($it.rechecks[0].gates_must)},
     items: [{id: $it.id, rows: $R3, status: $it.status, cites: [$it.cites[] | anchor(.)], evidence: [$it.evidence[] | ev(.)],
              rechecks: [{label: $it.rechecks[0].label, what: $it.rechecks[0].what, cite: anchor($it.rechecks[0].gates_must),
                          report: ($X.rechecks[$it.rechecks[0].file] | {file, sha256, in_git: false})}],
              disclosures: [$D25[].id], disclosure_rows: ([$D25[] | {(.id): $R3}] | add)}],
     figs78: {branch: "ii", included: false,
              rule: "FINAL only if 09's GATES line 'figs78 results commit <sha> verified' exists at freeze time and its fresh recheck is CONFIRMED (09, via 2e); otherwise v2.6"}}
| .flip_groups.group_4 = {what: "v2.5 ruling: rows 119-121 FINAL for draft 0029's visible text (printed token and anchor from v19, A2)", evidence: ["rows_119_121"]}
| .superseded_fields = {scope: "v2.5 = v2.4 + the v2.5 rulings, by key; everything not listed is byte-identical to v2.4",
     v2_4: $META.v24,
     fields: ["version", "repo_path", "date_pt", "supersedes", "builder", "printed_items.all_rows[119-121].status", "printed_items.all_rows_status_counts",
              "printed_items.groups_n", "printed_items.items_n", "in_scope_outputs[the slot-23 group: row 122 -> rows 119-122]",
              "preliminary_outputs[- the slot-23 group, rows 119-121]", "disclosures[+v25.*]", "row_rulings_v2_5", "flip_groups.group_4"],
     v2_4_superseded_fields: $v.superseded_fields}
# ---- figs78 branch (i): only when the wrapper passes the verified record ($META.figs78 != null; 09: GATES L2430 @ ae77708bf) ----
| if $META.figs78 == null then . else
    item("figs78") as $f
    | $f.rows as $S24
    | (if ($S24 | sort) != ([69, 70, 71, 72, 73, 74] + [range(241; 249)]) then error("figs78 rows") else . end)
    | ([atext($f.reason_must) | match("figs78 results commit [0-9a-f]+ verified \\(fresh recheck CONFIRMED\\); slot 24 \\+ Figs 7/8 FINAL-eligible for v2\\.5 branch \\(i\\)") | .string][0]
       // error("the branch (i) reason is not on its GATES line")) as $REASON
    | (item_disclosures($f)) as $DP
    | (if ($DP | length) != 1 then error("figs78: not exactly one pins disclosure") else . end)
    | ([$v.preliminary_outputs[] | select(.name == "Figs 7/8")][0] // error("no v2.4 Figs 7/8 output")) as $f78o
    | ([$v.preliminary_outputs[] | select(.name == "rc1b printed items (App H / Fig 7-8 captions)")][0] // error("no v2.4 slot-24 group")) as $g24
    | ([.figure_labels[] | select(.file | test("figs78"))] | length) as $nlab
    | (if $nlab != 2 then error("figs78: not exactly 2 figure labels") else . end)
    # 09 (c4, #2): v2.4's pre-stated "applies only when the figs78 item is in the version" disclosure, attached figure-level (no disclosure_rows
    # entry); its id copied from v2.4's item figs78_axis_presentation, and its disclosure object stays v2.4's (already in .disclosures)
    | ([$v.row_rulings_v2_4.items[] | select(.id == "figs78_axis_presentation")
        | select(.status == "disclosure (applies only when the figs78 item is in the version)") | .disclosures[]]) as $AXIS
    | (if ($AXIS | length) != 1 or ([.disclosures[] | select(.id == $AXIS[0])] | length) != 1 then error("figs78: the v2.4 axis disclosure") else . end)
    # 09 (c4, #1): the branch (i) reading; build-time sanity only (SC5 asserts the full string)
    | (.row_rulings_v2_5.reading | split(" (09, GATES L2364)")[0] + " (09, GATES L2364)") as $rpre
    | (if (READING_I | startswith($rpre)) and (READING_I | endswith("; everything else as v2.4")) then . else error("the branch (i) reading") end)
    | .row_rulings_v2_5.reading = READING_I
    | .printed_items.all_rows |= map(if isin(.row; $S24) then .status = ("FINAL (" + $REASON + ")") else . end)
    | .printed_items.all_rows_status_counts = (.printed_items.all_rows | group_by(.status) | map({(.[0].status): length}) | add)
    # 09 (finding 2, (a)): the Figs 7/8 downstream tag records its recompute = the same record pin as pending_downstream.recomputed["Figs 7/8"]
    # 09 (c4): its downstream_tag.text = TAG24_L103 + "; results commit <record commit, 9 hex> verified, GATES L<verified line> @ <GATES commit>"
    | anchor($f.cites[0]) as $vc
    | .in_scope_outputs += [($f78o | .status = "FINAL" | del(.reason, .pending_downstream_label) | . + {ruling: "figs78 (v2.5 branch (i))", moved_from: "preliminary_outputs (v2.4)"}
                             | .downstream_tag.recomputed_output = $META.figs78.record
                             | .downstream_tag.text = (TAG24_L103 + "; results commit " + $META.figs78.record.commit[0:9] + " verified, GATES L" + ($vc.line | tostring) + " @ " + $vc.commit)),
                           ($g24 | .name = "rc1b printed items, slot 24 (App H / Fig 7-8 captions)" | .status = "FINAL" | del(.reason)
                                 | .ruling = "figs78 (v2.5 branch (i))")]
    | .preliminary_outputs |= map(select(.name != "Figs 7/8" and .name != "rc1b printed items (App H / Fig 7-8 captions)"))
    | .printed_items.groups_n = ([(.in_scope_outputs + .preliminary_outputs)[] | select(.kind == "printed_items")] | length)
    | .printed_items.items_n = ([(.in_scope_outputs + .preliminary_outputs)[] | select(.kind == "printed_items") | .items_n] | add)
    | .figure_labels |= map(if (.file | test("figs78")) then (.label = "FINAL" | del(.reason) | .note = ("FINAL (" + $REASON + ")") | .disclosures = [$DP[].id]) else . end)
    | .element_tables += [ $F78[0].element_tables[] | {output: "Figs 7/8", kind: "figs78_item", file, commit, sha256, figure_status, inventory_complete,
                           elements_all_final, classes, metadata_not_drawn, elements, source_record: $META.figs78.record} ]
    | .pending_downstream |= (.outputs = [] | .figure_files = [] | .recomputed["Figs 7/8"] = {figs78_record: $META.figs78.record} | .waits = (.waits | del(.["Figs 7/8"])))
    | .disclosures += $DP
    | .row_rulings_v2_5.items += [{id: $f.id, rows: $S24, status: ("FINAL (" + $REASON + ")"), cites: [$f.cites[] | anchor(.)], evidence: [$f.evidence[] | ev(.)],
                                  rechecks: [{label: $f.rechecks[0].label, what: $f.rechecks[0].what, cite: anchor($f.rechecks[0].gates_must),
                                              report: ($X.rechecks[$f.rechecks[0].file] | {file, sha256, in_git: false})}],
                                  disclosures: ([$DP[].id] + $AXIS), disclosure_rows: ([$DP[] | {(.id): $S24}] | add),
                                  label_disclosures: [.figure_labels[] | select(.file | test("figs78")) | .file]}]
    | .row_rulings_v2_5.figs78 = {branch: "i", included: true, verified_cite: anchor($f.cites[0]), reason_cite: anchor($f.reason_must),
                                  record: $META.figs78.record, recheck: $META.figs78.recheck,
                                  rule: "FINAL only with 09's GATES line 'figs78 results commit <sha> verified' and a CONFIRMED fresh recheck (09, via 2e): both present"}
    | .flip_groups.group_4.what += "; figs78 branch (i): slot-24 rows 69-74 / 241-248 and Figs 7/8 FINAL (GATES L2430)"
    | .flip_groups.group_4.evidence += ["figs78"]
    | .superseded_fields.fields += ["printed_items.all_rows[slot 24].status", "in_scope_outputs[+Figs 7/8, +slot-24 group]", "preliminary_outputs[-Figs 7/8, -slot-24 group]",
                                    "figure_labels[figs78: FINAL, note, disclosures]", "element_tables[+2 figs78]", "pending_downstream[Figs 7/8 recomputed]",
                                    "in_scope_outputs[Figs 7/8].downstream_tag.recomputed_output", "in_scope_outputs[Figs 7/8].downstream_tag.text"]
  end
# ---- c5 (09 via 2e; both branches): the v2.4 Figs 1/6 mirror defect. P = the Figs 1/6 labels' .reason (both equal); the tag text and the two
# label notes become P + C5_JOINER + their old text; appended after the branch (i) stage so earlier superseded_fields indices do not move ----
| ([.figure_labels[] | select(.output == "Figs 1/6") | .reason]) as $P2
| (if ($P2 | length) != 2 or $P2[0] != $P2[1] or ($P2[0] | startswith("PRELIMINARY") | not) then error("c5: the Figs 1/6 label reasons") else . end)
| $P2[0] as $P
| (if ([.preliminary_outputs[] | select(.name == "Figs 1/6") | .downstream_tag.text | select(type == "string" and startswith("FINAL"))] | length) != 1
      or ([.figure_labels[] | select(.output == "Figs 1/6") | .note | select(type == "string" and startswith("FINAL"))] | length) != 2
   then error("c5: the Figs 1/6 tag text / notes are not the v2.4 FINAL-prefixed strings") else . end)
| ([.figure_labels | to_entries[] | select(.value.output == "Figs 1/6") | .key]) as $LI
| .preliminary_outputs |= map(if .name == "Figs 1/6" then .downstream_tag.text = ($P + C5_JOINER + .downstream_tag.text) else . end)
| .figure_labels |= map(if .output == "Figs 1/6" then .note = ($P + C5_JOINER + .note) else . end)
| .superseded_fields.fields += (["preliminary_outputs[Figs 1/6].downstream_tag.text (v2.4 erratum)"] + [$LI[] | "figure_labels[\(.)].note (Figs 1/6; v2.4 erratum)"])
