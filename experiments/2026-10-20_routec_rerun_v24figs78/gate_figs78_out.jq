# Result gate r3 for map_figs78_a1.py's keys-only record (jq -e; prints true or false only). Counts are pre-stated in PRESTATED §5 /
# AMEND2 from the source files' KEYS. r2 (Codex r1 REQUIRED 6-8): totals recomputed from the elements against constants here; slot-24
# evidence flags rechecked. r3 (Codex r2 REQUIRED 8, 12): a CLOSED schema (exact key sets and types for every object; no string leaf
# that parses as a number), and the reported inventory fields (n_elements, n_drawn, metadata_not_drawn, classes, expected_class_counts,
# inventory_complete, ids_unique, elements_all_final) must equal what the elements give.
def NOTDRAWN: "disclosure only (not drawn)";
def EXPECT: {
  "fig_pertask_onpolicy_figs78_routecA_source.json": {
    "union-stream r2 point -> a1 anchor_pt (pooled_all_origins point)": 2, "union-stream r2 interval end -> a1 anchor_ci": 4,
    "union-stream n_cells -> a1 anchor_ci": 1, "Fig 7 scatter coordinate (Stage D max_minus_tau) -> a1 stage_d_pt": 3924,
    "point count -> a1 anchor_pt lead pooled_all_origins n_cells": 1, "drawn-point count -> a1 anchor_pt lead pooled_all_origins n_cells": 2,
    "metadata: gate boolean (not drawn; disclosure only)": 1, "metadata: bootstrap set descriptor (not drawn; disclosure only)": 1},
  "fig_pertask_summary_figs78_routecA_source.json": {
    "union-stream r2 point -> a1 anchor_pt (pooled_all_origins point)": 12, "union-stream r2 interval end -> a1 anchor_ci": 24,
    "union-stream n_cells -> a1 anchor_ci": 6,
    "metadata: gate boolean (not drawn; disclosure only)": 1, "metadata: bootstrap set descriptor (not drawn; disclosure only)": 1}};
def PRINTED10: ["24.n_cells.lead.own_all_rs_uncut_L48 C", "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",
  "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]", "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]",
  "24.r2.lead.own_all_rs_uncut_L48 C.main", "24.r2.lead.own_all_rs_uncut_L48 C.yardstick",
  "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",
  "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]",
  "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]",
  "24.r2.yardstick_halfwidth.lead.own_all_rs_uncut_L48 C"] | sort;
def INPUTS: ["anchor_followup24_2026-10-20_141527.json", "cleanroom_anchor_2026-10-20_135936.json", "fig_pertask_onpolicy_figs78_routecA_source.json",
             "fig_pertask_summary_figs78_routecA_source.json", "provenance_v84_final17_0141.jsonl"];
# ---- closed schema helpers ----
def int: type == "number" and . == floor;
def str: type == "string";
def boolean: type == "boolean";
def bool_or_null: type == "boolean" or type == "null";
def str_or_null: type == "string" or type == "null";
def keys_in($allowed): (keys - $allowed) == [];
def EL_TYPES: {source: "str", value_index: "int", input: "str", key_path: "str", id: "str", class: "str", status: "str", component: "str",
               a1_family: "str", a1_key: "str", a1_rows: "int", a1_verdict: "bool_or_null", a1_match: "bool_or_null", a1_kind_ok: "boolean",
               value_identity: "bool_or_null", reason: "str", label: "str", cell: "str", origin: "str", arm: "str", panel: "str"};
def S24_TYPES: {row: "int", provenance_row_unique: "boolean", record_is_rc1b_600f63614: "boolean", record_sha_is_a1_pin: "boolean", slot: "str_or_null",
                component: "str_or_null", a1_key: "str_or_null", in_a1_printed10: "boolean", a1_value_rows: "int", a1_value_match: "boolean",
                a1_display_rows: "int", a1_display_match: "boolean", halfwidth_followup: "object", status: "str"};
def typed($t): if $t == "str" then str elif $t == "int" then int elif $t == "boolean" then boolean elif $t == "bool_or_null" then bool_or_null
               elif $t == "str_or_null" then str_or_null elif $t == "object" then type == "object" else false end;
def obj_ok($types): type == "object" and keys_in($types | keys) and (to_entries | all(.key as $k | .value | typed($types[$k])));
# ---- r4 (Codex r3 REQUIRED 8, 12): exact field sets per element type, identifier patterns, closed family keys, full histograms ----
def A1F: ["a1_family", "a1_key", "a1_rows", "a1_verdict", "a1_match", "a1_kind_ok", "value_identity"];
def LABELS: ["FIBS|uncut C", "harmfulness|uncut C", "high-stakes|uncut C", "own_all_rs_uncut_L48 C", "refusal|uncut C", "sol_a_uncut C"];
def ORIGIN_RE: "^S_(old|new|newmains|paper493) \\(all-origins draws\\) \\[pooled: pooled_all_origins\\]$";
def FAMS: ["anchor_ci", "anchor_diag", "anchor_pt", "rc1b_24", "rc1b_24_display", "stage_d_pt", "stage_d_tau"];
def lab_ok: . as $l | LABELS | index([$l]) != null;
def el_exact:
  if .source == "values" and (.class | startswith("metadata: ")) then
    (keys == (["source", "value_index", "input", "key_path", "id", "class", "status"] | sort))
    and ((.input == "anchor" and .key_path == "gate/pass" and .id == "values/gate/pass" and .class == "metadata: gate boolean (not drawn; disclosure only)")
         or (.input == "pertask" and .key_path == "bootstrap/sets/pooled_all_origins" and .id == "values/bootstrap/sets/pooled_all_origins"
             and .class == "metadata: bootstrap set descriptor (not drawn; disclosure only)"))
  elif .source == "values" then
    (keys == ((["source", "value_index", "input", "key_path", "id", "component", "class", "status"] + A1F) | sort)) and .input == "anchor"
    and (.key_path | test("^union_stream/lead/[^/]+/sets/pooled_all_origins \\(union stream\\)/(r2/(yardstick|main)|n_cells)$"))
    and ((.key_path | split("/")[2]) | lab_ok)
    and (if .component == "point" then .a1_family == "anchor_pt" and .id == "values/\(.key_path)/point"
           and .a1_key == "lead|\(.key_path | split("/")[2])|pooled_all_origins|r2.\(.key_path | split("/")[6])"
         elif .component == "ci95[0]" or .component == "ci95[1]" then .a1_family == "anchor_ci" and .id == "values/\(.key_path)/\(.component)"
           and .a1_key == "lead|\(.key_path | split("/")[2])|union|r2.\(.key_path | split("/")[6]).\(.component)"
         elif .component == "value" then .a1_family == "anchor_ci" and .id == "values/\(.key_path)" and (.key_path | endswith("/n_cells"))
           and .a1_key == "lead|\(.key_path | split("/")[2])|union|n_cells"
         else false end)
  elif .source == "scatter_points" then
    (keys == ((["source", "label", "cell", "origin", "arm", "id", "class", "status"] + A1F) | sort))
    and .label == "own_all_rs_uncut_L48 C" and (.cell | test("^[A-Za-z0-9_.|:@ -]{1,64}$")) and (.origin | test(ORIGIN_RE))
    and (.arm as $a | ["run1", "run2", "sim"] | index([$a]) != null) and .a1_family == "stage_d_pt"
    and .id == "scatter/\(.label)/\(.cell)/\(.origin)/\(.arm)"
    and .a1_key == "\(.label)|\(.origin | split(" ")[0])|\(.cell)|\(.arm)|max_minus_tau"
  elif .source == "n_points" then
    (keys == ((["source", "label", "id", "class", "status"] + A1F) | sort)) and .label == "own_all_rs_uncut_L48 C" and .id == "n_points/\(.label)"
    and .a1_family == "anchor_pt" and .a1_key == "lead|\(.label)|pooled_all_origins|n_cells"
  elif .source == "drawn_points_per_panel" then
    (keys == ((["source", "label", "panel", "id", "class", "status"] + A1F) | sort)) and .label == "own_all_rs_uncut_L48 C"
    and (.panel as $p | ["main", "yardstick"] | index([$p]) != null) and .id == "drawn/\(.label)/\(.panel)"
    and .a1_family == "anchor_pt" and .a1_key == "lead|\(.label)|pooled_all_origins|n_cells"
  else false end;
def histogram: reduce .[] as $e ({}; .[$e.class] = ((.[$e.class] // {n: 0, FINAL: 0, PRELIMINARY: 0, (NOTDRAWN): 0}) | .n += 1 | .[$e.status] += 1));
# ---- r5 (Codex r4 REQUIRED 8): class bound to source / component; slot-24 rows with exact fields and slot / component / a1_key consistent ----
def C_PT: "union-stream r2 point -> a1 anchor_pt (pooled_all_origins point)";
def C_CI: "union-stream r2 interval end -> a1 anchor_ci";
def C_NC: "union-stream n_cells -> a1 anchor_ci";
def C_SC: "Fig 7 scatter coordinate (Stage D max_minus_tau) -> a1 stage_d_pt";
def C_NP: "point count -> a1 anchor_pt lead pooled_all_origins n_cells";
def C_DP: "drawn-point count -> a1 anchor_pt lead pooled_all_origins n_cells";
def class_ok:
  if .source == "values" and (.class | startswith("metadata: ")) then true
  elif .source == "values" then (if .component == "point" then .class == C_PT elif .component == "value" then .class == C_NC else .class == C_CI end)
  elif .source == "scatter_points" then .class == C_SC elif .source == "n_points" then .class == C_NP elif .source == "drawn_points_per_panel" then .class == C_DP
  else false end;
def S24_KEY: {"69": "24.n_cells.lead.own_all_rs_uncut_L48 C", "70": "24.n_cells.lead.own_all_rs_uncut_L48 C", "71": "24.r2.lead.own_all_rs_uncut_L48 C.yardstick",
  "72": "24.r2.yardstick_halfwidth.lead.own_all_rs_uncut_L48 C", "73": "24.n_cells.lead.own_all_rs_uncut_L48 C", "74": "24.r2.lead.own_all_rs_uncut_L48 C.yardstick",
  "241": "24.r2.lead.own_all_rs_uncut_L48 C.yardstick", "242": "24.r2.lead.own_all_rs_uncut_L48 C.main",
  "243": "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick", "244": "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]",
  "245": "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]", "246": "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",
  "247": "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]",
  "248": "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]"};
def SUFFIX: {"point": "", "interval_low": ".ci95[0]", "interval_high": ".ci95[1]"};
# r6 (Codex r5 REQUIRED 8): the component is fixed per row, and the slot is the canonical base key (no interval suffix)
def S24_COMP: {"69": null, "70": null, "71": null, "72": null, "73": null, "74": null, "241": "point", "242": "point", "243": "point",
  "244": "interval_low", "245": "interval_high", "246": "point", "247": "interval_low", "248": "interval_high"};
def s24_exact:
  (keys == ((["row", "provenance_row_unique", "record_is_rc1b_600f63614", "record_sha_is_a1_pin", "slot", "component", "a1_key", "in_a1_printed10",
              "a1_value_rows", "a1_value_match", "a1_display_rows", "a1_display_match", "status"] + (if .row == 72 then ["halfwidth_followup"] else [] end)) | sort))
  and .a1_key == S24_KEY[.row | tostring] and .component == S24_COMP[.row | tostring]
  and .slot == (.a1_key | sub("\\.ci95\\[[01]\\]$"; ""))
  and (.slot | type) == "string" and (.slot | test("^24\\.(n_cells|r2)\\."))
  and (if .row <= 74 then .component == null and .a1_key == .slot
       else (.component as $c | (SUFFIX | has($c // "none")) and .a1_key == .slot + SUFFIX[$c]) end);
def schema_ok:
  (keys == ["a1_duplicate_keys_by_family", "a1_values_file", "app", "element_tables", "exit", "gates", "inputs", "negative_controls", "slot24_quantities", "slot24_rows", "stamp"])
  and (.stamp | str) and (.app | str) and (.exit | str)
  and (.inputs | keys == INPUTS and all(.[]; keys == ["sha256", "source"] and (.sha256 | test("^[0-9a-f]{64}$")) and (.source | test("^[0-9a-f]{40}:experiments/"))))
  and (.a1_values_file | keys == ["mount", "path", "sha256", "volume"] and all(.[]; str))
  and (.a1_duplicate_keys_by_family | type == "object" and keys == FAMS and all(.[]; int))
  and (.gates | type == "array" and all(.[]; keys == ["name", "ok"] and (.name | str) and (.ok | boolean)))
  and (.negative_controls | keys == ["NC1_planted_key_absent", "NC2_a1_match_flipped"] and all(.[]; keys == ["as_predicted", "predicted"] and (.predicted | str) and (.as_predicted | boolean)))
  and (.slot24_quantities | type == "array" and all(.[]; str))
  and (.slot24_rows | type == "array" and all(.[]; obj_ok(S24_TYPES) and s24_exact
        and (if has("halfwidth_followup") then (.halfwidth_followup | keys == ["all_pairs_match", "ident_match", "identified_readings_is_half_range"] and all(.[]; boolean)) else true end)))
  and (.element_tables | type == "array" and all(.[];
        keys == ["classes", "commit", "elements", "elements_all_final", "expected_class_counts", "figure_status", "file", "ids_unique", "inventory_complete",
                 "metadata_not_drawn", "n_drawn", "n_elements", "sha256"]
        and (.file | str) and (.commit | test("^[0-9a-f]{40}$")) and (.sha256 | test("^[0-9a-f]{64}$")) and (.n_elements | int) and (.n_drawn | int)
        and (.figure_status | str) and (.inventory_complete | boolean) and (.ids_unique | boolean) and (.elements_all_final | boolean)
        and (.metadata_not_drawn | type == "array" and all(.[]; str))
        and (.expected_class_counts | type == "object" and all(.[]; int))
        and (.classes | type == "object" and all(.[]; keys == (["FINAL", "PRELIMINARY", NOTDRAWN, "n"] | sort) and all(.[]; int)))
        and (.elements | type == "array" and all(.[]; obj_ok(EL_TYPES) and el_exact and class_ok))))
  # identifiers: no string leaf anywhere parses as a number (values never travel as strings either)
  and ([.. | strings | select((tonumber? // null) != null)] | length) == 0;
def drawn: select(.status != NOTDRAWN);
def ok_el: .status == "FINAL" and .a1_rows == 1 and .a1_verdict == true and .a1_match == true and .a1_kind_ok == true and .value_identity == true;
. as $r
| ($r | schema_ok) as $schema
| ($r.exit == "OK")
  and $schema
  and ([$r.gates[] | .name] | sort) == ([
        "inputs_sha", "a1_values_sha", "real_by_family_equal", "real_n_verdict_equal", "real_printed10_equal", "figs78_inputs_are_a1_pins",
        "element_table_complete_and_final:fig_pertask_onpolicy_figs78_routecA_source.json",
        "element_table_complete_and_final:fig_pertask_summary_figs78_routecA_source.json",
        "slot24_14_rows_10_quantities_all_final", "negative_controls_fail_as_predicted"] | sort)
  and all($r.gates[]; .ok == true)
  and ($r.element_tables | length) == 2
  and ([$r.element_tables[] | .file | split("/") | last] | sort) == (EXPECT | keys)
  and all($r.element_tables[]; (.file | split("/") | last) as $f
        | ([.elements[] | .class] | group_by(.) | map({(.[0]): length}) | add) as $cnt
        | .figure_status == "FINAL"
        and (.elements | length) > 0
        and $cnt == EXPECT[$f]
        and .expected_class_counts == EXPECT[$f]
        and .classes == (.elements | histogram)
        and .n_elements == (.elements | length)
        and .n_drawn == ([.elements[] | drawn] | length)
        and .n_drawn == ([EXPECT[$f] | to_entries[] | select(.key | startswith("metadata: ") | not) | .value] | add)
        and ([.elements[] | .id] | (unique | length) == length) and .ids_unique == true
        and all(.elements[] | drawn; ok_el) and .elements_all_final == true and .inventory_complete == true
        and ([.elements[] | select(.status == NOTDRAWN) | .key_path] | sort) == ["bootstrap/sets/pooled_all_origins", "gate/pass"]
        and .metadata_not_drawn == ["bootstrap/sets/pooled_all_origins", "gate/pass"]
        and all(.elements[] | select(.status == NOTDRAWN); (.class | startswith("metadata: ")) and .source == "values"))
  and ($r.slot24_rows | length) == 14
  and ([$r.slot24_rows[].row] == [69, 70, 71, 72, 73, 74, 241, 242, 243, 244, 245, 246, 247, 248])
  and all($r.slot24_rows[]; .status == "FINAL" and .provenance_row_unique == true and .record_is_rc1b_600f63614 == true and .record_sha_is_a1_pin == true
          and .in_a1_printed10 == true and .a1_value_rows == 1 and .a1_value_match == true and .a1_display_rows == 1 and .a1_display_match == true)
  and ([$r.slot24_rows[] | select(has("halfwidth_followup")) | .row] == [72])
  and all($r.slot24_rows[] | select(.row == 72) | .halfwidth_followup; .identified_readings_is_half_range == true and .ident_match == true and .all_pairs_match == true)
  and ([$r.slot24_rows[].a1_key] | unique) == PRINTED10
  and ($r.slot24_quantities | sort) == PRINTED10
  and all($r.negative_controls[]; .as_predicted == true)
