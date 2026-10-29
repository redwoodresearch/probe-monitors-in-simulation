# Companion to gate_figs78_out.jq (fresh review N2), r2 (Codex r7 REQUIRED 3-5, OPTIONAL 14-15): validate a STOP_TO_09 / PRELIMINARY record
# from map_figs78_a1.py and list, by key, every element that is not FINAL, so a PRELIMINARY outcome can be told apart from a broken record.
# Keys, booleans and counts only. Output: {valid, kind, exit, gates_false, prelim: {file: [element ids]}, slot24_not_final: [rows],
# figure_status: {file: status}, n_prelim}. Predictions frozen first: PREDICTIONS_gate_figs78_prelim_r2_2026-10-21_0055.md (e3fcd884…).
# Lines 7-111 of this file are gate_figs78_out.jq lines 6-110, byte for byte (its constants and closed-schema helpers; diff-checked). r3: see PREDICTIONS r3.
# The RUN_MEs run guard_one_object.jq (exactly one JSON object) before this file.
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
# ---- companion r2: the closed schema for a STOP_TO_09 record (as the main gate's schema_ok, plus PRELIMINARY element shapes) ----
def ROSTER: ["inputs_sha", "a1_values_sha", "real_by_family_equal", "real_n_verdict_equal", "real_printed10_equal", "figs78_inputs_are_a1_pins",
  "element_table_complete_and_final:fig_pertask_onpolicy_figs78_routecA_source.json",
  "element_table_complete_and_final:fig_pertask_summary_figs78_routecA_source.json",
  "slot24_14_rows_10_quantities_all_final", "negative_controls_fail_as_predicted"] | sort;
def S24_ROWS: [69, 70, 71, 72, 73, 74, 241, 242, 243, 244, 245, 246, 247, 248];
def TOP: ["a1_duplicate_keys_by_family", "a1_values_file", "app", "element_tables", "exit", "gates", "inputs", "negative_controls", "slot24_quantities", "slot24_rows", "stamp"];
def NUM_FIELDS: ["value_index", "a1_rows", "row", "a1_value_rows", "a1_display_rows", "n_elements", "n_drawn", "n", "FINAL", "PRELIMINARY", NOTDRAWN];
# PRELIMINARY element shapes the mapper can emit: map_one's no-unique-row form (no a1_kind_ok) or its not-ok form, an UNCLASSIFIED values
# entry, or a scatter point whose origin is not one of a1's set keys. Each carries a string reason.
def el_prelim:
  (.reason | str)
  and (if .class == "UNCLASSIFIED" then .source == "values" and (keys == (["class", "id", "input", "key_path", "reason", "source", "status", "value_index"] | sort))
         and .id == "values/\(.value_index)"
       elif .source == "scatter_points" and (has("a1_family") | not) then
         (keys == (["arm", "cell", "class", "id", "label", "origin", "reason", "source", "status"] | sort)) and .class == C_SC
         and (.arm as $a | ["run1", "run2", "sim"] | index([$a]) != null) and .id == "scatter/\(.label)/\(.cell)/\(.origin)/\(.arm)"
       else (del(.reason) | (if has("a1_kind_ok") then . else .a1_kind_ok = false end) | el_exact and class_ok) end);
def el_p:
  obj_ok(EL_TYPES)
  and (if .status == "FINAL" then el_exact and class_ok and (has("reason") | not)
         and .a1_rows == 1 and .a1_verdict == true and .a1_match == true and .a1_kind_ok == true and .value_identity == true
       elif .status == "PRELIMINARY" then el_prelim
       elif .status == NOTDRAWN then el_exact and class_ok and .source == "values" and (.class | startswith("metadata: "))
       else false end);
def schema_p:
  (keys == TOP)
  and (.stamp | str) and (.app | str) and (.exit | str)
  and (.inputs | type == "object" and keys == INPUTS and all(.[]; type == "object" and keys == ["sha256", "source"] and (.sha256 | test("^[0-9a-f]{64}$")) and (.source | test("^[0-9a-f]{40}:experiments/"))))
  and (.a1_values_file | type == "object" and keys == ["mount", "path", "sha256", "volume"] and all(.[]; str))
  and (.a1_duplicate_keys_by_family | type == "object" and keys == FAMS and all(.[]; int))
  and (.gates | type == "array" and all(.[]; type == "object" and keys == ["name", "ok"] and (.name | str) and (.ok | boolean)))
  and (.negative_controls | type == "object" and keys == ["NC1_planted_key_absent", "NC2_a1_match_flipped"]
       and all(.[]; type == "object" and keys == ["as_predicted", "predicted"] and (.predicted | str) and (.as_predicted | boolean)))
  and (.slot24_quantities | type == "array" and all(.[]; str))
  and (.slot24_rows | type == "array" and all(.[]; obj_ok(S24_TYPES) and s24_exact and (.status == "FINAL" or .status == "PRELIMINARY")
        and (if has("halfwidth_followup") then (.halfwidth_followup | keys == ["all_pairs_match", "ident_match", "identified_readings_is_half_range"] and all(.[]; boolean)) else true end)))
  and (.element_tables | type == "array" and all(.[]; type == "object" and
        keys == ["classes", "commit", "elements", "elements_all_final", "expected_class_counts", "figure_status", "file", "ids_unique", "inventory_complete",
                 "metadata_not_drawn", "n_drawn", "n_elements", "sha256"]
        and (.file | str) and (.commit | test("^[0-9a-f]{40}$")) and (.sha256 | test("^[0-9a-f]{64}$")) and (.n_elements | int) and (.n_drawn | int)
        and (.inventory_complete | boolean) and (.ids_unique | boolean) and (.elements_all_final | boolean)
        and (.metadata_not_drawn | type == "array" and all(.[]; str))
        and (.expected_class_counts | type == "object" and all(.[]; int))
        # r3 (Codex r8b item 5): the pre-stated counts for this file exactly; class keys = those plus at most UNCLASSIFIED
        and ((.file | split("/") | last) as $bn | (EXPECT | has($bn)) and .expected_class_counts == EXPECT[$bn]
             and ((.classes | keys) - (EXPECT[$bn] | keys) - ["UNCLASSIFIED"]) == [])
        and (.classes | type == "object" and all(.[]; type == "object" and keys == (["FINAL", "PRELIMINARY", NOTDRAWN, "n"] | sort) and all(.[]; int)))
        and (.elements | type == "array" and length > 0 and all(.[]; el_p))
        and ([.elements[] | .id] | (unique | length) == length)
        and .classes == (.elements | histogram)))
  # every number is an integer under a count / index field; no string leaf parses as a number
  # r3 (Codex r8b item 18): bind the leaf's path first; r2 tested the allowlist's own last entry
  and all(paths(type == "number"); . as $p | ($p[-1] | type) == "string"
          and ((NUM_FIELDS | index([$p[-1]])) != null or (($p | length) >= 2 and $p[-2] == "expected_class_counts")
               or (($p | length) == 2 and $p[0] == "a1_duplicate_keys_by_family")))
  and all(.. | numbers; . == floor)
  and ([.. | strings | select((tonumber? // null) != null)] | length) == 0;
. as $r
| if ($r | type) != "object" then {valid: false, kind: "not an object"}
  elif $r.exit == "PLUMBING" then {valid: false, kind: "PLUMBING", exit: $r.exit, gates_false: [($r.gates // [])[] | select(.ok != true) | .name? | strings]}
  elif $r.exit != "STOP_TO_09" then {valid: false, kind: "not STOP_TO_09"}
  else
    (($r | schema_p)
     and ([$r.gates[] | .name] | sort) == ROSTER
     and ($r.element_tables | length) == 2
     and ([$r.element_tables[] | .file | split("/") | last] | sort) == (EXPECT | keys)
     and all($r.element_tables[]; .figure_status == "PRELIMINARY")
     and ([$r.slot24_rows[] | .row] == S24_ROWS)
     # r3 (Codex r8b item 16): the mapper returns PLUMBING when any of these 5 fails, so a STOP_TO_09 record must have them all true
     and all($r.gates[] | select(.name as $n | ["inputs_sha", "a1_values_sha", "real_by_family_equal", "real_n_verdict_equal", "real_printed10_equal"] | index([$n]) != null); .ok == true)
     and any($r.gates[]; .ok == false)) as $valid
    | if $valid then
        {valid: true, kind: "STOP_TO_09", exit: $r.exit,
         gates_false: [$r.gates[] | select(.ok != true) | .name],
         prelim: ([$r.element_tables[] | {((.file | split("/") | last)): [.elements[] | select(.status == "PRELIMINARY") | .id]}] | add),
         slot24_not_final: [$r.slot24_rows[] | select(.status != "FINAL") | .row],
         figure_status: ([$r.element_tables[] | {((.file | split("/") | last)): .figure_status}] | add),
         n_prelim: ([$r.element_tables[].elements[] | select(.status == "PRELIMINARY")] | length)}
      else {valid: false, kind: "STOP_TO_09"} end
  end
