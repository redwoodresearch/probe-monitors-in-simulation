# Synthetic keys-only record shaped like map_figs78_a1.py's output (synthetic cell ids; no value), for testing gate_figs78_out.jq only.
# jq -n -f toy_gate_record.jq  -> predicted: the gate returns true on it; each mutation in toy_gate_negatives.zsh -> false.
def NOTDRAWN: "disclosure only (not drawn)";
def U: "pooled_all_origins (union stream)";
def LABELS: ["FIBS|uncut C", "harmfulness|uncut C", "high-stakes|uncut C", "own_all_rs_uncut_L48 C", "refusal|uncut C", "sol_a_uncut C"];
def ok($fam; $key): {a1_family: $fam, a1_key: $key, a1_rows: 1, a1_verdict: true, a1_match: true, a1_kind_ok: true, value_identity: true, status: "FINAL"};
def meta: [{source: "values", value_index: 0, input: "pertask", key_path: "bootstrap/sets/pooled_all_origins", id: "values/bootstrap/sets/pooled_all_origins",
            class: "metadata: bootstrap set descriptor (not drawn; disclosure only)", status: NOTDRAWN},
           {source: "values", value_index: 1, input: "anchor", key_path: "gate/pass", id: "values/gate/pass", class: "metadata: gate boolean (not drawn; disclosure only)", status: NOTDRAWN}];
def r2($i; $lab; $q): ("union_stream/lead/\($lab)/sets/\(U)/r2/\($q)") as $kp
  | [({source: "values", value_index: $i, input: "anchor", key_path: $kp, id: "values/\($kp)/point", component: "point",
       class: "union-stream r2 point -> a1 anchor_pt (pooled_all_origins point)"} + ok("anchor_pt"; "lead|\($lab)|pooled_all_origins|r2.\($q)")),
     ({source: "values", value_index: $i, input: "anchor", key_path: $kp, id: "values/\($kp)/ci95[0]", component: "ci95[0]",
       class: "union-stream r2 interval end -> a1 anchor_ci"} + ok("anchor_ci"; "lead|\($lab)|union|r2.\($q).ci95[0]")),
     ({source: "values", value_index: $i, input: "anchor", key_path: $kp, id: "values/\($kp)/ci95[1]", component: "ci95[1]",
       class: "union-stream r2 interval end -> a1 anchor_ci"} + ok("anchor_ci"; "lead|\($lab)|union|r2.\($q).ci95[1]"))];
def nc($i; $lab): ("union_stream/lead/\($lab)/sets/\(U)/n_cells") as $kp
  | {source: "values", value_index: $i, input: "anchor", key_path: $kp, id: "values/\($kp)", component: "value",
     class: "union-stream n_cells -> a1 anchor_ci"} + ok("anchor_ci"; "lead|\($lab)|union|n_cells");
def tally: reduce .[] as $e ({}; .[$e.class] = ((.[$e.class] // {n: 0, FINAL: 0, PRELIMINARY: 0, (NOTDRAWN): 0}) | .n += 1 | .[$e.status] += 1));
def table($file; $els): {file: ("experiments/x/" + $file), commit: ("a" * 40), sha256: ("b" * 64), n_elements: ($els | length),
    n_drawn: ([$els[] | select(.status != NOTDRAWN)] | length), classes: ($els | tally),
    expected_class_counts: ($els | group_by(.class) | map({(.[0].class): length}) | add), inventory_complete: true, ids_unique: true,
    elements_all_final: true, metadata_not_drawn: ["bootstrap/sets/pooled_all_origins", "gate/pass"], elements: $els, figure_status: "FINAL"};
"own_all_rs_uncut_L48 C" as $L7
| "S_old (all-origins draws) [pooled: pooled_all_origins]" as $O
| (meta + r2(2; $L7; "yardstick") + r2(3; $L7; "main") + [nc(4; $L7)]
   + [range(0; 1308) as $c | ("run1", "run2", "sim") as $a | {source: "scatter_points", label: $L7, cell: "cell_\($c)", origin: $O, arm: $a,
       id: "scatter/\($L7)/cell_\($c)/\($O)/\($a)", class: "Fig 7 scatter coordinate (Stage D max_minus_tau) -> a1 stage_d_pt"}
       + ok("stage_d_pt"; "\($L7)|S_old|cell_\($c)|\($a)|max_minus_tau")]
   + [{source: "n_points", label: $L7, id: "n_points/\($L7)", class: "point count -> a1 anchor_pt lead pooled_all_origins n_cells"}
      + ok("anchor_pt"; "lead|\($L7)|pooled_all_origins|n_cells")]
   + [("main", "yardstick") as $p | {source: "drawn_points_per_panel", label: $L7, panel: $p, id: "drawn/\($L7)/\($p)",
       class: "drawn-point count -> a1 anchor_pt lead pooled_all_origins n_cells"} + ok("anchor_pt"; "lead|\($L7)|pooled_all_origins|n_cells")]) as $f7
| (meta + ([range(0; 6) as $j | r2(2 + $j; LABELS[$j]; "yardstick"), r2(2 + $j; LABELS[$j]; "main")] | add) + [range(0; 6) as $j | nc(20 + $j; LABELS[$j])]) as $f8
| {stamp: "2026-10-20_000000", app: "toy", exit: "OK",
   inputs: (["anchor_followup24_2026-10-20_141527.json", "cleanroom_anchor_2026-10-20_135936.json", "fig_pertask_onpolicy_figs78_routecA_source.json",
             "fig_pertask_summary_figs78_routecA_source.json", "provenance_v84_final17_0141.jsonl"] | map({(.): {sha256: ("c" * 64), source: (("d" * 40) + ":experiments/x")}}) | add),
   a1_values_file: {volume: "v", path: "p", sha256: ("e" * 64), mount: "read-only"},
   a1_duplicate_keys_by_family: {anchor_ci: 0, anchor_diag: 0, anchor_pt: 0, rc1b_24: 0, rc1b_24_display: 0, stage_d_pt: 0, stage_d_tau: 0},
   gates: (["inputs_sha", "a1_values_sha", "real_by_family_equal", "real_n_verdict_equal", "real_printed10_equal", "figs78_inputs_are_a1_pins",
            "element_table_complete_and_final:fig_pertask_onpolicy_figs78_routecA_source.json", "element_table_complete_and_final:fig_pertask_summary_figs78_routecA_source.json",
            "slot24_14_rows_10_quantities_all_final", "negative_controls_fail_as_predicted"] | map({name: ., ok: true})),
   element_tables: [table("fig_pertask_onpolicy_figs78_routecA_source.json"; $f7), table("fig_pertask_summary_figs78_routecA_source.json"; $f8)],
   slot24_rows: ([[69, "24.n_cells.lead.own_all_rs_uncut_L48 C"], [70, "24.n_cells.lead.own_all_rs_uncut_L48 C"], [71, "24.r2.lead.own_all_rs_uncut_L48 C.yardstick"],
                  [72, "24.r2.yardstick_halfwidth.lead.own_all_rs_uncut_L48 C"], [73, "24.n_cells.lead.own_all_rs_uncut_L48 C"], [74, "24.r2.lead.own_all_rs_uncut_L48 C.yardstick"],
                  [241, "24.r2.lead.own_all_rs_uncut_L48 C.yardstick"], [242, "24.r2.lead.own_all_rs_uncut_L48 C.main"],
                  [243, "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick"], [244, "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]"],
                  [245, "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]"], [246, "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick"],
                  [247, "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]"],
                  [248, "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]"]]
                 | map(. as [$row, $key] | ($key | sub("\\.ci95\\[[01]\\]$"; "")) as $slot | {row: $row, provenance_row_unique: true, record_is_rc1b_600f63614: true, record_sha_is_a1_pin: true, slot: $slot, component: (if $row <= 74 then null elif ($key | endswith(".ci95[0]")) then "interval_low" elif ($key | endswith(".ci95[1]")) then "interval_high" else "point" end), a1_key: $key,
                        in_a1_printed10: true, a1_value_rows: 1, a1_value_match: true, a1_display_rows: 1, a1_display_match: true, status: "FINAL"}
                        + (if $row == 72 then {halfwidth_followup: {identified_readings_is_half_range: true, ident_match: true, all_pairs_match: true}} else {} end))),
   slot24_quantities: ["24.n_cells.lead.own_all_rs_uncut_L48 C", "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",
     "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]", "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]",
     "24.r2.lead.own_all_rs_uncut_L48 C.main", "24.r2.lead.own_all_rs_uncut_L48 C.yardstick",
     "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",
     "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[0]",
     "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[1]", "24.r2.yardstick_halfwidth.lead.own_all_rs_uncut_L48 C"],
   negative_controls: {NC1_planted_key_absent: {predicted: "p", as_predicted: true}, NC2_a1_match_flipped: {predicted: "p", as_predicted: true}}}
