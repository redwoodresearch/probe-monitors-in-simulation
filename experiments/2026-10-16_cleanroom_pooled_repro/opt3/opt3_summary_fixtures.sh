#!/bin/bash
# The summary gate's frozen fixtures (OPT3 addenda 8 §6, 8a, 8b §2, 8c §3 and §6, 8d §4, 8e §3, 8f §4, 9a §4, 9b §3): the 100 cases of cleanroom_boot_opt3.py _summary_fixtures, on synthetic
# JSON, with the committed node lists (first node of each list). jq only; no summary, record or value is read. Usage, from the folder
# experiments/2026-10-16_cleanroom_pooled_repro:   bash opt3/opt3_summary_fixtures.sh
# Prints "FIXTURES OK (100/100 as predicted)" or exits 3 at the first case whose outcome differs from its prediction.
set -euo pipefail
GATE=opt3/opt3_summary_shape.jq
NA=opt3/opt3_nodes_A.txt
NB=opt3/opt3_nodes_fig4.txt
PRED=opt3/PREDICTIONS_r2_11.json
SLOTS=opt3/opt3_d5_slots.txt
NL=$(wc -l < cleanroom_boot_opt3.py | tr -d " ")  # stop_line's bounds: the script's and the reducer's line counts (the results RUN_ME checks both unmodified)
NLR=$(wc -l < cleanroom_reduce_v2.py | tr -d " ")
NLJ="{\"boot\": $NL, \"reducer\": $NLR}"
g() { jq -e -s --rawfile nodesA "$NA" --rawfile nodesB "$NB" --slurpfile pred "$PRED" --rawfile slots "$SLOTS" --argjson nlines "$NLJ" -f "$GATE" > /dev/null 2>&1; }
A1=$(head -n 1 "$NA")
B1=$(head -n 1 "$NB")
NAN0='{"d7": 0, "d7_units_ok": 0, "d6": 0, "empty": 0, "other": 0}'
BLANK='{"record": "A", "stop": "STOP: not started", "stop_file": null, "stop_line": null, "stop_ids": [], "input_file": null, "stages": [], "thread_env_pre": {}, "blas_pools_n": null,
 "blas_threads_max": null, "selftests_ok": null, "selftests_failed": [], "calib_sha_ok": null, "c5_trials": null, "c5_compared": null,
 "c5_skipped_d7": null, "n_degenerate": null, "mains_outside_universe": null, "draw0_ok": null, "draw0_failures": [], "c0_scale": null, "c0_ok": null,
 "d1_identified": null, "d1_reading": null, "d1_s1_equals_s2": null, "d1_candidate_failures": [], "pilot_projected_s": null,
 "pilot_elapsed_s": null, "pilot_rss_peak_mib": null, "pilot_rss_child_mib": null, "pilot_mem_projected_mib": null,
 "pilot_mem_source": null, "c1_ok": null, "c1_n_labels": null, "c1_n_labels_ok": null, "c1_gap_compared": null, "c1_gap_mismatch": null,
 "c1_gap_flip": null, "c1_ok_mask_diffs": null, "c1_first_failures": [], "c2_method": null, "c2_n_nodes": null, "c4_n": null,
 "c4_n_stopped": null, "n_nodes": null, "n_reproduced": null, "n_reproduced_flip": null, "n_not_reproduced": null,
 "n_not_like_for_like": null, "n_unresolved": null, "n_incomplete_draws": null, "mc_n_agree": null, "nan_by_node": {}, "d5_status": null,
 "d5_pair": null, "d5_ok": null, "d5_not_reproduced": [], "d5_unresolved": [], "n_d5_flip_sensitive": null, "verdict": "STOP", "failing_nodes": [], "values_dir": null,
 "values_sha256_node_series": null, "values_sha256_compare": null, "seconds": 0, "d9": null}'
STAGES='["code_pins", "selftests", "pins", "manifests", "nodes", "resolve", "c4", "c2", "draw0", "c0", "d1", "pilot", "draws", "c1", "compare", "values", "d5"]'
FULL=$(jq -cn --argjson b "$BLANK" '{launch_epoch: 1792345678.123, launch_id: "1792345678123", run_mode: "AB", fan_out: 2, driver_stop: null,
  workers: [$b, ($b | .record = "B")]}')
EXACT=$(jq -cn --argjson b "$BLANK" --argjson st "$STAGES" --argjson z "$NAN0" --arg a "$A1" --arg bn "$B1" '
  def ex($rec; $node; $stg): $b | .record = $rec | .stop = null | .stages = $stg | .selftests_ok = true | .draw0_ok = true
    | .c0_ok = true | .c0_scale = "fraction" | .c1_ok = true | .c4_n = 5 | .c4_n_stopped = 5 | .n_nodes = 1 | .n_reproduced = 1
    | .nan_by_node = {($node): $z} | .verdict = "INTERVALS REPRODUCED (EXACT)" | .values_dir = "/o/opt3_\($rec)_20261021_120000_0123abcd";
  {launch_epoch: 1792345678.123, launch_id: "1792345678123", run_mode: "AB", fan_out: 2, driver_stop: null,
   workers: [(ex("A"; $a; $st) | .d5_status = "fit"), ex("B"; $bn; ($st | map(select(. != "d5"))))]}')
STOPPED='{"launch_epoch": 1792345678.123, "launch_id": "1792345678123", "run_mode": "AB", "fan_out": 0, "driver_stop": "STOP: summary schema", "workers": []}'
n=0
# want: ok | fail; $1 name, $2 basis JSON, $3 jq mutation
chk() {
  local got
  if printf '%s' "$2" | jq -c "$3" | g; then got=ok; else got=fail; fi
  [ "$got" = "$4" ] || { echo "STOP: fixture $1 gave $got, predicted $4"; exit 3; }
  n=$((n + 1))
}
# The real node lists' sizes in the EXACT rule: n_nodes must equal each list's length, so the EXACT basis uses those lengths.
LA=$(grep -c . "$NA"); LB=$(grep -c . "$NB")
EXACT=$(printf '%s' "$EXACT" | jq -c --argjson la "$LA" --argjson lb "$LB" --rawfile na "$NA" --rawfile nb "$NB" --argjson z "$NAN0" '
  .workers[0] |= (.n_nodes = $la | .n_reproduced = $la | .nan_by_node = ([$na | split("\n")[] | select(length > 0) | {(.): $z}] | add))
  | .workers[1] |= (.n_nodes = $lb | .n_reproduced = $lb | .nan_by_node = ([$nb | split("\n")[] | select(length > 0) | {(.): $z}] | add))')
chk valid_full "$FULL" '.' ok
chk valid_stop "$STOPPED" '.' ok
chk valid_exact "$EXACT" '.' ok
chk crashed_worker "$FULL" '.workers[0].stop = "STOP: worker exception RemoteError" | .workers[1].stop = "STOP: worker exception other"' ok
chk extra_key "$FULL" '.workers[0].extra = 1' fail
chk float_in_int "$FULL" '.workers[0].c4_n = 1.5 | .workers[0].c4_n_stopped = 1.5' fail
chk int_as_float "$FULL" '.workers[0].seconds = 1.0' fail
chk c4_unequal "$FULL" '.workers[0].c4_n = 3 | .workers[0].c4_n_stopped = 2' fail
chk c4_null_vs_3 "$FULL" '.workers[0].c4_n = null | .workers[0].c4_n_stopped = 3' fail
chk c4_zero "$FULL" '.workers[0].c4_n = 0 | .workers[0].c4_n_stopped = 0' fail
chk compare_without_c4 "$FULL" '.workers[0].stages = ["compare"]' fail
chk verdict_free_text "$FULL" '.workers[0].stop = null | .workers[0].verdict = "NOT REPRODUCED: 0.52"' fail
chk stop_free_text "$FULL" '.workers[0].stop = "STOP: 0.52"' fail
chk stop_exception_unknown "$FULL" '.workers[0].stop = "STOP: exception Foo"' fail
chk stop_null_verdict_stop "$FULL" '.workers[0].stop = null' fail
chk workers_order "$FULL" '.workers |= reverse' fail
chk fan_out_false "$STOPPED" '.fan_out = false' fail
chk fan_out_float "$STOPPED" '.fan_out = 0.0' fail
chk launch_id_free "$FULL" '.launch_id = "0.52"' fail
chk values_dir_free "$EXACT" '.workers[0].values_dir = "/o/0.52"' fail
chk id_free_text "$FULL" '.workers[0].d1_candidate_failures = ["0.52"]' fail
chk id_valid "$FULL" '.workers[0].d1_candidate_failures = ["#3 joint FIBS|uncut C tau_B@0.01@draw7"]' ok
chk gap_id_valid "$(printf '%s' "$FULL" | jq -c --arg a "$A1" '.workers[0].c1_first_failures = ["gap " + $a + "@draw17"]')" '.' ok
chk gap_id_unknown_node "$FULL" '.workers[0].c1_first_failures = ["gap zz@draw17"]' fail
chk failing_node_unknown "$FULL" '.workers[0].failing_nodes = ["zz"]' fail
chk nan_missing_node "$FULL" '.workers[0].stages = ["draws"]' fail
chk nan_extra_key "$EXACT" '.workers[0].nan_by_node |= (to_entries | .[0].value.extra = 0 | from_entries)' fail
chk nan_negative "$EXACT" '.workers[0].nan_by_node |= (to_entries | .[0].value.d7 = -1 | from_entries)' fail
chk exact_without_c1 "$EXACT" '.workers[0].c1_ok = false' fail
chk exact_a_d5_ambiguous "$EXACT" '.workers[0].d5_status = "ambiguous"' fail
chk c0_verdict_without_c0_false "$FULL" '.workers[0].stop = null | .workers[0].verdict = "NOT LIKE-FOR-LIKE (C0 scale unidentified)"' fail
chk driver_stop_free "$STOPPED" '.driver_stop = "STOP: x"' fail
chk verdict_with_null_c4 "$FULL" '.workers[0].stop = null | .workers[0].verdict = "UNRESOLVED (stream gate)"' fail
chk stream_gate_with_c4 "$FULL" '.workers[0].stop = null | .workers[0].verdict = "UNRESOLVED (stream gate)" | .workers[0].c4_n = 5 | .workers[0].c4_n_stopped = 5 | .workers[0].stages = ["code_pins", "selftests", "pins", "manifests", "nodes", "c4", "draw0", "c0"]' ok
chk c4_stage_without_counts "$FULL" '.workers[0].stages = ["c4"]' fail
chk counts_without_c4_stage "$FULL" '.workers[0].c4_n = 5 | .workers[0].c4_n_stopped = 5' fail
chk launch_epoch_infinite "$FULL" '.launch_epoch = 1e400' fail
chk planted_selftest "$FULL" '.workers[0].selftests_failed = ["0.52"]' fail
chk selftest_valid "$FULL" '.workers[0].selftests_failed = ["c5.failures"]' ok
chk planted_slot "$FULL" '.workers[0].d5_not_reproduced = ["0.52"]' fail
chk slot_valid "$FULL" '.workers[0].d5_not_reproduced = ["10.halfwidth_C.max_all"]' ok
chk planted_level "$FULL" '.workers[0].draw0_failures = ["FIBS|uncut C:tau_B@0.52"]' fail
chk level_valid "$FULL" '.workers[0].draw0_failures = ["FIBS|uncut C:tau_B@0.01"]' ok
chk stop_line_valid "$FULL" '.workers[0].stop = "STOP: exception KeyError @c4" | .workers[0].stop_file = "boot" | .workers[0].stop_line = 10 | .workers[0].stages = ["code_pins", "selftests", "pins", "manifests", "nodes", "resolve"]' ok
chk stop_line_zero "$FULL" '.workers[0].stop_line = 0' fail
chk stop_line_too_big "$FULL" ".workers[0].stop_line = $((NL + 1))" fail
chk stop_line_float "$FULL" '.workers[0].stop_line = 10.0' fail
chk stop_line_on_non_stop "$EXACT" '.workers[0].stop_line = 10' fail
chk stage_suffix_valid "$FULL" '.workers[0].stop = "STOP: reducer blob @code_pins"' ok
chk stage_suffix_unknown "$FULL" '.workers[0].stop = "STOP: reducer blob @c9"' fail
chk stage_suffix_completed "$FULL" '.workers[0].stop = "STOP: exception KeyError @nodes" | .workers[0].stages = ["code_pins", "selftests", "pins", "manifests", "nodes"]' fail
chk n_degenerate_negative "$FULL" '.workers[0].n_degenerate = -1' fail
chk n_degenerate_float "$FULL" '.workers[0].n_degenerate = 1.0' fail
chk stop_file_valid "$FULL" '.workers[0].stop_file = "reducer" | .workers[0].stop_line = 100' ok
chk stop_file_unknown "$FULL" '.workers[0].stop_file = "q1_eval" | .workers[0].stop_line = 10' fail
chk stop_line_reducer_too_big "$FULL" ".workers[0].stop_file = \"reducer\" | .workers[0].stop_line = $((NLR + 1))" fail
chk stop_line_without_file "$FULL" '.workers[0].stop_line = 10' fail
chk input_file_valid "$FULL" '.workers[0].stop = "STOP: input missing @code_pins" | .workers[0].input_file = "override_S_old_C_2026-10-17_151056_gated.json"' ok
chk input_file_unknown "$FULL" '.workers[0].input_file = "x.json"' fail
chk input_file_without_stop "$EXACT" '.workers[0].input_file = "override_S_old_C_2026-10-17_151056_gated.json"' fail
# addendum 9a §4: the 29 d9 cases. DIAG: diagB, #0 and #2..#6 FAIL with the 7 control ids, #1 MATCH; FULLB: fullB, candidate 1.
CTRL='["FIBS|uncut C tau_B@0.0237137@draw38", "harmfulness|uncut C tau_B@0.0237137@draw38", "high-stakes|uncut C tau_B@0.0421697@draw49",
 "own_all_rs_uncut_L48 C tau_B@0.00562341@draw24", "refusal|uncut C tau_B@0.0237137@draw38", "sol_a_uncut C tau_B@0.00562341@draw24",
 "gap pooled|refusal|uncut C|pooled_all_origins|joint|0.0237137|gap|delta_full@draw38"]'
DIAG=$(jq -cn --argjson b "$BLANK" --argjson ctrl "$CTRL" '
  def ent($i; $m): {cand: $i, joint_match: $m, fixed_match: true, draw0_ok: true, calib_compared: 18000, calib_mom_compared: 4800, calib_mismatch: (if $m then 0 else 6 end),
    gap_compared: 9000, gap_mismatch: (if $m then 0 else 1 end), fixed_compared: 27000, fixed_mismatch: 0,
    first_failures: (if $m then [] else ($ctrl | sort) end)};
  {launch_epoch: 1792345678.123, launch_id: "1792345678123", run_mode: "diagB", fan_out: 1, driver_stop: null,
   workers: [$b | .record = "B" | .stop = null | .c4_n = 5 | .c4_n_stopped = 5 | .verdict = "UNRESOLVED (stream gate)"
     | .stages = ["code_pins", "selftests", "pins", "manifests", "nodes", "resolve", "c4", "c2", "draw0", "c0", "d1", "diag"]
     | .d9 = {run_mode: "diagB", candidate: null, status: "identified", baseline_ok: true, identified: 1, grid_forms_equal: true,
              grid_exact_levels_equal: true, candidates: [range(7) as $i | ent($i; $i == 1)]}]}')
FULLB=$(jq -cn --argjson b "$BLANK" '{launch_epoch: 1792345678.123, launch_id: "1792345678123", run_mode: "fullB", fan_out: 1, driver_stop: null,
   workers: [$b | .record = "B" | .stop = null | .c4_n = 5 | .c4_n_stopped = 5 | .verdict = "UNRESOLVED (stream gate)"
     | .stages = ["code_pins", "selftests", "pins", "manifests", "nodes", "c4", "draw0", "c0"]
     | .d9 = {run_mode: "fullB", candidate: 1, status: null, baseline_ok: null, identified: null, grid_forms_equal: true,
              grid_exact_levels_equal: true, candidates: []}]}')
E='.workers[0].d9'
CF="$E.candidates[0].first_failures |= map(select(startswith(\"gap \") | not)) | $E.candidates[0].gap_mismatch = 0"  # the control without its gap id
JO="$E.candidates[1] |= (.fixed_match = false | .fixed_mismatch = 2 | .first_failures = [\"FIBS|uncut C fixed_tau_B@0.01@draw1\"])"
M3="$E.candidates[3] = ($E.candidates[1] + {cand: 3})"
chk d9_diag_valid "$DIAG" '.' ok
chk d9_diag_none_matched "$DIAG" "$E.candidates[1] = ($E.candidates[0] + {cand: 1}) | $E.status = \"none matched\" | $E.identified = null" ok
chk d9_diag_control_failed "$DIAG" "$CF | $E.baseline_ok = false | $E.status = \"control failed\" | $E.identified = null" ok
chk d9_control_failed_identified "$DIAG" "$CF | $E.baseline_ok = false" fail
chk d9_two_match_first_wins "$DIAG" "$M3" ok
chk d9_identified_not_first "$DIAG" "$M3 | $E.identified = 3" fail
chk d9_joint_only_no_match "$DIAG" "$JO | $E.status = \"none matched\" | $E.identified = null" ok
chk d9_joint_only_identified "$DIAG" "$JO" fail
chk d9_match_inconsistent "$DIAG" "$E.candidates[1].calib_mismatch = 2" fail
chk d9_baseline_ok_wrong "$DIAG" "$CF" fail
chk d9_planted_value_id "$DIAG" "$E.candidates[2].first_failures = [\"FIBS|uncut C tau_B@0.52@draw1\"]" fail
chk d9_draw0_false_identified "$DIAG" "$E.candidates[1].draw0_ok = false" ok
chk d9_entry_extra_key "$DIAG" "$E.candidates[0].extra = 1" fail
chk d9_entries_order "$DIAG" "$E.candidates |= reverse" fail
chk d9_entries_six "$DIAG" "$E.candidates |= .[:6]" fail
chk d9_candidate_in_diag "$DIAG" "$E.candidate = 1" fail
chk d9_null_in_diag "$DIAG" "$E = null" fail
chk d9_stop_report_null "$(printf '%s' "$DIAG" | jq -c --argjson b "$BLANK" '.workers = [$b | .record = "B" | .stop = "STOP: worker exception other"]')" '.' ok
chk d9_diag_stage_no_candidates "$DIAG" "$E |= (.candidates = [] | .status = null | .baseline_ok = null | .identified = null)" fail
chk d9_diag_verdict_exact "$DIAG" '.workers[0].verdict = "INTERVALS REPRODUCED (EXACT)"' fail
chk d9_run_mode_mismatch "$DIAG" "$E.run_mode = \"fullB\"" fail
chk d9_run_mode_unknown "$DIAG" '.run_mode = "diagA"' fail
chk d9_fan_out_2_in_diag "$(printf '%s' "$DIAG" | jq -c --argjson b "$BLANK" '.fan_out = 2 | .workers = [$b] + .workers')" '.' fail
chk d9_fan_out_1_in_ab "$DIAG" '.run_mode = "AB"' fail
chk d9_in_ab "$(jq -cn --argjson f "$FULL" --argjson d "$DIAG" '$f | .workers[1].d9 = $d.workers[0].d9')" '.' fail
chk d9_full_valid "$FULLB" '.' ok
chk d9_full_candidate_7 "$FULLB" "$E.candidate = 7" fail
chk d9_full_candidate_0 "$FULLB" "$E.candidate = 0" fail
chk d9_full_with_candidates "$(jq -cn --argjson f "$FULLB" --argjson d "$DIAG" '$f | .workers[0].d9.candidates = $d.workers[0].d9.candidates')" '.' fail
# addendum 9b §3: isolated control / upgrade / fullB-entries cases, coverage and mismatch <= compared
CFAIL="$E.baseline_ok = false | $E.status = \"control failed\" | $E.identified = null"
chk d9_diag_verdict_upgrade "$DIAG" '.workers[0].verdict = "NOT REPRODUCED"' fail
chk d9_control_fixed_false "$DIAG" "$E.candidates[0].fixed_match = false | $E.candidates[0].fixed_mismatch = 1" fail
chk d9_control_fixed_false_consistent "$DIAG" "$E.candidates[0].fixed_match = false | $E.candidates[0].fixed_mismatch = 1 | $CFAIL" ok
chk d9_control_draw0_false "$DIAG" "$E.candidates[0].draw0_ok = false" fail
chk d9_control_draw0_false_consistent "$DIAG" "$E.candidates[0].draw0_ok = false | $CFAIL" ok
chk d9_full_candidates_isolated "$(jq -cn --argjson f "$FULLB" --argjson d "$DIAG" '$f | .workers[0].d9 |= (.candidates = $d.workers[0].d9.candidates
  | .baseline_ok = true | .status = "identified" | .identified = 1) | .workers[0].stages += ["diag"]')" '.' fail
chk d9_mismatch_over_compared "$DIAG" "$E.candidates[2].gap_mismatch = 9001" fail
chk d9_coverage_calib_wrong "$DIAG" "$E.candidates[2].calib_compared = 17999" fail
chk d9_coverage_gap_wrong "$DIAG" "$E.candidates[2].gap_compared = 9001" fail
chk d9_coverage_fixed_wrong "$DIAG" "$E.candidates[1].fixed_compared = 36000" fail
chk d9_mom_negative "$DIAG" "$E.candidates[0].calib_mom_compared = -1" fail
# two documents in one file must fail too (the gate is slurped)
{ printf '%s\n' "$FULL"; printf '%s\n' "$FULL"; } | g && { echo "STOP: fixture two_documents passed"; exit 3; }
[ "$n" = "100" ] || { echo "STOP: fixture count $n"; exit 3; }
echo "FIXTURES OK (100/100 as predicted, plus two_documents)"
