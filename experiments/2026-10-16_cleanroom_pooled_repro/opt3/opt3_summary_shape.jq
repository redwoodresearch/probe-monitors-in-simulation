# Exact-schema gate for the option (3) driver summary (opt3_summary_<launch_id>_<stamp>_<nonce>.json), r2.9 (addenda 8-9b).
# Run as: jq -e -s --rawfile nodesA opt3/opt3_nodes_A.txt --rawfile nodesB opt3/opt3_nodes_fig4.txt --slurpfile pred opt3/PREDICTIONS_r2_11.json
#          --rawfile slots opt3/opt3_d5_slots.txt --argjson nlines {"boot": <wc -l of the script blob>, "reducer": <wc -l of the reducer blob>} -f opt3/opt3_summary_shape.jq
# Mirrors cleanroom_boot_opt3.py validate_summary / validate_report / validate_d9 (C4 rule per addendum 8b; run modes and d9 per 9a): exact key sets, integers only where integers are due (a number
# whose JSON literal is all digits, so 1.0 fails), fixed STOP codes and verdicts (no free text), identifier grammars, node keys by
# membership in the record's committed node list, and the shared rules (stop <-> verdict, C4, nan_by_node, EXACT, C0, stages).
def isint: type == "number" and (tojson | test("\\A-?[0-9]+\\z"));
def in_vocab($v): . as $x | any($v[]; . == $x);
def re($r): type == "string" and length <= 200 and test("\\A(?:" + $r + ")\\z");
def stage_names: ["code_pins", "selftests", "pins", "manifests", "nodes", "resolve", "c4", "c2", "draw0", "c0", "d1", "pilot", "draws", "c1", "compare", "values", "d5", "diag"];
def methods: ["linear", "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull",
              "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest"];
def d5pairs: [("half_range", "max_side") as $r | ("fraction", "points", "invariant") as $s | "\($r):\($s)"];
def d5status: ["fit", "ambiguous", "mixed", "unresolved", "not_reproduced", "error", "flip_sensitive"];
def verdicts: ["INTERVALS REPRODUCED (EXACT)", "NOT REPRODUCED", "NOT LIKE-FOR-LIKE", "NOT LIKE-FOR-LIKE (C0 scale unidentified)",
               "UNRESOLVED (stream gate)", "UNRESOLVED (D5 ambiguous)", "UNRESOLVED (D5 mixed readings)",
               "UNRESOLVED (D5 unresolved elements)", "UNRESOLVED (D5 error)", "UNRESOLVED (D5 flip-sensitive)", "STOP"];
def stop_bases: [
    "not started", "unknown stop code", "reducer blob", "pins config / q1_eval sha", "node list sha", "predictions file sha",
    "self-tests differ from the frozen predictions", "pin scope", "pin calib_json", "pin perdraw_json", "pin fig1_main",
    "pin fig1_main_R", "pin fig4", "pin rc1a", "D5 not ready: key_path form", "D5 not ready: slot missing",
    "D5 not ready: slot not a string", "D5 not ready: aggregator unknown", "D5 not ready: node not in A's list",
    "D5 not ready: no halfwidth elements", "record sha", "meta.merge layout", "calibration npz sha / manifest", "per-draw npz pin",
    "node list rebuild", "scope a_node form", "unparsed node", "a node level is not among the record's levels",
    "PLUMBING: pool / cohort mains outside meta.merge.mains", "engine draw 0 differs from compute_label", "unknown call pattern",
    "pilot projection over 0.8 x timeout", "pilot memory projection over 0.8 x 64 GiB", "calibration member missing",
    "calibration layout", "C1: a stored gap member is missing", "C1: stored gap array length vs the identified index rule",
    "C2: A record sha", "C2: A per-draw npz sha", "C2: expected members missing", "C2: stored array layout",
    "C2: no percentile method reproduces A's fixed-mode delta_full ci95", "fixture key absent", "C4 fixture did not stop",
    "report schema", "past the launch deadline", "worker report schema", "spawn marker exists (idempotency)", "summary schema",
    "BLAS threads not 1", "no BLAS pool found", "allow-list sha", "D5 not ready: slot list", "a record level is not in the pinned level list",
    "resolve:catch", "resolve:coheadline", "resolve:gap", "input missing", "input sha", "input pinned twice", "input missing: catalogue",
    "input missing: manifest", "level not on the registered grid", "d9: candidate not in the pre-stated list", "d9: unknown run mode",
    "d9: fullB not authorized by the diagnostic summary"];
def exc_names: [
    "AssertionError", "AttributeError", "BadZipFile", "BrokenProcessPool", "ConnectionError", "EOFError", "ExecutionError",
    "FileExistsError", "FileNotFoundError", "FloatingPointError", "FunctionTimeoutError", "ImportError", "IndexError",
    "InputCancellation", "InternalFailure", "JSONDecodeError", "KeyError", "LinAlgError", "MemoryError", "ModuleNotFoundError",
    "NotADirectoryError", "OSError", "OverflowError", "PermissionError", "RecursionError", "RemoteError", "RuntimeError",
    "TimeoutError", "TypeError", "UnicodeDecodeError", "ValueError", "ZeroDivisionError", "other"];
def stop_codes: (stop_bases | map("STOP: " + .)) + [("", "worker ", "driver ") as $w | exc_names[] | "STOP: \($w)exception \(.)"];
def input_files: ["E12_BLOCK_2026-10-16_092322.json", "INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json", "INPUTS_FULL_2026-10-17_233212.json", "authorized_overlaps_routec_2026-10-18_123612.json", "exclusions_2026-10-16_102159.json", "new_mains_cells_v1.json", "nofilter_exclusions_2026-10-16_012305.json", "override_S_new25f_C_2026-10-16_045304_gated.json", "override_S_new25f_R_2026-10-16_045058_gated.json", "override_S_new_C_2026-10-16_104513_gated.json", "override_S_new_R_2026-10-16_104459_gated.json", "override_S_newmains_C_2026-10-16_053303_gated.json", "override_S_newmains_R_2026-10-16_053350_gated.json", "override_S_old_C_2026-10-17_151056_gated.json", "override_S_old_R_2026-10-17_151100_gated.json", "override_S_paper493_C_2026-10-16_101944_gated.json", "override_S_paper493_R_2026-10-16_102114_gated.json", "override_pools_ep2_C_2026-10-16_053038_gated.json", "override_pools_ep2_R_2026-10-16_052820_gated.json", "own_all_rs_folds_2026-10-16.json", "pilot_tasks_amendment9_all196.json", "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz", "routec_B_alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652.json.gz", "runs.json", "scaleup_tasks_plus311_2026-10-14_125135.json", "simh_counted_keys_pilot_v3.json", "sp493_facts_2026-10-15_165730.json"];  # addendum 8f: the 27 /cfg basenames
def thread_vars: ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"];
def count_keys: ["d7", "d7_units_ok", "d6", "empty", "other"];
def LABEL: "[A-Za-z0-9_|\\-]{1,40} [CR]";
def level_list: ["0.001", "0.00133352", "0.00177828", "0.00237137", "0.00316228", "0.00421697", "0.00562341", "0.00749894", "0.01",
                  "0.0133352", "0.0177828", "0.0237137", "0.0316228", "0.0421697", "0.05"];
def LEVEL: "(?:" + (level_list | map(gsub("\\."; "\\.")) | join("|")) + ")";  # membership, never a free decimal (addendum 8c)
def QTY: "(?:(?:fixed_)?tau_[BS]@" + LEVEL + "|(?:mu|sd)_[BS]\\[[0-4]\\]|(?:mu0|sd0)_[BS]|(?:mu|sd)_[BS] layout)";
def MEMBER: "(?:(?:joint|fixed)__" + LEVEL + "__[BS]__T|joint__[BS]__(?:ok|mu|sd|mu0|sd0))";
def id_re($kind):
  {"draw0": (LABEL + ":(?:tau_[BS]|tpr_(?:run1|run2|sim|simB))@" + LEVEL),
   "d1": ("#(?:[1-9]|1[0-2]) (?:joint|joint_s507) " + LABEL + " " + QTY + "@draw[0-9]{1,5}"),
   "c1": (LABEL + " " + QTY + "@draw[0-9]{1,5}"),
   "stop_id": (LABEL + " " + MEMBER)}[$kind];
def id_ok($kind; $nodes; $allow):
  type == "string" and length <= 200 and (
    if $kind == "selftest" or $kind == "slot" then (. as $x | any($allow[$kind][]; . == $x))  # membership (addendum 8c)
    elif $kind == "c1" and startswith("gap ") then
      (.[4:] | rindex("@draw")) as $i
      | $i != null and (.[4:][:$i] as $n | any($nodes[]; . == $n)) and (.[4:][$i + 5:] | re("[0-9]{1,5}"))
    elif $kind == "stop_id" and (. as $x | any($nodes[]; . == $x)) then true
    else re(id_re($kind)) end);
def k($kind; $v; $nodes; $allow):
  if $kind == "enum" then (type == "string" and in_vocab($v))
  elif $kind == "enum|null" then (. == null or (type == "string" and in_vocab($v)))
  elif $kind == "stop" then (. == null or (type == "string" and (in_vocab(stop_codes)
                               or ((split(" @") | length) == 2 and (split(" @")[0] | in_vocab(stop_codes)) and (split(" @")[1] | in_vocab(stage_names))))))
  elif $kind == "stop_line" then (. == null or (isint and . >= 1))  # its per-file bound is a rule (addendum 8f)
  elif $kind == "count|null" then (. == null or (isint and . >= 0))
  elif $kind == "bool|null" then (. == null or type == "boolean")
  elif $kind == "int|null" then (. == null or isint)
  elif $kind == "int" then isint
  elif $kind == "list_enum" then (type == "array" and all(.[]; type == "string" and in_vocab($v)))
  elif $kind == "ids" then (type == "array" and all(.[]; id_ok($v; $nodes; $allow)))
  elif $kind == "nodes" then (type == "array" and all(.[]; type == "string" and (. as $x | any($nodes[]; . == $x))) and . == (unique))
  elif $kind == "nan_map" then (type == "object" and all(to_entries[]; (.key as $x | any($nodes[]; . == $x))
                               and (.value | type == "object" and (keys == (count_keys | sort)) and all(.[]; isint and . >= 0))))
  elif $kind == "thread_map" then (type == "object" and (. == {} or keys == (thread_vars | sort)) and all(.[]; isint and . >= -2 and . <= 9999))
  elif $kind == "values_dir" then (. == null or re("/o/opt3_[AB]_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}"))
  elif $kind == "hex64|null" then (. == null or re("[0-9a-f]{64}"))
  elif $kind == "d9" then (. == null or type == "object")  # its rules: d9_ok (addendum 9a)
  else false end;
def fields: {
  "record": ["enum", ["A", "B"]], "stop": ["stop", null], "stop_file": ["enum|null", ["boot", "reducer"]], "stop_line": ["stop_line", null],
  "stop_ids": ["ids", "stop_id"], "input_file": ["enum|null", input_files],
  "stages": ["list_enum", stage_names],
  "thread_env_pre": ["thread_map", null], "blas_pools_n": ["int|null", null], "blas_threads_max": ["int|null", null],
  "selftests_ok": ["bool|null", null], "selftests_failed": ["ids", "selftest"], "calib_sha_ok": ["bool|null", null],
  "c5_trials": ["int|null", null], "c5_compared": ["int|null", null], "c5_skipped_d7": ["int|null", null],
  "n_degenerate": ["count|null", null], "mains_outside_universe": ["int|null", null], "draw0_ok": ["bool|null", null], "draw0_failures": ["ids", "draw0"],
  "c0_scale": ["enum|null", ["fraction", "points"]], "c0_ok": ["bool|null", null],
  "d1_identified": ["int|null", null], "d1_reading": ["enum|null", ["joint", "joint_s507"]], "d1_s1_equals_s2": ["bool|null", null],
  "d1_candidate_failures": ["ids", "d1"],
  "pilot_projected_s": ["int|null", null], "pilot_elapsed_s": ["int|null", null], "pilot_rss_peak_mib": ["int|null", null],
  "pilot_rss_child_mib": ["int|null", null], "pilot_mem_projected_mib": ["int|null", null],
  "pilot_mem_source": ["enum|null", ["cgroup_v2", "cgroup_v1", "getrusage"]],
  "c1_ok": ["bool|null", null], "c1_n_labels": ["int|null", null], "c1_n_labels_ok": ["int|null", null], "c1_gap_compared": ["int|null", null],
  "c1_gap_mismatch": ["int|null", null], "c1_gap_flip": ["int|null", null], "c1_ok_mask_diffs": ["int|null", null], "c1_first_failures": ["ids", "c1"],
  "c2_method": ["enum|null", methods], "c2_n_nodes": ["int|null", null], "c4_n": ["int|null", null], "c4_n_stopped": ["int|null", null],
  "n_nodes": ["int|null", null], "n_reproduced": ["int|null", null], "n_reproduced_flip": ["int|null", null], "n_not_reproduced": ["int|null", null],
  "n_not_like_for_like": ["int|null", null], "n_unresolved": ["int|null", null], "n_incomplete_draws": ["int|null", null], "mc_n_agree": ["int|null", null],
  "nan_by_node": ["nan_map", null], "d5_status": ["enum|null", d5status], "d5_pair": ["enum|null", d5pairs], "d5_ok": ["bool|null", null],
  "d5_not_reproduced": ["ids", "slot"], "d5_unresolved": ["ids", "slot"], "n_d5_flip_sensitive": ["int|null", null],
  "verdict": ["enum", verdicts], "failing_nodes": ["nodes", null],
  "values_dir": ["values_dir", null], "values_sha256_node_series": ["hex64|null", null], "values_sha256_compare": ["hex64|null", null],
  "seconds": ["int", null], "d9": ["d9", null]};
def has_stage($s): any(.stages[]; . == $s);
# addendum 9a §3: the d9 object (null in AB; diagB / fullB: record B, null only on a STOP report), its 7 entries, and
# (baseline_ok, status, identified) recomputed from the entries as cleanroom_boot_opt3.py d9_status
def d9_control_ids: ["FIBS|uncut C tau_B@0.0237137@draw38", "harmfulness|uncut C tau_B@0.0237137@draw38", "high-stakes|uncut C tau_B@0.0421697@draw49",
                     "own_all_rs_uncut_L48 C tau_B@0.00562341@draw24", "refusal|uncut C tau_B@0.0237137@draw38", "sol_a_uncut C tau_B@0.00562341@draw24",
                     "gap pooled|refusal|uncut C|pooled_all_origins|joint|0.0237137|gap|delta_full@draw38"];  # addendum 9 §4.3
def d9_keys: ["run_mode", "candidate", "status", "baseline_ok", "identified", "grid_forms_equal", "grid_exact_levels_equal", "candidates"];
def d9_entry_keys: ["cand", "joint_match", "fixed_match", "draw0_ok", "calib_compared", "calib_mom_compared", "calib_mismatch", "gap_compared",
                    "gap_mismatch", "fixed_compared", "fixed_mismatch", "first_failures"];
def d9_cover: {"calib_compared": 18000, "gap_compared": 9000, "fixed_compared": 27000};  # addendum 9b §3 (c): 6 x 15 x 2 x 100; 90 x 100; 18,000 + 9,000
def d9_entry_ok($i; $nodes; $allow):
  type == "object" and (keys == (d9_entry_keys | sort))
  and (.cand | isint) and .cand == $i
  and all(.joint_match, .fixed_match, .draw0_ok; type == "boolean")
  and all(.calib_compared, .calib_mom_compared, .calib_mismatch, .gap_compared, .gap_mismatch, .fixed_compared, .fixed_mismatch; isint and . >= 0)
  and (. as $e | d9_cover | to_entries | all(.value == $e[.key]))  # the real loops' coverage, and mismatch <= compared (9b §3 (c))
  and .calib_mismatch <= .calib_compared + .calib_mom_compared and .gap_mismatch <= .gap_compared and .fixed_mismatch <= .fixed_compared
  and (.first_failures | type == "array" and all(.[]; id_ok("c1"; $nodes; $allow)) and . == sort)
  and (.joint_match == (.calib_mismatch == 0 and .gap_mismatch == 0)) and (.fixed_match == (.fixed_mismatch == 0));
def d9_status:
  if length == 0 then [null, null, null]
  elif (.[0].first_failures == (d9_control_ids | sort) and .[0].fixed_match == true and .[0].draw0_ok == true) | not then [false, "control failed", null]
  else [.[1:][] | select(.joint_match == true and .fixed_match == true) | .cand] as $h
       | [true, (if $h == [] then "none matched" else "identified" end), (if $h == [] then null else $h[0] end)] end;
def d9_ok($nodes; $allow; $mode):
  .d9 as $d
  | if $mode == "AB" then ($d == null and (has_stage("diag") | not))
    elif .record != "B" then false
    elif $d == null then .stop != null
    else ($d | type == "object" and keys == (d9_keys | sort) and .run_mode == $mode)
      and (if $mode == "diagB" then $d.candidate == null else ($d.candidate | isint and . >= 1 and . <= 6) end)
      and all($d.baseline_ok, $d.grid_forms_equal, $d.grid_exact_levels_equal; . == null or type == "boolean")
      and ($d.status == null or ($d.status | type == "string" and in_vocab(["identified", "none matched", "control failed"])))
      and ($d.identified == null or ($d.identified | isint and . >= 1 and . <= 6))
      and ($d.candidates | type == "array")
      and ($d.candidates == [] or ($mode == "diagB" and ($d.candidates | length) == 7
                                   and ($d.candidates | to_entries | all(.key as $i | .value | d9_entry_ok($i; $nodes; $allow)))))
      and ([$d.baseline_ok, $d.status, $d.identified] == ($d.candidates | d9_status))
      and (has_stage("diag") == ($d.candidates != []))
      and ($mode != "diagB" or .stop != null or (has_stage("diag") and .verdict == "UNRESOLVED (stream gate)"))
    end;
def report_ok($nodes; $allow; $mode):
  type == "object"
  and (keys == (fields | keys))
  and (. as $r | fields | to_entries | all(.value as $f | $r[.key] | k($f[0]; $f[1]; $nodes; $allow)))
  and ((.stop == null) == (.verdict != "STOP"))
  and ((.stop_line == null) == (.stop_file == null))
  and (.stop_line == null or (.stop != null and .stop_line <= $nlines[.stop_file]))  # within that file's line count (addenda 8d, 8f)
  and (.input_file == null or .stop != null)
  and (if (.stop | type == "string") and ((.stop | split(" @") | length) == 2) then ((.stop | split(" @")[1]) as $s | (has_stage($s) | not)) else true end)
  and (.c4_n == .c4_n_stopped) and (.c4_n == null or .c4_n >= 1) and (has_stage("c4") == (.c4_n != null))
  and (.verdict == "STOP" or has_stage("c4")) and ((has_stage("compare") | not) or .c4_n != null)
  and ((.nan_by_node | keys) == (if has_stage("draws") then ($nodes | unique) else [] end))
  and (.verdict != "INTERVALS REPRODUCED (EXACT)"
       or (.selftests_ok == true and .draw0_ok == true and .c0_ok == true and .c1_ok == true and .n_nodes == ($nodes | length)
           and .n_reproduced == .n_nodes and .failing_nodes == [] and has_stage("values")
           and .d5_status == (if .record == "A" then "fit" else null end)))
  and (.verdict != "NOT LIKE-FOR-LIKE (C0 scale unidentified)" or .c0_ok == false)
  and ((.stages | length) == (.stages | unique | length))
  and d9_ok($nodes; $allow; $mode);
($nodesA | split("\n") | map(select(length > 0))) as $nA
| ($nodesB | split("\n") | map(select(length > 0))) as $nB
| {selftest: ($pred[0] | to_entries | map(select(.key | startswith("_") | not))
             | map(.key as $g | [$g] + (.value | if type == "object" then (keys | map("\($g).\(.)")) else [] end)) | add),
   slot: ($slots | split("\n") | map(select(length > 0)))} as $allow
| length == 1
  and (.[0] | type == "object"
    and (keys == ["driver_stop", "fan_out", "launch_epoch", "launch_id", "run_mode", "workers"])
    and (.launch_epoch | type == "number" and (isinfinite | not) and (isnan | not))  # finite, as Python's _isnum (addendum 8c)
    and (.launch_id | re("[0-9]{10,16}"))
    and (.run_mode | type == "string" and in_vocab(["AB", "diagB", "fullB"]))  # addendum 9a
    and (.driver_stop == null or (.driver_stop | type == "string" and in_vocab(stop_codes)))
    and (.fan_out | isint)
    and ((.run_mode == "AB" and .fan_out == 2 and .driver_stop == null
          and (.workers | type == "array" and length == 2 and (map(.record) == ["A", "B"])
               and (.[0] | report_ok($nA; $allow; "AB")) and (.[1] | report_ok($nB; $allow; "AB"))))
         or (.run_mode != "AB" and .fan_out == 1 and .driver_stop == null  # diagB / fullB: exactly one B worker (addendum 9a)
             and (.run_mode as $m | .workers | type == "array" and length == 1
                  and (.[0] | type == "object" and .record == "B" and report_ok($nB; $allow; $m))))
         or (.fan_out == 0 and .workers == [] and .driver_stop != null)))
