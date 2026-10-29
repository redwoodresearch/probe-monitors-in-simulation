# COUNTSCHECK keys-and-counts extract of an honest_all record (writer: run_honest_all.py, cited, not read).
# Keeps run-key maps, label maps, count leaves and certification booleans. Drops every score, AUROC, catch, threshold,
# null draw and bootstrap leaf: run_scores keep their KEYS only.
{
  run_keys, run_set, run_epoch, run_cell, run_main, run_folds, run_source,
  epochs_per_class, per_class_set_epoch,
  pool_sizes: .inputs.pool_sizes,
  routec_sold_sim: (if .inputs.routec_sold_sim == null then null
                    else .inputs.routec_sold_sim | {sold_sim, registered, pilot} end),
  routec_prior_match: (if .inputs.routec_prior_match == null then null
                       else .inputs.routec_prior_match | {equal, n_prior, n_pilot, n_mains} end),
  drops: [.inputs.drops[] | {world, source, reason, n}],
  meta_checked: .skip.meta_checked,
  zero_row_runs: (.zero_row_runs | map_values({n, by_class, by_set, runs: [.runs[] | {key, name, set}]})),
  uncut_zero_row_runs: .uncut_gates.zero_row_runs,
  n_uncut_failures: (.uncut_gates.failures | length),
  n_failed: (.failed | length),
  n_required_gaps: (.required_gaps | length),
  rows_per_run_equal_across_layers,
  cells: (.cells | map_values({
    rows, rows_per_run, runs_without_rows,
    certified_per_column: [.certificates[]? | .certified_per_column],
    problems: (.problems | map_values({
      n_runs, observed_complete, null_complete, observed_certified,
      null_certified: .null.certified,
      run_score_keys: (.run_scores | keys)
    }))
  }))
}
