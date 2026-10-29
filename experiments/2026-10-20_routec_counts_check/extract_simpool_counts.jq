# COUNTSCHECK counts-only extract of routec_simpool_numbers (writer: run_routec_simpool_numbers.py, cited, not read).
# Items 1, 2 and 4 only. Drops every share, percent, printed-precision field, percent-based boolean and percent-based flag
# (09's condition 1: those are compared inside the Modal function), item 5, the gates and record A's sizes.
def counts: {n, sol_using, sol_free, unknown};
.items as $it
| {
    item1: ($it["1_sim_footnote"] | {
      pool: (.pool | map_values(counts)),
      detector_cells: (.detector_cells | {routec: (.routec | map_values(counts)), v83: (.v83 | map_values(counts))}),
      n_paths_by_number_of_sources,
      pilot_header_scan: (.pilot_header_scan | {n_files, n_header_read, n_unknown,
                          rows: [.rows[] | {path, has_header, models_any_tokens}]}),
      v83_counts_equal_relay: .v83_reproduces_printed["pool counts equal the 10-17 relay (sol, n)"]
    }),
    item2: ($it["2_failed_footnote"] | {
      headline_sets,
      v83: (.v83 | {failed, launched, rate_limit}),
      routec: (.routec | {failed, launched, rate_limit}),
      pilot_runs_in_headline_sets,
      pilot_attempts_S_old_apart: (.pilot_attempts_S_old_apart | {files,
        counted_attempts_without_eval_simh: .counted_attempts_without_eval.simh,  # a count in the record (review O4)
        manifest_counts})
    }),
    item4: ($it["4_honest_detector_cohort"] | {routec, v83, pool_n, flags})
  }
