L=/home/aswin/phase2/nm/runs/2026-10-15_sim2gap_s4r/launch.json
jq -c '{finished: (.finished // null), n_runs: (.runs|length), arms: ([.runs[].arm] | group_by(.) | map({(.[0]): length}) | add),
  runs: [.runs[] | select(.key|test("/nm242_")) | {key, arm, state, n_attempts: ((.attempts // [])|length),
    attempts: [(.attempts // [])[] | {kind, exit_code, policy_called, clean: .outcome.clean, eval_status: .outcome.eval_status,
      n_sample_errors: ((.outcome.sample_errors // [])|length), sample_error_lens: [(.outcome.sample_errors // [])[] | tostring | length],
      n_errors: ((.outcome.errors // [])|length), samples: .outcome.samples}]}],
  reference_state: ((.reference_state // {}) | with_entries(select(.key|test("^nm242_"))) | map_values({state, tries,
    preflight: ((.preflight // {}) | map_values({state, exit_code}))}))}' "$L"
