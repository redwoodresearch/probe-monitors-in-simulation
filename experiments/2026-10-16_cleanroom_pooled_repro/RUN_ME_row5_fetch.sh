#!/bin/bash
# RUN_ME (addendum 26; a1 runs): after the detached fit, fetch <stamp>/a_vs_b/fit_summary.json (writer = fit_driver @ blob c9cfd9e8:
# counts, shas, booleans, wall_s, projection; no score values) and gate it. Keys / counts only on the laptop.
set -euo pipefail
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
D=$S/ps-repro/experiments/2026-10-16_cleanroom_pooled_repro
OUT=$D/results/honest_refit_fit_summary_a_vs_b_2026-10-20_140924.json
[ -e "$OUT" ] && { echo "STOP: $OUT exists"; exit 3; }
cd "$S/ps-repro"
uv run --no-project --python 3.12 --with modal==1.6.1 modal volume get probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16 /2026-10-20_140924/a_vs_b/fit_summary.json "$OUT"
shasum -a 256 "$OUT"
jq -c '{problems, sub, go, n_cells_done, n_uncertified_total, exit_nonzero_required, cells: [.cells[] | {cell, n_rows, n_fits, n_uncertified, zero_row_runs, wall_s, record_sha256}]}' "$OUT"
jq -e '.problems == ["a_vs_b"] and .sub == "a_vs_b" and .exit_nonzero_required == false and .n_cells_done == 2' "$OUT" > /dev/null \
  && echo "GATE PASS" || echo "GATE FAIL: fit summary gate false (committed as-is and labelled; no relaunch without a new GO)"
