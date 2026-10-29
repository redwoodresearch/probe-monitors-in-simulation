#!/usr/bin/env bash
# Every step of the Q2 real->sim shift analysis, in dependency order, from the repo root:
#   bash experiments/2026-10-06_q2_impactful_call_transfer/shift/run_all.sh
# Local CPU only (saved scores); about 10 minutes. Inputs: see README.md.
set -euo pipefail
D=experiments/2026-10-06_q2_impactful_call_transfer/shift
run() { set -- $1; echo "=== $*"; uv run --with pandas --with pyarrow --with matplotlib python "$D/$1.py" "${@:2}"; }
for step in assemble assemble_qwen35 stage1 plot_stage1 stage2 stage2c stage3 context_features history format_matched \
            behavior insert_history str_replace_history insert_experiment examples view_format edit_format level7_pieces compare_levels str_replace_levels \
            attempted_only "attempted_only paired" batch_split; do
  run "$step"
done
