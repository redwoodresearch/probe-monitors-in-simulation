set -e
S=/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad
SW=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/80d5a744-25a9-4506-a519-d6d787a845e5/scratchpad/aswin4/repairs/out/swap_sets.json
LT=/Users/aswinsurya/probe-distribution-shift-from-sims/.claude/worktrees/agent-a41654b9677ba3292/experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548
OL=$S/wt_olmo/experiments/2026-10-12_q2_olmo_reader/results
LL=$S/wt_llama/experiments/2026-10-12_q2_llama_reader/results/run_scores_2026-10-12_143129
PS=$S/wt_ps/experiments/2026-10-12_plantslip_transfer/results
REP=$S/wt_rep/experiments/2026-10-12_q2_per_repair/results
R2=$S/wt_row2/experiments/2026-10-12_plantslip_heldout/results
LPL=$S/wt_lplant/experiments/2026-10-12_q2_llama_plant/results
MAIN=$S/wt_sens/experiments
D=$S/wt_sens/experiments/2026-10-12_q2_repair_sensitivity
PY="uv run --no-project --python 3.12 --with numpy python"
SUB=$1
cd $S/wt_sens
$PY $D/analyze_subset_per_repair.py --subset $SUB --swap-sets $SW --committed $REP/per_repair_939_2026-10-12_145825.json \
  --stage-record $REP/stage_per_repair_2026-10-12_133921.json --qwen-scores-dir $LT \
  --qwen-metrics $MAIN/2026-10-06_q3_attack_detector/results/metrics_2026-10-06_181548__2026-10-06_182739.json \
  --fix939-runs $HOME/probe-shift-q2-fix939/fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl \
  --fix939-result $MAIN/2026-10-08_q3_swap_cue/results/fix939_POSTHOC_2026-10-11_020519.json \
  --qwen-sr $REP/per_repair_qwen3_frozen_runs_2026-10-12_143541.jsonl --qwen-insert $REP/per_repair_qwen3_frozen_runs_2026-10-12_144249.jsonl \
  --olmo-scores-dir $OL/run_scores_2026-10-12_110549 --olmo-both-npz $OL/fix_scores_olmo_2026-10-12_114920.npz \
  --olmo-part2 $OL/edit_effect_olmo_2026-10-12_115023.json --olmo-sr $REP/per_repair_olmo_scores_sr_2026-10-12_145740.npz \
  --olmo-insert $REP/per_repair_olmo_scores_insert_2026-10-12_145721.npz | tail -1
uv run --no-project --python 3.12 --with numpy --with "modal==1.6.1" python $D/analyze_subset_reply_delta.py P4 $PS/q2_reply_scores_P4_planted_2026-10-12_151508.json $SUB $SW $PS/q2_reply_delta_P4_2026-10-12_151742.json | tail -1
uv run --no-project --python 3.12 --with numpy --with "modal==1.6.1" python $D/analyze_subset_reply_delta.py P5 $PS/q2_reply_scores_P5_planted_2026-10-12_150508.json $SUB $SW $PS/q2_reply_delta_P5_2026-10-12_150711.json | tail -1
$PY $D/analyze_subset_plant_llama.py $LL $LPL/llama_plant_scores_planted-e2_2026-10-12_212056.json $SUB $SW $LPL/llama_plant_delta_2026-10-12_212724.json | tail -1
$PY $D/analyze_subset_y.py $R2/q2_step2d_2026-10-12_211734.json $S/q2_reply_stage_2026-10-12_135318.json $R2/real_only_2026-10-12_191044 2026-10-12_192842 \
  $R2/q2_frozen_scores_planted_2026-10-12_202420.json $R2/q2_olmo_scores_planted_2026-10-12_203343.json $R2/q2_llama_scores_planted_2026-10-12_212315.json \
  $R2/q2_sim_scores_qwen3_2026-10-12_215648.json $OL/run_scores_2026-10-12_110549 $LL $S/row2/manifest_part1_2026-10-12_185811.json $SUB $SW | tail -1
