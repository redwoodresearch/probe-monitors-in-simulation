#!/bin/bash
# RUN_ME (option (3), the record-B debug of OPT3 addenda 9 / 9a / 9b, results; after the driver has written its summary). Usage:
#   bash RUN_ME_opt3_d9_results.sh opt3_summary_<launch_id>_<YYYYMMDD_HHMMSS>_<8 hex>.json   (the exact name; a names-only volume ls)
# Fetches exactly that file from the output Volume into results/ (create-only), runs the committed gate's frozen fixtures (100 cases,
# jq on synthetic JSON), then the gate on the real summary (one document; exact keys / types / fixed codes / identifier grammars; node
# keys by membership in the committed node lists; the shared rules; d9 recomputed from its entries), and prints codes, counts and
# keys only. 0d runs it. Addendum 9b §3 (e): the file goes straight into ~/probe-shift-reruns-2026-10-20/opt3_run/<launch_id>/ (create-only),
# never into a1's tree; 0d commits it on its own branch (09's isolation rule).
set -euo pipefail
NAME=${1:?exact summary file name}
[[ "$NAME" =~ ^opt3_summary_[0-9]{10,16}_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}\.json$ ]] || { echo "STOP: bad name"; exit 3; }
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
cd "$S/ps-repro/experiments/2026-10-16_cleanroom_pooled_repro"
GATE=opt3/opt3_summary_shape.jq; GATE_BLOB=1f771d7a8c72924e7c947a7eab0f1a41036a1fc9
FIX=opt3/opt3_summary_fixtures.sh; FIX_BLOB=2c084d9555a94fc0177875b88e446d2108630e81
NA=opt3/opt3_nodes_A.txt; NA_BLOB=671b015d73e29f75c214f9682179d4eb21588105
NB=opt3/opt3_nodes_fig4.txt; NB_BLOB=f1178581d7e88720e48b91c223956d84d1441d47
PRED=opt3/PREDICTIONS_r2_11.json; PRED_BLOB=d5c23c00f7f4261125933da506e0320d7111c217
SLOTS=opt3/opt3_d5_slots.txt; SLOTS_BLOB=8ad8519e7557db1a4de9111b483e64f67d51316a
BOOT=cleanroom_boot_opt3.py; BOOT_BLOB=25255f9a701e4866dfff038e8d22e121b1a0f456  # stop_line's bound: the pinned script's line count (addendum 8d)
RED=cleanroom_reduce_v2.py; RED_BLOB=b8c136c8c84a7a24a0c75c6120c17b21ca021627  # stop_line's bound for stop_file reducer (addendum 8f)
for f in "$GATE:$GATE_BLOB" "$FIX:$FIX_BLOB" "$NA:$NA_BLOB" "$NB:$NB_BLOB" "$PRED:$PRED_BLOB" "$SLOTS:$SLOTS_BLOB" "$BOOT:$BOOT_BLOB" "$RED:$RED_BLOB"; do
  p=${f%%:*}; b=${f##*:}
  git diff --quiet HEAD -- "$p" || { echo "STOP: $p modified"; exit 3; }
  [ "$(git rev-parse "HEAD:./$p")" = "$b" ] || { echo "STOP: $p blob"; exit 3; }
done
NLINES="{\"boot\": $(git cat-file -p "$BOOT_BLOB" | wc -l | tr -d " "), \"reducer\": $(git cat-file -p "$RED_BLOB" | wc -l | tr -d " ")}"
bash "$FIX" || { echo "STOP: gate fixtures"; exit 3; }
LID=${NAME#opt3_summary_}; LID=${LID%%_*}  # the launch id: the digits after opt3_summary_
OUTD=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_run/$LID; mkdir -p "$OUTD"; OUT=$OUTD/$NAME
[ ! -e "$OUT" ] || { echo "STOP: $OUT exists"; exit 3; }
uv run --no-project --python 3.12 --with modal==1.6.1 modal volume get probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16 "/$NAME" "$OUT"
shasum -a 256 "$OUT"
jq -e -s --rawfile nodesA "$NA" --rawfile nodesB "$NB" --slurpfile pred "$PRED" --rawfile slots "$SLOTS" --argjson nlines "$NLINES" -f "$GATE" "$OUT" > /dev/null || { echo "STOP: summary schema"; exit 3; }
echo "SCHEMA OK"
jq -c '{launch_id, run_mode, fan_out, driver_stop}' "$OUT"
jq -c '.workers[] | {record, stop, stop_file, stop_line, input_file, stop_ids, verdict, stages, n_degenerate, blas_pools_n, blas_threads_max, selftests_ok,
  selftests_failed, mains_outside_universe, draw0_ok, n_draw0_failures: (.draw0_failures | length), c0_scale, c0_ok, d1_identified, d1_reading,
  d1_s1_equals_s2, d1_candidate_failures, c4_n, c4_n_stopped, c2_method, c2_n_nodes, pilot_projected_s, pilot_elapsed_s, c1_ok, c1_n_labels_ok,
  c1_n_labels, c1_gap_compared, c1_gap_mismatch, c1_gap_flip, c1_first_failures, n_nodes, n_reproduced, n_reproduced_flip, n_not_reproduced,
  n_not_like_for_like, n_unresolved, n_incomplete_draws, mc_n_agree, n_failing_nodes: (.failing_nodes | length), seconds}' "$OUT"
jq -c '.workers[] | .d9 | if . == null then {d9: null} else {run_mode, candidate, status, baseline_ok, identified, grid_forms_equal, grid_exact_levels_equal} end' "$OUT"
jq -c '.workers[] | (.d9.candidates // [])[] | {cand, joint_match, fixed_match, draw0_ok, calib_compared, calib_mismatch, gap_compared, gap_mismatch,
  fixed_compared, fixed_mismatch, n_first_failures: (.first_failures | length), first_failures}' "$OUT"
