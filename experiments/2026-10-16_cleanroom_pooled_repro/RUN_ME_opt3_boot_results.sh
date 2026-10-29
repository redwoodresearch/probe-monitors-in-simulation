#!/bin/bash
# RUN_ME (option (3) main run, results; after the driver has written its summary). Usage:
#   bash RUN_ME_opt3_boot_results.sh opt3_summary_<launch_id>_<YYYYMMDD_HHMMSS>_<8 hex>.json   (the exact name; a names-only volume ls)
# Fetches exactly that file from the output Volume into results/ (create-only), runs the committed gate's frozen fixtures (OPT3
# addenda 8-8f: 60 cases, jq on synthetic JSON), then the gate on the real summary (one document; exact keys / types / fixed codes /
# identifier grammars; node keys by membership in the committed node lists; the shared rules), and prints codes, counts and keys only.
set -euo pipefail
NAME=${1:?exact summary file name}
[[ "$NAME" =~ ^opt3_summary_[0-9]{10,16}_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}\.json$ ]] || { echo "STOP: bad name"; exit 3; }
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
cd "$S/ps-repro/experiments/2026-10-16_cleanroom_pooled_repro"
GATE=opt3/opt3_summary_shape.jq; GATE_BLOB=96366a14651ea69316e1c7ddd20d9ce62c62f471
FIX=opt3/opt3_summary_fixtures.sh; FIX_BLOB=e370869e0a59c3797c9719ff81edac80509f0011
NA=opt3/opt3_nodes_A.txt; NA_BLOB=671b015d73e29f75c214f9682179d4eb21588105
NB=opt3/opt3_nodes_fig4.txt; NB_BLOB=f1178581d7e88720e48b91c223956d84d1441d47
PRED=opt3/PREDICTIONS_r2_8.json; PRED_BLOB=097663aebcbf2c7558fe369cb4b380225b2fd706
SLOTS=opt3/opt3_d5_slots.txt; SLOTS_BLOB=8ad8519e7557db1a4de9111b483e64f67d51316a
BOOT=cleanroom_boot_opt3.py; BOOT_BLOB=c65bf420ba792a73d958aac9f52b6ce7577e271f  # stop_line's bound: the pinned script's line count (addendum 8d)
RED=cleanroom_reduce_v2.py; RED_BLOB=b8c136c8c84a7a24a0c75c6120c17b21ca021627  # stop_line's bound for stop_file reducer (addendum 8f)
for pair in "$GATE $GATE_BLOB" "$FIX $FIX_BLOB" "$NA $NA_BLOB" "$NB $NB_BLOB" "$PRED $PRED_BLOB" "$SLOTS $SLOTS_BLOB" "$BOOT $BOOT_BLOB" "$RED $RED_BLOB"; do
  set -- $pair
  git diff --quiet HEAD -- "$1" || { echo "STOP: $1 modified"; exit 3; }
  [ "$(git rev-parse HEAD:./$1)" = "$2" ] || { echo "STOP: $1 blob"; exit 3; }
done
NLINES="{\"boot\": $(git cat-file -p "$BOOT_BLOB" | wc -l | tr -d " "), \"reducer\": $(git cat-file -p "$RED_BLOB" | wc -l | tr -d " ")}"
bash "$FIX" || { echo "STOP: gate fixtures"; exit 3; }
OUT=results/$NAME
[ ! -e "$OUT" ] || { echo "STOP: $OUT exists"; exit 3; }
uv run --no-project --python 3.12 --with modal==1.6.1 modal volume get probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16 "/$NAME" "$OUT"
shasum -a 256 "$OUT"
jq -e -s --rawfile nodesA "$NA" --rawfile nodesB "$NB" --slurpfile pred "$PRED" --rawfile slots "$SLOTS" --argjson nlines "$NLINES" -f "$GATE" "$OUT" > /dev/null || { echo "STOP: summary schema"; exit 3; }
echo "SCHEMA OK"
jq -c '{launch_id, fan_out, driver_stop}' "$OUT"
jq -c '.workers[] | {record, stop, stop_file, stop_line, input_file, stop_ids, verdict, stages, n_degenerate, thread_env_pre, blas_pools_n, blas_threads_max, selftests_ok, selftests_failed,
  c5_trials, c5_compared, c5_skipped_d7, mains_outside_universe, draw0_ok, n_draw0_failures: (.draw0_failures | length), c0_scale,
  d1_identified, d1_reading, d1_s1_equals_s2, d1_candidate_failures, pilot_projected_s, pilot_elapsed_s, pilot_rss_peak_mib,
  pilot_rss_child_mib, pilot_mem_projected_mib, pilot_mem_source, c1_ok, c1_n_labels_ok, c1_n_labels, c1_gap_compared, c1_gap_mismatch,
  c1_gap_flip, c1_ok_mask_diffs, c1_first_failures, c2_method, c2_n_nodes, c4_n, c4_n_stopped, n_nodes, n_reproduced, n_reproduced_flip,
  n_not_reproduced, n_not_like_for_like, n_unresolved, n_incomplete_draws, mc_n_agree,
  nan_totals: ([.nan_by_node[]] | if length == 0 then null else reduce .[] as $c ({}; reduce ($c | to_entries[]) as $e (.; .[$e.key] += $e.value)) end),
  d5_status, d5_pair, d5_ok, d5_not_reproduced, d5_unresolved, n_d5_flip_sensitive, n_failing_nodes: (.failing_nodes | length), seconds}' "$OUT"
