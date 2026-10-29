#!/bin/bash
# RUN_ME (option (3) real_line scope extension, results; ADDENDUM_OPT3_10_rlext.md). Adapted from RUN_ME_opt3_boot_results.sh (the r2.7
# results script). Differences only: the worktree; the A node list and predictions for the REAL summary are the rlext ones; the script is
# cleanroom_boot_opt3.py (= r2.7 + 3 pinned constants); the r2.7-era gate and fixtures (blobs 96366a14 / e370869e, as that script pinned) are materialized by
# blob into a fresh folder (the branch head carries later versions); modal by its named binary; two extra keys-only prints.
# Usage: bash opt3/rlext/RUN_ME_opt3_rlext_results.sh <exact summary name>
set -euo pipefail
NAME=${1:?exact summary file name}
[[ "$NAME" =~ ^opt3_summary_[0-9]{10,16}_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}\.json$ ]] || { echo "STOP: bad name"; exit 3; }
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
W=/Users/aswinsurya/probe-shift-opt3-rlext-2026-10-25/experiments/2026-10-16_cleanroom_pooled_repro
cd "$W"
NA=opt3/opt3_nodes_A_rlext.txt; BOOT=cleanroom_boot_opt3.py; RED_BLOB=b8c136c8c84a7a24a0c75c6120c17b21ca021627
for f in "$NA" "$BOOT" cleanroom_reduce_v2.py; do git diff --quiet HEAD -- "$f" || { echo "STOP: $f modified"; exit 3; }; done
[ "$(head -60 "$NA" | shasum -a 256 | cut -c1-64)" = "8f6f42788ceb2d48a375e8dec21106ee8859debca7f75e41d5a15c1ecad5702b" ] || { echo "STOP: A's 60 nodes changed"; exit 3; }
[ "$(shasum -a 256 "$NA" | cut -c1-64)" = "a5c12f43cfa9d2358ea189409701a2f6067f79870a524f08715b440495c1b5d7" ] || { echo "STOP: extended node list sha"; exit 3; }
[ "$(shasum -a 256 opt3/rlext/PREDICTIONS_r2_8_rlext.json | cut -c1-64)" = "d0b665dbcae5852587beab0c8b530e435a51d6e2ed33c55457b75220ce16d337" ] || { echo "STOP: rlext predictions sha"; exit 3; }
T=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_run_rlext/gate_r27_$(TZ=America/Los_Angeles date '+%Y%m%d_%H%M%S')
[ ! -e "$T" ] || { echo "STOP: $T exists"; exit 3; }
mkdir -p "$T/opt3"
while read -r b n; do git cat-file -p "$b" > "$T/opt3/$n"; [ "$(git hash-object "$T/opt3/$n")" = "$b" ] || { echo "STOP: blob $n"; exit 3; }; done <<'BLOBS'
96366a14651ea69316e1c7ddd20d9ce62c62f471 opt3_summary_shape.jq
e370869e0a59c3797c9719ff81edac80509f0011 opt3_summary_fixtures.sh
671b015d73e29f75c214f9682179d4eb21588105 opt3_nodes_A.txt
f1178581d7e88720e48b91c223956d84d1441d47 opt3_nodes_fig4.txt
097663aebcbf2c7558fe369cb4b380225b2fd706 PREDICTIONS_r2_8.json
8ad8519e7557db1a4de9111b483e64f67d51316a opt3_d5_slots.txt
BLOBS
git cat-file -p c65bf420ba792a73d958aac9f52b6ce7577e271f > "$T/cleanroom_boot_opt3.py"  # the fixtures' stop_line bound (same line count as $BOOT)
git cat-file -p "$RED_BLOB" > "$T/cleanroom_reduce_v2.py"
[ "$(wc -l < "$T/cleanroom_boot_opt3.py")" = "$(git cat-file -p "HEAD:./$BOOT" | wc -l)" ] || { echo "STOP: line counts differ"; exit 3; }
cp "$NA" "$T/opt3/opt3_nodes_A_rlext.txt"
NLINES="{\"boot\": $(git cat-file -p "HEAD:./$BOOT" | wc -l | tr -d " "), \"reducer\": $(git cat-file -p "$RED_BLOB" | wc -l | tr -d " ")}"
(cd "$T" && bash opt3/opt3_summary_fixtures.sh) || { echo "STOP: gate fixtures"; exit 3; }
OUT=results/$NAME
[ ! -e "$OUT" ] || { echo "STOP: $OUT exists"; exit 3; }
"$M" volume get probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16 "/$NAME" "$OUT"
shasum -a 256 "$OUT"
jq -e -s --rawfile nodesA "$T/opt3/opt3_nodes_A_rlext.txt" --rawfile nodesB "$T/opt3/opt3_nodes_fig4.txt" --slurpfile pred opt3/rlext/PREDICTIONS_r2_8_rlext.json \
  --rawfile slots "$T/opt3/opt3_d5_slots.txt" --argjson nlines "$NLINES" -f "$T/opt3/opt3_summary_shape.jq" "$OUT" > /dev/null || { echo "STOP: summary schema"; exit 3; }
echo "SCHEMA OK"
jq -c '{launch_id, fan_out, driver_stop}' "$OUT"
jq -c '.workers[] | {record, stop, stop_file, stop_line, input_file, stop_ids, verdict, stages, selftests_ok, mains_outside_universe, draw0_ok,
  d1_identified, d1_reading, c1_ok, c1_n_labels_ok, c1_n_labels, c1_gap_compared, c1_gap_mismatch, c1_gap_flip, c1_first_failures, c4_n, c4_n_stopped,
  n_nodes, n_reproduced, n_reproduced_flip, n_not_reproduced, n_not_like_for_like, n_unresolved, n_incomplete_draws, d5_status, d5_ok,
  n_failing_nodes: (.failing_nodes | length), failing_nodes, values_dir, seconds}' "$OUT"
