#!/bin/bash
# RUN_ME (option (3), stage 0 results; run after the launch). Usage: bash RUN_ME_opt3_names_results.sh <record file name>
# <record file name> = the exact name the launch log printed after "wrote" (opt3_npz_names_<stamp>.json); exactly one file must match
# (no selection by recency). The shape gate is the committed opt3/opt3_names_shape.jq (blob pinned below), run with jq -e -s -f:
# exactly one JSON document, exact key sets, members = array of strings, n_members = its length (integer), sha256_matches_pin = true,
# no number anywhere except n_members. Its negative fixtures (predictions frozen in OPT3 addendum 3) run first; any deviation STOPs.
set -euo pipefail
NAME=${1:?exact record file name (opt3_npz_names_<stamp>.json)}
case "$NAME" in opt3_npz_names_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].json) ;; *) echo "STOP: bad name"; exit 3;; esac
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
cd "$S/ps-repro/experiments/2026-10-16_cleanroom_pooled_repro"
GATE=opt3/opt3_names_shape.jq
GATE_BLOB=dcb41836e384460cf71300af972352ab9d25997a
git diff --quiet HEAD -- "$GATE" || { echo "STOP: gate modified"; exit 3; }
[ "$(git rev-parse HEAD:./$GATE)" = "$GATE_BLOB" ] || { echo "STOP: gate blob"; exit 3; }
# --- frozen fixtures (OPT3 addendum 3 §1): valid -> pass; the 4 negatives -> fail
V='{"files":{"A_calib_five_sol_a_C":{"path":"p","sha256_matches_pin":true,"n_members":1,"members":["a.npy"]},"A_calib_single_FIBS_C":{"path":"p","sha256_matches_pin":true,"n_members":1,"members":["a.npy"]},"B_calib_five_sol_a_C":{"path":"p","sha256_matches_pin":true,"n_members":1,"members":["a.npy"]},"B_gap_perdraw":{"path":"p","sha256_matches_pin":true,"n_members":2,"members":["a.npy","b.npy"]}},"pins_from":[{"file":"f","sha256":"0000000000000000000000000000000000000000000000000000000000000000"},{"file":"g","sha256":"1111111111111111111111111111111111111111111111111111111111111111"}]}'
g() { jq -e -s -f "$GATE" > /dev/null 2>&1; }
printf '%s' "$V" | g || { echo "STOP: fixture valid did not pass"; exit 3; }
printf '%s' "$V" | jq -c '.files.A_calib_five_sol_a_C.members = {"a":"b"}' | g && { echo "STOP: fixture members_object passed"; exit 3; }
printf '%s' "$V" | jq -c '.extra = 1' | g && { echo "STOP: fixture extra_integer passed"; exit 3; }
{ printf '%s\n' "$V"; printf '%s\n' "$V"; } | g && { echo "STOP: fixture two_documents passed"; exit 3; }
printf '%s' "$V" | jq -c 'del(.files.A_calib_single_FIBS_C.path)' | g && { echo "STOP: fixture missing_path passed"; exit 3; }
echo "FIXTURES OK (5/5 as predicted)"
# --- the real record
N=$(find results -maxdepth 1 -type f -name "$NAME" | wc -l | tr -d ' ')
[ "$N" = "1" ] || { echo "STOP: $N files named $NAME"; exit 3; }
REC=results/$NAME
shasum -a 256 "$REC"
jq -e -s -f "$GATE" "$REC" > /dev/null || { echo "STOP: names record shape"; exit 3; }
echo "SHAPE OK"
jq -c '.files | to_entries[] | {role: .key, n_members: .value.n_members, sha_ok: .value.sha256_matches_pin}' "$REC"
jq -r '.files | to_entries[] | select(.key | startswith("B_gap") | not) | .key as $k | .value.members[] | "\($k)\t\(.)"' "$REC"
