#!/bin/bash
# RUN_ME (option (3) scope extension: the six probes' real_line at joint 1%; ADDENDUM_OPT3_10_rlext.md). 0d wrote and runs it.
# Adapted from RUN_ME_opt3_boot.sh (the r2.7 run that gave record A EXACT, 2026-10-21 13:15). Differences, and only these:
#   - the script keeps its name, cleanroom_boot_opt3.py (STOP_FILES keys on it), = r2.7's blob c65bf420 with exactly three
#     constants changed (NODES["A"], SCOPE, PREDICTIONS -> opt3/rlext/PREDICTIONS_r2_8_rlext.json: one frozen expectation 60|60 -> 72|72);
#   - the A node list is opt3/opt3_nodes_A_rlext.txt (A's 60 nodes, unchanged and in order, plus 12 real_line nodes);
#   - the scope file is opt3/rlext/a1_interval_scope_rlext_20261025.jsonl (2e's 106 lines, unchanged, plus 24 interval-end lines);
#   - the 27 reducer inputs are copied from the 13:15 run's own input folder (the /tmp scratchpad copy is gone) and sha-checked
#     against the same pins; the branch is local (no push), so HEAD must equal the pinned commit instead of origin;
#   - modal by its named binary (modal 1.6.1).
# Usage: bash opt3/rlext/RUN_ME_opt3_rlext.sh "Claude-Session: https://claude.ai/code/session_<id>" <40-hex commit>
set -euo pipefail
TRAILER=${1:?trailer required}
COMMIT=${2:?pinned commit required}
case "$TRAILER" in *$'\n'*) echo "STOP: trailer has more than one line"; exit 3;; esac
[[ "$TRAILER" =~ ^Claude-Session:\ https://claude\.ai/code/session_[A-Za-z0-9]+$ ]] || { echo "STOP: trailer format"; exit 3; }
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "STOP: commit format"; exit 3; }
W=/Users/aswinsurya/probe-shift-opt3-rlext-2026-10-25; D=experiments/2026-10-16_cleanroom_pooled_repro; scr=cleanroom_boot_opt3.py
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
APP=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
PREV=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_run/20261021_131516  # the A EXACT run's inputs (read-only copies)
H=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_run_rlext
cd "$W/$D"
[ "$(git symbolic-ref --short HEAD)" = "aswin/opt3-realline-ext" ] || { echo "STOP: branch"; exit 3; }
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || { echo "STOP: HEAD != pinned commit"; exit 3; }
for f in $scr cleanroom_reduce_v2.py config/routec_A18_pins.json config/routec_BS18_pins.json opt3/opt3_nodes_A_rlext.txt opt3/opt3_nodes_fig4.txt opt3/PREDICTIONS_r2_8.json opt3/opt3_d5_slots.txt opt3/opt3_record_path_templates.tsv opt3/rlext/a1_interval_scope_rlext_20261025.jsonl opt3/rlext/PREDICTIONS_r2_8_rlext.json ../../src/probe_shift/q1_eval.py; do
  git diff --quiet HEAD -- "$f" || { echo "STOP: $f modified"; exit 3; }
done
STAGED=$(git diff --cached --name-only) || { echo "STOP: git diff --cached failed"; exit 3; }
[ -z "$STAGED" ] || { echo "STOP: staged"; exit 3; }
# the script = r2.7's blob with exactly the three pinned-constant lines changed (68 NODES["A"], 70 SCOPE, 1812 PREDICTIONS)
DL=$(diff <(git cat-file -p c65bf420ba792a73d958aac9f52b6ce7577e271f) "$scr" | grep -c '^[<>]' || true)
[ "$DL" = "6" ] || { echo "STOP: script differs from r2.7 by $DL diff lines, not 6"; exit 3; }
[ "$(diff <(git cat-file -p c65bf420ba792a73d958aac9f52b6ce7577e271f) "$scr" | grep '^[0-9]' | tr '\n' ' ')" = "68c68 70c70 1812c1812 " ] || { echo "STOP: changed lines"; exit 3; }
[ "$(git rev-parse HEAD:./cleanroom_reduce_v2.py)" = "b8c136c8c84a7a24a0c75c6120c17b21ca021627" ] || { echo "STOP: reducer blob"; exit 3; }
[ "$(git rev-parse HEAD:./config/routec_A18_pins.json)" = "ec99948ff772a9aa2fa1ab386aa31fec26a40e51" ] || { echo "STOP: A18 pins blob"; exit 3; }
[ "$(git rev-parse HEAD:./config/routec_BS18_pins.json)" = "cf84ee2abd5cb80260e26d57dd3657a7465d550a" ] || { echo "STOP: BS18 pins blob"; exit 3; }
[ "$(git rev-parse HEAD:src/probe_shift/q1_eval.py)" = "0a08acf42c671034f82efef268d8bb87b4039b32" ] || { echo "STOP: q1_eval blob"; exit 3; }
[ "$(git rev-parse HEAD:./opt3/PREDICTIONS_r2_8.json)" = "097663aebcbf2c7558fe369cb4b380225b2fd706" ] || { echo "STOP: predictions blob"; exit 3; }
[ "$(git rev-parse HEAD:./opt3/opt3_nodes_fig4.txt)" = "f1178581d7e88720e48b91c223956d84d1441d47" ] || { echo "STOP: fig4 node list blob"; exit 3; }
[ "$(head -60 opt3/opt3_nodes_A_rlext.txt | shasum -a 256 | cut -c1-64)" = "8f6f42788ceb2d48a375e8dec21106ee8859debca7f75e41d5a15c1ecad5702b" ] || { echo "STOP: A's 60 nodes changed"; exit 3; }
[ "$(head -106 opt3/rlext/a1_interval_scope_rlext_20261025.jsonl | shasum -a 256 | cut -c1-64)" = "957d9c75710f2f7eff3bcda0ad79df1a44550a19a59008461511dd4dd12f3c5d" ] || { echo "STOP: 2e's 106 scope lines changed"; exit 3; }
# --- input folders under ~ (fresh, read-only), every file sha-checked against the same pins as the r2.7 run
STAMP=$(TZ=America/Los_Angeles date '+%Y%m%d_%H%M%S')
CRC=$H/$STAMP/cr_cfg; OPC=$H/$STAMP/opt3_in
[ ! -e "$CRC" ] && [ ! -e "$OPC" ] || { echo "STOP: input folder exists"; exit 3; }
mkdir -p "$CRC" "$OPC"
for P in config/routec_A18_pins.json config/routec_BS18_pins.json; do
  jq -r '.files[] | .[0] + " " + (.[2] | split("/") | last)' "$P" | while read -r h n; do
    [ -f "$CRC/$n" ] || cp "$PREV/cr_cfg/$n" "$CRC/$n"
    [ "$(shasum -a 256 "$CRC/$n" | cut -c1-64)" = "$h" ] || { echo "STOP: pin $n"; exit 3; }
  done
done
while read -r h n; do
  [ -f "$CRC/$n" ] || cp "$PREV/cr_cfg/$n" "$CRC/$n"
  [ "$(shasum -a 256 "$CRC/$n" | cut -c1-64)" = "$h" ] || { echo "STOP: pin $n"; exit 3; }
done <<'EXTRA'
da33c4f0e3fce682908518cf230e369ab48ebe4358811b039595f8304daab13b override_S_old_R_2026-10-17_151100_gated.json
f8ee4ddc603dd9e38f0fd650055bcc9d324df903bbfd5a51809ca6b59ffceafd override_S_old_C_2026-10-17_151056_gated.json
2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27 simh_counted_keys_pilot_v3.json
EXTRA
[ "$(ls "$CRC" | wc -l | tr -d ' ')" = "27" ] || { echo "STOP: cr_cfg count"; exit 3; }
cp opt3/rlext/a1_interval_scope_rlext_20261025.jsonl "$OPC/"
cp /Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_inputs/opt3_calibration_paths_AB.json "$OPC/"
cp /Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_inputs/opt3_perdraw_npz_AB.json "$OPC/"
git show f35c8a0dc:experiments/2026-10-15_fpr_sweep_507_prep/results/fig1_main_figs156_routecA_source.json > "$OPC/fig1_main_figs156_routecA_source.json"
git show f35c8a0dc:experiments/2026-10-15_fpr_sweep_507_prep/results/fig1_main_R_figs156_routecA_source.json > "$OPC/fig1_main_R_figs156_routecA_source.json"
git show f35c8a0dc:experiments/2026-10-15_fpr_sweep_507_prep/results/fig4_pooled_fig4_routecB_source.json > "$OPC/fig4_pooled_fig4_routecB_source.json"
git show 6bee032a9:experiments/2026-10-17_routec_slot_readout/results/slots_routeC_writing_rc1a.json > "$OPC/slots_routeC_writing_rc1a.json"
while read -r h n; do [ "$(shasum -a 256 "$OPC/$n" | cut -c1-64)" = "$h" ] || { echo "STOP: pin $n"; exit 3; }; done <<'PINS'
4dbffaa36a5569b3f76dae0f701558357539c71bec2288c75849bc0e9d29e41b a1_interval_scope_rlext_20261025.jsonl
c1ea94312e9b5ce07a4d7f24f54ed0f28e7a4c83d9c45aa982deabe44480dbeb opt3_calibration_paths_AB.json
28c8302c7c2128e1d802ba9e7684d45fa25dddb631104a045fceb9c801aab111 opt3_perdraw_npz_AB.json
4f31f71b9d8b92fe25845d715c91d1dcfbd9910fbd9c11f8ab185b3cb97c74e1 fig1_main_figs156_routecA_source.json
69b30ddcd3fcd8db7d7d634e4a258849848ca00168792c68ccb25af202ed6536 fig1_main_R_figs156_routecA_source.json
0de398ad8a6e6115a67e473c42a772f4def525dae11a481e23654e4d504e9e5d fig4_pooled_fig4_routecB_source.json
3d7b036b445dca6b467ce709071c7003ab94fe3b854c0bc3d0f82ec4bedc9238 slots_routeC_writing_rc1a.json
PINS
[ "$(ls "$OPC" | wc -l | tr -d ' ')" = "7" ] || { echo "STOP: input folder count"; exit 3; }
chmod -R a-w "$CRC" "$OPC"
# live-app gate (fail closed), as the r2.7 RUN_ME
FILT='[.[] | select(.description == $a and .state != "stopped")] | length'
SCHEMA='type == "array" and all(.[]; type == "object" and ((.description | type) == "string") and ((.state | type) == "string"))'
FX1=$(jq -cn --arg a "$APP" '[{description: $a, state: "ephemeral"}, {description: "other-app", state: "ephemeral"}]')
FX0=$(jq -cn --arg a "$APP" '[{description: $a, state: "stopped"}]')
FXN1='[{"description": null, "state": "ephemeral"}]'
FXN2='["not-an-object"]'
[ "$(printf '%s' "$FX1" | jq --arg a "$APP" "$FILT")" = "1" ] || { echo "STOP: live filter fixture (live) != 1"; exit 3; }
[ "$(printf '%s' "$FX0" | jq --arg a "$APP" "$FILT")" = "0" ] || { echo "STOP: live filter fixture (stopped) != 0"; exit 3; }
printf '%s' "$FX1" | jq -e "$SCHEMA" > /dev/null || { echo "STOP: schema fixture (valid) rejected"; exit 3; }
printf '%s' "$FXN1" | jq -e "$SCHEMA" > /dev/null 2>&1 && { echo "STOP: schema fixture (null description) accepted"; exit 3; }
printf '%s' "$FXN2" | jq -e "$SCHEMA" > /dev/null 2>&1 && { echo "STOP: schema fixture (non-object entry) accepted"; exit 3; }
"$M" --version 2>&1 | grep -q '1\.6\.1' || { echo "STOP: modal binary is not 1.6.1"; exit 3; }
APPS=$("$M" app list --json)
printf '%s' "$APPS" | jq -e "$SCHEMA" > /dev/null || { echo "STOP: app list schema"; exit 3; }
LIVE=$(printf '%s' "$APPS" | jq --arg a "$APP" "$FILT")
[ "$LIVE" = "0" ] || { echo "STOP: $LIVE live apps"; exit 3; }
LOG=$H/$STAMP/launch.log
printf '%s\n' "$TRAILER" > "$LOG"
echo "launch $scr at $(git rev-parse --short HEAD) $(TZ=America/Los_Angeles date '+%H:%M:%S %Z'); inputs $CRC $OPC; log $LOG"
rc=0
CR_CFG="$CRC" OPT3_CFG="$OPC" "$M" run --detach $scr >> "$LOG" 2>&1 || rc=$?
grep -E "driver_call_id|STOP|Error|ap-[A-Za-z0-9]{10,}" "$LOG" | cut -c1-300 || true
echo "launch exit code $rc"
exit $rc
