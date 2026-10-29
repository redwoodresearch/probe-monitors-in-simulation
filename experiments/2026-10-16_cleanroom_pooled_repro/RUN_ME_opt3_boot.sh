#!/bin/bash
# RUN_ME (option (3), main bootstrap; a1 wrote, 2e reads, 0d runs). Usage:
#   bash RUN_ME_opt3_boot.sh "Claude-Session: https://claude.ai/code/session_<id>"   ($1 = exactly one such line; logged)
# Run only after 09's GO and refresh3's ARMED line (cap $2.865288) with 0 live apps. Per attempt: workers <= 3960 s, driver <= 4560 s;
# Modal's input plane may retry internal failures independently of retries=0 (SDK 1.6.1 _functions.py:475); the overall spend is
# enforced by refresh3's arm cap ($2.865288, stop on over-cap). In-function entry deadlines: driver launch + 300 s, workers launch + 600 s.
# Builds two input folders under ~ (not /tmp), each file sha-checked against its pin, then ONE detached driver spawn.
set -euo pipefail
TRAILER=${1:?trailer required}
case "$TRAILER" in *$'\n'*) echo "STOP: trailer has more than one line"; exit 3;; esac
[[ "$TRAILER" =~ ^Claude-Session:\ https://claude\.ai/code/session_[A-Za-z0-9]+$ ]] || { echo "STOP: trailer format"; exit 3; }
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
R=$S/ps-repro; D=experiments/2026-10-16_cleanroom_pooled_repro; scr=cleanroom_boot_opt3.py
BLOB=c65bf420ba792a73d958aac9f52b6ce7577e271f
APP=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
H=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_run  # 09 #1: outside the worktree; off-limits to a1; only 0d runs this
cd "$R/$D"
git fetch -q origin
[ "$(git symbolic-ref --short HEAD)" = "aswin/repro-audit" ] || { echo "STOP: branch"; exit 3; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/aswin/repro-audit)" ] || { echo "STOP: HEAD != origin"; exit 3; }
for f in $scr cleanroom_reduce_v2.py config/routec_A18_pins.json config/routec_BS18_pins.json opt3/opt3_nodes_A.txt opt3/opt3_nodes_fig4.txt opt3/PREDICTIONS_r2_8.json opt3/opt3_d5_slots.txt opt3/opt3_record_path_templates.tsv ../../src/probe_shift/q1_eval.py; do
  git diff --quiet HEAD -- "$f" || { echo "STOP: $f modified"; exit 3; }
done
[ "$(git rev-parse HEAD:./$scr)" = "$BLOB" ] || { echo "STOP: blob"; exit 3; }
# Codex r1 #4: immutable pins of the reviewed configs and q1_eval.py (blobs), beside the script's
[ "$(git rev-parse HEAD:./config/routec_A18_pins.json)" = "ec99948ff772a9aa2fa1ab386aa31fec26a40e51" ] || { echo "STOP: A18 pins blob"; exit 3; }
[ "$(git rev-parse HEAD:./config/routec_BS18_pins.json)" = "cf84ee2abd5cb80260e26d57dd3657a7465d550a" ] || { echo "STOP: BS18 pins blob"; exit 3; }
[ "$(git rev-parse HEAD:src/probe_shift/q1_eval.py)" = "0a08acf42c671034f82efef268d8bb87b4039b32" ] || { echo "STOP: q1_eval blob"; exit 3; }
STAGED=$(git diff --cached --name-only) || { echo "STOP: git diff --cached failed"; exit 3; }
[ -z "$STAGED" ] || { echo "STOP: staged"; exit 3; }
# --- input folders under ~ (fresh, read-only), every file sha-checked
STAMP=$(TZ=America/Los_Angeles date '+%Y%m%d_%H%M%S')
CRC=$H/$STAMP/cr_cfg; OPC=$H/$STAMP/opt3_in  # 09 #1 (ii): cp + sha256 only; nothing here is read or printed
[ ! -e "$CRC" ] && [ ! -e "$OPC" ] || { echo "STOP: input folder exists"; exit 3; }
mkdir -p "$CRC" "$OPC"
for P in config/routec_A18_pins.json config/routec_BS18_pins.json; do
  jq -r '.files[] | .[0] + " " + (.[2] | split("/") | last)' "$P" | while read -r h n; do
    [ -f "$CRC/$n" ] || cp "$S/cleanroom/cfg_rc/$n" "$CRC/$n"
    [ "$(shasum -a 256 "$CRC/$n" | cut -c1-64)" = "$h" ] || { echo "STOP: pin $n"; exit 3; }
  done
done
# 09 (addendum 8f): the 3 /cfg inputs the reducer opens that neither files[] lists, cp + sha256 against INPUTS_FULL's own pins
while read -r h n; do
  [ -f "$CRC/$n" ] || cp "$S/cleanroom/cfg_rc/$n" "$CRC/$n"
  [ "$(shasum -a 256 "$CRC/$n" | cut -c1-64)" = "$h" ] || { echo "STOP: pin $n"; exit 3; }
done <<'EXTRA'
da33c4f0e3fce682908518cf230e369ab48ebe4358811b039595f8304daab13b override_S_old_R_2026-10-17_151100_gated.json
f8ee4ddc603dd9e38f0fd650055bcc9d324df903bbfd5a51809ca6b59ffceafd override_S_old_C_2026-10-17_151056_gated.json
2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27 simh_counted_keys_pilot_v3.json
EXTRA
[ "$(ls "$CRC" | wc -l | tr -d ' ')" = "27" ] || { echo "STOP: cr_cfg count"; exit 3; }
cp /Users/aswinsurya/probe-shift-routec-final-release-2026-10-20/v2_4r/q1/a1_interval_scope_20261021_004310.jsonl "$OPC/"
cp /Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_inputs/opt3_calibration_paths_AB.json "$OPC/"
cp /Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_inputs/opt3_perdraw_npz_AB.json "$OPC/"
git show f35c8a0dc:experiments/2026-10-15_fpr_sweep_507_prep/results/fig1_main_figs156_routecA_source.json > "$OPC/fig1_main_figs156_routecA_source.json"
git show f35c8a0dc:experiments/2026-10-15_fpr_sweep_507_prep/results/fig1_main_R_figs156_routecA_source.json > "$OPC/fig1_main_R_figs156_routecA_source.json"
git show f35c8a0dc:experiments/2026-10-15_fpr_sweep_507_prep/results/fig4_pooled_fig4_routecB_source.json > "$OPC/fig4_pooled_fig4_routecB_source.json"
git show 6bee032a9:experiments/2026-10-17_routec_slot_readout/results/slots_routeC_writing_rc1a.json > "$OPC/slots_routeC_writing_rc1a.json"
while read -r h n; do [ "$(shasum -a 256 "$OPC/$n" | cut -c1-64)" = "$h" ] || { echo "STOP: pin $n"; exit 3; }; done <<'PINS'
957d9c75710f2f7eff3bcda0ad79df1a44550a19a59008461511dd4dd12f3c5d a1_interval_scope_20261021_004310.jsonl
c1ea94312e9b5ce07a4d7f24f54ed0f28e7a4c83d9c45aa982deabe44480dbeb opt3_calibration_paths_AB.json
28c8302c7c2128e1d802ba9e7684d45fa25dddb631104a045fceb9c801aab111 opt3_perdraw_npz_AB.json
4f31f71b9d8b92fe25845d715c91d1dcfbd9910fbd9c11f8ab185b3cb97c74e1 fig1_main_figs156_routecA_source.json
69b30ddcd3fcd8db7d7d634e4a258849848ca00168792c68ccb25af202ed6536 fig1_main_R_figs156_routecA_source.json
0de398ad8a6e6115a67e473c42a772f4def525dae11a481e23654e4d504e9e5d fig4_pooled_fig4_routecB_source.json
3d7b036b445dca6b467ce709071c7003ab94fe3b854c0bc3d0f82ec4bedc9238 slots_routeC_writing_rc1a.json
PINS
[ "$(ls "$OPC" | wc -l | tr -d ' ')" = "7" ] || { echo "STOP: input folder count"; exit 3; }
chmod -R a-w "$CRC" "$OPC"
# live-app gate: schema (each entry an object with string .description and .state) fails closed; fixtures first
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
APPS=$(uv run --no-project --python 3.12 --with modal==1.6.1 modal app list --json)
printf '%s' "$APPS" | jq -e "$SCHEMA" > /dev/null || { echo "STOP: app list schema"; exit 3; }
LIVE=$(printf '%s' "$APPS" | jq --arg a "$APP" "$FILT")
[ "$LIVE" = "0" ] || { echo "STOP: $LIVE live apps"; exit 3; }
LOG=$H/$STAMP/launch.log
printf '%s\n' "$TRAILER" > "$LOG"
echo "launch $scr at $(git rev-parse --short HEAD) blob ${BLOB:0:8} $(TZ=America/Los_Angeles date '+%H:%M:%S %Z'); inputs $CRC $OPC; log $LOG"
rc=0
CR_CFG="$CRC" OPT3_CFG="$OPC" uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach $scr >> "$LOG" 2>&1 || rc=$?
grep -E "driver_call_id|STOP|Error|ap-[A-Za-z0-9]{10,}" "$LOG" | cut -c1-300 || true
echo "launch exit code $rc"
exit $rc
