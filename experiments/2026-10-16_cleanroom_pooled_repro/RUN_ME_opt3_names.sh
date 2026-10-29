#!/bin/bash
# RUN_ME (option (3), stage 0: npz member names only; a1 wrote, 2e reads, 0d runs).
# Usage: bash RUN_ME_opt3_names.sh "Claude-Session: https://claude.ai/code/session_<id>"   ($1 = exactly one such line; logged)
# Run only after 09's GO and refresh3's ARMED line for this stage with 0 live apps. Per attempt <= 420 s; Modal's input plane may retry
# internal failures independently of retries=0 (SDK 1.6.1 _functions.py:475); the overall spend is enforced by refresh3's arm cap
# ($0.0064344, stop on over-cap). The function also STOPs at entry if called later than launch + 420 s (OPT3 addendum 3 §1).
set -euo pipefail
TRAILER=${1:?trailer required}
case "$TRAILER" in *$'\n'*) echo "STOP: trailer has more than one line"; exit 3;; esac
[[ "$TRAILER" =~ ^Claude-Session:\ https://claude\.ai/code/session_[A-Za-z0-9]+$ ]] || { echo "STOP: trailer format"; exit 3; }
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
R=$S/ps-repro; D=experiments/2026-10-16_cleanroom_pooled_repro; scr=opt3_npz_names.py
BLOB=0a8e1876819c4913c281918dd87e5c5dfc668ffd
APP=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
cd "$R/$D"
git fetch -q origin
[ "$(git symbolic-ref --short HEAD)" = "aswin/repro-audit" ] || { echo "STOP: branch"; exit 3; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/aswin/repro-audit)" ] || { echo "STOP: HEAD != origin"; exit 3; }
git diff --quiet HEAD -- $scr || { echo "STOP: modified"; exit 3; }
[ "$(git rev-parse HEAD:./$scr)" = "$BLOB" ] || { echo "STOP: blob"; exit 3; }
STAGED=$(git diff --cached --name-only) || { echo "STOP: git diff --cached failed"; exit 3; }
[ -z "$STAGED" ] || { echo "STOP: staged"; exit 3; }
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
mkdir -p "$S/opt3"
LOG=$S/opt3/names_$(TZ=America/Los_Angeles date '+%Y-%m-%d_%H%M%S').log
printf '%s\n' "$TRAILER" > "$LOG"
echo "launch $scr at $(git rev-parse --short HEAD) blob ${BLOB:0:8} $(TZ=America/Los_Angeles date '+%H:%M:%S %Z'); log $LOG"
rc=0
uv run --no-project --python 3.12 --with modal==1.6.1 modal run $scr >> "$LOG" 2>&1 || rc=$?
grep -E "STOP|Error|wrote|ap-[A-Za-z0-9]{10,}" "$LOG" | cut -c1-300 || true
echo "launch exit code $rc"
exit $rc
