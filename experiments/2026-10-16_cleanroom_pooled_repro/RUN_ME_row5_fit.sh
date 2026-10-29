#!/bin/bash
# RUN_ME (addendum 26 / 26b; a1 wrote, 2e reads, 0d runs): Table 7 row 5 (a_vs_b) clean-room refit = ONE detached fit_driver spawn.
# Run only after 09's launch OK and refresh3's ARMED line for the fit arm ($4.70) with 0 live apps. Prints ids, booleans and counts only.
# Code bound by bc (26b): 2*0.00148596*1560 + 3660*0.00001754 = $4.7003916.
set -euo pipefail
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
R=$S/ps-repro; D=experiments/2026-10-16_cleanroom_pooled_repro; scr=honest_refit_cleanroom.py
BLOB=c9cfd9e8cdfd7de8dce6c06bc74123942baa9cee
APP=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
cd "$R/$D"
git fetch -q origin
[ "$(git symbolic-ref --short HEAD)" = "aswin/repro-audit" ] || { echo "STOP: branch"; exit 3; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/aswin/repro-audit)" ] || { echo "STOP: HEAD != origin"; exit 3; }
git diff --quiet HEAD -- $scr || { echo "STOP: modified"; exit 3; }
[ "$(git rev-parse HEAD:./$scr)" = "$BLOB" ] || { echo "STOP: blob"; exit 3; }
[ -z "$(git diff --cached --name-only)" ] || { echo "STOP: staged"; exit 3; }
# live-app gate (2e, 10-20 21:5x): schema check that fails closed + two inline fixtures, then the real count
FILT='[.[] | select(.description == $a and .state != "stopped")] | length'
FX1=$(jq -cn --arg a "$APP" '[{description: $a, state: "ephemeral"}, {description: "other-app", state: "ephemeral"}]')
FX0=$(jq -cn --arg a "$APP" '[{description: $a, state: "stopped"}]')
[ "$(printf '%s' "$FX1" | jq --arg a "$APP" "$FILT")" = "1" ] || { echo "STOP: live filter fixture (live) != 1"; exit 3; }
[ "$(printf '%s' "$FX0" | jq --arg a "$APP" "$FILT")" = "0" ] || { echo "STOP: live filter fixture (stopped) != 0"; exit 3; }
APPS=$(uv run --no-project --python 3.12 --with modal==1.6.1 modal app list --json)
printf '%s' "$APPS" | jq -e 'type == "array" and all(.[]; has("description") and has("state"))' > /dev/null || { echo "STOP: app list schema"; exit 3; }
LIVE=$(printf '%s' "$APPS" | jq --arg a "$APP" "$FILT")
[ "$LIVE" = "0" ] || { echo "STOP: $LIVE live apps"; exit 3; }
mkdir -p "$S/row5"
LOG=$S/row5/fit_$(TZ=America/Los_Angeles date '+%Y-%m-%d_%H%M%S').log
echo "launch $scr --stage fit --problem a_vs_b at $(git rev-parse --short HEAD) blob ${BLOB:0:8} $(TZ=America/Los_Angeles date '+%H:%M:%S %Z'); log $LOG"
rc=0
uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach $scr --stage fit --stamp 2026-10-20_140924 --problem a_vs_b > "$LOG" 2>&1 || rc=$?
grep -E "fit_driver_call_id|ap-[A-Za-z0-9]{10,}|STOP|Error" "$LOG" | cut -c1-300 || true
echo "launch exit code $rc"
exit $rc
