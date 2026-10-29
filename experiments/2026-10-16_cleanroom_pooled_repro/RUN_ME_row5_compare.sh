#!/bin/bash
# RUN_ME (addendum 26 / 26d / 26e; Codex r3 + R1 r2 SHIP on blob f424f944; a1 wrote, 2e reads, 0d runs): Table 7 row 5 comparison, honest_refit_compare_row5.py, one CPU function.
# Usage: bash RUN_ME_row5_compare.sh <commit holding the committed fit-summary record>. Run only after refresh3's ARMED line for the
# compare arm ($0.013) with 0 live apps. Bound by bc: (120+300+60)*(0.0000131+4*0.00000222) = $0.0105504. Booleans and counts only.
# Gate (26d, 09): the record must show fixtures n == n_stopped == 52; a fixture that did not raise STOPs the container (no record).
set -euo pipefail
C=${1:?fit-summary commit}
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
R=$S/ps-repro; D=experiments/2026-10-16_cleanroom_pooled_repro; scr=honest_refit_compare_row5.py
BLOB=f424f944eafe6868a34247251a729aec48941965
FS=$D/results/honest_refit_fit_summary_a_vs_b_2026-10-20_140924.json
APP=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
cd "$R/$D"
git fetch -q origin
[ "$(git symbolic-ref --short HEAD)" = "aswin/repro-audit" ] || { echo "STOP: branch"; exit 3; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/aswin/repro-audit)" ] || { echo "STOP: HEAD != origin"; exit 3; }
git diff --quiet HEAD -- $scr || { echo "STOP: modified"; exit 3; }
[ "$(git rev-parse HEAD:./$scr)" = "$BLOB" ] || { echo "STOP: blob"; exit 3; }
[ -z "$(git diff --cached --name-only)" ] || { echo "STOP: staged"; exit 3; }
git merge-base --is-ancestor "$C" HEAD || { echo "STOP: fit-summary commit not on this branch"; exit 3; }
FSHA=$(git show "$C:$FS" | shasum -a 256 | cut -c1-64)
[ "${#FSHA}" = "64" ] || { echo "STOP: fit-summary sha"; exit 3; }
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
echo "launch $scr at $(git rev-parse --short HEAD) blob ${BLOB:0:8}; fit summary $C sha256 ${FSHA:0:16} $(TZ=America/Los_Angeles date '+%H:%M:%S %Z')"
rc=0
BEFORE=$(ls results/ | grep -c '^honest_refit_compare_row5_' || true)
HM_CFG=$S/cleanroom/cfg_hm uv run --no-project --python 3.12 --with modal==1.6.1 modal run $scr --fit-summary-sha "$FSHA" > "$S/row5/compare_run.log" 2>&1 || rc=$?
grep -E "STOP|Error|wrote" "$S/row5/compare_run.log" | cut -c1-300 || true; echo "compare exit code $rc"
[ "$(ls results/ | grep -c '^honest_refit_compare_row5_' || true)" = "$((BEFORE + 1))" ] || { echo "STOP: no new compare record (a container STOP writes none, so no verdict exists)"; exit 3; }
REC=$(ls -t results/honest_refit_compare_row5_*.json | head -1)
shasum -a 256 "$REC"
# 26d gate (09): all 52 removed-key fixtures stopped; only then is the verdict line read
jq -e '.fixtures.n == 52 and .fixtures.n_stopped == 52' "$REC" > /dev/null || { echo "STOP: fixtures gate failed; verdict not read"; exit 3; }
jq -c '{verdict, like_for_like, fixtures, printed: {n: .printed.n, n_match: .printed.n_match, boundary_warnings: .printed.boundary_warnings}, secondary: {n: .secondary.n, n_match: .secondary.n_match, breaches: .secondary.breaches}, fit_summary_gate, label}' "$REC"
exit $rc
