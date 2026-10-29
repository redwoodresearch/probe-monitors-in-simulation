#!/bin/bash
# RUN_ME (option (3), the record-B debug of OPT3 addenda 9 / 9a / 9b; a1 wrote, 2e reads, 0d runs). Usage:
#   bash RUN_ME_opt3_d9.sh "Claude-Session: https://claude.ai/code/session_<id>" diagB
#   bash RUN_ME_opt3_d9.sh "Claude-Session: https://claude.ai/code/session_<id>" fullB <cand 1..6> <opt3-results commit> <path> <sha256>
# Run only after 09's GO, the launch's own SPEND line and refresh3's ARMED line, with 0 live apps. Arm caps: diagB $0.3599895
# (bound $0.4704792: worker_B_diag <= 900 + 360 s, driver_diag <= 1200 + 360 s); fullB $0.719979 (bound $1.4726352: worker_B
# <= 3600 + 360 s, driver <= 4200 + 360 s). fullB (addendum 9b §2): the diagnostic's summary is read from origin's
# aswin/opt3-results (the commit an ancestor of its tip, fetched into a bare repo under the run folder, never into this worktree),
# its full sha256 must equal <sha256> BEFORE the gate, then the committed gate and 9a's precondition (run_mode diagB, baseline_ok,
# status "identified", identified = <cand>, that entry's draw0_ok); the driver re-checks the same bytes before the spawn.
# Builds the same two input folders as RUN_ME_opt3_boot.sh under ~ (not /tmp), each file sha-checked, then ONE detached spawn.
set -euo pipefail
TRAILER=${1:?trailer required}
MODE=${2:?run mode required (diagB or fullB)}
case "$TRAILER" in *$'\n'*) echo "STOP: trailer has more than one line"; exit 3;; esac
[[ "$TRAILER" =~ ^Claude-Session:\ https://claude\.ai/code/session_[A-Za-z0-9]+$ ]] || { echo "STOP: trailer format"; exit 3; }
S=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/3411bfa8-f1b8-4c15-bf10-536e101b5c37/scratchpad
R=$S/ps-repro; D=experiments/2026-10-16_cleanroom_pooled_repro; scr=cleanroom_boot_opt3.py
BLOB=25255f9a701e4866dfff038e8d22e121b1a0f456
APP=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
H=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_run  # 09 #1: outside the worktree; off-limits to a1; only 0d runs this
cd "$R/$D"
GIT_SSH_COMMAND="ssh -o ConnectTimeout=20 -o BatchMode=yes" git fetch -q origin aswin/repro-audit  # this branch only (never a bare fetch)
[ "$(git symbolic-ref --short HEAD)" = "aswin/repro-audit" ] || { echo "STOP: branch"; exit 3; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/aswin/repro-audit)" ] || { echo "STOP: HEAD != origin"; exit 3; }
for f in $scr cleanroom_reduce_v2.py config/routec_A18_pins.json config/routec_BS18_pins.json opt3/opt3_nodes_A.txt opt3/opt3_nodes_fig4.txt opt3/PREDICTIONS_r2_11.json opt3/opt3_d5_slots.txt opt3/opt3_record_path_templates.tsv opt3/opt3_summary_shape.jq opt3/opt3_summary_fixtures.sh ../../src/probe_shift/q1_eval.py; do
  git diff --quiet HEAD -- "$f" || { echo "STOP: $f modified"; exit 3; }
done
[ "$(git rev-parse HEAD:./$scr)" = "$BLOB" ] || { echo "STOP: blob"; exit 3; }
[ "$(git rev-parse HEAD:./opt3/opt3_summary_shape.jq)" = "1f771d7a8c72924e7c947a7eab0f1a41036a1fc9" ] || { echo "STOP: gate blob"; exit 3; }  # addendum 9b §2
[ "$(git rev-parse HEAD:./opt3/opt3_summary_fixtures.sh)" = "2c084d9555a94fc0177875b88e446d2108630e81" ] || { echo "STOP: fixture helper blob"; exit 3; }
[ "$(git rev-parse HEAD:./config/routec_A18_pins.json)" = "ec99948ff772a9aa2fa1ab386aa31fec26a40e51" ] || { echo "STOP: A18 pins blob"; exit 3; }
[ "$(git rev-parse HEAD:./config/routec_BS18_pins.json)" = "cf84ee2abd5cb80260e26d57dd3657a7465d550a" ] || { echo "STOP: BS18 pins blob"; exit 3; }
[ "$(git rev-parse HEAD:src/probe_shift/q1_eval.py)" = "0a08acf42c671034f82efef268d8bb87b4039b32" ] || { echo "STOP: q1_eval blob"; exit 3; }
STAGED=$(git diff --cached --name-only) || { echo "STOP: git diff --cached failed"; exit 3; }
[ -z "$STAGED" ] || { echo "STOP: staged"; exit 3; }
STAMP=$(TZ=America/Los_Angeles date '+%Y%m%d_%H%M%S')
[ ! -e "$H/$STAMP" ] || { echo "STOP: run folder exists"; exit 3; }
# --- the run mode; fullB is bound to the committed diagnostic summary (addenda 9a §3, 9b §2)
if [ "$MODE" = "diagB" ]; then
  [ "$#" = "2" ] || { echo "STOP: diagB takes no further arguments"; exit 3; }
  ARGS=(--run-mode diagB)
elif [ "$MODE" = "fullB" ]; then
  [ "$#" = "6" ] || { echo "STOP: fullB needs <cand> <opt3-results commit> <path> <sha256>"; exit 3; }
  CAND=$3; RCOMMIT=$4; RPATH=$5; RSHA=$6
  [[ "$CAND" =~ ^[1-6]$ ]] || { echo "STOP: cand not in 1..6"; exit 3; }
  [[ "$RCOMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "STOP: commit must be 40 hex"; exit 3; }
  [[ "$RSHA" =~ ^[0-9a-f]{64}$ ]] || { echo "STOP: sha256 must be 64 hex"; exit 3; }
  [[ "$RPATH" =~ ^([A-Za-z0-9_.-]+/)*opt3_summary_[0-9]{10,16}_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}\.json$ ]] && [[ "$RPATH" != *..* ]] || { echo "STOP: path form"; exit 3; }
  mkdir -p "$H/$STAMP"
  RB=$H/$STAMP/results_check.git; DSUM=$H/$STAMP/diag_summary.json  # a bare repo under 0d's run folder: no results ref enters this worktree
  git init -q --bare "$RB"
  GIT_SSH_COMMAND="ssh -o ConnectTimeout=20 -o BatchMode=yes" git -C "$RB" fetch -q "$(git remote get-url origin)" "refs/heads/aswin/opt3-results:refs/heads/tip" \
    || { echo "STOP: fetch aswin/opt3-results"; exit 3; }
  git -C "$RB" merge-base --is-ancestor "$RCOMMIT" tip || { echo "STOP: commit not on aswin/opt3-results"; exit 3; }
  git -C "$RB" show "$RCOMMIT:$RPATH" > "$DSUM" || { echo "STOP: path not in that commit"; exit 3; }
  [ "$(shasum -a 256 "$DSUM" | cut -c1-64)" = "$RSHA" ] || { echo "STOP: diagnostic summary sha256 differs"; exit 3; }  # BEFORE the gate
  NLINES="{\"boot\": $(git cat-file -p "$BLOB" | wc -l | tr -d " "), \"reducer\": $(git cat-file -p HEAD:./cleanroom_reduce_v2.py | wc -l | tr -d " ")}"
  bash opt3/opt3_summary_fixtures.sh || { echo "STOP: gate fixtures"; exit 3; }
  jq -e -s --rawfile nodesA opt3/opt3_nodes_A.txt --rawfile nodesB opt3/opt3_nodes_fig4.txt --slurpfile pred opt3/PREDICTIONS_r2_11.json \
    --rawfile slots opt3/opt3_d5_slots.txt --argjson nlines "$NLINES" -f opt3/opt3_summary_shape.jq "$DSUM" > /dev/null || { echo "STOP: diagnostic summary schema"; exit 3; }
  jq -e --argjson c "$CAND" '.run_mode == "diagB" and .fan_out == 1 and .driver_stop == null and (.workers[0].d9 as $d
      | $d != null and $d.baseline_ok == true and $d.status == "identified" and $d.identified == $c and $d.candidates[$c].draw0_ok == true)' "$DSUM" > /dev/null \
    || { echo "STOP: the diagnostic does not identify candidate $CAND with draw0_ok (no fullB)"; exit 3; }
  echo "diagnostic summary $RSHA at $RCOMMIT identifies #$CAND"
  ARGS=(--run-mode fullB --cand "$CAND" --diag-summary "$DSUM" --diag-sha "$RSHA")
else
  echo "STOP: run mode must be diagB or fullB"; exit 3
fi
# --- input folders under ~ (fresh, read-only), every file sha-checked (as RUN_ME_opt3_boot.sh, addendum 8f's 27 /cfg files)
CRC=$H/$STAMP/cr_cfg; OPC=$H/$STAMP/opt3_in  # 09 #1 (ii): cp + sha256 only; nothing here is read or printed
[ ! -e "$CRC" ] && [ ! -e "$OPC" ] || { echo "STOP: input folder exists"; exit 3; }
mkdir -p "$CRC" "$OPC"
for P in config/routec_A18_pins.json config/routec_BS18_pins.json; do
  jq -r '.files[] | .[0] + " " + (.[2] | split("/") | last)' "$P" | while read -r h n; do
    [ -f "$CRC/$n" ] || cp "$S/cleanroom/cfg_rc/$n" "$CRC/$n"
    [ "$(shasum -a 256 "$CRC/$n" | cut -c1-64)" = "$h" ] || { echo "STOP: pin $n"; exit 3; }
  done
done
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
echo "launch $scr ${ARGS[*]} at $(git rev-parse --short HEAD) blob ${BLOB:0:8} $(TZ=America/Los_Angeles date '+%H:%M:%S %Z'); inputs $CRC $OPC; log $LOG"
rc=0
CR_CFG="$CRC" OPT3_CFG="$OPC" uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach $scr "${ARGS[@]}" >> "$LOG" 2>&1 || rc=$?
grep -E "driver_call_id|STOP|Error|ap-[A-Za-z0-9]{10,}" "$LOG" | cut -c1-300 || true
echo "launch exit code $rc"
exit $rc
