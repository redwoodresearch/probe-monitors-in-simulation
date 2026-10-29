#!/usr/bin/env bash
# unpause_one.sh (0d ~13:08 via d9: unpause delegated, one launch at a time, on an objective census condition): lift the RL
# pause of ONE registered sim-honest run dir by renaming its PAUSE_SIMS aside (PAUSE_SIMS.lifted_<epoch>; nothing deleted;
# the launcher checks the exact name). Box side: unpause_remote.sh through stdin. It fails closed unless the pause is ours
# alone and no guard has ever paused the run; a detached box-side job owns the final checks, the rename and the 3 s / 150 s
# rechecks that restore PAUSE_SIMS on any other guard's sign (Codex r19, r19b, r19c, r19d). LIFTED only once the box-side job reports
# OK at 150 s and PAUSE_SIMS is absent.
#   unpause_one.sh <instance> <run dir>
set -euo pipefail
INST=$1 RUN=$2
[[ $INST =~ ^i-[0-9a-f]+$ && $RUN =~ ^/home/aswin/phase2/runs/2026-10-15_simh_((q[1-4]|nm_(s1|s2|s3|s4d|s4r|s5))(_rlr|_rlr2)?|sp493_t196|t196fill|sp493_new)$ ]] || { echo "bad args"; exit 2; }
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
R=$(cd "$(dirname "$0")" && pwd)/unpause_remote.sh
N=${RUN##*/2026-10-15_simh_}
if [[ $N == q* ]]; then HARN=/home/aswin/phase2/attack_check_311; SES=simh311${N}; else HARN=/home/aswin/phase2/nm/attack_check_nm; SES=simhnm${N#nm_}; fi
[[ $N == *_rlr || $N == *_rlr2 ]] && SES=simh${N//_/}  # replacement launches (simh_rlr_box.sh): session simh<launch without underscores>
[[ $N == sp493_* || $N == t196fill ]] && HARN=/home/aswin/phase2/attack_check_311 SES=simh${N//_/}  # S_paper493 (simh_sp493_box.sh): +311 harness
box() { (cd $SP/simh_launch_X/h && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 60; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "bash -s -- $*") < "$R" 2>&1 | grep -E '^(LIFT_STARTED|RESULT|NOW|NO |NOT )' || true; }
pre=$(box lift "$RUN" "$HARN" "$SES"); echo "$pre"
[[ $pre =~ LIFT_STARTED\ ([0-9]+) ]] || { echo "NOT LIFTED"; exit 1; }
TS=${BASH_REMATCH[1]}
sleep 160
res=$(box result "$RUN" "$TS"); echo "$res"
if [[ $res == *"RESULT OK 150s"* && $res == *"NOW PAUSE_SIMS absent"* && $res != *RESTORED* ]]; then
  echo "LIFTED $(TZ=America/Los_Angeles date '+%F %T PT') $RUN (box-side checks OK at 3 s and 150 s)"
else echo "NOT LIFTED or uncertain: see above (the box-side job restores PAUSE_SIMS on its own)"; exit 1; fi
