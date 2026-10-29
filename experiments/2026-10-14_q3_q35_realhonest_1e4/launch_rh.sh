#!/usr/bin/env bash
# ON THE DEVBOX, as the only command of tmux session rh1e4 (tmux -L phase2), under rr-env with the vLLM key only:
#   RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.vllm-only.tpl -- bash launch_rh.sh <base urls> <run dir>
# One launch of the 980 real honest runs (RULE.md, Amendment 2): the frozen launcher with the pinned images manifest,
# its log at <run dir>.launcher.log (never overwritten), then gate 1's per-replicate counts from the final launch.json
# (gate1.py final) and the upload manifests, written into the run folder before the session ends and the laptop
# watcher copies it. Exit: the launcher's code if non-zero, else 4 if gate1.py final failed, else 0.
set -uo pipefail
URLS=$1 RUN=$2
[[ $RUN == /home/aswin/phase2/runs_rh1e4/* && ! -e $RUN && ! -e $RUN.launcher.log ]] ||
  { echo "run dir must be new and under /home/aswin/phase2/runs_rh1e4/: $RUN" >&2; exit 2; }
mkdir -p /home/aswin/phase2/runs_rh1e4 || exit 2
set -C  # noclobber: the redirection fails rather than truncate a log that appeared meanwhile
exec > "$RUN.launcher.log" 2>&1 || exit 2
set +C
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
cd ~/phase2/attack_check_rh || exit 1
PY=~/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json  # sims-astra's, copied
PIN=20d4df7e120b9d3a06e92b49d5c68a884c93331cdc3b18f9ab613b32c0e874ce  # digest_sets_tagform_2026-10-08.json's
[[ $(sha256sum < "$IMAGES" | cut -d' ' -f1) == "$PIN" ]] || { log "$IMAGES is not the pinned manifest"; exit 2; }
log "launch -> $RUN"
PATH=$PWD/fixed_digest_lookup:$PATH IMAGES_JSON=$IMAGES "$PY" run_pilot.py --pilot-config config/pilot_honest_rh980.yaml \
  --base-urls "$URLS" --commit c42666d7 --out "$RUN"
rc=$?
log "launcher exit $rc"
"$PY" gate1.py final "$RUN"
grc=$?
log "gate1.py final exit $grc (1: a replicate below 177 usable of 196, or not all five present)"
(( rc != 0 )) && exit "$rc"
(( grc != 0 )) && exit 4
exit 0
