#!/usr/bin/env bash
# ON q3-realhonest, as the only command of tmux session ep2t196 (tmux -L phase2), under rr-env with the vLLM key only:
#   RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.vllm-only.tpl -- bash launch_ep2_t196.sh <base urls> <run dir>
# Real honest epoch 2, T196 (RULE.md): the 1e-4 job's launch_rh2.sh with only the config (pilot_honest_ep2_t196.yaml), the run
# folder (~/phase2/runs_ep2) and the final gate (gate_ep2.py) changed. The frozen launcher with the pinned images manifest, its log
# at <run dir>.launcher.log (never overwritten), then gate_ep2.py final (per-batch counts and upload manifests) before the session
# ends. Exit: the launcher's code if non-zero, else 4 if gate_ep2.py final failed, else 0.
set -uo pipefail
URLS=$1 RUN=$2
[[ $RUN == /home/aswin/phase2/runs_ep2/* && ! -e $RUN && ! -e $RUN.launcher.log ]] ||
  { echo "run dir must be new and under /home/aswin/phase2/runs_ep2/: $RUN" >&2; exit 2; }
mkdir -p /home/aswin/phase2/runs_ep2 || exit 2
set -C
exec > "$RUN.launcher.log" 2>&1 || exit 2
set +C
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
cd ~/phase2/attack_check_rh || exit 1
PY=~/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
PIN=20d4df7e120b9d3a06e92b49d5c68a884c93331cdc3b18f9ab613b32c0e874ce
[[ $(sha256sum < "$IMAGES" | cut -d' ' -f1) == "$PIN" ]] || { log "$IMAGES is not the pinned manifest"; exit 2; }
log "launch -> $RUN"
PATH=$PWD/fixed_digest_lookup:$PATH IMAGES_JSON=$IMAGES "$PY" run_pilot.py --pilot-config config/pilot_honest_ep2_t196.yaml \
  --base-urls "$URLS" --commit c42666d7 --out "$RUN"
rc=$?
log "launcher exit $rc"
"$PY" gate_ep2.py final "$RUN"
grc=$?
log "gate_ep2.py final exit $grc (1: a batch below 90% usable of its tasks, or a batch missing)"
(( rc != 0 )) && exit "$rc"
(( grc != 0 )) && exit 4
exit 0
