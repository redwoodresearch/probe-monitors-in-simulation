#!/usr/bin/env bash
# ON q3-realhonest, as the only command of tmux session ep2s311b (tmux -L phase2), under rr-env with the vLLM key only:
#   RR_OWNER_ID=aswin-surya rr-env run --env-file ~/phase2/ep2_311b/env.vllm-only.tpl -- bash ~/phase2/ep2_311b/launch_ep2_311b.sh <base urls> <run dir>
# Real honest epoch 2, +311 tail launch ep2_311b (RULE.md amendment 3): launch_ep2_311.sh with the staged folder ~/phase2/ep2_311b,
# its config and tasks file (launch 1's cells[265:], built from launch 1's tasks file, pinned here at 3b645674...), on the same
# frozen +311 harness at ~/phase2/attack_check_311 (freeze gate against the pinned manifest). Exit: 2 if a pin or the freeze fails
# (nothing started), else the launcher's code if non-zero, else 4 if the final gate failed, else 0.
set -uo pipefail
URLS=$1 RUN=$2
[[ $RUN == /home/aswin/phase2/runs_ep2/* && ! -e $RUN && ! -e $RUN.launcher.log ]] ||
  { echo "run dir must be new and under /home/aswin/phase2/runs_ep2/: $RUN" >&2; exit 2; }
mkdir -p /home/aswin/phase2/runs_ep2 || exit 2
set -C
exec > "$RUN.launcher.log" 2>&1 || exit 2
set +C
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
C=/home/aswin/phase2/attack_check_311 S=/home/aswin/phase2/ep2_311b
cd "$C" || exit 2
PY=~/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
pin() { [[ $(sha256sum < "$1" | cut -d' ' -f1) == "$2" ]] || { log "$1 is not the pinned file ($2)"; exit 2; }; }
pin "$IMAGES" 20d4df7e120b9d3a06e92b49d5c68a884c93331cdc3b18f9ab613b32c0e874ce
pin "$S/harness_manifest_311.tsv" 5d17a3966386446e41450a6c85e9948c4d7ee9e98ae7e424cb1c6d82ef874591
pin "$S/pilot_honest_ep2_311b.yaml" 313d06d46b659f1c0c4a9f3e914f99c7ed1f5c5b6cd385a1895a6f96ec1cce74
pin /home/aswin/phase2/ep2_311/pilot_tasks_ep2_311_309.json 3b645674937459b697a9fec427aa703b322082260b34f0485a22eb6ac37a8a84  # launch 1's file, the split's source
[[ -s $S/pilot_tasks_ep2_311b_44.json ]] || { log "no tasks file (make_tasks_ep2_311b.py, stage_ep2_311b.sh)"; exit 2; }  # its sha256: the staging record and launch.json
# the freeze gate (prelaunch_gates.sh's step 1): every harness file, and no other, hashes as the committed manifest says
diff <(find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort |
  xargs -d '\n' sha256sum) "$S/harness_manifest_311.tsv" > "$RUN.manifest_diff.txt" || { log "harness differs from the frozen manifest ($RUN.manifest_diff.txt)"; exit 2; }
log "freeze: all $(wc -l < "$S/harness_manifest_311.tsv") harness files match the frozen manifest; launch -> $RUN"
PATH=$C/fixed_digest_lookup:$PATH IMAGES_JSON=$IMAGES "$PY" "$C/run_pilot.py" --pilot-config "$S/pilot_honest_ep2_311b.yaml" \
  --base-urls "$URLS" --commit "$(cat "$C/LAUNCH_COMMIT")" --approaches /home/aswin/phase2/approaches --out "$RUN"
rc=$?
log "launcher exit $rc"
"$PY" "$S/gate_ep2_arms.py" final "$RUN"
grc=$?
log "gate_ep2_arms.py final exit $grc (1: an arm below 90% usable of its runs, or an arm missing)"
(( rc != 0 )) && exit "$rc"
(( grc != 0 )) && exit 4
exit 0
