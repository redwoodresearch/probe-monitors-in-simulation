#!/usr/bin/env bash
# ON q3-scaleup4, as the only command of tmux session ep2s311 (tmux -L phase2), under rr-env with the vLLM key only:
#   RR_OWNER_ID=aswin-surya rr-env run --env-file ~/phase2/ep2_311/env.vllm-only.tpl -- bash ~/phase2/ep2_311/launch_ep2_311.sh <base urls> <run dir>
# Real honest epoch 2, +311 (RULE.md, amendment 1): launch_ep2_t196.sh on the +311 harness. The harness must hash exactly as its
# committed frozen manifest (the scale-up's prelaunch freeze gate, against ~/phase2/ep2_311/harness_manifest_311.tsv, pinned); the
# epoch-2 config, tasks file and Q4's 85-cell launch file live in ~/phase2/ep2_311/ (outside the harness) and are pinned here.
# Then the frozen launcher with the pinned images manifest and the canary's command (fixed_digest_lookup, --commit LAUNCH_COMMIT,
# --approaches), its log at <run dir>.launcher.log (never overwritten), then gate_ep2_arms.py final (per-arm counts and upload
# manifests). Exit: 2 if a pin or the freeze fails (nothing started), else the launcher's code if non-zero, else 4 if the final
# gate failed, else 0.
set -uo pipefail
URLS=$1 RUN=$2
[[ $RUN == /home/aswin/phase2/runs_ep2/* && ! -e $RUN && ! -e $RUN.launcher.log ]] ||
  { echo "run dir must be new and under /home/aswin/phase2/runs_ep2/: $RUN" >&2; exit 2; }
mkdir -p /home/aswin/phase2/runs_ep2 || exit 2
set -C
exec > "$RUN.launcher.log" 2>&1 || exit 2
set +C
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
C=/home/aswin/phase2/attack_check_311 S=/home/aswin/phase2/ep2_311
cd "$C" || exit 2
PY=~/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
pin() { [[ $(sha256sum < "$1" | cut -d' ' -f1) == "$2" ]] || { log "$1 is not the pinned file ($2)"; exit 2; }; }
pin "$IMAGES" 20d4df7e120b9d3a06e92b49d5c68a884c93331cdc3b18f9ab613b32c0e874ce
pin "$S/harness_manifest_311.tsv" 5d17a3966386446e41450a6c85e9948c4d7ee9e98ae7e424cb1c6d82ef874591
pin "$S/pilot_honest_ep2_311.yaml" 4f133b89fa8a1bcfb099025a7098a9088b24679e74764ff997dcdbccbdb3e4c3
pin "$S/tasks_311_q4_launch_85.json" 033d4bebf7bb93c78cada607f698e9ce32e157a6d808adb735d24574d6fc8a2e
[[ -s $S/pilot_tasks_ep2_311_309.json ]] || { log "no tasks file (make_tasks_ep2_311.py, stage_ep2_311.sh)"; exit 2; }  # its sha256: the staging record and launch.json
# the freeze gate (prelaunch_gates.sh's step 1): every harness file, and no other, hashes as the committed manifest says
diff <(find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort |
  xargs -d '\n' sha256sum) "$S/harness_manifest_311.tsv" > "$RUN.manifest_diff.txt" || { log "harness differs from the frozen manifest ($RUN.manifest_diff.txt)"; exit 2; }
log "freeze: all $(wc -l < "$S/harness_manifest_311.tsv") harness files match the frozen manifest; launch -> $RUN"
PATH=$C/fixed_digest_lookup:$PATH IMAGES_JSON=$IMAGES "$PY" "$C/run_pilot.py" --pilot-config "$S/pilot_honest_ep2_311.yaml" \
  --base-urls "$URLS" --commit "$(cat "$C/LAUNCH_COMMIT")" --approaches /home/aswin/phase2/approaches --out "$RUN"
rc=$?
log "launcher exit $rc"
"$PY" "$S/gate_ep2_arms.py" final "$RUN"
grc=$?
log "gate_ep2_arms.py final exit $grc (1: an arm below 90% usable of its runs, or an arm missing)"
(( rc != 0 )) && exit "$rc"
(( grc != 0 )) && exit 4
exit 0
