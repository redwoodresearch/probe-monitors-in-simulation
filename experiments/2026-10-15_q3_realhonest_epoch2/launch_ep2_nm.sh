#!/usr/bin/env bash
# ON q3-ep2-nm, as the only command of tmux session ep2snm (tmux -L phase2), under rr-env with the vLLM key only:
#   RR_OWNER_ID=aswin-surya rr-env run --env-file ~/phase2/ep2_nm/env.vllm-only.tpl -- bash ~/phase2/ep2_nm/launch_ep2_nm.sh <base urls> <run dir>
# Real honest epoch 2, nm360 (RULE.md, amendment 2): launch_ep2_311.sh on the frozen nm harness (~/phase2/nm/attack_check_nm, its
# committed 178-file manifest, LAUNCH_COMMIT 99c2dc38). The harness must hash exactly as that manifest (the freeze gate, against
# ~/phase2/ep2_nm/harness_manifest_nm.tsv, pinned); the epoch-2 config and tasks file live in ~/phase2/ep2_nm/ (outside the
# harness), the config pinned here. Then the frozen launcher with the pinned images manifest and the nm canary's command
# (fixed_digest_lookup, --commit LAUNCH_COMMIT; --approaches ~/phase2/approaches: the content-addressed files relayed by
# relay_approaches.sh, whose fixed path that is), its log at <run dir>.launcher.log (never overwritten), then gate_ep2_arms.py final
# (per-arm counts of the counted attempts, and upload manifests). Exit: 2 if a pin or the freeze fails (nothing started), else the
# launcher's code if non-zero, else 4 if the final gate failed, else 0.
set -uo pipefail
URLS=$1 RUN=$2
[[ $RUN == /home/aswin/phase2/runs_ep2/* && ! -e $RUN && ! -e $RUN.launcher.log ]] ||
  { echo "run dir must be new and under /home/aswin/phase2/runs_ep2/: $RUN" >&2; exit 2; }
mkdir -p /home/aswin/phase2/runs_ep2 || exit 2
set -C
exec > "$RUN.launcher.log" 2>&1 || exit 2
set +C
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
C=/home/aswin/phase2/nm/attack_check_nm S=/home/aswin/phase2/ep2_nm
cd "$C" || exit 2
PY=~/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
pin() { [[ $(sha256sum < "$1" | cut -d' ' -f1) == "$2" ]] || { log "$1 is not the pinned file ($2)"; exit 2; }; }
pin "$IMAGES" 20d4df7e120b9d3a06e92b49d5c68a884c93331cdc3b18f9ab613b32c0e874ce
pin "$S/harness_manifest_nm.tsv" 59bccf88aab48df1c2d562634d36137363e88c4f05667a44d431486cd4443a51
pin "$S/pilot_honest_ep2_nm.yaml" 9585a30c35783efd3689239c3ba72860447ca5ab5851c8bd949b5d3c8e9645e5
[[ -s $S/pilot_tasks_ep2_nm_351.json ]] || { log "no tasks file (make_tasks_ep2_nm.py, stage_ep2_nm.sh)"; exit 2; }  # its sha256: the staging record and launch.json
# the freeze gate (prelaunch_gates.sh's step 1): every harness file, and no other, hashes as the committed manifest says
diff <(find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort |
  xargs -d '\n' sha256sum) "$S/harness_manifest_nm.tsv" > "$RUN.manifest_diff.txt" || { log "harness differs from the frozen manifest ($RUN.manifest_diff.txt)"; exit 2; }
log "freeze: all $(wc -l < "$S/harness_manifest_nm.tsv") harness files match the frozen manifest; launch -> $RUN"
PATH=$C/fixed_digest_lookup:$PATH IMAGES_JSON=$IMAGES "$PY" "$C/run_pilot.py" --pilot-config "$S/pilot_honest_ep2_nm.yaml" \
  --base-urls "$URLS" --commit "$(cat "$C/LAUNCH_COMMIT")" --approaches /home/aswin/phase2/approaches --out "$RUN"
rc=$?
log "launcher exit $rc"
"$PY" "$S/gate_ep2_arms.py" final "$RUN"
grc=$?
log "gate_ep2_arms.py final exit $grc (1: an arm below 90% usable of its runs, or an arm missing)"
(( rc != 0 )) && exit "$rc"
(( grc != 0 )) && exit 4
exit 0
