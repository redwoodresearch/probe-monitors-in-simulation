#!/usr/bin/env bash
# ON THE DEVBOX, as the only command of tmux session sp493<X> (tmux -L phase2), under rr-env with the vLLM key only:
#   RR_OWNER_ID=aswin-surya rr-env run --env-file ~/phase2/sp493/env.vllm-only.tpl -- bash ~/phase2/sp493/launch_sp493.sh <A|B|C> <base urls> <run dir> <launch tasks sha256>
# S_paper493 real arms (launch_ep2_311.sh @ 474c95fee with this box's config and tasks file): the harness must hash exactly as
# its committed frozen +311 manifest (5d17a396...); the config, the tasks file and the gate live in ~/phase2/sp493/ (outside the
# harness) and are pinned here (the tasks file by the sha256 that prep_sp493.sh printed, the 4th argument). Then the
# frozen launcher with the pinned images manifest and the canary's command, its log at <run dir>.launcher.log (never
# overwritten), then gate_sp493.py final (per-group counts and upload manifests). Exit: 2 if a pin or the freeze fails (nothing
# started), else the launcher's code if non-zero, else 4 if the final gate failed, else 0.
set -uo pipefail
X=$1 URLS=$2 RUN=$3 TSHA=${4:-}
[[ $X == A || $X == B || $X == C ]] || exit 2
[[ $TSHA =~ ^[0-9a-f]{64}$ ]] || { echo "the 4th argument must be the launch tasks sha256" >&2; exit 2; }
[[ $RUN == /home/aswin/phase2/runs_sp493/* && ! -e $RUN && ! -e $RUN.launcher.log ]] ||
  { echo "run dir must be new and under /home/aswin/phase2/runs_sp493/: $RUN" >&2; exit 2; }
mkdir -p /home/aswin/phase2/runs_sp493 || exit 2
set -C
exec > "$RUN.launcher.log" 2>&1 || exit 2
set +C
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
C=/home/aswin/phase2/attack_check_311 S=/home/aswin/phase2/sp493
cd "$C" || exit 2
PY=~/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
pin() { [[ $(sha256sum < "$1" | cut -d' ' -f1) == "$2" ]] || { log "$1 is not the pinned file ($2)"; exit 2; }; }
pin "$IMAGES" 20d4df7e120b9d3a06e92b49d5c68a884c93331cdc3b18f9ab613b32c0e874ce
pin "$S/harness_manifest_311.tsv" 5d17a3966386446e41450a6c85e9948c4d7ee9e98ae7e424cb1c6d82ef874591
pin "$S/pilot_sp493_real_box$X.yaml" "$(cat "$S/pilot_sp493_real_box$X.yaml.sha256")"
pin "$S/tasks_sp493_box${X}_launch.json" "$TSHA"
# the freeze gate (prelaunch_gates.sh's step 1): every harness file, and no other, hashes as the committed manifest says.
# Box B only (xbox1): the committed exemption list freeze_exempt_boxB.tsv (pinned) names 28 EXTRA files (other launches'
# configs and fu_launch.sh, none read by this launch), each by exact "sha256  path" line; only exact matches are dropped, so a
# changed or unlisted extra still fails, and every manifest file must still match.
EX=$S/freeze_exempt_box$X.tsv
if [[ -e $EX ]]; then pin "$EX" "$(cat "$EX.sha256")"; else EX=/dev/null; fi
# Codex r3: hash into a file first, so a hashing failure aborts (pipefail) instead of hiding inside a process substitution
find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort |
  xargs -d '\n' sha256sum > "$RUN.harness_list.txt" || { log "hashing the harness failed"; exit 2; }
grep -vxF -f "$EX" "$RUN.harness_list.txt" > "$RUN.harness_list_kept.txt" || { log "no harness file left after the exemption"; exit 2; }
diff "$RUN.harness_list_kept.txt" "$S/harness_manifest_311.tsv" > "$RUN.manifest_diff.txt" || { log "harness differs from the frozen manifest ($RUN.manifest_diff.txt)"; exit 2; }
log "freeze: all $(wc -l < "$S/harness_manifest_311.tsv") harness files match the frozen manifest; launch box $X -> $RUN"
PATH=$C/fixed_digest_lookup:$PATH IMAGES_JSON=$IMAGES "$PY" "$C/run_pilot.py" --pilot-config "$S/pilot_sp493_real_box$X.yaml" \
  --base-urls "$URLS" --commit "$(cat "$C/LAUNCH_COMMIT")" --approaches /home/aswin/phase2/approaches --out "$RUN"
rc=$?
log "launcher exit $rc"
"$PY" "$S/gate_sp493.py" final "$RUN" "$S/tasks_sp493_box${X}_launch.json"
grc=$?
log "gate_sp493.py final exit $grc (1: a group below 90% usable of its runs, or a group missing)"
(( rc != 0 )) && exit "$rc"
(( grc != 0 )) && exit 4
exit 0
