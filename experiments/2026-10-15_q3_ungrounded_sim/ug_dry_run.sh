#!/usr/bin/env bash
# ug_dry_run.sh (RULE.md): ON q3-xbox3, run ug_dry_preflight.py under the box launch lock, only when ug_ready_check.sh's
# conditions hold (re-checked inside the same lock). Output: ~/phase2/attack_check_ug/results/ug_dry_<stamp>/.
#   bash ~/phase2/ug_stage/ug_dry_run.sh
set -euo pipefail
ST=/home/aswin/phase2/ug_stage; H=/home/aswin/phase2/attack_check_ug
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
(cd "$ST" && sha256sum -c --quiet SHA256SUMS) || { echo "FAIL: staged files differ from SHA256SUMS"; exit 1; }
exec 9> /home/aswin/phase2/.launch.lock
flock -n 9 || { echo "FAIL: the launch lock is held"; exit 1; }
# the readiness conditions, inside this lock (ug_ready_check.sh's own lock attempt would fail here, so they are inlined)
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" /home/aswin/phase2/nm/runs/2026-10-14_il_s2/launch.json \
  || { echo "FAIL: s2 not finished"; exit 1; }
out=""; rc=0; out=$(tmux -L nm ls -F '#S' 2>&1) || rc=$?
if (( rc == 0 )); then [[ $'\n'"$out"$'\n' != *$'\n'ilnms2$'\n'* ]] || { echo "FAIL: ilnms2 live"; exit 1; }
elif [[ $out != *"no server running"* && $out != "error connecting to /tmp/tmux-$UID/nm (No such file or directory)" ]]; then echo "FAIL: tmux -L nm: $out"; exit 1; fi
rc=0; live=$(pgrep -af 'run_pilot\.py') || rc=$?
[[ $rc == 1 ]] || { echo "FAIL: a run_pilot.py is running, or pgrep failed (rc $rc)"; exit 1; }
for f in "$H/results/PAUSE_SIMS" /home/aswin/phase2/attack_check_311/results/PAUSE_SIMS; do [[ ! -e $f && ! -L $f ]] || { echo "FAIL: $f exists"; exit 1; }; done
[[ -s $IMAGES ]] || { echo "FAIL: $IMAGES missing"; exit 1; }
OUT="$H/results/ug_dry_$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)"
cd "$H" && PATH="$H/fixed_digest_lookup:$PATH" IMAGES_JSON="$IMAGES" PYTHONDONTWRITEBYTECODE=1 "$PY" "$ST/ug_dry_preflight.py" --out-dir "$OUT"
