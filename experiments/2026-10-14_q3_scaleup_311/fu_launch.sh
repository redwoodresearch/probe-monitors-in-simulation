#!/usr/bin/env bash
# fu_launch.sh (RULE.md Amendment 7): ON THE DEVBOX, run by end_main.py after the quarter's main launch has ended. The
# follow-up's gates, then its launcher (tmux -L phase2 session f311q<N>) with --references-from <main run>. Exit 0 only if
# every gate passed and the launcher is running. Names, counts and hashes only.
#   fu_launch.sh <quarter> <main run dir> <follow-up run dir> <follow-up pilot config> <base URLs>
# Gates: 1. freeze: every harness file hashes as config/harness_manifest_fu.tsv says; 2. the follow-up config runs no real
# arm (real_arms []), only sim_attack@september, and reads config/tasks_311_q<N>_fu.json; 3. that file's cells are the
# main launch file's r1 entries, byte-identical (canonical JSON), all of them, and no r2 entry; 4. settings equality
# (gate_settings.py --harness .); 5. the follow-up run folder doesn't exist and no f311q<N> session is live.
set -euo pipefail
Q=$1 MAIN=$2 FU=$3 CFG=$4 BASES=$5
C=/home/aswin/phase2/attack_check_311; PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
export PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src
cd "$C"
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [fu Q$Q] $*"; }
fail() { log "FOLLOW-UP GATE FAIL: $*"; exit 1; }
[[ ! -e $FU ]] || fail "$FU exists"
! tmux -L phase2 has-session -t "=f311q$Q" 2>/dev/null || fail "a f311q$Q session is live"
diff <(find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' ! -path './config/harness_manifest.tsv' ! -path './config/harness_manifest_fu.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) config/harness_manifest_fu.tsv > /dev/null || fail "harness files differ from config/harness_manifest_fu.tsv"
log "freeze: every harness file matches config/harness_manifest_fu.tsv"
"$PY" - "$CFG" "$Q" <<'PY' || fail "follow-up config or tasks"
import json, sys, yaml
cfg, q = yaml.safe_load(open(sys.argv[1])), sys.argv[2]
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_attack"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_311_q{q}_fu.json", cfg["tasks"]
fu = json.load(open(cfg["tasks"]))["cells"]
main = [c for c in json.load(open(f"config/tasks_311_q{q}_launch.json"))["cells"] if "real_arms" not in c]
canon = lambda c: json.dumps(c, sort_keys=True)
assert [canon(c) for c in fu] == [canon(c) for c in main], "the follow-up cells are not the main launch's r1 entries"
assert not any("real_arms" in c or c["key"].endswith("_r2") for c in fu)
print(f"follow-up config: no real arm, sim_attack@september only; {len(fu)} cells = the main launch's r1 entries")
PY
GS="results/gate_settings_fu_q${Q}_$(date +%s).json"  # a fresh file every time (Codex round 12: no stale PASS)
[[ ! -e $GS ]] || fail "$GS exists"
set +e; "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" --harness . > "$GS.log" 2>&1; rc=$?; set -e
[[ $rc == 0 && -s $GS ]] || fail "settings gate exit $rc"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$GS" || fail "settings equality"
log "settings equality PASS"
printf '{"quarter": %s, "run1_launch": "%s", "sim_launch": "%s", "sim_launch_differs_from_run1": true, "references_from": "%s"}\n' \
  "$Q" "$(basename "$MAIN")" "$(basename "$FU")" "$MAIN" > "$FU.meta.json"
tmux -L phase2 new -d -s "f311q$Q" "cd $C && PATH=$C/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $C/run_pilot.py --pilot-config $CFG --base-urls $BASES --commit \$(cat $C/LAUNCH_COMMIT_FU) --approaches /home/aswin/phase2/approaches --references-from $MAIN --out $FU > $FU.launcher.log 2>&1"
sleep 15
tmux -L phase2 has-session -t "=f311q$Q" || fail "the follow-up launcher is not running"
log "follow-up launched: $FU (session f311q$Q), references from $MAIN"
