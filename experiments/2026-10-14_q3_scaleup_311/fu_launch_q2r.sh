#!/usr/bin/env bash
# fu_launch_q2r.sh (RULE.md Amendment 11): ON BOX 2, the Q2 RERUN's gates, then its launcher (tmux -L phase2 session f311q2r)
# with --references-from the Q2 main run. fu_launch.sh with: its own manifest; the 55-cell task file; its own session.
# Exit 0 only if every gate passed and the launcher is running. Names, counts and hashes only.
#   fu_launch_q2r.sh <Q2 main run dir> <Q2 first follow-up run dir> <rerun run dir> <base URLs>
# Gates: 1. freeze: every harness file hashes as config/harness_manifest_fu_q2r.tsv says; 2. the config runs no real arm,
# only sim_attack@september, reads config/tasks_311_q2r_fu.json, whose cells are exactly the Q2 main launch file's r1
# entries (byte-identical, canonical JSON) whose simulated run is "stopped" or "not run" in the first follow-up's FINAL
# launch.json (read here, sha256 pinned), 55 of them; 3. settings equality (gate_settings.py --harness .); 4. the rerun
# folder doesn't exist and no f311q2r session is live.
set -euo pipefail
MAIN=$1 FU1=$2 FU=$3 BASES=$4
CFG=config/pilot_311_fu_q2r_iii.yaml
C=/home/aswin/phase2/attack_check_311; PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
export PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src
cd "$C"
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [Q2 rerun] $*"; }
fail() { log "RERUN GATE FAIL: $*"; exit 1; }
[[ ! -e $FU ]] || fail "$FU exists"
! tmux -L phase2 has-session -t "=f311q2r" 2>/dev/null || fail "a f311q2r session is live"
diff <(find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' ! -path './config/harness_manifest.tsv' ! -path './config/harness_manifest_fu.tsv' ! -path './config/harness_manifest_fu_q2r.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) config/harness_manifest_fu_q2r.tsv > /dev/null || fail "harness files differ from config/harness_manifest_fu_q2r.tsv"
log "freeze: every harness file matches config/harness_manifest_fu_q2r.tsv"
"$PY" - "$CFG" "$FU1" <<'PY' || fail "rerun config or tasks"
import hashlib, json, sys, yaml
from pathlib import Path
cfg, fu1 = yaml.safe_load(open(sys.argv[1])), Path(sys.argv[2])
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_attack"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == "config/tasks_311_q2r_fu.json", cfg["tasks"]
raw = (fu1 / "launch.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == "6359c27350c0335ead56ea51b423ae6e9c86731e7822582edce376423ba533ba", "first follow-up record changed"
first = json.loads(raw)
assert first.get("finished")
redo = {r["task"] for r in first["runs"] if r["state"] in ("stopped", "not run")}
rr = json.load(open(cfg["tasks"]))["cells"]
main = [c for c in json.load(open("config/tasks_311_q2_launch.json"))["cells"] if "real_arms" not in c]
canon = lambda c: json.dumps(c, sort_keys=True)
assert [canon(c) for c in rr] == [canon(c) for c in main if c["key"] in redo], "the rerun cells are not the main's r1 entries left without a run"
assert len(rr) == len(redo) == 55 and not any("real_arms" in c or c["key"].endswith("_r2") for c in rr)
print(f"rerun config: no real arm, sim_attack@september only; {len(rr)} cells = the Q2 main's r1 entries stopped or not run in {fu1.name}")
PY
GS="results/gate_settings_fu_q2r_$(date +%s).json"
[[ ! -e $GS ]] || fail "$GS exists"
set +e; "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" --harness . > "$GS.log" 2>&1; rc=$?; set -e
[[ $rc == 0 && -s $GS ]] || fail "settings gate exit $rc"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$GS" || fail "settings equality"
log "settings equality PASS"
printf '{"quarter": 2, "run1_launch": "%s", "sim_launch": "%s", "sim_launch_differs_from_run1": true, "references_from": "%s", "rerun_of": "%s", "rerun_rule": "the 55 cells stopped or not run in rerun_of; selected on in-flight status, which correlates with run duration and possibly outcome; a separate launch from rerun_of"}\n' \
  "$(basename "$MAIN")" "$(basename "$FU")" "$MAIN" "$(basename "$FU1")" > "$FU.meta.json"
tmux -L phase2 new -d -s "f311q2r" "cd $C && PATH=$C/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $C/run_pilot.py --pilot-config $CFG --base-urls $BASES --commit \$(cat $C/LAUNCH_COMMIT_Q2R) --approaches /home/aswin/phase2/approaches --references-from $MAIN --out $FU > $FU.launcher.log 2>&1"
sleep 15
tmux -L phase2 has-session -t "=f311q2r" || fail "the rerun launcher is not running"
log "rerun launched: $FU (session f311q2r), references from $MAIN"
