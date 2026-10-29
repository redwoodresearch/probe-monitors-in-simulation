#!/usr/bin/env bash
# simh311_box.sh v2 (experiments/2026-10-15_q3_simhonest_311/RULE.md): ON A HOST BOX, start one +311 quarter's
# SIMULATED-HONEST launch (sim_honest@september only, forked at 0 from the Q<k> main's honest_a via --references-from) from
# the host's 311 harness and settings. Derived from experiments/2026-10-15_q3_interleaved_run3_sim2/il311_box.sh v9 @
# d7f43e2fd: same helpers, guards and abort path; changed only where marked "simh:". v2 runs on ANY host (the main's own box,
# or another box after relay_simh.sh relayed the main's honest-only inputs to the same paths). Every check fails closed. It
# refuses unless: the host's PREVIOUS launch (given) has finished; NO run_pilot.py and NO spend or pause guard
# (sim_cap_stop.py, gate_pause*.py) runs on the box (one launcher per settings dir: simulated runs write the compose file
# too, run_pilot.py:20/450/617); the harness equals one of the three committed frozen 311 manifests (staged, sha-pinned);
# the config, follow-up task file and the main's own task file (staged, sha-pinned) check out; and the quarter's committed
# generation manifest (09 Append 25f; staged, sha given) matches the main's pinned references cell by cell.
#   bash ~/phase2/simh_stage/simh311_box.sh <k> <6 base URLs, comma-separated> <branch commit> <host's previous run dir> \
#     <host label> <generation manifest sha256>
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
K=$1 BASES=$2 COMMIT=$3 PREV=$4 HOST=$5 GSHA=$6  # simh: previous launch, host label, generation manifest
case $K in  # simh: caps, the frozen follow-up task files' and the simh configs' sha256
  1) LSHA=ee413a10eb2c448575fa34a2eafad25428f0947e8ac3450dfbc7e2c7152e4124; MAIN=/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q1; CAP=2115; STOP=2215; TSHA=b5b02a57e6e4f02eeed7b7dc38ec7f48a1e8ab7669a4e9a97b98c6f23f860818; CSHA=ddae84b5e0704c21cce7e038e62b8445206e1f449bbada8d8817b030432b7e17 ;;
  2) LSHA=eb72f9532e42953fb813d6b226b2eb815c9d161dc3f3e47c5062694de97eaf4d; MAIN=/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q2; CAP=4120; STOP=2940; TSHA=31ee96143aa74b35eb898edb64d9c7b33b79ff779d6b8b84b360b754207bd1cf; CSHA=dace5410a1c408159eebcdf2b126ff3c22e7b18415c26c3f2de3f17795d9513e ;;
  3) LSHA=186633516c694d6087ad2454ee5f8f7cb854dd8bf319c1717682330673a916a9; MAIN=/home/aswin/phase2/runs/2026-10-14_155838_scaleup311_q3; CAP=1977; STOP=2077; TSHA=a477bb8ee83d0e29943caae41f8d43c5c7892aa620240f6095076461d0631305; CSHA=38be55521261d1a23ea88ff0008311c51af2ad99335bde33594671a82842d57e ;;
  4) LSHA=033d4bebf7bb93c78cada607f698e9ce32e157a6d808adb735d24574d6fc8a2e; MAIN=/home/aswin/phase2/runs/2026-10-14_160246_scaleup311_q4; CAP=2334; STOP=2434; TSHA=dab4bf2d00b890b2add9e7f9746f66dbf12457f11d66b5004f00599dd344122a; CSHA=942f88b0c1c34d9ae8d83b9658727e1ec93f5c0674b0b32105b37b306cdade87 ;;
  *) echo "box $K: not a +311 box"; exit 2 ;;
esac
SRC=/home/aswin/phase2/attack_check_311; ST=/home/aswin/phase2/simh_stage;  # simh: own stage dir
 CHECKS=/home/aswin/phase2/s311_checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
RUN=/home/aswin/phase2/runs/2026-10-15_simh_q$K; SESSION=simh311q$K  # simh
LOG=/home/aswin/phase2/simh_q$K.log; TMP=$(mktemp -d)
# simh: the three committed frozen 311 harness manifests (origin/aswin/q3-scaleup-311 @ cfdd656c96cd, config/), staged in $ST
FROZEN="harness_manifest_fu_q123.tsv:71fd27724a045d3df5d8dba9b6e63d5f86514377c025a11a57fd78f6bf1aa287
harness_manifest_fu_q2r.tsv:b8f813c9a1f4ff4371a981b3aea8a9406b7fa0143eda3212a8b053e8123a4992
harness_manifest_fu_q4.tsv:32ed72b3e3aa95b21a41ae0dfcf47cacb9efb6e760e9577536e49b57e898bd3d"
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [simh q$K] $*"
  echo "$line" 2>/dev/null || true; echo "$line" >> "$LOG" 2>/dev/null || true
}
fail() { log "SIMH GATE FAIL: $*"; exit 1; }
sessions() {  # the session names on a tmux socket; no server = none; any other tmux error returns 1 (callers fail)
  local out rc=0; out=$(tmux -L "$1" ls -F '#S' 2>&1) || rc=$?
  # no server = none: "no server running", or (a socket never created on this box) exactly this one message
  if (( rc == 0 )); then printf '%s\n' "$out"
  elif [[ $out == *"no server running"* || $out == "error connecting to /tmp/tmux-$UID/$1 (No such file or directory)" ]]; then :
  else echo "tmux -L $1: $out" >&2; return 1; fi
}
has_session() {  # 0 = live, 1 = absent; a tmux or grep error exits the script through fail
  local list rc=0; list=$(sessions "$1") || fail "cannot list tmux -L $1 sessions"
  in_list "$list" "$2"
}
no_launcher() {  # fails unless NO run_pilot.py process runs on the box (pgrep's own status checked; it excludes itself)
  local live rc=0; live=$(pgrep -af 'run_pilot\.py') || rc=$?
  (( rc == 0 || rc == 1 )) || fail "pgrep failed ($rc)"
  (( rc == 1 )) && [[ -z $live ]] || fail "another launcher is running: $(grep -o -- '--out [^ ]*' <<< "$live" | sort -u | tr '\n' ' ')"
}
in_list() {  # 0 if line $2 is in the newline-separated list $1 (pure bash: no redirection or process that could fail)
  [[ $'\n'"$1"$'\n' == *$'\n'"$2"$'\n'* ]]
}
absent() {  # fails unless nothing exists at path $1 (a symlink counts, even dangling) and its parent is inspectable
  [[ -d ${1%/*} && -r ${1%/*} && -x ${1%/*} ]] || fail "cannot inspect ${1%/*}"
  [[ ! -e $1 && ! -L $1 ]] || fail "$1 exists"
}
sha_ok() {  # $1 file, $2 expected sha256
  local h; h=$(sha256sum < "$1") || fail "sha256sum $1"; [[ ${h:0:64} == "$2" ]] || fail "$1: sha256 ${h:0:16}... != expected ${2:0:16}..."
}
[[ $BASES =~ ^https://[^,]+(,https://[^,]+){5}$ ]] || fail "expected 6 base URLs"
absent "$RUN"
absent "$SRC/results/PAUSE_SIMS"
! has_session phase2 "$SESSION" || fail "session $SESSION is live"
# (a) simh: the host's previous launch has FINISHED, and no spend or pause guard of any launch is still running (a spend
# guard's late harness-wide PAUSE_SIMS write could otherwise pause this launch: Codex r1 follow-up)
[[ $PREV == /home/aswin/phase2/* && -r $PREV/launch.json ]] || fail "cannot read the previous launch $PREV/launch.json"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$PREV/launch.json" \
  || fail "the previous launch $PREV has not finished"
log "host $HOST: previous launch $PREV finished"
rc=0; gl=$(pgrep -af 'sim_cap_stop\.py|gate_pause[a-z0-9_]*\.py') || rc=$?
(( rc == 1 )) && [[ -z $gl ]] || fail "a spend or pause guard is still running (pgrep rc $rc): ${gl:0:300}"
[[ $GSHA =~ ^[0-9a-f]{64}$ ]] || fail "generation manifest sha256 malformed"
[[ -r $MAIN/launch.json ]] || fail "the Q$K main is not on this host at $MAIN"
# (b) one launcher per settings dir: no run_pilot.py process at all on the box
no_launcher
log "no other run_pilot.py on the box"
# 1. the harness is still frozen: it must equal ONE of the three committed frozen manifests (each staged copy sha-pinned).
# Other launches' staged config files are excluded by name, as the interleaved scripts do: pilot_il*, tasks_il* (incl. the
# rate-limit relaunches' pilot_ilrr*/tasks_ilrr*), ilrr* and pilot_simh_*.
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il*' ! -path './config/tasks_il*' ! -path './config/ilrr*' ! -path './config/pilot_simh_*' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
[[ -s $TMP/now.tsv ]] || fail "empty hash list"
MATCH=""
while IFS=: read -r mf ms; do
  sha_ok "$ST/$mf" "$ms"
  grep -v '  ./config/harness_manifest' "$ST/$mf" > "$TMP/frozen.tsv" || fail "reading $ST/$mf failed"
  [[ -s $TMP/frozen.tsv ]] || fail "empty $mf"
  if cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv"; then MATCH=$mf; fi
done <<< "$FROZEN"
[[ -n $MATCH ]] || fail "$SRC equals none of the three committed frozen manifests"
log "freeze: $SRC matches $MATCH ($(wc -l < "$TMP/now.tsv") files)"
# 2. simh: this launch's config (staged) and the frozen follow-up task file (already in config/ and in the manifest)
cp "$ST/pilot_simh_q${K}.yaml" "$SRC/config/"
# simh: the Q<k> main's own task file: a separately staged copy, sha-pinned to the scale-up branch's (= the main box's),
# because hosts' frozen harnesses hold different Q4 launch files (Codex r2)
LTASK=$TMP/tasks_311_q${K}_launch.json
cp "$ST/tasks_311_q${K}_launch.json" "$LTASK" || fail "staged Q$K main task file"
sha_ok "$LTASK" "$LSHA"
# simh: the committed generation manifest (09 Append 25f), staged: its sha256 is the one given, and each pinned cell's
# selected reference .eval sha256 equals the main's pinned reference sha256 (the launcher then re-checks each copy)
GM=$(ls "$ST"/manifest_simh_q${K}_*.json 2>/dev/null | head -2) || true
[[ -n $GM && $(wc -l <<< "$GM") == 1 ]] || fail "need exactly one staged generation manifest for q$K"
sha_ok "$GM" "$GSHA"
"$PY" - "$GM" "$MAIN/launch.json" "$K" <<'PY' || fail "generation manifest does not match the main's references"
import json, sys
m, launch, k = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
assert m["quarter"] == f"q{k}" and m["references_from"] == sys.argv[2].rsplit("/", 1)[0], (m["quarter"], m["references_from"])
refs = launch["references"]
pinned = {c["task"]: c["eval_sha256"] for c in m["cells"] if c["status"] == "pinned"}
assert pinned and pinned == {t: r["sha256"] for t, r in refs.items() if t in pinned}, "per-cell sha256 mismatch"
assert set(pinned) == set(refs) & {c["task"] for c in m["cells"]}, "pinned set differs from the main's references"
print(f"generation manifest: {len(pinned)} pinned cells match the main's references")
PY
sha_ok "$SRC/config/tasks_311_q${K}_fu.json" "$TSHA"
sha_ok "$SRC/config/pilot_simh_q${K}.yaml" "$CSHA"
(cd "$SRC" && "$PY" - "$K" "$LTASK" <<'PY') || fail "config or task file"
import json, sys, yaml
k = sys.argv[1]
cfg = yaml.safe_load(open(f"config/pilot_simh_q{k}.yaml"))
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_honest"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_311_q{k}_fu.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"]
assert cfg["spend"]["worst_per_sim_run_usd"] == 106, cfg["spend"]
cells = json.load(open(cfg["tasks"]))["cells"]
main = [c for c in json.load(open(sys.argv[2]))["cells"] if "real_arms" not in c]
canon = lambda c: json.dumps(c, sort_keys=True)
assert sorted(map(canon, cells)) == sorted(map(canon, main)), "cells are not the main launch's r1 entries"
print(f"config: sim_honest@september only; {len(cells)} cells = the main's r1 entries; cap {cfg['spend']['cap_usd']}")
PY
# 3. settings equality (as fu_launch.sh)
GS="results/gate_settings_simh_q${K}_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS"
printf '{"launch": "simh_q%s", "world": "simh_311", "arm": "sim_honest@september", "stratum": "S_new (+311)", "host": "%s", "previous_launch": "%s", "references_from": "%s", "generation_manifest_sha256": "%s", "frozen_manifest": "%s", "settings_dir": "/home/aswin/phase2/settings", "sim_attempt_timeout_s": 300, "cap_usd": %s, "commit": "%s"}\n' \
  "$K" "$HOST" "$PREV" "$MAIN" "$GSHA" "$MATCH" "$CAP" "$COMMIT" > "$RUN.simh_meta.json"
# 4. the launcher, then the two reviewed guards (pause only); both must be alive 30 s later, or the launcher is
# interrupted and its exit CONFIRMED (its server-start checks take minutes, so no run has started by then). The
# exclusivity check is repeated right before the launch and 30 s after; later, the forks start no launcher on this box
# while simh311q<k> runs (RULE.md). From the launch on, errexit is off and every failure goes through abort().
OURS="--out $RUN( |\$)"  # this launch's processes (pgrep/grep -E): the exact run dir, not a prefix of another
gone() {  # 0 = confirmed gone; 1 = still there; 2 = could not tell (an inspection error is never read as gone)
  local list rc=0; list=$(sessions phase2) || return 2
  if in_list "$list" "$SESSION"; then return 1; fi
  pgrep -f -- "$OURS" > /dev/null || rc=$?
  case $rc in 0) return 1 ;; 1) return 0 ;; *) return 2 ;; esac
}
abort() {  # interrupt the launcher and CONFIRM it ended; escalate to kill-session; never report success unconfirmed
  log "ABORT: $1"
  local i j
  for i in 1 2; do
    tmux -L phase2 send-keys -t "=$SESSION:" C-c > /dev/null 2>&1 || true
    for j in $(seq 30); do gone; if [[ $? == 0 ]]; then log "launcher ended after Ctrl+C"; exit 1; fi; sleep 2; done
  done
  tmux -L phase2 kill-session -t "=$SESSION" > /dev/null 2>&1 || true
  for j in $(seq 15); do gone; if [[ $? == 0 ]]; then log "launcher session killed"; exit 1; fi; sleep 2; done
  log "LAUNCHER NOT CONFIRMED STOPPED after abort: $SESSION, $RUN (stop it by hand)"; exit 3
}
no_launcher
tmux -L phase2 new -d -s "$SESSION" "cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES PYTHONPATH=$TREE RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $SRC/run_pilot.py --pilot-config config/pilot_simh_q${K}.yaml --base-urls $BASES --commit $COMMIT --approaches /home/aswin/phase2/approaches --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L guards new -d -s "scsimhq$K" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $STOP" || abort "tmux new (sim_cap_stop) failed"
tmux -L guards new -d -s "gpsimhq$K" "$PY $CHECKS/gate_pause_311.py $RUN $SESSION" || abort "tmux new (gate_pause_311) failed"
log "guards started: scsimhq$K (at $STOP), gpsimhq$K"
sleep 30
list=$(sessions phase2) || abort "cannot list tmux -L phase2 sessions"
in_list "$list" "$SESSION" || abort "the launcher session is gone (see $RUN.launcher.log)"
pgrep -f "sim_cap_stop.py $RUN $SESSION" > /dev/null || abort "sim_cap_stop is not running (or pgrep failed)"
pgrep -f "gate_pause_311.py $RUN $SESSION" > /dev/null || abort "gate_pause_311 is not running (or pgrep failed)"
rc=0; all=$(pgrep -af 'run_pilot\.py') || rc=$?
[[ $rc == 0 ]] || abort "pgrep finds no launcher process at all (rc $rc)"
others=""; ours=0
while IFS= read -r line; do  # pure bash over pgrep's lines: ours match the exact run dir, anything else is another launcher
  if [[ $line =~ --out\ $RUN(\ |$) ]]; then ours=$((ours + 1)); elif [[ -n $line ]]; then others+="$line"$'\n'; fi
done <<< "$all"
(( ours > 0 )) || abort "no process of this launch found"
[[ -z $others ]] || abort "another launcher is running: ${others:0:300}"
log "STARTED: $SESSION, run $RUN, cap $CAP, guards alive, no other launcher"
