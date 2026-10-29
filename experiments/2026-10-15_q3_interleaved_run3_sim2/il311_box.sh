#!/usr/bin/env bash
# il311_box.sh v9 (RULE.md, "Overnight +311 launch"): ON BOX k, start that box's interleaved +311 launch (run3 + sim2) from
# the box's ORIGINAL 311 harness and settings (no copy: a copied settings dir renames control-tower's generated compose
# files and breaks the pinned references' digests, Codex r2 item 1). It refuses unless NO other run_pilot.py runs on the
# box (one launcher per settings dir). Every check fails closed before anything starts; names, counts and hashes only.
# Staged files (this script, pilot_il_q<k>_all.yaml, tasks_il_q<k>_all.json) are in ~/phase2/il_stage/.
#   bash ~/phase2/il_stage/il311_box.sh <k> <base URLs, comma-separated> <branch commit>
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
K=$1 BASES=$2 COMMIT=$3
case $K in
  1) MAIN=/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q1; CAP=1811; TSHA=d605f9b5ef145ba979956b00fbcb56e79fea4b88df945b1190625ce9846202c9; CSHA=1dab6eb4c0feb24d1d84ee2783373a73b6b945d0963f92c8ee1c5bb8d269eb00 ;;
  2) MAIN=/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q2; CAP=1747; TSHA=b5b2760cb7af1ad30671eb0a501f204a63853c7fd4ab669b16522519a8430e49; CSHA=e70edb16923bcba9ec418cfff77ff3ff08627756270af5ddab22642c6497aab9 ;;
  3) MAIN=/home/aswin/phase2/runs/2026-10-14_155838_scaleup311_q3; CAP=1721; TSHA=c612d676d206226ff2f0d2440fc2035b161fc89c049e3e212472263182d6bdfc; CSHA=1e38314127517cb8363dc278fa1b46a34cefc1658b71339a800bc369f323e04a ;;
  4) MAIN=/home/aswin/phase2/runs/2026-10-14_160246_scaleup311_q4; CAP=1937; TSHA=f683cf5d28ba8427516cc144bd738f3f61ef634d2479830e12ce215a7251d505; CSHA=37de54e3a59365cfe70ea9ab1e9445d2bf95e4513d2415311b5435456926d975 ;;
  *) echo "box $K: not a +311 box"; exit 2 ;;
esac
SRC=/home/aswin/phase2/attack_check_311; ST=/home/aswin/phase2/il_stage; CHECKS=/home/aswin/phase2/s311_checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
RUN=/home/aswin/phase2/runs/2026-10-14_il_q$K; SESSION=il311q$K; EM=/home/aswin/phase2/nm/checks/end_main_s$K.log
LOG=/home/aswin/phase2/il_q$K.log; TMP=$(mktemp -d)
MANIFEST=$SRC/config/harness_manifest_fu.tsv
if [[ $K == 2 ]]; then MANIFEST=$SRC/config/harness_manifest_fu_q2r.tsv; fi  # box 2's last launch
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [il q$K] $*"
  echo "$line" 2>/dev/null || true; echo "$line" >> "$LOG" 2>/dev/null || true
}
fail() { log "IL GATE FAIL: $*"; exit 1; }
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
# (a) the box's new-mains main has ended: a handover line in end_main's log, and no main session
[[ -r $EM ]] || fail "cannot read $EM"
if grep -q "FOLLOW-UP STARTED" "$EM"; then log "new-mains main ended: $(grep "FOLLOW-UP STARTED" "$EM" | tail -1 | cut -c1-90)"
elif grep -q "main ended" "$EM" && sed -n '/main ended/,$p' "$EM" | grep -q "NO FOLLOW-UP"; then log "new-mains main ended (no follow-up)"
else fail "the S$K main has not ended (no handover line in $EM)"; fi
! has_session nm "snms$K" || fail "the S$K main launcher session snms$K is live"
# (b) one launcher per settings dir: no run_pilot.py process at all on the box
no_launcher
log "no other run_pilot.py on the box"
# 1. the harness is still frozen (each producer's status checked; the staged il files are excluded by name)
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il_*' ! -path './config/tasks_il_*' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
grep -v '  ./config/harness_manifest' "$MANIFEST" > "$TMP/frozen.tsv" || fail "reading $MANIFEST failed"
[[ -s $TMP/now.tsv && -s $TMP/frozen.tsv ]] || fail "empty hash list"
cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv" || fail "$SRC differs from $MANIFEST"
log "freeze: $SRC matches $MANIFEST ($(wc -l < "$TMP/now.tsv") files)"
# 2. this launch's two files
cp "$ST/pilot_il_q${K}_all.yaml" "$ST/tasks_il_q${K}_all.json" "$SRC/config/"
sha_ok "$SRC/config/tasks_il_q${K}_all.json" "$TSHA"
sha_ok "$SRC/config/pilot_il_q${K}_all.yaml" "$CSHA"
(cd "$SRC" && "$PY" - "$K" <<'PY') || fail "config or task file"
import json, sys, yaml
k = sys.argv[1]
cfg = yaml.safe_load(open(f"config/pilot_il_q{k}_all.yaml"))
assert cfg["real_arms"] == ["attack"] and cfg["sim_arms"] == ["sim_attack"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_il_q{k}_all.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"]
cells = json.load(open(cfg["tasks"]))["cells"]
main = [c for c in json.load(open(f"config/tasks_311_q{k}_launch.json"))["cells"] if "real_arms" not in c]
canon = lambda c: json.dumps(c, sort_keys=True)
assert sorted(map(canon, cells)) == sorted(map(canon, main)), "cells are not the main launch's r1 entries"
print(f"config: real attack + sim_attack@september; {len(cells)} cells = the main's r1 entries; cap {cfg['spend']['cap_usd']}")
PY
# 3. settings equality (as fu_launch.sh)
GS="results/gate_settings_il_q${K}_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS"
printf '{"launch": "il_q%s", "interleaved_launch": true, "worlds": {"attack": "run3", "sim_attack@september": "sim2"}, "references_from": "%s", "settings_dir": "/home/aswin/phase2/settings", "sim_attempt_timeout_s": 300, "cap_usd": %s, "commit": "%s"}\n' \
  "$K" "$MAIN" "$CAP" "$COMMIT" > "$RUN.il_meta.json"
# 4. the launcher, then the two reviewed guards (pause only); both must be alive 30 s later, or the launcher is
# interrupted and its exit CONFIRMED (its server-start checks take minutes, so no run has started by then). The
# exclusivity check is repeated right before the launch and 30 s after; later, the forks start no launcher on this box
# while il311q<k> runs (RULE.md). From the launch on, errexit is off and every failure goes through abort().
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
tmux -L phase2 new -d -s "$SESSION" "cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES PYTHONPATH=$TREE RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $SRC/run_pilot.py --pilot-config config/pilot_il_q${K}_all.yaml --base-urls $BASES --commit $COMMIT --approaches /home/aswin/phase2/approaches --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L guards new -d -s "scil311q$K" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $((CAP + 100))" || abort "tmux new (sim_cap_stop) failed"
tmux -L guards new -d -s "gpil311q$K" "$PY $CHECKS/gate_pause_311.py $RUN $SESSION" || abort "tmux new (gate_pause_311) failed"
log "guards started: scil311q$K (at $((CAP + 100))), gpil311q$K"
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
