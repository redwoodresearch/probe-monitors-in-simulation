#!/usr/bin/env bash
# il_nm_box.sh (RULE.md, "New mains"): start split S<k>'s interleaved launch (run3 + sim2 on the S_k main's cells) from the
# NEW-MAINS harness, derived from il311_box.sh v9 (Codex-reviewed) with only the nm changes: harness ~/phase2/nm/attack_check_nm
# and its guards in ~/phase2/nm/checks (tmux -L nm / -L nmguards, as the nm launches), references from the S_k main
# ~/phase2/nm/runs/2026-10-14_newmains_s<k> (it must have finished), cells checked against config/tasks_nm_s<k>_launch.json's
# r1 entries, the launcher command as canary.sh's (approaches ~/phase2/nm/approaches), run ~/phase2/nm/runs/2026-10-14_il_s<k>.
# Runs on whichever box holds the S_k main and harness (xbox2: s1, xbox3: s2, box 3: s3, sims-astra: s5; box 4: 4d =
# S4's 19 four-arm cells, references from the S4 main, cells checked against the staged tasks_nm_s4d_fu.json @ a5ce1226,
# sha256 a16c73db...; runs on any host with the nm harness and the S4 main relayed: xbox2 after s1).
#   bash ~/phase2/il_stage/il_nm_box.sh <k> <base URLs, comma-separated> <branch commit>
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
K=$1 BASES=$2 COMMIT=$3
case $K in
  1) CAP=1353; TSHA=ee3d865713f08d806116a48bec0c4c1b853f3e8e5dc54134c702e217d5e0fbf9; CSHA=f76c3d655142978c3f17f5e2b9abca2997b605fea6a059c9251336a6efd53b94 ;;
  2) CAP=1660; TSHA=ab9e06493f6573c7777fa3c9d78d9647945323875da9a912e64e89d4a9c22741; CSHA=bd8205856da7c2b5a0218c9bdd7e9a55a3643d05bf34e0799e6eb9f5a9038f43 ;;
  3) CAP=1271; TSHA=a1a4863d51d94bf3477b87a05c03fd10330beede313628c2aefd15ea2d7006ee; CSHA=2f64f5d766203baf6afeb0c7b2fcc1343d8bd60f013e97e36cd96a2b73c054b4 ;;
  5) CAP=1455; TSHA=c7979d88bd11f9b21efc51431af62231b19a6d78897071f0dde3f63f0392e457; CSHA=3b6daef8b1d97feeb31f1d2054510fe8edfefc4d9d21ada28a5817031eb02f96 ;;
  4d) CAP=389; TSHA=f51ab2049add49c42c7b48baf908958939f58ee07ddecf1d6c69039ae2e77fde; CSHA=0c3f2aa77517a0a4deaa9cacc7bf3152615759476227c16e34d51a2754f6622e ;;  # S4's 19 four-arm cells (the 4d set)
  *) echo "split $K: not prepared"; exit 2 ;;
esac
SRC=/home/aswin/phase2/nm/attack_check_nm; ST=/home/aswin/phase2/il_stage; CHECKS=/home/aswin/phase2/nm/checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s${K%d}; RUN=/home/aswin/phase2/nm/runs/2026-10-14_il_s$K; SESSION=ilnms$K
LOG=/home/aswin/phase2/il_s$K.log; TMP=$(mktemp -d)
MANIFEST=$SRC/config/harness_manifest_fu.tsv
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [il s$K] $*"
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
! has_session nm "$SESSION" || fail "session $SESSION is live"
# (a) the S_k main has finished (its launch.json says so); its references are what this launch forks from
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$MAIN/launch.json" \
  || fail "the S$K main $MAIN has not finished (or its launch.json can't be read)"
log "S$K main finished: $MAIN"
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
cp "$ST/pilot_il_s${K}.yaml" "$ST/tasks_il_s${K}.json" "$SRC/config/"
sha_ok "$SRC/config/tasks_il_s${K}.json" "$TSHA"
sha_ok "$SRC/config/pilot_il_s${K}.yaml" "$CSHA"
CELLSRC=-  # 4d: the 4d set's cell file, staged (not in the frozen harness on every host) and checked against its commit
if [[ $K == 4d ]]; then CELLSRC=$ST/tasks_nm_s4d_fu.json; sha_ok "$CELLSRC" a16c73db952dd77a415bea0ad686860b7e5ffd039eec9d80408d3e5ab9d112d9; fi
(cd "$SRC" && "$PY" - "$K" "$CELLSRC" <<'PY') || fail "config or task file"
import json, sys, yaml
k = sys.argv[1]
cfg = yaml.safe_load(open(f"config/pilot_il_s{k}.yaml"))
assert cfg["real_arms"] == ["attack"] and cfg["sim_arms"] == ["sim_attack"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_il_s{k}.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"]
cells = json.load(open(cfg["tasks"]))["cells"]
main = (json.load(open(sys.argv[2]))["cells"] if k == "4d"  # the 4d set (make_s4d.py @ a5ce1226): S4's 19 four-arm r1 cells
        else [c for c in json.load(open(f"config/tasks_nm_s{k}_launch.json"))["cells"] if "real_arms" not in c])
canon = lambda c: json.dumps(c, sort_keys=True)
assert sorted(map(canon, cells)) == sorted(map(canon, main)), "cells are not the main launch's r1 entries"
print(f"config: real attack + sim_attack@september; {len(cells)} cells = the main's r1 entries; cap {cfg['spend']['cap_usd']}")
PY
# 3. settings equality (as fu_launch.sh)
GS="results/gate_settings_il_s${K}_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS"
printf '{"launch": "il_s%s", "interleaved_launch": true, "worlds": {"attack": "run3", "sim_attack@september": "sim2"}, "references_from": "%s", "settings_dir": "/home/aswin/phase2/settings", "harness": "nm", "sim_attempt_timeout_s": 300, "cap_usd": %s, "commit": "%s"}\n' \
  "$K" "$MAIN" "$CAP" "$COMMIT" > "$RUN.il_meta.json"
# 4. the launcher, then the two reviewed guards (pause only); both must be alive 30 s later, or the launcher is
# interrupted and its exit CONFIRMED (its server-start checks take minutes, so no run has started by then). The
# exclusivity check is repeated right before the launch and 30 s after; later, the forks start no launcher on this box
# while il311q<k> runs (RULE.md). From the launch on, errexit is off and every failure goes through abort().
OURS="--out $RUN( |\$)"  # this launch's processes (pgrep/grep -E): the exact run dir, not a prefix of another
gone() {  # 0 = confirmed gone; 1 = still there; 2 = could not tell (an inspection error is never read as gone)
  local list rc=0; list=$(sessions nm) || return 2
  if in_list "$list" "$SESSION"; then return 1; fi
  pgrep -f -- "$OURS" > /dev/null || rc=$?
  case $rc in 0) return 1 ;; 1) return 0 ;; *) return 2 ;; esac
}
abort() {  # interrupt the launcher and CONFIRM it ended; escalate to kill-session; never report success unconfirmed
  log "ABORT: $1"
  local i j
  for i in 1 2; do
    tmux -L nm send-keys -t "=$SESSION:" C-c > /dev/null 2>&1 || true
    for j in $(seq 30); do gone; if [[ $? == 0 ]]; then log "launcher ended after Ctrl+C"; exit 1; fi; sleep 2; done
  done
  tmux -L nm kill-session -t "=$SESSION" > /dev/null 2>&1 || true
  for j in $(seq 15); do gone; if [[ $? == 0 ]]; then log "launcher session killed"; exit 1; fi; sleep 2; done
  log "LAUNCHER NOT CONFIRMED STOPPED after abort: $SESSION, $RUN (stop it by hand)"; exit 3
}
no_launcher
tmux -L nm new -d -s "$SESSION" "cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $SRC/run_pilot.py --pilot-config config/pilot_il_s${K}.yaml --base-urls $BASES --commit $COMMIT --approaches /home/aswin/phase2/nm/approaches --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L nmguards new -d -s "scilnms$K" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $((CAP + 100)) $((CAP + 400))" || abort "tmux new (sim_cap_stop) failed"
tmux -L nmguards new -d -s "gpilnms$K" "$PY $CHECKS/gate_pause_nm.py $RUN $SESSION" || abort "tmux new (gate_pause_nm) failed"
log "guards started: scilnms$K (pause at $((CAP + 100)), hard Ctrl+C at $((CAP + 400))), gpilnms$K"
sleep 30
list=$(sessions nm) || abort "cannot list tmux -L nm sessions"
in_list "$list" "$SESSION" || abort "the launcher session is gone (see $RUN.launcher.log)"
pgrep -f "sim_cap_stop.py $RUN $SESSION" > /dev/null || abort "sim_cap_stop is not running (or pgrep failed)"
pgrep -f "gate_pause_nm.py $RUN $SESSION" > /dev/null || abort "gate_pause_nm is not running (or pgrep failed)"
rc=0; all=$(pgrep -af 'run_pilot\.py') || rc=$?
[[ $rc == 0 ]] || abort "pgrep finds no launcher process at all (rc $rc)"
others=""; ours=0
while IFS= read -r line; do  # pure bash over pgrep's lines: ours match the exact run dir, anything else is another launcher
  if [[ $line =~ --out\ $RUN(\ |$) ]]; then ours=$((ours + 1)); elif [[ -n $line ]]; then others+="$line"$'\n'; fi
done <<< "$all"
(( ours > 0 )) || abort "no process of this launch found"
[[ -z $others ]] || abort "another launcher is running: ${others:0:300}"
log "STARTED: $SESSION, run $RUN, cap $CAP, guards alive, no other launcher"
