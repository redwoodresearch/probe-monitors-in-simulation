#!/usr/bin/env bash
# ug_box.sh (experiments/2026-10-15_q3_ungrounded_sim/RULE.md): ON q3-xbox3, start the UNGROUNDED simulator arm
# (sim_attack@september_ungrounded on the interleaved pilot slice minus t139), derived from il_old_box.sh (il311_box.sh
# v9, Codex-reviewed) with only these changes: the harness COPY ~/phase2/attack_check_ug (build_ug.sh; its manifest's
# sha256 is an argument and checked), references from the merged index ~/phase2/il_refindex_ug (sha256 an argument),
# the config check (no real arm; simulators [september_ungrounded]; cells = the task file's 149, each with a pinned
# reference), the previous launch on xbox3 (s2) finished, run ug / session ugsim / guards scugsim and gpugsim, and the
# box launch lock. Every check fails closed before anything starts.
#   bash ~/phase2/ug_stage/ug_box.sh <base URLs, comma-separated> <branch commit> <index launch.json sha256> <harness_manifest_ug.tsv sha256>
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
K=ug BASES=$1 COMMIT=$2 RSHA=$3 MSHA=$4
[[ $RSHA =~ ^[0-9a-f]{64}$ && $MSHA =~ ^[0-9a-f]{64}$ ]] || { echo "usage: ug_box.sh <bases> <commit> <index sha256> <manifest sha256>"; exit 1; }
MAIN=/home/aswin/phase2/il_refindex_ug; CAP=3000; TSHA=cf89e595a37b85375ba1a84796718211d07b48a6e6f697c69a5faad72f76b80b; CSHA=73c525a46801afd40b5dc855c2a987662d37e16a7fa102eeb01df7e6d36f6974
SRC=/home/aswin/phase2/attack_check_ug; CHECKS=/home/aswin/phase2/s311_checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
RUN=/home/aswin/phase2/runs/2026-10-15_ug; SESSION=ugsim
LOG=/home/aswin/phase2/ug.log; TMP=$(mktemp -d)
MANIFEST=$SRC/config/harness_manifest_ug.tsv
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [ug] $*"
  echo "$line" 2>/dev/null || true; echo "$line" >> "$LOG" 2>/dev/null || true
}
fail() { log "UG GATE FAIL: $*"; exit 1; }
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
[[ $BASES =~ ^https://[^,]+(,https://[^,]+){3}$ ]] || fail "expected 4 base URLs"
absent "$RUN"
absent "$SRC/results/PAUSE_SIMS"
absent "/home/aswin/phase2/attack_check_311/results/PAUSE_SIMS"  # sim_cap_stop.py also writes the frozen harness's (its HARNESS constant)
! has_session phase2 "$SESSION" || fail "session $SESSION is live"
# (a) the box's previous launch, the s2 interleaved launch (nm harness), has FINISHED: its record says so and its
# session (tmux -L nm ilnms2) is gone
IL_PREV=/home/aswin/phase2/nm/runs/2026-10-14_il_s2
! has_session nm ilnms2 || fail "ilnms2 (s2) is still live"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$IL_PREV/launch.json" \
  || fail "$IL_PREV/launch.json is not finished (or unreadable)"
log "s2 finished: $IL_PREV/launch.json has finished; ilnms2 gone"
# (b) one launcher per settings dir: no run_pilot.py process at all on the box
no_launcher
log "no other run_pilot.py on the box"
# 1. the harness COPY is what build_ug.sh built (each producer's status checked)
sha_ok "$MANIFEST" "$MSHA"
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il_*' ! -path './config/tasks_il_*' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
grep -v '  ./config/harness_manifest' "$MANIFEST" > "$TMP/frozen.tsv" || fail "reading $MANIFEST failed"
[[ -s $TMP/now.tsv && -s $TMP/frozen.tsv ]] || fail "empty hash list"
cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv" || fail "$SRC differs from $MANIFEST"
log "freeze: $SRC matches $MANIFEST ($(wc -l < "$TMP/now.tsv") files)"
# 2. this launch's two files (installed by build_ug.sh, so also in the manifest) and the reference index
sha_ok "$SRC/config/tasks_ug_pilot.json" "$TSHA"
sha_ok "$SRC/config/pilot_ug.yaml" "$CSHA"
sha_ok "$MAIN/launch.json" "$RSHA"
(cd "$SRC" && "$PY" - "$MAIN/launch.json" <<'PY') || fail "config, task file or references"
import json, sys, yaml
cfg = yaml.safe_load(open("config/pilot_ug.yaml"))
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_attack"] and cfg["simulators"] == ["september_ungrounded"], cfg
assert cfg["tasks"] == "config/tasks_ug_pilot.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"] == 3000
sim = yaml.safe_load(open("config/run.yaml"))["simulators"]["september_ungrounded"]
assert sim["ungrounded"] is True and sim["env"] == {"BA_NO_HONEST_REFERENCE": "1"}, sim
cells = json.load(open(cfg["tasks"]))["cells"]
keys = [c["key"] for c in cells]
assert len(keys) == len(set(keys)) == 149 and not {"t98_model_registry", "t139_model_registry", "t151_model_registry"} & set(keys)
refs = json.load(open(sys.argv[1]))["references"]
assert set(keys) <= set(refs), sorted(set(keys) - set(refs))
print(f"config: sim_attack@september_ungrounded only; {len(cells)} cells, each with a pinned reference; cap {cfg['spend']['cap_usd']}")
PY
# 3. settings equality (as fu_launch.sh), on the FROZEN harness: its gate hashes run.yaml whole, and the copy's run.yaml
# differs by design. The copy is then shown equal to it: sim.yaml byte-identical, run.yaml equal once the one added
# simulator entry is removed (parsed), and the pinned september tree unchanged.
FROZEN=/home/aswin/phase2/attack_check_311
GS="results/gate_settings_ug_$(date +%s).json"
(cd "$FROZEN" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$FROZEN/$GS" || fail "settings gate"
cmp -s "$FROZEN/config/sim.yaml" "$SRC/config/sim.yaml" || fail "the copy's sim.yaml differs from the frozen one"
"$PY" - "$FROZEN/config/run.yaml" "$SRC/config/run.yaml" <<'PY' || fail "the copy's run.yaml differs beyond september_ungrounded"
import sys, yaml
frozen, copy = (yaml.safe_load(open(p)) for p in sys.argv[1:3])
added = copy["simulators"].pop("september_ungrounded")
assert copy == frozen and added["ungrounded"] is True, "run.yaml differs beyond the september_ungrounded entry"
PY
log "settings equality PASS (frozen harness); the copy's sim.yaml identical, run.yaml = frozen + september_ungrounded"
printf '{"launch": "%s", "ungrounded_arm": true, "worlds": {"sim_attack@september_ungrounded": "sim_ug"}, "references_from": "%s", "harness": "%s", "settings_dir": "/home/aswin/phase2/settings", "sim_attempt_timeout_s": 300, "cap_usd": %s, "commit": "%s"}\n' \
  "$K" "$MAIN" "$SRC" "$CAP" "$COMMIT" > "$RUN.ug_meta.json"
# 4. the launcher, then the two reviewed guards (pause only); both must be alive 30 s later, or the launcher is
# interrupted and its exit CONFIRMED (its server-start checks take minutes, so no run has started by then). The
# exclusivity check is repeated right before the launch and 30 s after; later, the forks start no launcher on this box
# while ugsim runs (RULE.md). From the launch on, errexit is off and every failure goes through abort().
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
# the box launch lock, held from the final checks through the launch and the +30 s check (fd 9 stays open until this
# script exits): another script using this lock cannot start a launcher in between; the pause files and the
# no-other-launcher scan are repeated inside it. Scripts that don't take the lock are covered by coordination (RULE.md).
exec 9> /home/aswin/phase2/.launch.lock || fail "cannot open the launch lock"
flock -n 9 || fail "another launch holds /home/aswin/phase2/.launch.lock"
absent "$SRC/results/PAUSE_SIMS"
absent "/home/aswin/phase2/attack_check_311/results/PAUSE_SIMS"
absent "$RUN"
no_launcher
tmux -L phase2 new -d -s "$SESSION" "cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES PYTHONPATH=$TREE RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $SRC/run_pilot.py --pilot-config config/pilot_ug.yaml --base-urls $BASES --commit $COMMIT --approaches /home/aswin/phase2/approaches --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L guards new -d -s "scugsim" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $((CAP + 100))" || abort "tmux new (sim_cap_stop) failed"
tmux -L guards new -d -s "gpugsim" "$PY $CHECKS/gate_pause_311.py $RUN $SESSION" || abort "tmux new (gate_pause_311) failed"
log "guards started: scugsim (at $((CAP + 100))), gpugsim"
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
