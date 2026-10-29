#!/usr/bin/env bash
# sim2gap_box.sh: ON an nm box whose LIVE harness equals the IL frozen nm manifest by full hash (xbox2), start the sim2 gap-fill
# part s4r: ONE simulated attack (sim2) per cell for the 59 S4 (4r-owned) cells; no real arm; references from the S4 (4r) main,
# relayed by relay_simh.sh and pinned by key (config/sim2gap/sim2gap_relay_allowlist_s4r.json). A DESCRIPTIVE add-on registered
# by 09 @ b9081cd57 (+ 4d44c4e2f): folder ilnm_s4rgap_sim2_v1__sim_attack, run 2026-10-15_sim2gap_s4r, stage "gap"; outcomes
# unread until 09's analysis note. Derived from il_nm_box.sh @ 2fc7f0104 (its helpers, freeze, settings gate, guards and
# confirmed start verbatim); changed: sim-only config check; the freeze manifest pinned by full sha256 and the box's staged IL
# rerun files exempted by EXACT path + pinned sha256; every launch.json on the box finished; references pinned by key.
#   bash ~/phase2/il_stage/sim2gap_box.sh s4r <6 base URLs, comma-separated> <branch commit>
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
K=$1 BASES=$2 COMMIT=$3
[[ $K == s4r ]] || { echo "part $K: not prepared"; exit 2; }
CAP=2100; TSHA=699376f532fabf5054796f69a896e173da6d36530c97c5a1311883a8ef84e8f9; CSHA=05e9cb21fad0d0ace22b1a4584b91a7565e8fdacf6983b69f0c24245d520607b; ASHA=e7cccdc4157b9312871a1adcd05f1c7d7640099765d3ab4083015fb37390e7aa
MSHA=18fbbaa225bbb0f70890f1436c17d254d676495136f8bd2aa616cb626281b7ed  # the IL frozen nm manifest (equal on box3, ep2nm, xbox2 at 16:4x)
MAINSHA=d2f091cbdf39f9de21fb3fb05d57754a9693dc61023bfcd85fc61e82fdf12719  # newmains_s4r launch.json (s4 source map @ b68ab0a53)
EXEMPT="ilrr_rerun_of_s1.json:12f54ee5421502b20584a24fcbe82eb370fcb25238287cbbb9f78ad239ce51a2 tasks_ilrr_s1.json:7b03f29dec3883650930e62946df94970f6c65563f7a3841b6afa968b407cef9 pilot_ilrr2_s1.yaml:e76c0d79e1257f42ddd003be26d398f1ce2176f3d5b160c1f383057a6329f35e"  # xbox2's staged IL s1 rerun files (= their commits)
SRC=/home/aswin/phase2/nm/attack_check_nm; ST=/home/aswin/phase2/il_stage; CHECKS=/home/aswin/phase2/nm/checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r; RUN=/home/aswin/phase2/nm/runs/2026-10-15_sim2gap_s4r; SESSION=sim2gaps4r
LOG=/home/aswin/phase2/sim2gap_s4r.log; TMP=$(mktemp -d)
MANIFEST=$SRC/config/harness_manifest_fu.tsv
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [sim2gap $K] $*"
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
# (a) the S4 (4r) main: the pinned launch.json, finished
sha_ok "$MAIN/launch.json" "$MAINSHA"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$MAIN/launch.json" \
  || fail "the main $MAIN has not finished (or its launch.json can't be read)"
log "S4 (4r) main pinned and finished: $MAIN"
# (a2) EVERY launch on this box has finished (as il_sel97_box.sh v2): parents proven inspectable, run dirs enumerated
[[ -d /home/aswin/phase2 && -r /home/aswin/phase2 && -x /home/aswin/phase2 ]] || fail "cannot inspect /home/aswin/phase2"
for q in /home/aswin/phase2/runs /home/aswin/phase2/nm /home/aswin/phase2/nm/runs; do
  [[ ( ! -e $q && ! -L $q ) || ( -d $q && -r $q && -x $q ) ]] || fail "cannot inspect $q"
done
shopt -s nullglob; RDIRS=(/home/aswin/phase2/runs/*/ /home/aswin/phase2/nm/runs/*/); shopt -u nullglob
for d in "${RDIRS[@]}"; do
  [[ -r $d && -x $d ]] || fail "cannot inspect $d"
  l=${d}launch.json
  [[ -e $l || -L $l ]] || continue
  [[ -f $l && -r $l ]] || fail "$l is not a readable regular file"
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$l" \
    || fail "${d%/} has not finished (or its launch.json can't be read)"
done
log "every launch.json under runs/ and nm/runs/ reports finished (${#RDIRS[@]} run dirs inspected)"
# (b) one launcher per settings dir: no run_pilot.py process at all on the box
no_launcher
log "no other run_pilot.py on the box"
# 1. the live harness equals the IL frozen nm manifest (pinned by full sha256); staged files excluded by name, this box's IL
# rerun files by EXACT path, each then required absent or at its pinned sha256
sha_ok "$MANIFEST" "$MSHA"
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il_*' ! -path './config/tasks_il_*' \
  ! -path './config/pilot_sim2gap_s4r.yaml' ! -path './config/tasks_sim2gap_s4r.json' \
  ! -path './config/ilrr_rerun_of_s1.json' ! -path './config/tasks_ilrr_s1.json' ! -path './config/pilot_ilrr2_s1.yaml' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
grep -v '  ./config/harness_manifest' "$MANIFEST" > "$TMP/frozen.tsv" || fail "reading $MANIFEST failed"
[[ -s $TMP/now.tsv && -s $TMP/frozen.tsv ]] || fail "empty hash list"
cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv" || fail "$SRC differs from $MANIFEST"
for pin in $EXEMPT; do
  f=$SRC/config/${pin%%:*}; [[ ! -e $f && ! -L $f ]] || sha_ok "$f" "${pin##*:}"
done
log "freeze: $SRC matches $MANIFEST (sha256 $MSHA; $(wc -l < "$TMP/now.tsv") files; exempt by exact path, absent or pinned: $EXEMPT)"
# 2. this launch's files
cp "$ST/pilot_sim2gap_s4r.yaml" "$ST/tasks_sim2gap_s4r.json" "$SRC/config/"
sha_ok "$SRC/config/tasks_sim2gap_s4r.json" "$TSHA"
sha_ok "$SRC/config/pilot_sim2gap_s4r.yaml" "$CSHA"
sha_ok "$ST/sim2gap_relay_allowlist_s4r.json" "$ASHA"
# 3. the config (sim only), the cells (= the allowlist's 59 keys) and the references by key (copy and source sha256 = the pin)
(cd "$SRC" && "$PY" - "$MAIN" "$ST/sim2gap_relay_allowlist_s4r.json" <<'PY') || fail "config, cells or references"
import hashlib, json, sys
import yaml
main, alf = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("config/pilot_sim2gap_s4r.yaml"))
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_attack"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == "config/tasks_sim2gap_s4r.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"] == 2100
cells = json.load(open(cfg["tasks"]))["cells"]
keys = [c["key"] for c in cells]
al = json.load(open(alf))["files"]
pins = {f["key"]: {} for f in al if f["role"] != "launch.json"}
for f in al:
    if f["role"] != "launch.json":
        pins[f["key"]][f["role"]] = (f["path"], f["sha256"])
assert len(keys) == len(set(keys)) == 59 and set(keys) == set(pins), "cells != the pinned references' keys"
refs = json.load(open(main + "/launch.json"))["references"]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for k in keys:
    e = refs[k]
    for role in ("copy", "source"):
        p, s = pins[k][role]
        assert e[role] == p and h(p) == s, (k, role)
print(f"config: sim only (sim_attack@september); {len(keys)} cells; references by key = the pins (copy and source); cap {cfg['spend']['cap_usd']}")
PY
# 4. settings equality (as fu_launch.sh)
GS="results/gate_settings_sim2gap_s4r_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS ($GS, sha256 $(sha256sum < "$SRC/$GS" | cut -c1-16)...)"
printf '{"launch": "sim2gap_s4r", "addon": "sim2 gap-fill (descriptive; 09 @ b9081cd57 + 4d44c4e2f)", "worlds": {"sim_attack@september": "sim2"}, "folder": "ilnm_s4rgap_sim2_v1__sim_attack", "stage": "gap", "references_from": "%s", "references_pins": "config/sim2gap/sim2gap_relay_allowlist_s4r.json sha256 %s", "settings_dir": "/home/aswin/phase2/settings", "harness": "nm", "manifest_sha256": "%s", "settings_gate": "%s", "sim_attempt_timeout_s": 300, "cap_usd": %s, "commit": "%s"}\n' \
  "$MAIN" "$ASHA" "$MSHA" "$GS" "$CAP" "$COMMIT" > "$RUN.il_meta.json"
# 5. the launcher, then the two reviewed guards (pause only); both must be alive 30 s later, or the launcher is
# interrupted and its exit CONFIRMED. From the launch on, errexit is off and every failure goes through abort().
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
tmux -L nm new -d -s "$SESSION" "cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $SRC/run_pilot.py --pilot-config config/pilot_sim2gap_s4r.yaml --base-urls $BASES --commit $COMMIT --approaches /home/aswin/phase2/nm/approaches --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L nmguards new -d -s "sc$SESSION" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $((CAP + 100)) $((CAP + 400))" || abort "tmux new (sim_cap_stop) failed"
tmux -L nmguards new -d -s "gp$SESSION" "$PY $CHECKS/gate_pause_nm.py $RUN $SESSION" || abort "tmux new (gate_pause_nm) failed"
log "guards started: sc$SESSION (pause at $((CAP + 100)), hard Ctrl+C at $((CAP + 400))), gp$SESSION"
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
