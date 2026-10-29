#!/usr/bin/env bash
# sp493sim_box.sh: ON box3 (key L1) or box4 (key L2), start an S_paper493 SIM-ATTACK follow-up (09 Append 39 + amendment 1;
# group map v2 @ c8ff270da: L1 = g1, L2 = g2): ONE simulated attack per cell, no real arm; references from this launch's
# reference index (fork pins by key, results/sp493_sim_refpins_check_2026-10-15.json); +311 harness = the IL frozen manifest
# (71fd2772..., staged) by full hash, with 09's exact-path exemptions pinned per box; max_runs_per_server 5 (09).
# Derived from il_sel97_box.sh (its helpers, settings gate, guards and confirmed start verbatim).
#   bash ~/phase2/il_stage/sp493sim_box.sh <L1|L2> <6 base URLs, comma-separated> <branch commit>
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
K=$1 BASES=$2 COMMIT=$3
case $K in
  L1) STALE=""; BOXL=box3; TSHA=8a2a4cd6684f4ad7c1b2ef707171eca8c4cb2e1d13398dc70f6c00b0eb882c9b; CSHA=f7f882a7d8378740dcc69e36c28b558e132bfe5063660e47d46b584108929bba; RSHA=291c22b17e7965b02c63388d7a21cf6b357ffcbea4843eca4e0a5a56e6c6c0a9; XSHA=cd0fa40708e7a7d30fb31d3765bddef5387a567ffad65c35b8040004dea63797; XF=freeze_exempt_box3.tsv ;;
  L2) STALE="/home/aswin/phase2/runs/2026-10-14_153435_canary_a:031d6972d0f7c5c179e18419faa2bdde09ee225ebf82349f13250918dfbcf37a:1792017901 /home/aswin/phase2/runs/2026-10-14_153435_canary_b:e99b2af227b7acf0186644aff2b24b7ddb0a6a1ff345db83485246a9a682aafb:1792017845"  # v2 (d9 ~19:12): dead 10-14 canaries (finished null), pinned by sha256 + mtime
      BOXL=box4; TSHA=075e0c1a2aa587c24e2bd0a27cb9a8b478838506fb8071f5a37a221d8fb5c83b; CSHA=008d30eb57bdc2fce1326a810141438c0284299713b9cd49d2c666f127626926; RSHA=3a8c569ad6894dac1e7450a61ffaefcf5ccdf59d5f270f136672354d307c11f6; XSHA=536bf688318304c04e31db436b447036906adb0d0a3d88f162b20a56adf0fec0; XF=freeze_exempt_box4.tsv ;;
  *) echo "launch $K: not prepared"; exit 2 ;;
esac
MSHA=71fd27724a045d3df5d8dba9b6e63d5f86514377c025a11a57fd78f6bf1aa287  # the IL frozen +311 manifest (xbox3/xbox1/box3), staged as il_stage/harness_manifest_il_frozen_311.tsv
CAPSTOP_SHA=6689a761a6a5d879310dcb6c7e0e10eea8840dd4eb88fd44b9b3403222ba1ccb; GATE_SHA=624e35e33f393146978a65b63050fbc8de8a8726a9135137c72f67d68a6cc795
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json; ST=/home/aswin/phase2/il_stage
SRC=/home/aswin/phase2/attack_check_311; SOCK=phase2; GSOCK=guards; CHECKS=$ST/checks; CAPDIR=/home/aswin/phase2/s311_checks
RIDX=/home/aswin/phase2/sp493sim/refindex_$K; RUN=/home/aswin/phase2/runs/2026-10-15_sp493_sim_$K; SESSION=sp493sim$K
GATEPAUSE=gate_pause_311_rl.py; APPR=/home/aswin/phase2/approaches; MANIFEST=$ST/harness_manifest_il_frozen_311.tsv
LOG=/home/aswin/phase2/sp493_sim_$K.log; TMP=$(mktemp -d)
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [sp493 sim $K @ $BOXL] $*"
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
CAP=$(sed -nE 's/^  cap_usd: ([0-9]+).*/\1/p' "$ST/pilot_sp493_sim_$K.yaml"); [[ $CAP =~ ^[0-9]+$ ]] || { echo "no cap"; exit 2; }
CAPSTOP=$((CAP + 100))
LAUNCH_CMD="cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES PYTHONPATH=$TREE RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $SRC/run_pilot.py --pilot-config config/pilot_sp493_sim_$K.yaml --base-urls $BASES --commit $COMMIT --approaches $APPR --references-from $RIDX --out $RUN > $RUN.launcher.log 2>&1"
[[ $BASES =~ ^https://[^,]+(,https://[^,]+){5}$ ]] || fail "expected 6 base URLs"
absent "$RUN"
absent "$SRC/results/PAUSE_SIMS"
! has_session $SOCK "$SESSION" || fail "session $SESSION is live"
# (a) every launch on this box has finished; parents proven inspectable; run dirs enumerated (as sim2gap_box.sh)
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
  rd=${d%/}; st=""; sm=""; for s in $STALE; do IFS=: read -r sn ss sms <<< "$s"; [[ $sn == "$rd" ]] && { st=$ss; sm=$sms; }; done  # Codex r48: matched by the full path
  if [[ -n $st ]]; then  # v2: a pinned dead launch (finished null): launch.json byte-identical, mtime unchanged, no python process names it
    sha_ok "$l" "$st"; [[ $(stat -c %Y "$l") == "$sm" ]] || fail "$rd: launch.json mtime changed"
    PSO=$(ps -eo comm=,args=) || fail "ps failed"  # Codex r48: a failing ps never reads as "no process"
    if awk -v r="$rd" '$1 ~ /^python/ && index($0, r)' <<< "$PSO" | grep -q .; then fail "$rd: a python process still names it"; fi
    log "pinned dead launch accepted (launch.json sha256 ${st:0:16}..., mtime $sm, no python process): $rd"; continue
  fi
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$l" \
    || fail "${d%/} has not finished (or its launch.json can't be read)"
done
log "every launch.json under runs/ and nm/runs/ reports finished (${#RDIRS[@]} run dirs inspected)"
no_launcher
log "no other run_pilot.py on the box"
# 1. staged files and pins
sha_ok "$ST/tasks_sp493_sim_$K.json" "$TSHA"; sha_ok "$ST/pilot_sp493_sim_$K.yaml" "$CSHA"; sha_ok "$ST/refindex_$K.json" "$RSHA"
sha_ok "$ST/$XF" "$XSHA"; sha_ok "$MANIFEST" "$MSHA"; sha_ok "$CAPDIR/sim_cap_stop.py" "$CAPSTOP_SHA"; sha_ok "$CHECKS/$GATEPAUSE" "$GATE_SHA"
# 2. (Codex r47) no symlink in the harness (outside results/ and __pycache__); NO name-pattern exclusions: the box's staged
# IL configs are exact-path exemptions in $XF too. Then the harness equals the IL frozen manifest; this box's exempt paths (09) are excluded from both sides and then each
# required to be absent or at its pinned sha256; this launch's own staged config/task names are excluded by name
SL=$(cd "$SRC" && find . -type l ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' -print -quit) || fail "symlink scan failed"
[[ -z $SL ]] || fail "symlink in the harness: $SL"
XPATHS=$(awk '{print $2}' "$ST/$XF"); [[ -n $XPATHS ]] || fail "empty exemption file"
FINDX=(); for p in $XPATHS; do [[ $p =~ ^\./[A-Za-z0-9_./-]+$ ]] || fail "bad exempt path $p"; FINDX+=(! -path "$p"); done
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' \
  ! -path "./config/pilot_sp493_sim_$K.yaml" ! -path "./config/tasks_sp493_sim_$K.json" "${FINDX[@]}" -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
(set -o pipefail; grep -v '  ./config/harness_manifest' "$MANIFEST" | awk 'NR == FNR {x[$2] = 1; next} !($2 in x)' "$ST/$XF" - > "$TMP/frozen.tsv") || fail "reading $MANIFEST failed"
[[ -s $TMP/now.tsv && -s $TMP/frozen.tsv ]] || fail "empty hash list"
cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv" || fail "$SRC differs from the IL frozen manifest (outside 09's exemptions)"
while read -r h p; do f=$SRC/${p#./}; [[ ! -e $f && ! -L $f ]] || sha_ok "$f" "$h"; done < "$ST/$XF"
log "freeze: $SRC = the IL frozen +311 manifest ($MSHA; $(wc -l < "$TMP/now.tsv") files) except $XF ($XSHA): exempt paths absent or pinned"
# 3. this launch's files in place
cp "$ST/pilot_sp493_sim_$K.yaml" "$ST/tasks_sp493_sim_$K.json" "$SRC/config/"
sha_ok "$SRC/config/tasks_sp493_sim_$K.json" "$TSHA"; sha_ok "$SRC/config/pilot_sp493_sim_$K.yaml" "$CSHA"
mkdir -p "$RIDX"; [[ ! -e $RIDX/launch.json ]] || cmp -s "$ST/refindex_$K.json" "$RIDX/launch.json" || fail "$RIDX/launch.json exists and differs"
cp "$ST/refindex_$K.json" "$RIDX/launch.json"; sha_ok "$RIDX/launch.json" "$RSHA"
# 4. config (sims only, max_runs_per_server 5), cells = the index's keys, every reference copy at its pin, every approach present
(cd "$SRC" && "$PY" - "$K" "$RIDX/launch.json" "$APPR" <<'PY') || fail "config, cells, references or approaches"
import hashlib, json, os, sys
import yaml
k, ridx, appr = sys.argv[1:]
cfg = yaml.safe_load(open(f"config/pilot_sp493_sim_{k}.yaml"))
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_attack"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_sp493_sim_{k}.json" and cfg["max_runs_per_server"] == 5
assert cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"]
cells = json.load(open(cfg["tasks"]))["cells"]
refs = json.load(open(ridx))["references"]
keys = [c["key"] for c in cells]
assert len(keys) == len(set(keys)) and set(keys) == set(refs), "cells != the reference index's keys"
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for c in cells:
    e = refs[c["key"]]
    assert os.path.isfile(e["copy"]) and h(e["copy"]) == e["sha256"], (c["key"], "reference")
    assert "real_arms" not in c
    a = os.path.join(appr, c["approach_sha256"] + ".txt")
    assert os.path.isfile(a) and h(a) == c["approach_sha256"], (c["key"], "approach")
print(f"config: sims only, max_runs_per_server 5; {len(keys)} cells = the index's keys; references and approaches at their pins; cap {cfg['spend']['cap_usd']}")
PY
# 5. settings equality (as fu_launch.sh)
GS="results/gate_settings_sp493_sim_${K}_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS ($GS, sha256 $(sha256sum < "$SRC/$GS" | cut -c1-16)...)"
printf '{"launch": "sp493_sim_%s", "group": "%s", "folder": "sp493_%s_sim_v1__sim_attack", "worlds": {"sim_attack@september": "sim"}, "references_from": "%s", "references_index_sha256": "%s", "harness": "311", "manifest_sha256": "%s", "freeze_exemptions": "%s %s", "max_runs_per_server": 5, "max_runs_per_server_reason": "09: concurrency bounded independently of the cap; 6 servers x 5 = 30 per launch, 60 total, <= 95", "settings_gate": "%s", "cap_usd": %s, "commit": "%s"}\n' \
  "$K" "$([[ $K == L1 ]] && echo g1 || echo g2)" "$([[ $K == L1 ]] && echo g1 || echo g2)" "$RIDX" "$RSHA" "$MSHA" "$XF" "$XSHA" "$GS" "$CAP" "$COMMIT" > "$RUN.il_meta.json"
# 6. the launcher, then the two reviewed guards (pause only); both alive 30 s later, or the launcher is interrupted and
# its exit CONFIRMED. From the launch on, errexit is off and every failure goes through abort().
OURS="--out $RUN( |\$)"  # this launch's processes (pgrep/grep -E): the exact run dir, not a prefix of another
gone() {  # 0 = confirmed gone; 1 = still there; 2 = could not tell (an inspection error is never read as gone)
  local list rc=0; list=$(sessions $SOCK) || return 2
  if in_list "$list" "$SESSION"; then return 1; fi
  pgrep -f -- "$OURS" > /dev/null || rc=$?
  case $rc in 0) return 1 ;; 1) return 0 ;; *) return 2 ;; esac
}
abort() {  # interrupt the launcher and CONFIRM it ended; escalate to kill-session; never report success unconfirmed
  log "ABORT: $1"
  local i j
  for i in 1 2; do
    tmux -L $SOCK send-keys -t "=$SESSION:" C-c > /dev/null 2>&1 || true
    for j in $(seq 30); do gone; if [[ $? == 0 ]]; then log "launcher ended after Ctrl+C"; exit 1; fi; sleep 2; done
  done
  tmux -L $SOCK kill-session -t "=$SESSION" > /dev/null 2>&1 || true
  for j in $(seq 15); do gone; if [[ $? == 0 ]]; then log "launcher session killed"; exit 1; fi; sleep 2; done
  log "LAUNCHER NOT CONFIRMED STOPPED after abort: $SESSION, $RUN (stop it by hand)"; exit 3
}
no_launcher
tmux -L $SOCK new -d -s "$SESSION" "$LAUNCH_CMD" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L $GSOCK new -d -s "sc$SESSION" "$PY $CAPDIR/sim_cap_stop.py $RUN $SESSION $CAPSTOP" || abort "tmux new (sim_cap_stop) failed"
tmux -L $GSOCK new -d -s "gp$SESSION" "$PY $CHECKS/$GATEPAUSE $RUN $SESSION" || abort "tmux new ($GATEPAUSE) failed"
log "guards started: sc$SESSION (sim_cap_stop $CAPSTOP), gp$SESSION ($GATEPAUSE) on tmux -L $GSOCK"
sleep 30
list=$(sessions $SOCK) || abort "cannot list tmux -L $SOCK sessions"
in_list "$list" "$SESSION" || abort "the launcher session is gone (see $RUN.launcher.log)"
pgrep -f "sim_cap_stop.py $RUN $SESSION" > /dev/null || abort "sim_cap_stop is not running (or pgrep failed)"
pgrep -f "$GATEPAUSE $RUN $SESSION" > /dev/null || abort "$GATEPAUSE is not running (or pgrep failed)"
rc=0; all=$(pgrep -af 'run_pilot\.py') || rc=$?
[[ $rc == 0 ]] || abort "pgrep finds no launcher process at all (rc $rc)"
others=""; ours=0
while IFS= read -r line; do  # pure bash over pgrep's lines: ours match the exact run dir, anything else is another launcher
  if [[ $line =~ --out\ $RUN(\ |$) ]]; then ours=$((ours + 1)); elif [[ -n $line ]]; then others+="$line"$'\n'; fi
done <<< "$all"
(( ours > 0 )) || abort "no process of this launch found"
[[ -z $others ]] || abort "another launcher is running: ${others:0:300}"
log "STARTED: $SESSION, run $RUN, cap $CAP, guards alive, no other launcher"
