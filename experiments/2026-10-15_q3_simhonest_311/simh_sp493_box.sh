#!/usr/bin/env bash
# simh_sp493_box.sh (S_paper493 sim-honest Phase A: 27 T196 mains; 09 Append 39, rulings via d9 ~17:30): ON A HOST BOX,
# start launch sp493_t196 (sim_honest@september only), every cell forked at step 0 from il_refindex_old's pinned reference
# (--references-from /home/aswin/phase2/il_refindex_old). DERIVED from simh_rlr_box.sh (Codex r12-r31); changed only where
# marked "sp493:": the launch table, the generation manifest's checks (the refindex reference rule instead of the counted-
# attempt rule; a 09-confirmed routed root required) and the log label. Fails closed exactly as simh_rlr_box.sh.
#   bash ~/phase2/simh_rlr_stage/simh_sp493_box.sh sp493_t196 <6 base URLs> <branch commit> <host's previous run dir> <host label> \
#     <generation manifest sha256> [<env file>]
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
G=$1 BASES=$2 COMMIT=$3 PREV=$4 HOST=$5 GSHA=$6 ENVF=${7:-config/env.aws.tpl}
# sp493: pause at 2 x expected; admission cap = expected + K x $106 (make_sp493.py)
case $G in  # sp493: the one Phase A launch (make_sp493.py; results/sp493_simh_plan_2026-10-15.json)
  sp493_t196) FAM=311; MAIN=/home/aswin/phase2/il_refindex_old; CAP=1546; STOP=548; TSHA=8c2469269f3b3226367c6c66875ce27ffb4c471ffd128c428e6bbac77be27c2d; CSHA=a206fa697d2af33b1bf851ca06056dc7e3bad06c846644d177bcfe2997d52b81; LTF=tasks_il_old_all.json; LSHA=69c709d91da2f8952c0a3f4ceddf3e05cac5044f73a596eb253bb5763f70ee4e ;;
  sp493_new) FAM=311; MAIN=/home/aswin/phase2/sp493sim/refindex_L3; CAP=2700; STOP=1372; TSHA=5c44c87c2c85ff077dda260e9216ca4e1dfed67ed1d770dbf2cd821f9136aed6; CSHA=f241f82317fb0ce2c9093acbdb92688afa3bcc0414e64f54d252e6f510512544; LTF=sp493_v2_r1_entries.json; LSHA=2f005d1b51d295fa547d4fcda4a0efd0a5cabba35e6890743be075580565d1f7 ;;  # S_paper493 g5 (make_sp493_g5.py)
  t196fill) FAM=311; MAIN=/home/aswin/phase2/il_refindex_old; CAP=899; STOP=500; TSHA=ac0d043ac18573e21cc404a7815de3259a84759ef08fce2d71f2ca18afdf0536; CSHA=a5fac258fb43cb02bff74cef1caf764c85fd780a7f8bbde28bbaf8a54b7fb37a; LTF=tasks_il_old_all.json; LSHA=69c709d91da2f8952c0a3f4ceddf3e05cac5044f73a596eb253bb5763f70ee4e ;;  # the A25 fill-in (09 @ 286b2f518)
  *) echo "launch $G: not an S_paper493 / fill-in sim-honest launch"; exit 2 ;;
esac
ST=/home/aswin/phase2/simh_rlr_stage  # rlr: one stage dir for both families
if [[ $FAM == 311 ]]; then SRC=/home/aswin/phase2/attack_check_311; APPR=/home/aswin/phase2/approaches; PYP="PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src "
else SRC=/home/aswin/phase2/nm/attack_check_nm; APPR=/home/aswin/phase2/nm/approaches; PYP=""; fi  # rlr: each family's own launch line
CHECKS=/home/aswin/phase2/s311_checks  # the +311 sim-honest spend guard (socket phase2), as simh_q<k>
RLGATE_SHA=624e35e33f393146978a65b63050fbc8de8a8726a9135137c72f67d68a6cc795  # gate_pause_311_rl.py @ e46ba05d0
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
RUN=/home/aswin/phase2/runs/2026-10-15_simh_$G; SESSION=simh${G//_/}  # rlr: runs/ so the census glob covers it
LOG=/home/aswin/phase2/simh_$G.log; TMP=$(mktemp -d)
# rlr: the family's committed frozen harness manifests, staged in $ST; box4's s4r-armed one only for nm_s4d_rlr / nm_s4r_rlr on box4; nm_s4d_rlr2 only on box1 (fu_nm)
if [[ $FAM == 311 ]]; then FROZEN="harness_manifest_fu_q123.tsv:71fd27724a045d3df5d8dba9b6e63d5f86514377c025a11a57fd78f6bf1aa287
harness_manifest_fu_q2r.tsv:b8f813c9a1f4ff4371a981b3aea8a9406b7fa0143eda3212a8b053e8123a4992
harness_manifest_fu_q4.tsv:32ed72b3e3aa95b21a41ae0dfcf47cacb9efb6e760e9577536e49b57e898bd3d"
else FROZEN="harness_manifest_fu_nm.tsv:18fbbaa225bbb0f70890f1436c17d254d676495136f8bd2aa616cb626281b7ed
harness_manifest_fu_box4.tsv:363d5b65a03900495372b5160f28bdd96673e37b563a42dd46148a67e4eda45d
harness_manifest_fu_simsastra.tsv:deb5f193780be039ca55b948b2083ac458a0cdaa7231440bbdb8358db6247b83"; fi
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [simh sp493 $G] $*"
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
# nm: the env file selects the simulator API key by its Secrets Manager label (aws-sm://...); never the key itself
[[ $ENVF == config/env.aws.tpl || $ENVF =~ ^/home/aswin/phase2/simh_nm_stage/env_[a-z0-9_]+\.aws\.tpl$ ]] || fail "env file $ENVF not allowed"
[[ $ENVF == /* ]] && { [[ -f $ENVF ]] || fail "env file $ENVF missing"; } || [[ -f $SRC/$ENVF ]] || fail "env file $SRC/$ENVF missing"
absent "$RUN"
absent "$SRC/results/PAUSE_SIMS"  # rlr: the family harness's own marker
! has_session phase2 "$SESSION" || fail "session $SESSION is live"
# (a) the host's previous launch has FINISHED, and no spend or pause guard of any launch is still running
[[ $PREV == /home/aswin/phase2/* && -r $PREV/launch.json ]] || fail "cannot read the previous launch $PREV/launch.json"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$PREV/launch.json" \
  || fail "the previous launch $PREV has not finished"
log "host $HOST: previous launch $PREV finished"
rc=0; gl=$(pgrep -af 'sim_cap_stop\.py|gate_pause[a-z0-9_]*\.py') || rc=$?
(( rc == 1 )) && [[ -z $gl ]] || fail "a spend or pause guard is still running (pgrep rc $rc): ${gl:0:300}"
[[ $GSHA =~ ^[0-9a-f]{64}$ ]] || fail "generation manifest sha256 malformed"
[[ -r $MAIN/launch.json ]] || fail "the $G main is not on this host at $MAIN"
# (b) one launcher per settings dir: no run_pilot.py process at all on the box
no_launcher
log "no other run_pilot.py on the box"
# 1. the nm harness is still frozen: it must equal ONE of the three committed frozen nm manifests (each staged copy
# sha-pinned). Other launches' staged config files are excluded by name: pilot_il*, tasks_il*, ilrr*, pilot_simh_* and
# (rlr:) tasks_simh_* (all sim-honest task files).
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il*' ! -path './config/tasks_il*' ! -path './config/ilrr*' ! -path './config/pilot_simh_*' \
  ! -path './config/tasks_simh_*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
[[ -s $TMP/now.tsv ]] || fail "empty hash list"
MATCH=""
while IFS=: read -r mf ms; do
  sha_ok "$ST/$mf" "$ms"
  grep -v '  ./config/harness_manifest' "$ST/$mf" > "$TMP/frozen.tsv" || fail "reading $ST/$mf failed"
  [[ -s $TMP/frozen.tsv ]] || fail "empty $mf"
  if cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv"; then MATCH=$mf; fi
done <<< "$FROZEN"
[[ -n $MATCH ]] || fail "$SRC equals none of the committed frozen manifests allowed for $G on $HOST"
log "freeze: $SRC matches $MATCH ($(wc -l < "$TMP/now.tsv") files)"
sha_ok "$ST/gate_pause_311_rl.py" "$RLGATE_SHA"
# 2. nm: this launch's config and task file (staged, sha-pinned) into the harness config/; the main's own task file
sha_ok "$ST/pilot_simh_$G.yaml" "$CSHA"; sha_ok "$ST/tasks_simh_$G.json" "$TSHA"
cp "$ST/pilot_simh_$G.yaml" "$ST/tasks_simh_$G.json" "$SRC/config/"
sha_ok "$SRC/config/pilot_simh_$G.yaml" "$CSHA"; sha_ok "$SRC/config/tasks_simh_$G.json" "$TSHA"
LTASK=$TMP/$LTF
cp "$ST/$LTF" "$LTASK" || fail "staged $G main task file"
sha_ok "$LTASK" "$LSHA"
# the committed generation manifest (09 Append 25g item 4), staged: its sha256 is the one given; it is this group's; its
# 25g census is clean (every pinned cell's fork source is the unique counted attempt, source == copy sha256) and every
# pinned cell carries the rendered reference hash; and each pinned cell's sha256 equals the main's pinned reference now
GM=$(ls "$ST"/manifest_simh_${G}_*.json 2>/dev/null | head -2) || true
[[ -n $GM && $(wc -l <<< "$GM") == 1 ]] || fail "need exactly one staged generation manifest for $G"
sha_ok "$GM" "$GSHA"
"$PY" - "$GM" "$MAIN/launch.json" "$G" "$SRC/config/tasks_simh_$G.json" "$ST/$MATCH" <<'PY' || fail "generation manifest does not match the main's references or the live harness"
import json, sys
m, launch, g = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
# 09 @ df46b9be1, on every host: the live harness (it equals $MATCH exactly, checked above) must be the generation
# manifest's PINNED baseline, by name and full sha256 (no fallback between baselines)
import hashlib as _h
_live = open(sys.argv[5], "rb").read()
assert m["harness"]["frozen_manifest"].rsplit("/", 1)[-1] == sys.argv[5].rsplit("/", 1)[-1], (m["harness"]["frozen_manifest"], sys.argv[5])
assert _h.sha256(_live).hexdigest() == m["harness"]["frozen_manifest_sha256"], "live baseline != the manifest's pinned baseline"
# sp493: the refindex rule. The manifest is this launch's; its reference census is clean; a 09-confirmed routed root is
# recorded (tested regex); il_refindex_old's launch.json is the pinned one; every pinned cell's sha256 equals the index's
# pin for its key AND the copy on this host hashes to it now; the pinned set equals the task file's cells
import hashlib, re
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
assert m["launch"] == g and m["family"] == "311" and m["world"] == g and m["references_from"] == sys.argv[2].rsplit("/", 1)[0], (m["launch"], m["references_from"])
assert m["reference_census"]["equal"] and not m["reference_census"]["violations"], m["reference_census"]
rr = m.get("routed_root") or {}
ROOT_RE = {"sp493_t196": r"^sp493_g[0-9]+_simh(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$",
           "t196fill": r"^simh_t196fill(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_honest$",
           "sp493_new": r"^sp493_g[0-9]+_simh(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest$"}[g]
assert re.match(ROOT_RE, rr.get("root", "")), "no 09-confirmed routed root for this launch in the manifest"
assert sha(sys.argv[2]) == m["index"]["launch_json_sha256"], "il_refindex_old/launch.json is not the manifest's"
refs = launch["references"]
pinned = {c["task"]: c["eval_sha256"] for c in m["cells"] if c["status"] == "pinned"}
assert pinned and len(pinned) == len(m["cells"]) and all(c.get("rendered_reference_sha256") for c in m["cells"]), "unpinned cell or rendered hash missing"
assert pinned == {t: refs[t]["sha256"] for t in pinned}, "per-cell sha256 != il_refindex_old's pin"
assert {c["task"] for c in m["cells"]} == {c["key"] for c in json.load(open(sys.argv[4]))["cells"]}, "manifest cells != task file cells"
for t, pin in pinned.items():
    assert sha(refs[t]["copy"]) == pin, f"{t}: the copy on this host != the pin"
print(f"generation manifest: {len(pinned)} pinned cells == il_refindex_old pins; copies re-hashed now; routed root {rr['root']}")
PY
(cd "$SRC" && "$PY" - "$G" "$LTASK" <<'PY') || fail "config or task file"
import json, sys, yaml
g = sys.argv[1]
cfg = yaml.safe_load(open(f"config/pilot_simh_{g}.yaml"))
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_honest"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_simh_{g}.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"]
assert cfg["spend"]["worst_per_sim_run_usd"] == 106 and cfg["spend"]["prior_per_sim_run_usd"] == 21.13, cfg["spend"]
cells = json.load(open(cfg["tasks"]))["cells"]
main = {c["key"]: c for c in json.load(open(sys.argv[2]))["cells"] if "real_arms" not in c}
canon = lambda c: json.dumps(c, sort_keys=True)
assert cells and all(c["key"] in main and canon(main[c["key"]]) == canon(c) for c in cells), "cells are not the main's r1 entries"
print(f"config: sim_honest@september only; {len(cells)} cells, each the main's r1 entry; cap {cfg['spend']['cap_usd']}")
PY
# 3. settings equality (as the nm fu_launch.sh)
GS="results/gate_settings_simh_${G}_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS"
printf '{"launch": "simh_%s", "world": "simh_311 or simh_nm360 (see the manifest)", "arm": "sim_honest@september", "stratum": "S_paper493 Phase A (sp493)", "host": "%s", "previous_launch": "%s", "references_from": "%s", "generation_manifest_sha256": "%s", "frozen_manifest": "%s", "env_file": "%s", "settings_dir": "/home/aswin/phase2/settings", "cap_usd": %s, "commit": "%s", "spec": "09 Append 39 + amendment 1; 25g machinery"}\n' \
  "$G" "$HOST" "$PREV" "$MAIN" "$GSHA" "$MATCH" "$ENVF" "$CAP" "$COMMIT" > "$RUN.simh_meta.json"
# 4. the launcher (nm: the nm follow-up launch line: nm approaches store, no PYTHONPATH; on socket phase2 for the +311
# sim-honest guards), then the two guards (pause only); both must be alive 30 s later, or the launcher is interrupted and
# its exit CONFIRMED. From the launch on, errexit is off and every failure goes through abort().
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
tmux -L phase2 new -d -s "$SESSION" "cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES ${PYP}RR_OWNER_ID=aswin-surya rr-env run --env-file $ENVF -- $PY $SRC/run_pilot.py --pilot-config config/pilot_simh_$G.yaml --base-urls $BASES --commit $COMMIT --approaches $APPR --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L guards new -d -s "scsimh${G//_/}" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $STOP" || abort "tmux new (sim_cap_stop) failed"
tmux -L guards new -d -s "gpsimh${G//_/}" "$PY $ST/gate_pause_311_rl.py $RUN $SESSION" || abort "tmux new (gate_pause_311_rl) failed"
log "guards started: scsimh${G//_/} (at $STOP), gpsimh${G//_/}"
sleep 30
list=$(sessions phase2) || abort "cannot list tmux -L phase2 sessions"
in_list "$list" "$SESSION" || abort "the launcher session is gone (see $RUN.launcher.log)"
pgrep -f "sim_cap_stop.py $RUN $SESSION" > /dev/null || abort "sim_cap_stop is not running (or pgrep failed)"
pgrep -f "gate_pause_311_rl.py $RUN $SESSION" > /dev/null || abort "gate_pause_311_rl is not running (or pgrep failed)"
rc=0; all=$(pgrep -af 'run_pilot\.py') || rc=$?
[[ $rc == 0 ]] || abort "pgrep finds no launcher process at all (rc $rc)"
others=""; ours=0
while IFS= read -r line; do  # pure bash over pgrep's lines: ours match the exact run dir, anything else is another launcher
  if [[ $line =~ --out\ $RUN(\ |$) ]]; then ours=$((ours + 1)); elif [[ -n $line ]]; then others+="$line"$'\n'; fi
done <<< "$all"
(( ours > 0 )) || abort "no process of this launch found"
[[ -z $others ]] || abort "another launcher is running: ${others:0:300}"
log "STARTED: $SESSION, run $RUN, cap $CAP, guards alive, no other launcher"
