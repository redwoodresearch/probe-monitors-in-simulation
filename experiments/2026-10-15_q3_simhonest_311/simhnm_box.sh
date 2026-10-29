#!/usr/bin/env bash
# simhnm_box.sh (nm360 sim-honest; 09 Append 25g @ e6ded1811; d9's names): ON A HOST BOX, start one nm launch group's
# SIMULATED-HONEST launch (sim_honest@september only, forked at 0 from the group's nm main's COUNTED honest_a via
# --references-from) on the nm harness's follow-up path (Codex nm-delta review: SAFE). Derived from simh311_box.sh v2
# (Codex r5/r6 AGREE): same helpers, guards (the +311 sim-honest sim_cap_stop, plus the RL-as-infra gate_pause_311_rl.py
# @ e46ba05d0, staged) and abort path; changed for nm where marked "nm:". Every check fails closed. It refuses unless:
# the host's PREVIOUS launch has finished; NO run_pilot.py and NO spend or pause guard runs on the box; the nm harness
# equals one of the three committed frozen nm manifests (staged, sha-pinned); the staged config, task file (the nm
# follow-up file minus the broken-setup and 25g cells) and the main's own task file check out; the group's committed
# generation manifest (staged, sha given) shows every pinned cell's fork source == the unique counted attempt (25g item 2)
# and matches the main's references cell by cell; no nm-harness PAUSE_SIMS; settings equality.
#   bash ~/phase2/simh_nm_stage/simhnm_box.sh <g> <6 base URLs, comma-separated> <branch commit> <host's previous run dir> \
#     <host label> <generation manifest sha256> [<env file: config/env.aws.tpl (default) or $ST/env_<label>.aws.tpl>]
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
G=$1 BASES=$2 COMMIT=$3 PREV=$4 HOST=$5 GSHA=$6 ENVF=${7:-config/env.aws.tpl}
case $G in  # nm: per group the nm main, cap, sim_cap_stop pause threshold (STOP; Aswin ~12:06 raise: 2x expected spend), our task file's and config's sha256, the main's own task file
  s1) MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s1; CAP=1676; STOP=1776; TSHA=6054c0493544756d65787962c67a2ef1f20c43bc617f52d8c8949c7d58f1fafd; CSHA=a65890bcae3859dc287d18dbb2b55bb506e1a847c2c521c3a3d2eaaaa2eabb91; LTF=tasks_nm_s1_launch.json; LSHA=8249c208d3dc44a033f8949bd1f323caf7cbe2f96f1d1a0ff3447678dab280ee ;;
  s2) MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s2; CAP=4159; STOP=3018; TSHA=06ac7c1ef5080c302b28ab5db22379dc73653a309cb01cb924ab031573411eb5; CSHA=bbf6f66a24e9a16178f2f75f751fe82cd76cc98a99bec58684fdb5321db01b45; LTF=tasks_nm_s2_launch.json; LSHA=4e83a59f99c4d03872f904fb7c18e09a69be6a72b6590ad208821ae7cdab4d9f ;;
  s3) MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s3; CAP=3806; STOP=2312; TSHA=53856cef12148a1d0bc279b399f460e8f258480fe4bc34de0dcffb10026565e3; CSHA=6220927d602a78ba6b0f2333c0b716bf5a6f3ead5d514e1d5c2b988aacd7af61; LTF=tasks_nm_s3_launch.json; LSHA=83ec89869204591d5390eb95ba4c4f6663f9a9d8fc591c97939cec2a3cbac196 ;;
  s4d) MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4; CAP=1060; STOP=1160; TSHA=bf54b635bfadca2addccb0b10092181b9ad0c28222228f85106a5ec37599eba3; CSHA=c340e9bc6a4679139257170e5106ea99ab6793efc7c4f64d68d3fc76689bec0c; LTF=tasks_nm_s4_launch.json; LSHA=88ddad22c047ab2258010947d01cc40af769ac45106e57e15e5d5ba62cc5e9ab ;;
  s4r) MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r; CAP=3787; STOP=2274; TSHA=92aaff8bc144f9d34da9a10a568c7ef9c481ec39b4d62ba7b4a287a6117da0d1; CSHA=1cd72f42b6f9f7425afce96c31294c474b5bc622dfdb006e79daa41b7bf35c52; LTF=tasks_nm_s4r_launch.json; LSHA=9b7e3f4d1a5b7735fd0ca1a6aee428d420c41f26f21b82cfd012b3e49127fe7f ;;
  s5) MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s5; CAP=4002; STOP=2704; TSHA=ac316e27b2384787084207409957cb9eb33af6f3abb4e8b28b085ec5a66d4000; CSHA=8d1be79f77aa864a472aa166e5767f93cf334c44abb4fa36ffe84f4b88e82d98; LTF=tasks_nm_s5_launch.json; LSHA=f5589918a368d3509709206b5eefbcd4b3da40f435a9c1e3af37fc0192d984f7 ;;
  *) echo "group $G: not an nm360 sim-honest group"; exit 2 ;;
esac
SRC=/home/aswin/phase2/nm/attack_check_nm; ST=/home/aswin/phase2/simh_nm_stage  # nm: the nm harness; own stage dir
CHECKS=/home/aswin/phase2/s311_checks  # the +311 sim-honest spend guard (socket phase2), as simh_q<k>
RLGATE_SHA=624e35e33f393146978a65b63050fbc8de8a8726a9135137c72f67d68a6cc795  # gate_pause_311_rl.py @ e46ba05d0
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
RUN=/home/aswin/phase2/runs/2026-10-15_simh_nm_$G; SESSION=simhnm$G  # nm: runs/ so the census glob (2026-10-15_simh_*) covers it
LOG=/home/aswin/phase2/simh_nm_$G.log; TMP=$(mktemp -d)
# nm: the three committed frozen nm harness manifests (origin/aswin/q3-new-mains-launch, experiments/2026-10-14_q3_new_mains_launch/config/), staged in $ST
FROZEN="harness_manifest_fu_nm.tsv:18fbbaa225bbb0f70890f1436c17d254d676495136f8bd2aa616cb626281b7ed
harness_manifest_fu_box4.tsv:363d5b65a03900495372b5160f28bdd96673e37b563a42dd46148a67e4eda45d
harness_manifest_fu_simsastra.tsv:deb5f193780be039ca55b948b2083ac458a0cdaa7231440bbdb8358db6247b83"
# 4th, box4 AND groups s4d/s4r ONLY (09's ruling @ df46b9be1): box4's nm harness as the reviewed S4-rerun arming left it
# (s4r_arm.sh @ 6dde150b1, Amendment 9 @ ef6f05eec: edited ./prelaunch_gates.sh 8951059e -> b08e671d, a main-launch gate the
# follow-up path never runs, and regenerated config/harness_manifest_fu.tsv; newmains_s4r and its follow-up ran on it).
[[ $HOST == box4 && ( $G == s4d || $G == s4r ) ]] && FROZEN+=$'\nharness_manifest_fu_box4_s4r.tsv:543394b2f6c0d590bea597c83feac7d7f69550dfb5b0c040d83cbd78a4143cc1'
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [simh nm $G] $*"
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
absent "$SRC/results/PAUSE_SIMS"  # nm: the nm harness's own marker (Codex nm-delta review)
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
# (nm:) tasks_simh_nm_*.
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il*' ! -path './config/tasks_il*' ! -path './config/ilrr*' ! -path './config/pilot_simh_*' \
  ! -path './config/tasks_simh_nm_*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
[[ -s $TMP/now.tsv ]] || fail "empty hash list"
MATCH=""
while IFS=: read -r mf ms; do
  sha_ok "$ST/$mf" "$ms"
  grep -v '  ./config/harness_manifest' "$ST/$mf" > "$TMP/frozen.tsv" || fail "reading $ST/$mf failed"
  [[ -s $TMP/frozen.tsv ]] || fail "empty $mf"
  if cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv"; then MATCH=$mf; fi
done <<< "$FROZEN"
[[ -n $MATCH ]] || fail "$SRC equals none of the committed frozen nm manifests allowed on $HOST"
log "freeze: $SRC matches $MATCH ($(wc -l < "$TMP/now.tsv") files)"
sha_ok "$ST/gate_pause_311_rl.py" "$RLGATE_SHA"
# 2. nm: this launch's config and task file (staged, sha-pinned) into the harness config/; the main's own task file
sha_ok "$ST/pilot_simh_nm_$G.yaml" "$CSHA"; sha_ok "$ST/tasks_simh_nm_$G.json" "$TSHA"
cp "$ST/pilot_simh_nm_$G.yaml" "$ST/tasks_simh_nm_$G.json" "$SRC/config/"
sha_ok "$SRC/config/pilot_simh_nm_$G.yaml" "$CSHA"; sha_ok "$SRC/config/tasks_simh_nm_$G.json" "$TSHA"
LTASK=$TMP/$LTF
cp "$ST/$LTF" "$LTASK" || fail "staged $G main task file"
sha_ok "$LTASK" "$LSHA"
# the committed generation manifest (09 Append 25g item 4), staged: its sha256 is the one given; it is this group's; its
# 25g census is clean (every pinned cell's fork source is the unique counted attempt, source == copy sha256) and every
# pinned cell carries the rendered reference hash; and each pinned cell's sha256 equals the main's pinned reference now
GM=$(ls "$ST"/manifest_simh_nm_${G}_*.json 2>/dev/null | head -2) || true
[[ -n $GM && $(wc -l <<< "$GM") == 1 ]] || fail "need exactly one staged generation manifest for $G"
sha_ok "$GM" "$GSHA"
"$PY" - "$GM" "$MAIN/launch.json" "$G" "$SRC/config/tasks_simh_nm_$G.json" "$ST/$MATCH" <<'PY' || fail "generation manifest does not match the main's references or the live harness"
import json, sys
m, launch, g = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
# 09 @ df46b9be1, on every host: the live harness (it equals $MATCH exactly, checked above) must be the generation
# manifest's PINNED baseline, by name and full sha256 (no fallback between baselines)
import hashlib as _h
_live = open(sys.argv[5], "rb").read()
assert m["harness"]["frozen_manifest"].rsplit("/", 1)[-1] == sys.argv[5].rsplit("/", 1)[-1], (m["harness"]["frozen_manifest"], sys.argv[5])
assert _h.sha256(_live).hexdigest() == m["harness"]["frozen_manifest_sha256"], "live baseline != the manifest's pinned baseline"
assert m["group"] == g and m["world"] == "simh_nm360" and m["references_from"] == sys.argv[2].rsplit("/", 1)[0], (m["group"], m["references_from"])
assert m["selected_run_census"]["equal"] and not m["selected_run_census"]["not_counted"], m["selected_run_census"]
refs = launch["references"]
pinned = {c["task"]: c["eval_sha256"] for c in m["cells"] if c["status"] == "pinned"}
assert pinned and all(c.get("rendered_reference_sha256") for c in m["cells"] if c["status"] == "pinned"), "rendered hash missing"
assert pinned == {t: r["sha256"] for t, r in refs.items() if t in pinned}, "per-cell sha256 mismatch"
assert set(pinned) == set(refs) & {c["task"] for c in m["cells"]}, "pinned set differs from the main's references"
assert {c["task"] for c in m["cells"]} == {c["key"] for c in json.load(open(sys.argv[4]))["cells"]}, "manifest cells != task file cells"
# Codex r12: re-run 25g item 2 on the CURRENT main launch.json right before the launch: its sha256 equals the one the
# manifest froze (a finished main's launch.json never changes), every pinned cell's fork source is the UNIQUE counted
# attempt (not kind rerun, no redrawn_because), and source, copy and pin hash equal
import hashlib
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
assert sha(sys.argv[2]) == m["main_launch_json_sha256"], "the main's launch.json changed since the manifest"
runs = launch["runs"]; runs = list(runs.values()) if isinstance(runs, dict) else runs
ha = {r["key"].split("/", 1)[1]: r for r in runs if r.get("arm") == "honest_a"}
for t, pin in pinned.items():
    ref, atts = refs[t], ha[t]["attempts"]
    counted = [i for i, a in enumerate(atts) if not a.get("redrawn_because") and a.get("kind") != "rerun"]
    src_dir = Path(ref["source"]).resolve().parent
    sel = [i for i, a in enumerate(atts) if a.get("log_dir") and Path(a["log_dir"]).resolve() == src_dir]
    assert len(counted) == 1 and sel == counted, f"{t}: fork source is not the unique counted attempt"
    assert sha(ref["source"]) == sha(ref["copy"]) == pin, f"{t}: source, copy and pin differ"
print(f"generation manifest: {len(pinned)} pinned cells match the main's references; 25g census clean; re-checked now: launch.json frozen, fork source = unique counted attempt, source = copy = pin")
PY
(cd "$SRC" && "$PY" - "$G" "$LTASK" <<'PY') || fail "config or task file"
import json, sys, yaml
g = sys.argv[1]
cfg = yaml.safe_load(open(f"config/pilot_simh_nm_{g}.yaml"))
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_honest"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_simh_nm_{g}.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"]
assert cfg["spend"]["worst_per_sim_run_usd"] == 106 and cfg["spend"]["prior_per_sim_run_usd"] == 21.13, cfg["spend"]
cells = json.load(open(cfg["tasks"]))["cells"]
main = {c["key"]: c for c in json.load(open(sys.argv[2]))["cells"] if "real_arms" not in c}
canon = lambda c: json.dumps(c, sort_keys=True)
assert cells and all(c["key"] in main and canon(main[c["key"]]) == canon(c) for c in cells), "cells are not the main's r1 entries"
print(f"config: sim_honest@september only; {len(cells)} cells, each the main's r1 entry; cap {cfg['spend']['cap_usd']}")
PY
# 3. settings equality (as the nm fu_launch.sh)
GS="results/gate_settings_simh_nm_${G}_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS"
printf '{"launch": "simh_nm_%s", "world": "simh_nm360", "arm": "sim_honest@september", "stratum": "S_newmains (nm360)", "host": "%s", "previous_launch": "%s", "references_from": "%s", "generation_manifest_sha256": "%s", "frozen_manifest": "%s", "env_file": "%s", "settings_dir": "/home/aswin/phase2/settings", "cap_usd": %s, "commit": "%s", "spec": "09 Append 25g @ e6ded1811"}\n' \
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
tmux -L phase2 new -d -s "$SESSION" "cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES RR_OWNER_ID=aswin-surya rr-env run --env-file $ENVF -- $PY $SRC/run_pilot.py --pilot-config config/pilot_simh_nm_$G.yaml --base-urls $BASES --commit $COMMIT --approaches /home/aswin/phase2/nm/approaches --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L guards new -d -s "scsimhnm$G" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $STOP" || abort "tmux new (sim_cap_stop) failed"
tmux -L guards new -d -s "gpsimhnm$G" "$PY $ST/gate_pause_311_rl.py $RUN $SESSION" || abort "tmux new (gate_pause_311_rl) failed"
log "guards started: scsimhnm$G (at $STOP), gpsimhnm$G"
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
