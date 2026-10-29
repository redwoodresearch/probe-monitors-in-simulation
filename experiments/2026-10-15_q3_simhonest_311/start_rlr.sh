#!/bin/bash
# start_rlr.sh (sim-honest replacement launches; derived from start_nm.sh, Codex r12-r19): one replacement launch: census
# gate, server liveness, approach preflight (the family's approaches store), RL-watcher registration, simh_rlr_box.sh.
#   start_rlr.sh <launch> <host instance> <host label> <host's previous run dir> <servers json> <generation manifest sha256> [<env file>]
set -euo pipefail
G=$1 INST=$2 HOST=$3 PREV=$4 SJ=$5 GSHA=$6 ENVF=${7:-config/env.aws.tpl}
[[ $G =~ ^(q1_rlr|q2_rlr|q3_rlr|q4_rlr|nm_s3_rlr|nm_s4d_rlr|nm_s2_rlr|nm_s4r_rlr|q1_rlr2|q2_rlr2|nm_s4d_rlr2|q3_rlr2|sp493_t196|t196fill|sp493_new)$ ]] || { echo "bad launch $G"; exit 2; }
K=$G  # file-name tag for this starter's local outputs
[[ $G == q* || $G == sp493_* || $G == t196fill ]] && ADIR=/home/aswin/phase2/approaches || ADIR=/home/aswin/phase2/nm/approaches
CDIR=config_rlr BOXSCRIPT=simh_rlr_box.sh; [[ $G == sp493_* || $G == t196fill ]] && CDIR=config_sp493 BOXSCRIPT=simh_sp493_box.sh  # S_paper493 (d9 ~17:40)
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
WT=$SP/wt_simh311; H=$SP/simh_launch_X/h; REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
echo "== $(TZ=America/Los_Angeles date '+%F %T PT') start_rlr $G on $HOST ($INST)"
# 1. census gate (rule of d9 09:27 PT 10-15; from ~14:30 census_rl_v6.py gate: <= 95, running launches at admission capacity (0d after the 14:23 RL)): the newest committed census, <= 30 min old, complete,
# < 90 sims running, no RateLimitError in the latest 10-min bin; and no rate-limit pause in force (rl_pause_watch.sh)
[[ ! -e $SP/simh_drive/RL_PAUSED ]] || { echo "RL PAUSE IN FORCE ($SP/simh_drive/RL_PAUSED): no start"; exit 2; }
git -C "$REPO" fetch -q origin aswin/q3-new-mains-launch
CF=$(git -C "$REPO" ls-tree --name-only origin/aswin/q3-new-mains-launch experiments/2026-10-14_q3_new_mains_launch/results/ | grep 'sim_error_census_' | sort | tail -1)
# v3 (d9 ~11:45, 0d): projected = running + this launch's own concurrency (cap / $106, rounded up) + sel97's reserve gap
# (flags in scratchpad/sel97/) [+ ug's gap while il_q4c runs]; <= 110 (< 90 while il_q4c runs)
CONC=$(python3 -I -c 'import math, re, sys; t = open(sys.argv[1]).read(); print(math.ceil(int(re.search(r"^  cap_usd: ([1-9][0-9]*)(?:[ \t]+#[^\n]*)?$", t, re.M).group(1)) / 106))' "$(dirname "$0")/$CDIR/pilot_simh_$G.yaml")
git -C "$REPO" show "origin/aswin/q3-new-mains-launch:$CF" | python3 -I "$(dirname "$0")/census_rl_v12.py" gate --start-conc "$CONC" --sel97-dir "$SP/sel97" || { echo "CENSUS GATE FAIL ($CF)"; exit 2; }
echo "census gate PASS ($CF)"
# 2. servers: the record has 6 URLs and app IDs, and every app is deployed with >= 1 task
URLS=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); s=d["servers"]; assert d.get("status") == "up" and len(s)==6 and all(x.get("base_url", "").endswith("/v1") and x.get("app_id") for x in s); print(",".join(x["base_url"] for x in s))' "$SJ") || { echo "SERVER RECORD INCOMPLETE"; exit 3; }
perl -e 'alarm 60; exec @ARGV' "$M" app list --json > "$SP/simh_drive/applist_q$K.json"
python3 -c '
import json, sys
apps = {a["app_id"]: a for a in json.load(open(sys.argv[1]))}
ids = [x["app_id"] for x in json.load(open(sys.argv[2]))["servers"]]
bad = [(i, apps.get(i, {}).get("state"), apps.get(i, {}).get("tasks")) for i in ids if i not in apps or apps[i].get("stopped_at") or str(apps[i].get("tasks")) in ("0", "None")]
print("servers:", [(i, apps.get(i, {}).get("state"), apps.get(i, {}).get("tasks")) for i in ids])
sys.exit(1 if bad else 0)' "$SP/simh_drive/applist_q$K.json" "$SJ" || { echo "SERVER LIVENESS FAIL"; exit 3; }
echo "servers live"
# 2b. approaches: every approach_sha256 in this group's staged task file (the one the launcher materializes) is a regular,
# non-symlink ~/phase2/nm/approaches/<sha>.txt (nm: the nm approaches store, as the nm follow-up launch line) hashing to
# its name. Hash fields and sha256 only; no text is opened beyond hashing.
APPR=$( (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  "python3 -I -c 'import json, sys, os, hashlib
s = sorted({c[\"approach_sha256\"] for c in json.load(open(sys.argv[1]))[\"cells\"]})
ok = [h for h in s if os.path.isfile(p := f\"$ADIR/{h}.txt\") and not os.path.islink(p) and hashlib.sha256(open(p, \"rb\").read()).hexdigest() == h]
print(f\"APPROACHES needed={len(s)} ok={len(ok)}\")' /home/aswin/phase2/simh_rlr_stage/tasks_simh_${G}.json") 2> /dev/null | grep -E '^APPROACHES ' || true)
echo "${APPR:-approach check gave no answer}"
[[ $APPR =~ ^APPROACHES\ needed=([0-9]+)\ ok=([0-9]+)$ && ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" && ${BASH_REMATCH[1]} -gt 0 ]] || { echo "APPROACH FILES MISSING"; exit 4; }
# 2c. q1_rlr only (09's ack @ dc521bdbe via d9): t389's fork source is its 25f-pinned rerun, sha256 d160ffdc..., the same
# reference as the original sim-honest run. Fail the start unless exactly one COMMITTED generation manifest (HEAD blobs in
# $WT, by the given sha256) pins exactly that, and on the host the Q1 main's launch.json reference for t389 records it and
# its copy and source both hash to it. Hashes only; nothing is opened beyond hashing. Every verdict, pass or fail (also no or
# several committed manifests, or an unreadable one), goes into a start record in results/ before any exit (Codex r25).
if [[ $G == q1_rlr ]]; then
  T389=d160ffdc956ffc89180043b3575b002d4b0f1cc3f7600082f38decc81f28dae0
  GM_OUT=$(python3 -I -c 'import hashlib, json, subprocess, sys
wt, d, want = sys.argv[1:4]
g = lambda *a: subprocess.run(["git", "-C", wt, *a], capture_output=True, check=True).stdout
try:
    paths = [p for p in g("ls-tree", "-r", "--name-only", "HEAD", "--", d).decode().splitlines() if p.rsplit("/", 1)[-1].startswith("manifest_simh_q1_rlr_") and p.endswith(".json")]
    hits = [p for p in paths if hashlib.sha256(g("cat-file", "blob", "HEAD:" + p)).hexdigest() == want]
    assert len(hits) == 1, f"{len(hits)} committed manifests with sha256 {want}"
    pin = [c["eval_sha256"] for c in json.loads(g("cat-file", "blob", "HEAD:" + hits[0]))["cells"] if c["task"] == "t389_model_registry" and c["status"] == "pinned"]
    print("GMF=" + hits[0]); print("MAN=" + json.dumps(pin))
except Exception as e:
    print("ERR=" + type(e).__name__ + ": " + str(e).replace("\n", " "))' "$WT" experiments/2026-10-15_q3_simhonest_311/results/ "$GSHA" || true)
  GMF=$(sed -n 's/^GMF=//p' <<< "$GM_OUT"); MAN=$(sed -n 's/^MAN=//p' <<< "$GM_OUT"); GERR=$(sed -n 's/^ERR=//p' <<< "$GM_OUT")
  [[ -n $GMF || -n $GERR ]] || GERR="manifest lookup gave no answer"
  BOXT=$( (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 120; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
    "python3 -I -c 'import json, sys, hashlib
r = json.load(open(sys.argv[1]))[\"references\"][\"t389_model_registry\"]
h = lambda p: hashlib.sha256(open(p, \"rb\").read()).hexdigest()
print(\"T389 recorded=\" + r[\"sha256\"] + \" copy=\" + h(r[\"copy\"]) + \" source=\" + h(r[\"source\"]))' /home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q1/launch.json") 2> /dev/null | grep -E '^T389 ' || true)
  echo "t389 manifest ${GMF:-none} pin ${MAN:-none} ${GERR:+error: $GERR}; host: ${BOXT:-no answer}"
  OK=false; [[ -z $GERR && $MAN == "[\"$T389\"]" && $BOXT == "T389 recorded=$T389 copy=$T389 source=$T389" ]] && OK=true
  python3 -I -c 'import json, sys, datetime, zoneinfo
json.dump({"check": "q1_rlr t389 fork source == 25f-pinned rerun (09 ack @ dc521bdbe via d9)", "expected_sha256": sys.argv[1], "generation_manifest": sys.argv[2],
           "generation_manifest_sha256": sys.argv[3], "manifest_error": sys.argv[9], "manifest_pin": sys.argv[4], "host": sys.argv[5], "host_answer": sys.argv[6], "pass": sys.argv[7] == "true",
           "checked_pt": datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%F %T")}, open(sys.argv[8], "w"), indent=1)' \
    "$T389" "$GMF" "$GSHA" "$MAN" "$INST" "$BOXT" "$OK" "$(dirname "$0")/results/start_simh_q1_rlr_t389_check_$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S).json" "$GERR"
  [[ $OK == true ]] || { echo "T389 ASSERT FAIL: no start"; exit 6; }
  echo "t389 fork source == $T389 (manifest pin, recorded, copy, source): PASS"
fi
# 3. the reviewed on-box start (fail-closed); the branch commit recorded by the launcher. Codex r8: the run dir is
# registered for rl_pause_watch.sh BEFORE the start (the watcher pauses it as soon as it exists, so a pause that trips
# during this start still reaches it), the RL flag is re-checked last, and an SSH failure cannot skip the verdict.
RUN=/home/aswin/phase2/runs/2026-10-15_simh_$G
[[ ! -e $SP/simh_drive/RL_PAUSED ]] || { echo "RL PAUSE IN FORCE ($SP/simh_drive/RL_PAUSED): no start"; exit 2; }
echo "$INST $RUN" >> "$SP/simh_drive/simh_running.txt"
COMMIT=$(git -C "$WT" rev-parse HEAD)
set +e
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 900; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  "bash ~/phase2/simh_rlr_stage/$BOXSCRIPT $G $URLS $COMMIT $PREV $HOST $GSHA $ENVF") > "$SP/simh_drive/start_q${K}_last.out" 2>&1
rc=$?
set -e
tail -25 "$SP/simh_drive/start_q${K}_last.out"
LBL="simh rlr"; [[ $G == sp493_* || $G == t196fill ]] && LBL="simh sp493"  # Codex r33: the box script's own log label
if grep -Eq "^[0-9-]+ [0-9:]+ PT \[$LBL $G\] STARTED: simh${G//_/}, run $RUN, " "$SP/simh_drive/start_q${K}_last.out"; then
  echo "start_q: STARTED (ssh rc $rc); registered for the RL pause watcher"
  [[ ! -e $SP/simh_drive/RL_PAUSED ]] || echo "start_q: an RL pause tripped during this start; rl_pause_watch.sh pauses $RUN"
else echo "start_q: NOT STARTED or UNCERTAIN (ssh rc $rc): check the box before anything else"; exit 5; fi
