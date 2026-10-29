#!/bin/bash
# Start one +311 simulated-honest quarter (d9's GO): census gate, server liveness, then the reviewed on-box start script.
#   start_q.sh <k> <host instance> <host label> <host's previous run dir> <servers json> <generation manifest sha256>
set -euo pipefail
K=$1 INST=$2 HOST=$3 PREV=$4 SJ=$5 GSHA=$6
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
WT=$SP/wt_simh311; H=$SP/simh_launch_X/h; REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
echo "== $(TZ=America/Los_Angeles date '+%F %T PT') start_q q$K on $HOST ($INST)"
# 1. census gate (rule of d9 09:27 PT 10-15; from ~14:30 census_rl_v6.py gate: <= 95, running launches at admission capacity (0d after the 14:23 RL)): the newest committed census, <= 30 min old, complete,
# < 90 sims running, no RateLimitError in the latest 10-min bin; and no rate-limit pause in force (rl_pause_watch.sh)
[[ ! -e $SP/simh_drive/RL_PAUSED ]] || { echo "RL PAUSE IN FORCE ($SP/simh_drive/RL_PAUSED): no start"; exit 2; }
git -C "$REPO" fetch -q origin aswin/q3-new-mains-launch
CF=$(git -C "$REPO" ls-tree --name-only origin/aswin/q3-new-mains-launch experiments/2026-10-14_q3_new_mains_launch/results/ | grep 'sim_error_census_' | sort | tail -1)
# v3 (d9 ~11:45, 0d): projected = running + this launch's own concurrency (cap / $106, rounded up) + sel97's reserve gap
# (flags in scratchpad/sel97/) [+ ug's gap while il_q4c runs]; <= 110 (< 90 while il_q4c runs)
CONC=$(python3 -I -c 'import math, re, sys; t = open(sys.argv[1]).read(); print(math.ceil(int(re.search(r"^  cap_usd: ([1-9][0-9]*)(?:[ \t]+#[^\n]*)?$", t, re.M).group(1)) / 106))' "$(dirname "$0")/config/pilot_simh_q$K.yaml")
git -C "$REPO" show "origin/aswin/q3-new-mains-launch:$CF" | python3 -I "$(dirname "$0")/census_rl_v6.py" gate --start-conc "$CONC" --sel97-dir "$SP/sel97" || { echo "CENSUS GATE FAIL ($CF)"; exit 2; }
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
# 2b. approaches (d9 10-15): every approach_sha256 in the follow-up task file the launcher materializes (cfg tasks =
# config/tasks_311_q<k>_fu.json) is a regular, non-symlink ~/phase2/approaches/<sha>.txt hashing to its name. Hash fields
# and sha256 only; no text is opened beyond hashing.
APPR=$( (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  "python3 -I -c 'import json, sys, os, hashlib
s = sorted({c[\"approach_sha256\"] for c in json.load(open(sys.argv[1]))[\"cells\"]})
ok = [h for h in s if os.path.isfile(p := f\"/home/aswin/phase2/approaches/{h}.txt\") and not os.path.islink(p) and hashlib.sha256(open(p, \"rb\").read()).hexdigest() == h]
print(f\"APPROACHES needed={len(s)} ok={len(ok)}\")' /home/aswin/phase2/attack_check_311/config/tasks_311_q${K}_fu.json") 2> /dev/null | grep -E '^APPROACHES ' || true)
echo "${APPR:-approach check gave no answer}"
[[ $APPR =~ ^APPROACHES\ needed=([0-9]+)\ ok=([0-9]+)$ && ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" && ${BASH_REMATCH[1]} -gt 0 ]] || { echo "APPROACH FILES MISSING"; exit 4; }
# 3. the reviewed on-box start (fail-closed); the branch commit recorded by the launcher. Codex r8: the run dir is
# registered for rl_pause_watch.sh BEFORE the start (the watcher pauses it as soon as it exists, so a pause that trips
# during this start still reaches it), the RL flag is re-checked last, and an SSH failure cannot skip the verdict.
RUN=/home/aswin/phase2/runs/2026-10-15_simh_q$K
[[ ! -e $SP/simh_drive/RL_PAUSED ]] || { echo "RL PAUSE IN FORCE ($SP/simh_drive/RL_PAUSED): no start"; exit 2; }
echo "$INST $RUN" >> "$SP/simh_drive/simh_running.txt"
COMMIT=$(git -C "$WT" rev-parse HEAD)
set +e
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 900; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  "bash ~/phase2/simh_stage/simh311_box.sh $K $URLS $COMMIT $PREV $HOST $GSHA") > "$SP/simh_drive/start_q${K}_last.out" 2>&1
rc=$?
set -e
tail -25 "$SP/simh_drive/start_q${K}_last.out"
if grep -Eq "^[0-9-]+ [0-9:]+ PT \[simh q$K\] STARTED: simh311q$K, run $RUN, " "$SP/simh_drive/start_q${K}_last.out"; then
  echo "start_q: STARTED (ssh rc $rc); registered for the RL pause watcher"
  [[ ! -e $SP/simh_drive/RL_PAUSED ]] || echo "start_q: an RL pause tripped during this start; rl_pause_watch.sh pauses $RUN"
else echo "start_q: NOT STARTED or UNCERTAIN (ssh rc $rc): check the box before anything else"; exit 5; fi
