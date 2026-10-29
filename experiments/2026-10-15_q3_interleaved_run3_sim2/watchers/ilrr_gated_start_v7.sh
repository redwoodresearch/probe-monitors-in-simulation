#!/usr/bin/env bash
# ilrr_gated_start.sh (d9 02:55 GO): ONE gated start of one paired rate-limit rerun (ilrr_box.sh <key>) on its host.
# Every 150 s, read-only, until ALL hold (then one start, then exit; ends 07:00 PT):
#  1. servers: the server-launch results JSON (never retyped) has status "up" and 6 servers; all 6 app IDs are deployed
#     with >= 1 running task (modal app list). With HOLD=1 (servers that are already up, e.g. q7) a server that is not
#     live makes the starter EXIT with "HOLD" instead of waiting (d9: if q7 idles out, hold and tell me);
#  2. host: the staged ilrr_box.sh and the key's three files match the committed sha256s; every earlier launch on the
#     host finished; no run_pilot.py; no PAUSE_SIMS in the key's harness; no pause sleeper;
#  3. census: the latest committed census on origin/aswin/q3-new-mains-launch is <= 30 min old, its last-30-min
#     RateLimitError rate < 5% and "sims running now" < 60;
#  4. pacing: >= 15 min since the last IL start by any starter (a line "starting il" in scratchpad il_watch/*.log),
#     checked again under a lock (mkdir) so two starters never start together.
# The start itself is ilrr_box.sh, which re-checks everything fail-closed. Never stops or edits anything on the host.
# v3 (d9 03:20, s3 pre-authorized): key s3 dispatches il_nm_box.sh 3 (its staged files checked at COMMIT) after q3's
# reruns finished; every key also refuses while a pause/kill sleeper of the nm cleanup's pattern exists. Otherwise v2.
# v4 (completion launches, d9 03:27 GO after r19 + 09's registration + the gate change): keys c<h> dispatch
# ilc_box.sh <h> on the named host; staged files checked at COMMIT (ilc_box.sh, the host's three completion files, the gate).
# v5 (0d 03:52): keys s1r2/s2r2 dispatch ilrr2_box.sh (the s1/s2 reruns with admitting caps); xbox2/xbox3 completions wait
# for those; and EVERY key refuses unless the committed config's cap >= $75 x its minimum parallel cells (c* 10, else 1).
# v6 (addendum 04:08): key q4r2 dispatches ilrr2_box.sh q4b1 (q4's rerun at the amended cap); keys r3<case> dispatch
# ilrr3_<host>_box.sh <case> (rlr3); an optional 7th argument adds run dirs that must have finished (rlr3 before the host's
# completion); the admission floor for non-completion keys is min(cells, 5) x $75 from the committed task file.
# v7 (d9 04:28, 0d): the rlr3 window. For completion keys whose box ran a paired rerun that may need rlr3 (cq2: q2_rlr,
# cold: old_rlr, cq4: q1_rlr), once that rerun has finished: if it left rlr3-eligible cells (BOTH arms 'not run', 0 attempts,
# sim2 why 'spend cap'), the completion WAITS up to 15 min after the rerun's finish time; a file il_watch/yield_<key> in
# that window makes this starter exit without starting (rlr3 registered: it goes first). After 15 min, the completion starts.
#   ilrr_gated_start_v7.sh <key> <instance> <servers results JSON> <commit> [HOLD|-] [PACE seconds, default 900]
# v2 (Codex r18b probes): base URLs must match a strict pattern (no shell metacharacters); the census rate is recomputed
# from its counts; PACE is a parameter (d9 03:00: with a clean census and < 60 sims running, starts may overlap: PACE 0).
# Codex r18b fixes: the commit must resolve locally and every committed-file read must succeed; distinct apps and URLs;
# remote arguments shell-quoted; census only after a successful fetch, read at the fetched revision; the start line must
# be recorded before dispatch; the deadline is re-checked right before dispatch. (Other starters do not take this lock;
# with PACE 0 that only matters for one launcher per box, which ilrr_box.sh itself enforces.)
set -uo pipefail
K=$1 INST=$2 SJSON=$3 COMMIT=$4 HOLD=${5:-}; [[ $HOLD == - ]] && HOLD=; PACE=${6:-900}; EXTRA=${7:-}
[[ $PACE =~ ^[0-9]+$ && $COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "bad PACE or COMMIT"; exit 2; }
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; W=$SP/il_watch; LOG=$W/rr_$K.log; M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
EXP=$SP/wt_interleave/experiments/2026-10-15_q3_interleaved_run3_sim2
END=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' "2026-10-15 07:00:00" +%s)
R311=/home/aswin/phase2/runs; RNM=/home/aswin/phase2/nm/runs
case $K in
  q1) HF=q1; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q1" ;;
  q2) HF=q2; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q2" ;;
  q3) HF=q3; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q3" ;;
  old) HF=old; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_old" ;;
  q4b1) HF=q4; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q4 $R311/2026-10-14_il_q1 $R311/2026-10-15_il_q1_rlr" ;;
  q4b2) HF=q4; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q4 $R311/2026-10-14_il_q2 $R311/2026-10-15_il_q2_rlr" ;;
  s1) HF=s1; HARN=nm/attack_check_nm; PREVS="$RNM/2026-10-14_il_s1 $RNM/2026-10-14_il_s4d" ;;
  s2) HF=s2; HARN=nm/attack_check_nm; PREVS="$RNM/2026-10-14_il_s2" ;;
  s3) HF=s3; HARN=nm/attack_check_nm; PREVS="$R311/2026-10-14_il_q3 $R311/2026-10-15_il_q3_rlr" ;;
  cq4) HF=q4; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q4 $R311/2026-10-14_il_q1 $R311/2026-10-15_il_q1_rlr $R311/2026-10-15_il_q4_rlr2" ;;
  cq2) HF=q2; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q2 $R311/2026-10-15_il_q2_rlr" ;;
  cold) HF=old; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_old $R311/2026-10-15_il_old_rlr" ;;
  cq1) HF=q1; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q1 $RNM/2026-10-14_il_s1 $RNM/2026-10-14_il_s4d $RNM/2026-10-15_il_s1_rlr2" ;;
  cq3) HF=q3; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q3 $RNM/2026-10-14_il_s2 $RNM/2026-10-15_il_s2_rlr $RNM/2026-10-15_il_s2_rlr2" ;;
  s1r2) HF=s1; HARN=nm/attack_check_nm; PREVS="$RNM/2026-10-14_il_s1 $RNM/2026-10-14_il_s4d" ;;
  s2r2) HF=s2; HARN=nm/attack_check_nm; PREVS="$RNM/2026-10-14_il_s2 $RNM/2026-10-15_il_s2_rlr" ;;
  q4r2) HF=q4; HARN=attack_check_311; PREVS="$R311/2026-10-14_il_q4 $R311/2026-10-14_il_q1 $R311/2026-10-15_il_q1_rlr" ;;
  r3q1|r3q2|r3q3|r3old|r3q4b1|r3s1|r3s2) CASE=${K#r3}; HF=${CASE%b1}; [[ $HF == s? ]] && HARN=nm/attack_check_nm || HARN=attack_check_311
      PREVS="" ;;  # rlr3: the box script pins its own PREVS; the 7th argument lists them here too
  *) echo "unknown key $K"; exit 2 ;;
esac
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [rr $K] $*" >> "$LOG"; }
# the committed sha256s of what must be staged (from the worktree at COMMIT, never retyped)
(cd "$EXP" && git cat-file -e "$COMMIT^{commit}") 2> /dev/null || { log "commit $COMMIT not found"; echo "bad commit"; exit 2; }
SHAS=""; BLOB=$(mktemp "$W/.rr_${K}_XXXXXX") || { echo "mktemp failed"; exit 2; }
[[ -z $EXTRA ]] || PREVS="$PREVS ${EXTRA//,/ }"
[[ -n $PREVS ]] || { echo "no must-finish list"; exit 2; }
case $K in cq2) RLR=$R311/2026-10-15_il_q2_rlr ;; cold) RLR=$R311/2026-10-15_il_old_rlr ;; cq4) RLR=$R311/2026-10-15_il_q1_rlr ;; *) RLR= ;; esac
if [[ $K == c* ]]; then FILES="ilc_box.sh config/completion/pilot_ilc_$HF.yaml config/completion/tasks_ilc_$HF.json config/completion/ilc_supersession_tier3_$HF.json gate_pause_311_rl.py"
  STAGED="ilc_box.sh pilot_ilc_$HF.yaml tasks_ilc_$HF.json ilc_supersession_tier3_$HF.json checks/gate_pause_311_rl.py"; DISPATCH=(bash /home/aswin/phase2/il_stage/ilc_box.sh "$HF")
elif [[ $K == q4r2 ]]; then FILES="ilrr2_box.sh config/pilot_ilrr2_q4.yaml config/tasks_ilrr_q4.json config/ilrr_rerun_of_q4.json"
  STAGED="ilrr2_box.sh pilot_ilrr2_q4.yaml tasks_ilrr_q4.json ilrr_rerun_of_q4.json"; DISPATCH=(bash /home/aswin/phase2/il_stage/ilrr2_box.sh q4b1)
elif [[ $K == r3* ]]; then FILES="ilrr3_${HF}_box.sh config/pilot_ilrr3_$HF.yaml config/tasks_ilrr3_$HF.json config/ilrr3_rerun_of_$HF.json"
  STAGED="ilrr3_${HF}_box.sh pilot_ilrr3_$HF.yaml tasks_ilrr3_$HF.json ilrr3_rerun_of_$HF.json"; DISPATCH=(bash /home/aswin/phase2/il_stage/ilrr3_${HF}_box.sh "$CASE")
elif [[ $K == s?r2 ]]; then FILES="ilrr2_box.sh config/pilot_ilrr2_$HF.yaml config/tasks_ilrr_$HF.json config/ilrr_rerun_of_$HF.json"
  STAGED="ilrr2_box.sh pilot_ilrr2_$HF.yaml tasks_ilrr_$HF.json ilrr_rerun_of_$HF.json"; DISPATCH=(bash /home/aswin/phase2/il_stage/ilrr2_box.sh "$HF")
elif [[ $K == s3 ]]; then FILES="il_nm_box.sh config/pilot_il_s3.yaml config/tasks_il_s3.json"; STAGED="il_nm_box.sh pilot_il_s3.yaml tasks_il_s3.json"
  DISPATCH=(bash /home/aswin/phase2/il_stage/il_nm_box.sh 3)
else FILES="ilrr_box.sh config/pilot_ilrr_$HF.yaml config/tasks_ilrr_$HF.json config/ilrr_rerun_of_$HF.json"
  STAGED="ilrr_box.sh pilot_ilrr_$HF.yaml tasks_ilrr_$HF.json ilrr_rerun_of_$HF.json"; DISPATCH=(bash /home/aswin/phase2/il_stage/ilrr_box.sh "$K"); fi
for f in $FILES; do
  B=$(cd "$EXP" && git cat-file -p "$COMMIT:experiments/2026-10-15_q3_interleaved_run3_sim2/$f" > "$BLOB" && shasum -a 256 < "$BLOB") \
    || { log "cannot read $f at $COMMIT"; echo "no committed shas"; exit 2; }
  [[ -s $BLOB && ${B:0:64} =~ ^[0-9a-f]{64}$ ]] || { log "empty or unhashable $f at $COMMIT"; echo "no committed shas"; exit 2; }
  SHAS+="${B:0:64} "
done
rm -f "$BLOB"; [[ $(wc -w <<< "$SHAS") == $(wc -w <<< "$FILES") ]] || { log "hash count"; echo "no committed shas"; exit 2; }
# admission (0d 03:52): the committed config's cap must admit the minimum parallel cells, or this starter refuses
CFGF=$(tr ' ' '\n' <<< "$FILES" | grep -E 'pilot_il' | head -1)
CAPV=$(cd "$EXP" && git cat-file -p "$COMMIT:experiments/2026-10-15_q3_interleaved_run3_sim2/$CFGF" | sed -nE 's/^  cap_usd: ([0-9]+).*/\1/p')
if [[ $K == c* ]]; then MINPAR=10
else  # min(cells, 5) from the committed task file
  TF=$(tr ' ' '\n' <<< "$FILES" | grep -E 'tasks_il' | head -1)
  NC=$(cd "$EXP" && git cat-file -p "$COMMIT:experiments/2026-10-15_q3_interleaved_run3_sim2/$TF" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["cells"]))')
  [[ $NC =~ ^[0-9]+$ && $NC -ge 1 ]] || { log "cannot count cells in $TF"; echo "admission refused"; exit 2; }
  MINPAR=$(( NC < 5 ? NC : 5 ))
fi
[[ $CAPV =~ ^[0-9]+$ ]] && (( CAPV >= 75 * MINPAR )) || { log "ADMISSION REFUSED: $CFGF cap '$CAPV' < \$75 x $MINPAR"; echo "admission refused"; exit 2; }
BOX="cd /home/aswin/phase2/il_stage 2>/dev/null || { echo 'WAIT no il_stage'; exit 0; }
[[ \$(sha256sum $STAGED 2>/dev/null | cut -c1-64 | tr '\n' ' ') == '$SHAS' ]] || { echo 'WAIT staged files differ from $COMMIT'; exit 0; }
for p in $PREVS; do python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get(\"finished\") else 1)' \$p/launch.json 2>/dev/null || { echo \"WAIT \$p not finished\"; exit 0; }; done
pgrep -f '[r]un_pilot\.py' > /dev/null; [[ \$? == 1 ]] || { echo 'WAIT launcher running (or pgrep error)'; exit 0; }
[[ ! -e /home/aswin/phase2/$HARN/results/PAUSE_SIMS ]] || { echo 'WAIT PAUSE_SIMS present'; exit 0; }
pgrep -f '[t]ouch /home/aswin/phase2/$HARN/results/PAUSE_SIMS' > /dev/null; [[ \$? == 1 ]] || { echo 'WAIT pause sleeper'; exit 0; }
pgrep -f 'sleep [0-9]+ && (touch /home/aswin/phase2/nm|tmux -L nm send-keys)' > /dev/null; [[ \$? == 1 ]] || { echo 'WAIT nm sleeper'; exit 0; }
echo BOX_READY"
servers() {  # prints SERVERS_LIVE <bases> | WAIT ... | DEAD ...
  rm -f "$W/applist_rr_$K.json"
  perl -e 'alarm 60; exec @ARGV' $M app list --json > "$W/applist_rr_$K.json" 2>/dev/null || { echo "WAIT modal app list failed"; return; }
  python3 - "$SJSON" "$W/applist_rr_$K.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("status") != "up" or len(d.get("servers", [])) != 6: print(f"WAIT servers status {d.get('status')}"); raise SystemExit
try: apps = {a["app_id"]: a for a in json.load(open(sys.argv[2]))}
except Exception: print("WAIT app list unreadable"); raise SystemExit
ids = [s["app_id"] for s in d["servers"]]
if len(set(ids)) != 6: print("DEAD duplicate app IDs"); raise SystemExit
live = [i for i in ids if i in apps and apps[i]["state"] == "deployed" and str(apps[i]["tasks"]).isdigit() and int(apps[i]["tasks"]) >= 1]
bases = [s["base_url"] for s in d["servers"]]
import re
ok = len(live) == 6 and len(set(bases)) == 6 and all(re.fullmatch(r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9._-]+)*", b) for b in bases)
print(("SERVERS_LIVE " + ",".join(bases)) if ok else f"DEAD live {len(live)} of 6")
PY
}
census() {
  (cd $SP/wt_interleave && perl -e 'alarm 60; exec @ARGV' git fetch -q origin aswin/q3-new-mains-launch) > /dev/null 2>&1 || { echo "WAIT census fetch failed"; return; }
  (cd $SP/wt_interleave && REV=$(git rev-parse --verify -q origin/aswin/q3-new-mains-launch) && F=$(git ls-tree -r --name-only "$REV" -- experiments/2026-10-14_q3_new_mains_launch/results/ | grep 'sim_error_census_2026-10-15_' | sort | tail -1) \
   && [[ -n $F ]] && S=$(git log -1 --format=%h --abbrev=9 "$REV" -- "$F") && git show "$REV:$F" | python3 -c "
import json, re, sys, datetime as dt
from zoneinfo import ZoneInfo
d = json.load(sys.stdin); t = d.get('census_time_pt', ''); txt = d.get('rates_text', '')
m = re.search(r'last 30 min[^\n]*RateLimitError (\d+) of (\d+) ended = ([0-9.]+)%', txt); n = re.search(r'sims running now (\d+)', txt)
now = dt.datetime.now(ZoneInfo('America/Los_Angeles')).replace(tzinfo=None)
try: ct = dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
except ValueError: print('WAIT census time unparsed: ' + t); raise SystemExit
if not m or not n or (now - ct).total_seconds() > 1800 or ct > now: print('WAIT census older than 30 min (or unparsed): ' + t); raise SystemExit
x, y = int(m.group(1)), int(m.group(2)); rate = 100 * x / y if y else 100.0  # no ended sims in the window: not evidence of a clean rate
ok = 0 < y and 0 <= x <= y and rate < 5 and float(m.group(3)) < 5 and int(n.group(1)) < 60
print(('CENSUS_OK ' if ok else 'WAIT census ') + sys.argv[1] + ' @ ' + sys.argv[2] + ' ' + t + ' RL ' + m.group(1) + '/' + m.group(2) + ' = ' + m.group(3) + '% running ' + n.group(1))
" "$F" "$S") 2>&1 | tail -1
}
last_start_age() {  # seconds since the newest "starting il" line in any il_watch log (none: a large number)
  python3 - "$W" <<'PY'
import glob, re, sys, datetime as dt
from zoneinfo import ZoneInfo
tz = ZoneInfo("America/Los_Angeles"); now = dt.datetime.now(tz); newest = None
for f in glob.glob(sys.argv[1] + "/*.log"):
    for line in open(f, errors="replace"):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) PT .*starting il", line)
        if m:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
            newest = t if newest is None or t > newest else newest
print(int((now - newest).total_seconds()) if newest else 10**9)
PY
}
log "v7 armed (cap $CAPV >= 75 x $MINPAR) (PACE ${PACE}s): key $K on $INST, servers $SJSON, commit $COMMIT, hold-if-dead '${HOLD}'; ends 07:00 PT"
while (( $(date +%s) < END )); do
  S=$(servers)
  if [[ $S != SERVERS_LIVE* ]]; then
    log "servers: $S"
    if [[ -n $HOLD && $S == DEAD* ]]; then log "HOLD: servers no longer live; not starting"; echo "HOLD $S"; exit 0; fi
    sleep 150; continue
  fi
  BASES=${S#SERVERS_LIVE }
  R=$(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 90; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$BOX" 2>&1 | tail -1)
  if [[ $R != BOX_READY ]]; then log "box: $R"; sleep 150; continue; fi
  if [[ -n $RLR ]]; then  # the rlr3 window (read-only: run states and the finished time of the box's rerun)
    E=$(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 90; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "python3 -c 'import json,sys,datetime as dt; d=json.load(open(sys.argv[1]+\"/launch.json\")); rs=d[\"runs\"]; rs=list(rs.values()) if isinstance(rs,dict) else rs; c={}; [c.setdefault(r[\"key\"].split(\"/\",1)[1],[]).append(r) for r in rs]; n=sum(1 for k,v in c.items() if len(v)==2 and all(r[\"state\"]==\"not run\" and not r.get(\"attempts\") for r in v) and any((not r[\"real\"]) and str(r.get(\"why\",\"\")).startswith(\"spend cap\") for r in v)); print(\"RLR3\", n, int(dt.datetime.fromisoformat(d[\"finished\"]).timestamp()))' $RLR" 2>/dev/null | grep '^RLR3 ' | tail -1)
    read -r _ NEL FINT <<< "$E"
    if [[ ! $NEL =~ ^[0-9]+$ || ! $FINT =~ ^[0-9]+$ ]]; then log "rlr3 window: cannot read $RLR ('$E')"; sleep 150; continue; fi
    if (( NEL > 0 )); then
      if [[ -e $W/yield_$K ]]; then  # only a registration made inside the window counts (Codex r23): the file's mtime <= finish + 15 min
        YMT=$(stat -f %m "$W/yield_$K" 2>/dev/null)
        if [[ $YMT =~ ^[0-9]+$ ]] && (( YMT <= FINT + 900 )); then log "rlr3 window: YIELD file from $(TZ=America/Los_Angeles date -r $YMT '+%T') PT ($NEL eligible cells); rlr3 goes first; not starting"; echo "YIELD to rlr3"; exit 0; fi
        log "rlr3 window: yield file unreadable or made after the window ($YMT); ignored"
      fi
      if (( $(date +%s) < FINT + 900 )); then log "rlr3 window: $NEL eligible cells; waiting until $(TZ=America/Los_Angeles date -r $((FINT + 900)) '+%T') PT for rlr3"; sleep 60; continue; fi
      log "rlr3 window closed ($NEL eligible cells, no rlr3 within 15 min): the completion starts first, rlr3 after it"
    fi
  fi
  C=$(census)
  if [[ $C != CENSUS_OK* ]]; then log "census: $C"; sleep 150; continue; fi
  A=$(last_start_age)
  if (( A < PACE )); then log "pacing: last IL start ${A}s ago"; sleep 150; continue; fi
  if ! mkdir "$W/start.lock" 2> /dev/null; then log "pacing: another starter holds the lock"; sleep 150; continue; fi
  if ! mkdir "$W/box_$INST.lock" 2> /dev/null; then rmdir "$W/start.lock"; log "another starter holds the $INST box lock"; sleep 150; continue; fi
  trap 'rmdir "$W/start.lock" "$W/box_$INST.lock" 2>/dev/null' EXIT
  A=$(last_start_age)
  if (( A < PACE )); then rmdir "$W/start.lock" "$W/box_$INST.lock"; trap - EXIT; log "pacing (under lock): last IL start ${A}s ago"; sleep 150; continue; fi
  if (( $(date +%s) >= END )); then log "deadline 07:00 reached at dispatch; not starting"; echo "deadline reached"; exit 0; fi
  echo "$(TZ=America/Los_Angeles date '+%F %T PT') [rr $K] ALL CONDITIONS HOLD (servers 6/6 live from $SJSON; box ready; $C; last start ${A}s ago): starting il ${DISPATCH[*]:1}" >> "$LOG" \
    || { echo "cannot record the start line; not starting"; exit 3; }
  RCMD=$(printf '%q ' "${DISPATCH[@]}" "$BASES" "$COMMIT")
  OUT=$(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 480; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "${RCMD}; echo EXIT=\$?" 2>&1 | grep -v '^config:' | tail -14)
  echo "$OUT" >> "$LOG"; echo "$OUT" | tail -4; echo "$C"
  if [[ $OUT == *$'\n'EXIT=0 || $OUT == EXIT=0 ]]; then log "dispatch returned EXIT=0"; exit 0; fi
  log "DISPATCH NOT CONFIRMED (no EXIT=0 from the host): check the host before anything else"; echo "DISPATCH NOT CONFIRMED"; exit 4
done
log "deadline 07:00 reached without starting"; echo "deadline reached"
