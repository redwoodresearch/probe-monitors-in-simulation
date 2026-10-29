#!/bin/bash
# The Seed-OSS-36B acting pilot's job watcher, the Llama-3.3-70B check's (e05dfe82) unchanged but for this comment: one
# hard stop for the WHOLE job (servers, smoke and devbox together), set from RULE.md's estimate (about 2x, Aswin 13:36 PT
# via 0d: "do this on policy, just run it ... go"; no price gate). Run from a FROZEN copy, started
# before anything paid, and left running from the smoke to the end.
#   usage: watch_job_frozen.sh <stop_dollars> <done_file> <apps_file> <devbox instance id> [<launcher results dir>]
# Apps: the union of every app ID ever seen in <apps_file> (must exist; created empty before the start) and, if given,
# in the launcher's servers_*.json files in <launcher results dir> written after this watcher started, plus every app
# whose name is one of those files' server names: launch_servers.py writes each name before it starts that server and
# its app ID only after the earlier servers' URLs, so the names catch every server of a launch as soon as its app
# exists (Codex 10-12). An ID once seen is never dropped. <apps_file>.launchers lists the PIDs of launch_servers.py
# runs (one per line); the stop sends each live one SIGINT (its own cleanup stops the apps it made) and is confirmed only
# once none is alive, so no launch in flight can add an app after the stop.
# Cost = max(Modal's billed $ for those apps over a UTC-safe window [UTC date - 1, UTC date + 2], their live containers
# x RATE integrated since the start + PRIOR_MODAL_USD) + the devbox's running time x DEVBOX_RATE + PRIOR_DEVBOX_USD.
# The stop acts when cost + the live rate x LOOKAHEAD_MIN reaches <stop_dollars>, so polling and teardown (about two
# minutes) finish under it: with 4 servers and the devbox live ($39.2/h) it acts at about $157.4.
# Fails closed (stops everything) when: the app list or the apps file is unreadable for FAIL_CLOSED_S, or a cost
# calculation fails. The stop first creates <apps_file>.stopping (no launch may start once it exists), then stops every
# live app (`modal app stop -y`, in parallel) and the devbox, re-reading the apps, and repeats until each app is stopped
# or gone from `modal app list` (a running app is always listed) with no containers and the devbox is stopping or
# stopped, twice in a row 20 s apart; it never gives up: after 9 rounds every further round prints an ALARM. On the done file it verifies the same
# teardown (and does it if anything is still live) before exiting 0.
STOP=$1; DONE=$2; APPS_FILE=$3; INSTANCE=$4; SERVERS_DIR=${5:-}
CALL_TIMEOUT=${CALL_TIMEOUT:-45}; POLL_S=${POLL_S:-30}; FAIL_CLOSED_S=${FAIL_CLOSED_S:-300}; LOOKAHEAD_MIN=${LOOKAHEAD_MIN:-4}
RATE=${RATE:-9.4}                    # $/h per container: 2 x H200 at $4.67 all-in (PRICE.md), rounded up
DEVBOX_RATE=${DEVBOX_RATE:-1.61}     # $/h, PRICE.md (SPEND_scaleup.md line 39)
PRIOR_MODAL_USD=${PRIOR_MODAL_USD:-0}; PRIOR_DEVBOX_USD=${PRIOR_DEVBOX_USD:-0}
M=(uv run --no-project --python 3.12 --with "modal==1.6.1" modal)   # pinned: 1.4.3's --json has other keys (Codex 10-12)
A=(aws --profile "${AWS_PROFILE:-redwood-sandbox}" --region us-west-2)
TMP=$(mktemp -d); MARK="$TMP/started"; touch "$MARK"
bounded() {  # run "$@" in its own process group; on timeout TERM then KILL the whole group, exit 124; signal -> 128+n
  perl -e 'my $t = shift; my $pid = fork(); die "fork" unless defined $pid;
    if ($pid == 0) { setpgrp(0, 0); exec @ARGV; exit 127 }
    my $end = time + $t; my $st;
    while (1) { my $r = waitpid($pid, 1); if ($r == $pid) { $st = $?; last }
      if (time >= $end) { kill "TERM", -$pid; select(undef, undef, undef, 2); kill "KILL", -$pid; waitpid($pid, 0); exit 124 }
      select(undef, undef, undef, 0.2) }
    exit($st & 127 ? 128 + ($st & 127) : $st >> 8)' "$CALL_TIMEOUT" "$@"; }
ts() { TZ=America/Los_Angeles date +%H:%M:%S; }
known=""; names=""
union() { tr ',' '\n' <<< "$1,$2" | grep -Ev '^-?$' | sort -u | paste -sd, -; }
registry() {  # union the apps file and the launcher's new servers_*.json (IDs and names) into $known, $names; 1 on failure
  [ -r "$APPS_FILE" ] || return 1
  local out
  out=$(python3 -c "
import json, os, re, sys
apps_file, d, mark = sys.argv[1:4]
ids, names = set(re.findall(r'ap-[A-Za-z0-9]+', open(apps_file).read())), set()
if d:
    t0 = os.path.getmtime(mark)
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if f.startswith('servers_') and f.endswith('.json') and os.path.getmtime(p) > t0:
            text = open(p).read()
            ids |= set(re.findall(r'ap-[A-Za-z0-9]+', text))
            try: names |= {r['name'] for r in json.loads(text).get('servers', []) if r.get('name')}
            except ValueError: pass  # caught mid-write: its IDs counted; its names at the next read
print(','.join(sorted(ids)) or '-', ','.join(sorted(names)) or '-')" "$APPS_FILE" "$SERVERS_DIR" "$MARK") || return 1
  known=$(union "$known" "${out% *}"); names=$(union "$names" "${out#* }"); return 0; }
launchers_alive() { local p n=0; [ -r "$APPS_FILE.launchers" ] || { echo 0; return; }
  for p in $(grep -Eo '^[0-9]+$' "$APPS_FILE.launchers"); do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo $n; }
poll() {  # the three reads in parallel, each bounded; results in $TMP
  rm -f "$TMP"/*.out "$TMP"/*.rc
  ( bounded "${M[@]}" app list --json > "$TMP/apps.out" 2>/dev/null; echo $? > "$TMP/apps.rc" ) &
  ( bounded "${M[@]}" billing report --start "$(date -u -v-1d +%Y-%m-%d)" --end "$(date -u -v+2d +%Y-%m-%d)" \
      --resolution h --json > "$TMP/bill.out" 2>/dev/null; echo $? > "$TMP/bill.rc" ) &
  ( bounded "${A[@]}" ec2 describe-instances --instance-ids "$INSTANCE" \
      --query 'Reservations[0].Instances[0].State.Name' --output text > "$TMP/dev.out" 2>/dev/null; echo $? > "$TMP/dev.rc" ) &
  wait; }
apps_state() {  # "<live tasks> <live app IDs or -> <all known IDs>" for $known plus apps named in $names, or FAIL
  [ "$(cat "$TMP/apps.rc" 2>/dev/null)" = 0 ] || { echo FAIL; return; }
  python3 -c "
import json,sys
try:
    apps = json.load(open(sys.argv[3]))
    assert isinstance(apps, list) and all({'app_id', 'state', 'tasks', 'description'} <= set(a) for a in apps)
    by = {a['app_id']: a for a in apps}
    names = [n for n in sys.argv[2].split(',') if n not in ('', '-')]
    want = sorted({w for w in sys.argv[1].split(',') if w not in ('', '-')}
                  | {a['app_id'] for a in apps if any(n in (a['description'] or '') for n in names)})
    tasks = sum(int(by[w]['tasks'] or 0) for w in want if w in by)
    live = [w for w in want if w in by and (by[w]['state'] != 'stopped' or int(by[w]['tasks'] or 0) > 0)]
    print(tasks, ','.join(live) or '-', ','.join(want) or '-')
except Exception: print('FAIL')" "$known" "$names" "$TMP/apps.out"; }
billed() {  # summed billed $ of $known, or FAIL
  [ "$(cat "$TMP/bill.rc" 2>/dev/null)" = 0 ] || { echo FAIL; return; }
  python3 -c "
import json,sys
try:
    rows = json.load(open(sys.argv[2]))
    assert isinstance(rows, list) and all({'object_id', 'cost'} <= set(r) for r in rows)
    want = set(w for w in sys.argv[1].split(',') if w)
    print(round(sum(float(r['cost']) for r in rows if r['object_id'] in want), 3))
except Exception: print('FAIL')" "$known" "$TMP/bill.out"; }
devbox() {  # the instance state, or FAIL; instance "none": no devbox in the job (tests)
  [ "$INSTANCE" = none ] && { echo stopped; return; }
  local d; d=$(cat "$TMP/dev.out" 2>/dev/null | tr -d '[:space:]')
  [ "$(cat "$TMP/dev.rc" 2>/dev/null)" = 0 ] && [[ "$d" =~ ^(pending|running|stopping|stopped|shutting-down|terminated)$ ]] \
    && echo "$d" || echo FAIL; }
stop_all() {  # stop until confirmed twice in a row; never gives up
  touch "$APPS_FILE.stopping"; local round=0 ok=0 st t live all d l p
  while true; do
    round=$((round + 1)); registry || echo "$(ts) ALARM: apps file unreadable during the stop; using the IDs already seen"
    l=$(launchers_alive)
    if [ "$l" != 0 ]; then for p in $(grep -Eo '^[0-9]+$' "$APPS_FILE.launchers"); do kill -INT "$p" 2>/dev/null; done; fi
    poll; st=$(apps_state); d=$(devbox)
    if [ "$st" = FAIL ]; then t="?"; live=$known
    else read -r t live all <<< "$st"; known=$(union "$known" "$all"); fi
    if [ "$st" != FAIL ] && [ "$t" = 0 ] && [ "$live" = "-" ] && [ "$l" = 0 ] && { [ "$d" = stopped ] || [ "$d" = stopping ]; }; then
      ok=$((ok + 1))
      if [ "$ok" -ge 2 ]; then echo "$(ts) stop confirmed (round $round, twice): apps ${known:--} stopped or gone, 0 containers, no launcher, devbox $d"; return 0; fi
      sleep 20; continue
    fi
    ok=0; echo "$(ts) stop round $round: live apps ${live:--} ($t containers), launchers alive $l, devbox $d"
    [ "$round" -gt 9 ] && echo "$(ts) ALARM: not confirmed stopped after $round rounds; still retrying"
    for a in ${live//,/ }; do [ "$a" = "-" ] || bounded "${M[@]}" app stop "$a" -y >/dev/null 2>&1 & done
    { [ "$d" = stopped ] || [ "$d" = stopping ]; } || bounded "${A[@]}" ec2 stop-instances --instance-ids "$INSTANCE" >/dev/null 2>&1 &
    wait; sleep 20
  done; }
num() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
est=$PRIOR_MODAL_USD; dev_usd=$PRIOR_DEVBOX_USD; tasks=0; last=$(date +%s); fail_since=""
while true; do
  if [ -f "$DONE" ]; then
    registry; echo "$(ts) done file present; verifying teardown (integrated \$$est, devbox \$$dev_usd)"
    stop_all; echo "$(ts) exiting"; exit 0; fi
  reg_ok=1; registry || { reg_ok=0; echo "$(ts) ALARM: apps file unreadable"; }
  poll; st=$(apps_state)
  if [ "$st" != FAIL ]; then read -r t_now live all <<< "$st"; known=$(union "$known" "$all"); fi
  b=$(billed); d=$(devbox); now=$(date +%s)
  if [ "$st" = FAIL ]; then echo "$(ts) ALARM: poll failed (app list); counting the last known $tasks containers"
  else tasks=$t_now; fi
  [ "$b" = FAIL ] && echo "$(ts) ALARM: poll failed (billing report)"
  [ "$d" = FAIL ] && echo "$(ts) ALARM: poll failed (devbox state); counting it as running"
  if [ "$st" = FAIL ] || [ "$reg_ok" = 0 ]; then fail_since=${fail_since:-$now}; else fail_since=""; fi
  out=$(python3 -c "
est = $est + $tasks * $RATE * ($now - $last) / 3600
on = '$d' not in ('stopped',)
dev = $dev_usd + ($DEVBOX_RATE * ($now - $last) / 3600 if on else 0)
cost = max(est, float('$b') + $PRIOR_MODAL_USD if '$b' != 'FAIL' else 0.0) + dev  # est already holds the prior
rate = $tasks * $RATE + ($DEVBOX_RATE if on else 0)
print(round(est, 3), round(dev, 3), round(cost, 3), round(cost + rate * $LOOKAHEAD_MIN / 60, 3))" 2>/dev/null)
  read -r e2 dv2 cost proj <<< "$out"
  if ! { num "$e2" && num "$dv2" && num "$cost" && num "$proj"; }; then
    echo "$(ts) ALARM: cost calculation failed ('$out'); failing closed: stopping apps and devbox"; stop_all; exit 4; fi
  est=$e2; dev_usd=$dv2; last=$now
  echo "$(ts) heartbeat apps=${known:--} tasks=$tasks integrated=\$$est billed=\$$b devbox=$d \$$dev_usd cost=\$$cost projected=\$$proj stop=\$$STOP"
  if [ -n "$fail_since" ] && [ $((now - fail_since)) -ge "$FAIL_CLOSED_S" ]; then
    echo "$(ts) ALARM: app list or apps file unreadable for $((now - fail_since)) s; failing closed: stopping apps and devbox"
    stop_all; exit 4; fi
  if python3 -c "import sys; sys.exit(0 if float('$proj') >= float('$STOP') else 1)"; then
    echo "$(ts) STOP threshold reached (\$$cost now, \$$proj in $LOOKAHEAD_MIN min >= \$$STOP): stopping apps and devbox"
    stop_all; exit 2; fi
  sleep "$POLL_S"
done
