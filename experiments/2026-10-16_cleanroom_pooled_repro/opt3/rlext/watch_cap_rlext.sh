#!/bin/bash
# Spend-cap enforcement for the rlext run (stands in for refresh3's ARMED cap, which r2.7's RUN_ME required; refresh3 is closed).
# ARMED BEFORE LAUNCH (no arguments): snapshot the cleanroom app's existing app_ids (refuse if any is live), then track EVERY NEW
# app_id ever seen under that description. It never disarms on its own except on a confirmed stop: exit 0 = every seen app is confirmed
# "stopped" by a validated listing; exit 2 = the cap stop was confirmed. If no new app ever appears (RUN_ME failed before launch),
# the operator ends the watcher (TaskStop) only after checking the RUN_ME log (no app id, non-zero exit). Every external call
# (list, stop) is bounded: started in the background, polled each second, TERM at 45 s, KILL 3 s later, then reaped. A listing is
# accepted only if it is exactly ONE JSON document, an array of objects with string app_id / state / description, with unique
# app_ids; the tracked app must match exactly one row. Anything else is "unknown" and never "done".
# Cap: the deadline is arm + 5100 s. Before it, each cycle is <= 48 s (list) + 30 s (sleep), so the stop starts by arm + 5178 s.
# Stop cycles are <= 48 s (stop) + 30 s + 48 s (list) = 126 s. If Modal confirms the stop within 3 cycles, containers run
# <= 5178 + 378 = 5556 s; worst case (bc) 5556 x (2 x $0.00035168 + $0.00001754) = $4.0053 < $5 (at most the 2 workers + driver at once),
# IF the app is identified by the first valid listing after launch (else it is stopped on discovery; if none is identified by the
# deadline the automatic cap is NOT in force and 0d stops it by hand).
# If the stop is not confirmed, the watcher prints an ALERT every cycle (about $0.09 per extra cycle) and keeps trying; 0d escalates.
# Usage: bash opt3/rlext/watch_cap_rlext.sh > <log> 2>&1   (start BEFORE RUN_ME_opt3_rlext.sh; wait for "WATCH ARMED")
set -uo pipefail
APP=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
BUDGET=5100
VALID='length == 1 and (.[0] | type == "array" and all(.[]; type == "object" and ((.app_id | type) == "string") and ((.state | type) == "string") and ((.description | type) == "string")) and ((map(.app_id) | unique | length) == length))'
bounded() {  # bounded <outfile> <cmd...>: runs cmd with stdout to outfile; returns its status, or 124 if it had to be killed
  local out=$1; shift
  "$@" > "$out" 2>/dev/null &
  local p=$! i=0
  while kill -0 "$p" 2>/dev/null && [ "$i" -lt 45 ]; do sleep 1; i=$((i + 1)); done
  if kill -0 "$p" 2>/dev/null && jobs -p | grep -qx "$p"; then
    kill -TERM "$p" 2>/dev/null; sleep 3
    kill -0 "$p" 2>/dev/null && jobs -p | grep -qx "$p" && kill -KILL "$p" 2>/dev/null
    wait "$p" 2>/dev/null; return 124
  fi
  wait "$p"; return $?
}
lst() {  # prints the validated listing, or the word INVALID
  local f; f=$(mktemp)
  if bounded "$f" "$M" app list --json && jq -e -s "$VALID" "$f" > /dev/null 2>&1; then jq -c . "$f"; else echo INVALID; fi
  rm -f "$f"
}
state_of() {  # state_of <json> <id>: the state of the ONE row with that id, else "not-listed" / "ambiguous"
  printf '%s' "$1" | jq -r --arg i "$2" '[.[] | select(.app_id == $i) | .state] | if length == 1 then .[0] elif length == 0 then "not-listed" else "ambiguous" end'
}
# fixtures (fail closed if the validators misbehave)
v() { printf '%s' "$1" | jq -e -s "$VALID" > /dev/null 2>&1; }
v '[{"app_id":"ap-a","state":"stopped","description":"d"}]' || { echo "WATCH STOP: fixture valid rejected"; exit 3; }
v '[]' || { echo "WATCH STOP: fixture empty rejected"; exit 3; }
v '[{"app_id":"ap-a","description":"d"}]' && { echo "WATCH STOP: fixture missing state accepted"; exit 3; }
v '[{"app_id":"ap-a","state":"stopped","description":"d"},{"app_id":"ap-a","state":"ephemeral","description":"d"}]' && { echo "WATCH STOP: fixture duplicate id accepted"; exit 3; }
v '[] []' && { echo "WATCH STOP: fixture two documents accepted"; exit 3; }
[ "$(state_of '[{"app_id":"ap-a","state":"ephemeral","description":"d"}]' ap-a)" = "ephemeral" ] || { echo "WATCH STOP: fixture state_of"; exit 3; }
[ "$(state_of '[]' ap-a)" = "not-listed" ] || { echo "WATCH STOP: fixture state_of not-listed"; exit 3; }
f=$(mktemp); bounded "$f" sleep 60; rc=$?; rm -f "$f"; [ "$rc" = "124" ] || { echo "WATCH STOP: fixture bounded did not kill a 60 s call (rc $rc)"; exit 3; }
# unres <json> <seen>: every seen id whose ONE matching row is not "stopped" (missing or ambiguous rows count as unresolved)
unres() { printf '%s' "$1" | jq -c --argjson s "$2" '. as $J | [$s[] as $i | ($J | map(select(.app_id == $i)) | if length == 1 then .[0].state elif length == 0 then "not-listed" else "ambiguous" end) as $st | select($st != "stopped") | {id: $i, state: $st}]'; }
[ "$(unres '[{"app_id":"a","state":"stopped","description":"d"},{"app_id":"b","state":"ephemeral","description":"d"}]' '["a","b"]' | jq -c 'map(.id)')" = '["b"]' ] || { echo "WATCH STOP: fixture unres live"; exit 3; }
[ "$(unres '[{"app_id":"a","state":"stopped","description":"d"}]' '["a","b"]' | jq -c 'map(.state)')" = '["not-listed"]' ] || { echo "WATCH STOP: fixture unres not-listed"; exit 3; }
[ "$(unres '[{"app_id":"a","state":"stopped","description":"d"},{"app_id":"b","state":"stopped","description":"d"}]' '["a","b"]')" = '[]' ] || { echo "WATCH STOP: fixture unres done"; exit 3; }
T0=$(date +%s); DL=$((T0 + BUDGET))
J=$(lst); [ "$J" = "INVALID" ] && { echo "WATCH STOP: first listing invalid; do not launch"; exit 3; }
BASE=$(printf '%s' "$J" | jq -c --arg a "$APP" '[.[] | select(.description == $a) | .app_id]')
NLIVE=$(printf '%s' "$J" | jq --arg a "$APP" '[.[] | select(.description == $a and .state != "stopped")] | length')
[ "$NLIVE" = "0" ] || { echo "WATCH STOP: $NLIVE live $APP apps before launch; do not launch"; exit 3; }
echo "WATCH ARMED $(TZ=America/Los_Angeles date '+%H:%M:%S') budget ${BUDGET}s deadline $(TZ=America/Los_Angeles date -r "$DL" '+%H:%M:%S'); $(printf '%s' "$BASE" | jq length) existing ids snapshotted"
SEEN='[]'; prev=""
while true; do
  J=$(lst); now=$(date +%s)
  if [ "$J" = "INVALID" ]; then line="unknown(listing invalid or timed out)"; U=""
  else
    NEWNOW=$(printf '%s' "$J" | jq -c --arg a "$APP" --argjson b "$BASE" '[.[] | select(.description == $a and (.app_id as $i | $b | index($i) | not)) | .app_id]')
    S2=$(jq -cn --argjson s "$SEEN" --argjson n "$NEWNOW" '$s + $n | unique')
    [ "$S2" != "$SEEN" ] && { echo "WATCH NEW-APPS $(TZ=America/Los_Angeles date '+%H:%M:%S') seen=$S2"; SEEN=$S2; }
    [ "$(printf '%s' "$SEEN" | jq length)" -gt 1 ] && echo "WATCH ALERT more than one new $APP app seen: $SEEN (all are tracked; operator check)"
    U=$(unres "$J" "$SEEN"); line="seen=$(printf '%s' "$SEEN" | jq length) unresolved=$U"
  fi
  [ "$line" != "$prev" ] && { echo "WATCH $(TZ=America/Los_Angeles date '+%H:%M:%S') $line"; prev=$line; }
  if [ -n "$U" ] && [ "$(printf '%s' "$SEEN" | jq length)" -ge 1 ] && [ "$U" = "[]" ]; then echo "WATCH DONE all seen apps stopped (confirmed): $SEEN"; exit 0; fi
  if [ "$now" -ge "$DL" ]; then
    if [ "$(printf '%s' "$SEEN" | jq length)" = "0" ]; then echo "WATCH ALERT deadline passed with no new app identified: AUTOMATIC CAP NOT IN FORCE (0d: check the RUN_ME log and stop any $APP app by hand now)"; sleep 30; continue; fi
    echo "WATCH CAP deadline reached -> stopping every unresolved app: $U"
    n=0
    while true; do
      n=$((n + 1))
      for id in $(printf '%s' "${U:-[]}" | jq -r '.[].id') $( [ -z "$U" ] && printf '%s' "$SEEN" | jq -r '.[]' ); do
        f=$(mktemp); bounded "$f" "$M" app stop -y "$id"; rc=$?; rm -f "$f"
        [ "$rc" = "0" ] || echo "WATCH ALERT app stop $id attempt $n returned $rc"
      done
      sleep 30
      J=$(lst)
      if [ "$J" != "INVALID" ]; then
        NEWNOW=$(printf '%s' "$J" | jq -c --arg a "$APP" --argjson b "$BASE" '[.[] | select(.description == $a and (.app_id as $i | $b | index($i) | not)) | .app_id]')
        S2=$(jq -cn --argjson s "$SEEN" --argjson n "$NEWNOW" '$s + $n | unique')
        [ "$S2" != "$SEEN" ] && { echo "WATCH ALERT NEW-APPS during cap stop: $S2"; SEEN=$S2; }
        U=$(unres "$J" "$SEEN")
        [ "$U" = "[]" ] && { echo "WATCH CAP STOPPED all seen apps (confirmed, attempt $n): $SEEN"; exit 2; }
      else U=""; fi
      echo "WATCH ALERT not all confirmed stopped after attempt $n at $(TZ=America/Los_Angeles date '+%H:%M:%S'): ${U:-listing invalid}; retrying"
    done
  fi
  sleep 30
done
