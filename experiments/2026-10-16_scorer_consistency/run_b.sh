#!/bin/zsh
# (b) runner (CUT fork, 10-16): a bounded billing read before (UTC today-3 .. today+2, <= 7 days; a failed read STOPs), the ONE detached driver
# (client up for the record; output to a log), and a WALL-CLOCK watchdog independent of the client: every 15 s it captures the app id from
# the log and, every 60 s, the app's state from `modal app list`; at 45 min it stops the app with `--yes </dev/null` and requires a POSITIVE
# "stopped" state (else STOP NOT CONFIRMED, exit 5). Every 5 min it also reads the app's billed cost (unrounded: whole micro-dollars rounded
# UP); above CAP ($5) it stops the app the same way (exit 8). A failed billing read after the run exits non-zero (7); a cost above CAP in that
# read exits 8. If the client exits and no app id is in the log, exit 10 (check `modal app list`). From just BEFORE launch, a signal
# (INT/TERM/HUP) or an unplanned exit stops the owned app the same way and reaps the client (exit 11); after a normal finish a signal exits 130.
# Exit = the run's exit code. RUNB_* env overrides exist ONLY for the local shell-logic test (test_run_b.sh): an ABSOLUTE path to the test's
# own fake binary (marker on its line 2), never the real modal; the real run uses the defaults.
#   zsh run_b.sh <reviewed code commit, full sha>
# Launch guard: the script sha is PIN; the argument is the last commit that touched run_b.sh / test_run_b.sh / the script (the one both
# reviewers passed), and on top of it only SPEND.md, NOTE_*.md and results/ records may change (diff --stat printed for the record).
set -u
set -o pipefail
REAL_M=/Users/aswinsurya/probe-distribution-shift-from-sims/.venv/bin/modal
if [ "${RUNB_TEST:-}" = "fake-modal-only" ]; then  # the local test: overrides allowed only with the test's fake binary (never the real one)
  F_=${RUNB_MODAL:-}
  [[ $F_ == /* && -f $F_ && -x $F_ && ${F_:A} != ${REAL_M:A} && "$(sed -n 2p -- $F_)" == "# run_b.sh test fake: not modal" ]] \
    || { echo "STOP: test mode needs RUNB_MODAL = the absolute path of the test's fake binary"; exit 3; }
  RUNB_MODAL=${F_:A}  # run exactly the file that was checked
else  # a real launch: refuse any leftover RUNB_* override (they could weaken the bounds)
  [ -z "$(env | grep '^RUNB_')" ] || { echo "STOP: RUNB_* overrides are set outside the test: $(env | grep -o '^RUNB_[A-Z_]*' | tr '\n' ' ')"; exit 3; }
fi
M=${RUNB_MODAL:-$REAL_M}
LIMIT=${RUNB_LIMIT_S:-2700}; POLL=${RUNB_POLL_S:-15}; IDWAIT=${RUNB_ID_WAIT_S:-30}; BILLEVERY=${RUNB_BILL_EVERY_S:-300}; POSTSLEEP=${RUNB_POST_SLEEP_S:-60}; CAP=5
REPO=${0:A:h:h:h}
cd $REPO
bounded() { perl -e 'alarm shift; exec @ARGV' "$@"; }
S=$(date -u -v-3d +%Y-%m-%d); E=$(date -u -v+2d +%Y-%m-%d)
bill() {  # $1 = app id or "" -> "rows micro-dollars" (the app's exact decimal total, rounded UP to a whole micro-dollar); a failed,
  local out  # timed-out, empty or non-finite read returns non-zero
  out=$(bounded 120 $M billing report --start $S --end $E --resolution h --json 2>/dev/null) || return 1
  print -r -- "$out" | /usr/bin/python3 -I -c '
import json, math, sys
from decimal import Decimal
d = json.load(sys.stdin, parse_float=Decimal); assert isinstance(d, list) and d
a = sys.argv[1]
t = sum((Decimal(str(r["cost"])) for r in d if a and r.get("object_id") == a), Decimal(0)); assert t.is_finite()
print(len(d), math.ceil(t * 1000000))' "$1"
}
capcheck() { [[ $1 == <-> ]] || return 1; (( $1 > CAP * 1000000 )) && echo over || echo under; }  # $1 = micro-dollars; integers only
state() {  # $1 = app id -> its state from app list ("absent" if not listed); non-zero on a failed read
  local out
  out=$(bounded 120 $M app list --json 2>/dev/null) || return 1
  print -r -- "$out" | /usr/bin/python3 -I -c 'import json,sys; d=json.load(sys.stdin); m=[x["state"] for x in d if x["app_id"] == sys.argv[1]]; print(m[0] if m else "absent")' "$1"
}
stop_confirmed() {
  for i in 1 2 3; do
    bounded 120 $M app stop --yes $1 </dev/null >/dev/null 2>&1
    [ "$(state $1)" = "stopped" ] && return 0
    sleep ${RUNB_STOP_SLEEP_S:-20}
  done
  return 1
}
echo "effective: modal $M; time limit ${LIMIT} s; poll ${POLL} s; cost read every ${BILLEVERY} s; cap \$$CAP"
b0=$(bill "") || { echo "STOP: billing read failed ($S..$E)"; exit 2; }
echo "billing before: rows / micro-dollars $b0"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: dirty tree"; exit 3; }
DIR=experiments/2026-10-16_scorer_consistency; PIN=70e38e9a509909181aad7a4af04d3f4a514248a10c15ec02869f0b1d8d15c76d
[ "$(shasum -a 256 $DIR/run_scorer_consistency.py | cut -d' ' -f1)" = "$PIN" ] || { echo "STOP: the script is not the pinned ${PIN:0:8}..."; exit 3; }
REV=${1:-}
[[ $REV =~ '^[0-9a-f]{40}$' ]] || { echo "STOP: usage: run_b.sh <reviewed code commit, full sha>"; exit 3; }
CODE=$(git log -1 --format=%H -- $DIR/run_b.sh $DIR/test_run_b.sh $DIR/run_scorer_consistency.py) || { echo "STOP: git log failed"; exit 3; }
[ "$CODE" = "$REV" ] || { echo "STOP: the last commit touching run_b.sh / test_run_b.sh / the script is ${CODE:0:9}, not the reviewed ${REV:0:9}"; exit 3; }
ch=$(git diff --no-renames --name-only $REV HEAD) || { echo "STOP: git diff vs ${REV:0:9} failed"; exit 3; }  # a rename lists both paths
[ -z "$(print -r -- "$ch" | grep -v -E -x 'SPEND\.md|(.*/)?NOTE_[^/]*\.md|(.*/)?results/.*')" ] \
  || { echo "STOP: files beyond SPEND.md / NOTE_*.md / results records changed since the reviewed ${REV:0:9}"; exit 3; }
echo "reviewed code commit $REV; head $(git rev-parse HEAD); diff --stat vs it:"; git diff --stat $REV HEAD
echo "diff --stat vs 7b700d3c3 (for the record):"; git diff --stat 7b700d3c3 HEAD
STAMP=$(date +%Y-%m-%d_%H%M%S); LOG=${RUNB_LOG_DIR:-$REPO/$DIR/results}/run_b_$STAMP.log
mkdir -p ${LOG:h}
APP=""; CL=""; RC=""; ARMED=1
appid() { [ -n "$APP" ] || APP=$(grep -o 'ap-[A-Za-z0-9]\{22\}' $LOG 2>/dev/null | head -1); }
reap() {  # stop the client (TERM, then KILL after 10 s) and reap it; nothing to do before it was started
  [ -n "$CL" ] || return 0
  kill $CL 2>/dev/null
  for i in {1..10}; do kill -0 $CL 2>/dev/null || break; sleep 1; done
  kill -9 $CL 2>/dev/null; wait $CL 2>/dev/null
}
halt() {  # $1 = message, $2 = exit code, $3 = exit code if no app id (default 10): reap the client, reread the id, stop the owned app
  trap '' INT TERM HUP                  # with a POSITIVE 'stopped' confirmation (else exit 5); signals are ignored BEFORE disarming, to the end
  seam halt                             # test only: between ignoring and disarming
  ARMED=0
  echo "$1"; appid
  for ((i = 0; i < IDWAIT; i++)); do [ -n "$APP" ] || [ -z "$CL" ] || ! kill -0 $CL 2>/dev/null && break; sleep 1; appid; done  # a live client may still print the id
  reap; appid
  [ -n "$APP" ] || { echo "STOP: no app id in $LOG; check modal app list and stop it by hand"; exit ${3:-10}; }
  stop_confirmed $APP || { echo "STOP NOT CONFIRMED: $APP is not in state 'stopped' after 3 tries; stop it by hand"; exit 5; }
  echo "stopped $APP (state 'stopped' confirmed)"; exit $2
}
onexit() { [ $ARMED = 1 ] || return 0; halt "STOP: the runner got a signal or exited early; stopping the app" 11; }
seam() {  # test mode only (RUNB_SEAM): ack the point $1, then HOLD until the test has delivered its signal (a two-way barrier, <= 20 s);
  [ "${RUNB_SEAM:-}" = "$1" ] || return 0  # a barrier that expires leaves $1.timeout, which fails the test
  : > $RUNB_ACK_DIR/$1
  for ((k = 0; k < 200; k++)); do [ -f $RUNB_ACK_DIR/$1.sent ] && return 0; sleep 0.1; done
  : > $RUNB_ACK_DIR/$1.timeout
}
hold_signals() { trap 'PEND=1' INT TERM HUP; seam prelaunch; }  # the hold trap and its seam are one unit: PEND must be cleared BEFORE it
trap onexit EXIT
PEND=""; hold_signals  # a signal from here to recording CL is held, then handled (before the launch: nothing is launched)
[ -z "$PEND" ] || { ARMED=0; echo "STOP: a signal arrived before the launch; nothing was launched"; exit 130; }
$M run --detach $DIR/run_scorer_consistency.py > $LOG 2>&1 &
CL=$!
trap 'onexit; exit 130' INT TERM HUP
[ -z "$PEND" ] || { onexit; exit 130; }
T0=$(date +%s); LASTB=$T0; BFAIL=0
while true; do
  appid
  if [ -z "$RC" ] && ! kill -0 $CL 2>/dev/null; then wait $CL; RC=$?; appid; fi  # reread the id once the client has exited
  NOW=$(date +%s)
  if [ -n "$RC" ]; then  # the client has exited: done once the owned app is 'stopped'
    [ -n "$APP" ] || halt "STOP: the client exited (rc $RC) and no app id is in its log" 10
    st=$(state $APP) || st="unknown"
    [ "$st" = "stopped" ] && break
  fi
  if [ -n "$APP" ] && (( NOW - LASTB >= BILLEVERY )); then  # the cost cap (hourly billing rows lag; the time limit is the hard bound)
    LASTB=$NOW
    bc_=$(bill "$APP") || bc_=""
    v_=$(capcheck "${bc_#* }") || v_=""
    if [ -z "$v_" ]; then  # a failed or non-numeric cost read: loud, and two in a row stop the app
      BFAIL=$((BFAIL + 1)); echo "WARN: cost-cap billing read failed ($BFAIL in a row)"
      [ $BFAIL -ge 2 ] && halt "STOP: two failed cost-cap reads; stopping $APP" 9
    else
      BFAIL=0
    fi
    [ "$v_" = "over" ] && halt "STOP: $APP billed ${bc_#* } micro-dollars > cap \$$CAP" 8
  fi
  (( NOW - T0 > LIMIT )) && halt "time limit (${LIMIT} s) passed: stopping ${APP:-the app}" 6 4
  sleep $POLL & wait $!  # an interruptible wait, so a signal is handled at once
done
ARMED=0
grep -v '^[│├└]' $LOG | grep -E '"record"|"gate_pass"|"controls"|STOP|FAIL|Error|Traceback|ap-' | cut -c1-3000 | head -40
echo "exit $RC; app $APP; log $LOG"
sleep $POSTSLEEP
b1=$(bill "$APP") || { echo "STOP: billing read after the run failed"; exit 7; }
echo "billing after (rows / app micro-dollars so far; hourly rows lag): $b1"
v_=$(capcheck "${b1#* }") || { echo "STOP: the billing read after the run gave no integer cost"; exit 7; }
[ "$v_" = "over" ] && { echo "STOP: $APP billed ${b1#* } micro-dollars > cap \$$CAP (read after the run)"; exit 8; }
exit $RC
