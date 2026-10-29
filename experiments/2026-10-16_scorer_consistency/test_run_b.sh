#!/bin/zsh
# Local shell-logic test of run_b.sh (no Modal, no network): a fake `modal` binary driven by files in a temp dir; every log goes to that
# temp dir (RUNB_LOG_DIR), never results/. Each case checks the exit code AND the fake app's final state (none = never launched):
# (T1) pipefail preflight -> 2; (T2) a failed post-run billing read -> 7; (T3) the limit passes and the stop never takes -> STOP NOT
# CONFIRMED 5, app still running; (T4) the client dies at once while the app runs on -> stopped at the limit, 6; (T5) billed cost over the $5
# cap -> 8, stopped; (T6) two failed cost-cap reads in a row -> 9, stopped; (T7) no app id by the limit -> 4; (T8) RUNB_* set outside test
# mode (RUNB_TEST unset, the fake supplied so a regression could never reach the real binary) -> 3 with the outside-test diagnostic; (T9) the
# read after the run shows the cost over the cap -> 8; (T10) the client exits and no app id is in the log -> 10; (T11) SIGTERM to the runner
# mid-run -> 11, stopped; (T12) test mode with a bare-name RUNB_MODAL (PATH holds only the fake) -> 3 with the test-mode diagnostic; (T13)
# a final cost of $5.00004 (just over the cap, unrounded) -> 8; (T14) a reviewed-commit argument that is not the last code commit -> 3.
# (T15) a second SIGTERM while halt is stopping the app (the stop takes 3 s) -> ignored; 11, stopped; (T16) the final cost as a JSON NUMBER
# 5.0000000000000001 (a float would read 5.0) -> 8; (T17) SIGTERM after halt ignores signals, before it disarms -> ignored, the limit stop
# completes: 6, stopped; (T18) SIGTERM right after the hold trap, before the launch -> held; nothing launched: 130. T17 / T18 use a two-way
# barrier (RUNB_SEAM, test mode only): the runner acks the point and waits there until the test has sent its signal; a barrier that expires
# (~20 s) leaves a .timeout file and T17 / T18 FAIL; (T19) a harness that never sends sees that expiry reported.
# Every run passes the last commit touching run_b.sh / test_run_b.sh / the script as the reviewed commit (the launch argument).
set -u
D=${0:A:h}; T=$(mktemp -d); trap 'rm -rf $T' EXIT
mkdir -p $T/logs
REV=$(git -C $D log -1 --format=%H -- run_b.sh test_run_b.sh run_scorer_consistency.py)
cat > $T/modal <<'FAKE'
#!/bin/zsh
# run_b.sh test fake: not modal
S=$FAKE_DIR
case "$1 $2" in
  "billing report")
    n=$(( $(cat $S/bill_n 2>/dev/null || echo 0) + 1 )); echo $n > $S/bill_n
    q='"'; [ -f $S/cost_numeric ] && q=""  # the cost as a JSON string, or as a JSON number (exercises parse_float=Decimal)
    echo "[{\"object_id\": \"ap-AAAAAAAAAAAAAAAAAAAAAA\", \"cost\": $q$(cat $S/cost 2>/dev/null || echo 0.01)$q}]"
    [ -f $S/bill_fail_from ] && [ $n -ge $(cat $S/bill_fail_from) ] && exit 1
    exit 0;;
  "run --detach")
    [ -f $S/no_app_id ] || echo "Running app ap-AAAAAAAAAAAAAAAAAAAAAA"; echo "running" > $S/state
    sleep $(cat $S/client_s); [ -f $S/finish_app ] && echo "stopped" > $S/state; exit 0;;
  "app list")
    echo "[{\"app_id\": \"ap-AAAAAAAAAAAAAAAAAAAAAA\", \"state\": \"$(cat $S/state)\", \"description\": \"x\", \"created_at\": \"\", \"stopped_at\": null, \"tasks\": 0}]";;
  "app stop")
    [ -f $S/stop_delay ] && sleep $(cat $S/stop_delay)
    [ -f $S/stop_works ] && echo "stopped" > $S/state && exit 0; exit 1;;
esac
FAKE
chmod +x $T/modal
RENV=(RUNB_TEST=fake-modal-only RUNB_MODAL=$T/modal RUNB_LOG_DIR=$T/logs FAKE_DIR=$T RUNB_POLL_S=1 RUNB_POST_SLEEP_S=0 RUNB_STOP_SLEEP_S=1 RUNB_ID_WAIT_S=2)
fin() { echo "$1/$(cat $T/state 2>/dev/null || echo none)"; }  # exit code / the fake app's final state
run() {  # $1 = case; LIM / BE set per case
  env $RENV RUNB_LIMIT_S=${LIM:-60} RUNB_BILL_EVERY_S=${BE:-999} zsh $D/run_b.sh ${ARG:-$REV} > $T/out_$1 2>&1
  fin $?
}
reset() { rm -f $T/bill_n $T/bill_fail_from $T/stop_works $T/finish_app $T/state $T/cost $T/no_app_id $T/stop_delay $T/cost_numeric; }
reset; echo 1 > $T/bill_fail_from; echo 1 > $T/client_s; r1=$(run T1)
reset; echo 2 > $T/bill_fail_from; echo 1 > $T/client_s; touch $T/finish_app; r2=$(run T2)
reset; echo 30 > $T/client_s; r3=$(LIM=3 run T3)
reset; echo 0 > $T/client_s; touch $T/stop_works; r4=$(LIM=4 run T4)
reset; echo 30 > $T/client_s; touch $T/stop_works; echo 6.0 > $T/cost; r5=$(BE=0 run T5)
reset; echo 30 > $T/client_s; touch $T/stop_works; echo 2 > $T/bill_fail_from; r6=$(BE=0 run T6)
reset; echo 30 > $T/client_s; touch $T/no_app_id; r7=$(LIM=3 run T7)
reset; env -u RUNB_TEST RUNB_MODAL=$T/modal FAKE_DIR=$T RUNB_LOG_DIR=$T/logs RUNB_LIMIT_S=5 zsh $D/run_b.sh $REV > $T/out_T8 2>&1; r8=$(fin $?)
grep -q 'RUNB_\* overrides are set outside the test' $T/out_T8 || r8="$r8-wrong-guard"
reset; echo 1 > $T/client_s; touch $T/finish_app; echo 6.0 > $T/cost; r9=$(run T9)
reset; echo 0 > $T/client_s; touch $T/no_app_id; r10=$(run T10)
reset; echo 30 > $T/client_s; touch $T/stop_works
env $RENV RUNB_LIMIT_S=60 RUNB_BILL_EVERY_S=999 zsh $D/run_b.sh $REV > $T/out_T11 2>&1 &
P=$!
for i in {1..20}; do [ -f $T/state ] && break; sleep 1; done; sleep 2  # the app is up and the trap is set
kill -TERM $P; wait $P; r11=$(fin $?)
reset; env RUNB_TEST=fake-modal-only RUNB_MODAL=modal PATH=$T:/usr/bin:/bin FAKE_DIR=$T RUNB_LOG_DIR=$T/logs RUNB_LIMIT_S=5 \
  /bin/zsh $D/run_b.sh $REV > $T/out_T12 2>&1; r12=$(fin $?)
grep -q 'test mode needs RUNB_MODAL = the absolute path' $T/out_T12 || r12="$r12-wrong-guard"
reset; echo 1 > $T/client_s; touch $T/finish_app; echo 5.00004 > $T/cost; r13=$(run T13)
reset; echo 1 > $T/client_s; r14=$(ARG=$(printf '0%.0s' {1..40}) run T14)
grep -q 'not the reviewed' $T/out_T14 || r14="$r14-wrong-guard"
reset; echo 30 > $T/client_s; touch $T/stop_works; echo 3 > $T/stop_delay  # T15: a second SIGTERM while halt is stopping the app
env $RENV RUNB_LIMIT_S=60 RUNB_BILL_EVERY_S=999 zsh $D/run_b.sh $REV > $T/out_T15 2>&1 &
P=$!
for i in {1..20}; do [ -f $T/state ] && break; sleep 1; done; sleep 2
kill -TERM $P; sleep 1; kill -TERM $P 2>/dev/null; wait $P; r15=$(fin $?)
reset; echo 1 > $T/client_s; touch $T/finish_app $T/cost_numeric; echo 5.0000000000000001 > $T/cost; r16=$(run T16)
reset; echo 30 > $T/client_s; touch $T/stop_works  # T17: SIGTERM in the gap where halt has started but not yet disarmed (limit stop)
rm -f $T/halt $T/halt.sent $T/halt.timeout $T/prelaunch $T/prelaunch.sent $T/prelaunch.timeout
env $RENV RUNB_LIMIT_S=3 RUNB_BILL_EVERY_S=999 RUNB_SEAM=halt RUNB_ACK_DIR=$T zsh $D/run_b.sh $REV > $T/out_T17 2>&1 &
P=$!
for i in {1..60}; do [ -f $T/halt ] && break; sleep 0.5; done  # the runner is HELD inside halt's window (ignore set, not yet disarmed)
[ -f $T/halt ] && kill -TERM $P && k17=ok || k17=missed; : > $T/halt.sent; wait $P; r17=$(fin $?); [ $k17 = ok ] || r17="$r17-boundary-$k17"
[ ! -f $T/halt.timeout ] || r17="$r17-barrier-expired"  # the signal may have landed after the boundary: fail
reset; echo 30 > $T/client_s; touch $T/stop_works  # T18: SIGTERM after the hold trap is set, before the launch
rm -f $T/halt $T/halt.sent $T/halt.timeout $T/prelaunch $T/prelaunch.sent $T/prelaunch.timeout
env $RENV RUNB_LIMIT_S=60 RUNB_BILL_EVERY_S=999 RUNB_SEAM=prelaunch RUNB_ACK_DIR=$T zsh $D/run_b.sh $REV > $T/out_T18 2>&1 &
P=$!
for i in {1..60}; do [ -f $T/prelaunch ] && break; sleep 0.5; done  # the runner is HELD right after the hold trap, nothing launched yet
[ -f $T/prelaunch ] && kill -TERM $P && k18=ok || k18=missed; : > $T/prelaunch.sent; wait $P; r18=$(fin $?); [ $k18 = ok ] || r18="$r18-boundary-$k18"
[ ! -f $T/prelaunch.timeout ] || r18="$r18-barrier-expired"
reset; echo 30 > $T/client_s; touch $T/stop_works  # T19: the harness never sends -> the barrier expires and says so (detection check)
rm -f $T/halt $T/halt.sent $T/halt.timeout
env $RENV RUNB_LIMIT_S=3 RUNB_BILL_EVERY_S=999 RUNB_SEAM=halt RUNB_ACK_DIR=$T zsh $D/run_b.sh $REV > $T/out_T19 2>&1 &
P=$!
for i in {1..60}; do [ -f $T/halt ] && break; sleep 0.5; done
wait $P; fin $? > /dev/null; [ -f $T/halt.timeout ] && r19=expiry-detected || r19=expiry-missed
fails=0
for c in "T1 pipefail preflight:$r1:2/none" "T2 failed post-run billing read:$r2:7/stopped" "T3 unconfirmed stop:$r3:5/running" \
         "T4 client dies, app runs on:$r4:6/stopped" "T5 cost over the cap:$r5:8/stopped" "T6 two failed cost reads:$r6:9/stopped" \
         "T7 no app id by the limit:$r7:4/running" "T8 RUNB_* outside test mode:$r8:3/none" "T9 final read over the cap:$r9:8/stopped" \
         "T10 client exits with no app id:$r10:10/running" "T11 SIGTERM to the runner:$r11:11/stopped" "T12 bare-name RUNB_MODAL:$r12:3/none" \
         "T13 final read 5.00004 over the cap:$r13:8/stopped" "T14 wrong reviewed commit:$r14:3/none" \
         "T15 second SIGTERM during the stop:$r15:11/stopped" "T16 numeric JSON cost 5.0000000000000001:$r16:8/stopped" \
         "T17 SIGTERM as halt starts:$r17:6/stopped" "T18 SIGTERM before the launch:$r18:130/none" "T19 barrier expiry is reported:${r19}:expiry-detected"; do
  n=${c%%:*}; got=${${c#*:}%:*}; want=${c##*:}
  [ "$got" = "$want" ] && echo "$n: $got (want $want) PASS" || { echo "$n: $got (want $want) FAIL"; fails=$((fails + 1)); }
done
[ $fails = 0 ] && echo "ALL NINETEEN PASS" || { echo "$fails CASES FAILED"; exit 1; }
