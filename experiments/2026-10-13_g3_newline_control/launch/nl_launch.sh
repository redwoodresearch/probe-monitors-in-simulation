#!/bin/bash
# Newline-control launch helper (q2 fork, 10-13), from G3's g3_launch.sh @ b2389750: starts `modal run <args>` from this
# worktree (no --detach), takes the app id from the client's log (the URL line), and within seconds starts the v2.5 billing
# watcher (stop $, deadline), watch_cost v2.5 --app-id (NL_COST_RATE $/h per container; --startup-polls 40 for long capacity
# waits) and guard v2 (cap) on that id, recording every pid. usage: nl_launch.sh <tag> <cap> <stop $> <deadline s> <modal run args...>
S=/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad
L=$S/nl_logs; W=$S/wt_nlctl/experiments/2026-10-13_g3_newline_control
tag=$1; cap=$2; stop=$3; dl=$4; shift 4
[ -z "$NL_COST_RATE" ] && { echo "set NL_COST_RATE ($/h per container: 5.52 for one H200, 0.6 for 8 CPUs / 32 GiB, 1.3 for 16 CPUs / 64 GiB)"; exit 2; }
cd $S/wt_nlctl || exit 1
PYTHONPATH=src nohup uv run --no-project --python 3.12 --with "modal==1.6.1" $NL_EXTRA_WITH modal run "$@" > $L/client_$tag.log 2>&1 &
CP=$!; echo $CP > $L/client_$tag.pid
until grep -q -E 'ap-[A-Za-z0-9]{22}' $L/client_$tag.log 2>/dev/null || ! kill -0 $CP 2>/dev/null; do sleep 1; done
APPID=$(grep -o -E 'ap-[A-Za-z0-9]{22}' $L/client_$tag.log | head -1)
[ -z "$APPID" ] && { echo "NO APP ID; client exited"; tail -5 $L/client_$tag.log; exit 2; }
echo $APPID > $L/$tag.app
cd $W || exit 1
nohup python3 watch_billing_strict.py $APPID $stop $(( $(date +%s) + dl )) > $L/watch_billing_$tag.log 2>&1 & echo $! > $L/watch_billing_$tag.pid
nohup uv run --no-project --python 3.12 --with 'modal==1.6.1' python watch_cost.py --app-id $APPID --stop $stop --rate $NL_COST_RATE --cap-min 150 --startup-polls 40 --log $L/watch_cost_$tag.log > $L/watch_cost_$tag.out 2>&1 & echo $! > $L/watch_cost_$tag.pid
nohup python3 guard_tasks_v2.py $APPID $cap > $L/guard_$tag.log 2>&1 & echo $! > $L/guard_$tag.pid
echo "$(date +%H:%M:%S) $tag rate $NL_COST_RATE app $APPID client $CP billing $(cat $L/watch_billing_$tag.pid) cost $(cat $L/watch_cost_$tag.pid) guard $(cat $L/guard_$tag.pid) stop \$$stop cap $cap"
