#!/bin/bash
# G3 launch helper (q2, 10-13): starts `modal run <args>` from the G3 worktree (no --detach), takes the app id from the
# client's own log, and within seconds starts the v2.5 billing watcher (stop $, deadline), watch_cost v2.5 --app-id and
# guard v2 (cap) on that id, recording every pid. usage: g3_launch.sh <tag> <cap> <stop $> <deadline s> <modal run args...>
S=/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad
L=$S/g3_resume_logs; W=$S/g3_watch25
tag=$1; cap=$2; stop=$3; dl=$4; shift 4
[ -z "$G3_COST_RATE" ] && { echo "set G3_COST_RATE ($/h per container for watch_cost: 5.52 for one H200, 0.6 for an 8-CPU 32 GiB container)"; exit 2; }  # 10-13: no default rate
cd $S/wt_g3 || exit 1
PYTHONPATH=src nohup uv run --no-project --python 3.12 --with "modal==1.6.1" $G3_EXTRA_WITH modal run "$@" > $L/client_$tag.log 2>&1 &
CP=$!; echo $CP > $L/client_$tag.pid
until grep -q -E 'ap-[A-Za-z0-9]{22}' $L/client_$tag.log 2>/dev/null || ! kill -0 $CP 2>/dev/null; do sleep 1; done
APPID=$(grep -o -E 'ap-[A-Za-z0-9]{22}' $L/client_$tag.log | head -1)
[ -z "$APPID" ] && { echo "NO APP ID; client exited"; tail -5 $L/client_$tag.log; exit 2; }
echo $APPID > $L/$tag.app
cd $W || exit 1
nohup python3 watch_billing_strict.py $APPID $stop $(( $(date +%s) + dl )) > $L/watch_billing_$tag.log 2>&1 & echo $! > $L/watch_billing_$tag.pid
nohup uv run --no-project --python 3.12 --with 'modal==1.6.1' python watch_cost.py --app-id $APPID --stop $stop --rate $G3_COST_RATE --cap-min 150 --log $L/watch_cost_$tag.log > $L/watch_cost_$tag.out 2>&1 & echo $! > $L/watch_cost_$tag.pid
nohup python3 guard_tasks_v2.py $APPID $cap > $L/guard_$tag.log 2>&1 & echo $! > $L/guard_$tag.pid
echo "$(date +%H:%M:%S) $tag rate $G3_COST_RATE app $APPID client $CP billing $(cat $L/watch_billing_$tag.pid) cost $(cat $L/watch_cost_$tag.pid) guard $(cat $L/guard_$tag.pid) stop \$$stop cap $cap"
