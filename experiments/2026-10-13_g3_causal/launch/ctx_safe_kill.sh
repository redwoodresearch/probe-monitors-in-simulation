#!/bin/bash
# Kill a launch's watchers ONLY when (1) its client pid has exited AND (2) `modal app list` shows its app stopped.
# Both conditions are logged before any kill. usage: ctx_safe_kill.sh <tag> [watcher-pid-file suffix, default <tag>]
S=/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad; L=$S/g3ctx_logs
tag=$1; suf=${2:-$1}; APPID=$(cat $L/$tag.app)
if kill -0 $(cat $L/client_$tag.pid) 2>/dev/null; then echo "$(date +%T) $tag: client $(cat $L/client_$tag.pid) still running; NOT killing watchers" | tee -a $L/safe_kill.log; exit 1; fi
STATE=$(cd /tmp && uv run --no-project --with 'modal==1.6.1' modal app list --json 2>/dev/null | python3 -c "import json,sys; print(next((r['state'] for r in json.load(sys.stdin) if r['app_id']=='$APPID'), 'not listed'))")
echo "$(date +%T) $tag: client exited; app $APPID state=$STATE" | tee -a $L/safe_kill.log
case "$STATE" in stopped|"not listed") ;; *) echo "$(date +%T) $tag: app not stopped; NOT killing watchers" | tee -a $L/safe_kill.log; exit 1;; esac
for p in watch_billing watch_cost guard; do f=$L/${p}_$suf.pid; [ -f $f ] && kill $(cat $f) 2>/dev/null && echo "$(date +%T) killed $p pid $(cat $f)" | tee -a $L/safe_kill.log; done
