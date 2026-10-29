#!/usr/bin/env bash
# Negative tests of modal_day_guard.py's fail-closed path (RULE.md, Stops), laptop, $0: a Modal CLI that always fails
# (`false`), polls every 1 s, one fake box (no devbox, no apps). Expected: three unreadable polls, status STOP, a stop
# attempted on the box and retried (never confirmed, so the guard keeps owning it), and no exit while unconfirmed.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d)
echo '[{"name":"fake","devbox":"nobody@i-00000000000000000","session":"s311bX","app_ids_file":"'"$T"'/none.txt","harness":"'"$T"'"}]' > "$T/boxes.json"
TEST_MODAL_CMD=false TEST_POLL_S=1 perl -e 'alarm 12; exec @ARGV' python3 -I "$HERE/modal_day_guard.py" "$T/boxes.json" "$T/status" > "$T/log" 2>&1
rc=$?
pass=1
[[ $rc == 142 ]] || { echo "FAIL: the guard exited ($rc) while its box was unconfirmed"; pass=0; }
[[ $(cut -d' ' -f1 "$T/status") == STOP ]] || { echo "FAIL: status is not STOP"; pass=0; }
grep -q "no Modal reading 3 polls in a row (fail closed)" "$T/log" || { echo "FAIL: no fail-closed line"; pass=0; }
(( $(grep -c "fake: stop" "$T/log") >= 2 )) || { echo "FAIL: the stop was not retried"; pass=0; }
(( pass )) && echo "PASS: unreadable billing -> STOP, stop attempted and retried every poll, guard still owns the box"
tail -4 "$T/log"; rm -rf "$T"
