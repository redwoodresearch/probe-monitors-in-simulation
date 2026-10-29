#!/usr/bin/env bash
# ug_wait.sh (RULE.md; d9's conditional GO, step 1): FROM THE LAPTOP, poll q3-xbox3 with ug_ready_check.sh until it
# reports READY (exit 0) or 06:00 PT (exit 2). READY counts only if the ssh call itself succeeded (remote exit 0), its
# last line starts with READY:, and the deadline has not passed when it is read. Every poll and sleep is bounded by the
# time left; polls are >= 150 s apart; a failed sleep aborts (exit 3) rather than polling early. Each poll runs in its
# own process group, writing to a file (not a pipe): on timeout the WHOLE group is killed (ssh and its SSM proxy), and
# the group is killed after a normal exit too, so nothing can hold the poll open past its bound. Read-only on the box.
#   DEVBOX_SH=<devbox.sh> bash ug_wait.sh
set -uo pipefail
D=${DEVBOX_SH:?set DEVBOX_SH}; BOX=aswin@i-0f2299f29dd296259
OUTF=$(mktemp -t ug_wait) || exit 3
deadline=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' "$(TZ=America/Los_Angeles date +%Y-%m-%d) 06:00:00" +%s) || exit 3
bounded() {  # bounded <seconds> <command...>: the command's exit status (128+N if killed by signal N), or 124 on
             # timeout; output in $OUTF
  local t=$1; shift
  perl -e '
    my $t = shift @ARGV;
    my $pid = fork(); die "fork failed\n" unless defined $pid;
    if ($pid == 0) { setpgrp(0, 0); exec @ARGV or exit 127; }
    local $SIG{ALRM} = sub { kill "KILL", -$pid; waitpid($pid, 0); exit 124; };
    alarm $t; waitpid($pid, 0); my $st = $?; alarm 0; kill "KILL", -$pid; exit(($st & 127) ? 128 + ($st & 127) : ($st >> 8));
  ' "$t" "$@" > "$OUTF" 2>&1
}
while :; do
  left=$(( deadline - $(date +%s) ))
  (( left > 5 )) || { echo "deadline 06:00 PT passed"; exit 2; }
  t=$(( left < 120 ? left : 120 ))
  DEVBOX=$BOX bounded "$t" bash "$D" 'bash /home/aswin/phase2/ug_stage/ug_ready_check.sh'; rc=$?
  last=$(tail -1 "$OUTF")
  echo "$(TZ=America/Los_Angeles date '+%H:%M:%S') rc=$rc $last"
  if (( rc == 0 )) && [[ $last == READY:* ]]; then
    (( $(date +%s) < deadline )) && exit 0
    echo "READY arrived after the deadline: ignored"; exit 2
  fi
  left=$(( deadline - $(date +%s) ))
  (( left > 5 )) || { echo "deadline 06:00 PT passed"; exit 2; }
  sleep $(( left < 150 ? left : 150 )) || { echo "sleep failed: aborting"; exit 3; }
done
