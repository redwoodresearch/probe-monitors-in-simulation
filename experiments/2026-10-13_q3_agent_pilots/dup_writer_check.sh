#!/usr/bin/env bash
# Duplicate-writer check for the agent pilots (0d's standing rule, 10-13: a container guard of N + max(4, 10% of N)
# plus a check that stops only when two containers actually serve traffic). Each serving container writes its own
# request log, requests_<stamp>_<function call id>.jsonl.gz (run_vllm_tunnel.py), with no app ID in the name;
# watch_and_stop's rescue (RESCUE=1) copies each watched container's log as requests_<stamp>_<function call id>_<app id>.jsonl.gz.
# Every 60 s, per watched app, this counts the DISTINCT function call IDs among the rescued files carrying that app ID.
# More than one means two containers served it: ALARM, then `modal app stop <app id>`. Needs watch_and_stop with
# RESCUE=1 (RESCUE_EVERY_S 120). Exits when every app is stopped.
#   usage: dup_writer_check.sh <app id>[,<app id>...] <log file>
set -uo pipefail
IFS=, read -ra APPS <<< "$1"; LOG=$2
VOL=probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06
M=(uv run --no-project --python 3.12 --with modal==1.6.1 modal)
bounded() { perl -e 'alarm shift; exec @ARGV' "$@"; }
say() { echo "$(TZ=America/Los_Angeles date '+%H:%M:%S PT') $*" >> "$LOG"; }
say "start: ${APPS[*]}"
while true; do
  files=$(bounded 90 "${M[@]}" volume ls "$VOL" / --json 2>/dev/null) || { say "volume ls failed; retry"; sleep 60; continue; }
  live=0
  for a in "${APPS[@]}"; do
    n=$(uv run --no-project --python 3.12 python -I -c "import json,re,sys; print(len({m.group(0) for r in json.loads(sys.argv[2]) if sys.argv[1] in r['filename'] for m in [re.search(r'fc-[A-Za-z0-9]+', r['filename'])] if m}))" "$a" "$files")
    st=$(bounded 60 "${M[@]}" app list --json 2>/dev/null | uv run --no-project --python 3.12 python -I -c "import json,sys; print(next((r['state'] for r in json.load(sys.stdin) if r.get('app_id')==sys.argv[1]),'gone'))" "$a")
    [[ "$st" =~ stopped|gone ]] || live=1
    if (( n > 1 )); then say "ALARM $a: $n distinct serving containers (function call IDs); stopping"; bounded 60 "${M[@]}" app stop "$a" --yes >/dev/null 2>&1; fi
    say "$a writers=$n state=$st"
  done
  (( live )) || { say "all apps stopped; exit"; exit 0; }
  sleep 60
done
