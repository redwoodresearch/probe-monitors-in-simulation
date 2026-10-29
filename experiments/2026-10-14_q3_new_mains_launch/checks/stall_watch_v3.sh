#!/usr/bin/env bash
# Laptop: after each new census file (the loop's or a one-shot), run stall_tag.py on it; log every ALARM with its tag to
# nm_launch/stall_watch.log. On any UNEXPLAINED stall, write nm_launch/STALL_UNEXPLAINED.txt and exit 3 (the fork tells d9 at
# once). Until 21:40 PT (loop11 ends 21:30; v3 of stall_watch.sh). Read-only: census files only.
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
RD=$SP/wt_nm/experiments/2026-10-14_q3_new_mains_launch/results; LOG=$SP/nm_launch/stall_watch.log
END=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M' '2026-10-15 21:40' +%s)
last=""
while (( $(date +%s) < END )); do
  f=$(ls -t $RD/sim_error_census_2026-10-15_*.json 2>/dev/null | head -1)
  if [[ -n $f && $f != "$last" ]] && python3 -I -c "import json,sys; json.load(open(sys.argv[1]))" "$f" 2>/dev/null; then
    last=$f; out=$(python3 -I "$SP/nm_launch/stall_tag.py" "$f"); rc=$?
    { echo "$(TZ=America/Los_Angeles date '+%T') $(basename "$f")"; echo "$out" | sed 's/^/  /'; } >> "$LOG"
    (( rc == 3 )) && { echo "$out" > "$SP/nm_launch/STALL_UNEXPLAINED.txt"; exit 3; }
  fi
  sleep 30
done
