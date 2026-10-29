#!/usr/bin/env bash
# Laptop: after each new census file (the loop's or a one-shot), run stall_tag.py on it; log every ALARM with its tag to
# nm_launch/stall_watch.log. On any UNEXPLAINED stall, write nm_launch/STALL_UNEXPLAINED.txt and exit 3 (the fork tells d9 at
# once). Until 04:40 PT 10-16 (loop15 ends 04:30). v4 (d9 ~22:00): also runs cap_rl_check.py and exits 4 with
# CAP_OR_RL_ALERT.txt if total sims > 95, any RL in the exact last 30 min, or an incomplete or unreadable census (exit 4 wins
# over 3; both alert files are written). Read-only: census files only.
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
RD=$SP/wt_nm/experiments/2026-10-14_q3_new_mains_launch/results; LOG=$SP/nm_launch/stall_watch.log
END=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M' '2026-10-16 04:40' +%s)
last=""; tf=""; tries=0
while (( $(date +%s) < END )); do
  f=$(ls -t $RD/sim_error_census_2026-10-1[56]_*.json 2>/dev/null | head -1)
  if [[ -n $f && $f != "$last" ]]; then
    [[ $f != "$tf" ]] && { tf=$f; tries=0; }  # retries count per file
    # Codex r84: both checks run on every new census before the exit status is chosen; a census that stays unreadable or
    # malformed for 3 tries (~90 s; one try may catch the file mid-write) is an alert, never a silent retry.
    out=$(python3 -I "$SP/nm_launch/stall_tag.py" "$f" 2>/dev/null); src=$?
    cap=$(python3 -I "$SP/nm_launch/cap_rl_check.py" "$f" 2>/dev/null); crc=$?
    if (( (src != 0 && src != 3) || (crc != 0 && crc != 4) )); then
      tries=$((tries + 1)); (( tries < 3 )) && { sleep 30; continue; }
      cap="$(basename "$f"): unreadable or malformed after 3 tries (stall_tag exit $src, cap check exit $crc)"; crc=4
    fi
    last=$f; tries=0
    { echo "$(TZ=America/Los_Angeles date '+%T') $(basename "$f")"; echo "$out" | sed 's/^/  /'; echo "  $cap"; } >> "$LOG"
    (( src == 3 )) && echo "$out" > "$SP/nm_launch/STALL_UNEXPLAINED.txt"
    (( crc == 4 )) && echo "$cap" > "$SP/nm_launch/CAP_OR_RL_ALERT.txt"
    (( crc == 4 )) && exit 4
    (( src == 3 )) && exit 3
  fi
  sleep 30
done
