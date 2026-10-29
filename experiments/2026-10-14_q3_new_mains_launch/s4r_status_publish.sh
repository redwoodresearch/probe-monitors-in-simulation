#!/usr/bin/env bash
# Laptop side, read-only on box 4 (0d chose d9's option (d), 2026-10-15 ~06:20 PT): publish the 4r hold's state for the
# ungrounded fork's watcher, which pauses ug near 4r's main end (so the census drops below 60 simulated runs and 4r's
# follow-up releases) and resumes ug once 4r's follow-up sims have started. Separate from the release watcher (pid 36099,
# s4r_census_hold.sh; neither it nor its box script is changed). Every 300 s until 09:25 PT:
#   - one JSON line appended to results/s4r_hold_status_2026-10-15.jsonl on aswin/q3-new-mains-launch, committed and
#     pushed under the census lock (never racing a census commit; a busy lock defers the commit to the next poll);
#   - the same line written atomically to the laptop file nm_launch/s4r_hold_status_latest.json.
# Fields: time; readable (false = box unreadable: every other field absent, never a default); 4r real runs open, not
# started, running; main session live (positive tmux read); hold marker present; released (census_hold_released_*.txt present); follow-up session live; follow-up
# simulated runs started / total (from its launch.json; null before it exists); harness PAUSE_SIMS; end_main's last line.
#   nohup bash s4r_status_publish.sh > ../../../nm_launch/s4r_status_publish.out 2>&1 &
set -uo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
HERE=$(cd "$(dirname "$0")" && pwd); W=$SP/wt_nm; REL=experiments/2026-10-14_q3_new_mains_launch/results/s4r_hold_status_2026-10-15.jsonl
OUT=$W/$REL; LATEST=$SP/nm_launch/s4r_hold_status_latest.json; LK=$SP/nm_launch/census.lock
MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r
END=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' '2026-10-15 09:25:00' +%s)
export DEVBOX=aswin@i-07bce857a818b6f69
held=0; trap '(( held )) && rmdir "$LK"' EXIT; trap 'exit 143' TERM INT HUP
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
perl -e 'alarm 120; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync -a "$HERE/checks/s4r_fu_started.py" box:phase2/nm/tmp/ < /dev/null || exit 1
while (( $(date +%s) < END )); do
  t=$(TZ=America/Los_Angeles date '+%F %T')
  s=$(box 120 "python3 /home/aswin/phase2/nm/checks/s4r_hold_state.py") || s=""
  # Codex r36 item 1: a failed listing makes the line unreadable (never "released: false")
  r=$(box 60 "L=\$(ls $MAIN/) || exit 3; printf '%s\n' \"\$L\" | grep -c '^census_hold_released_.*[.]txt\$' || true") || r=""
  f=$(box 60 "python3 /home/aswin/phase2/nm/tmp/s4r_fu_started.py") || f=""
  line=$(python3 -c '
import json, sys
t, s, r, f = sys.argv[1:5]
try:
    d, fu, n = json.loads(s), json.loads(f), int(r)
except Exception:
    print(json.dumps({"time_pt": t, "readable": False})); sys.exit()
present = fu["fu_launch_json"] == "present"
ses = fu["nm_sessions"]  # Codex r36 item 3: session state from a positive tmux read, not the hold-state helper
print(json.dumps({"time_pt": t, "readable": True, "open_real_4r_main": d["open_real"],
                  "main_real_not_started": fu["main_real_not_started"], "main_real_running": fu["main_real_running"],
                  "main_live": "snms4r" in ses, "hold_marker_present": d["hold"], "released": n > 0, "fu_live": "fnms4r" in ses,
                  "fu_sims_started": fu["fu_sims_started"] if present else None, "fu_sims_total": fu["fu_sims_total"] if present else None,
                  "harness_pause_sims": d["harness_pause_sims"], "end_main_last": d["em_last"]}))' "$t" "$s" "$r" "$f")
  echo "$line" >> "$OUT"
  echo "$line" > "$LATEST.tmp" && mv "$LATEST.tmp" "$LATEST"
  if mkdir "$LK" 2>/dev/null; then
    held=1  # Codex r36 item 2: the lock is freed on any exit while held; each git call is bounded
    (cd "$W" && perl -e 'alarm 30; exec @ARGV' git add "$REL" && perl -e 'alarm 60; exec @ARGV' git commit -q -m "4r hold status $t PT" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EaTTB7CWcfzNHhcqan35WH" > /dev/null 2>&1
     perl -e 'alarm 60; exec @ARGV' git -c url."https://github.com/".insteadOf="git@github.com:" push -q origin HEAD > /dev/null 2>&1)
    rmdir "$LK"; held=0
  fi
  sleep 300
done
