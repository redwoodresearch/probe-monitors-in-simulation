#!/usr/bin/env bash
# Laptop: run the census on all boxes, write results/sim_error_census_2026-10-15_<HHMM>.json, commit and push it; prints
# the rates and the commit. Read-only on the boxes.
set -uo pipefail
E=$(cd "$(dirname "$0")" && pwd); W=$E/../wt_nm
# one census at a time (the 30-min cadence and s4r_census_hold.sh release share census/ and rates_now.txt)
n=0; until mkdir "$E/census.lock" 2>/dev/null; do n=$((n + 1)); (( n > 60 )) && { echo "census lock busy 10 min"; exit 1; }; sleep 10; done
trap 'rmdir "$E/census.lock"' EXIT; trap 'exit 143' TERM  # SIGTERM (the 4r watcher's 600 s bound) still frees the lock
# census_loop3 (Codex r41): skip when a census newer than the one the caller planned from was committed meanwhile
# (the newest committed, clean census file)
if [ -n "${SKIP_IF_NEWER_THAN:-}" ]; then
  nt=$(cd "$W" && for f in $(ls -r experiments/2026-10-14_q3_new_mains_launch/results/sim_error_census_2026-10-15_*.json 2>/dev/null); do git ls-files --error-unmatch "$f" > /dev/null 2>&1 && [ -z "$(git status --porcelain -- "$f")" ] && { echo "$f"; break; }; done | xargs -I{} python3 -c 'import json,sys; from datetime import datetime; from zoneinfo import ZoneInfo; print(int(datetime.strptime(json.load(open(sys.argv[1]))["census_time_pt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()))' {} 2>/dev/null)
  [ -n "$nt" ] && (( nt > SKIP_IF_NEWER_THAN )) && { echo "skipped: a newer census ($nt) was committed while waiting"; exit 0; }
fi
[ -n "${NOT_AFTER:-}" ] && (( $(date +%s) >= NOT_AFTER )) && { echo "skipped: past the caller's end"; exit 0; }  # Codex r42-r43: right before collection
bash "$E/census_all.sh" > /dev/null 2>&1
T=$(date '+%F %T'); TS=$(date '+%H%M')
python3 "$E/census_rates.py" > "$E/rates_now.txt" || exit 1
cd "$E/.." && python3 - "$T" "$TS" <<'EOF' || exit 1
import json, glob, os, sys
out = {"census_time_pt": sys.argv[1], "rates_text": open("nm_launch/rates_now.txt").read(),
       "per_box": {os.path.basename(f)[:-5]: (json.load(open(f)) if open(f).read().strip() else None) for f in sorted(glob.glob("nm_launch/census/*.json")) if not f.endswith("summary.json")}}
open(f"wt_nm/experiments/2026-10-14_q3_new_mains_launch/results/sim_error_census_2026-10-15_{sys.argv[2]}.json", "w").write(json.dumps(out, indent=1))
EOF
A() { perl -e 'alarm shift; exec @ARGV' "$@"; }  # d9 10-15 10:10: GitHub ops are hanging; every git network/commit call is bounded
cd "$W" && A 30 git add "experiments/2026-10-14_q3_new_mains_launch/results/sim_error_census_2026-10-15_$TS.json" && \
  A 60 git commit -q -m "Rate-limit census $T PT

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EaTTB7CWcfzNHhcqan35WH" > /dev/null 2>&1
for i in 1 2 3; do A 90 git -c url."https://github.com/".insteadOf="git@github.com:" push -q origin HEAD > /dev/null 2>&1
  [ "$(A 30 git ls-remote origin refs/heads/aswin/q3-new-mains-launch | cut -c1-9)" = "$(git rev-parse --short=9 HEAD)" ] && break; sleep 5; done
echo "census $T committed @ $(git rev-parse --short=9 HEAD) (remote $(A 30 git ls-remote origin refs/heads/aswin/q3-new-mains-launch | cut -c1-9)): results/sim_error_census_2026-10-15_$TS.json"
grep "last 30 min" "$E/rates_now.txt"
