#!/usr/bin/env bash
# Laptop: run the census on all boxes (census_all_v2.sh), write results/sim_error_census_2026-10-15_<HHMM>.json, commit and push it;
# prints the rates and the commit. Read-only on the boxes. v4 (10-15 ~11:12; a NEW file): v3 with q2_replay_v4.py (the replay record has
# "rows" and "rate_limit_first_last" like every box launch record). v3 (10-15 ~10:30): v2 + df/q2's Modal replay as source
# "q2_replay" (q2_replay_v4.py: census/q2_replay.json is removed first and rewritten every census; a reader failure or timeout
# writes a read error; "not started" is complete), counted by the same rates code. v2 (10-15 ~10:20):
#  - 0d's powered-off rule (powered_off_v2.py): a box with no data this round is recorded under "powered_off_boxes" only if a fresh
#    EC2 read says stopped/terminated AND every launch in its last-seen census was finished; "powered_off_boxes" is always present;
#  - Codex r47: add/commit failures exit 1 (no push, no "committed" line); the push is verified against ls-remote and the verified
#    hash is printed; an unverified push prints "committed locally ... push NOT verified" and exits 2; every git call is bounded.
set -uo pipefail
E=$(cd "$(dirname "$0")" && pwd); W=$E/../wt_nm; RD=experiments/2026-10-14_q3_new_mains_launch/results
A() { perl -e 'alarm shift; exec @ARGV' "$@"; }
n=0; until mkdir "$E/census.lock" 2>/dev/null; do n=$((n + 1)); (( n > 60 )) && { echo "census lock busy 10 min"; exit 1; }; sleep 10; done
trap 'rmdir "$E/census.lock"' EXIT; trap 'exit 143' TERM
if [ -n "${SKIP_IF_NEWER_THAN:-}" ]; then
  nt=$(cd "$W" && for f in $(ls -r $RD/sim_error_census_2026-10-15_*.json 2>/dev/null); do A 30 git ls-files --error-unmatch "$f" > /dev/null 2>&1 && st=$(A 30 git status --porcelain -- "$f") && [ -z "$st" ] && { echo "$f"; break; }; done | xargs -I{} python3 -c 'import json,sys; from datetime import datetime; from zoneinfo import ZoneInfo; print(int(datetime.strptime(json.load(open(sys.argv[1]))["census_time_pt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()))' {} 2>/dev/null)
  [ -n "$nt" ] && (( nt > SKIP_IF_NEWER_THAN )) && { echo "skipped: a newer census ($nt) was committed while waiting"; exit 0; }
fi
[ -n "${NOT_AFTER:-}" ] && (( $(date +%s) >= NOT_AFTER )) && { echo "skipped: past the caller's end"; exit 0; }
bash "$E/census_all_v2.sh" > /dev/null 2>&1
QF="$E/census/q2_replay.json"  # Codex r51: never keep an earlier census's replay status
rm -f "$QF"; [ -e "$QF" ] && { echo "census: cannot remove $QF"; exit 1; }
A 90 python3 "$W/experiments/2026-10-14_q3_new_mains_launch/checks/q2_replay_v4.py" "$E/census" > /dev/null 2>&1 && [ -s "$QF" ] ||
  { echo '{"q2_replay": {"error": "replay reader failed or timed out"}}' > "$QF" && [ -s "$QF" ] || { echo "census: cannot write $QF"; exit 1; }; }
T=$(date '+%F %T'); TS=$(date '+%H%M')
python3 "$E/census_rates.py" > "$E/rates_now.txt" || exit 1
A 600 python3 "$W/experiments/2026-10-14_q3_new_mains_launch/checks/powered_off_v2.py" "$E/census" "$W" "$RD" > "$E/powered_off_now.json" 2> "$E/powered_off_now.err" || echo '{}' > "$E/powered_off_now.json"
cd "$E/.." && python3 - "$T" "$TS" <<'EOF' || exit 1
import json, glob, os, sys
try:
    po = json.load(open("nm_launch/powered_off_now.json"))
    assert isinstance(po, dict)
except Exception:
    po = {}
out = {"census_time_pt": sys.argv[1], "rates_text": open("nm_launch/rates_now.txt").read(),
       "per_box": {os.path.basename(f)[:-5]: (json.load(open(f)) if open(f).read().strip() else None) for f in sorted(glob.glob("nm_launch/census/*.json")) if not f.endswith("summary.json")},
       "powered_off_boxes": po}
open(f"wt_nm/experiments/2026-10-14_q3_new_mains_launch/results/sim_error_census_2026-10-15_{sys.argv[2]}.json", "w").write(json.dumps(out, indent=1))
EOF
F=$RD/sim_error_census_2026-10-15_$TS.json
cd "$W" || exit 1
A 30 git add "$F" || { echo "census $T: git add FAILED ($F not committed)"; exit 1; }
A 60 git commit -q -m "Rate-limit census $T PT

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EaTTB7CWcfzNHhcqan35WH" > /dev/null 2>&1 || { echo "census $T: git commit FAILED ($F not committed)"; exit 1; }
lg=$(A 30 git log -1 --format=%H -- "$F") && [ -n "$lg" ] || { echo "census $T: $F not in history"; exit 1; }
HF=$(A 30 git rev-parse HEAD) || { echo "census $T: rev-parse failed"; exit 1; }; H=${HF:0:9}; ok=""
for i in 1 2 3; do
  # Codex r48-r49: success = the push call exits 0 AND a successful ls-remote shows HEAD's FULL hash
  A 90 git -c url."https://github.com/".insteadOf="git@github.com:" push -q origin HEAD > /dev/null 2>&1 &&
    lr=$(A 30 git ls-remote origin refs/heads/aswin/q3-new-mains-launch) && [ "$(cut -f1 <<< "$lr")" = "$HF" ] && { ok=$H; break; }
  sleep 5; done
if [ -n "$ok" ]; then echo "census $T committed @ $H (remote $ok): $F"; else echo "census $T committed locally @ $H; push NOT verified: $F"; fi
grep "last 30 min" "$E/rates_now.txt"
[ -n "$ok" ] || exit 2
