#!/usr/bin/env bash
# v5 (10-15 ~10:30): runs census_commit_v3.sh (v2 + the q2 replay source). Laptop: the census cadence (d9 10-15 09:27 PT, Aswin asked for speed; 0d raised the start cap 60 -> 90): census_commit_v3.sh
# 10 min after the latest COMMITTED census while it shows > 60 simulated runs running, otherwise 25 min after it, until
# 14:00 PT (no census starts after 14:00; the loop keeps re-planning until then, so another process's census can still
# move the clock). Codex r41: censuses are read only from committed, clean files, newest readable first; a census that
# lands while waiting re-plans; census_commit_v3.sh gets SKIP_IF_NEWER_THAN=<the census this plan was based on> and skips
# when a newer one was committed while it waited for the lock (no duplicates); a failed census retries after 120 s, never
# back-to-back; each census runs in the background and is stopped (with its children) after 900 s. Log: census_loop.log.
E=$(cd "$(dirname "$0")" && pwd); W=$E/../wt_nm; RD=experiments/2026-10-14_q3_new_mains_launch/results
END=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' '2026-10-15 14:00:00' +%s)
latest() {  # "<census epoch> <simulated running>" of the newest committed, clean, readable census; "0 999" if none
  local f
  for f in $(cd "$W" && ls -r $RD/sim_error_census_2026-10-15_*.json 2>/dev/null); do
    perl -e 'alarm 30; exec @ARGV' git -C "$W" ls-files --error-unmatch "$f" > /dev/null 2>&1 || continue  # Codex r48: bounded
    st=$(perl -e 'alarm 30; exec @ARGV' git -C "$W" status --porcelain -- "$f") && [ -z "$st" ] || continue
    python3 - "$W/$f" <<'PY' 2>/dev/null && return 0
import json, sys
from datetime import datetime
from zoneinfo import ZoneInfo
c = json.load(open(sys.argv[1]))
t = datetime.strptime(c["census_time_pt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()
n = sum(x["sim_running"] for v in c["per_box"].values() if isinstance(v, dict) for x in v.values() if isinstance(x, dict) and "sim_running" in x)
print(int(t), n)
PY
  done
  echo "0 999"
}
retry_at=0
while :; do
  read -r t n <<< "$(latest)"
  (( t > 0 )) || { t=$(date +%s); n=999; }  # none readable: plan 10 min from now, never back-to-back
  next=$(( t + (n > 60 ? 600 : 1500) )); (( next < retry_at )) && next=$retry_at
  if (( next > END )); then  # nothing to run before 14:00 unless another census moves the clock
    (( $(date +%s) >= END )) && break
    sleep 60; continue
  fi
  while (( $(date +%s) < next )); do sleep 20; read -r t2 n2 <<< "$(latest)"; (( t2 > t )) && break; done
  read -r t2 n2 <<< "$(latest)"; (( t2 > t )) && continue  # a newer census landed: re-plan from it
  (( $(date +%s) < END )) || break  # Codex r42: never start after 14:00 (census_commit_v3.sh re-checks after its lock)
  SKIP_IF_NEWER_THAN=$t NOT_AFTER=$END bash "$E/census_commit_v3.sh" > "$E/census_loop5_out.txt" 2>&1 & c=$!
  for k in $(seq 1 180); do kill -0 $c 2>/dev/null || break; sleep 5; done
  if kill -0 $c 2>/dev/null; then pkill -TERM -P $c; kill -TERM $c; wait $c 2>/dev/null; rc=124; else wait $c; rc=$?; fi
  out=$(head -2 "$E/census_loop5_out.txt" | tr '\n' ' ')
  echo "$(TZ=America/Los_Angeles date '+%F %T PT') [loop5, prev running $n, rc $rc] $out" >> "$E/census_loop.log"
  # Codex r42: success = a NEWER committed census exists now; anything else (failure, skip, git error) backs off 120 s
  read -r t3 n3 <<< "$(latest)"
  if (( rc == 0 && t3 > t )); then retry_at=0; else retry_at=$(( $(date +%s) + 120 )); fi
done
