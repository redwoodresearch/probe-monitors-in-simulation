#!/bin/bash
# test_census_rl_pause_v2.sh: replays committed censuses (origin/aswin/q3-new-mains-launch) through census_rl.py (old) and
# census_rl_pause_v2.py (new). Expected: the 00:14 census across midnight trips the old test (3) and not the new one (0);
# the 13:06 and 13:23 censuses (real RateLimitError, 2.2% and 6.5%) trip the new one (3); 00:00 and 13:00 do not (0);
# 01:02 (before the exact count existed) is a parse failure (2). Prints one line per case and PASS/FAIL.
set -u
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims; B=origin/aswin/q3-new-mains-launch
D=experiments/2026-10-14_q3_new_mains_launch/results; HERE=$(cd "$(dirname "$0")" && pwd); fail=0
run() {  # $1 script, $2 census stamp, $3 since, $4 expected rc
  out=$(git -C "$REPO" show "$B:$D/sim_error_census_$2.json" | python3 -I "$HERE/$1" pause --since "$3" 2>&1); rc=$?
  [[ $rc == "$4" ]] && v=PASS || { v=FAIL; fail=1; }
  echo "$v $1 census $2 since '$3' rc $rc (expected $4): $out"
}
run census_rl.py          2026-10-16_0014 "2026-10-15 23:02" 3   # reproduces v9's false trip
run census_rl_pause_v2.py 2026-10-16_0014 "2026-10-15 23:02" 0   # negative test across midnight
run census_rl_pause_v2.py 2026-10-16_0000 "2026-10-15 23:02" 0
run census_rl_pause_v2.py 2026-10-15_1300 "2026-10-15 12:50" 0
run census_rl_pause_v2.py 2026-10-15_1306 "2026-10-15 12:50" 3   # positive: 2 of 90
run census_rl_pause_v2.py 2026-10-15_1323 "2026-10-15 12:50" 3   # positive: 6 of 93
run census_rl_pause_v2.py 2026-10-15_1306 "2026-10-15 13:10" 0   # census before --since is not used
run census_rl_pause_v2.py 2026-10-15_0102 "2026-10-15 00:00" 2   # no exact count -> parse failure
drop() {  # $1 census stamp, $2 python edit of d, $3 label: each missing field alone -> parse failure (2)
  out=$(git -C "$REPO" show "$B:$D/sim_error_census_$1.json" | python3 -I -c "import json, sys; d = json.load(sys.stdin); $2; print(json.dumps(d))" | python3 -I "$HERE/census_rl_pause_v2.py" pause --since "2026-10-15 23:02" 2>&1); rc=$?
  [[ $rc == 2 ]] && v=PASS || { v=FAIL; fail=1; }
  echo "$v census_rl_pause_v2.py census $1 with $3 rc $rc (expected 2): $out"
}
drop 2026-10-16_0014 'd.pop("census_time_pt")' "census_time_pt removed"
drop 2026-10-16_0014 'd["rates_text"] = d["rates_text"].replace("exact last 30 min", "last 30 min")' "the exact line removed"
(( fail == 0 )) && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }
