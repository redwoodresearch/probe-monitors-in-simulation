#!/usr/bin/env bash
# Laptop side, read-only: copy sim_ratelimit_list.py to each box with a running launch and run it over the box's
# interleaved (il_*) launch folders and S5's follow-up; one JSON per box in nm_launch/rl/.
E=$(cd "$(dirname "$0")" && pwd)
S=$E/../wt_nm/experiments/2026-10-14_q3_new_mains_launch/checks/sim_ratelimit_list.py  # as run from scratchpad/nm_launch/rl_all.sh
mkdir -p "$E/rl"
while read -r N B; do
  DEVBOX=aswin@$B perl -e 'alarm 60; exec @ARGV' bash "$E/devbox.sh" "mkdir -p /home/aswin/phase2/nm/tmp" < /dev/null > /dev/null 2>&1
  DEVBOX=aswin@$B perl -e 'alarm 60; exec @ARGV' bash "$E/devbox.sh" rsync -a "$S" box:phase2/nm/tmp/ < /dev/null > /dev/null 2>&1
  DEVBOX=aswin@$B perl -e 'alarm 240; exec @ARGV' bash "$E/devbox.sh" \
    "python3 /home/aswin/phase2/nm/tmp/sim_ratelimit_list.py $N \$(ls -d /home/aswin/phase2/runs/2026-10-1[45]_il_* /home/aswin/phase2/nm/runs/2026-10-1[45]_il_* /home/aswin/phase2/nm/runs/2026-10-14_newmains_s*_fu 2>/dev/null | grep -v '[.]')" \
    < /dev/null 2> /dev/null | tail -1 > "$E/rl/$N.json"
  echo "$N $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$E/rl/$N.json" 2>&1)"
done <<'BOXES'
box1 i-002bbb22c185c0f07
box2 i-0c02a38abcde32bfb
box3 i-0cc7555acb95c694b
box4 i-07bce857a818b6f69
xbox1 i-04331f272d6778df0
xbox2 i-00db08777bb29d430
xbox3 i-0f2299f29dd296259
simsastra i-05b1d910fd70016e6
BOXES
