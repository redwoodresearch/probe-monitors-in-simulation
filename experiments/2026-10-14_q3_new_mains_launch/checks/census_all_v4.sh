#!/usr/bin/env bash
# Laptop side, read-only: copy sim_error_census.py to each box with a running launch and run it over the box's interleaved
# (il_*) launch folders and S5's follow-up; one JSON per box in nm_launch/census/. 10-15 06:15: also the ungrounded
# fork's /home/aswin/phase2/runs/2026-10-15_ug. 10-15 09:08 (d9): every folder is listed only once its launch.json exists
# (a run dir that aborted before writing one drops out); renamed folders (any "." in the name, e.g. *.aborted_0855) drop
# out. Fail closed (Codex r38): a folder is dropped only when it is searchable AND launch.json is confirmed absent (not a
# file, not a dangling link); an unsearchable folder or an unreadable launch.json is listed and becomes a read error.
# Codex r39: a runs/ parent that exists but can't be listed or searched adds "<parent>/PARENT_UNREADABLE", a read error.
# 10-15 09:41 (d9): also today's sim-honest launches /home/aswin/phase2/runs/2026-10-15_simh_* (same OpenAI key), same rule.
# v3 (10-15 ~13:40, d9; a NEW file, swapped in at a census boundary): each box read records ONLY its steps' exit codes and
# seconds (mkdir, rsync, read; 142 = the step's perl alarm fired, a timeout; 255 = ssh/SSM failure) in census_err/<box>.json,
# outside census/ (whose *.json are the per-box records). No stderr text is captured or written anywhere (d9: Aswin's standing
# rule, no credential-detection or masking code); stderr still goes to /dev/null as in v2. The box JSONs are unchanged.
# v4 (10-15 ~21:55, d9; a NEW file, registry addition only): also /home/aswin/phase2/runs/2026-10-15_sp493_* (the design fork's
# S_paper493 simulated launches: sp493_sim_L1 on box3, _L2 on box4, later L3/g6/g7), same rule as the other folders.
E=$(cd "$(dirname "$0")" && pwd)
S=$E/../wt_nm/experiments/2026-10-14_q3_new_mains_launch/checks/sim_error_census_v2.py
mkdir -p "$E/census" "$E/census_err"
while read -r N B; do
  t0=$(date +%s)
  DEVBOX=aswin@$B timeout_s=60 perl -e 'alarm 60; exec @ARGV' bash "$E/devbox.sh" "mkdir -p /home/aswin/phase2/nm/tmp" < /dev/null > /dev/null 2>&1; r1=$?; t1=$(date +%s)
  DEVBOX=aswin@$B perl -e 'alarm 60; exec @ARGV' bash "$E/devbox.sh" rsync -a "$S" box:phase2/nm/tmp/ < /dev/null > /dev/null 2>&1; r2=$?; t2=$(date +%s)
  DEVBOX=aswin@$B perl -e 'alarm 150; exec @ARGV' bash "$E/devbox.sh" \
    'python3 /home/aswin/phase2/nm/tmp/sim_error_census_v2.py $(for p in /home/aswin/phase2/runs /home/aswin/phase2/nm/runs; do [ ! -e "$p" ] || { [ -d "$p" ] && [ -r "$p" ] && [ -x "$p" ]; } || echo "$p/PARENT_UNREADABLE"; done; for d in /home/aswin/phase2/runs/2026-10-1[45]_il_* /home/aswin/phase2/nm/runs/2026-10-1[45]_il_* /home/aswin/phase2/nm/runs/2026-10-14_newmains_s*_fu /home/aswin/phase2/runs/2026-10-15_ug /home/aswin/phase2/runs/2026-10-15_simh_* /home/aswin/phase2/runs/2026-10-15_sp493_*; do [ -d "$d" ] || continue; [ -x "$d" ] && [ ! -e "$d/launch.json" ] && [ ! -L "$d/launch.json" ] && continue; echo "$d"; done | grep -v "[.]")' \
    < /dev/null 2> /dev/null | tail -1 > "$E/census/$N.json"; r3=${PIPESTATUS[0]}; t3=$(date +%s)
  printf '{"instance": "%s", "mkdir": {"rc": %d, "seconds": %d}, "rsync": {"rc": %d, "seconds": %d}, "read": {"rc": %d, "seconds": %d}, "output_bytes": %d}\n' \
    "$B" "$r1" $((t1 - t0)) "$r2" $((t2 - t1)) "$r3" $((t3 - t2)) "$(wc -c < "$E/census/$N.json" | tr -d ' ')" > "$E/census_err/$N.json"
  echo "$N done"
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
