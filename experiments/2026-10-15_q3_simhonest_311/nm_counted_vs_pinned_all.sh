#!/usr/bin/env bash
# nm_counted_vs_pinned_all.sh (nm360 sim-honest prep, d9 ~11:15): READ-ONLY. For each reachable nm main, copy the reviewed
# counted_vs_pinned.py (09's Append 25f census script: reads <main>/launch.json and sha256-hashes each counted honest_a .eval;
# writes nothing on the box except the copy) to /tmp/counted_vs_pinned_simh.py on that box (left there; not removed), run it
# with python3 -I on the main's run dir, and write its one-line JSON to <out dir>/nm_counted_vs_pinned_<label>.json on the
# laptop, plus <out dir>/nm_counted_vs_pinned_all.txt (one summary line per main). box3 first (powers off 11:45 PT).
#   bash nm_counted_vs_pinned_all.sh <out dir>
set -uo pipefail
OUT=$1
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; E=$SP/wt_simh311/experiments/2026-10-15_q3_simhonest_311
mkdir -p "$OUT"
while read -r LAB INST MAIN; do
  ( cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 60; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
      rsync -a "$E/counted_vs_pinned.py" box:/tmp/counted_vs_pinned_simh.py ) < /dev/null > /dev/null 2>&1
  ( cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
      "python3 -I /tmp/counted_vs_pinned_simh.py $MAIN" ) < /dev/null 2> /dev/null | tail -1 > "$OUT/nm_counted_vs_pinned_$LAB.json"
  python3 -I -c 'import json, sys; d = json.load(open(sys.argv[1])); print(sys.argv[2], {k: (v if not isinstance(v, (list, dict)) else len(v)) for k, v in d.items()})' \
    "$OUT/nm_counted_vs_pinned_$LAB.json" "$LAB" >> "$OUT/nm_counted_vs_pinned_all.txt" 2>&1 || echo "$LAB: NO RESULT" >> "$OUT/nm_counted_vs_pinned_all.txt"
done <<'MAINS'
s3 i-0cc7555acb95c694b /home/aswin/phase2/nm/runs/2026-10-14_newmains_s3
s1 i-002bbb22c185c0f07 /home/aswin/phase2/nm/runs/2026-10-14_newmains_s1
s4 i-07bce857a818b6f69 /home/aswin/phase2/nm/runs/2026-10-14_newmains_s4
s4r i-07bce857a818b6f69 /home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r
s5 i-05b1d910fd70016e6 /home/aswin/phase2/nm/runs/2026-10-14_newmains_s5
MAINS
cat "$OUT/nm_counted_vs_pinned_all.txt"
