#!/usr/bin/env bash
# nm_counted_vs_pinned_one.sh (nm360 sim-honest, d9 ~11:25): the same READ-ONLY step as nm_counted_vs_pinned_all.sh for ONE
# main on one box (S2's main on xbox3, box2 being down): copy the reviewed counted_vs_pinned.py to /tmp on the box (left
# there), run it with python3 -I on the main's run dir, write its JSON line to <out dir>/nm_counted_vs_pinned_<label>.json.
#   bash nm_counted_vs_pinned_one.sh <label> <instance> <main run dir> <out dir>
set -uo pipefail
LAB=$1 INST=$2 MAIN=$3 OUT=$4
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; E=$SP/wt_simh311/experiments/2026-10-15_q3_simhonest_311
mkdir -p "$OUT"
( cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 60; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
    rsync -a "$E/counted_vs_pinned.py" box:/tmp/counted_vs_pinned_simh.py ) < /dev/null > /dev/null 2>&1
( cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
    "python3 -I /tmp/counted_vs_pinned_simh.py $MAIN" ) < /dev/null 2> /dev/null | tail -1 > "$OUT/nm_counted_vs_pinned_$LAB.json"
python3 -I -c 'import json, sys; d = json.load(open(sys.argv[1])); print(sys.argv[2], {k: (v if not isinstance(v, (list, dict)) else len(v)) for k, v in d.items()})' \
  "$OUT/nm_counted_vs_pinned_$LAB.json" "$LAB" || echo "$LAB: NO RESULT"
