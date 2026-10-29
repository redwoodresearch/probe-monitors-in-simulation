#!/bin/bash
# G3 step-7 launch (q2, 10-13): literally `python3 .../g3_step7_barrier.py && <launch> --forecast-commit 6d7b1f50... --forecast-sha256 4243f2cd...`
# (09's requirement), then a launch record (the barrier's line, the exact command, app id, pids) in the experiment's results.
# usage: g3_step7.sh <tag> <max containers> <stop $> <deadline s> <run_extract_g3.py or g3_score_sim.py args (without the forecast args)>
S=/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad
L=$S/g3_resume_logs; tag=$1; cap=$2; stop=$3; dl=$4; shift 4
cd $S/wt_g3 || exit 1
FC=6d7b1f50c2c08a5887962b5066681684d6e1ae63; FS=4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77
CMD=(python3 experiments/2026-10-13_g3_new_reader/g3_step7_barrier.py)
LAUNCH=($S/g3_launch.sh $tag $cap $stop $dl "$@" --forecast-commit $FC --forecast-sha256 $FS)
B=$("${CMD[@]}") && OUT=$("${LAUNCH[@]}"); rc=$?
echo "$B"; echo "$OUT"
python3 - "$tag" "$B" "$OUT" "${CMD[*]} && ${LAUNCH[*]}" "$rc" <<'PY'
import json, sys, time
tag, b, out, cmd, rc = sys.argv[1:6]
p = f"/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad/wt_g3/experiments/2026-10-13_g3_new_reader/results/step7_launch_{tag}_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
json.dump({"tag": tag, "made (PT)": time.strftime("%Y-%m-%d %H:%M:%S"), "barrier line": b, "exact launch command": cmd, "launch helper output": out, "rc": int(rc)}, open(p, "x"), indent=1)
print(p)
PY
