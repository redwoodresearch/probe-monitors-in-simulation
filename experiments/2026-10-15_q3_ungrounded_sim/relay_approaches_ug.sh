#!/usr/bin/env bash
# relay_approaches_ug.sh (RULE.md): FROM THE LAPTOP, add the approach texts that q3-xbox1 lacks for this launch's cells
# (run_pilot.py restores each cell's approach from ~/phase2/approaches/<sha256>.txt and refuses a missing or wrong one).
# Bytes only: the texts are never opened, printed or parsed here; tar moves them box k -> laptop -> xbox1, and xbox1
# checks every file's sha256 against its name. Never overwrites (tar --skip-old-files; an existing name is left as is).
# bash 3.2-safe.   DEVBOX_SH=<path to devbox.sh> bash relay_approaches_ug.sh <config/tasks_ug_pilot.json>
set -euo pipefail
TASKS=$1; D=${DEVBOX_SH:?set DEVBOX_SH}
XBOX1=aswin@i-04331f272d6778df0; A=/home/aswin/phase2/approaches
box() { case $1 in q1) echo aswin@i-002bbb22c185c0f07 ;; q2) echo aswin@i-0c02a38abcde32bfb ;;
                   q3) echo aswin@i-0cc7555acb95c694b ;; q4) echo aswin@i-07bce857a818b6f69 ;; esac; }
# the shas each quarter needs (the +311 cells: their key's quarter comes from the interleaved pilot slices; old cells' texts are on xbox1)
for q in q1 q2 q3 q4; do
  want=$(python3 - "$TASKS" "$q" <<'PY'
import json, re, subprocess, sys
tasks, q = sys.argv[1], sys.argv[2]
src = "experiments/2026-10-15_q3_interleaved_run3_sim2/config/tasks_il_%s_pilot.json" % q
qkeys = {c["key"] for c in json.loads(subprocess.run(["git", "show", "d7f43e2fd:" + src], check=True, capture_output=True).stdout)["cells"]}
shas = sorted({c["approach_sha256"] for c in json.load(open(tasks))["cells"] if c["key"] in qkeys})
assert all(re.fullmatch(r"[0-9a-f]{64}", s) for s in shas)
print(" ".join(shas))
PY
)
  missing=$(DEVBOX=$XBOX1 bash "$D" "cd $A && for s in $want; do [[ -e \$s.txt ]] || echo \$s; done" | tr '\n' ' ')
  n=$(printf '%s' "$missing" | wc -w | tr -d ' ')
  echo "$q: $(printf '%s' "$want" | wc -w | tr -d ' ') needed, $n missing on xbox1"
  [[ $n -gt 0 ]] || continue
  DEVBOX=$(box $q) bash "$D" "cd $A && tar -cf - $(for s in $missing; do printf '%s.txt ' "$s"; done)" \
    | DEVBOX=$XBOX1 bash "$D" "cd $A && tar --skip-old-files -xf -"
  DEVBOX=$XBOX1 bash "$D" "cd $A && for s in $missing; do [[ \$(sha256sum < \$s.txt | cut -c1-64) == \$s ]] || { echo \"FAIL: \$s.txt\"; exit 1; }; done; echo '$q: $n texts added, each sha256 = its name'"
done
