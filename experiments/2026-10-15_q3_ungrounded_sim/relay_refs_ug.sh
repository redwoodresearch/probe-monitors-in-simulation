#!/usr/bin/env bash
# relay_refs_ug.sh (RULE.md): FROM THE LAPTOP, relay the +311 pilot cells' pinned honest_a reference files (the fork
# source, BA_REAL_LOG) from boxes 1-4's Q_k main runs to q3-xbox1, plus each main's launch.json (the index source).
# The laptop only MOVES bytes (tar through two ssh sessions); every file's sha256 is checked on xbox1 against the main's
# launch.json record by make_reference_index.py --check-files (run next, on xbox1). Refuses an existing target dir.
# bash 3.2-safe (macOS): no associative arrays.
#   DEVBOX_SH=<path to devbox.sh> bash relay_refs_ug.sh <config/tasks_ug_pilot.json>
set -euo pipefail
TASKS=$1; D=${DEVBOX_SH:?set DEVBOX_SH}; HERE=$(cd "$(dirname "$0")" && pwd)
XBOX1=aswin@i-04331f272d6778df0; DEST=/home/aswin/phase2/refs_ug
box() { case $1 in q1) echo aswin@i-002bbb22c185c0f07 ;; q2) echo aswin@i-0c02a38abcde32bfb ;;
                   q3) echo aswin@i-0cc7555acb95c694b ;; q4) echo aswin@i-07bce857a818b6f69 ;; esac; }
main() { case $1 in q1) echo /home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q1 ;; q2) echo /home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q2 ;;
                    q3) echo /home/aswin/phase2/runs/2026-10-14_155838_scaleup311_q3 ;; q4) echo /home/aswin/phase2/runs/2026-10-14_160246_scaleup311_q4 ;; esac; }
keys=$(python3 -c 'import json,sys; print(",".join(c["key"] for c in json.load(open(sys.argv[1]))["cells"]))' "$TASKS")
DEVBOX=$XBOX1 bash "$D" "set -e; [[ ! -e $DEST && ! -L $DEST ]] || { echo '$DEST exists'; exit 1; }; mkdir -p $DEST/q1 $DEST/q2 $DEST/q3 $DEST/q4"
total=0
for q in q1 q2 q3 q4; do
  # on box q: the reference copies of this quarter's wanted keys, from its main's launch.json
  list=$(DEVBOX=$(box $q) bash "$D" "python3 - $(main $q)/launch.json $keys" < "$HERE/refs_list.py")
  n=$(printf '%s' "$list" | grep -c . || true)
  echo "$q: $n reference files"
  [[ $n -gt 0 ]] || continue
  # stream: box q (tar of exactly those files, flat names) -> laptop -> xbox1 refs_ug/<q>/
  DEVBOX=$(box $q) bash "$D" "tar -cf - --transform='s|.*/||' $(printf '%s ' $list)" \
    | DEVBOX=$XBOX1 bash "$D" "tar -xf - -C $DEST/$q"
  DEVBOX=$(box $q) bash "$D" "cat $(main $q)/launch.json" | DEVBOX=$XBOX1 bash "$D" "cat > $DEST/${q}_launch.json"
  got=$(DEVBOX=$XBOX1 bash "$D" "ls $DEST/$q | grep -c '\.eval$'")
  [[ $got == "$n" ]] || { echo "FAIL: $q: $got files arrived, $n expected"; exit 1; }
  total=$((total + n))
done
echo "relayed $total reference files + 4 launch.json -> $XBOX1:$DEST (sha256s checked next by make_reference_index.py --check-files)"
