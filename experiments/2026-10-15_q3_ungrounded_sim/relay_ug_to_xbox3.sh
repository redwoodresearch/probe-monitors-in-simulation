#!/usr/bin/env bash
# relay_ug_to_xbox3.sh (RULE.md; d9 moved the arm to q3-xbox3): FROM THE LAPTOP, copy the ungrounded arm's inputs from
# q3-xbox1 (where relay_refs_ug.sh and relay_approaches_ug.sh put them, verified) to q3-xbox3, at the SAME absolute
# paths: ~/phase2/refs_ug (the 115 +311 references + the 4 Q_k launch.json), ~/phase2/il_refindex_old/launch.json (the
# S_old index, pinned 79acbfc1… by build_index_ug.sh) and the 34 old cells' reference files it names, and the approach
# texts xbox3 lacks for the 149 cells. Bytes only (tar through two ssh sessions); nothing is opened. Integrity is checked
# on xbox3: build_index_ug.sh pins every index source by sha256 and --check-files re-hashes every reference file; each
# approach text's sha256 must equal its name. Refuses existing reference targets; never overwrites an approach text.
# bash 3.2-safe.   DEVBOX_SH=<devbox.sh> bash relay_ug_to_xbox3.sh <config/tasks_ug_pilot.json>
set -euo pipefail
TASKS=$1; D=${DEVBOX_SH:?set DEVBOX_SH}; HERE=$(cd "$(dirname "$0")" && pwd)
SRC=aswin@i-04331f272d6778df0; DST=aswin@i-0f2299f29dd296259; P=/home/aswin/phase2
keys=$(python3 -c 'import json,sys; print(",".join(c["key"] for c in json.load(open(sys.argv[1]))["cells"]))' "$TASKS")
shas=$(python3 -c 'import json,sys; print(" ".join(sorted({c["approach_sha256"] for c in json.load(open(sys.argv[1]))["cells"]})))' "$TASKS")
DEVBOX=$DST bash "$D" "for p in $P/refs_ug $P/il_refindex_old $P/refs_old; do [[ ! -e \$p && ! -L \$p ]] || { echo \"FAIL: \$p exists\"; exit 1; }; done"
# 1. refs_ug, whole (the relayed +311 references and the 4 Q_k launch.json)
DEVBOX=$SRC bash "$D" "tar -C $P -cf - refs_ug" | DEVBOX=$DST bash "$D" "tar -C $P -xf -"
# 2. the S_old index and the old cells' reference files it names (same absolute paths)
list=$(DEVBOX=$SRC bash "$D" "python3 - $P/il_refindex_old/launch.json $keys" < "$HERE/refs_list.py")
n=$(printf '%s' "$list" | grep -c . || true)
echo "old cells' reference files: $n"
DEVBOX=$SRC bash "$D" "tar -C / -cf - ${P#/}/il_refindex_old/launch.json $(for f in $list; do printf '%s ' "${f#/}"; done)" \
  | DEVBOX=$DST bash "$D" "tar -C / -xf -"
# 3. approach texts xbox3 lacks (by sha; never overwrite)
missing=$(DEVBOX=$DST bash "$D" "cd $P/approaches && for s in $shas; do [[ -e \$s.txt ]] || echo \$s; done" | tr '\n' ' ')
m=$(printf '%s' "$missing" | wc -w | tr -d ' ')
echo "approach texts: $(printf '%s' "$shas" | wc -w | tr -d ' ') needed, $m missing on xbox3"
if [[ $m -gt 0 ]]; then
  DEVBOX=$SRC bash "$D" "cd $P/approaches && tar -cf - $(for s in $missing; do printf '%s.txt ' "$s"; done)" \
    | DEVBOX=$DST bash "$D" "cd $P/approaches && tar --skip-old-files -xf -"
  DEVBOX=$DST bash "$D" "cd $P/approaches && for s in $missing; do [[ \$(sha256sum < \$s.txt | cut -c1-64) == \$s ]] || { echo \"FAIL: \$s.txt\"; exit 1; }; done; echo '$m approach texts added, each sha256 = its name'"
fi
DEVBOX=$DST bash "$D" "ls $P/refs_ug/q1 $P/refs_ug/q2 $P/refs_ug/q3 $P/refs_ug/q4 | grep -c '\.eval$'; sha256sum $P/il_refindex_old/launch.json $P/refs_ug/q?_launch.json"
