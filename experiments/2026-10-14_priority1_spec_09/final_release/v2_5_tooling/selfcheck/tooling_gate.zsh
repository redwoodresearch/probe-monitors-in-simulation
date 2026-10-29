#!/bin/zsh -f
# tooling_gate.zsh: run tooling_gate.jq over every file under a folder (the staged v2_4_tooling/). jq + shasum; no Python.
# Prints counts, file names, check names and line numbers only (never a token). Exit 0 = TOOLING_GATE_OK, 1 = TOOLING_GATE_FAIL, 2 = STOP.
#   zsh -f tooling_gate.zsh <folder>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; T=${1:A}
P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl
[[ $(shasum -a 256 $P18 | cut -c1-16) == 13d29fb90cac3e3e ]] || { echo "STOP v18 sha"; exit 2; }
[[ -d $T ]] || { echo "STOP no folder"; exit 2; }
RUN=$HERE/_runs/tooling_gate_$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_$$; mkdir -p $RUN
: > $RUN/files.jsonl
find $T -type f | LC_ALL=C sort | while IFS= read -r f; do
  rel=${f#$T/}
  if [[ $f == *.json ]]; then js=$(jq -c . $f) || { echo "STOP unparseable json: $rel"; exit 2; }; else js=null; fi
  jq -R -s -c --arg name "$rel" --argjson json "$js" '{name: $name, raw: ., json: $json}' $f >> $RUN/files.jsonl
done
[[ $(find $T -type l | wc -l | tr -d ' ') == 0 ]] || { echo "STOP symlinks in the folder"; exit 2; }
jq -n -c -f $HERE/tooling_gate.jq --slurpfile F $RUN/files.jsonl --slurpfile P18 $P18 > $RUN/gate.json 2> $RUN/gate.err || { echo "STOP tooling_gate.jq failed (stderr in $RUN/gate.err)"; exit 2; }
jq -r '"TOOLING_GATE " + (.n | to_entries | map("\(.key) \(.value)") | join(" ")) + " findings \(.bad | length)"' $RUN/gate.json
jq -r '.bad[] | "FLAG " + .' $RUN/gate.json
if jq -e '.bad | length == 0' $RUN/gate.json > /dev/null; then print TOOLING_GATE_OK; else print TOOLING_GATE_FAIL; exit 1; fi
