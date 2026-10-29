#!/bin/zsh -f
# msg_v27.zsh: the v2.6 release commit message checks on a record and its message. Prints counts and check names only.
#   zsh -f msg_v27.zsh <record.json> <message.txt> <number of files in v2_7_tooling/>
# (1) msg_v27_check.jq: independent checks (its own constants; counts re-derived from the record).
# (2) b8 (Codex v24 r1 item 6): the message must equal, byte for byte, the message msg_v27_make.jq generates from the record, so no extra
#     line, number or wording can be added (MISMATCH message: not the generated message).
# The v2.3 message checks (record sha, sensitivity, FINAL outputs, figure labels, exclusions) run separately:
#   SC_REST=<this folder>/sc_rest_v27.jq zsh -f sc_core.zsh <record.json> <message.txt>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; REC=${1:A}; MSG=${2:A}; NTOOL=${3:-}
[[ -f $REC && -f $MSG && $NTOOL =~ ^[0-9]+$ ]] || { echo "STOP usage"; exit 2; }
RECSHA=$(shasum -a 256 $REC | cut -c1-64)
OUT=$(jq -n -c -f $HERE/msg_v27_check.jq --slurpfile REC $REC --rawfile MSG $MSG --arg RECSHA $RECSHA --argjson NTOOL $NTOOL) || { echo "STOP msg_v27_check.jq failed"; exit 2; }
GEN=$(jq -n -r -f $HERE/msg_v27_make.jq --slurpfile REC $REC --arg RECSHA $RECSHA --argjson NTOOL $NTOOL) || { echo "STOP msg_v27_make.jq failed"; exit 2; }
EXACT=$(cmp -s <(print -r -- "$GEN") $MSG && print true || print false)
OUT=$(print -r -- "$OUT" | jq -c --argjson ex $EXACT '.n += 1 | if $ex then . else .bad += ["message: not the generated message"] end')
print -r -- "MSGCHECK_V27 checks $(print -r -- "$OUT" | jq '.n') failed $(print -r -- "$OUT" | jq '.bad | length') exact $EXACT"
print -r -- "$OUT" | jq -r '.bad[] | "MISMATCH " + .'
if print -r -- "$OUT" | jq -e '.bad | length == 0' > /dev/null; then print MSGCHECK_V27_OK; else print MSGCHECK_V27_FAIL; exit 1; fi
