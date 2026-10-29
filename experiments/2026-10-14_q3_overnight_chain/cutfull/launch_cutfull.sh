#!/bin/bash
# d9's CUT fork (Append 41): launch the cutfull batch's stagings or extractions, each its own detached `modal run` whose fan-out runs
# in a Modal driver (stage_driver / orchestrate; started markers and call ids on the batch's Volume), with the batch's OWN app,
# Volume and leases. At most <par> launcher clients at a time; one log per target or config in <logdir>, exits in <logdir>/exits.txt;
# the launcher exits nonzero if any launch did.
#   launch_cutfull.sh stage   <private targets json> <logdir> <par>     every target in the file
#   launch_cutfull.sh extract <logdir> <par> <config yaml under config/> ...
#   launch_cutfull.sh format  <private targets json> <plan json> <logdir> <par>   (stage_chain --format-all, every epoch of the planned arm)
# "dirty" = a modified TRACKED file (untracked results and logs are allowed), as stage_chain.py's own check.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
export MODAL=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
export REL=experiments/2026-10-14_q3_overnight_chain
VOLUME=probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14
cd "$REPO"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "dirty tree; commit first" >&2; exit 1; fi
posint() { [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "par must be a positive integer" >&2; exit 2; }; }
stage_one() {
  "$MODAL" run --detach "$REL/stage_chain.py" --target "$1" > "$LOGDIR/stage_$1.log" 2>&1
  rc=$?; echo "$1 exit $rc" >> "$LOGDIR/exits.txt"; return $rc
}
extract_one() {
  "$MODAL" run --detach "$REL/extract_chain.py" --config "$1" --attempt "$ATTEMPT" > "$LOGDIR/extract_$(basename "$1" .yaml).log" 2>&1
  rc=$?; echo "$1 exit $rc" >> "$LOGDIR/exits.txt"; return $rc
}
format_one() {  # "<target> <arm> <shards>"
  set -- $1
  "$MODAL" run --detach "$REL/stage_chain.py" --target "$1" --format-all --arm "$2" --format-shards "$3" > "$LOGDIR/format_$1__${2%%@*}.log" 2>&1
  rc=$?; echo "$1 $2 exit $rc" >> "$LOGDIR/exits.txt"; return $rc
}
export -f stage_one extract_one format_one
mode=$1; shift
if [ "$mode" = stage ]; then
  tf="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; export LOGDIR=$2; posint "$3"
  mkdir -p "$LOGDIR"
  uv run --no-project --python 3.12 python -c "import json,sys; sys.stdout.write(''.join(t + '\0' for t in json.load(open(sys.argv[1]))))" "$tf" |
    CHAIN_STAGE_APP=probe-shift_aswin-surya_sub-q3cutfull_stage_kill-2026-11-14 CHAIN_STAGE_VOLUME=$VOLUME CHAIN_TARGETS_FILE="$tf" \
    PYTHONPATH="$REPO/src" xargs -0 -n 1 -P "$3" bash -c 'stage_one "$1"' _
elif [ "$mode" = format ]; then  # every-epoch format check: <targets json> <plan json {targets: {T: [arms]}}> <logdir> <par>; ~8 epochs per shard
  tf="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; pf=$2; export LOGDIR=$3; posint "$4"
  mkdir -p "$LOGDIR"
  uv run --no-project --python 3.12 python -c "
import json, sys
T, P = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))['targets']
E = json.load(open(sys.argv[2])).get('epochs') or {t: T[t]['arm_list']['epochs'] for t in P}  # staged epochs per arm (the plan's, else the arm list's)
sys.stdout.write(''.join(f'{t} {a} {min(48, -(-E[t][a] // 8))}' + '\0' for t, arms in sorted(P.items()) for a in arms))" "$tf" "$pf" |
    CHAIN_STAGE_APP=probe-shift_aswin-surya_sub-q3cutfull_stage_kill-2026-11-14 CHAIN_STAGE_VOLUME=$VOLUME CHAIN_TARGETS_FILE="$tf" \
    PYTHONPATH="$REPO/src" xargs -0 -n 1 -P "$4" bash -c 'format_one "$1"' _
elif [ "$mode" = extract ]; then
  export LOGDIR=$1 ATTEMPT=${ATTEMPT:-1}; posint "$2"; par=$2; shift 2
  [ $# -ge 1 ] || { echo "no extraction configs" >&2; exit 2; }
  mkdir -p "$LOGDIR"
  printf '%s\0' "$@" |
    XREAD_APP=probe-shift_aswin-surya_sub-q3chain_cutfull_kill-2026-11-14 Q4OWN_VOLUME=$VOLUME \
    XREAD_LEASES=probe-shift_aswin-surya_sub-q3cutfull_leases_kill-2026-11-14 XREAD_TIMEOUT_S="${XREAD_TIMEOUT_S:-5400}" \
    PYTHONPATH="$REPO/src" xargs -0 -n 1 -P "$par" bash -c 'extract_one "$1"' _
else
  echo "usage: stage|extract" >&2; exit 2
fi
