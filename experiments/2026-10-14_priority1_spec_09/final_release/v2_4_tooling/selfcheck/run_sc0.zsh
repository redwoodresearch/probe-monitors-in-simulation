#!/bin/zsh -f
# SC0 (calibration, real data first): run the jq/zsh port (sc_core.zsh) on the committed v2.3 record with v2.3's frozen commit message,
# and require its output to equal the frozen Python self-check log line for line.
#   zsh -f run_sc0.zsh
# Prints shas, the port's log (counts / ids / slot names only) and one verdict line: SC0_REPRODUCED or SC0_DIFF.
set -e; setopt pipefail; export LC_ALL=C
D=/Users/aswinsurya/probe-shift-routec-final-release-2026-10-20/v2_4r
HERE=$D/selfcheck
REC=$D/inputs/v2.3_record_80a77ecdc.json;   REC_SHA=ebb47e28f00596535f4fbaa5bdadb2b93b413bb768fb05e99b1abd36cac7de4f
MSG=$D/inputs/msg_final_release_v2.3.txt;    MSG_SHA=deeb65d0bf6085f2c623c0181d7970036b7c6423597d78d4f6e990dec4f489b0
LOG=$D/inputs/selfcheck_frozen_v2.3.log;     LOG_SHA=08ab0f10397f61a5
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
[[ $(shasum -a 256 $REC | cut -c1-64) == $REC_SHA ]] || { echo "STOP record sha"; exit 2; }
[[ $(git -C $REPO cat-file blob 80a77ecdc:experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.3_2026-10-20.json | shasum -a 256 | cut -c1-64) == $REC_SHA ]] || { echo "STOP record != committed blob"; exit 2; }
[[ $(shasum -a 256 $MSG | cut -c1-64) == $MSG_SHA ]] || { echo "STOP message sha"; exit 2; }
[[ $(shasum -a 256 $LOG | cut -c1-16) == $LOG_SHA ]] || { echo "STOP reference log sha"; exit 2; }
cmp -s $LOG /Users/aswinsurya/probe-shift-routec-final-release-2026-10-20/out/v2_3_combined_frozen_1251/selfcheck_frozen.log || { echo "STOP reference log != frozen original"; exit 2; }
for f in sc_core.zsh sc_nodes.jq sc_needs.jq sc_rest.jq; do print -r -- "PORT_FILE $f $(shasum -a 256 $HERE/$f | cut -c1-64)"; done
mkdir -p $HERE/_runs; OUT=$HERE/_runs/sc0_$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S).out
print -r -- "SC0_START $(TZ=America/Los_Angeles date '+%H:%M:%S %Z')"
set +e
zsh -f $HERE/sc_core.zsh $REC $MSG > $OUT
rc=$?
set -e
print -r -- "SC0_PORT_EXIT $rc"
cat $OUT
n=$( (diff $LOG $OUT || true) | wc -l | tr -d ' ')
if [[ $n == 0 && $rc == 0 ]]; then print -r -- "SC0_REPRODUCED lines $(wc -l < $OUT | tr -d ' ') out $OUT sha256 $(shasum -a 256 $OUT | cut -c1-64)"
else print -r -- "SC0_DIFF diff_lines $n port_exit $rc"; diff $LOG $OUT || true; exit 1; fi
print -r -- "SC0_END $(TZ=America/Los_Angeles date '+%H:%M:%S %Z')"
