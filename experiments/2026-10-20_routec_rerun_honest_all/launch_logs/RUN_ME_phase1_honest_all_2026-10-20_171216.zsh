#!/bin/zsh
# honest_all route A, PHASE 1 (addenda F3, G, H, H2). v2. COMMIT 82ea3e926 (v12 + F/G/H/H2; 0d OK COMMIT 17:03:27). Bound (H7 terms, bc): hash $0.03173912 + toys $0.06858140 = $0.10032052.
# v2 = v1 (92bf3de7) + the 2 REQUIRED fixes of Codex (17:10, FAIL): (a) hash-shards writes to a provisional .unverified path; the manifest is
# published at $MANIFEST (mv -n, then frozen) only after exit, RESULT, every gate and the sha all pass, so a failure never publishes it;
# (b) the APP_ID watcher exits 1 when no app id is found, and launch() STOPs on a failed watcher instead of ignoring its status.
# Run by 0d ONLY on 2e's relayed ARMED line:  zsh RUN_ME_phase1_honest_all_<stamp>.zsh "<ARMED_AT>"
# Sequence: assert the committed worktree -> live-app filter self-test -> (1) `--mode hash-shards` (CPU): writes the replay manifest to a
# NEW local file; the file's sha256 must equal the RESULT line's manifest_sha256; the file is frozen (a-w) -> (2) the 5 CPU toys with the
# exact signatures of H6 as amended by H2 (UNITNEG 16). Every call: 0 live apps of the name first, FOREGROUND, a fresh log, no tee, the
# APP_ID printed with `date` as soon as it appears, 15 s pause after it returns (refresh3's guard binds exactly one new id per ~5 s sighting).
# Prints only OK/STOP lines, counts, APP_ID/RETURNED lines, the structured RESULT lines, and the manifest pin (a sha). STOPs on anything
# unexpected. Check with `zsh -n` only.
set -euo pipefail
ARMED_AT=${1:?usage: zsh RUN_ME_phase1_honest_all.zsh "<ARMED_AT as relayed by 2e>"}
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
WT=/Users/aswinsurya/probe-shift-wt-rerun-honest-all-2026-10-20
BR=aswin/routec-rerun-honest-all
COMMIT=82ea3e9265fec1fec12a6ce17aa447068a7dfada
REL=experiments/2026-10-20_routec_rerun_honest_all
D=$WT/$REL
ITEM=/Users/aswinsurya/probe-shift-reruns-2026-10-20/honest_all
APP=probe-shift_aswin-surya_sub-rerun_honall_kill-2026-11-19
CODE_BLOB=177f19b5a88116eb8ebc04b80833aaec351eb146
CODE_SHA=b6c45a573054d05f728843eeca319782973494b6a8f03c8c08c77f2817a34f2f
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
LOGDIR=$ITEM/phase1_$STAMP
MANIFEST=$LOGDIR/replay_manifest_$STAMP.json              # published only after every hash check passes
MANIFEST_CAND=$LOGDIR/hash_out_$STAMP.unverified.json     # where hash-shards writes; never committed

stop() { echo "STOP: $1 ($(date))" >&2; exit 1; }
echo "honest_all phase 1: ARMED_AT=[$ARMED_AT] stamp=$STAMP start=$(date)"
[[ $COMMIT != __FILL_AFTER_CODE_COMMIT__ ]] || stop "draft: COMMIT not filled in"

# ---- 1. the committed worktree: attached, HEAD == origin == COMMIT, clean (git's exit checked first), code blob and sha
[[ -d $WT ]] || stop "worktree missing"
[[ $(git -C $WT symbolic-ref --short HEAD) == $BR ]] || stop "worktree not attached to $BR"
git -C $WT fetch -q origin $BR
LH=$(git -C $WT rev-parse HEAD) || stop "rev-parse HEAD failed"
OH=$(git -C $WT rev-parse origin/$BR) || stop "rev-parse origin/$BR failed"
[[ -n $LH && $LH == $OH && $LH == $COMMIT ]] || stop "HEAD != origin/$BR != $COMMIT"
ST=$(git -C $WT status --porcelain) || stop "git status failed"
[[ -z $ST ]] || stop "worktree not clean"
[[ $(git -C $WT rev-parse HEAD:$REL/check_honest_all.py) == $CODE_BLOB ]] || stop "committed code blob"
[[ $(git -C $WT hash-object $D/check_honest_all.py) == $CODE_BLOB ]] || stop "working-copy code blob"
[[ $(shasum -a 256 $D/check_honest_all.py | cut -c1-64) == $CODE_SHA ]] || stop "code sha256"
echo "OK worktree: $BR @ $COMMIT == origin, clean, code blob $CODE_BLOB"
[[ ! -e $LOGDIR ]] || stop "log folder exists"
mkdir -p $LOGDIR

# ---- 2. live-app gate (schema-checked, fails closed), self-tested on fixtures first (as the reviewed 14:44 launcher)
SCHEMA='type == "array" and all(.[]; type == "object" and ((.description | type) == "string") and ((.state | type) == "string"))'
schema_ok() { jq -e "$SCHEMA" $1 > /dev/null; }
count_live() {
  schema_ok $1 || stop "app list schema ($1)"
  jq --arg a $APP '[.[] | select(.description == $a and ((.state | ascii_downcase) != "stopped"))] | length' $1
}
printf '[{"description":"%s","state":"ephemeral"},{"description":"other","state":"deployed"}]' $APP > $LOGDIR/fx_live.json
printf '[{"description":"%s","state":"stopped"}]' $APP > $LOGDIR/fx_stopped.json
printf '[]' > $LOGDIR/fx_empty.json
printf '[{"Description":"%s","State":"ephemeral"}]' $APP > $LOGDIR/fx_upper.json
printf '[{"description":null,"state":"ephemeral"}]' > $LOGDIR/fx_null.json
printf '{"description":"%s","state":"ephemeral"}' $APP > $LOGDIR/fx_notarray.json
[[ $(count_live $LOGDIR/fx_live.json) == 1 && $(count_live $LOGDIR/fx_stopped.json) == 0 && $(count_live $LOGDIR/fx_empty.json) == 0 ]] \
  || stop "live-app filter self-test"
for fx in fx_upper fx_null fx_notarray; do
  if schema_ok $LOGDIR/$fx.json; then stop "schema self-test: $fx passed"; fi
done
echo "OK live-app filter self-test: live=1 stopped=0 empty=0; upper-case keys, null description and non-array rejected"

gate_zero_live() {
  "$M" app list --json > $LOGDIR/apps_before_$1.json || stop "app list failed before $1"
  local n
  n=$(count_live $LOGDIR/apps_before_$1.json) || exit 1
  [[ $n == 0 ]] || stop "$n live app(s) named $APP before $1"
  echo "OK 0 live apps named $APP before $1 ($(date))"
}

LAST_RC=0
launch() {  # label log args... : FOREGROUND call; watcher prints the APP_ID; 15 s pause after the call returns
  local label=$1 log=$2
  shift 2
  [[ ! -e $log ]] || stop "log $log exists"
  gate_zero_live $label
  : > $log
  (
    while [[ ! -e $log.returned ]]; do
      id=$(grep -o 'ap-[A-Za-z0-9]\{10,\}' $log 2> /dev/null | head -1 || true)
      if [[ -n $id ]]; then echo "APP_ID $label $id $(date)"; exit 0; fi
      sleep 1
    done
    id=$(grep -o 'ap-[A-Za-z0-9]\{10,\}' $log 2> /dev/null | head -1 || true)
    if [[ -n $id ]]; then echo "APP_ID $label $id $(date) (seen after the call returned)"; exit 0; fi
    echo "APP_ID $label none $(date)" >&2
    exit 1
  ) &
  local wpid=$! wrc=0
  LAST_RC=0
  (cd $D && "$M" run "$@" >> $log 2>&1) || LAST_RC=$?
  touch $log.returned
  wait $wpid || wrc=$?
  echo "RETURNED $label rc=$LAST_RC $(date)"
  [[ $wrc == 0 ]] || stop "APP_ID watcher for $label failed or found no app id (watcher rc=$wrc)"
  sleep 15
}

result_line() {  # exactly one structured RESULT line and no CRASH line
  local log=$1 n
  n=$(grep -c '^HONALL RESULT ' $log || true)
  [[ $n == 1 ]] || stop "$log: $n HONALL RESULT lines"
  [[ $(grep -c '^HONALL CRASH' $log || true) == 0 ]] || stop "$log: CRASH line"
  grep '^HONALL RESULT ' $log
}
cnt() { grep -c "$2" $1 || true; }
exact() { [[ $(cnt $1 "$2") == $3 ]] || stop "${1:t}: count of [$2] != $3"; }

# ---- 3. (1) hash-shards: the manifest pin (F3, G, H)
[[ ! -e $MANIFEST && ! -e $MANIFEST_CAND ]] || stop "manifest path exists"
launch hash-shards $LOGDIR/hash.log check_honest_all.py --mode hash-shards --manifest-out $MANIFEST_CAND
line=$(result_line $LOGDIR/hash.log) || exit 1
echo "hash-shards rc=$LAST_RC predicted=0 $line"
[[ $LAST_RC == 0 ]] || stop "hash-shards exit $LAST_RC != 0 (the provisional ${MANIFEST_CAND:t} is not published)"
for g in G-HASH-RESULT G-HASH-NPZ G-HASH-SHARDS G-HASH-MYW; do exact $LOGDIR/hash.log "^HONALL GATE $g PASS" 1; done
exact $LOGDIR/hash.log '^HONALL GATE .* STOP' 0
[[ -f $MANIFEST_CAND ]] || stop "provisional manifest file missing"
MSHA=$(shasum -a 256 $MANIFEST_CAND | cut -c1-64)
print -r -- ${line#HONALL RESULT } | jq -e --arg s $MSHA '.exit == 0 and .manifest_sha256 == $s' > /dev/null || stop "manifest sha != RESULT manifest_sha256"
# every check passed: publish (no-clobber move), freeze, re-check the published file's sha
chmod a-w $MANIFEST_CAND
[[ ! -e $MANIFEST ]] || stop "manifest path appeared"
mv -n $MANIFEST_CAND $MANIFEST
[[ -f $MANIFEST && ! -e $MANIFEST_CAND && ! -w $MANIFEST ]] || stop "manifest publish"
[[ $(shasum -a 256 $MANIFEST | cut -c1-64) == $MSHA ]] || stop "published manifest sha256 != $MSHA"
echo "OK manifest pin: $MANIFEST sha256 $MSHA (published after every check, frozen a-w)"

# ---- 4. (2) the 5 CPU toys, exact signatures (H6 as amended by H2: UNITNEG 16)
typeset -A WANT
WANT=(test 0 test-replay 0 test-replay-fail-fold 1 test-replay-fail-producer 1 test-replay-fail-manifest 1)
for mode in test test-replay test-replay-fail-fold test-replay-fail-producer test-replay-fail-manifest; do
  log=$LOGDIR/toy_$mode.log
  launch $mode $log check_honest_all.py --mode $mode --stamp $STAMP
  line=$(result_line $log) || exit 1
  echo "$mode rc=$LAST_RC predicted=$WANT[$mode] $line"
  [[ $LAST_RC == $WANT[$mode] ]] || stop "$mode exit $LAST_RC != predicted $WANT[$mode]"
  print -r -- ${line#HONALL RESULT } | jq -e --argjson x $WANT[$mode] '.exit == $x' > /dev/null || stop "$mode RESULT exit field"
  if [[ $mode == test ]]; then
    exact $log '^HONALL TOYBOUNDARY cases_as_predicted=8/8 TRUE$' 1
    exact $log '^HONALL FIT_VERDICT .* PASS_STRICT ' 4
    exact $log '^HONALL TOYREF weights_within_1e-6 TRUE$' 1
  else
    exact $log '^HONALL UNITNEG .* AS_PREDICTED$' 16
    exact $log '^HONALL UNITNEG .* UNEXPECTED$' 0
    exact $log '^HONALL GATE G-TOY-FIRST-PASS STOP' 0
  fi
  case $mode in
    test-replay)               exact $log '^HONALL REPLAY .* MATCH mode=stored tier=tight ' 4
                               exact $log '^HONALL NEGCTRL N1 .* FAIL_AS_PREDICTED$' 4
                               exact $log '^HONALL FIT_VERDICT .* PASS_STRICT ' 4
                               exact $log '^HONALL GATE .* STOP' 0 ;;
    test-replay-fail-fold)     exact $log '^HONALL GATE .* STOP' 1
                               exact $log "^HONALL GATE G-LAYOUT STOP parsed=3/4 record_folds_0to4=TRUE reasons=\['command|L48 simh_vs_b:keys'\]$" 1 ;;
    test-replay-fail-producer) exact $log '^HONALL REPLAY .* MISMATCH ' 4
                               exact $log '^HONALL GATE .* STOP' 0 ;;
    test-replay-fail-manifest) exact $log '^HONALL GATE .* STOP' 1
                               exact $log '^HONALL GATE G-MANIFEST-FILE STOP' 1 ;;
  esac
  echo "OK $mode: exit and exact signature as predicted"
done
echo "honest_all phase 1 done: $(date). Manifest pin $MSHA at $MANIFEST. Next: commit the manifest and these logs, then phase 2 on a new ARMED line."
