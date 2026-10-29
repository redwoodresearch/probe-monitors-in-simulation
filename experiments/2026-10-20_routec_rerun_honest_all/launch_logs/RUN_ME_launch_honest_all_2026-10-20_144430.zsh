#!/bin/zsh
# honest_all route C rerun, step 4 (launch). Run by 0d ONLY on 2e's relayed ARMED line:  zsh RUN_ME_launch_honest_all_<stamp>.zsh "<ARMED_AT>"
# Sequence: assert the committed launch worktree -> materialize pinned inputs -> fail-closed dry-resolve -> 7 toy modes (CPU) with predicted
# exits and signatures -> the real run (`modal run --detach`). Before every modal call: 0 live apps of the name. A fresh log per call, no tee.
# Prints only OK/STOP lines, counts, app ids with `date`, and each call's structured `HONALL RESULT` line (exit, Volume path, sha256).
# STOPs (exit 1) on any unexpected exit or signature. Check with `zsh -n` only.
# WHY strictly sequential (2e, 14:3x): refresh3's guard (armed 2026-10-20 14:29:28 PDT, cap $3.00, up to 8 sequential apps) binds the first
# NEW app id it sees and needs exactly ONE new id at each sighting (it polls every ~5 s). So the 8 `"$M" run` calls run one after another
# in the FOREGROUND (only the last, the real run, uses --detach); each waits for 0 live apps of the name; after each call returns the script
# sleeps 15 s before the next live-0 check. A background watcher only greps the call's own log so the APP_ID line prints at once.
set -euo pipefail
ARMED_AT=${1:?usage: zsh RUN_ME_launch_honest_all.zsh "<ARMED_AT as relayed by 2e>"}
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
WT=/Users/aswinsurya/probe-shift-wt-rerun-honest-all-2026-10-20
BR=aswin/routec-rerun-honest-all
COMMIT=e29be485a6daf6f23df61f62a5bea59ddf8f6503
REL=experiments/2026-10-20_routec_rerun_honest_all
D=$WT/$REL
ITEM=/Users/aswinsurya/probe-shift-reruns-2026-10-20/honest_all
APP=probe-shift_aswin-surya_sub-rerun_honall_kill-2026-11-19
CODE_BLOB=7cd2061aa5a8690f7224d32f2cf6e207d080f797
CODE_SHA=2a551d383224c0e4ff03ad4571c1885074620ac2148c6469a97d09c5036ec159
MAT_SHA=9de41d9b1b1f84fbc19c5c41f834e6c5e79713ca87a071ad02df7f82182ec1a3
DRY_SHA=27c04a5194c312ced0ee26d3583436bb409de702756f4c1959716ea7d2c60fb2
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
LOGDIR=$ITEM/launch_$STAMP

stop() { echo "STOP: $1 ($(date))" >&2; exit 1; }   # stderr, so a STOP raised inside $(...) stays visible (Codex launch review)
echo "honest_all launch: ARMED_AT=[$ARMED_AT] stamp=$STAMP start=$(date)"

# ---- 1. the committed launch worktree: attached branch, HEAD == origin == the reviewed commit, committed code blob, clean
[[ -d $WT ]] || stop "worktree missing"
[[ $(git -C $WT symbolic-ref --short HEAD) == $BR ]] || stop "worktree not attached to $BR"
git -C $WT fetch -q origin $BR
[[ $(git -C $WT rev-parse HEAD) == $(git -C $WT rev-parse origin/$BR) ]] || stop "HEAD != origin/$BR"
[[ $(git -C $WT rev-parse HEAD) == $COMMIT ]] || stop "HEAD != $COMMIT"
ST=$(git -C $WT status --porcelain) || stop "git status failed"   # git's own exit first: empty output on failure must not read as clean
[[ -z $ST ]] || stop "worktree not clean"
[[ $(git -C $WT rev-parse HEAD:$REL/check_honest_all.py) == $CODE_BLOB ]] || stop "committed code blob"
[[ $(git -C $WT hash-object $D/check_honest_all.py) == $CODE_BLOB ]] || stop "working-copy code blob"
[[ $(shasum -a 256 $D/check_honest_all.py | cut -c1-64) == $CODE_SHA ]] || stop "code sha256"
[[ $(shasum -a 256 $D/materialize_pinned.zsh | cut -c1-64) == $MAT_SHA ]] || stop "materialize_pinned sha256"
[[ $(shasum -a 256 $D/dryresolve.zsh | cut -c1-64) == $DRY_SHA ]] || stop "dryresolve sha256"
echo "OK worktree: $BR @ $COMMIT == origin, clean, code blob $CODE_BLOB"

# ---- 2. fresh log folder
[[ ! -e $LOGDIR ]] || stop "log folder exists"
mkdir -p $LOGDIR

# ---- 3. live-app gate (lower-case keys .description/.state; schema-checked; fails closed), self-tested on fixtures first
SCHEMA='type == "array" and all(.[]; type == "object" and ((.description | type) == "string") and ((.state | type) == "string"))'
schema_ok() { jq -e "$SCHEMA" $1 > /dev/null; }   # every entry: string description and string state (Codex: null must fail closed)
count_live() {  # file -> count of apps named $APP not in state "stopped"; STOP on schema failure
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

gate_zero_live() {  # label
  "$M" app list --json > $LOGDIR/apps_before_$1.json || stop "app list failed before $1"
  local n
  n=$(count_live $LOGDIR/apps_before_$1.json) || exit 1
  [[ $n == 0 ]] || stop "$n live app(s) named $APP before $1"
  echo "OK 0 live apps named $APP before $1 ($(date))"
}

# ---- 4. one modal call, in the FOREGROUND: fresh log; a watcher prints the APP_ID with `date` the moment it appears; exit code kept;
#         15 s pause after the call returns (the guard's ~5 s polling must see the app stopped before the next one starts)
LAST_RC=0
launch() {  # label log args...
  local label=$1 log=$2
  shift 2
  [[ ! -e $log ]] || stop "log $log exists"
  gate_zero_live $label
  : > $log
  (  # watcher: reads only this call's log, prints one APP_ID line, exits when the id appears or the call has returned
    while [[ ! -e $log.returned ]]; do
      id=$(grep -o 'ap-[A-Za-z0-9]\{10,\}' $log 2> /dev/null | head -1 || true)
      if [[ -n $id ]]; then echo "APP_ID $label $id $(date)"; exit 0; fi
      sleep 1
    done
    id=$(grep -o 'ap-[A-Za-z0-9]\{10,\}' $log 2> /dev/null | head -1 || true)
    echo "APP_ID $label ${id:-none} $(date) (seen after the call returned)"
  ) &
  local wpid=$!
  LAST_RC=0
  (cd $D && "$M" run "$@" >> $log 2>&1) || LAST_RC=$?   # foreground: the script waits here until the call returns
  touch $log.returned
  wait $wpid || true
  echo "RETURNED $label rc=$LAST_RC $(date)"
  sleep 15
}

result_line() {  # log -> the single structured line; STOP unless exactly one and no CRASH line
  local log=$1 n
  n=$(grep -c '^HONALL RESULT ' $log || true)
  [[ $n == 1 ]] || stop "$log: $n HONALL RESULT lines"
  [[ $(grep -c '^HONALL CRASH' $log || true) == 0 ]] || stop "$log: CRASH line"
  grep '^HONALL RESULT ' $log
}

has() { [[ $(grep -c "$2" $1 || true) -ge 1 ]]; }      # log pattern: at least one line
cnt() { grep -c "$2" $1 || true; }                     # log pattern: count of lines

# ---- 5. pinned inputs (committed script; sha-checked against the pre-stated shas) and the fail-closed dry-resolve
zsh $D/materialize_pinned.zsh $LOGDIR/pinned > $LOGDIR/materialize.log 2>&1 || { cat $LOGDIR/materialize.log; stop "materialize"; }
cat $LOGDIR/materialize.log
zsh $D/dryresolve.zsh $LOGDIR/pinned $LOGDIR/real.log $STAMP > $LOGDIR/dryresolve.log 2>&1 || { cat $LOGDIR/dryresolve.log; stop "dry-resolve"; }
cat $LOGDIR/dryresolve.log
[[ $(tail -1 $LOGDIR/dryresolve.log) == "DRY-RESOLVE PASS" ]] || stop "dry-resolve did not PASS"

# ---- 6. the 7 toy modes (CPU, orchestrate_toy), each with its predicted exit and signature (PRESTATED v2 section 7; addenda B8, C1, D2)
typeset -A WANT
WANT=(test 0 test-ruling 0 test-fail-value 1 test-fail-pilot 1 test-fail-sha 1 test-fail-token 1 test-fail-defgap 1)
for mode in test test-ruling test-fail-value test-fail-pilot test-fail-sha test-fail-token test-fail-defgap; do
  log=$LOGDIR/toy_$mode.log
  launch $mode $log check_honest_all.py --mode $mode --stamp $STAMP
  line=$(result_line $log) || exit 1
  echo "$mode rc=$LAST_RC predicted=$WANT[$mode] $line"
  [[ $LAST_RC == $WANT[$mode] ]] || stop "$mode exit $LAST_RC != predicted $WANT[$mode]"
  print -r -- ${line#HONALL RESULT } | jq -e --argjson x $WANT[$mode] '.exit == $x' > /dev/null || stop "$mode RESULT exit field"
  has $log '^HONALL TOYBOUNDARY cases_as_predicted=8/8 TRUE$' || stop "$mode: TOYBOUNDARY not 8/8"
  case $mode in
    test)             [[ $(cnt $log '^HONALL FIT_VERDICT .* PASS_STRICT ') == 4 ]] || stop "test: PASS_STRICT != 4"
                      has $log '^HONALL TOYREF weights_within_1e-6 TRUE$' || stop "test: TOYREF" ;;
    test-ruling)      [[ $(cnt $log '^HONALL FIT_VERDICT .* PASS_RULING ') == 4 ]] || stop "test-ruling: PASS_RULING != 4"
                      [[ $(cnt $log '^HONALL REPLAY .* MATCH mode=stored tier=tight ') == 4 ]] || stop "test-ruling: tight replay != 4"
                      has $log '^HONALL TOYREF weights_within_1e-6 TRUE$' || stop "test-ruling: TOYREF" ;;
    test-fail-value)  has $log '^HONALL REPLAY command|L48 simh_vs_b MISMATCH ' || stop "test-fail-value: no REPLAY MISMATCH on the perturbed cell"
                      has $log '^HONALL REFIT command|L48 simh_vs_b tight_tier_1e-3=MISMATCH ' || stop "test-fail-value: no tight-tier REFIT MISMATCH" ;;
    test-fail-pilot)  has $log '^HONALL GATE G-PILOT STOP' || stop "test-fail-pilot: G-PILOT did not STOP" ;;
    test-fail-sha)    has $log '^HONALL GATE G-STAGED STOP' || stop "test-fail-sha: G-STAGED did not STOP" ;;
    test-fail-token)  has $log '^HONALL DIAG D1 rows_per_run command|L48 FALSE' || stop "test-fail-token: D1 not FALSE" ;;
    test-fail-defgap) [[ $(cnt $log '^HONALL REPLAY .* MATCH mode=') == 4 ]] || stop "test-fail-defgap: REPLAY MATCH != 4"
                      [[ $(cnt $log '^HONALL REFIT .* tight_tier_1e-3=MISMATCH ') -ge 1 ]] || stop "test-fail-defgap: no tight-tier REFIT MISMATCH"
                      np=$(( $(cnt $log '^HONALL FIT_VERDICT .* PASS_STRICT ') + $(cnt $log '^HONALL FIT_VERDICT .* PASS_RULING ') ))
                      [[ $np -lt 4 ]] || stop "test-fail-defgap: all 4 FIT_VERDICT passed" ;;
  esac
  echo "OK $mode: exit and signature as predicted"
done

# ---- 7. the real run (detached; the local client stays attached to print the structured lines)
HEAD_SHA=$(git -C $WT rev-parse HEAD)
launch real $LOGDIR/real.log --detach check_honest_all.py --mode real --pinned $LOGDIR/pinned --git-head $HEAD_SHA --stamp $STAMP
line=$(result_line $LOGDIR/real.log) || exit 1
echo "real rc=$LAST_RC $line"
echo "honest_all launch done: $(date). Exit 0 = every check passed; 1 = a check failed (the HONALL lines in $LOGDIR/real.log say which)."
exit $LAST_RC
