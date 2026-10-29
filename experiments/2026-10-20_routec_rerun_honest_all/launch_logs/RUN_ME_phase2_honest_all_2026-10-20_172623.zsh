#!/bin/zsh
# honest_all route A, PHASE 2 (addenda F3, G, H5, H2): the replay-only run on the real stored shards, with the phase-1 manifest pin
# 7d52ffc3… (phase 1 rc 0, 17:16:50–17:21:13). Run by 0d ONLY on 2e's relayed ARMED line, after 09's own launch OK:
#   zsh RUN_ME_phase2_honest_all_<stamp>.zsh "<ARMED_AT>"
# 09 (17:1x, relayed by 2e): the H100 replay must be DETACHED with ONE remote call, so a client drop cannot lose it. main() makes exactly one
# client-side call, orchestrate_replay.remote (CPU), which calls replay_fit.remote (H100) once and commits the record to the output Volume
# (<stamp>_replay/) before it returns. `modal run --detach` keeps the app running if this client drops.
# The manifest commit is pinned by content, not by its sha (it is frozen before that commit exists): HEAD == origin; HEAD^ == 82ea3e926
# (the v12 code commit); BASE..HEAD changes exactly the 11 manifest-commit paths, each with its pinned blob (SPEND.md the only
# modification), plus at most .gitleaksignore with additions only (the sanctioned fingerprint route, if the hook blocks the first try).
# Sequence: assert all that, clean, code blob v12, the committed manifest's sha256 == the pin -> live-app filter self-test -> materialize the
# pinned inputs (committed script) -> fail-closed dry-resolve -> 0 live apps -> the one detached call, FOREGROUND, fresh log, no tee, the
# APP_ID printed to stdout with `date` as soon as it appears (a watcher failure or no id STOPs), 15 s pause after it returns.
# Prints only OK/STOP lines, counts, APP_ID/RETURNED lines and the structured RESULT line (exit, Volume path, sha256). No REPLAY or
# FIT_VERDICT value is predicted (H5): the script asserts the 13 gates PASS, N1 FAIL_AS_PREDICTED 4/4, no UNITNEG lines (toy-only), no
# CRASH, and prints the counts of the REPLAY/FIT_VERDICT outcome words. Bound (H7 terms, bc): replay $1.53471096 + orchestrator
# $0.03371188 = $1.56842284. Check with `zsh -n` only.
set -euo pipefail
ARMED_AT=${1:?usage: zsh RUN_ME_phase2_honest_all.zsh "<ARMED_AT as relayed by 2e>"}
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
WT=/Users/aswinsurya/probe-shift-wt-rerun-honest-all-2026-10-20
BR=aswin/routec-rerun-honest-all
BASE=82ea3e9265fec1fec12a6ce17aa447068a7dfada
REL=experiments/2026-10-20_routec_rerun_honest_all
D=$WT/$REL
ITEM=/Users/aswinsurya/probe-shift-reruns-2026-10-20/honest_all
APP=probe-shift_aswin-surya_sub-rerun_honall_kill-2026-11-19
CODE_BLOB=177f19b5a88116eb8ebc04b80833aaec351eb146
CODE_SHA=b6c45a573054d05f728843eeca319782973494b6a8f03c8c08c77f2817a34f2f
MAT_SHA=9de41d9b1b1f84fbc19c5c41f834e6c5e79713ca87a071ad02df7f82182ec1a3
DRY_SHA=27c04a5194c312ced0ee26d3583436bb409de702756f4c1959716ea7d2c60fb2
MANIFEST_REL=routeA/replay_manifest_2026-10-20_171650.json
MSHA=7d52ffc317f6a27b134c6bbc5d85f935f34c8ed68253ef0d929e048ab3846fd5
typeset -A WANTBLOB   # the manifest commit's 11 paths (under $REL) -> git blob of the frozen source
WANTBLOB=(routeA/replay_manifest_2026-10-20_171650.json                      cbe95e8743e82c04ed261629b670704f94bf25d6
          launch_logs/phase1_2026-10-20_171650/hash.log                      1439f1e02741d6832aea69530b16d808a649e80b
          launch_logs/phase1_2026-10-20_171650/toy_test.log                  d2ced519724436b350d936fcd39bd7b283d8e9d4
          launch_logs/phase1_2026-10-20_171650/toy_test-replay.log           74f33fb2c3e7a483c7e0a7b12141be98af3dd76d
          launch_logs/phase1_2026-10-20_171650/toy_test-replay-fail-fold.log 4d226a2b96afec5102d448d20d7e3b29b524bedc
          launch_logs/phase1_2026-10-20_171650/toy_test-replay-fail-producer.log 3c5c906a0452056bf93b2731e54f1bdbeb42180a
          launch_logs/phase1_2026-10-20_171650/toy_test-replay-fail-manifest.log 4031757b91bf1dd4409c09d1d3c28aee4f10d0ed
          launch_logs/run_phase1_171650.log                                  dffda7fa70a82c91aa5f5681e76d92542a5426a9
          launch_logs/RUN_ME_phase1_honest_all_2026-10-20_171216.zsh         f94ac7b32785e58690094f89249e9e44aef5785a
          NOTE_routeA_phase1_2026-10-20.md                                   eb1869857d4c22b36c71f168866dd41bedb6bbba
          SPEND.md                                                           8de4ffdaa165dac3a51137822d81179470750c48)
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
LOGDIR=$ITEM/phase2_$STAMP

stop() { echo "STOP: $1 ($(date))" >&2; exit 1; }
echo "honest_all phase 2: ARMED_AT=[$ARMED_AT] stamp=$STAMP start=$(date)"
[[ ${#WANTBLOB} == 11 ]] || stop "pinned blob list"

# ---- 1. the committed worktree and the manifest commit (pinned by content)
[[ -d $WT ]] || stop "worktree missing"
[[ $(git -C $WT symbolic-ref --short HEAD) == $BR ]] || stop "worktree not attached to $BR"
git -C $WT fetch -q origin $BR
LH=$(git -C $WT rev-parse HEAD) || stop "rev-parse HEAD failed"
OH=$(git -C $WT rev-parse origin/$BR) || stop "rev-parse origin/$BR failed"
[[ -n $LH && $LH == $OH ]] || stop "HEAD != origin/$BR"
PH=$(git -C $WT rev-parse HEAD^) || stop "rev-parse HEAD^ failed"
[[ $PH == $BASE ]] || stop "HEAD^ != $BASE (the manifest commit must sit directly on the v12 code commit)"
ST=$(git -C $WT status --porcelain) || stop "git status failed"
[[ -z $ST ]] || stop "worktree not clean"
CN=$(git -C $WT diff --name-only $BASE HEAD) || stop "git diff BASE HEAD failed"
CM=$(git -C $WT diff --name-only --diff-filter=M $BASE HEAD) || stop "git diff --diff-filter=M failed"
CX=$(git -C $WT diff --name-only --diff-filter=DRTUXB $BASE HEAD) || stop "git diff --diff-filter=DRTUXB failed"
[[ -z $CX ]] || stop "the manifest commit deletes, renames or retypes a path"
typeset -a CHANGED
CHANGED=(${(f)CN})
NG=0
for n in $CHANGED; do
  if [[ $n == .gitleaksignore ]]; then
    NG=1
    NS=$(git -C $WT diff --numstat $BASE HEAD -- .gitleaksignore) || stop "numstat .gitleaksignore failed"
    [[ ${${(ps:\t:)NS}[2]} == 0 ]] || stop ".gitleaksignore lost lines"
    continue
  fi
  [[ $n == $REL/* ]] || stop "changed path outside $REL"
  r=${n#$REL/}
  [[ -n ${WANTBLOB[$r]-} ]] || stop "unexpected changed path: $r"
  CB=$(git -C $WT rev-parse HEAD:$n) || stop "rev-parse HEAD:$r failed"
  [[ $CB == ${WANTBLOB[$r]} ]] || stop "committed blob != pinned: $r"
done
[[ ${#CHANGED} == $(( 11 + NG )) ]] || stop "changed path count ${#CHANGED} != $(( 11 + NG ))"
if (( NG )); then
  [[ $CM == ".gitleaksignore"$'\n'"$REL/SPEND.md" ]] || stop "modified paths other than SPEND.md and .gitleaksignore"
else
  [[ $CM == "$REL/SPEND.md" ]] || stop "modified paths other than SPEND.md"
fi
[[ $(git -C $WT rev-parse HEAD:$REL/check_honest_all.py) == $CODE_BLOB ]] || stop "committed code blob"
[[ $(git -C $WT hash-object $D/check_honest_all.py) == $CODE_BLOB ]] || stop "working-copy code blob"
[[ $(shasum -a 256 $D/check_honest_all.py | cut -c1-64) == $CODE_SHA ]] || stop "code sha256"
[[ $(shasum -a 256 $D/materialize_pinned.zsh | cut -c1-64) == $MAT_SHA ]] || stop "materialize_pinned sha256"
[[ $(shasum -a 256 $D/dryresolve.zsh | cut -c1-64) == $DRY_SHA ]] || stop "dryresolve sha256"
MF=$D/$MANIFEST_REL
[[ -f $MF ]] || stop "committed manifest missing"
[[ $(git -C $WT hash-object $MF) == ${WANTBLOB[$MANIFEST_REL]} ]] || stop "working-copy manifest != pinned blob"
[[ $(shasum -a 256 $MF | cut -c1-64) == $MSHA ]] || stop "manifest sha256 != the phase-1 pin"
echo "OK worktree: $BR @ $LH == origin, parent $BASE, clean; manifest commit pinned by content ($(( 11 + NG )) paths); code blob $CODE_BLOB; manifest sha256 $MSHA"
[[ ! -e $LOGDIR ]] || stop "log folder exists"
mkdir -p $LOGDIR

# ---- 2. live-app gate (schema-checked, fails closed), self-tested on fixtures first (as the reviewed launchers)
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

# ---- 3. pinned inputs (committed script, sha-checked) and the fail-closed dry-resolve (as the reviewed 14:44 launcher)
zsh $D/materialize_pinned.zsh $LOGDIR/pinned > $LOGDIR/materialize.log 2>&1 || { cat $LOGDIR/materialize.log; stop "materialize"; }
cat $LOGDIR/materialize.log
zsh $D/dryresolve.zsh $LOGDIR/pinned $LOGDIR/replay.log $STAMP > $LOGDIR/dryresolve.log 2>&1 || { cat $LOGDIR/dryresolve.log; stop "dry-resolve"; }
cat $LOGDIR/dryresolve.log
[[ $(tail -1 $LOGDIR/dryresolve.log) == "DRY-RESOLVE PASS" ]] || stop "dry-resolve did not PASS"

# ---- 4. the replay: ONE detached call (09), foreground client
launch replay $LOGDIR/replay.log --detach check_honest_all.py --mode replay --pinned $LOGDIR/pinned --git-head $LH --stamp $STAMP \
  --manifest $MF --manifest-sha $MSHA
NR=$(cnt $LOGDIR/replay.log '^HONALL RESULT ')
[[ $NR == 1 ]] || stop "the client returned (rc=$LAST_RC) with $NR RESULT lines. The DETACHED app may still be running: do NOT relaunch; tell 2e (check the app list for $APP and the Volume folder ${STAMP}_replay/ by key)"
line=$(result_line $LOGDIR/replay.log) || exit 1
echo "replay rc=$LAST_RC $line"
print -r -- ${line#HONALL RESULT } | jq -e --argjson x $LAST_RC '.exit == $x' > /dev/null || stop "RESULT exit field != rc $LAST_RC"
# counts first (printed whatever the outcome), then H5's asserts: the 13 gates PASS once each; no STOP; N1 FAIL_AS_PREDICTED 4/4; no UNITNEG
for g in G-PIN G-MANIFEST-FILE G-RESULT-LOCAL G-STEMS-LOCAL G-CLASS G-FOLD G-MANIFEST G-RESULT G-STEMS G-WIDTH G-LAYOUT G-MYW G-MYFIT; do
  echo "COUNT [GATE $g PASS] $(cnt $LOGDIR/replay.log "^HONALL GATE $g PASS")"
done
for w in 'GATE .* STOP' 'NEGCTRL N1 .* FAIL_AS_PREDICTED$' 'UNITNEG ' \
         'REPLAY .* MATCH mode=stored tier=tight ' 'REPLAY .* MATCH mode=stored tier=fp32 ' 'REPLAY .* MISMATCH ' 'REPLAY .* UNRESOLVED$' \
         'NEGCTRL N2 .* FAIL_AS_PREDICTED$' 'NEGCTRL .* UNEXPECTED_PASS$' 'FIT_VERDICT .* PASS_STRICT ' 'FIT_VERDICT .* PASS_RULING ' \
         'FIT_VERDICT .* UNRESOLVED ' 'FIT_VERDICT .* STOP_FITTER_CHECK ' 'FIT_VERDICT .* RECIPE_NOT_FULLY_REGISTERED '; do
  echo "COUNT [$w] $(cnt $LOGDIR/replay.log "^HONALL $w")"
done
for g in G-PIN G-MANIFEST-FILE G-RESULT-LOCAL G-STEMS-LOCAL G-CLASS G-FOLD G-MANIFEST G-RESULT G-STEMS G-WIDTH G-LAYOUT G-MYW G-MYFIT; do
  exact $LOGDIR/replay.log "^HONALL GATE $g PASS" 1
done
exact $LOGDIR/replay.log '^HONALL GATE .* STOP' 0
exact $LOGDIR/replay.log '^HONALL NEGCTRL N1 .* FAIL_AS_PREDICTED$' 4
exact $LOGDIR/replay.log '^HONALL UNITNEG ' 0
echo "OK 13 gates PASS, 0 STOP, N1 FAIL_AS_PREDICTED 4/4, UNITNEG 0"
echo "honest_all phase 2 done: $(date). rc=$LAST_RC (0 = every check passed; 1 = a check failed, the HONALL lines in ${LOGDIR:t}/replay.log say which)."
exit $LAST_RC
