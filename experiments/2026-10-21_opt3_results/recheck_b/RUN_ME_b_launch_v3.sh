#!/bin/bash
# RUN_ME_b_launch_v3.sh: recheck (b) v3, an independent in-container re-comparison of option (3) record A (PRESTATEMENT_b_v3.md, a
# delta note on PRESTATEMENT_b_v2.md). Supersedes the v1 and v2 launch RUN_MEs (kept unchanged; do not run them). Launches ONE Modal CPU
# job (recheck_b_v3.py, one .remote() call) with the named modal 1.6.1 binary. Run by the coordinator only, after: review PASS ->
# 09's key-check -> SPEND row -> guard ARMED.
#   bash RUN_ME_b_launch_v3.sh 'Claude-Session: https://claude.ai/code/session_<id>'
# Writes only in this folder: runs/<STAMP>/ (record A gz, the node list and the templates, extracted by git show; the raw launch log,
# mode 600) and, via the local entrypoint and only on success, results/recheck_b_v3_out_<STAMP>.json (create-only). Both Volumes are
# mounted read-only; nothing is written to any Volume. To the terminal it prints only its own gate lines (keys, shas, counts) and the
# RECHECK_B_STATUS line if it matches the closed grammar; never a value. The raw log keeps all of modal's stdout / stderr for diagnosis:
# framework diagnostics outside the script's control (image build, mounts, argument deserialization, the Modal runtime) MAY carry
# exception text there, so the log is mode 600 and is NOT relayed (PRESTATEMENT_b_v3 change 3; disclosed residual risk).
# Exits with modal run's exit code (0 ok; 2 remote stop code; 3 local fail code; others from modal itself).
# Cost (bc): rate 2 x $0.0000131 + 4 x $0.00000222 = $0.00003508 / s. Estimate (~120 s billed, a guess): $0.0042096 raw, $0.00547248
# with +30%. Hard bound per attempt: (600 + 180 + 2) s x $0.00003508 = $0.02743256 (< $5). Arm ceiling 2.5 x $0.00547248 = $0.0136812.
set -u
set -o pipefail
set -o noclobber

die() { echo "STOP: $*"; exit 1; }

HERE=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_recheck/b_incontainer
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
M_DIST=/Users/aswinsurya/.cache/probe-shift-modal161/lib/python3.12/site-packages/modal-1.6.1.dist-info
APP=probe-shift_aswin-surya_sub-opt3recheck_b_kill-2026-11-20
A1_VOL=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
PD_VOL=probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13
SHA_PRE=e299e560b7a01b70482bf30201cdb850d0da79ba95090bc939f6768b5a8148b4   # PRESTATEMENT_b_v3.md (delta note)
SHA_PRE_BASE=36e12f41e39dc7ffbbf3a3ed6fdaa09a13932880ba017b38419bbdcd0851d710   # PRESTATEMENT_b_v2.md (its base)
SHA_PY=2c8d70d3f9f2fb654e85df01c49b573c72ae9881165dc552e3bd619b63a5be91    # recheck_b_v3.py
SHA_RES=eaa09e1e09177eebf6f4d7b3ef05a0876b41254eaeabcf0ab4378fd08275c06f   # RUN_ME_b_results_v3.sh
REC_COMMIT=a936a5d9762763f9feed351aebdeeec2d8114f21
REC_FILE=routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz
REC_PATH=experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/$REC_FILE
REC_GZ=191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865
REC_JS=54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a
FRZ_COMMIT=22504a59619c4bfdd280a540b9306b15a3d89c9a
FRZ_DIR=experiments/2026-10-16_cleanroom_pooled_repro/opt3
NODES_FILE=opt3_nodes_A.txt
NODES_SHA=8f6f42788ceb2d48a375e8dec21106ee8859debca7f75e41d5a15c1ecad5702b
TPL_FILE=opt3_record_path_templates.tsv
TPL_SHA=58785b3064ac5a5d21666bc9f9b7365e5f85d02a8812734296196270c3b5649c
SUM_COMMIT=7d787550b999042ff60aeb3b208a3669ca38413f
SUM_PATH=experiments/2026-10-21_opt3_results/opt3_summary_1792613723775_20261021_202531_7e4f75e4.json
SUM_SHA=8283c81a8ab584297a9ea118619a85790dfadbbaedcbc86fcfa1e0230efce333
VALUES_REL=opt3_A_20261021_202527_acbf3404
VALUES_DIR=/o/$VALUES_REL
NS_SHA=9493e2b24caa34b42b8a04e5501ee1dc8213697a3fcaa10a1fdbbe9d24c88f4d
CMP_SHA=6d81961657f7075f50c008b0921aaed6403501b70d2b951768684e9c6556a208
PD_JSON=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_inputs/opt3_perdraw_npz_AB.json
PD_JSON_SHA=28c8302c7c2128e1d802ba9e7684d45fa25dddb631104a045fceb9c801aab111
PD_DIR=xfer/fpr507prep
PD_BASE=alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz
PD_REL=$PD_DIR/$PD_BASE
PD_SHA=5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1

# 0. Location, the trailer argument, tools, the named modal binary.
[ "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" = "$HERE" ] || die "run this file from its frozen folder $HERE"
[ $# -eq 1 ] || die "usage: bash RUN_ME_b_launch_v3.sh 'Claude-Session: https://claude.ai/code/session_<id>'"
TRAILER=$1
case $TRAILER in *$'\n'*|*$'\r'*) die "the trailer holds a line break";; esac
tre='^Claude-Session: https://claude\.ai/code/session_[A-Za-z0-9]+$'
[[ $TRAILER =~ $tre ]] || die "the trailer is not one Claude-Session line"
(( ${#APP} <= 63 )) || die "the app name is longer than 63 characters"
for t in git jq shasum gunzip awk grep sort tr; do command -v "$t" >/dev/null || die "tool $t missing"; done
[ -x "$M" ] || die "the named modal binary $M is missing or not executable"
[ -d "$M_DIST" ] || die "the named binary's venv does not hold modal-1.6.1.dist-info"

# 1. Frozen files: pinned shas (the v3 note and its v2 base), SHA256SUMS_v3 (which also covers this file), read-only modes, and the
#    bound / input / sanitizer lines.
pin() { local h; h=$(shasum -a 256 "$HERE/$1" | awk '{print $1}') || die "shasum $1"; [ "$h" = "$2" ] || die "$1: sha256 differs from the pinned value"; }
pin PRESTATEMENT_b_v3.md "$SHA_PRE"
pin PRESTATEMENT_b_v2.md "$SHA_PRE_BASE"
pin recheck_b_v3.py "$SHA_PY"
pin RUN_ME_b_results_v3.sh "$SHA_RES"
( cd "$HERE" && shasum -a 256 -c SHA256SUMS_v3 >/dev/null 2>&1 ) || die "SHA256SUMS_v3 check failed (a frozen file changed)"
names=$(awk '{print $2}' "$HERE/SHA256SUMS_v3" | LC_ALL=C sort | tr '\n' ' ') || die "read SHA256SUMS_v3"
[ "$names" = "PRESTATEMENT_b_v3.md RUN_ME_b_launch_v3.sh RUN_ME_b_results_v3.sh recheck_b_v3.py " ] || die "SHA256SUMS_v3 lists other files"
for f in PRESTATEMENT_b_v3.md PRESTATEMENT_b_v2.md recheck_b_v3.py RUN_ME_b_launch_v3.sh RUN_ME_b_results_v3.sh SHA256SUMS_v3; do
  [ -f "$HERE/$f" ] || die "$f missing"
  [ -w "$HERE/$f" ] && die "$f is writable (not frozen)"
done
S=$HERE/recheck_b_v3.py
one() { local c; c=$(grep -c -F -- "$1" "$S"); [ "$c" = 1 ] || die "recheck_b_v3.py: want exactly one line holding '$1' (found ${c:-none})"; }
none() { local c; c=$(grep -c -F -- "$1" "$S"); [ "$c" = 0 ] || die "recheck_b_v3.py: want no line holding '$1' (found ${c:-none})"; }
one "APP = \"$APP\""
one "A1_VOL = \"$A1_VOL\""
one "PD_VOL = \"$PD_VOL\""
one 'TIMEOUT, STARTUP, SCALEDOWN = 600, 180, 2'
one 'CPU, MEM_MIB = 2, 4096'
one 'cpu=(CPU, CPU), memory=(MEM_MIB, MEM_MIB), timeout=TIMEOUT, startup_timeout=STARTUP, scaledown_window=SCALEDOWN,'
one 'retries=0, max_containers=1, volumes={"/o": a1_vol.read_only(), "/p": pd_vol.read_only()})'
one '@app.function('
one 'recheck.remote('
one 'uv_pip_install("numpy==2.3.3", "threadpoolctl==3.6.0")'
one 'deadline = float(meta_in["launch_epoch"]) + TIMEOUT + STARTUP'
one 'with out_local.open("xb") as fh:'
one "PRESTATEMENT_SHA256 = \"$SHA_PRE\""
one "PRESTATEMENT_BASE_SHA256 = \"$SHA_PRE_BASE\""
one "VALUES_DIR = \"$VALUES_DIR\""
one "NS_SHA = \"$NS_SHA\""
one "CMP_SHA = \"$CMP_SHA\""
one "PD_PATH = \"/p/$PD_REL\""
one "PD_SHA = \"$PD_SHA\""
one "REC_FILE = \"$REC_FILE\""
one "REC_GZ_SHA = \"$REC_GZ\""
one "REC_JSON_SHA = \"$REC_JS\""
one "NODES_FILE, NODES_SHA = \"$NODES_FILE\", \"$NODES_SHA\""
one "TPL_FILE, TPL_SHA = \"$TPL_FILE\", \"$TPL_SHA\""
one 'D, MIN_FINITE, TOL, SCALE = 10000, 9500, 1e-9, 1.0'
one 'N_NODES, N_GAP, N_SAMPLE, SAMPLE_SEED, NC_DELTA = 60, 28, 2000, 20261021, 1e-6'
one 'def _guarded(body, *args):'                                           # v2: the sanitizing boundary (PRESTATEMENT_b_v2 §3a)
one 'except BaseException as e:  # noqa: BLE001  (the sanitizing boundary)'
one 'return _guarded(_body, rec_gz, nodes_b, tpl_b, meta_in)'              # the remote function's whole body
one 'r1 = _guarded(_fixture_value_error, 0.123456789)'                     # the in-container sanitizer fixture
one 'REMOTE_FALLBACK_CODE = "stop:sanitizer:Other"'                        # v3 change 2: the remote constant fallback
one 'return {"status": "stop", "code": REMOTE_FALLBACK_CODE}'
one 'def _local_guarded(body, *args):'                                     # v3 change 1: the local twin
one 'except BaseException as e:  # noqa: BLE001  (the local sanitizing boundary)'
one 'rc, line = _local_guarded(_local_body, stamp, run_dir, trailer)'      # main's whole body goes through it
one 'FALLBACK_LINE = "RECHECK_B_STATUS fail app_id=unknown code=fail:sanitizer:Other"'  # v3 change 2: the local constant
one 'return 3, FALLBACK_LINE'
one 'fx = _local_guarded(_local_fixture_value_error, 0.123456789)'         # v3 change 4: the local fixture
one 'with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):'
one '        sys.exit(rc)  # outside every except block'                   # main exits without an exception chain
none 'raise SystemExit'
none '.commit('
none 'create_if_missing'
none 'import cleanroom'
none 'from cleanroom'
none 'import traceback'
none 'print_exc'
none 'format_exc'
c=$(grep -c -F 'print(' "$S")
[ "$c" = 2 ] || die "recheck_b_v3.py: want exactly 2 print lines (the body's status line, main's failure line) (found ${c:-none})"
c=$(grep -c -F 'print(line, flush=True)' "$S")
[ "$c" = 2 ] || die "recheck_b_v3.py: the 2 print lines must both print the status line only (found ${c:-none})"
c=$(grep -c -E '^[[:space:]]+raise[[:space:]]*$' "$S")
[ "$c" = 0 ] || die "recheck_b_v3.py: a bare re-raise is present (found ${c:-none})"
c=$(grep -c -E '\.(map|starmap|spawn|spawn_map|for_each)\(' "$S")
[ "$c" = 0 ] || die "recheck_b_v3.py: a fan-out call is present (found ${c:-none})"

# 2. Commits.
for c in "$REC_COMMIT" "$FRZ_COMMIT" "$SUM_COMMIT"; do
  full=$(git -C "$REPO" rev-parse --verify --quiet "$c^{commit}") || die "commit $c not found in $REPO"
  [ "$full" = "$c" ] || die "commit $c resolves elsewhere"
done

# 3. Provenance (jq -e booleans; output discarded; no value is printed): a1's summary and 2e's per-draw pointer.
h=$(git -C "$REPO" show "$SUM_COMMIT:$SUM_PATH" | shasum -a 256 | awk '{print $1}') || die "git show the summary"
[ "$h" = "$SUM_SHA" ] || die "a1's summary sha256 differs"
git -C "$REPO" show "$SUM_COMMIT:$SUM_PATH" | jq -e --arg vd "$VALUES_DIR" --arg ns "$NS_SHA" --arg cs "$CMP_SHA" '
  [.workers[] | select(.record == "A")] as $w
  | ($w | length) == 1 and $w[0].values_dir == $vd and $w[0].values_sha256_node_series == $ns and $w[0].values_sha256_compare == $cs
    and $w[0].d1_identified == 1 and $w[0].c0_scale == "fraction" and $w[0].c2_method == "linear" and $w[0].n_nodes == 60
    and $w[0].verdict == "INTERVALS REPRODUCED (EXACT)" and .launch_id == "1792613723775"' >/dev/null 2>&1 \
  || die "a1's summary: record A's values_dir / value shas / d1 / scale / method / node count / verdict differ from the pre-statement"
h=$(shasum -a 256 "$PD_JSON" | awk '{print $1}') || die "shasum the per-draw pointer"
[ "$h" = "$PD_JSON_SHA" ] || die "opt3_perdraw_npz_AB.json sha256 differs"
jq -e --arg p "$PD_REL" --arg s "$PD_SHA" --arg v "$PD_VOL" '.A_npz == {path: $p, sha256: $s, volume: $v}' "$PD_JSON" >/dev/null 2>&1 \
  || die "opt3_perdraw_npz_AB.json A_npz differs from the pre-statement"

# 4. Fresh run folder and log (exclusive); the results folder; the output name must not exist yet.
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S) || die "date"
NAME=recheck_b_v3_out_$STAMP.json
RUN_DIR=$HERE/runs/$STAMP
mkdir -p "$HERE/runs" "$HERE/results" || die "mkdir runs / results"
[ -e "$HERE/results/$NAME" ] && die "results/$NAME already exists"
mkdir "$RUN_DIR" || die "the run folder runs/$STAMP exists"
LOG=$RUN_DIR/launch_$STAMP.log
( umask 077 && : > "$LOG" ) || die "the log exists"   # v3 change 3: the raw log is mode 600 before anything is written to it
chmod 600 "$LOG" || die "chmod 600 the raw log"
[ "$(stat -f %Lp "$LOG")" = 600 ] || die "the raw log is not mode 600"
{ echo "launch $STAMP PT (v3)"; echo "trailer $TRAILER"; echo "app $APP"; echo "modal_binary $M"; echo "prestatement_sha256 $SHA_PRE";
  echo "prestatement_base_sha256 $SHA_PRE_BASE";
  echo "script_sha256 $SHA_PY"; echo "results_runme_sha256 $SHA_RES"; echo "record $REC_COMMIT:$REC_PATH gz $REC_GZ json $REC_JS";
  echo "frozen $FRZ_COMMIT:$FRZ_DIR/{$NODES_FILE $NODES_SHA, $TPL_FILE $TPL_SHA}"; echo "summary $SUM_COMMIT:$SUM_PATH $SUM_SHA";
  echo "a1 $A1_VOL:$VALUES_REL node_series.npz $NS_SHA compare.json $CMP_SHA"; echo "perdraw $PD_VOL:$PD_REL $PD_SHA"; } >> "$LOG"
echo "provenance gates passed; run folder runs/$STAMP"

# 5. Inputs shipped as .remote() bytes: git show into the run folder, then sha256 (the record also decompressed; never parsed here).
extract() {  # commit path out_name sha [json_sha]
  local out=$RUN_DIR/$3 h
  git -C "$REPO" cat-file -e "$1:$2" || die "$3 is not in $1"
  git -C "$REPO" show "$1:$2" > "$out" || die "git show $3"
  h=$(shasum -a 256 "$out" | awk '{print $1}') || die "shasum $3"
  [ "$h" = "$4" ] || die "$3: sha256 differs"
  if [ $# -eq 5 ]; then
    h=$(gunzip -c "$out" | shasum -a 256 | awk '{print $1}') || die "gunzip $3"
    [ "$h" = "$5" ] || die "$3: decompressed sha256 differs"
  fi
  echo "input $3: sha256 match" >> "$LOG"
}
extract "$REC_COMMIT" "$REC_PATH" "$REC_FILE" "$REC_GZ" "$REC_JS"
extract "$FRZ_COMMIT" "$FRZ_DIR/$NODES_FILE" "$NODES_FILE" "$NODES_SHA"
extract "$FRZ_COMMIT" "$FRZ_DIR/$TPL_FILE" "$TPL_FILE" "$TPL_SHA"
echo "inputs extracted and sha-checked (record gz + json, node list, templates)"

# 6. Dry-resolve the Volume paths the container reads (read-only listings; names go to the log only; no spend).
listed() {  # listing basename -> how many lines are that basename or end with /basename
  printf '%s\n' "$1" | awk -v f="$2" '$0 == f || substr($0, length($0) - length(f)) == "/" f {c++} END {print c+0}'
}
L=$(cd "$HERE" && "$M" volume ls "$A1_VOL" "$VALUES_REL" 2>>"$LOG") || die "modal volume ls on a1's OUT_VOL failed (see the log)"
printf '%s\n' "$L" >> "$LOG"
for f in node_series.npz compare.json; do
  [ "$(listed "$L" "$f")" = 1 ] || die "$f is not listed exactly once under $VALUES_REL on $A1_VOL"
done
L=$(cd "$HERE" && "$M" volume ls "$PD_VOL" "$PD_DIR" 2>>"$LOG") || die "modal volume ls on the per-draw Volume failed (see the log)"
[ "$(listed "$L" "$PD_BASE")" = 1 ] || die "the per-draw npz is not listed exactly once under $PD_DIR on $PD_VOL"
echo "volume paths resolved: $PD_VOL:$PD_REL" >> "$LOG"
echo "volume paths resolved (a1's node_series.npz and compare.json; the per-draw npz)"

# 7. Live-app gate: lower-case keys .description / .state, schema-checked, fails closed. Fixture self-tests run first.
live_count() {
  jq -r --arg app "$APP" 'if type == "array" and length > 0 and all(.[]; type == "object" and has("state") and has("description")
      and (.state | type) == "string" and (.description | type) == "string")
    then ([.[] | select(.description == $app and .state != "stopped")] | length | tostring) else "SCHEMA" end'
}
gate_open() { local n; n=$(live_count 2>/dev/null) || return 1; [ "$n" = "0" ]; }
expect() {  # want(open|stop) label fixture
  local got
  if printf '%s' "$3" | gate_open; then got=open; else got=stop; fi
  [ "$got" = "$1" ] || die "live-app gate self-test '$2': got $got, want $1"
  echo "live-app gate self-test '$2': $got (want $1)" >> "$LOG"
}
expect open clean "[{\"app_id\":\"ap-t1\",\"description\":\"other-app\",\"state\":\"deployed\",\"tasks\":0,\"created_at\":\"t\",\"stopped_at\":null},{\"app_id\":\"ap-t2\",\"description\":\"$APP\",\"state\":\"stopped\",\"tasks\":0,\"created_at\":\"t\",\"stopped_at\":\"t\"}]"
expect stop live_ephemeral "[{\"app_id\":\"ap-t1\",\"description\":\"$APP\",\"state\":\"ephemeral\",\"tasks\":1,\"created_at\":\"t\",\"stopped_at\":null}]"
expect stop live_detached "[{\"app_id\":\"ap-t1\",\"description\":\"$APP\",\"state\":\"ephemeral (detached)\",\"tasks\":1,\"created_at\":\"t\",\"stopped_at\":null}]"
expect stop live_deployed "[{\"app_id\":\"ap-t1\",\"description\":\"$APP\",\"state\":\"deployed\",\"tasks\":0,\"created_at\":\"t\",\"stopped_at\":null}]"
expect stop uppercase_keys "[{\"App ID\":\"ap-t1\",\"Description\":\"$APP\",\"State\":\"ephemeral\"}]"
expect stop missing_description '[{"app_id":"ap-t1","state":"ephemeral"}]'
expect stop empty_list '[]'
expect stop not_a_list "{\"description\":\"$APP\",\"state\":\"stopped\"}"
expect stop garbage 'not json'
APPS=$(cd "$HERE" && "$M" app list --json 2>>"$LOG") || die "modal app list failed (see the log)"
printf '%s' "$APPS" | gate_open || die "live-app gate: $APP is live, or the app list failed its schema check"
echo "live-app gate: 0 live apps named $APP" >> "$LOG"
echo "live-app gate: 0 live apps named $APP (9 fixture self-tests passed)"

# 8. Launch with the named binary (no --detach); stdout and stderr go to the fresh mode-600 raw log only (never to the terminal).
echo "LAUNCH $STAMP PT: app $APP; raw log runs/$STAMP/launch_$STAMP.log (mode 600; not to be relayed)"
( cd "$HERE" && "$M" run recheck_b_v3.py --stamp "$STAMP" --run-dir "$RUN_DIR" --trailer "$TRAILER" ) >> "$LOG" 2>&1
rc=$?
echo "modal run exit code $rc at $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"

# 9. Only the structured status line is read from the raw log, and it is printed only if it matches its closed grammar:
#    RECHECK_B_STATUS ok app_id=<id> name=<name> sha256=<sha> local=results/<name>
#    RECHECK_B_STATUS stop app_id=<id> code=stop:<stage>:<ExceptionClass>   or   ...code=stop:<stage>:Stop:<reason>
#    RECHECK_B_STATUS fail app_id=<id> code=fail:<stage>:<ExceptionClass>   or   ...code=fail:<stage>:Stop:<reason>
NRE=${NAME//./\\.}
AID='(ap-[A-Za-z0-9]+|unknown)'
CODE_TAIL='[a-z0-9_]+:([A-Za-z]+|Stop:[a-z0-9_]+)'
st_re="^RECHECK_B_STATUS (ok app_id=$AID name=$NRE sha256=[0-9a-f]{64} local=results/$NRE|stop app_id=$AID code=stop:$CODE_TAIL|fail app_id=$AID code=fail:$CODE_TAIL)\$"
n=$(grep -c '^RECHECK_B_STATUS ' "$LOG")
line=""
if [ "$n" = 1 ]; then
  l1=$(grep '^RECHECK_B_STATUS ' "$LOG") || die "grep status line"
  if [[ $l1 =~ $st_re ]]; then
    line=$l1
    echo "$line"
  else
    echo "the RECHECK_B_STATUS line does not match its closed grammar (not shown)"
  fi
else
  echo "RECHECK_B_STATUS lines in the raw log: ${n:-none} (want 1)"
fi
if [ "$rc" -ne 0 ]; then
  echo "The job did not finish cleanly (exit $rc). Raw log (mode 600; for diagnosis only; not to be relayed): $LOG"
  exit "$rc"
fi
[ -n "$line" ] || die "exit 0 but no well-formed RECHECK_B_STATUS line in the raw log"
set -- $line
[ "${1:-}" = RECHECK_B_STATUS ] && [ "${2:-}" = ok ] || die "exit 0 but the status line is not 'ok'"
[ "${4:-}" = "name=$NAME" ] && [ "${6:-}" = "local=results/$NAME" ] || die "the status line names a different output"
lsha=${5:-}
[ -f "$HERE/results/$NAME" ] || die "the local output results/$NAME is missing"
h=$(shasum -a 256 "$HERE/results/$NAME" | awk '{print $1}') || die "shasum output"
[ "sha256=$h" = "$lsha" ] || die "the local output's sha256 differs from the status line"
echo "OUTPUT results/$NAME sha256 $h"
echo "NEXT: bash $HERE/RUN_ME_b_results_v3.sh $NAME"
exit 0
