#!/bin/bash
# RUN_ME_b_launch_v4.sh: recheck (b) v4, the in-container re-comparison of option (3) parametrized by record (PRESTATEMENT_b_v4.md, a
# delta note on PRESTATEMENT_b_v3.md, itself a delta note on PRESTATEMENT_b_v2.md). v3's launch RUN_ME (RUN_ME_b_launch_v3.sh
# 9fd28eff...) copied, with every record-specific constant in ONE case table (PRESTATEMENT_b_v4 §2) and a1's summary given as
# arguments (§3.8). Launches ONE Modal CPU job (recheck_b_v4.py, one .remote() call) with the named modal 1.6.1 binary. Run by the
# coordinator only, after: review PASS -> 09's key-check -> SPEND row -> guard ARMED.
#   bash RUN_ME_b_launch_v4.sh '<Claude-Session line>' <A|B> <opt3-results commit> <summary path> <summary sha256>
#   record B (PRESTATEMENT_b_v4 §8):
#   bash RUN_ME_b_launch_v4.sh 'Claude-Session: https://claude.ai/code/session_<id>' B fcac6b803e6cd96d3e0be6fb85e2b95846f90d2a \
#     experiments/2026-10-21_opt3_results/opt3_summary_1792628072873_20261022_001844_b4f7d17e.json \
#     6ccd7619c6e09d6a108fe2c4dd9487429b1eca6f308fd106155b057c31b88f2b
# The summary arguments must equal the table's for that record. The summary is fetched by git show, its full sha256 compared, and
# values_dir / the two value shas read by jq (keys only) and passed to the script.
# Writes only in this folder: runs/<STAMP>/ (the record gz, the node list and the templates, extracted by git show; the raw launch log,
# mode 600) and, via the local entrypoint and only on success, results/recheck_b_v4_out_<record>_<STAMP>.json (create-only). Both
# Volumes are mounted read-only; nothing is written to any Volume. To the terminal it prints only its own gate lines (keys, shas,
# counts) and the RECHECK_B_STATUS line if it matches the closed grammar; never a value. The raw log keeps all of modal's stdout /
# stderr for diagnosis: framework diagnostics outside the script's control (image build, mounts, argument deserialization, the Modal
# runtime) MAY carry exception text there, so the log is mode 600 and is NOT relayed (PRESTATEMENT_b_v3 change 3; disclosed residual
# risk). Exits with modal run's exit code (0 ok; 2 remote stop code; 3 local fail code; others from modal itself).
# Cost (bc): rate 2 x $0.0000131 + 4 x $0.00000222 = $0.00003508 / s. Estimate (~180 s billed, a guess): $0.0063144 raw, $0.00820872
# with +30%. Hard bound per attempt: (1200 + 180 + 2) s x $0.00003508 = $0.04848056 (< $5). Arm ceiling 2.5 x $0.00820872 = $0.0205218.
set -u
set -o pipefail
set -o noclobber

die() { echo "STOP: $*"; exit 1; }

HERE=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_recheck/b_incontainer_B
V3=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_recheck/b_incontainer   # holds the v3 and v2 notes (this note's bases); read only
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
M_DIST=/Users/aswinsurya/.cache/probe-shift-modal161/lib/python3.12/site-packages/modal-1.6.1.dist-info
APP=probe-shift_aswin-surya_sub-opt3recheck_b_kill-2026-11-20
A1_VOL=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
PD_VOL=probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13
SHA_PRE=1cbcac690f4ae005c7661dbe1cd987fb8d3b222ad509261b97d66c833346eeff        # PRESTATEMENT_b_v4.md (delta note)
SHA_PRE_BASE=e299e560b7a01b70482bf30201cdb850d0da79ba95090bc939f6768b5a8148b4   # PRESTATEMENT_b_v3.md (its base, in $V3)
SHA_PRE_ROOT=36e12f41e39dc7ffbbf3a3ed6fdaa09a13932880ba017b38419bbdcd0851d710   # PRESTATEMENT_b_v2.md (v3's base, in $V3)
SHA_PY=7c9b6ad83a56435a9eb41eaf190ee0f2250fe3d4c1ef366457cf0162447d28df         # recheck_b_v4.py
SHA_RES=b2a28fb3ad34cb95c0f56d12ca8febca07e612958aeb2c77336ea4b104722fb3        # RUN_ME_b_results_v4.sh
REC_DIR=experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records
FRZ_COMMIT=22504a59619c4bfdd280a540b9306b15a3d89c9a
FRZ_DIR=experiments/2026-10-16_cleanroom_pooled_repro/opt3
TPL_FILE=opt3_record_path_templates.tsv
TPL_SHA=58785b3064ac5a5d21666bc9f9b7365e5f85d02a8812734296196270c3b5649c
SUM_DIR=experiments/2026-10-21_opt3_results
PD_JSON=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_inputs/opt3_perdraw_npz_AB.json
PD_JSON_SHA=28c8302c7c2128e1d802ba9e7684d45fa25dddb631104a045fceb9c801aab111

# 0. Location, the arguments, the per-record table, tools, the named modal binary.
[ "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" = "$HERE" ] || die "run this file from its frozen folder $HERE"
[ $# -eq 5 ] || die "usage: bash RUN_ME_b_launch_v4.sh '<Claude-Session line>' <A|B> <opt3-results commit> <summary path> <summary sha256>"
TRAILER=$1
REC=$2
ARG_SUM_COMMIT=$3
ARG_SUM_PATH=$4
ARG_SUM_SHA=$5
case $TRAILER in *$'\n'*|*$'\r'*) die "the trailer holds a line break";; esac
tre='^Claude-Session: https://claude\.ai/code/session_[A-Za-z0-9]+$'
[[ $TRAILER =~ $tre ]] || die "the trailer is not one Claude-Session line"
# THE per-record table (PRESTATEMENT_b_v4 §2): the only place a record's constants live in this file. A's values are v3's.
case $REC in
  A)
    REC_COMMIT=a936a5d9762763f9feed351aebdeeec2d8114f21
    REC_FILE=routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz
    REC_GZ=191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865
    REC_JS=54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a
    NODES_FILE=opt3_nodes_A.txt
    NODES_SHA=8f6f42788ceb2d48a375e8dec21106ee8859debca7f75e41d5a15c1ecad5702b
    PD_REL=xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz
    PD_SHA=5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1
    N_NODES=60; N_GAP=28; DRAWS=10000; MIN_FINITE=9500
    SUM_COMMIT=7d787550b999042ff60aeb3b208a3669ca38413f
    SUM_PATH=$SUM_DIR/opt3_summary_1792613723775_20261021_202531_7e4f75e4.json
    SUM_SHA=8283c81a8ab584297a9ea118619a85790dfadbbaedcbc86fcfa1e0230efce333
    LAUNCH_ID=1792613723775
    ;;
  B)
    REC_COMMIT=f4e21b0f76acde02d1ae4e7408028d6eaeb559d8
    REC_FILE=routec_B_alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652.json.gz
    REC_GZ=5d5c58f76f7c0cfda0c09666ea8a1651e055a486a48bece8dafcb8657ad2c435
    REC_JS=ef5c765716ee338a1e8d207a820eecd5c083ba1d51262ff72361d6d254bf6ad7
    NODES_FILE=opt3_nodes_fig4.txt
    NODES_SHA=7d984dcf9107a43db4ae9188aaab693280f55502b011e445247e718d95a1b35c
    PD_REL=xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652_perdraw.npz
    PD_SHA=8a2a2afda1ff5271695191be04911fcd9ecb4bf7199d5715fe6a526c9b2ffaa3
    N_NODES=240; N_GAP=180; DRAWS=10000; MIN_FINITE=9500
    SUM_COMMIT=fcac6b803e6cd96d3e0be6fb85e2b95846f90d2a
    SUM_PATH=$SUM_DIR/opt3_summary_1792628072873_20261022_001844_b4f7d17e.json
    SUM_SHA=6ccd7619c6e09d6a108fe2c4dd9487429b1eca6f308fd106155b057c31b88f2b
    LAUNCH_ID=1792628072873
    ;;
  *) die "the record must be A or B";;
esac
REC_PATH=$REC_DIR/$REC_FILE
PD_DIR=${PD_REL%/*}
PD_BASE=${PD_REL##*/}
[ "$ARG_SUM_COMMIT" = "$SUM_COMMIT" ] || die "the summary commit is not record $REC's pre-stated one"
[ "$ARG_SUM_PATH" = "$SUM_PATH" ] || die "the summary path is not record $REC's pre-stated one"
[ "$ARG_SUM_SHA" = "$SUM_SHA" ] || die "the summary sha256 is not record $REC's pre-stated one"
(( ${#APP} <= 63 )) || die "the app name is longer than 63 characters"
for t in git jq shasum gunzip awk grep sort tr stat; do command -v "$t" >/dev/null || die "tool $t missing"; done
[ -x "$M" ] || die "the named modal binary $M is missing or not executable"
[ -d "$M_DIST" ] || die "the named binary's venv does not hold modal-1.6.1.dist-info"

# 1. Frozen files: pinned shas (the v4 note and its v3 / v2 bases), SHA256SUMS_v4 (which also covers this file), read-only modes, and
#    the bound / input / sanitizer / record-table lines of the script.
pin() { local h; h=$(shasum -a 256 "$1" | awk '{print $1}') || die "shasum $1"; [ "$h" = "$2" ] || die "$1: sha256 differs from the pinned value"; }
pin "$HERE/PRESTATEMENT_b_v4.md" "$SHA_PRE"
pin "$V3/PRESTATEMENT_b_v3.md" "$SHA_PRE_BASE"
pin "$V3/PRESTATEMENT_b_v2.md" "$SHA_PRE_ROOT"
pin "$HERE/recheck_b_v4.py" "$SHA_PY"
pin "$HERE/RUN_ME_b_results_v4.sh" "$SHA_RES"
( cd "$HERE" && shasum -a 256 -c SHA256SUMS_v4 >/dev/null 2>&1 ) || die "SHA256SUMS_v4 check failed (a frozen file changed)"
names=$(awk '{print $2}' "$HERE/SHA256SUMS_v4" | LC_ALL=C sort | tr '\n' ' ') || die "read SHA256SUMS_v4"
[ "$names" = "PRESTATEMENT_b_v4.md RUN_ME_b_launch_v4.sh RUN_ME_b_results_v4.sh recheck_b_v4.py " ] || die "SHA256SUMS_v4 lists other files"
for f in "$HERE/PRESTATEMENT_b_v4.md" "$V3/PRESTATEMENT_b_v3.md" "$V3/PRESTATEMENT_b_v2.md" "$HERE/recheck_b_v4.py" \
         "$HERE/RUN_ME_b_launch_v4.sh" "$HERE/RUN_ME_b_results_v4.sh" "$HERE/SHA256SUMS_v4"; do
  [ -f "$f" ] || die "$f missing"
  [ -w "$f" ] && die "$f is writable (not frozen)"
done
S=$HERE/recheck_b_v4.py
one() { local c; c=$(grep -c -F -- "$1" "$S"); [ "$c" = 1 ] || die "recheck_b_v4.py: want exactly one line holding '$1' (found ${c:-none})"; }
none() { local c; c=$(grep -c -F -- "$1" "$S"); [ "$c" = 0 ] || die "recheck_b_v4.py: want no line holding '$1' (found ${c:-none})"; }
one "APP = \"$APP\""
one "A1_VOL = \"$A1_VOL\""
one "PD_VOL = \"$PD_VOL\""
one 'TIMEOUT, STARTUP, SCALEDOWN = 1200, 180, 2'
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
one "PRESTATEMENT_ROOT_SHA256 = \"$SHA_PRE_ROOT\""
one "TPL_FILE, TPL_SHA = \"$TPL_FILE\", \"$TPL_SHA\""
one 'TOL, SCALE = 1e-9, 1.0'
one 'N_SAMPLE, SAMPLE_SEED, NC_DELTA = 2000, 20261021, 1e-6'
one 'RESULTS_DIR = Path("/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_recheck/b_incontainer_B/results")'
# v4: the script's RECORDS block for this record must equal this table, line by line and in order (PRESTATEMENT_b_v4 §2).
want_blk=$(printf '%s\n' "        rec_file=\"$REC_FILE\"," "        rec_gz_sha=\"$REC_GZ\"," "        rec_json_sha=\"$REC_JS\"," \
  "        nodes_file=\"$NODES_FILE\"," "        nodes_sha=\"$NODES_SHA\"," "        pd_path=\"/p/$PD_REL\"," "        pd_sha=\"$PD_SHA\"," \
  "        n_nodes=$N_NODES," "        n_gap=$N_GAP," "        draws=$DRAWS," "        min_finite=$MIN_FINITE,") || die "build the record block"
got_blk=$(awk -v h="    \"$REC\": dict(" 'on && $0 == "    )," {exit} on {print} $0 == h {on = 1}' "$S") || die "read the script's record block"
[ "$got_blk" = "$want_blk" ] || die "recheck_b_v4.py RECORDS[\"$REC\"] differs from this RUN_ME's table for record $REC"
one "    \"$REC\": dict("
c=$(grep -c -E '^    "[AB]": dict\($' "$S")
[ "$c" = 2 ] || die "recheck_b_v4.py: want exactly 2 record blocks, A and B (found ${c:-none})"
# v4: the record parameter and the passed-in values (PRESTATEMENT_b_v4 §3.1-§3.3).
one 'rec = str(meta_in["record"])'
one 'R = RECORDS[rec]'
one 'N_NODES, N_GAP, D, MIN_FINITE = R["n_nodes"], R["n_gap"], R["draws"], R["min_finite"]'
one 'VALUES_DIR, NS_SHA, CMP_SHA = str(meta_in["values_dir"]), str(meta_in["ns_sha"]), str(meta_in["cmp_sha"])'
one 'meta_in["name"] != f"recheck_b_v4_out_{rec}_{stamp}.json"'
one 'name = f"recheck_b_v4_out_{record}_{stamp}.json"'
c=$(grep -c -F '{rec}\t' "$S")
[ "$c" = 6 ] || die "recheck_b_v4.py: want the record letter in exactly 6 template-row lookups (found ${c:-none})"
none '"A\t'
none "'A\\t"
none 'b_incontainer/'
none 'recheck_b_v3_out'
# v3's sanitizer and boundary lines (PRESTATEMENT_b_v2 §3a; PRESTATEMENT_b_v3 changes 1-4).
one 'def _guarded(body, *args):'
one 'except BaseException as e:  # noqa: BLE001  (the sanitizing boundary)'
one 'return _guarded(_body, rec_gz, nodes_b, tpl_b, meta_in)'
one 'r1 = _guarded(_fixture_value_error, 0.123456789)'
one 'REMOTE_FALLBACK_CODE = "stop:sanitizer:Other"'
one 'return {"status": "stop", "code": REMOTE_FALLBACK_CODE}'
one 'def _local_guarded(body, *args):'
one 'except BaseException as e:  # noqa: BLE001  (the local sanitizing boundary)'
one 'rc, line = _local_guarded(_local_body, stamp, run_dir, trailer, record, values_dir, ns_sha, cmp_sha)'
one 'FALLBACK_LINE = "RECHECK_B_STATUS fail app_id=unknown code=fail:sanitizer:Other"'
one 'return 3, FALLBACK_LINE'
one 'fx = _local_guarded(_local_fixture_value_error, 0.123456789)'
one 'with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):'
one '        sys.exit(rc)  # outside every except block'
none 'raise SystemExit'
none '.commit('
none 'create_if_missing'
none 'import cleanroom'
none 'from cleanroom'
none 'import traceback'
none 'print_exc'
none 'format_exc'
c=$(grep -c -F 'print(' "$S")
[ "$c" = 2 ] || die "recheck_b_v4.py: want exactly 2 print lines (the body's status line, main's failure line) (found ${c:-none})"
c=$(grep -c -F 'print(line, flush=True)' "$S")
[ "$c" = 2 ] || die "recheck_b_v4.py: the 2 print lines must both print the status line only (found ${c:-none})"
c=$(grep -c -E '^[[:space:]]+raise[[:space:]]*$' "$S")
[ "$c" = 0 ] || die "recheck_b_v4.py: a bare re-raise is present (found ${c:-none})"
c=$(grep -c -E '\.(map|starmap|spawn|spawn_map|for_each)\(' "$S")
[ "$c" = 0 ] || die "recheck_b_v4.py: a fan-out call is present (found ${c:-none})"

# 2. Commits (full shas that resolve to themselves).
for c in "$REC_COMMIT" "$FRZ_COMMIT" "$SUM_COMMIT"; do
  full=$(git -C "$REPO" rev-parse --verify --quiet "$c^{commit}") || die "commit $c not found in $REPO"
  [ "$full" = "$c" ] || die "commit $c resolves elsewhere"
done

# 3. Provenance (jq -e booleans with output discarded, then jq -r for three keys; no value is printed): a1's summary for this record,
#    and 2e's per-draw pointer.
h=$(git -C "$REPO" show "$SUM_COMMIT:$SUM_PATH" | shasum -a 256 | awk '{print $1}') || die "git show the summary"
[ "$h" = "$SUM_SHA" ] || die "a1's summary sha256 differs from the given / pre-stated one"
vre="^/o/opt3_${REC}_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}\$"
git -C "$REPO" show "$SUM_COMMIT:$SUM_PATH" | jq -e --arg r "$REC" --arg lid "$LAUNCH_ID" --argjson nn "$N_NODES" --arg vre "$vre" '
  [.workers[] | select(.record == $r)] as $w
  | ($w | length) == 1 and .launch_id == $lid
    and ($w[0].values_dir | type) == "string" and ($w[0].values_dir | test($vre))
    and ($w[0].values_sha256_node_series | type) == "string" and ($w[0].values_sha256_node_series | test("^[0-9a-f]{64}$"))
    and ($w[0].values_sha256_compare | type) == "string" and ($w[0].values_sha256_compare | test("^[0-9a-f]{64}$"))
    and $w[0].d1_identified == 1 and $w[0].c0_scale == "fraction" and $w[0].c2_method == "linear" and $w[0].n_nodes == $nn
    and $w[0].verdict == "INTERVALS REPRODUCED (EXACT)"' >/dev/null 2>&1 \
  || die "a1's summary: record $REC's worker / launch_id / values_dir / value shas / d1 / scale / method / node count / verdict differ from the pre-statement"
VALS=$(git -C "$REPO" show "$SUM_COMMIT:$SUM_PATH" | jq -r --arg r "$REC" \
  '[.workers[] | select(.record == $r)][0] | "\(.values_dir) \(.values_sha256_node_series) \(.values_sha256_compare)"') \
  || die "read values_dir / value shas from the summary"
read -r VALUES_DIR NS_SHA CMP_SHA EXTRA <<< "$VALS"
[ -z "${EXTRA:-}" ] || die "the summary's values_dir / value shas did not split into exactly three fields"
[[ $VALUES_DIR =~ $vre ]] || die "values_dir is not /o/opt3_${REC}_<8 digits>_<6 digits>_<8 hex>"
[[ $NS_SHA =~ ^[0-9a-f]{64}$ && $CMP_SHA =~ ^[0-9a-f]{64}$ ]] || die "a value sha is not 64 lower-case hex"
VALUES_REL=${VALUES_DIR#/o/}
h=$(shasum -a 256 "$PD_JSON" | awk '{print $1}') || die "shasum the per-draw pointer"
[ "$h" = "$PD_JSON_SHA" ] || die "opt3_perdraw_npz_AB.json sha256 differs"
jq -e --arg k "${REC}_npz" --arg p "$PD_REL" --arg s "$PD_SHA" --arg v "$PD_VOL" '.[$k] == {path: $p, sha256: $s, volume: $v}' "$PD_JSON" \
  >/dev/null 2>&1 || die "opt3_perdraw_npz_AB.json ${REC}_npz differs from the pre-statement"

# 4. Fresh run folder and log (exclusive); the results folder; the output name must not exist yet.
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S) || die "date"
NAME=recheck_b_v4_out_${REC}_$STAMP.json
RUN_DIR=$HERE/runs/$STAMP
mkdir -p "$HERE/runs" "$HERE/results" || die "mkdir runs / results"
[ -e "$HERE/results/$NAME" ] && die "results/$NAME already exists"
mkdir "$RUN_DIR" || die "the run folder runs/$STAMP exists"
LOG=$RUN_DIR/launch_$STAMP.log
( umask 077 && : > "$LOG" ) || die "the log exists"   # v3 change 3: the raw log is mode 600 before anything is written to it
chmod 600 "$LOG" || die "chmod 600 the raw log"
[ "$(stat -f %Lp "$LOG")" = 600 ] || die "the raw log is not mode 600"
{ echo "launch $STAMP PT (v4) record $REC"; echo "trailer $TRAILER"; echo "app $APP"; echo "modal_binary $M"; echo "prestatement_sha256 $SHA_PRE";
  echo "prestatement_base_sha256 $SHA_PRE_BASE"; echo "prestatement_root_sha256 $SHA_PRE_ROOT";
  echo "script_sha256 $SHA_PY"; echo "results_runme_sha256 $SHA_RES"; echo "record $REC_COMMIT:$REC_PATH gz $REC_GZ json $REC_JS";
  echo "frozen $FRZ_COMMIT:$FRZ_DIR/{$NODES_FILE $NODES_SHA, $TPL_FILE $TPL_SHA}"; echo "summary $SUM_COMMIT:$SUM_PATH $SUM_SHA launch_id $LAUNCH_ID";
  echo "a1 $A1_VOL:$VALUES_REL node_series.npz $NS_SHA compare.json $CMP_SHA"; echo "perdraw $PD_VOL:$PD_REL $PD_SHA";
  echo "counts n_nodes $N_NODES n_gap $N_GAP draws $DRAWS min_finite $MIN_FINITE"; } >> "$LOG"
echo "provenance gates passed (record $REC; summary $SUM_SHA; values_dir and value shas read by key); run folder runs/$STAMP"

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
echo "inputs extracted and sha-checked (record $REC gz + json, node list $NODES_FILE, templates)"

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
echo "LAUNCH $STAMP PT: record $REC; app $APP; raw log runs/$STAMP/launch_$STAMP.log (mode 600; not to be relayed)"
( cd "$HERE" && "$M" run recheck_b_v4.py --stamp "$STAMP" --run-dir "$RUN_DIR" --trailer "$TRAILER" --record "$REC" \
    --values-dir "$VALUES_DIR" --ns-sha "$NS_SHA" --cmp-sha "$CMP_SHA" ) >> "$LOG" 2>&1
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
echo "NEXT: bash $HERE/RUN_ME_b_results_v4.sh $NAME"
exit 0
