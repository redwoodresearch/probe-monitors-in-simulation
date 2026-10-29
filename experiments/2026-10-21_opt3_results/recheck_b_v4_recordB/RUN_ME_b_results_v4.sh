#!/bin/bash
# RUN_ME_b_results_v4.sh: recheck (b) v4, parametrized by record (PRESTATEMENT_b_v4.md, a delta note on PRESTATEMENT_b_v3.md /
# PRESTATEMENT_b_v2.md). v3's results RUN_ME (RUN_ME_b_results_v3.sh eaa09e1e...) copied; it differs only in its pins, file names, the
# record read from the output name, the record's counts in the spec, and the `record` key (PRESTATEMENT_b_v4 §3.9). Checks the job's
# output JSON and prints its table (booleans and ints, plus the record letter).
# No Modal call, no Python, no write: reads results/<name> (the exact local file name) and runs/<STAMP>/launch_<STAMP>.log only.
#   bash RUN_ME_b_results_v4.sh recheck_b_v4_out_<A|B>_<YYYY-MM-DD_HHMMSS>.json
# Gates (each STOPs with exit 3; nothing from the output file is rendered before gate 5 passes):
#   1. SHA256SUMS_v4 holds and lists exactly the four v4 files; the pinned pre-statement / script shas match; files read-only;
#   2. the output's sha256 equals the launch log's single 'RECHECK_B_STATUS ok' line, which names this output;
#   3. the spec below (PRESTATEMENT_b_v4 §4, with the record's counts) has 37 keys, each declared once in recheck_b_v4.py's OUT_KEYS
#      with the same type;
#   4. fixture self-tests: gate_text must pass the valid fixture and fail every negative one (incl. invalid-then-valid,
#      valid-then-invalid and two valid documents), printing nothing each time; the verdict rule must name exactly the broken key;
#   5. the real file through gate_text: exactly ONE top-level JSON value and it is an object, 37 key lines, no float / NaN / Infinity
#      literal, and the exact schema on .[0] only (`record` is the only string, from {"A", "B"}).
# Then it prints the table (key, value, PASS needs) from .[0] and ONE verdict line: RECHECK_B PASS (exit 0) or
# RECHECK_B FAIL failing=[...] (exit 1).
set -u
set -o pipefail

die() { echo "STOP: $*"; exit 3; }

HERE=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_recheck/b_incontainer_B
SHA_PRE=1cbcac690f4ae005c7661dbe1cd987fb8d3b222ad509261b97d66c833346eeff   # PRESTATEMENT_b_v4.md
SHA_PY=7c9b6ad83a56435a9eb41eaf190ee0f2250fe3d4c1ef366457cf0162447d28df    # recheck_b_v4.py

[ "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" = "$HERE" ] || die "run this file from its frozen folder $HERE"
[ $# -eq 1 ] || die "usage: bash RUN_ME_b_results_v4.sh recheck_b_v4_out_<A|B>_<YYYY-MM-DD_HHMMSS>.json"
NAME=$1
re='^recheck_b_v4_out_(A|B)_([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6})\.json$'
[[ $NAME =~ $re ]] || die "the name is not recheck_b_v4_out_<A|B>_<YYYY-MM-DD_HHMMSS>.json"
REC=${BASH_REMATCH[1]}
STAMP=${BASH_REMATCH[2]}
# The record's counts (PRESTATEMENT_b_v4 §2; the same values as recheck_b_v4.py's RECORDS and the launch RUN_ME's table).
case $REC in
  A) NN=60; NG=28; OTHER=B;;
  B) NN=240; NG=180; OTHER=A;;
  *) die "the record must be A or B";;
esac
F=$HERE/results/$NAME
LOG=$HERE/runs/$STAMP/launch_$STAMP.log
for t in jq shasum awk grep sort tr; do command -v "$t" >/dev/null || die "tool $t missing"; done
[ -f "$F" ] && [ ! -L "$F" ] || die "no regular output file results/$NAME"
[ -f "$LOG" ] || die "no launch log runs/$STAMP/launch_$STAMP.log"

# 1. Frozen files.
( cd "$HERE" && shasum -a 256 -c SHA256SUMS_v4 >/dev/null 2>&1 ) || die "SHA256SUMS_v4 check failed (a frozen file changed)"
names=$(awk '{print $2}' "$HERE/SHA256SUMS_v4" | LC_ALL=C sort | tr '\n' ' ') || die "read SHA256SUMS_v4"
[ "$names" = "PRESTATEMENT_b_v4.md RUN_ME_b_launch_v4.sh RUN_ME_b_results_v4.sh recheck_b_v4.py " ] || die "SHA256SUMS_v4 lists other files"
pin() { local h; h=$(shasum -a 256 "$HERE/$1" | awk '{print $1}') || die "shasum $1"; [ "$h" = "$2" ] || die "$1: sha256 differs from the pinned value"; }
pin PRESTATEMENT_b_v4.md "$SHA_PRE"
pin recheck_b_v4.py "$SHA_PY"
for f in PRESTATEMENT_b_v4.md recheck_b_v4.py RUN_ME_b_launch_v4.sh RUN_ME_b_results_v4.sh SHA256SUMS_v4; do
  [ -w "$HERE/$f" ] && die "$f is writable (not frozen)"
done

# 2. The output's sha256 against the launch log's single status line:
#    RECHECK_B_STATUS ok app_id=<id> name=<name> sha256=<sha> local=results/<name>
n=$(grep -c '^RECHECK_B_STATUS ' "$LOG")
[ "$n" = 1 ] || die "the launch log holds ${n:-no} RECHECK_B_STATUS lines (want 1)"
line=$(grep '^RECHECK_B_STATUS ok ' "$LOG") || die "the status line is not 'ok'"
lname=$(printf '%s\n' "$line" | awk '{print $4}') || die "awk status line"
lsha=$(printf '%s\n' "$line" | awk '{print $5}') || die "awk status line"
lloc=$(printf '%s\n' "$line" | awk '{print $6}') || die "awk status line"
[ "$lname" = "name=$NAME" ] && [ "$lloc" = "local=results/$NAME" ] || die "the status line names a different output"
want=${lsha#sha256=}
[[ $lsha == sha256=* && $want =~ ^[0-9a-f]{64}$ ]] || die "the status line carries no sha256"
got=$(shasum -a 256 "$F" | awk '{print $1}') || die "shasum output"
[ "$got" = "$want" ] || die "the output's sha256 differs from the launch log's status line"

# 3. The spec (PRESTATEMENT_b_v2 §4 + PRESTATEMENT_b_v4 §3.4 / §4): key, type, PASS needs ("true", a number, "ge1" = at least 1,
#    "any" = reported only, or, for the one string key `record`, the record letter from the output name).
SPEC="record str $REC
sha_ok_record_gz bool true
sha_ok_record_json bool true
sha_ok_nodes_file bool true
sha_ok_templates bool true
sha_ok_node_series bool true
sha_ok_compare bool true
sha_ok_perdraw_npz bool true
sanitizer_selftest_ok bool true
pins_ok bool true
blas_env_all_one bool true
threadpools_n int ge1
threadpools_all_one bool true
reader_control_ok bool true
templates_ok bool true
nodes_same_order_as_frozen bool any
method_is_linear bool true
scale_is_fraction bool true
d1_num_is_1 bool true
i_rule_is_I1 bool any
n_nodes_checked int $NN
n_ends_ok int $NN
n_counts_ok int $NN
n_a1_compare_ok int $NN
n_withheld_mine int 0
n_gap_nodes int $NG
n_gap_members_found int $NG
n_gap_lengths_ok int $NG
n_sample int 2000
n_sample_ok int 2000
n_sample_both_nonfinite int any
negative_control_ends_baseline_ok bool true
negative_control_ends_flagged bool true
negative_control_draw_baseline_ok bool true
negative_control_draw_flagged bool true
negative_control_flagged bool true
elapsed_s int any"
SPEC_JSON=$(printf '%s\n' "$SPEC" | jq -R -s -c '[split("\n")[] | select(length > 0) | split(" ") | {k: .[0], t: .[1], r: .[2]}]') || die "spec to JSON"
n_spec=$(printf '%s' "$SPEC_JSON" | jq -e '[.[] | select((.k | type) == "string" and (.t == "bool" or .t == "int" or .t == "str") and (.r | type) == "string")] | length') || die "spec shape"
[ "$n_spec" = 37 ] || die "the spec holds $n_spec well-formed rows (want 37)"
n_uniq=$(printf '%s' "$SPEC_JSON" | jq 'map(.k) | unique | length') || die "spec keys"
[ "$n_uniq" = 37 ] || die "the spec repeats a key"
n_str=$(printf '%s' "$SPEC_JSON" | jq '[.[] | select(.t == "str")] | length') || die "spec string keys"
[ "$n_str" = 1 ] || die "the spec holds $n_str string keys (want 1: record)"
while read -r k t r; do
  c=$(grep -c -F -x "    \"$k\": $t," "$HERE/recheck_b_v4.py")
  [ "$c" = 1 ] || die "recheck_b_v4.py OUT_KEYS: want exactly one line declaring $k as $t (found ${c:-none})"
done <<< "$SPEC"
c=$(grep -c -E '^    "[a-zA-Z0-9_]+": (bool|int|str),$' "$HERE/recheck_b_v4.py")
[ "$c" = 37 ] || die "recheck_b_v4.py declares ${c:-no} OUT_KEYS lines (want 37)"

SCHEMA_JQ='. as $o
| if type != "object" then false
  elif keys != ($spec | map(.k) | sort) then false
  else all($spec[]; . as $s | $o[$s.k] as $v
         | if $s.t == "bool" then ($v | type) == "boolean"
           elif $s.t == "int" then (($v | type) == "number") and ($v | tojson | test("^[0-9]+$"))
           elif $s.t == "str" then (($v | type) == "string") and ($v == "A" or $v == "B")
           else false end)
  end'
VERDICT_JQ='. as $o
| [ $spec[] | . as $s | $o[$s.k] as $v
    | select(if $s.r == "any" then false
             elif $s.t == "str" then $v != $s.r
             elif $s.r == "true" then $v != true
             elif $s.r == "ge1" then $v < 1
             else $v != ($s.r | tonumber) end)
    | $s.k ]'
VALID_JQ='$spec | map({key: .k, value: (if .t == "bool" then true elif .t == "str" then .r elif .r == "ge1" then 1 elif .r == "any" then 0
  else (.r | tonumber) end)})
| from_entries'

gate_text() {  # $1 = a document's text. Prints nothing. Returns 0 only if ALL hold: exactly one top-level JSON value and it is an
               # object; 37 key lines; no float / NaN / Infinity literal; the exact schema on .[0].
  local c
  printf '%s' "$1" | jq -s -e 'length == 1 and (.[0] | type) == "object"' >/dev/null 2>&1 || return 1
  c=$(printf '%s\n' "$1" | grep -c '^ "')
  [ "$c" = 37 ] || return 1
  c=$(printf '%s\n' "$1" | grep -c -E '[0-9][.eE]|NaN|Infinity')
  [ "$c" = 0 ] || return 1
  printf '%s' "$1" | jq -s -e --argjson spec "$SPEC_JSON" ".[0] | $SCHEMA_JQ" >/dev/null 2>&1 || return 1
  return 0
}
verdict_fails() { jq -s -c --argjson spec "$SPEC_JSON" ".[0] | $VERDICT_JQ"; }

# 4. Fixture self-tests (in memory). Every negative fixture must differ from the valid one; every gate call must print nothing.
VALID=$(jq -n -c --argjson spec "$SPEC_JSON" "$VALID_JQ") || die "build the valid fixture"
VPRETTY=$(printf '%s' "$VALID" | jq --indent 1 -S .) || die "build the pretty valid fixture"
mutp() {  # pretty (as the job writes it); runs inside $(...): its STOP goes to stderr and the caller exits 3
  local fx
  fx=$(printf '%s' "$VALID" | jq --indent 1 -S "$1") || { echo "STOP: build fixture '$1'" >&2; return 1; }
  [ "$fx" != "$VPRETTY" ] || { echo "STOP: fixture '$1' does not differ from the valid one" >&2; return 1; }
  printf '%s' "$fx"
}
mutc() {  # compact, for the verdict rule
  local fx
  fx=$(printf '%s' "$VALID" | jq -c "$1") || { echo "STOP: build fixture '$1'" >&2; return 1; }
  [ "$fx" != "$VALID" ] || { echo "STOP: fixture '$1' does not differ from the valid one" >&2; return 1; }
  printf '%s' "$fx"
}
expect_gate() {  # want(pass|fail) label text
  local o r g
  o=$(gate_text "$3" 2>&1); r=$?
  [ -z "$o" ] || die "gate self-test '$2' printed output"
  if [ "$r" -eq 0 ]; then g=pass; else g=fail; fi
  [ "$g" = "$1" ] || die "gate self-test '$2': got $g, want $1"
}
expect_verdict() {  # label json want_failing_keys
  local g
  g=$(printf '%s' "$2" | verdict_fails) || die "verdict self-test '$1': jq error"
  [ "$g" = "$3" ] || die "verdict self-test '$1': got $g, want $3"
}
expect_gate pass valid "$VPRETTY"
expect_gate fail invalid_then_valid '{"leak":"x"}'$'\n'"$VPRETTY"
expect_gate fail valid_then_invalid "$VPRETTY"$'\n''{"leak":"x"}'
expect_gate fail two_valid "$VPRETTY"$'\n'"$VPRETTY"
fx=$(mutp '.n_ends_ok = 60.5') || exit 3; expect_gate fail float "$fx"
fx=$(mutp 'del(.elapsed_s)') || exit 3; expect_gate fail missing_key "$fx"
fx=$(mutp '.extra_key = 1') || exit 3; expect_gate fail extra_key "$fx"
fx=$(mutp '.pins_ok = "true"') || exit 3; expect_gate fail string_value "$fx"
fx=$(mutp '.n_sample = true') || exit 3; expect_gate fail bool_as_int "$fx"
fx=$(mutp '.pins_ok = 1') || exit 3; expect_gate fail int_as_bool "$fx"
fx=$(mutp '.elapsed_s = -1') || exit 3; expect_gate fail negative_int "$fx"
fx=$(mutp '.n_ends_ok = null') || exit 3; expect_gate fail null_value "$fx"
fx=$(mutp '.record = 1') || exit 3; expect_gate fail record_as_int "$fx"
fx=$(mutp '.record = "C"') || exit 3; expect_gate fail record_outside_set "$fx"
expect_gate fail not_an_object '[]'
expect_gate fail garbage 'not json'
expect_verdict valid "$VALID" '[]'
fx=$(mutc '.i_rule_is_I1 = false | .nodes_same_order_as_frozen = false | .n_sample_both_nonfinite = 7 | .elapsed_s = 99') || exit 3
expect_verdict reported_only "$fx" '[]'
fx=$(mutc ".n_ends_ok = $((NN - 1))") || exit 3; expect_verdict ends_minus_1 "$fx" '["n_ends_ok"]'
fx=$(mutc '.n_withheld_mine = 1') || exit 3; expect_verdict withheld_1 "$fx" '["n_withheld_mine"]'
fx=$(mutc '.n_sample_ok = 1999') || exit 3; expect_verdict sample_1999 "$fx" '["n_sample_ok"]'
fx=$(mutc '.threadpools_n = 0') || exit 3; expect_verdict pools_0 "$fx" '["threadpools_n"]'
fx=$(mutc '.negative_control_flagged = false') || exit 3; expect_verdict nc_false "$fx" '["negative_control_flagged"]'
fx=$(mutc '.sha_ok_perdraw_npz = false') || exit 3; expect_verdict sha_false "$fx" '["sha_ok_perdraw_npz"]'
fx=$(mutc '.sanitizer_selftest_ok = false') || exit 3; expect_verdict sanitizer_false "$fx" '["sanitizer_selftest_ok"]'
fx=$(mutc ".record = \"$OTHER\"") || exit 3; expect_verdict record_other "$fx" '["record"]'
echo "self-tests: gate 16 fixtures (each printed nothing), verdict 10 fixtures, all as predicted (record $REC)"

# 5. The real file through the same gate; nothing from it is rendered before this passes.
DOC=$(<"$F") || die "read the output"
gate_text "$DOC" || die "the output fails the gate (exactly one JSON object; 37 key lines; no float / NaN / Infinity; exact schema)"

echo "RECHECK_B table (record $REC, results/$NAME, sha256 $got): key, value, PASS needs"
printf '%s' "$DOC" | jq -s -r --argjson spec "$SPEC_JSON" '.[0] as $o | $spec[] | "\(.k)\t\($o[.k])\t\(.r)"' || die "print the table"
fails=$(printf '%s' "$DOC" | verdict_fails) || die "verdict"
if [ "$fails" = "[]" ]; then
  echo "RECHECK_B PASS (record $REC; every PASS-needs entry of PRESTATEMENT_b_v4 §4 holds)"
  exit 0
fi
echo "RECHECK_B FAIL (record $REC) failing=$fails"
exit 1
