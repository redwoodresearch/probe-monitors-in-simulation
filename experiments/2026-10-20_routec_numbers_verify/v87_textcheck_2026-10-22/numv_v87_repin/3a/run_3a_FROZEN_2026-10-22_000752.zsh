#!/bin/zsh
# (3a') AMEND1: the text-only numbers check of the posted v87 against the numbers-verified X (09, GATES L3501 @ dcfc03c71; AMEND1 rulings
# L3515 @ 5f8bbb758 and L3533 @ eb7247196). zsh + jq + diff only; free; no Python, no Modal, no git writes. Prints keys, shas, counts, row
# ids and booleans only; every text-bearing intermediate is a file under the run's work/ folder and is never printed.
#   zsh run_3a_FROZEN_<stamp>.zsh real    (identity positive, real check, hunk-8 consistency, 4 planted v87 negatives, 2 planted X negatives)
#   zsh run_3a.zsh fixture <fixture dir>  (a disclosed DRY RUN on synthetic files only; same code path)
setopt errexit nounset pipefail
F=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87_repin
A=$F/3a
MODE=${1:?mode}
die() { print -r -- "STOP: $*"; exit 3; }
sha() { shasum -a 256 "$1" | cut -c1-64; }
now() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z'; }
if [[ $MODE == real ]]; then
  [[ "${0:A:t}" == run_3a_FROZEN_*.zsh && ! -w "${0:A}" ]] || die "real mode runs only the frozen, read-only driver"
  EMIT=$A/emit_3a_FROZEN_2026-10-22_000752.jq; CHECK=$A/check_3a_FROZEN_2026-10-22_000752.jq; H8=$A/h8_consistency_FROZEN_2026-10-22_000752.jq
  [[ ! -w $EMIT && "$(sha $EMIT)" == 2bdaf6414829959f116fe44ce97b01e131dc96d777f1521d1135ab19130541c2 && ! -w $CHECK && "$(sha $CHECK)" == d24ba440582a8db4653068cb82feef7032b6614dfca2d4f817594a70ffab57b3 && ! -w $H8 && "$(sha $H8)" == 4691dafc80f0832d56d6dcc068d18aa1f678e5746f19c7fe777512d743ad70a0 ]] \
    || die "frozen jq files: sha or mode"
else EMIT=$A/emit_3a.jq; CHECK=$A/check_3a.jq; H8=$A/h8_consistency.jq; fi
RUN=$A/run_${MODE}_$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S); [[ ! -e $RUN ]] || die "run folder exists"
mkdir -p $RUN/work $RUN/out || die "mkdir"
print -r -- "START $(now) mode=$MODE run=$RUN emit_sha256=$(sha $EMIT) check_sha256=$(sha $CHECK) h8_sha256=$(sha $H8) self_sha256=$(sha ${0:A})"

if [[ $MODE == real ]]; then
  D1R=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1/run_real_2026-10-21_205711
  D1=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1
  REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
  E=experiments/2026-10-20_routec_numbers_verify
  XS=$D1R/base.stripped;  XS_SHA=92b6a030d69275b1c6bbe86c91f900a0c4be27cf0f599764937cce62e8fd373b
  V87=/Users/aswinsurya/probe-shift-writing-handoff-2026-10-15/w23/drafts/draft_v87_final_frozen_0029.md; V87_SHA=75af0e8e5a10cf762bc4dc327ced8c1eb1386b64c5d680c4e7102086bad3e512
  HT=$D1R/hunks.txt;       HT_SHA=74495f1a3dc4bf43a524b2cce5deddee99e7dcd4e44a7592718bbfa91d510640
  RE=$D1/writing_core_regex.bin; RE_SHA=9414451d677f0d34b11fc2d4d80fc1b3f880c82445a5d4d64da156ddfc913bf3
  R123=$D1/R123_out_2026-10-21_220301.txt; R123_SHA=428d4e3d3f3002664860bb0b323e84a523d803fb246703f03f65e2f59c970c01
  D1SUMS=$D1/SHA256SUMS; D1SUMS_SHA=a26839e2ba17fd8be689aefcf438501e9edb518ea761899033b32a4e40be0566
  HP=$D1R/hidden_pattern.bin; HP_SHA=39d3d18fd21298dd2ea96526389f29db50575fe2b20f99d3306ddf6b269e79fa
  WSF=$D1R/posthoc_ws_move.txt; WSF_SHA=b79680b7b2aad1e64f1fbe35183b37b750b7d2a7d4285b4ae121dbfb03c28f7d
  typeset -A GP GS
  GP=(v19 c20699a5d:$E/post/provenance_v84_final19_X.jsonl v10 c20699a5d:$E/post/constants_v84_final10_X.jsonl
      dfr b58b25384:$E/post/deferred_v84_final8_0141.jsonl final5 d1c8ec348:$E/post/constants_sources_final5_0141_030913.jsonl
      add9 64d962a67:$E/post/constants_sources_v9_additions_0141.jsonl xres afe11c809:$E/results/numbers_verify_2026-10-21_025253.json
      xraw c20699a5d:$E/post/draft_v84_X_0141_appC0029.md)
  GS=(v19 a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda v10 35a780d0d69ab2f3e8217c7efc8db62106ee24acd318d678451b5791c313e6b5
      dfr 9f3260a3b4be241e53603610f0ac502386fd09cc83900a9473ca5ccf2da3d180 final5 f5f316cb84320035517f11ae34025f05991a5c473101b8ad31151cd46a674d05
      add9 eb03888f433c6b045b2eaf73d7c76958256956c70fa27bfb158e6f5282adc3f2 xres d59138c43422e2e50766611e1781737075bc9aaf9d10af1211a30c7deef7e618
      xraw 4d01234cabb79ff09695c0dffec2183e805c3f4129914dc6a7f62b423b4336fa)
  for f_ s_ in $XS $XS_SHA $V87 $V87_SHA $HT $HT_SHA $RE $RE_SHA $R123 $R123_SHA $D1SUMS $D1SUMS_SHA $HP $HP_SHA $WSF $WSF_SHA; do
    [[ -f $f_ && "$(sha $f_)" == "$s_" ]] || die "input sha: ${f_:t}"; done
  for k_ in ${(k)GP}; do git -C $REPO show "$GP[$k_]" > $RUN/work/$k_.in || die "git show $k_"; [[ "$(sha $RUN/work/$k_.in)" == "$GS[$k_]" ]] || die "pin sha $k_"; done
  W=$RUN/work
  V19=$W/v19.in; V10=$W/v10.in; DFR=$W/dfr.in; FIN5=$W/final5.in; ADD9=$W/add9.in; XR=$W/xraw.in
  R241OK=$(jq -c '.diagnostics.post.map_rows["241"] | (.ok == true and .bucket == "verified")' $W/xres.in) || die "row 241 in X's result"
else
  FX=${2:?fixture dir}; [[ -d $FX ]] || die "no fixture dir"
  XS=$FX/x.stripped; V87=$FX/v87.md; HT=$FX/hunks.txt; RE=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1/writing_core_regex.bin
  R123=$FX/r123.txt; V19=$FX/v19.jsonl; V10=$FX/v10.jsonl; DFR=$FX/dfr.jsonl; FIN5=$FX/final5.jsonl; ADD9=$FX/add9.jsonl; WSF=$FX/ws.txt
  R241OK=$(cat $FX/r241ok.json); W=$RUN/work; XR=$FX/x.raw; HP=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1/run_real_2026-10-21_205711/hidden_pattern.bin
fi
print -r -- "PINS_OK mode=$MODE x=$(sha $XS) xraw=$(sha $XR) v87=$(sha $V87) hunks=$(sha $HT) regex=$(sha $RE) hidden_pattern=$(sha $HP) v19=$(sha $V19) v10=$(sha $V10) dfr=$(sha $DFR) final5=$(sha $FIN5) add9=$(sha $ADD9) ws=$(sha $WSF) r241_verified_in_X=$R241OK"

# ---- approved literals (09 condition 4): a v10 row is approved iff final5 holds its row with text_sha256 == sha256(its text), or the v9
# additions hold its row with byte-equal text. Values stay in variables; only row ids and booleans are written.
: > $W/appr.jsonl
jq -c '{row, t: .text}' $V10 | while IFS= read -r l_; do
  r_=$(print -r -- "$l_" | jq -r .row); ts_=$(print -r -- "$l_" | jq -j .t | shasum -a 256 | cut -c1-64)
  f5_=$(jq -r --argjson r "$r_" 'select(.row == $r) | .text_sha256' $FIN5 | head -n 1)
  a9n_=$(jq -r --argjson r "$r_" 'select(.constants_row == $r) | .constants_row' $ADD9 | wc -l | tr -d ' ')
  a9_=$(jq -j --argjson r "$r_" '[select(.constants_row == $r) | .text][0] // ""' $ADD9 | shasum -a 256 | cut -c1-64)
  jq -n -c --argjson r "$r_" --argjson f5 "$([[ -n $f5_ && $f5_ == $ts_ ]] && print true || print false)" \
     --argjson a9 "$([[ $a9n_ -ge 1 && $a9_ == $ts_ ]] && print true || print false)" '{row: $r, final5: $f5, add9: $a9, approved: ($f5 or $a9)}' >> $W/appr.jsonl
done
jq -s -c '.' $W/appr.jsonl > $W/appr.json
print -r -- "APPROVED_LITERALS rows=$(jq length $W/appr.json) approved=$(jq '[.[] | select(.approved)] | length' $W/appr.json) final5=$(jq '[.[] | select(.final5)] | length' $W/appr.json) add9=$(jq '[.[] | select(.add9)] | length' $W/appr.json)"

# ---- D1's per-hunk multiset counts (its R1 output, CORES lines): {"idx": [removed, added]}
D1C=$(grep -E '^CORES hunk=[0-9]+ removed=[0-9]+ added=[0-9]+$' $R123 | awk '{split($2,a,"="); split($3,b,"="); split($4,c,"="); printf "%s\"%s\":[%s,%s]", (n++?",":""), a[2], b[2], c[2]} END {}' | sed 's/^/{/; s/$/}/')
print -r -- "$D1C" | jq -e 'type == "object"' >/dev/null || die "D1 cores map"

# ---- one check: stage 1 (code points per kind-c hunk), POSIX diff per hunk, stage 2
check_one() {  # check_one <label> <v87-side file> <hunks file>
  local lab_=$1 vs_=$2 ht_=$3 d_=$W/$1 rc_ i_
  mkdir -p $d_ || die "mkdir $d_"
  jq -n -r -f $EMIT --rawfile xs $XS --rawfile vs $vs_ --rawfile ht $ht_ > $d_/cp.txt || die "emit ($lab_)"
  awk -v D=$d_ '{f = D "/h" $1 "." $2; if (f != cur) { if (cur != "") close(cur); cur = f }; print $3 >> f}' $d_/cp.txt || die "split ($lab_)"
  : > $d_/cmds.jsonl
  for i_ in $(awk '{print $1}' $d_/cp.txt | sort -n -u); do
    [[ -f $d_/h$i_.O && -f $d_/h$i_.N ]] || die "hunk $i_ sides ($lab_)"
    rc_=0; diff $d_/h$i_.O $d_/h$i_.N > $d_/h$i_.diff || rc_=$?
    [[ $rc_ == 0 || $rc_ == 1 ]] || die "diff rc $rc_ hunk $i_ ($lab_)"
    grep -E '^[0-9]+(,[0-9]+)?[acd][0-9]+(,[0-9]+)?$' $d_/h$i_.diff | awk -v I=$i_ '{
      match($0, /[acd]/); k = substr($0, RSTART, 1); o = substr($0, 1, RSTART - 1); n = substr($0, RSTART + 1)
      no = split(o, oo, ","); nn = split(n, ee, ",")
      printf "{\"idx\":%d,\"k\":\"%s\",\"o1\":%d,\"o2\":%d,\"n1\":%d,\"n2\":%d}\n", I, k, oo[1], (no > 1 ? oo[2] : oo[1]), ee[1], (nn > 1 ? ee[2] : ee[1]) }' >> $d_/cmds.jsonl
  done
  jq -n -f $CHECK --rawfile xs $XS --rawfile xr $XR --rawfile hp $HP --rawfile vs $vs_ --rawfile re $RE --rawfile ht $ht_ --slurpfile cmds $d_/cmds.jsonl \
     --slurpfile v19 $V19 --slurpfile v10 $V10 --slurpfile dfr $DFR --slurpfile appr $W/appr.json \
     --argjson r241ok "$R241OK" --argjson d1cores "$D1C" --arg mode "$lab_" > $RUN/out/$lab_.json || die "check ($lab_)"
  jq -r '"RESULT \(.mode) a_ok=\(.a.ok) strip_ok=\(.lines.raw_x_strips_to_x_stripped) a_rows=\(.a.rows_in_changed_hunks) a_kept=\(.a.kept) a_fail=\([.a.fail[] | "\(.set):\(.row)"] | tojson) a_not_printed=\(.a.not_printed | tojson) a_no_core_kept=\(.a.no_core | tojson) outside=\(.a.outside_rows) outside_fail=\([.a.outside_fail[] | "\(.set):\(.row)"] | tojson) lines_unequal=\(.lines.unchanged_unequal) runs_bad=\(.lines.runs_bad) b_ok_script=\(.b.ok_script) b=\([.b.per_hunk[] | "\(.hunk):n\(.n_added):\(.matched_by):\(.approved_rows + .h8_rows_19_83 | tojson)"] | tojson) c_ok=\(.c.ok) c_verdict=\(.c.verdict) c_na=\(.c.na) d1_multiset_match=\(.d1_cross_check.all_match) d1_mismatch=\(.d1_cross_check.mismatched_hunks | tojson) deferred=\([.a.deferred[] | "\(.row):\(.status)"] | tojson) multi=\([.a.multi_occurrence_in_v87[] | "\(.set):\(.row):\(.status)"] | tojson) hidden_overlap=\(.a.anchors_overlapping_hidden_comments | tojson)"' $RUN/out/$lab_.json
}
# ---- hunk-8 consistency (09 L3533): X's own definition of v10 rows 19 / 83 against v87's hunk-8 "N%"
h8_one() {  # h8_one <label> <raw X file> <v87-side file> <hunk-8 added core position in it>
  jq -n -f $H8 --rawfile xr $2 --rawfile hp $HP --rawfile re $RE --rawfile vs $3 --argjson h8pos "${4:-null}" --slurpfile v10 $V10 > $RUN/out/h8_$1.json || die "h8 ($1)"
  jq -r --arg l "$1" '"H8 \($l) hunk8_consistency=\(.hunk8_consistency) forms=\(.forms | tojson) core_has_pct=\(.h8_core_has_pct) rows=\([.rows[] | "\(.row):found=\(.anchor_once_raw):resolved=\(.resolved):lit_eq_N=\(.literal_eq_N):x_core_eq_N=\(.x_core_eq_N // false):form=\(.form):x_lines=\(.sentence_x_lines // [] | tojson)"] | tojson)"' $RUN/out/h8_$1.json
}

# ---- identity positive: X against itself, no hunks (09 / 2e: (c) is N/A by construction, c_na = no hunk 32 in the identity input)
: > $W/empty_hunks.txt
check_one identity $XS $W/empty_hunks.txt
# ---- the real check and the real hunk-8 consistency
check_one real $V87 $HT
P8=$(jq -r '.neg_targets.h8_added_vpos // empty' $RUN/out/real.json)
h8_one real $XR $V87 "$P8"

# ---- planted v87 negatives (positions from the real run's keys; every copy keeps every line break)
plant() {  # plant <label> <code point position> <rule: plus1 | not_approved>: one digit changed, 1 byte
  local p_=$2
  [[ "$p_" == <-> ]] || die "plant $1: no position"
  jq -n -r -j --rawfile v $V87 --argjson p $p_ --arg rule $3 --slurpfile v10 $V10 --slurpfile appr $W/appr.json --rawfile re $RE '
    ($v | explode) as $E | if ($E[$p] < 48 or $E[$p] > 57) then error("not a digit") else . end
    | ($appr[0] | map(select(.approved) | .row)) as $ok | ([$v10[] | select(.row as $r | $ok | index($r)) | .text]) as $lits
    | [range(1; 10) as $k | ($E | .[$p] = 48 + ((.[$p] - 48 + $k) % 10)) as $N
       | ($N | implode) as $t | ([$t | match($re; "g") | select(.offset <= $p and .offset + (.string | length) > $p) | .string | sub(",+$"; "")][0]) as $c
       | select($rule == "plus1" or (($lits | index($c)) | not)) | $t][0] // error("no planted digit")' > $W/neg_$1.md || die "plant $1"
  [[ "$(cmp -l $V87 $W/neg_$1.md 2>/dev/null | wc -l | tr -d ' ')" == 1 ]] || die "plant $1: the copy does not differ in exactly 1 byte"
  print -r -- "PLANTED $1 pos=$p_ rule=$3 bytes_differing=1 sha256=$(sha $W/neg_$1.md)"
}
PA=$(jq -r '.neg_targets.v19_105_core_vpos[0] // empty' $RUN/out/real.json); plant a "$PA" plus1
PB=$(jq -r '.neg_targets.h24_added_vpos // empty' $RUN/out/real.json);        plant b "$PB" not_approved
PC=$(jq -r '.neg_targets.h32_added_vpos // empty' $RUN/out/real.json)
PC_LAST=$(jq -n -r --rawfile v $V87 --argjson p "$PC" --rawfile re $RE '[$v | match($re; "g") | select(.offset == $p) | .string | sub(",+$"; "") | sub("%$"; "")][0] as $c | $p + ($c | length) - 1')
plant c "$PC_LAST" plus1
# neg_d (09 L3515 / L3533): hunk 8's added core replaced by L + "%", L = the approved v10 literal of lowest row id in fraction form
# (^[0-9]*\.[0-9]+$) with L + "%" itself not an approved literal and L not rows 19 / 83's literal. A generic "core minus % = literal" rule
# would accept it; the hunk-8 rule and the consistency check must not.
[[ "$P8" == <-> ]] || die "neg_d: no hunk-8 position"
jq -n -r -j --rawfile v $V87 --argjson p $P8 --slurpfile v10 $V10 --slurpfile appr $W/appr.json --rawfile re $RE '
  ($appr[0] | map(select(.approved) | .row)) as $ok | ([$v10[] | select(.row as $r | $ok | index($r))]) as $A
  | ([$A[] | .text]) as $lits | ([$v10[] | select(.row == 19 or .row == 83) | .text]) as $n
  | ([$A[] | . as $a | select(($a.text | test("^[0-9]*\\.[0-9]+$")) and (($lits | index($a.text + "%")) | not) and (($n | index($a.text)) | not))] | sort_by(.row) | .[0].text) as $L
  | if $L == null then error("no fraction-form literal") else . end
  | ([$v | match($re; "g") | select(.offset == $p) | .string | sub(",+$"; "")][0]) as $c
  | if $c == null then error("no core at the hunk-8 position") else . end
  | ($v | explode) as $E | (($E[0:$p] + (($L + "%") | explode) + $E[$p + ($c | length):]) | implode)' > $W/neg_d.md || die "plant d"
[[ "$(wc -l < $V87)" == "$(wc -l < $W/neg_d.md)" ]] || die "plant d: line count changed"
print -r -- "PLANTED d pos=$P8 rule=fraction_literal_plus_pct sha256=$(sha $W/neg_d.md)"
check_one neg_a $W/neg_a.md $HT
check_one neg_b $W/neg_b.md $HT
check_one neg_c $W/neg_c.md $HT
check_one neg_d $W/neg_d.md $HT
h8_one neg_d $XR $W/neg_d.md "$P8"
# ---- planted X negatives for the hunk-8 consistency (09 L3533 (2)): on a copy of raw X, row 19's matched marker becomes "per 1,000"
# ("per 100" -> "per 1,000"; "in 100" -> "in 1,000"; "%" -> " per 1,000"); and, separately, row 19's core N gets its first digit +1
M19=$(jq -r '.rows[] | select(.row == 19) | .marker_raw_pos // empty' $RUN/out/h8_real.json)
F19=$(jq -r '.rows[] | select(.row == 19) | .form' $RUN/out/h8_real.json)
K19=$(jq -r '.rows[] | select(.row == 19) | .core_raw_pos // empty' $RUN/out/h8_real.json)
if [[ "$M19" == <-> && "$K19" == <-> ]]; then
  jq -n -r -j --rawfile x $XR --argjson p $M19 --arg f "$F19" '($x | explode) as $E
    | (if $f == "per 100" then {old: "per 100", new: "per 1,000"} elif $f == "in 100" then {old: "in 100", new: "in 1,000"}
       elif $f == "%" then {old: "%", new: " per 1,000"} else error("no form") end) as $m
    | if ($E[$p:$p + ($m.old | length)] | implode) != $m.old then error("marker not at its position") else . end
    | (($E[0:$p] + ($m.new | explode) + $E[$p + ($m.old | length):]) | implode)' > $W/negx_scale.md || die "plant x scale"
  jq -n -r -j --rawfile x $XR --argjson p $K19 '($x | explode) as $E | if ($E[$p] < 48 or $E[$p] > 57) then error("not a digit") else . end
    | ($E | .[$p] = 48 + ((.[$p] - 48 + 1) % 10)) | implode' > $W/negx_digit.md || die "plant x digit"
  [[ "$(cmp -l $XR $W/negx_digit.md 2>/dev/null | wc -l | tr -d ' ')" == 1 ]] || die "plant x digit: not exactly 1 byte"
  print -r -- "PLANTED x_scale pos=$M19 form=$F19 sha256=$(sha $W/negx_scale.md) ; PLANTED x_digit pos=$K19 bytes_differing=1 sha256=$(sha $W/negx_digit.md)"
  h8_one negx_scale $W/negx_scale.md $V87 "$P8"
  h8_one negx_digit $W/negx_digit.md $V87 "$P8"
else
  print -r -- "PLANTED x_scale/x_digit SKIPPED (row 19 has no matched marker or core in the real consistency run)"
  print -r -- '{"hunk8_consistency": null}' > $RUN/out/h8_negx_scale.json; print -r -- '{"hunk8_consistency": null}' > $RUN/out/h8_negx_digit.json
fi

# ---- the fresh-reader list, AMEND1 (09 condition 6; L3515: + v19:176 and the no-core rows; deferred rows in hunks too), keys only
WSJ=$(awk '$1 == "WS" {printf "%s\"%s\":%s", (n++?",":""), $2, $3}' $WSF | sed 's/^/{/; s/$/}/')
jq -n --slurpfile r $RUN/out/real.json --rawfile ht $HT --argjson ws "$WSJ" '
  [$ht | split("\n")[] | select(startswith("HUNK ")) | split(" ") | {idx: (.[1] | tonumber), kind: .[2], x: [(.[3] | tonumber), (.[4] | tonumber)], v87: [(.[5] | tonumber), (.[6] | tonumber)]}] as $H
  | ($r[0].hunks | map({key: (.idx | tostring), value: .}) | from_entries) as $HR
  | [$H[] | . as $h | ([$r[0].reader_rows[] | select(.hunks | index($h.idx)) | {set, row, status, v87_lines}]) as $rows
     | select((($ws[$h.idx | tostring] | not) and (($HR[$h.idx | tostring].x_cores + $HR[$h.idx | tostring].v_cores) > 0))
              or ([3, 4, 5, 21] | index($h.idx)) or ($rows | length) > 0)
     | {hunk: $h.idx, kind: $h.kind, x_stripped_lines: $h.x, v87_lines: $h.v87, whitespace_only: $ws[$h.idx | tostring],
        why: ([(if (($ws[$h.idx | tostring] | not) and (($HR[$h.idx | tostring].x_cores + $HR[$h.idx | tostring].v_cores) > 0)) then "reworded_number_bearing" else empty end),
               (if ([3, 4] | index($h.idx)) then "moved_line" else empty end), (if ([5, 21] | index($h.idx)) then "punctuation_only" else empty end),
               (if ($rows | length) > 0 then "re_anchored_rows" else empty end), (if $h.idx == 8 then "hunk8_pct_reader_check_ii" else empty end)]),
        rows: $rows}]' > $RUN/out/reader_list_amend1.json || die "reader list"
print -r -- "READER_LIST_AMEND1 hunks=$(jq length $RUN/out/reader_list_amend1.json) reworded_number_bearing=$(jq '[.[] | select(.why | index("reworded_number_bearing"))] | length' $RUN/out/reader_list_amend1.json) rows=$(jq '[.[].rows[]] | length' $RUN/out/reader_list_amend1.json) no_core_rows=$(jq '[.[].rows[] | select(.status | test("^no_core"))] | length' $RUN/out/reader_list_amend1.json) has_v19_176=$(jq '[.[].rows[] | select(.set == "v19" and .row == 176)] | length > 0' $RUN/out/reader_list_amend1.json) sha256=$(sha $RUN/out/reader_list_amend1.json)"

# ---- OUTCOME (AMEND1; per check, so each flag is defined whatever the real checks gave). identity_positive must be true before anything
# else counts. Negative flags: the check the negative targets must fail as stated, and every other check must equal the real run's value.
jq -n -r --slurpfile i $RUN/out/identity.json --slurpfile r $RUN/out/real.json --slurpfile na $RUN/out/neg_a.json \
   --slurpfile nb $RUN/out/neg_b.json --slurpfile nc $RUN/out/neg_c.json --slurpfile nd $RUN/out/neg_d.json \
   --slurpfile hr $RUN/out/h8_real.json --slurpfile hd $RUN/out/h8_neg_d.json --slurpfile hs $RUN/out/h8_negx_scale.json --slurpfile hg $RUN/out/h8_negx_digit.json '
  def h8($j; $i): [$j[0].b.per_hunk[] | select(.hunk == $i) | .ok_script][0];
  {identity_positive: ($i[0].a.ok and $i[0].b.ok_script and $i[0].c.na == true and $i[0].a.rows_in_changed_hunks == 0
                       and ($i[0].a.outside_fail | length) == 0 and $i[0].cores.x == $i[0].cores.x_kept and $i[0].lines.raw_x_strips_to_x_stripped),
   real_a: $r[0].a.ok, real_b_script: $r[0].b.ok_script, real_b_h24: h8($r; 24), real_b_h29: h8($r; 29), real_b_h8_script: h8($r; 8), real_c: $r[0].c.ok,
   hunk8_consistency: $hr[0].hunk8_consistency, hunk8_forms: ($hr[0].forms | join("|")), d1_multiset_match: $r[0].d1_cross_check.all_match,
   neg_a_a_fails_on_v19_105: (($na[0].a.ok | not) and ([$na[0].a.fail[] | "\(.set):\(.row)"] == ["v19:105"])),
   neg_a_b_c_as_real: ($na[0].b.ok_script == $r[0].b.ok_script and $na[0].c.ok == $r[0].c.ok),
   neg_b_b_h24_fails: (h8($nb; 24) == false), neg_b_a_c_h8_h29_as_real: ($nb[0].a.ok == $r[0].a.ok and $nb[0].c.ok == $r[0].c.ok and h8($nb; 8) == h8($r; 8) and h8($nb; 29) == h8($r; 29)),
   neg_c_c_fails: ($nc[0].c.ok == false), neg_c_a_b_as_real: ($nc[0].a.ok == $r[0].a.ok and $nc[0].b.ok_script == $r[0].b.ok_script),
   neg_d_b_h8_fails: (h8($nd; 8) == false), neg_d_h8_consistency_fails: ($hd[0].hunk8_consistency == false),
   neg_d_a_c_h24_h29_as_real: ($nd[0].a.ok == $r[0].a.ok and $nd[0].c.ok == $r[0].c.ok and h8($nd; 24) == h8($r; 24) and h8($nd; 29) == h8($r; 29)),
   negx_scale_consistency_fails: ($hs[0].hunk8_consistency == false), negx_digit_consistency_fails: ($hg[0].hunk8_consistency == false)}
  | . + {negatives_all: ([.neg_a_a_fails_on_v19_105, .neg_a_b_c_as_real, .neg_b_b_h24_fails, .neg_b_a_c_h8_h29_as_real, .neg_c_c_fails,
                          .neg_c_a_b_as_real, .neg_d_b_h8_fails, .neg_d_h8_consistency_fails, .neg_d_a_c_h24_h29_as_real,
                          .negx_scale_consistency_fails, .negx_digit_consistency_fails] | all),
        checks_all: ([.identity_positive, .real_a, .real_b_script, .real_c, .hunk8_consistency, .d1_multiset_match] | all)}
  | "OUTCOME \(to_entries | map("\(.key)=\(.value)") | join(" ")) reader_ii=PENDING_RELAY"'
(cd $RUN && find out -type f | sort | xargs shasum -a 256 > SHA256SUMS) || die "sums"
print -r -- "SUMS $RUN/SHA256SUMS sha256 $(sha $RUN/SHA256SUMS) files=$(wc -l < $RUN/SHA256SUMS | tr -d ' ') END $(now)"
