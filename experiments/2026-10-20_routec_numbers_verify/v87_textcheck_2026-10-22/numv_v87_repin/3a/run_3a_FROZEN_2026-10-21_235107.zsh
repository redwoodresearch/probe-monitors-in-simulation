#!/bin/zsh
# (3a') the text-only numbers check of the posted v87 against the numbers-verified X (09, GATES L3501 @ dcfc03c71). zsh + jq + diff only;
# free; no Python, no Modal, no git writes. Prints keys, shas, counts, row ids and booleans only; every text-bearing intermediate is a
# file under the run's work/ folder and is never printed.
#   zsh run_3a.zsh real                  (the real run: identity positive, real check, 3 planted negatives)
#   zsh run_3a.zsh fixture <fixture dir> (a disclosed DRY RUN on synthetic files only; same code path)
setopt errexit nounset pipefail
F=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87_repin
A=$F/3a
MODE=${1:?mode}
die() { print -r -- "STOP: $*"; exit 3; }
sha() { shasum -a 256 "$1" | cut -c1-64; }
now() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z'; }
if [[ $MODE == real ]]; then
  [[ "${0:A:t}" == run_3a_FROZEN_*.zsh && ! -w "${0:A}" ]] || die "real mode runs only the frozen, read-only driver"
  EMIT=$A/emit_3a_FROZEN_2026-10-21_235107.jq; CHECK=$A/check_3a_FROZEN_2026-10-21_235107.jq
  [[ ! -w $EMIT && "$(sha $EMIT)" == 2bdaf6414829959f116fe44ce97b01e131dc96d777f1521d1135ab19130541c2 && ! -w $CHECK && "$(sha $CHECK)" == b0b130bf23f03745c1efee0edfc62543f68a09895a012ad9ff6b2ea9507b789b ]] || die "frozen jq files: sha or mode"
else EMIT=$A/emit_3a.jq; CHECK=$A/check_3a.jq; fi
RUN=$A/run_${MODE}_$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S); [[ ! -e $RUN ]] || die "run folder exists"
mkdir -p $RUN/work $RUN/out || die "mkdir"
print -r -- "START $(now) mode=$MODE run=$RUN emit_sha256=$(sha $EMIT) check_sha256=$(sha $CHECK) self_sha256=$(sha ${0:A})"

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
  typeset -A GP GS
  GP=(v19 c20699a5d:$E/post/provenance_v84_final19_X.jsonl v10 c20699a5d:$E/post/constants_v84_final10_X.jsonl
      dfr b58b25384:$E/post/deferred_v84_final8_0141.jsonl final5 d1c8ec348:$E/post/constants_sources_final5_0141_030913.jsonl
      add9 64d962a67:$E/post/constants_sources_v9_additions_0141.jsonl xres afe11c809:$E/results/numbers_verify_2026-10-21_025253.json)
  GS=(v19 a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda v10 35a780d0d69ab2f3e8217c7efc8db62106ee24acd318d678451b5791c313e6b5
      dfr 9f3260a3b4be241e53603610f0ac502386fd09cc83900a9473ca5ccf2da3d180 final5 f5f316cb84320035517f11ae34025f05991a5c473101b8ad31151cd46a674d05
      add9 eb03888f433c6b045b2eaf73d7c76958256956c70fa27bfb158e6f5282adc3f2 xres d59138c43422e2e50766611e1781737075bc9aaf9d10af1211a30c7deef7e618)
  for f_ s_ in $XS $XS_SHA $V87 $V87_SHA $HT $HT_SHA $RE $RE_SHA $R123 $R123_SHA; do [[ -f $f_ && "$(sha $f_)" == "$s_" ]] || die "input sha: ${f_:t}"; done
  [[ "$(sha $D1SUMS)" == $D1SUMS_SHA ]] || die "D1 SUMS"
  for k_ in ${(k)GP}; do git -C $REPO show "$GP[$k_]" > $RUN/work/$k_.in || die "git show $k_"; [[ "$(sha $RUN/work/$k_.in)" == "$GS[$k_]" ]] || die "pin sha $k_"; done
  W=$RUN/work
  V19=$W/v19.in; V10=$W/v10.in; DFR=$W/dfr.in; FIN5=$W/final5.in; ADD9=$W/add9.in
  R241OK=$(jq -c '.diagnostics.post.map_rows["241"] | (.ok == true and .bucket == "verified")' $W/xres.in) || die "row 241 in X's result"
else
  FX=${2:?fixture dir}; [[ -d $FX ]] || die "no fixture dir"
  XS=$FX/x.stripped; V87=$FX/v87.md; HT=$FX/hunks.txt; RE=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1/writing_core_regex.bin
  R123=$FX/r123.txt; V19=$FX/v19.jsonl; V10=$FX/v10.jsonl; DFR=$FX/dfr.jsonl; FIN5=$FX/final5.jsonl; ADD9=$FX/add9.jsonl
  R241OK=$(cat $FX/r241ok.json); W=$RUN/work
fi
print -r -- "PINS_OK mode=$MODE x=$(sha $XS) v87=$(sha $V87) hunks=$(sha $HT) regex=$(sha $RE) v19=$(sha $V19) v10=$(sha $V10) dfr=$(sha $DFR) final5=$(sha $FIN5) add9=$(sha $ADD9) r241_verified_in_X=$R241OK"

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
  jq -n -f $CHECK --rawfile xs $XS --rawfile vs $vs_ --rawfile re $RE --rawfile ht $ht_ --slurpfile cmds $d_/cmds.jsonl \
     --slurpfile v19 $V19 --slurpfile v10 $V10 --slurpfile dfr $DFR --slurpfile appr $W/appr.json \
     --argjson r241ok "$R241OK" --argjson d1cores "$D1C" --arg mode "$lab_" > $RUN/out/$lab_.json || die "check ($lab_)"
  jq -r '"RESULT \(.mode) a_ok=\(.a.ok) a_rows=\(.a.rows_in_changed_hunks) a_kept=\(.a.kept) a_fail=\([.a.fail[] | "\(.set):\(.row)"] | tojson) a_not_printed=\(.a.not_printed | tojson) a_no_core=\(.a.no_core | length) outside=\(.a.outside_rows) outside_fail=\(.a.outside_fail | length) lines_unequal=\(.lines.unchanged_unequal) runs_bad=\(.lines.runs_bad) b_ok=\(.b.ok) b=\([.b.per_hunk[] | "\(.hunk):n\(.n_added):\(.approved_rows | tojson)"] | tojson) c_ok=\(.c.ok) c_verdict=\(.c.verdict) d1_match=\(.d1_cross_check.all_match) d1_mismatch=\(.d1_cross_check.mismatched_hunks | tojson) deferred=\([.a.deferred[] | "\(.row):\(.status)"] | tojson) multi=\([.a.multi_occurrence_in_v87[] | "\(.set):\(.row):\(.status)"] | tojson)"' $RUN/out/$lab_.json
}

# ---- identity positive: X against itself, no hunks (every row must map to itself)
: > $W/empty_hunks.txt
check_one identity $XS $W/empty_hunks.txt
# ---- the real check
check_one real $V87 $HT
# ---- planted negatives (one digit changed on a copy of v87; positions from the real run's keys; the copy keeps every line break)
plant() {  # plant <label> <code point position> <rule: plus1 | not_approved>
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
check_one neg_a $W/neg_a.md $HT
check_one neg_b $W/neg_b.md $HT
check_one neg_c $W/neg_c.md $HT

# ---- the fresh-reader list (09 condition 6), keys only: every changed hunk that is not whitespace-only (D1's WS flags) and holds a core
# on either side, plus hunks 3 / 4 / 5 / 21 and every hunk a checked row is anchored in; with X / v87 line ranges and row ids
if true; then
  if [[ $MODE == real ]]; then
    WSF=$D1R/posthoc_ws_move.txt; [[ "$(sha $WSF)" == b79680b7b2aad1e64f1fbe35183b37b750b7d2a7d4285b4ae121dbfb03c28f7d ]] || die "D1 WS flags sha"
  else WSF=$FX/ws.txt; fi
  WSJ=$(awk '$1 == "WS" {printf "%s\"%s\":%s", (n++?",":""), $2, $3}' $WSF | sed 's/^/{/; s/$/}/')
  jq -n --slurpfile r $RUN/out/real.json --rawfile ht $HT --argjson ws "$WSJ" '
    [$ht | split("\n")[] | select(startswith("HUNK ")) | split(" ") | {idx: (.[1] | tonumber), kind: .[2], x: [(.[3] | tonumber), (.[4] | tonumber)], v87: [(.[5] | tonumber), (.[6] | tonumber)]}] as $H
    | ($r[0].hunks | map({key: (.idx | tostring), value: .}) | from_entries) as $HR
    | [$H[] | . as $h | ([$r[0].reader_rows[] | select(.hunks | index($h.idx)) | {set, row, v87_lines}]) as $rows
       | select((($ws[$h.idx | tostring] | not) and (($HR[$h.idx | tostring].x_cores + $HR[$h.idx | tostring].v_cores) > 0))
                or ([3, 4, 5, 21] | index($h.idx)) or ($rows | length) > 0)
       | {hunk: $h.idx, kind: $h.kind, x_stripped_lines: $h.x, v87_lines: $h.v87, whitespace_only: $ws[$h.idx | tostring],
          why: ([(if (($ws[$h.idx | tostring] | not) and (($HR[$h.idx | tostring].x_cores + $HR[$h.idx | tostring].v_cores) > 0)) then "reworded_number_bearing" else empty end),
                 (if ([3, 4] | index($h.idx)) then "moved_line" else empty end), (if ([5, 21] | index($h.idx)) then "punctuation_only" else empty end),
                 (if ($rows | length) > 0 then "re_anchored_rows" else empty end)]),
          rows: $rows}]' > $RUN/out/reader_list.json || die "reader list"
  print -r -- "READER_LIST hunks=$(jq length $RUN/out/reader_list.json) reworded_number_bearing=$(jq '[.[] | select(.why | index("reworded_number_bearing"))] | length' $RUN/out/reader_list.json) rows=$(jq '[.[].rows[]] | length' $RUN/out/reader_list.json)"
fi

# ---- outcome against the frozen predictions (the pre-statement's section 5)
jq -n -r --slurpfile i $RUN/out/identity.json --slurpfile r $RUN/out/real.json --slurpfile na $RUN/out/neg_a.json \
   --slurpfile nb $RUN/out/neg_b.json --slurpfile nc $RUN/out/neg_c.json '
  {identity_positive: ($i[0].a.ok and $i[0].a.rows_in_changed_hunks == 0 and ($i[0].a.outside_fail | length) == 0
                       and $i[0].cores.x == $i[0].cores.x_kept),
   real_a: $r[0].a.ok, real_b: $r[0].b.ok, real_c: $r[0].c.ok,
   neg_a_fails_a_only_on_v19_105: (($na[0].a.ok | not) and ([$na[0].a.fail[] | "\(.set):\(.row)"] == ["v19:105"]) and $na[0].b.ok and $na[0].c.ok),
   neg_b_fails_b_only: ($nb[0].a.ok and ($nb[0].b.ok | not) and $nb[0].c.ok),
   neg_c_fails_c_only: ($nc[0].a.ok and $nc[0].b.ok and ($nc[0].c.ok | not))}
  | "OUTCOME \(to_entries | map("\(.key)=\(.value)") | join(" ")) all=\([.[]] | all)"'
(cd $RUN && find out -type f | sort | xargs shasum -a 256 > SHA256SUMS) || die "sums"
print -r -- "SUMS $RUN/SHA256SUMS sha256 $(sha $RUN/SHA256SUMS) files=$(wc -l < $RUN/SHA256SUMS | tr -d ' ') END $(now)"
