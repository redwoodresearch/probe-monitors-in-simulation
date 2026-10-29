#!/bin/zsh
# D1 R2 / R3 / R1 checks (09, GATES @ 3bd22fad4; PRESTATED_D1_R123_ADDENDUM_FROZEN_2026-10-21_220153.md, sha256 a5fca785...). Free: zsh + jq,
# no Python / Modal / git write. Prints and writes keys, shas (64 hex), counts and booleans only; never a line, core or text.
#   cd /Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1 && zsh -n d1_r123_check.zsh && zsh d1_r123_check.zsh
setopt errexit nounset pipefail
export LC_ALL=C
D1=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1
R=$D1/run_real_2026-10-21_205711
XS=$R/base.stripped;  XS_SHA=92b6a030d69275b1c6bbe86c91f900a0c4be27cf0f599764937cce62e8fd373b
VS=$R/other.stripped; VS_SHA=75af0e8e5a10cf762bc4dc327ced8c1eb1386b64c5d680c4e7102086bad3e512
HK=$R/hunks.txt;      HK_SHA=74495f1a3dc4bf43a524b2cce5deddee99e7dcd4e44a7592718bbfa91d510640
RE=$D1/writing_core_regex.bin; RE_SHA=9414451d677f0d34b11fc2d4d80fc1b3f880c82445a5d4d64da156ddfc913bf3
CJ=$D1/d1_cores.jq
die() { print -r -- "STOP: $*"; exit 3; }
sha() { shasum -a 256 "$1" | cut -c1-64; }
for f_ s_ in $XS $XS_SHA $VS $VS_SHA $HK $HK_SHA $RE $RE_SHA; do [[ -f $f_ && ! -L $f_ && "$(sha $f_)" == "$s_" ]] || die "input sha: ${f_:t}"; done
[[ -f $CJ && ! -w $CJ ]] || die "d1_cores.jq not frozen"
OUT=$D1/R123_out_$(date +%Y-%m-%d_%H%M%S).txt; [[ ! -e $OUT ]] || die "output exists"

# hunk 3 / 4 shapes from the frozen hunks.txt: 3 = c (X 7 -> v87 5), 4 = d (X 9-10, no added line)
H3_=$(grep -E '^HUNK 3 ' $HK | cut -d' ' -f3-7) || die "hunk 3"; H4_=$(grep -E '^HUNK 4 ' $HK | cut -d' ' -f3-7) || die "hunk 4"
[[ "$H3_" == "c 7 7 5 5" && "$H4_" == "d 9 10 6 6" ]] || die "hunks 3 / 4 are not the frozen shapes"
lsha() { jq -n -j --rawfile t "$1" --argjson n $2 '$t | split("\n") | .[$n - 1]' | shasum -a 256 | cut -c1-64; }   # the line's bytes, no newline
L3R_=$(lsha $XS 7); L3A_=$(lsha $VS 5); L4R1_=$(lsha $XS 9); L4R2_=$(lsha $XS 10)
REV_=false; [[ "$L3A_" == "$L4R1_" || "$L3A_" == "$L4R2_" ]] && REV_=true

grep -E '^HUNK ' $HK | jq -R -s -c 'split("\n") | map(select(length > 0) | split(" ")
  | {idx: (.[1] | tonumber), kind: .[2], os: (.[3] | tonumber), oe: (.[4] | tonumber), ns: (.[5] | tonumber), ne: (.[6] | tonumber)})' > $D1/_hunks_r123.json || die "hunks json"
CO_=$(jq -n -r -f $CJ --rawfile xs $XS --rawfile vs $VS --rawfile re $RE --slurpfile hunks $D1/_hunks_r123.json) || die "cores jq"
R35_=$(print -r -- "$CO_" | grep -c -E '^R3 hunk=5 .* equal=true$' || true); R321_=$(print -r -- "$CO_" | grep -c -E '^R3 hunk=21 .* equal=true$' || true)
COND_=$(print -r -- "$CO_" | grep -E '^R1_ADDED_CORE_CONDITION ' | cut -d' ' -f2)
MOOT0_=$(print -r -- "$CO_" | grep -c -E '^MOOT nonascii_decimal_digits=0 ' || true)
R3_=false; [[ $R35_ == 1 && $R321_ == 1 && $MOOT0_ == 1 ]] && R3_=true
R1REV_=false; [[ $REV_ == true && $R3_ == true && "$COND_" == true ]] && R1REV_=true
{
  print -r -- "R2_LINE_SHAS h3_removed_X7=$L3R_ h3_added_v87_5=$L3A_ h4_removed_X9=$L4R1_ h4_removed_X10=$L4R2_ h4_added=none"
  print -r -- "R2 literal_h3_removed_eq_h4_added=false (not evaluable: hunk 4 has no added line) reversed_h3_added_eq_a_h4_removed=$REV_ h3_removed_eq_h3_added=$([[ $L3R_ == $L3A_ ]] && echo true || echo false)"
  print -r -- "$CO_"
  print -r -- "R3 pass=$R3_"
  print -r -- "R1 literal=false reversed=$R1REV_ (09 rules which R2 reading applies)"
  print -r -- "INPUTS x_stripped=$XS_SHA v87_stripped=$VS_SHA hunks=$HK_SHA regex=$RE_SHA cores_jq=$(sha $CJ) check=$(sha ${0:A})"
} > $OUT || die "write output"
chmod a-w $OUT
cat $OUT
print -r -- "OUTPUT $OUT sha256 $(sha $OUT)"
