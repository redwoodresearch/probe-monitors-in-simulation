#!/bin/zsh
# D1: the free, keys-only diff gate X -> post-facing v87 (09, GATES @ ce614dc45 step (1); PRESTATED_D1_FROZEN_2026-10-21_205113.md, sha256
# 7f3002d8...). BUILT 2026-10-21 20:5x PDT by the numv_v87 builder. No Modal, no Python, no git write; reads git objects with `git show` only.
# PLACEHOLDERS: 2, each a name wrapped in two underscores on each side in the assignments below: V87_PATH and V87_SHA256. This file refuses
#   to run, in every mode, while either remains, and it runs only as a read-only copy named d1_gate_FROZEN_<stamp>.zsh in this folder.
# Modes (one argument):
#   c1   = positive control: X vs draft 1746 -> expected 0 hunks;
#   c2   = rehearsal negative: X vs a copy of X with one visible digit changed (the copy checked to differ by exactly 1 byte) -> expected exactly
#          1 hunk, 1 / 1 lines, digit-bearing tokens 1 removed / 1 added;
#   c3   = real negative: the same on v87;
#   real = X -> v87, with the row mapping (d1_map.jq) and the class.
# Strip = VP.HIDDEN's own pattern, extracted by key from verify_post.py @ f11453af9 (blob 027b0372) each run, its sha256 pinned; applied with jq
#   1.7.x gsub(pattern; ""; "m"): the flag maps Python's re.S ('.' matches newline) to Oniguruma's ONIG_OPTION_MULTILINE; synthetic self-tests
#   every run; STOP rather than translate if the pattern leaves the whitelist (literal text around one lazy .*?).
# Class (09): 0 hunks -> NO_DIFF; no hunk adds a digit-bearing token -> CARRY_OVER_SUBSET; any hunk adds one -> STOP (then option (3) is scoped).
# Prints keys, shas, counts, line numbers, row ids and booleans only; never text. The diff and stripped files stay in this folder.
# Exit: 0 = finished (the class or control verdict is printed), 6 = a control did not hold, 3 = this gate's own STOP.
# Run: cd /Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1 && zsh -n <frozen file> && zsh <frozen file> <mode> > run_d1_<mode>_$(date +%H%M%S).log 2>&1
setopt errexit nounset pipefail
export LC_ALL=C

D1=/Users/aswinsurya/probe-shift-reruns-2026-10-20/numv_v87/d1
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
E=experiments/2026-10-20_routec_numbers_verify
# ---- the 2 placeholders
V87=/Users/aswinsurya/probe-shift-writing-handoff-2026-10-15/w23/drafts/draft_v87_final_frozen_0029.md
V87_SHA=75af0e8e5a10cf762bc4dc327ced8c1eb1386b64c5d680c4e7102086bad3e512
# ---- fixed pins
BASEC=f11453af93dee8a32bd7e4befb90d8a590341033
VP_PATH=$E/verify_post.py; VP_BLOB=027b037246aea6e4d1159d910eb8be152b1d438b; VP_SHA=dbfc5911a38d1cb5a98a37b79828f2f6b8bef4de847adba2a6c822a74ec65d1f
PAT_SHA=39d3d18fd21298dd2ea96526389f29db50575fe2b20f99d3306ddf6b269e79fa
C1=c20699a5d4d4aed5344650fec54d9c166f421296
OX=$E/post/draft_v84_X_0141_appC0029.md;         XSHA=4d01234cabb79ff09695c0dffec2183e805c3f4129914dc6a7f62b423b4336fa
OV19=$E/post/provenance_v84_final19_X.jsonl;     V19SHA=a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda
OV10=$E/post/constants_v84_final10_X.jsonl;      V10SHA=35a780d0d69ab2f3e8217c7efc8db62106ee24acd318d678451b5791c313e6b5
DC=b58b253840ff8244ab92ea8fad968fec7bba8b72
ODF=$E/post/deferred_v84_final8_0141.jsonl;      DFSHA=9f3260a3b4be241e53603610f0ac502386fd09cc83900a9473ca5ccf2da3d180
D1746=/Users/aswinsurya/probe-shift-writing-handoff-2026-10-15/w23/drafts/draft_v84_frozen_1746.md
D1746_SHA=a31684f332ef735b196030325a4918df78e23df994a2d9212298a6dc9ed9dd8d
AWKF=$D1/d1_hunks.awk;  AWK_SHA=37e371c10d4a92397fa5355c07c32b005bfe274883fe49eaa19958559f711404
MAPF=$D1/d1_map.jq;     MAP_SHA=9a77971a4f087d2ef5d0c1aa56a67273af883fd40629259a70aafa10a85939dd
PH='__V87_(PATH|SHA256)__'

die() { print -r -- "STOP: $*"; exit 3; }
sha() { shasum -a 256 "$1" | cut -c1-64; }
now() { date '+%Y-%m-%d %H:%M:%S %Z'; }

# ---- 0. this frozen file; no placeholder; forms; the pinned helper files; the mode
RUNME=${0:A}
[[ "${RUNME:h}" == "$D1" && "${RUNME:t}" == d1_gate_FROZEN_*.zsh && -f "$RUNME" && ! -w "$RUNME" ]] || die "run the frozen, read-only copy from $D1"
rc_=0; grep -q -E -- "$PH" "$RUNME" || rc_=$?
[[ $rc_ == 1 ]] || die "a v87 placeholder remains in this file (or grep error, rc=$rc_)"
[[ "$V87" == /* && "$V87" == *.md && "$V87_SHA" =~ '^[0-9a-f]{64}$' && "$AWK_SHA" =~ '^[0-9a-f]{64}$' && "$MAP_SHA" =~ '^[0-9a-f]{64}$' ]] \
  || die "a filled value has the wrong form"
for f_ s_ in $AWKF $AWK_SHA $MAPF $MAP_SHA; do [[ -f $f_ && ! -L $f_ && ! -w $f_ && "$(sha $f_)" == "$s_" ]] || die "frozen helper sha: ${f_:t}"; done
(( $# == 1 )) && [[ "$1" == (c1|c2|c3|real) ]] || die "usage: one mode, c1 | c2 | c3 | real"
MODE=$1
[[ "$(jq --version)" == jq-1.7.* ]] || die "jq is not 1.7.x (the flag mapping and self-tests were fixed for jq 1.7.1)"
W=$D1/run_${MODE}_$(date +%Y-%m-%d_%H%M%S); [[ ! -e "$W" ]] || die "run folder exists"; mkdir -p "$W" || die "mkdir $W"
print -r -- "START $(now) mode=$MODE runme=$RUNME sha256 $(sha $RUNME)"

# ---- 1. VP.HIDDEN's pattern, by key, from the pinned verifier blob; whitelist; sha pin; synthetic self-tests of the flag mapping
[[ "$(git -C $REPO rev-parse $BASEC:$VP_PATH)" == $VP_BLOB ]] || die "verify_post.py at the base config's commit is not blob $VP_BLOB"
git -C $REPO cat-file blob $VP_BLOB > $W/verify_post.py || die "git cat-file"; [[ "$(sha $W/verify_post.py)" == $VP_SHA ]] || die "verify_post.py sha"
[[ "$(grep -c -E '(^|[^A-Za-z_.])HIDDEN[[:space:]]*=[^=]' $W/verify_post.py || true)" == 1 ]] || die "HIDDEN is not assigned exactly once"
[[ "$(grep -c -E '^HIDDEN = re\.compile\(r"[^"]*", re\.S\)$' $W/verify_post.py || true)" == 1 ]] || die "HIDDEN's line is not the expected form (raw string, flags exactly re.S)"
PAT=$(sed -n -E 's/^HIDDEN = re\.compile\(r"([^"]*)", re\.S\)$/\1/p' $W/verify_post.py) || die "sed"
[[ "$PAT" =~ '^[^].*?+(){}|^$\\[]+\.\*\?[^].*?+(){}|^$\\[]+$' ]] || die "the pattern leaves the whitelist (literal text around one lazy .*?): not expressible without translation"
PATF=$W/hidden_pattern.bin; printf '%s' "$PAT" > $PATF || die "write pattern"; [[ "$(sha $PATF)" == $PAT_SHA ]] || die "the pattern's sha is not the pinned one"
jq -n -e --rawfile p $PATF '
  ("a<!--x\ny-->b" | gsub($p; ""; "m")) == "ab"
  and ("a<!--x\ny-->b" | gsub($p; "")) == "a<!--x\ny-->b"
  and ("a<!--x-->b<!--y-->c" | gsub($p; ""; "m")) == "abc"
  and ("a<!--x" | gsub($p; ""; "m")) == "a<!--x"
  and ("a<!---->b" | gsub($p; ""; "m")) == "ab"
  and ("a<!--x--><!--y-->b" | gsub($p; ""; "m")) == "ab"
  and ("a<!--x-->y-->b" | gsub($p; ""; "m")) == "ay-->b"
  and ("é<!--ü\n-->ß" | gsub($p; ""; "m")) == "éß"' >/dev/null 2>&1 || die "a flag-mapping self-test failed"
print -r -- "PATTERN_OK by_key=true whitelist=true pattern_sha256=$PAT_SHA flags=re.S->m selftests=8/8"

# ---- 2. inputs, by sha, copied into this run's folder
src() {  # src <path> <sha> <dst>: a read-only, non-symlink file with this sha, copied and re-checked
  [[ -f "$1" && ! -L "$1" && ! -w "$1" && "$(sha $1)" == "$2" ]] || die "source not read-only / sha: ${1:t}"
  cp "$1" "$3" || die "cp ${1:t}"; [[ "$(sha $3)" == "$2" ]] || die "copy sha ${3:t}"
}
gsrc() { git -C $REPO show "$1" > "$3" || die "git show ${1:t}"; [[ "$(sha $3)" == "$2" ]] || die "git copy sha ${3:t}"; }
strip() {  # strip <in> <out>: VP.HIDDEN.sub("", text) via jq; strict UTF-8 first (the verifier decodes strictly)
  iconv -f UTF-8 -t UTF-8 "$1" > /dev/null 2>&1 || die "not valid UTF-8: ${1:t}"
  jq -n -j --rawfile t "$1" --rawfile p $PATF '$t | gsub($p; ""; "m")' > "$2" || die "strip ${1:t}"
}
ncom() { jq -n --rawfile t "$1" --rawfile p $PATF '[$t | match($p; "gm")] | length'; }
gsrc $C1:$OX $XSHA $W/x.md
case $MODE in
  c1)   src $D1746 $D1746_SHA $W/other.md; BASE=$W/x.md ;;
  c2)   BASE=$W/x.md ;;
  c3)   src $V87 $V87_SHA $W/v87.md; BASE=$W/v87.md ;;
  real) src $V87 $V87_SHA $W/other.md; BASE=$W/x.md ;;
esac
strip $BASE $W/base.stripped

# one visible digit changed (c2 / c3): the first digit byte whose change alters the stripped text; the copy must differ by exactly 1 byte
if [[ $MODE == (c2|c3) ]]; then
  OFFS=($(awk '{ for (i = 1; i <= length($0); i++) if (substr($0, i, 1) ~ /[0-9]/) { print off + i - 1; if (++n >= 400) exit }; off += length($0) + 1 }' $BASE))   # awk stops itself: no SIGPIPE under pipefail
  (( ${#OFFS} > 0 )) || die "no digit byte in the base file"
  BS_=$(sha $W/base.stripped); DONE_=0; TRIES_=0
  for o_ in $OFFS; do
    TRIES_=$((TRIES_+1))
    d_=$(dd if=$BASE bs=1 skip=$o_ count=1 2>/dev/null); [[ "$d_" == [0-9] ]] || die "the byte at a digit offset is not a digit"
    { head -c $o_ $BASE; printf '%s' $(( (d_ + 1) % 10 )); tail -c +$((o_ + 2)) $BASE; } > $W/try.md || die "build the copy"
    ND_=$( { cmp -l $BASE $W/try.md || true; } | wc -l | tr -d ' ')   # cmp exits 1 on a difference; counted, never fatal here
    [[ "$(wc -c < $BASE)" == "$(wc -c < $W/try.md)" && "$ND_" == 1 ]] || die "the copy does not differ by exactly 1 byte"
    strip $W/try.md $W/try.stripped
    if [[ "$(sha $W/try.stripped)" != "$BS_" ]]; then mv $W/try.md $W/other.md || die "mv"; DONE_=1; break; fi
  done
  (( DONE_ == 1 )) || die "no visible digit among the first ${#OFFS} digit bytes"
  print -r -- "ONE_DIGIT_COPY byte_diff=1 same_size=true visible=true tries=$TRIES_ base_sha256=$(sha $BASE) copy_sha256=$(sha $W/other.md)"
fi
strip $W/other.md $W/other.stripped
XNL_=$(tr -cd '\n' < $BASE | wc -c | tr -d ' '); XSNL_=$(tr -cd '\n' < $W/base.stripped | wc -c | tr -d ' ')
print -r -- "D1_STRIPPED base_raw_sha256=$(sha $BASE) base_stripped_sha256=$(sha $W/base.stripped) base_comments=$(ncom $BASE) other_raw_sha256=$(sha $W/other.md) other_stripped_sha256=$(sha $W/other.stripped) other_comments=$(ncom $W/other.md) base_line_numbers_kept=$([[ $XNL_ == $XSNL_ ]] && echo true || echo false)"

# ---- 3. diff by script; hunks by d1_hunks.awk (counts only)
rc_=0; diff $W/base.stripped $W/other.stripped > $W/diff.txt || rc_=$?
[[ $rc_ == 0 || $rc_ == 1 ]] || die "diff error rc=$rc_"
awk -f $AWKF $W/diff.txt > $W/hunks.txt || die "awk"
TL_=$(grep -E '^TOTAL ' $W/hunks.txt) || die "no TOTAL line"; NH_=$(print -r -- "$TL_" | cut -d' ' -f2); BAD_=$(print -r -- "$TL_" | cut -d' ' -f3)
[[ "$BAD_" == 0 ]] || die "the diff has $BAD_ unparsed lines"
[[ "$(grep -c -E '^HUNK ' $W/hunks.txt || true)" == "$NH_" ]] || die "hunk count"
[[ ( $rc_ == 0 && $NH_ == 0 ) || ( $rc_ == 1 && $NH_ -gt 0 ) ]] || die "diff rc and hunk count disagree"
SUMA_=0; SUMR_=0; NADD_=0
while read -r t_ i_ k_ os_ oe_ ns_ ne_ lr_ la_ r_ a_ to_ tn_; do
  print -r -- "D1_HUNK idx=$i_ kind=$k_ base_lines=$os_-$oe_ other_lines=$ns_-$ne_ lines_removed=$lr_ lines_added=$la_ digit_tokens_removed=$r_ digit_tokens_added=$a_ side_totals=$to_/$tn_"
  SUMA_=$((SUMA_ + a_)); SUMR_=$((SUMR_ + r_)); if (( a_ > 0 )); then NADD_=$((NADD_ + 1)); fi
done < <(grep -E '^HUNK ' $W/hunks.txt)
if (( NH_ == 0 )); then CLS_=NO_DIFF; elif (( NADD_ == 0 )); then CLS_=CARRY_OVER_SUBSET; else CLS_=STOP; fi
print -r -- "D1_SUMMARY hunks=$NH_ digit_tokens_removed=$SUMR_ digit_tokens_added=$SUMA_ hunks_adding_digit_tokens=$NADD_ class=$CLS_"

# ---- 4. the row mapping (real only), the control verdicts
if [[ $MODE == real ]]; then
  gsrc $C1:$OV19 $V19SHA $W/v19.jsonl; gsrc $C1:$OV10 $V10SHA $W/v10.jsonl; gsrc $DC:$ODF $DFSHA $W/deferred.jsonl
  { grep -E '^HUNK ' $W/hunks.txt || true; } | jq -R -s -c 'split("\n") | map(select(length > 0) | split(" ") | {idx: (.[1] | tonumber), kind: .[2], os: (.[3] | tonumber), oe: (.[4] | tonumber)})' > $W/hunks.json || die "hunks json"
  jq -n -r -f $MAPF --rawfile xs $W/base.stripped --rawfile vs $W/other.stripped --slurpfile v19 $W/v19.jsonl --slurpfile dfr $W/deferred.jsonl \
     --slurpfile cons $W/v10.jsonl --slurpfile hunks $W/hunks.json > $W/map.txt || die "map jq"
  cat $W/map.txt
fi
CTRL_=none
case $MODE in
  c1) [[ $NH_ == 0 && "$(sha $W/base.stripped)" == "$(sha $W/other.stripped)" ]] && CTRL_=true || CTRL_=false ;;
  c2|c3) [[ $NH_ == 1 && "$(grep -E '^HUNK ' $W/hunks.txt | cut -d' ' -f8-11)" == "1 1 1 1" ]] && CTRL_=true || CTRL_=false ;;
esac
print -r -- "D1_DONE mode=$MODE control_pass=$CTRL_ class=$CLS_ folder=$W $(now)"
[[ $CTRL_ != false ]] || exit 6
exit 0
