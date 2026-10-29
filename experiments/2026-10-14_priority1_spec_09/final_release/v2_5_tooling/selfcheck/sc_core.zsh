#!/bin/zsh -f
# sc_core.zsh: the laptop port of selfcheck_release_record.py v2.3 (sha 2b0eaec4…): jq + zsh + git + shasum, no Python.
#   zsh -f sc_core.zsh <record.json> [<commit-message.txt>]
# Prints the Python's log format: PERQ / FAMILY / SENS / PENDING_DOWNSTREAM / V2COND lines, the SELFCHECK counts line, MISMATCH lines
# (JSON paths, ids, slot names; never a value), then SELFCHECK_OK (exit 0) or SELFCHECK_FAIL (exit 1). Exit 2 = plumbing STOP.
# - SC1 (Python L56-86): every {path, commit, sha256} / anchor / checker cite / input ref re-derived from git with shasum; origin containment.
# - SC-rest (Python L89-573): sc_rest.jq on blobs streamed from git (never written to disk); only its counts/lines/mismatch texts are kept.
# Small key-only work files (shas, line texts of GATES / READ_LOG anchors, file lists) go to selfcheck/_runs/<stamp>/ (kept; no rm).
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
REC=${1:A}; MSG=${2:-}
[[ -f $REC ]] || { echo "STOP no record file"; exit 2; }
[[ -z $MSG || -f $MSG ]] || { echo "STOP no message file"; exit 2; }
RUN=$HERE/_runs/$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_$$
mkdir -p $RUN
G() { git -C $REPO "$@"; }
sha_of() { G cat-file blob "$1" | shasum -a 256 | cut -c1-64; }

# ---------------- SC1 ----------------
jq -r -f $HERE/sc_nodes.jq $REC > $RUN/nodes.tsv
typeset -A FSHA DSHA ORIG LSHA
typeset -a BAD
integer file_refs=0 gz=0 anchors=0 cites=0 inputs=0 commits=0
: > $RUN/lines.jsonl
while IFS=$'\t' read -r kind a b c d e f; do
  case $kind in
    F) where=$a; commit=$b; p=$c; want=$d; dec=$e; ob=$f
       (( ++file_refs ))
       [[ $commit =~ '^[0-9a-f]{40}$' ]] || BAD+=("$where commit not full")
       k="$commit:$p"
       (( ${+FSHA[$k]} )) || FSHA[$k]=$(sha_of "$k")
       [[ ${FSHA[$k]} == $want ]] || BAD+=("$where sha256")
       if [[ $dec != - ]]; then
         (( ++gz ))
         (( ${+DSHA[$k]} )) || DSHA[$k]=$(G cat-file blob "$k" | gunzip -c | shasum -a 256 | cut -c1-64)
         [[ ${DSHA[$k]} == $dec ]] || BAD+=("$where decompressed_sha256")
       fi
       if (( ! ${+ORIG[$commit]} )); then
         ORIG[$commit]=$(G branch -r --contains $commit | tr -s ' \t' '\n' | awk '/^origin\//' | sort | tr '\n' ' ')
         (( ++commits ))
       fi
       [[ -n ${ORIG[$commit]// /} ]] || BAD+=("$where commit on no origin branch")
       if [[ $ob != - ]]; then
         jq -e -n --argjson rec "$ob" --arg cur "${ORIG[$commit]}" '($cur | split(" ") | map(select(length > 0))) as $c | all($rec[]; . as $x | $c | index([$x]) != null)' > /dev/null \
           || BAD+=("$where origin_branches (a recorded branch no longer contains the commit)")
       fi ;;
    A) where=$a; commit=$b; file=$c; fsha=$d; ln=$e; lsha=$f
       (( ++anchors ))
       k="$commit:$file"
       (( ${+FSHA[$k]} )) || FSHA[$k]=$(sha_of "$k")
       [[ ${FSHA[$k]} == $fsha ]] || BAD+=("$where anchor file sha")
       lk="$k:$ln"
       (( ${+LSHA[$lk]} )) || LSHA[$lk]=$(G cat-file blob "$k" | sed -n "${ln}p" | tr -d '\n' | shasum -a 256 | cut -c1-64)
       [[ ${LSHA[$lk]} == $lsha ]] || BAD+=("$where anchor line sha") ;;
    C) where=$a; commit=$b; p=$c; lo=$d; hi=$e; csha=$f
       (( ++cites ))
       seg=$(G cat-file blob "$commit:$p" | awk -v lo=$lo -v hi=$hi 'NR >= lo && NR <= hi {if (NR > lo) printf "\n"; printf "%s", $0}' | shasum -a 256 | cut -c1-64)
       [[ $seg == $csha ]] || BAD+=("$where checker cite sha") ;;
    I) where=$a; commit=$b; inp=$c; want=$d
       (( ++inputs ))
       [[ ${#${inp//[^@]/}} == 1 ]] || { echo "STOP input ref $where has not exactly one @"; exit 2; }
       p=${inp%@*}; s=${inp##*@}
       k="$commit:$p"
       (( ${+FSHA[$k]} )) || FSHA[$k]=$(sha_of "$k")
       [[ ${FSHA[$k]} == $want ]] || BAD+=("$where input sha256")
       if [[ $p == *.gz ]]; then
         (( ${+DSHA[$k]} )) || DSHA[$k]=$(G cat-file blob "$k" | gunzip -c | shasum -a 256 | cut -c1-64)
         cs=${DSHA[$k]}
       else cs=${FSHA[$k]}; fi
       [[ $cs == $s ]] || BAD+=("$where input cover sha") ;;
    N) commit=$a; file=$b; ln=$c
       G cat-file blob "$commit:$file" | sed -n "${ln}p" | tr -d '\n' | jq -R -s -c --arg k "$commit:$file:$ln" '{k: $k, v: .}' >> $RUN/lines.jsonl ;;
    *) echo "STOP unknown node kind"; exit 2 ;;
  esac
done < $RUN/nodes.tsv

# ---------------- blobs and listings for SC-rest ----------------
jq -r -f $HERE/sc_needs.jq $REC | sort -u > $RUN/needs.tsv
: > $RUN/shows.jsonl; : > $RUN/lstree.jsonl
integer nstore=0
while IFS=$'\t' read -r kind c p; do
  case $kind in
    J|JL) (( ++nstore )) ;;
    SH) G show --name-only --format= $c | tr -s ' \t' '\n' | awk 'NF' | jq -R . | jq -s -c --arg k $c '{k: $k, v: .}' >> $RUN/shows.jsonl ;;
    LT) G ls-tree -r --name-only $c "$p" | tr -s ' \t' '\n' | awk 'NF' | jq -R . | jq -s -c --arg k "$c:$p" '{k: $k, v: .}' >> $RUN/lstree.jsonl ;;
  esac
done < $RUN/needs.tsv
build_store() {
  local kind c p s
  while IFS=$'\t' read -r kind c p; do
    case $kind in
      J)  s=${FSHA[$c:$p]:-$(sha_of "$c:$p")}; G cat-file blob "$c:$p" | jq -c --arg k "$c:$p" --arg sha $s '{k: $k, sha: $sha, v: .}' ;;
      JL) s=${FSHA[$c:$p]:-$(sha_of "$c:$p")}; G cat-file blob "$c:$p" | jq -c -s --arg k "$c:$p" --arg sha $s '{k: $k, sha: $sha, v: .}' ;;
    esac
  done < $RUN/needs.tsv
}
RECSHA=$(shasum -a 256 $REC | cut -c1-64)
HASMSG=false; MSGF=/dev/null
if [[ -n $MSG ]]; then HASMSG=true; MSGF=$MSG; fi
jq -n -c -f ${SC_REST:-$HERE/sc_rest.jq} --slurpfile REC $REC --slurpfile ST <(build_store) --slurpfile LN $RUN/lines.jsonl \
   --slurpfile SH $RUN/shows.jsonl --slurpfile LT $RUN/lstree.jsonl --argjson HASMSG $HASMSG --rawfile MSG $MSGF --arg RECSHA $RECSHA \
   --argjson NSTORE $nstore > $RUN/rest.json 2> $RUN/rest.err \
  || { echo "STOP the rest program failed (stderr in $RUN/rest.err; not printed)"; exit 2; }
jq -e '.store_n == .store_expected' $RUN/rest.json > /dev/null || { echo "STOP store incomplete"; exit 2; }

# ---------------- output, in the Python's order ----------------
jq -r '.lines[]' $RUN/rest.json
NREST=$(jq -r '.n | to_entries | map("\(.key) \(.value)") | .[]' $RUN/rest.json)
typeset -A NR
while read -r k v; do NR[$k]=$v; done <<< "$NREST"
integer nbad_rest=$(jq '.bad | length' $RUN/rest.json)
integer total=$(( ${#BAD} + nbad_rest ))
line="SELFCHECK file_refs $file_refs gz_decompressed $gz anchors $anchors checker_cites $cites input_refs $inputs outputs_reclassified ${NR[outputs_reclassified]} covers_reread ${NR[covers_reread]} commits_on_origin $commits pending_strings ${NR[pending_strings]}"
for k in printed_items sensitivity_entries exclusion_entries committed_in_git_nodes counts_items message_checks; do
  if (( ${+NR[$k]} )); then line+=" $k ${NR[$k]}"; fi
done
line+=" in_scope_set $(jq '.in_scope_set' $RUN/rest.json) mismatches $total"
print -r -- "$line"
for b in "${BAD[@]}"; do print -r -- "MISMATCH $b"; done
jq -r '.bad[] | "MISMATCH " + .' $RUN/rest.json
if (( total == 0 )); then print SELFCHECK_OK; exit 0; else print SELFCHECK_FAIL; exit 1; fi
