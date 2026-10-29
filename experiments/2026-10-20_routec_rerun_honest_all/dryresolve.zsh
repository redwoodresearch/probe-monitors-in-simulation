#!/bin/zsh
# honest_all rerun: dry-resolve every path check_honest_all.py reads (PRESTATED section 9), plus the local log path.
# Uses keys, paths and shas only (record .skip.staged_sha256 keys and sha strings; routing files; Volume listings). Prints OK/STOP and counts.
# Usage: zsh dryresolve.zsh <pinned dir> <launch log path> <stamp>
set -euo pipefail
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
P=${1:?pinned dir}; LOG=${2:?launch log path}; STAMP=${3:?stamp}
ACTS=probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14
WV=probe-shift_aswin-surya_sub-honsimdet_routec_w_kill-2026-11-15
OUTV=probe-shift_aswin-surya_sub-rerun_honall-out_kill-2026-11-19
T=$P.dry
[[ -e $T ]] && { echo "STOP: $T exists"; exit 1; }
mkdir -p $T

# local paths
[[ -d $P ]] || { echo "STOP: pinned dir missing"; exit 1; }
[[ -e $LOG ]] && { echo "STOP: log $LOG exists"; exit 1; }
[[ -d ${LOG:h} ]] || { echo "STOP: log dir missing"; exit 1; }
echo "OK local: pinned dir present, log path new, log dir present"

# routes: staged file -> (extraction root, meta arm), unique among routing versions whose staged_sha equals the record's
jq -r '.skip.staged_sha256 | to_entries[] | "\(.key) \(.value)"' $P/record.json > $T/staged.txt
jq -s '[.[] | .extraction_root as $r | (.arms // [])[] | {f: .staged_inputs_file, s: .staged_sha, r: $r, a: .meta_arm}]' $P/routing/*/*.json > $T/arms.json
: > $T/routes.txt
while read -r sk sh; do
  jq -r --arg f "$sk" --arg s "$sh" '[.[] | select(.f == $f and .s == $s) | "\(.r) \(.a)"] | unique | .[]' $T/arms.json > $T/opt.txt
  [[ $(wc -l < $T/opt.txt | tr -d ' ') == 1 ]] || { echo "STOP: route for $sk"; exit 1; }
  print -r -- "$sk $(cat $T/opt.txt)" >> $T/routes.txt
done < $T/staged.txt
echo "OK routes: $(wc -l < $T/routes.txt | tr -d ' ') staged files, $(cut -d' ' -f2 $T/routes.txt | sort -u | wc -l | tr -d ' ') extraction roots"

# staged inputs files
for d in $(cut -d' ' -f1 $T/routes.txt | xargs -n1 dirname | sort -u); do
  "$M" volume ls --json $ACTS $d | jq -r '.[].filename' >> $T/staged_ls.txt
done
miss=0
while read -r sk r a; do grep -qxF "$sk" $T/staged_ls.txt || miss=$((miss + 1)); done < $T/routes.txt
[[ $miss == 0 ]] || { echo "STOP: $miss staged files missing"; exit 1; }
echo "OK staged files: all $(wc -l < $T/routes.txt | tr -d ' ') present"

# per (root, arm): meta dir; per root: chunks and by_layer/L48
for ra in ${(f)"$(cut -d' ' -f2,3 $T/routes.txt | sort -u)"}; do
  r=${ra%% *}; a=${ra##* }
  "$M" volume ls --json $ACTS $r/meta/$a | jq -r '.[].filename | split("/") | last' >> $T/meta_names.txt
done
for r in $(cut -d' ' -f2 $T/routes.txt | sort -u); do
  n1=$("$M" volume ls --json $ACTS $r/chunks | jq 'length')
  n2=$("$M" volume ls --json $ACTS $r/by_layer/L48 | jq 'length')
  [[ $n1 -gt 0 && $n2 -gt 0 && $n1 == $n2 ]] || { echo "STOP: $r chunks=$n1 L48=$n2"; exit 1; }
done
echo "OK roots: chunks and by_layer/L48 present with equal counts"

# every run key's meta file name (sha1(key)[:20].json) is present in a routed meta dir
jq -r '.run_keys[]' $P/record.json > $T/evalkeys.txt
sort -u $T/meta_names.txt > $T/meta_names_u.txt
found=0; total=0
while read -r k; do
  total=$((total + 1))
  h=$(print -rn -- "$k" | shasum -a 1 | cut -c1-20)
  grep -qxF "$h.json" $T/meta_names_u.txt && found=$((found + 1))
done < $T/evalkeys.txt
[[ $found == $total ]] || { echo "STOP: meta files found $found of $total"; exit 1; }
echo "OK meta files: $found of $total run keys"

# weights npz (names only; never any cell_*.json)
"$M" volume ls --json $WV xfer/honest_sim_detector_all/2026-10-18_121211 | jq -r '.[].filename' > $T/w_ls.txt
for c in command reply; do for p in simh_vs_b a_vs_b; do
  grep -qxF "xfer/honest_sim_detector_all/2026-10-18_121211/w_${c}_L48__${p}.npz" $T/w_ls.txt || { echo "STOP: npz $c $p"; exit 1; }
done; done
echo "OK weights: 4 L48 npz present"

# output Volume: the listing must succeed and parse (Codex r5); then absent, or present without this stamp
"$M" volume list --json > $T/vol_list.json || { echo "STOP: volume list failed"; exit 1; }
jq -e 'type == "array"' $T/vol_list.json > /dev/null || { echo "STOP: volume list is not a JSON array"; exit 1; }
present=$(jq -r --arg v $OUTV 'map(.name // .Name) | index($v) != null' $T/vol_list.json)
if [[ $present == true ]]; then
  "$M" volume ls --json $OUTV > $T/out_ls.json || { echo "STOP: output Volume listing failed"; exit 1; }
  jq -e 'type == "array"' $T/out_ls.json > /dev/null || { echo "STOP: output Volume listing is not a JSON array"; exit 1; }
  newstamp=$(jq -r --arg s $STAMP 'map(.filename | rtrimstr("/")) | index($s) == null' $T/out_ls.json)
  [[ $newstamp == true ]] || { echo "STOP: stamp exists on output Volume"; exit 1; }
  echo "OK output Volume exists; stamp $STAMP is new"
elif [[ $present == false ]]; then
  echo "OK output Volume absent (created by the app)"
else
  echo "STOP: could not decide whether the output Volume exists"; exit 1
fi
echo "DRY-RESOLVE PASS"
