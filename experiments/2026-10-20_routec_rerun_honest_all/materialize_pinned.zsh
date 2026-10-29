#!/bin/zsh
# honest_all rerun: materialize the pinned committed inputs (PRESTATED_2026-10-20_123605.md section 2) for check_honest_all.py.
# Writes files with `git show <commit>:<path>` and checks each sha256. Prints names, counts and OK/STOP only; never file contents.
# Usage: zsh materialize_pinned.zsh <new pinned dir>
set -euo pipefail
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
OUT=${1:?usage: materialize_pinned.zsh <new pinned dir>}
[[ -e $OUT ]] && { echo "STOP: $OUT already exists"; exit 1; }
mkdir -p $OUT/routing

put() {  # name commit path sha256
  git -C $REPO show "$2:$3" > "$OUT/$1"
  [[ $(shasum -a 256 "$OUT/$1" | cut -c1-64) == $4 ]] || { echo "STOP: sha256 of $1"; exit 1; }
  echo "OK $1"
}
put record.json 596bc767e9f2bcb97ac187024ff7efd411b2d6f5 experiments/2026-10-16_honest_sim_detector_all/results/honest_all_routec_2026-10-18_121211.json 10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46
put simpool.json 104087089d65a4561f7c15634017b283edb68714 experiments/2026-10-20_routec_simpool_numbers/results/routec_simpool_numbers_2026-10-20_020501.json 2e17de0c852612d8a2b1d83468997c0cb6f787dcfee9a3c93eef6edee711fd11
put provenance.jsonl 98b5054fbf8306e54711b1a7bacd72b5f5e0af2a experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final17_0141.jsonl 220d25d21943226791637581afda92faac595a3b15b9b6d6bbd3fd1a1e575971
put release.json aee5aee0999260a25133221436e7ea1b319ca4a9 experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.2_2026-10-20.json 8ea44f19639cbee9a964f57a6003d965a171a9e9facc275a20eec891a582fa2d
put folds.json ec56720adde66baabe642a7fc842172c83737679 experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299
put pilot.json 344829af6dc4cd0473757398d542bb8e1f72f512 experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json 2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27
put sources_v2.json 596bc767e9f2bcb97ac187024ff7efd411b2d6f5 experiments/2026-10-16_honest_sim_detector_all/config/sources_v2.json d69ad9c41a091d6485cf8070b094cd62f2b820a74328e2d6540a76ffea8221ac

# the 606 routing entries of sources_v2 (role == "routing"), each at its own commit, checked against the manifest's sha256
n=0
jq -r '.files[] | select(.role == "routing") | "\(.commit) \(.path) \(.sha256)"' $OUT/sources_v2.json | while read -r c p h; do
  mkdir -p $OUT/routing/$c
  f=$OUT/routing/$c/${p:t}
  [[ -e $f ]] && { echo "STOP: duplicate routing name $c/${p:t}"; exit 1; }
  git -C $REPO show "$c:$p" > $f
  [[ $(shasum -a 256 $f | cut -c1-64) == $h ]] || { echo "STOP: routing sha256 $c/${p:t}"; exit 1; }
  n=$((n + 1))
done
cnt=$(find $OUT/routing -type f | wc -l | tr -d ' ')
[[ $cnt == 606 ]] || { echo "STOP: routing files $cnt != 606"; exit 1; }
chmod -R a-w $OUT
echo "OK routing files $cnt; pinned dir $OUT frozen (a-w)"
