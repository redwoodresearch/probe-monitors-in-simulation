#!/bin/bash
# Route C D (P-CD1): 09's binding on expect_strings (10-17). Run once at c4081678f with INPUTS_FULL 110447 (UNDER, 153); 09's (B), 10-17:
# re-run with the TEST INPUTS_FULL that C builds, now the 4th argument (this edit), and its kept set becomes D's expect_strings.
# Candidates: the pinned jq program (sha256 54e76edbdc1e02c5cde710a7db43684f8882fe8bd6525e68aa3854479df18f67) over the 10-16 dry-run
# sidecar (8ca194c5f6197066ba55b7f4fe0386230bf444c5, sha256 fcac0d836412aa2ba734d5c055b37c80236a1015414b04ea24c21aa2cf64b3a8).
# Rule (A) (09, 10-17): the wrapper's O2-1 number-token rule; integers inside names are allowed; any decimal / exponent / % / nan / inf
# token excludes a string (the program's `embed` and `numlike`).
# Filter: a candidate is KEPT only if it occurs as a literal (git grep -F) in the runner's code or config files at C; else DROPPED.
# The 18 files: the runner's 11 code modules (rp's PW_CODE_MODULES), 3 configs the code names, 3 configs D's flags pin, and the
# INPUTS_FULL config given as the 4th argument (a path under the experiment folder, present at C).
# Count: the refused paths the 10-16 plan would give with the kept strings pinned (the same rule, in jq), against MAX_PLAN_REFUSED = 200.
# Reads code, configs and the 10-16 sidecar's plan strings (a K-class record of keys, paths and counts). Writes four new files in OUT
# (a new directory; appends only).
#   usage: filter_expect_strings.sh <C, 40 hex> <OUT dir, must not exist> <esjq program file> <config/INPUTS_FULL[_TEST]_<stamp>.json>
set -euo pipefail
set -o noclobber
C="$1"; OUT="$2"; PROG="$3"; IFULL="$4"
E=experiments/2026-10-15_fpr_sweep_507_prep
# each guard on its own line, so `set -e` stops on it (inside an && list it would not)
[[ "$C" =~ ^[0-9a-f]{40}$ ]]
[ ! -e "$OUT" ]
[ -f "$PROG" ]
[[ "$IFULL" =~ ^config/INPUTS_FULL(_TEST)?_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}\.json$ ]]
git cat-file -e "$C:$E/$IFULL"
PS_=$(printf '%s' "$(cat "$PROG")" | shasum -a 256 | cut -c1-64)
[ "$PS_" = 54e76edbdc1e02c5cde710a7db43684f8882fe8bd6525e68aa3854479df18f67 ]
SB=$(git show "8ca194c5f6197066ba55b7f4fe0386230bf444c5:$E/results/dryrun_counts_FULL_TEST_2026-10-16_105031.json" | shasum -a 256 | cut -c1-64)
[ "$SB" = fcac0d836412aa2ba734d5c055b37c80236a1015414b04ea24c21aa2cf64b3a8 ]
FILES=(run_peraction311.py run_fprsweep_507.py run_common455.py run_common710.py cutfull_decisive.py full_override.py il_arms.py
       inputs_full.py inputs_final.py inputs_final_v2.py copies.py
       config/common455.json config/membership710.json config/inputs_final_v2_pins.json
       config/INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json config/E12_BLOCK_2026-10-16_092322.json
       config/ilarms_cut32768_2026-10-16_100056.json "$IFULL")
[ "${#FILES[@]}" = 18 ]
mkdir "$OUT"
PATHS=()
for f in "${FILES[@]}"; do
  p="$E/$f"; b=$(git rev-parse "$C:$p")
  printf '%s@%s\n' "$p" "$b" >> "$OUT/files.txt"
  PATHS+=("$p")
done
git show "8ca194c5f6197066ba55b7f4fe0386230bf444c5:$E/results/dryrun_counts_FULL_TEST_2026-10-16_105031.json" \
  | jq -c "$(cat "$PROG")" > "$OUT/candidates.json"
[ "$(jq length "$OUT/candidates.json")" = 105 ]
jq -r '.[]' "$OUT/candidates.json" | while IFS= read -r s; do
  if git grep -q -F -e "$s" "$C" -- "${PATHS[@]}"; then printf '%s\n' "$s" >> "$OUT/kept.txt"; else printf '%s\n' "$s" >> "$OUT/dropped.txt"; fi
done
touch "$OUT/kept.txt" "$OUT/dropped.txt"  # either may be empty
NK=$(grep -c . "$OUT/kept.txt" || true); ND=$(grep -c . "$OUT/dropped.txt" || true)
[ $((NK + ND)) = 105 ]
RJQ='def hexoff: gsub("(?<![0-9a-f.])([0-9a-f]{64}|[0-9a-f]{40})(?![0-9a-f])"; "x"); def numlike: test("^[+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][+-]?[0-9]+)?$|^[+-]?(nan|inf|infinity)$"; "i") and (test("^([0-9a-f]{40}|[0-9a-f]{64})$") | not); def embed: (hexoff | test("[.][0-9]|[0-9][eE][+-]?[0-9]|%|(?<![A-Za-z0-9])(nan|inf(inity)?)(?![A-Za-z0-9])"; "i")); def refused: (test("^[A-Za-z0-9._/@|:-]{1,400}$") | not) or numlike or embed; ($k | split("\n") | map(select(length > 0))) as $pin | ([.plan | .. | objects | keys[] | select(refused and (. as $s | $pin | index($s) | not))] | length) + ([.plan | paths(type == "string") as $q | getpath($q) | select(refused and (. as $s | $pin | index($s) | not))] | length)'
NR=$(git show "8ca194c5f6197066ba55b7f4fe0386230bf444c5:$E/results/dryrun_counts_FULL_TEST_2026-10-16_105031.json" | jq --rawfile k "$OUT/kept.txt" "$RJQ")
IB=$(git rev-parse "$C:$E/$IFULL")
printf 'C %s inputs_full %s@%s files %s kept %s dropped %s expected_refused_on_1016_plan %s cap 200 %s\n' "$C" "$IFULL" "$IB" "${#PATHS[@]}" "$NK" "$ND" "$NR" "$([ "$NR" -le 200 ] && echo UNDER || echo OVER)"
