#!/bin/zsh -f
# SC5 driver: the v2.5 invariants (sc_v25.jq) on a v2.5 draft against the committed v2.4 record. jq + git + shasum; no Python.
# Prints counts, keys and booleans only.   zsh -f sc_v25.zsh <draft.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
REC=${1:A}
V24=$D/inputs/v2.4_record_6a46541d5.json; P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl; P19=$D/inputs/provenance_final19_X_c20699a5d.jsonl
G() { git -C $REPO "$@"; }
[[ $(shasum -a 256 $V24 | cut -c1-64) == f14031d65dfe090f53d9e28045a200cff4b875bb41e60f8f7ebdaeba989c2c91 ]] || { echo "STOP v2.4 sha"; exit 2; }
[[ $(G cat-file blob 6a46541d5:experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.4_2026-10-21.json | shasum -a 256 | cut -c1-64) == $(shasum -a 256 $V24 | cut -c1-64) ]] || { echo "STOP v2.4 != commit 6a46541d5"; exit 2; }
[[ $(shasum -a 256 $P18 | cut -c1-16) == 13d29fb90cac3e3e ]] || { echo "STOP v18 sha"; exit 2; }
[[ $(shasum -a 256 $P19 | cut -c1-64) == a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda ]] || { echo "STOP v19 sha"; exit 2; }
[[ $(G cat-file blob c20699a5d:experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final19_X.jsonl | shasum -a 256 | cut -c1-64) == $(shasum -a 256 $P19 | cut -c1-64) ]] || { echo "STOP v19 != commit c20699a5d"; exit 2; }
RUN=$HERE/_runs/sc5_$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_$$; mkdir -p $RUN
A2OK=$(jq -n -f $HERE/a2_check.jq --slurpfile a $P18 --slurpfile b $P19)
# the cited GATES lines of the v2.5 disclosures and the draft_scope's recheck cite, read from git
jq -r '[.disclosures[] | select(.id | startswith("v25.")) | .cite] + [.row_rulings_v2_5.draft_scope.recheck_cite] + [.row_rulings_v2_5.items[].cites[]] | unique[] | "\(.commit)\t\(.file)\t\(.line)"' $REC \
  | while IFS=$'\t' read -r c f ln; do G cat-file blob "$c:$f" | sed -n "${ln}p" | tr -d '\n' | jq -R -s -c --arg k "$c:$f:$ln" '{k: $k, v: .}'; done > $RUN/lines.jsonl
# the B1-X GATES line itself (for the verifier-input phrase), by its pinned commit and line
G cat-file blob 16cd28aedc85bcb7ddf61ece0d8578144e81d9f3:experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md | sed -n '1974p' | tr -d '\n' \
  | jq -R -s -c '{k: "16cd28aedc85bcb7ddf61ece0d8578144e81d9f3:B1X:1974", v: .}' >> $RUN/lines.jsonl
# c4: build_v2_4.jq L103 at 6a46541d5 (the v2.4 $HAS78 tag-text line), by its pinned commit and line
G cat-file blob 6a46541d52b58f9626e14e7cc96339bb5d36ee3d:experiments/2026-10-14_priority1_spec_09/final_release/v2_4_tooling/build/build_v2_4.jq | sed -n '103p' | tr -d '\n' \
  | jq -R -s -c '{k: "6a46541d52b58f9626e14e7cc96339bb5d36ee3d:BUILD24:103", v: .}' >> $RUN/lines.jsonl
# the RV recheck: report sha, CONFIRMED, verifier blob sha, re-derived here
RVOK=$(jq -r '.row_rulings_v2_5.draft_scope.recheck | "\(.file)\t\(.sha256)\t\(.verdict)\t\(.of.commit)\t\(.of.path)\t\(.of.sha256)"' $REC | { IFS=$'\t' read -r f s v oc op os || true
  if [[ -f $f && $(shasum -a 256 $f | cut -c1-64) == $s && $v == CONFIRMED && $(grep -c CONFIRMED $f) -ge 1 && $(G cat-file blob "$oc:$op" | shasum -a 256 | cut -c1-64) == $os ]]; then print true; else print false; fi; })
# figs78 branch (i): 09's verified line (exactly once, at L2430 @ ae77708bf), the reason line L2426 (read by constant, not from the record's cite),
# the record from git at 854f901b6 (sha-checked) and the recheck report named by the record (sha as recorded, CONFIRMED line present)
GC=ae77708bf8fa980ec83acc0c456b89a365b95066; GPF=experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md
G cat-file -e "$GC:$GPF"
L78=$(G cat-file blob "$GC:$GPF" | grep -nF -- 'results commit 854f901b6 verified** (commit by key' | cut -d: -f1 | paste -sd ' ' -; true)
F78LINE=$([[ $L78 == 2430 ]] && print true || print false)
G cat-file blob "$GC:$GPF" | sed -n '2426p' | tr -d '\n' | jq -R -s -c --arg k "$GC:REASON:2426" '{k: $k, v: .}' >> $RUN/lines.jsonl
G cat-file blob 854f901b606cd8b44f30f87c263b441e7101a11f:experiments/2026-10-20_routec_rerun_v24figs78/results/figs78_map_2026-10-21_033233.json > $RUN/f78_record.json
F78ROK=$([[ $(shasum -a 256 $RUN/f78_record.json | cut -c1-64) == ee65f33f9a7856e1be6a6c9043345019785e0c395e1ccd0e0ac8e7f9677e35d1 ]] && print true || print false)
F78RCOK=$(jq -r '.row_rulings_v2_5.figs78.recheck // {} | "\(.file)\t\(.sha256)\t\(.verdict)"' $REC | { IFS=$'\t' read -r f s v || true
  if [[ -f $f && $(shasum -a 256 $f | cut -c1-64) == $s && $v == CONFIRMED && $(grep -c 'RECHECK_FIGS78 CONFIRMED' $f) -ge 1 ]]; then print true; else print false; fi; })
jq -n -c -f $HERE/sc_v25.jq --slurpfile REC $REC --slurpfile V24 $V24 --slurpfile P18 $P18 --slurpfile P19 $P19 --slurpfile LN $RUN/lines.jsonl \
   --argjson A2OK $A2OK --argjson RVOK $RVOK --argjson F78LINE $F78LINE --slurpfile F78R $RUN/f78_record.json --argjson F78ROK $F78ROK --argjson F78RCOK $F78RCOK \
   > $RUN/sc5.json 2> $RUN/sc5.err || { echo "STOP sc_v25.jq failed (stderr in $RUN/sc5.err; not printed)"; exit 2; }
jq -r '"SC5 " + (.n | to_entries | map("\(.key) \(.value)") | join(" ")) + " mismatches \(.bad | length)"' $RUN/sc5.json
jq -r '.bad[] | "MISMATCH " + .' $RUN/sc5.json
if jq -e '.bad | length == 0' $RUN/sc5.json > /dev/null; then print SC5_OK; else print SC5_FAIL; exit 1; fi
