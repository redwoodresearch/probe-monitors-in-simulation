#!/bin/zsh -f
# build_v2_7.zsh: builds the route C FINAL-release record v2.7 from the committed v2.6 record (0db7dc714), by key, with build_v2_7.jq.
# Pinned inputs: v27_pins.json (base; the GATES commits (40-hex from git rev-parse), lines and their straight-quote counts). Every GATES line is
# read by number at its commit from git; its quote count and origin branch are asserted before the build. Pre-statement PRESTATEMENT_v27.md.
# Prints keys, counts, shas and booleans only.   zsh -f build_v2_7.zsh <new out.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; OUT=${1:-}
[[ -n $OUT && ! -e $OUT ]] || { echo "STOP usage: zsh -f build_v2_7.zsh <new out.json>"; exit 2; }
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
G() { git -C $REPO "$@"; }
E=$HERE/v27_pins.json; V6=$D/inputs/v2.6_record_0db7dc714.json
WORK=$HERE/_work/build_$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_$$; mkdir -p $WORK
sha() { shasum -a 256 "$1" | cut -c1-64; }
ev() { jq -r "$1" $E; }
origins() { G branch -r --contains $1 | tr -s ' \t' '\n' | awk '/^origin\//' | sort | jq -R . | jq -s -c . ; }
# base: v2.6 from git == the input file at its pinned sha; parent pinned; on origin
[[ $(G rev-parse "$(ev .base.commit)^") == $(ev .base.parent) ]] || { echo "STOP base parent"; exit 2; }
[[ $(G cat-file blob "$(ev .base.commit):$(ev .base.path)" | shasum -a 256 | cut -c1-64) == $(ev .base.sha256) && $(sha $V6) == $(ev .base.sha256) ]] || { echo "STOP base record sha"; exit 2; }
G merge-base --is-ancestor $(ev .base.commit) origin/$(ev .base.branch) || { echo "STOP base not on origin"; exit 2; }
# GATES lines: by number at the pinned commits; quote counts, must-strings, origin branch; cites {commit, file, file_sha256, line, line_sha256}
GF=$(ev .gates_file); : > $WORK/lines.jsonl; : > $WORK/cites.jsonl
jq -r '.lines[] | "\(.k)\t\(.c)\t\(.l)\t\(.q)\t\(.must // "")"' $E | while IFS=$'\t' read -r k c l q must; do
  G merge-base --is-ancestor $c origin/$(ev .gates_branch) || { echo "STOP a GATES commit is not on origin"; exit 2; }
  G cat-file blob "${c}:${GF}" > $WORK/g_${c}.md
  [[ $(wc -l < $WORK/g_${c}.md | tr -d ' ') -ge $l ]] || { echo "STOP a GATES line is past the end"; exit 2; }
  txt=$(sed -n "${l}p" $WORK/g_${c}.md | tr -d '\n')
  [[ $(print -rn -- "$txt" | tr -cd '"' | wc -c | tr -d ' ') == $q ]] || { echo "STOP a GATES line's quote count is not the pinned one"; exit 2; }
  [[ -z $must || $txt == *"$must"* ]] || { echo "STOP a GATES line lacks its must-string"; exit 2; }
  print -rn -- "$txt" | jq -R -s -c --arg k $k '{k: $k, v: .}' >> $WORK/lines.jsonl
  jq -n -c --arg k $k --arg c $c --arg f $GF --argjson l $l --arg fs $(sha $WORK/g_${c}.md) --arg ls $(print -rn -- "$txt" | shasum -a 256 | cut -c1-64) \
    '{k: $k, v: {commit: $c, file: $f, file_sha256: $fs, line: $l, line_sha256: $ls}}' >> $WORK/cites.jsonl
done
[[ $(jq -s -r 'map({(.k): .v}) | add | (.g97_2982 == .g8e_2982)' $WORK/lines.jsonl) == true ]] || { echo "STOP L2982 differs at 8e9c6589f / 97b6e634d"; exit 2; }
META=$(jq -n -c --arg date "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M %Z')" --arg rp "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.7_2026-10-21.json" \
  --slurpfile e $E --argjson bo "$(origins $(ev .base.commit))" \
  --arg js $(sha $HERE/build_v2_7.jq) --arg ws $(sha $HERE/build_v2_7.zsh) --arg ps $(sha $E) \
  '{date_pt: $date, repo_path: $rp,
    v26: {path: $e[0].base.path, commit: $e[0].base.commit, sha256: $e[0].base.sha256, origin_branches: $bo},
    builder: {note: "v2.7 = v2.6 + the v2.7 rulings, built by key (build_v2_7.zsh + build_v2_7.jq) from the committed v2.6 record",
              script_sha256: $js, wrapper_sha256: $ws, strings_sha256: $ps}}')
jq -n -f $HERE/build_v2_7.jq --slurpfile V26 $V6 --slurpfile LN $WORK/lines.jsonl --slurpfile CI $WORK/cites.jsonl --argjson META "$META" > $OUT 2> $WORK/build.err \
  || { echo "STOP build_v2_7.jq failed (stderr kept in $WORK/build.err; not printed)"; exit 3; }
print -r -- "BUILT $OUT sha256 $(sha $OUT)"
jq -r '.printed_items.all_rows | "ROWS \(length) FINAL \([.[] | select(.status | startswith("FINAL"))] | length) PRELIMINARY \([.[] | select(.status | startswith("PRELIMINARY"))] | length) NOT_ROUTE_C \([.[] | select(.status | startswith("not route C"))] | length)"' $OUT
jq -r '"DISCLOSURES \(.disclosures | length) PRELIMINARY_OUTPUTS \([.preliminary_outputs[].name] | join("|"))"' $OUT
