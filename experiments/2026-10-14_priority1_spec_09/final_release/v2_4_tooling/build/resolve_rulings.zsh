#!/bin/zsh -f
# resolve_rulings.zsh: turn rulings_v2_4.json's cites, disclosure lines and evidence pins into git-derived, sha-pinned JSON.
#   zsh -f resolve_rulings.zsh <rulings_v2_4.json> <out.json>
# - each GATES must-contain string -> the unique line (exactly 1 match, else STOP) -> {file, commit (full), file_sha256, line, line_sha256}
#   (the anchor form the self-check re-derives). No line text is stored here (keys and shas only, 09 via 2e 01:08): the builder reads
#   each cited line from git and re-checks its line_sha256. The work folder (with its GATES copy) stays under _work/ and is never committed.
# - each evidence {path, commit} -> {path, commit (full), sha256, origin_branches} (+ decompressed_sha256 for .gz).
# - each recheck report given as a local file -> {file, sha256, label "from agent summary"} (sha prefix asserted).
# Prints counts and shas only.
set -e; setopt pipefail; export LC_ALL=C
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
RUL=${1:A}; OUT=${2:A}
[[ -f $RUL ]] || { echo "STOP no rulings file"; exit 2; }
G() { git -C $REPO "$@"; }
GP=$(jq -r '.gates.path' $RUL); GC=$(G rev-parse --verify "$(jq -r '.gates.commit' $RUL)^{commit}")
GSHA=$(G cat-file blob "$GC:$GP" | shasum -a 256 | cut -c1-64)
WORK=$(mktemp -d "${OUT:h}/_resolve.XXXXXX")
G cat-file blob "$GC:$GP" > $WORK/gates.md
origins() { G branch -r --contains $1 | tr -s ' \t' '\n' | awk '/^origin\//' | sort | jq -R . | jq -s -c . ; }
: > $WORK/anchors.jsonl
jq -r '[.items[] | (.cites[]?, .disclosure_lines[]?, (.rechecks[]? | .gates_must // empty), (.label_line // empty), (.counts_status_cite // empty), (.quote_lines[]? | .must))] | unique[]' $RUL | while IFS= read -r s; do
  n=$(grep -cF -- "$s" $WORK/gates.md || true)
  [[ $n == 1 ]] || { echo "STOP a GATES must-contain string matches $n lines"; exit 3; }
  ln=$(grep -nF -- "$s" $WORK/gates.md | cut -d: -f1)
  lsha=$(sed -n "${ln}p" $WORK/gates.md | tr -d '\n' | shasum -a 256 | cut -c1-64)
  jq -n -c --arg must "$s" --arg file "$GP" --arg commit $GC --arg fsha $GSHA --argjson line $ln --arg lsha $lsha \
    '{must: $must, anchor: {file: $file, commit: $commit, file_sha256: $fsha, line: $line, line_sha256: $lsha}}' >> $WORK/anchors.jsonl
done
: > $WORK/evidence.jsonl
jq -r '.items[] | .evidence[]? | "\(.path)\t\(.commit)"' $RUL | sort -u | while IFS=$'\t' read -r p c; do
  cf=$(G rev-parse --verify "$c^{commit}")
  s=$(G cat-file blob "$cf:$p" | shasum -a 256 | cut -c1-64)
  if [[ $p == *.gz ]]; then d=$(G cat-file blob "$cf:$p" | gunzip -c | shasum -a 256 | cut -c1-64); else d=""; fi
  o=$(origins $cf)
  [[ $(print -r -- "$o" | jq 'length') -gt 0 ]] || { echo "STOP evidence commit on no origin branch"; exit 3; }
  jq -n -c --arg key "$p@$c" --arg path "$p" --arg commit $cf --arg sha $s --arg dec "$d" --argjson ob "$o" \
    '{key: $key, path: $path, commit: $commit, sha256: $sha, origin_branches: $ob} + (if $dec != "" then {decompressed_sha256: $dec} else {} end)' >> $WORK/evidence.jsonl
done
: > $WORK/rechecks.jsonl
jq -r '.items[] | .rechecks[]? | select(.file) | "\(.file)\t\(.sha256_prefix)"' $RUL | sort -u | while IFS=$'\t' read -r f pre; do
  [[ -f $f ]] || { echo "STOP recheck report missing"; exit 3; }
  s=$(shasum -a 256 $f | cut -c1-64)
  [[ $s == $pre* ]] || { echo "STOP recheck report sha prefix"; exit 3; }
  jq -n -c --arg f "$f" --arg s $s '{file: $f, sha256: $s, label: "from agent summary", committed_in_git: false}' >> $WORK/rechecks.jsonl
done
jq -n -c --arg gp "$GP" --arg gc $GC --arg gs $GSHA --slurpfile A $WORK/anchors.jsonl --slurpfile E $WORK/evidence.jsonl --slurpfile K $WORK/rechecks.jsonl \
  '{gates: {path: $gp, commit: $gc, sha256: $gs}, anchors: ($A | map({(.must): .}) | add), evidence: ($E | map({(.key): del(.key)}) | add),
    rechecks: ($K | map({(.file): .}) | add)}' > $OUT
print -r -- "RESOLVED anchors $(jq '.anchors | length' $OUT) evidence $(jq '.evidence | length' $OUT) rechecks $(jq '.rechecks | length' $OUT) gates $GC out_sha256 $(shasum -a 256 $OUT | cut -c1-64)"
