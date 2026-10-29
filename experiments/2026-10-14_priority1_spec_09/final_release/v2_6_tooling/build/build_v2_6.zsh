#!/bin/zsh -f
# build_v2_6.zsh: builds the route C FINAL-release record v2.6 from the committed v2.5 record (4df86c4b9), by key, with build_v2_6.jq.
# Pinned inputs: evidence_v26.json (base, GATES lines, scope file, results / recheck pins; every 40-hex commit from git rev-parse) and
# v26_strings.json (09's texts, copied by script). Every pin is re-checked here against git or the file's full sha before the build.
# Prints keys, counts, shas and booleans only.   zsh -f build_v2_6.zsh <out.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}; OUT=${1:-}
[[ -n $OUT && ! -e $OUT ]] || { echo "STOP usage: zsh -f build_v2_6.zsh <new out.json>"; exit 2; }
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
G() { git -C $REPO "$@"; }
E=$HERE/evidence_v26.json; SF=$HERE/v26_strings.json; V5=$D/inputs/v2.5_record_4df86c4b9.json; SC=$D/inputs/a1_interval_scope_20261021_004310.jsonl
WORK=$HERE/_work/build_$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_$$; mkdir -p $WORK
sha() { shasum -a 256 "$1" | cut -c1-64; }
ev() { jq -r "$1" $E; }
origins() { G branch -r --contains $1 | tr -s ' \t' '\n' | awk '/^origin\//' | sort | jq -R . | jq -s -c . ; }
# base: v2.5 from git == the input file, at its pinned sha; parent pinned; on origin
[[ $(G rev-parse "$(ev .base.commit)^") == $(ev .base.parent) ]] || { echo "STOP base parent"; exit 2; }
[[ $(G cat-file blob "$(ev .base.commit):$(ev .base.path)" | shasum -a 256 | cut -c1-64) == $(ev .base.sha256) && $(sha $V5) == $(ev .base.sha256) ]] || { echo "STOP base record sha"; exit 2; }
# scope file
[[ $(sha $SC) == $(ev .scope.sha256) && $(jq -r '.row' $SC | sort -n | uniq | wc -l | tr -d ' ') == $(ev .scope.rows_n) ]] || { echo "STOP scope file"; exit 2; }
# results summary and recheck (b) from git at their commits; parents pinned; both on origin
[[ $(G rev-parse "$(ev .results.commit)^") == $(ev .results.parent) && $(G cat-file blob "$(ev .results.commit):$(ev .results.path)" | shasum -a 256 | cut -c1-64) == $(ev .results.sha256) ]] || { echo "STOP results pin"; exit 2; }
[[ $(G rev-parse "$(ev .recheck_b.commit)^") == $(ev .recheck_b.parent) && $(G cat-file blob "$(ev .recheck_b.commit):$(ev .recheck_b.path)" | shasum -a 256 | cut -c1-64) == $(ev .recheck_b.sha256) ]] || { echo "STOP recheck (b) pin"; exit 2; }
# recheck (a): the report file at its sha, with one line starting with each verdict prefix
RA=$(ev .recheck_a.file); [[ -f $RA && $(sha $RA) == $(ev .recheck_a.sha256) ]] || { echo "STOP recheck (a) report sha"; exit 2; }
jq -r '.recheck_a.verdict_lines[]' $E | while read -r vl; do [[ $(grep -c -F -- "$vl" $RA || true) -ge 1 && $(grep -c "^${vl}" $RA || true) -ge 1 ]] || { echo "STOP recheck (a) verdict line"; exit 2; }; done
# GATES lines: read by number at the pinned commits; each cite = {commit, file, file_sha256, line, line_sha256}
GC=$(ev .gates.commit); GF=$(ev .gates.file); GC78=$(ev .gates.figs78_verified.commit)
G cat-file blob "${GC}:${GF}" > $WORK/gates.md; G cat-file blob "${GC78}:${GF}" > $WORK/gates78.md
cite() { local c=$1 f=$2 ln=$3 src=$4; jq -n -c --arg c $c --arg f $f --argjson ln $ln --arg fs $(sha $src) --arg ls $(sed -n "${ln}p" $src | tr -d '\n' | shasum -a 256 | cut -c1-64) \
  '{commit: $c, file: $f, file_sha256: $fs, line: $ln, line_sha256: $ls}'; }
lt() { sed -n "${2}p" $1 | tr -d '\n' | jq -R -s -c --arg k "$3" '{k: $k, v: .}'; }
: > $WORK/lines.jsonl; : > $WORK/cites.jsonl
for k in ruling ruling_a ruling_b disclosure; do ln=$(ev ".gates.lines.$k"); lt $WORK/gates.md $ln "$k" >> $WORK/lines.jsonl
  cite $GC $GF $ln $WORK/gates.md | jq -c --arg k $k '{k: $k, v: .}' >> $WORK/cites.jsonl; done
ln=$(ev .gates.figs78_verified.line); lt $WORK/gates78.md $ln figs78_verified >> $WORK/lines.jsonl
cite $GC78 $GF $ln $WORK/gates78.md | jq -c '{k: "figs78_verified", v: .}' >> $WORK/cites.jsonl
# c2 (09 via 2e): the Fig 4 disclosure text = the one quoted string of GATES L3001 @ 89435ec9a (a line with exactly 2 straight double quotes)
GC2=$(G rev-parse 89435ec9a); G cat-file blob "${GC2}:${GF}" > $WORK/gates_c2.md
[[ $(sed -n 3001p $WORK/gates_c2.md | tr -cd '"' | wc -c | tr -d ' ') == 2 ]] || { echo "STOP GATES L3001 does not hold exactly 2 straight quotes"; exit 2; }
G merge-base --is-ancestor $GC2 origin/aswin/convergence-audit-09 || { echo "STOP 89435ec9a is not on origin/aswin/convergence-audit-09"; exit 2; }
lt $WORK/gates_c2.md 3001 fig4_disclosure >> $WORK/lines.jsonl
cite $GC2 $GF 3001 $WORK/gates_c2.md | jq -c '{k: "fig4_disclosure", v: .}' >> $WORK/cites.jsonl
# the shapes 09's rulings name (STOP if a line is not the expected one)
jq -e -s 'map({(.k): .v}) | add | (.ruling | startswith("### RULING: option (3)")) and (.ruling_a | startswith("- **RULING A:** INTERVALS REPRODUCED (EXACT)"))
  and (.ruling_b | startswith("- **RULING B:** UNRESOLVED")) and ([.disclosure | match("\"([^\"]+)\""; "g")] | length == 1)
  and (.figs78_verified | contains("results commit 854f901b6 verified"))' $WORK/lines.jsonl > /dev/null || { echo "STOP a GATES line is not the ruled one"; exit 2; }
# META: pins and builder shas
META=$(jq -n -c --arg date "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M %Z')" --arg rp "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.6_2026-10-21.json" \
  --slurpfile e $E --argjson bo "$(origins $(ev .base.commit))" --argjson ro "$(origins $(ev .results.commit))" --argjson bbo "$(origins $(ev .recheck_b.commit))" \
  --arg js $(sha $HERE/build_v2_6.jq) --arg ws $(sha $HERE/build_v2_6.zsh) --arg ss $(sha $SF) --arg es $(sha $E) \
  '{date_pt: $date, repo_path: $rp,
    v25: {path: $e[0].base.path, commit: $e[0].base.commit, sha256: $e[0].base.sha256, origin_branches: $bo},
    results: ($e[0].results | {path, commit, sha256, role} + {origin_branches: $ro}),
    recheck_b: ($e[0].recheck_b | {path, commit, sha256, role} + {origin_branches: $bbo}),
    recheck_a: ($e[0].recheck_a | {label, file, sha256, verdict_lines}),
    builder: {note: "v2.6 = v2.5 + the v2.6 rulings, built by key (build_v2_6.zsh + build_v2_6.jq) from the committed v2.5 record",
              script_sha256: $js, wrapper_sha256: $ws, strings_sha256: $ss, evidence_sha256: $es}}')
[[ $(origins $(ev .base.commit)) == *'"origin/aswin/routec-final-release"'* && $(origins $(ev .results.commit)) == *'"origin/aswin/opt3-results"'* \
   && $(origins $(ev .recheck_b.commit)) == *'"origin/aswin/opt3-results"'* ]] || { echo "STOP a pinned commit is not on its origin branch"; exit 2; }
jq -n -f $HERE/build_v2_6.jq --slurpfile V25 $V5 --slurpfile SCOPE $SC --slurpfile STR $SF --slurpfile LN $WORK/lines.jsonl --slurpfile CI $WORK/cites.jsonl \
   --argjson META "$META" > $OUT 2> $WORK/build.err || { echo "STOP build_v2_6.jq failed (stderr kept in $WORK/build.err; not printed)"; exit 3; }
print -r -- "BUILT $OUT sha256 $(sha $OUT)"
jq -r '.printed_items.all_rows | "ROWS \(length) FINAL \([.[] | select(.status | startswith("FINAL"))] | length) PRELIMINARY \([.[] | select(.status | startswith("PRELIMINARY"))] | length) NOT_ROUTE_C \([.[] | select(.status | startswith("not route C"))] | length)"' $OUT
