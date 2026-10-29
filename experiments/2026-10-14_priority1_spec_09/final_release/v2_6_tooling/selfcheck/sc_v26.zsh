#!/bin/zsh -f
# SC6 driver: the v2.6 invariants (sc_v26.jq) on a v2.6 draft against the committed v2.5 record. jq + git + shasum; no Python.
# Every pin is read from git by a short sha (expanded with git rev-parse) or from a file by its full sha; 64-character compares (B16).
# Prints counts, keys and booleans only.   zsh -f sc_v26.zsh <draft.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
REC=${1:A}; [[ -f $REC ]] || { echo "STOP usage"; exit 2; }
G() { git -C $REPO "$@"; }
sha() { shasum -a 256 "$1" | cut -c1-64; }
bsha() { G cat-file blob "$1" | shasum -a 256 | cut -c1-64; }
V25=$D/inputs/v2.5_record_4df86c4b9.json; SC=$D/inputs/a1_interval_scope_20261021_004310.jsonl
BRIEF=/Users/aswinsurya/probe-shift-reruns-2026-10-20/V26_BRIEF_implementer.md; STR=$D/build/v26_strings.json
GF=experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md
V25P=experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.5_2026-10-21.json
RSP=experiments/2026-10-21_opt3_results/opt3_summary_1792613723775_20261021_202531_7e4f75e4.json
RBP=experiments/2026-10-21_opt3_results/recheck_b/recheck_b_v3_out_2026-10-21_142703.json
RAF=/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_recheck/RECHECK_opt3_att3_134215.md
# full-sha pins (files) and short-sha pins (commits, expanded by git)
[[ $(sha $BRIEF) == f717e8f35b7098f42600783b498bd2b739745efc1d232c079f74a4c615ed4cd7 ]] || { echo "STOP brief sha"; exit 2; }
[[ $(sha $STR) == 3fcd0af09a5f566a39cbf86b62f2f295eb7b7d23031123e290f6bca167a1fe1a ]] || { echo "STOP strings sha"; exit 2; }
[[ $(sha $SC) == 957d9c75710f2f7eff3bcda0ad79df1a44550a19a59008461511dd4dd12f3c5d ]] || { echo "STOP scope sha"; exit 2; }
BASE=$(G rev-parse 4df86c4b9); GC=$(G rev-parse 4d43ba0a4); GC2=$(G rev-parse 89435ec9a); GC78=$(G rev-parse ae77708bf); RC=$(G rev-parse 7d787550b); BC=$(G rev-parse d1d74901a)
V25S=$(bsha "${BASE}:${V25P}"); RSS=$(bsha "${RC}:${RSP}"); RBS=$(bsha "${BC}:${RBP}"); RAS=$(sha $RAF)
[[ $V25S == 0d16330e8d1782809ec1c1be2154368d89edbfce81d8c26cac6c67ef833e5d30 && $(sha $V25) == $V25S ]] || { echo "STOP v2.5 record sha"; exit 2; }
PINOK=true
[[ $RSS == 8283c81a8ab584297a9ea118619a85790dfadbbaedcbc86fcfa1e0230efce333 ]] || PINOK=false
[[ $RBS == cb53eef30c58d1d3924da1d9426ffe6f07eb1a3a7b6cdca069c396bf06debfff ]] || PINOK=false
[[ $RAS == a2603440960b12e96b61b5e4d74db0eb8d81cc248feb342f908410bb89affe2f ]] || PINOK=false
[[ $(grep -c '^A: CONFIRMED' $RAF || true) -ge 1 && $(grep -c '^B: CONFIRMED' $RAF || true) -ge 1 ]] || PINOK=false
G merge-base --is-ancestor $RC origin/aswin/opt3-results && G merge-base --is-ancestor $BC origin/aswin/opt3-results || PINOK=false
G merge-base --is-ancestor $BASE origin/aswin/routec-final-release || PINOK=false
PIN=$(jq -n -c --arg gc $GC --arg gc2 $GC2 --arg gc78 $GC78 --arg gf $GF --arg base $BASE --arg v25p $V25P --arg v25s $V25S --arg rc $RC --arg rsp $RSP --arg rss $RSS \
  --arg bc $BC --arg rbp $RBP --arg rbs $RBS --arg raf $RAF --arg ras $RAS \
  '{gc: $gc, gc2: $gc2, gc78: $gc78, gf: $gf, base: {path: $v25p, commit: $base, sha256: $v25s}, results: {path: $rsp, commit: $rc, sha256: $rss},
    recheck_b: {path: $rbp, commit: $bc, sha256: $rbs}, recheck_a: {file: $raf, sha256: $ras}}')
FIG4=$(grep 'The new text, verbatim: "' $BRIEF | jq -R -r '[capture("The new text, verbatim: \"(?<t>[^\"]+)\"").t] | if length == 1 then .[0] else error("fig4") end')
READING=$(jq -r '.reading' $STR)
RUN=$HERE/_runs/sc6_$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_$$; mkdir -p $RUN
# GATES lines: the ruled ones by number, plus every cited (commit, file, line) in the draft; a line past the end of its file is left out
{ for ln in 2869 2876 2877 2878; do print -r -- "$GC	$GF	$ln"; done; print -r -- "$GC78	$GF	2430"; print -r -- "$GC2	$GF	3001"
  jq -r '[(.disclosures[] | .status_superseded_by? // empty | if type == "array" then .[] else . end | .gates_cite),
          (.row_rulings_v2_6.items[]?.cites[]?), (.disclosures[] | select(.id | startswith("v26.")) | .cite),
          (.row_rulings_v2_4.items[] | .status_superseded_by? // empty | if type == "array" then .[] else . end | .gates_cite)]
         | map(select(type == "object") | "\(.commit)\t\(.file)\t\(.line)")[]' $REC; } | sort -u > $RUN/cites.tsv
: > $RUN/lines.jsonl
while IFS=$'\t' read -r c f ln; do
  G cat-file -e "${c}:${f}" 2>/dev/null || continue
  n=$(G cat-file blob "${c}:${f}" | wc -l | tr -d ' ')
  [[ $ln =~ ^[0-9]+$ ]] && (( ln >= 1 && ln <= n )) || continue
  G cat-file blob "${c}:${f}" | sed -n "${ln}p" | tr -d '\n' | jq -R -s -c --arg k "${c}:${f}:${ln}" '{k: $k, v: .}' >> $RUN/lines.jsonl
done < $RUN/cites.tsv
jq -n -c -f $HERE/sc_v26.jq --slurpfile REC $REC --slurpfile V25 $V25 --slurpfile SCOPE $SC --slurpfile LN $RUN/lines.jsonl \
   --argjson PIN "$PIN" --arg FIG4 "$FIG4" --arg READING "$READING" --argjson PINOK $PINOK > $RUN/sc6.json 2> $RUN/sc6.err \
  || { echo "STOP sc_v26.jq failed (stderr in $RUN/sc6.err; not printed)"; exit 2; }
jq -r '"SC6 " + (.n | to_entries | map("\(.key) \(.value)") | join(" ")) + " mismatches \(.bad | length)"' $RUN/sc6.json
jq -r '.bad[] | "MISMATCH " + .' $RUN/sc6.json
if jq -e '.bad | length == 0' $RUN/sc6.json > /dev/null; then print SC6_OK; else print SC6_FAIL; exit 1; fi
