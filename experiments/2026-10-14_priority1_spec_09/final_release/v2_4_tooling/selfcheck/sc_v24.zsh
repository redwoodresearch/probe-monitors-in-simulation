#!/bin/zsh -f
# SC4 driver: the v2.4 invariants (sc_v24.jq) on a v2.4 draft. jq + git + shasum; no Python. Prints counts, keys and booleans only.
#   zsh -f sc_v24.zsh <draft.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
REC=${1:A}
V23=$D/inputs/v2.3_record_80a77ecdc.json; P17=$D/inputs/provenance_final17_98b5054fb.jsonl; P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl
Q1=$D/q1/frozen_0009/q1_rowtable_new145_20261020_232829_r251cite.json
[[ $(shasum -a 256 $V23 | cut -c1-64) == ebb47e28f00596535f4fbaa5bdadb2b93b413bb768fb05e99b1abd36cac7de4f ]] || { echo "STOP v2.3 sha"; exit 2; }
[[ $(shasum -a 256 $P17 | cut -c1-64) == 220d25d21943226791637581afda92faac595a3b15b9b6d6bbd3fd1a1e575971 ]] || { echo "STOP final17 sha"; exit 2; }
[[ $(shasum -a 256 $P18 | cut -c1-16) == 13d29fb90cac3e3e ]] || { echo "STOP v18 sha"; exit 2; }
[[ $(git -C $REPO cat-file blob 58bf4a3ac:experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final18_0141.jsonl | shasum -a 256 | cut -c1-64) == $(shasum -a 256 $P18 | cut -c1-64) ]] || { echo "STOP v18 != commit A"; exit 2; }
[[ $(shasum -a 256 $Q1 | cut -c1-16) == 285278990ccff77f ]] || { echo "STOP Q1 table sha"; exit 2; }
G() { git -C $REPO "$@"; }
RUN=$HERE/_runs/sc4_$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_$$; mkdir -p $RUN
# line texts for every {file, commit, line} node in the draft (the cited GATES lines), from git
jq -r -f $HERE/sc_nodes.jq $REC | awk -F'\t' '$1 == "N"' | sort -u | while IFS=$'\t' read -r kind c f ln; do
  G cat-file blob "$c:$f" | sed -n "${ln}p" | tr -d '\n' | jq -R -s -c --arg k "$c:$f:$ln" '{k: $k, v: .}'
done > $RUN/lines.jsonl
# the record (A) entries of Figs 1/6 that carry ci95, read from the committed sources (independently of the builder's wrapper)
IDX=$(jq -r '.element_tables[] | select(.file | test("fig1_main(_R)?_figs156_routecA_source\\.json$")) | "\(.commit)\t\(.file)"' $V23 | while IFS=$'\t' read -r c p; do
  G cat-file blob "$c:$p" | jq -c --arg p "$p" '{($p): [.values | to_entries[] | select(.value.input == "record" and (.value | has("ci95"))) | .key]}'; done | jq -s -c 'add')
# the row-122 recheck (if any): report sha, verdict CONFIRMED, and the verifier result blob sha, re-derived here (null when absent)
R122OK=$(jq -r '.row_rulings_v2_4.row122_recheck | if type == "object" then "\(.file)\t\(.sha256)\t\(.verdict // "")\t\(.of.commit // "")\t\(.of.path // "")\t\(.of.sha256 // "")" else "" end' $REC \
  | { IFS=$'\t' read -r f s v oc op os || true
      if [[ -z ${f:-} ]]; then print null
      elif [[ -f $f && $(shasum -a 256 $f | cut -c1-64) == $s && $v == CONFIRMED && $(G cat-file blob "$oc:$op" | shasum -a 256 | cut -c1-64) == $os ]]; then print true
      else print false; fi; })
# R4a (09): every power number of disclosure v24.slots_21_22.1 occurs on PRESTATED_W2 L26-27 at its pinned commit (counts only; nothing printed)
PL=$(G cat-file blob 0b6bdd4e7748ae0f0696a725ff051230f042288a:experiments/2026-10-20_routec_reruns/rc1b/PRESTATED_W2_2026-10-20_145024.md | sed -n '26,27p')
PN=$(jq -r '.disclosures[] | select(.id == "v24.slots_21_22.1") | .text' $REC | { grep -oE '[0-9]+(/[0-9]+)*%' || true; } | tr '/%' '\n\n' | { grep -v '^$' || true; })
integer pn_n=0 pn_ok=0
for x in ${(f)PN}; do (( ++pn_n )); print -r -- "$PL" | grep -qE "(^|[^0-9])${x}([^0-9]|\$)" && (( ++pn_ok )) || true; done
POWEROK=$([[ $pn_n -ge 1 && $pn_ok == $pn_n ]] && print true || print false)
jq -n -c -f $HERE/sc_v24.jq --slurpfile REC $REC --slurpfile V23 $V23 --slurpfile P17 $P17 --slurpfile P18 $P18 --slurpfile Q1 $Q1 \
   --slurpfile LN $RUN/lines.jsonl --argjson IDX "$IDX" --argjson R122OK "$R122OK" --argjson POWEROK "$POWEROK" > $RUN/sc4.json 2> $RUN/sc4.err || { echo "STOP sc_v24.jq failed (stderr in $RUN/sc4.err; not printed)"; exit 2; }
jq -r '"SC4 " + (.n | to_entries | map("\(.key) \(.value)") | join(" ")) + " mismatches \(.bad | length)"' $RUN/sc4.json
jq -r '.bad[] | "MISMATCH " + .' $RUN/sc4.json
print -r -- "SC4_NOTE rows 107-251 outside the ruled sets follow the Q1 table, which the builder also reads; the independent part is the fresh reviewer's seeded sample (14/14 + 15/15) and 09's own sample"
if jq -e '.bad | length == 0' $RUN/sc4.json > /dev/null; then print SC4_OK; else print SC4_FAIL; exit 1; fi
