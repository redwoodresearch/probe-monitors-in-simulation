#!/bin/zsh -f
# SC7 driver: the v2.7 invariants (sc_v27.jq) on a v2.7 draft against the committed v2.6 record. jq + git + shasum; no Python.
# GATES lines are read by number at their commits (short shas expanded with git rev-parse; quote counts asserted); full 64-character file shas
# (B16 / B66); every gates_cite line in the draft is read from git for the existence check. Prints counts, keys and booleans only.
#   zsh -f sc_v27.zsh <draft.json>
set -e; setopt pipefail; export LC_ALL=C
HERE=${0:A:h}; D=${HERE:h}
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
REC=${1:A}; [[ -f $REC ]] || { echo "STOP usage"; exit 2; }
P19OVR=${2:-}   # negatives only (run_sc7_neg.zsh N7): a planted v19 copy; it forces PINOK false, so SC7 cannot pass with it
G() { git -C $REPO "$@"; }
sha() { shasum -a 256 "$1" | cut -c1-64; }
V26=$D/inputs/v2.6_record_0db7dc714.json; P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl; P19=$D/inputs/provenance_final19_X_c20699a5d.jsonl
GF=experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md
V26P=experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.6_2026-10-21.json
[[ $(sha $P18) == 13d29fb90cac3e3e54145b5d9f41319614a0ff63ebf33b5b4de3d251419741b0 ]] || { echo "STOP v18 sha"; exit 2; }
[[ $(sha $P19) == a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda ]] || { echo "STOP v19 sha"; exit 2; }
if [[ -n $P19OVR ]]; then P19=${P19OVR:A}; print -r -- "SC7_P19_OVERRIDE (negative test only)"; fi
BASE=$(G rev-parse 0db7dc714)
[[ $(G cat-file blob "${BASE}:${V26P}" | shasum -a 256 | cut -c1-64) == 2db9cdee658f30637f76d7ce507789f72b4b5854879a9ce6bcdcfee3605d2b96 && $(sha $V26) == 2db9cdee658f30637f76d7ce507789f72b4b5854879a9ce6bcdcfee3605d2b96 ]] || { echo "STOP v2.6 record sha"; exit 2; }
RUN=$HERE/_runs/sc7_$(TZ=America/Los_Angeles date +%Y%m%d_%H%M%S)_$$; mkdir -p $RUN
PINOK=true
G merge-base --is-ancestor $BASE origin/aswin/routec-final-release || PINOK=false
[[ -z $P19OVR ]] || PINOK=false
: > $RUN/lines.jsonl; : > $RUN/cites.jsonl
jq -n -c --arg c $BASE '{k: "base", v: {commit: $c, sha256: "2db9cdee658f30637f76d7ce507789f72b4b5854879a9ce6bcdcfee3605d2b96"}}' >> $RUN/cites.jsonl
for spec in g80_3160:f80f3cfe6:3160:0 g80_3171:f80f3cfe6:3171:2 g0a_3060:0a719afaa:3060:0 g0a_3072:0a719afaa:3072:2 g0a_3074:0a719afaa:3074:2 \
            g97_2974:97b6e634d:2974:6 g97_2982:97b6e634d:2982:4 g97_3195:97b6e634d:3195:2 g97_3200:97b6e634d:3200:2 \
            g50_3230:5046d443f:3230:8 g50_3232:5046d443f:3232:2 g50_3234:5046d443f:3234:4 g50_3235:5046d443f:3235:2 g50_3241:5046d443f:3241:2; do
  k=${spec%%:*}; rest=${spec#*:}; s=${rest%%:*}; rest=${rest#*:}; l=${rest%%:*}; q=${rest#*:}
  c=$(G rev-parse $s); G merge-base --is-ancestor $c origin/aswin/convergence-audit-09 || PINOK=false
  G cat-file blob "${c}:${GF}" > $RUN/g_${s}.md
  txt=$(sed -n "${l}p" $RUN/g_${s}.md | tr -d '\n')
  [[ $(print -rn -- "$txt" | tr -cd '"' | wc -c | tr -d ' ') == $q ]] || PINOK=false
  print -rn -- "$txt" | jq -R -s -c --arg k $k '{k: $k, v: .}' >> $RUN/lines.jsonl
  jq -n -c --arg k $k --arg c $c --arg f $GF --argjson l $l --arg fs $(sha $RUN/g_${s}.md) --arg ls $(print -rn -- "$txt" | shasum -a 256 | cut -c1-64) \
    '{k: $k, v: {commit: $c, file: $f, file_sha256: $fs, line: $l, line_sha256: $ls}}' >> $RUN/cites.jsonl
done
# every gates_cite line named in the draft (status_superseded_by), read from git; a line past the end is left out
jq -r '[.disclosures[] | .status_superseded_by? // empty | if type == "array" then .[] else . end | .gates_cite | select(type == "object") | "\(.commit)\t\(.file)\t\(.line)"] | unique[]' $REC > $RUN/lx.tsv
: > $RUN/lx.jsonl
while IFS=$'\t' read -r c f ln; do
  G cat-file -e "${c}:${f}" 2>/dev/null || continue
  n=$(G cat-file blob "${c}:${f}" | wc -l | tr -d ' '); [[ $ln =~ ^[0-9]+$ ]] && (( ln >= 1 && ln <= n )) || continue
  G cat-file blob "${c}:${f}" | sed -n "${ln}p" | tr -d '\n' | jq -R -s -c --arg k "${c}:${f}:${ln}" '{k: $k, v: .}' >> $RUN/lx.jsonl
done < $RUN/lx.tsv
jq -n -c -f $HERE/sc_v27.jq --slurpfile REC $REC --slurpfile V26 $V26 --slurpfile P18 $P18 --slurpfile P19 $P19 --slurpfile LN $RUN/lines.jsonl \
   --slurpfile CI $RUN/cites.jsonl --slurpfile LX $RUN/lx.jsonl --argjson PINOK $PINOK > $RUN/sc7.json 2> $RUN/sc7.err \
  || { echo "STOP sc_v27.jq failed (stderr in $RUN/sc7.err; not printed)"; exit 2; }
jq -r '"SC7 " + (.n | to_entries | map("\(.key) \(.value)") | join(" ")) + " mismatches \(.bad | length)"' $RUN/sc7.json
jq -r '"SC7_EXEMPT " + (.exempt | to_entries | map("\(.key)=\(.value)") | join(" "))' $RUN/sc7.json
jq -r '.bad[] | "MISMATCH " + .' $RUN/sc7.json
if jq -e '.bad | length == 0' $RUN/sc7.json > /dev/null; then print SC7_OK; else print SC7_FAIL; exit 1; fi
