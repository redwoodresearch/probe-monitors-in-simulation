#!/bin/zsh -f
# build_v2_4.zsh: build a v2.4 release-record DRAFT from v2.3 by key (jq + git + shasum; no Python). Prints shas and counts only.
#   zsh -f build_v2_4.zsh <q1 table json> <resolved rulings json> <out.json>
# Optional, by environment (all default OFF / absent):
#   F78=<figs78 keys-only result json> F78REF=<its committed file ref json> F78VER=<09's verification anchor json>  (else Figs 7/8 + slot 24 PRELIMINARY)
#   R122=<json {file, sha256, label}> : the fresh recheck of the verifier results commit, CONFIRMED (else row 122 PRELIMINARY)
#   RNODE=<json {on: true, aswin_quote, time}> : the pre-stated R_node switch (09, GATES 00:21); only on Aswin's verbatim option (2)
set -e; setopt pipefail; export LC_ALL=C
H=${0:A:h}; D=${H:h}
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
Q1=${1:A}; RES=${2:A}; OUT=${3:A}
V23=$D/inputs/v2.3_record_80a77ecdc.json; P17=$D/inputs/provenance_final17_98b5054fb.jsonl; P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl; RUL=$H/rulings_v2_4.json
[[ $(shasum -a 256 $V23 | cut -c1-64) == ebb47e28f00596535f4fbaa5bdadb2b93b413bb768fb05e99b1abd36cac7de4f ]] || { echo "STOP v2.3 sha"; exit 2; }
[[ $(shasum -a 256 $P17 | cut -c1-64) == 220d25d21943226791637581afda92faac595a3b15b9b6d6bbd3fd1a1e575971 ]] || { echo "STOP final17 sha"; exit 2; }
[[ $(shasum -a 256 $P18 | cut -c1-64) == 13d29fb90cac3e3e* ]] || { echo "STOP v18 sha"; exit 2; }
G() { git -C $REPO "$@"; }
[[ $(G cat-file blob 58bf4a3ac:experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final18_0141.jsonl | shasum -a 256 | cut -c1-64) == $(shasum -a 256 $P18 | cut -c1-64) ]] || { echo "STOP v18 != commit A blob"; exit 2; }
cmp -s <(sed -n '1,118p;123,251p' $P17) <(sed -n '1,118p;123,251p' $P18) || { echo "STOP v18 != final17 outside rows 119-122"; exit 2; }
origins() { G branch -r --contains $1 | tr -s ' \t' '\n' | awk '/^origin\//' | sort | jq -R . | jq -s -c . ; }
mkdir -p $H/_work; WORK=$(mktemp -d "$H/_work/build.XXXXXX")
: > $WORK/cov.jsonl
jq -r '(.in_scope_outputs + .preliminary_outputs)[] | select(.s2_marker) | "\(.name)\t\(.s2_marker.commit)\t\(.s2_marker.path)"' $V23 | while IFS=$'\t' read -r nm c p; do
  G cat-file blob "$c:$p" | jq -c --arg nm "$nm" '{($nm): .covered}' >> $WORK/cov.jsonl
done
jq -s -c 'add' $WORK/cov.jsonl > $WORK/cov.json
SENS=$(jq -r '.preliminary_outputs[] | select(.name == "Table 6 plant_rows") | .files[] | "\(.commit)\t\(.path)"' $V23 | while IFS=$'\t' read -r c p; do G cat-file blob "$c:$p" | jq -c --arg p "$p" --argjson STEMS '["c_asis_uncut", "own_all_rs10_uncut_L48", "own_all_rs25_uncut_L48"]' --argjson OUTSIDE '["build_profile", "plan", "logs"]' -f $H/sens_scan.jq | jq -c --arg p "$p" '{($p): .}'; done | jq -s -c 'add')
[[ $(print -r -- "$SENS" | jq '[.[] | .mentions_outside_metadata_blocks] | add') == 0 ]] || { echo "STOP Table 6 sensitivity mentions outside metadata blocks need a ruling"; exit 3; }
IDX=$(jq -r '.element_tables[] | select(.file | test("fig1_main(_R)?_figs156_routecA_source.json$")) | "\(.commit)\t\(.file)"' $V23 | while IFS=$'\t' read -r c p; do G cat-file blob "$c:$p" | jq -c --arg p "$p" '{($p): [.values | to_entries[] | select((.value | has("ci95")) and .value.input == "record") | .key]}'; done | jq -s -c 'add')
[[ $(print -r -- "$IDX" | jq -c '[.[] | length]') == "[60,60]" ]] || { echo "STOP Figs 1/6 interval elements are not 60 per file"; exit 3; }
# b8 ([INT-FIG]): Fig 4's interval elements = its committed source's values with ci95 (240, per GATES)
FIG4N=$(jq -r '.in_scope_outputs[] | select(.name == "Fig 4") | .files[] | select(.path | test("fig4_pooled_fig4_routecB_source\\.json$")) | "\(.commit)\t\(.path)"' $V23 | while IFS=$'\t' read -r c p; do G cat-file blob "$c:$p" | jq '[.values[] | select(has("ci95"))] | length'; done)
[[ $FIG4N == 240 ]] || { echo "STOP Fig 4 interval elements are not 240"; exit 3; }
F18C=$(G rev-parse 58bf4a3ac); V23C=$(G rev-parse 80a77ecdc)
J() { if [[ -n $1 ]]; then cat $1; else print null; fi; }
if [[ -n ${R122:-} ]]; then   # row 122 FINAL only on a CONFIRMED fresh recheck whose report and verifier result are sha-pinned (09)
  jq -e '(keys == ["file", "gates", "label", "of", "sha256", "verdict"]) and .label == "from agent summary" and .verdict == "CONFIRMED"' $R122 > /dev/null || { echo "STOP R122 descriptor shape"; exit 2; }
  [[ $(shasum -a 256 "$(jq -r '.file' $R122)" | cut -c1-64) == $(jq -r '.sha256' $R122) ]] || { echo "STOP R122 report sha"; exit 2; }
  [[ $(G cat-file blob "$(jq -r '.of.commit' $R122):$(jq -r '.of.path' $R122)" | shasum -a 256 | cut -c1-64) == $(jq -r '.of.sha256' $R122) ]] || { echo "STOP R122 verifier result sha"; exit 2; }
  [[ $(G rev-parse "$(jq -r '.of.commit' $R122)^") == $(jq -r '.of.parent' $R122) ]] || { echo "STOP R122 verifier commit parent"; exit 2; }
  [[ $(G cat-file blob "$(jq -r '.gates.commit' $R122):experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md" | grep -cF -- "$(jq -r '.gates.must' $R122)") == 1 ]] || { echo "STOP R122 GATES log line"; exit 2; }
  # b8 (Codex v24 r1 item 5): the descriptor must be THE recheck 2e relayed (constants), not any self-consistent one
  jq -e '.sha256 == "870ed0de76ac62452711d35434a782dac3875abf69c838cd40b350b8e8963300" and .of.commit == "fa4a475c9069b9c5edc23d3da4f791e1d8c837b0"
         and .of.parent == "e14aa51f98830e1e3024d7cb65543e2975167368" and .of.sha256 == "45bfd0a3b8a0b3e3c4229e0d4d941c54f76c84bc2f8439f0f7bff4906f7bbafe"
         and .of.path == "experiments/2026-10-20_routec_numbers_verify/results/numbers_verify_2026-10-21_002558.json" and .gates.must == "RECHECK_R_fa4a475c9_004731"' $R122 > /dev/null \
    || { echo "STOP R122 descriptor is not the relayed recheck"; exit 2; }
fi
META=$(jq -n -c --arg date "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')" \
  --arg bsha $(shasum -a 256 $H/build_v2_4.jq | cut -c1-64) --arg wsha $(shasum -a 256 $H/build_v2_4.zsh | cut -c1-64) \
  --arg rsha $(shasum -a 256 $RUL | cut -c1-64) --arg ressha $(shasum -a 256 $RES | cut -c1-64) \
  --arg f18c $F18C --argjson f18o "$(origins $F18C)" --arg v23c $V23C --argjson v23o "$(origins $V23C)" \
  --arg q1sha $(shasum -a 256 $Q1 | cut -c1-64) --arg q1name ${Q1:t} \
  --argjson f78ref "$(J ${F78REF:-})" --argjson f78ver "$(J ${F78VER:-})" --argjson sens "$SENS" --argjson idx "$IDX" --argjson r122 "$(J ${R122:-})" --argjson rnode "$(J ${RNODE:-})" \
  --argjson fig4n $FIG4N \
  '{date_pt: $date, repo_path: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.4_2026-10-21.json", fig4_interval_n: $fig4n,
    builder: {script_sha256: $bsha, wrapper_sha256: $wsha, rulings_sha256: $rsha, resolved_sha256: $ressha,
              note: "v2.4 = v2.3 + the v2.4 rulings, by key (build_v2_4.jq); laptop jq + git + shasum, no Python"},
    provenance: {kind: "writing_provenance", path: "experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final18_0141.jsonl", commit: $f18c,
                 sha256: "13d29fb90cac3e3e", origin_branches: $f18o, equals_final17_except_rows: [119, 120, 121, 122],
                 final17: {path: "experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final17_0141.jsonl", commit: "98b5054fbf8306e54711b1a7bacd72b5f5e0af2a",
                           sha256: "220d25d21943226791637581afda92faac595a3b15b9b6d6bbd3fd1a1e575971"}},
    v23: {kind: "release_record", path: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.3_2026-10-20.json", commit: $v23c,
          sha256: "ebb47e28f00596535f4fbaa5bdadb2b93b413bb768fb05e99b1abd36cac7de4f", origin_branches: $v23o},
    q1_table: {file: $q1name, sha256: $q1sha, note: "Q1 per-row table (09, 22:46), computed on final17; v18 = final17 on every row except 119-122, which are ruled separately (rows_119_122); the row 251 half2_key cites GATES L1362 instead of the number (09, b8); committed with the tooling"},
    figs78: (if $f78ref == null then null else {record: $f78ref, verified_by_09: $f78ver} end), sens_scan_table6: $sens,
    row122_recheck: $r122, r_node_switch: ($rnode // {on: false}), figs16_interval_idx: $idx}')
# the full v18 sha256 goes in META (the literal above is a placeholder prefix, replaced here)
META=$(print -r -- "$META" | jq -c --arg s $(shasum -a 256 $P18 | cut -c1-64) '.provenance.sha256 = $s')
if [[ -n ${F78:-} ]]; then F78ARG=(--slurpfile F78 ${F78:A}); else F78ARG=(--slurpfile F78 /dev/null); fi
# the cited GATES line texts, read from git into the work folder (never into the resolved file); each line's sha256 re-checked
jq -r '.anchors[] | .anchor | "\(.commit)\t\(.file)\t\(.line)\t\(.line_sha256)"' $RES | sort -u | while IFS=$'\t' read -r c f ln lsha; do
  [[ $(G cat-file blob "$c:$f" | sed -n "${ln}p" | tr -d '\n' | shasum -a 256 | cut -c1-64) == $lsha ]] || { echo "STOP an anchor's line sha"; exit 3; }
  G cat-file blob "$c:$f" | sed -n "${ln}p" | tr -d '\n' | jq -R -s -c --arg k "$c:$f:$ln" '{k: $k, v: .}'
done > $WORK/lines.jsonl
[[ $(wc -l < $WORK/lines.jsonl | tr -d ' ') == $(jq -r '[.anchors[] | .anchor | "\(.commit):\(.file):\(.line)"] | unique | length' $RES) ]] || { echo "STOP line texts count"; exit 3; }
jq -n -f $H/build_v2_4.jq --slurpfile V23 $V23 --slurpfile P17 $P18 --slurpfile Q1 $Q1 --slurpfile RUL $RUL --slurpfile RES $RES --slurpfile LN $WORK/lines.jsonl \
   --slurpfile COV $WORK/cov.json "${F78ARG[@]}" --argjson META "$META" > $OUT 2> $WORK/build.err \
  || { echo "STOP build_v2_4.jq failed (stderr kept in $WORK/build.err; not printed)"; exit 3; }
print -r -- "BUILT $OUT sha256 $(shasum -a 256 $OUT | cut -c1-64) figs78 $([[ -n ${F78:-} ]] && print yes || print no) row122_recheck $([[ -n ${R122:-} ]] && print yes || print no) r_node_switch $(print -r -- "$META" | jq -r '.r_node_switch.on')"
jq -r '"ROWS \(.printed_items.all_rows | length) FINAL \([.printed_items.all_rows[] | select(.status | startswith("FINAL"))] | length) PRELIMINARY \([.printed_items.all_rows[] | select(.status | startswith("PRELIMINARY"))] | length) NOT_ROUTE_C \([.printed_items.all_rows[] | select(.status | startswith("not route C"))] | length)"' $OUT
