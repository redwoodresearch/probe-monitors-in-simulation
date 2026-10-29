#!/usr/bin/env bash
# Writes config/sources_v1.json: every committed file 4f's build_inputs reads (run remotely by run_refit3.py), plus 4f's record, pinned
# by full commit and sha256. git + jq only (no Python on the laptop). Run from this folder: bash make_sources.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/config/sources_v1.json"
[ -e "$OUT" ] && { echo "$OUT exists (config files are versioned, never overwritten)"; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
add() {  # commit path role
  local c s
  c=$(git rev-parse "$1^{commit}")
  s=$(git show "$c:$2" | shasum -a 256 | cut -c1-64)
  jq -nc --arg c "$c" --arg p "$2" --arg s "$s" --arg r "$3" '{commit: $c, path: $p, sha256: $s, role: $r}' >> "$TMP/files.jsonl"
}
add 59acf625e experiments/2026-10-16_cmd_sim_detector_uncut_all/results/attack_all_2026-10-16_042630.json record
add adbc791b6 experiments/2026-10-15_fpr_sweep_507_prep/results/exclusions_2026-10-16_033635.json cohort
FOLDS=experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json
add ec56720adde66baabe642a7fc842172c83737679 "$FOLDS" folds
git show "ec56720ad:$FOLDS" | jq -r '.pins.cells[], .pins.sp493_facts | "\(.[0]) \(.[1])"' | while read -r c p; do add "$c" "$p" fold_pin; done
for f in "$HERE"/config/inputs_all_*_v1.json; do
  jq -r '.inventory, .extraction_records[], .routing.records[] | "\(.commit) \(.path)"' "$f"
done | sort -u | while read -r c p; do add "$c" "$p" config_dep; done
jq -s '{label: "attack refit3: committed files 4f'"'"'s build_inputs reads, and 4f'"'"'s record (make_sources.sh; git + jq)", files: unique_by(.commit + ":" + .path)}' \
  "$TMP/files.jsonl" > "$OUT"
jq -c '{n_files: (.files | length), by_role: (.files | group_by(.role) | map({(.[0].role): length}) | add)}' "$OUT"
