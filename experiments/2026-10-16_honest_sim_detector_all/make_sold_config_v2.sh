#!/usr/bin/env bash
# Route C rerun (2026-10-18): writes config/inputs_S_old_routec_v2.json = 4f's S_old input config (inputs_all_S_old_v1.json @ 59acf625e) with
# exactly three changes: inventory -> route C's S_old inventory (inventory_b_fix_v2 @ 98575cc34 = fix_v1 + the 20 pilot sim-honest rows),
# + the pilot's routing record (sold__simh_q4f_v1_pilot_cutfull.json @ 59de09da9) and + its extraction record (@ f91317f83). Every pin is
# git show'd and sha256'd here; no value is typed. git + jq only (no Python on the laptop). Run from this folder: bash make_sold_config_v2.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/config/inputs_S_old_routec_v2.json"
[ -e "$OUT" ] && { echo "$OUT exists (config files are versioned, never overwritten)"; exit 1; }
spec() {  # commit path -> {commit (full), path, sha256}
  local c s
  c=$(git rev-parse "$1^{commit}")
  s=$(git show "$c:$2" | shasum -a 256 | cut -c1-64)
  jq -nc --arg c "$c" --arg p "$2" --arg s "$s" '{commit: $c, path: $p, sha256: $s}'
}
V1=$(git show "59acf625e:experiments/2026-10-16_cmd_sim_detector_uncut_all/config/inputs_all_S_old_v1.json")
INV=$(spec 98575cc34 experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v2_2026-10-17_081456.json)
ROUT=$(spec 59de09da9 experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing/sold__simh_q4f_v1_pilot_cutfull.json)
EXR=$(spec f91317f83 experiments/2026-10-14_q3_overnight_chain/results/extract_chain_simh_pilot_cutfull_extract_v1_a1_2026-10-17_014605.json)
ROOT=$(git show "59de09da9:experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing/sold__simh_q4f_v1_pilot_cutfull.json" | jq -r .extraction_root)
printf '%s' "$V1" | jq --argjson inv "$INV" --argjson rout "$ROUT" --argjson exr "$EXR" --arg root "$ROOT" '
  if (.extraction_records | has($root)) or ([.routing.records[].path] | index($rout.path)) then error("pilot already in the v1 config") else . end
  | .label = "ROUTE C RERUN (2026-10-18, make_sold_config_v2.sh): 4f inputs_all_S_old_v1.json @ 59acf625e with inventory -> route C INPUTS_FULL 233212 S_old inventory_b_fix_v2 (fix_v1 + the 20 pilot sim-honest rows), + the pilot routing record and its extraction record. Original label: " + .label
  | .inventory = $inv | .extraction_records[$root] = $exr | .routing.records += [$rout]' > "$OUT"
jq -c '{inventory: .inventory.path, n_routing: (.routing.records | length), n_extraction: (.extraction_records | length), pilot_root: (.extraction_records | has("simh_pilot_cutfull_extract_v1"))}' "$OUT"
