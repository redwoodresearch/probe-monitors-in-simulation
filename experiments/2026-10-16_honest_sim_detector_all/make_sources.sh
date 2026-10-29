#!/usr/bin/env bash
# Writes config/sources_v1.json: every committed file the remote input build (honest_inputs.build_all) reads, pinned by full commit and
# sha256, plus the routing directory listing at ROUTING_REF. git + jq only (no Python on the laptop). Run from this folder:
#   bash make_sources.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/config/sources_v1.json"
[ -e "$OUT" ] && { echo "$OUT exists (config files are versioned, never overwritten)"; exit 1; }
RDIR=experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing
REF=$(git rev-parse "16609a1b5502^{commit}")  # the routing records ref of the epoch-2 inventory (its rows cite it); used to select S_new's
D4F=experiments/2026-10-16_cmd_sim_detector_uncut_all
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

add() {  # commit path role
  local c s
  c=$(git rev-parse "$1^{commit}")
  s=$(git show "$c:$2" | shasum -a 256 | cut -c1-64)
  jq -nc --arg c "$c" --arg p "$2" --arg s "$s" --arg r "$3" '{commit: $c, path: $p, sha256: $s, role: $r}' >> "$TMP/files.jsonl"
}
cited() {  # commit path jq-row-filter: the routing records the inventory's honest rows cite ("<path> @ <commit>")
  git show "$1:$2" | jq -r ".runs[] | select(.world | test(\"honest\")) | select($3) | .uncut_planned.routing" | sort -u
}
ex_of() {  # commit path: that routing record's extraction records
  git show "$1:$2" | jq -r '(.extraction_record // [])[] | "\(.commit) \(.path)"'
}

P=experiments/2026-10-15_fpr_sweep_507_prep
add 927fabb0d "$P/config/INPUTS_FULL_2026-10-16_110447.json" inputs_full
add 22d7089f5 "$P/config/E12_BLOCK_2026-10-16_092322.json" e12
add 5b19c08d8 experiments/2026-10-15_epoch2_poolkeys/results/epoch2_poolkeys_2026-10-15_214512.json poolkeys
add 09b070617 experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json inv_bfix
add b3f8dea7b "$P/config/cutfull/inventory_v3_2026-10-15.json" inv_snew
add b880c3c3a experiments/2026-10-15_cut_sensitivity/results/inventory_ha25f_v2_2026-10-16_044849.json inv_ha25f
add 85447113f experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w1_v1_2026-10-16_025411.json inv_w1
add 102b3741e experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w23_v1_2026-10-16_033210.json inv_w23
add 06077a8a8 experiments/2026-10-15_cut_sensitivity/results/inventory_ep2_v1_2026-10-16_025227.json inv_ep2
add 06077a8a8 experiments/2026-10-15_cut_sensitivity/results/inventory_fill_v1_2026-10-16_025228.json inv_fill
FOLDS=experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json
add ec56720ad "$FOLDS" folds
add 59acf625e "$D4F/config/inputs_all_S_old_v1.json" sold_config
add 59acf625e "$D4F/config/inputs_all_S_new_v1.json" snew_attack_config

# the fold list's own pinned cell files and S_paper493 facts
git show "ec56720ad:$FOLDS" | jq -r '.pins.cells[] | "\(.[0]) \(.[1])"' | while read -r c p; do add "$c" "$p" fold_pin; done
git show "ec56720ad:$FOLDS" | jq -r '.pins.sp493_facts | "\(.[0]) \(.[1])"' | while read -r c p; do add "$c" "$p" fold_pin; done

# routing records: 4f's S_old config (+ its extraction records), 4f's S_new attack config (cross-check), every record cited by the honest rows
for cfg in inputs_all_S_old_v1 inputs_all_S_new_v1; do
  git show "59acf625e:$D4F/config/$cfg.json" | jq -r '.routing.records[] | "\(.commit) \(.path)"' >> "$TMP/routing.txt"
done
git show "59acf625e:$D4F/config/inputs_all_S_old_v1.json" | jq -r '.extraction_records[] | "\(.commit) \(.path)"' >> "$TMP/extract.txt"
{
  cited 09b070617 experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json '.task_set == "new mains"'
  cited b880c3c3a experiments/2026-10-15_cut_sensitivity/results/inventory_ha25f_v2_2026-10-16_044849.json '.set == "S_new"'
  cited 85447113f experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w1_v1_2026-10-16_025411.json '.set == "S_paper493"'
  cited 102b3741e experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w23_v1_2026-10-16_033210.json '.set == "S_paper493"'
  cited 06077a8a8 experiments/2026-10-15_cut_sensitivity/results/inventory_ep2_v1_2026-10-16_025227.json 'true'
  cited 06077a8a8 experiments/2026-10-15_cut_sensitivity/results/inventory_fill_v1_2026-10-16_025228.json 'true'
} | sed -E 's/^(.*) @ ([0-9a-f]+)$/\2 \1/' >> "$TMP/routing.txt"
# the whole routing directory at REF (S_new's records are selected remotely by 4f's rule)
git ls-tree --name-only "$REF:$RDIR" > "$TMP/names.txt"
sed "s|^|$REF $RDIR/|" "$TMP/names.txt" >> "$TMP/routing.txt"
sort -u "$TMP/routing.txt" | while read -r c p; do
  add "$c" "$p" routing
  case "$p" in *_attest.json) ;; *) ex_of "$c" "$p" >> "$TMP/extract.txt" ;; esac
done
sort -u "$TMP/extract.txt" | while read -r c p; do add "$c" "$p" extraction; done

jq -s --arg ref "$REF" --arg rdir "$RDIR" --rawfile names "$TMP/names.txt" \
  '{label: "honest all-tasks detector: committed files the remote input build reads (make_sources.sh; git + jq)", routing_ref: $ref, routing_dir: $rdir,
    routing_names: ($names | split("\n") | map(select(length > 0))), files: unique_by(.commit + ":" + .path)}' "$TMP/files.jsonl" > "$OUT"
jq -c '{n_files: (.files | length), by_role: (.files | group_by(.role) | map({(.[0].role): length}) | add), names: (.routing_names | length)}' "$OUT"
