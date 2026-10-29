#!/usr/bin/env bash
# Route C rerun (2026-10-18): writes config/sources_v2.json, make_sources.sh's list with exactly these changes (printed at the end against
# sources_v1.json, by role and commit:path):
#   inputs_full -> route C's INPUTS_FULL_2026-10-17_233212.json @ 1a842a856 (sha 1e186858);
#   + inv_sold = route C's S_old inventory (inventory_b_fix_v2 @ 98575cc34); inv_bfix (fix_v1) stays, now as INPUTS_FULL's S_newmains inventory;
#   + pilot_keys = INPUTS_FULL's s_old_simh_extra_roots file (simh_counted_keys_pilot_v3.json @ 344829af6);
#   sold_config -> config/inputs_S_old_routec_v2.json (make_sold_config_v2.sh) at its commit on this branch;
#   + the pilot routing record and extraction record (read from that config).
# git + jq only (no Python on the laptop). Run from this folder after inputs_S_old_routec_v2.json is committed: bash make_sources_v2.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/config/sources_v2.json"
[ -e "$OUT" ] && { echo "$OUT exists (config files are versioned, never overwritten)"; exit 1; }
RDIR=experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing
REF=$(git rev-parse "16609a1b5502^{commit}")  # unchanged from v1: the routing records ref of the epoch-2 inventory (S_new's selection)
D4F=experiments/2026-10-16_cmd_sim_detector_uncut_all
E=experiments/2026-10-16_honest_sim_detector_all
SOLD="$E/config/inputs_S_old_routec_v2.json"
SC=$(git log -1 --format=%H -- "$SOLD")
[ -n "$SC" ] || { echo "$SOLD is not committed"; exit 1; }
[ "$(git show "$SC:$SOLD" | shasum -a 256 | cut -c1-64)" = "$(shasum -a 256 < "$SOLD" | cut -c1-64)" ] || { echo "$SOLD differs from its commit $SC"; exit 1; }
TMP=$(mktemp -d)  # left in place (no deletions); holds only git-derived key lists

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
add 1a842a856 "$P/config/INPUTS_FULL_2026-10-17_233212.json" inputs_full
add 22d7089f5 "$P/config/E12_BLOCK_2026-10-16_092322.json" e12
add 5b19c08d8 experiments/2026-10-15_epoch2_poolkeys/results/epoch2_poolkeys_2026-10-15_214512.json poolkeys
add 98575cc34 experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v2_2026-10-17_081456.json inv_sold
add 09b070617 experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json inv_bfix
add 344829af6 experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json pilot_keys
add b3f8dea7b "$P/config/cutfull/inventory_v3_2026-10-15.json" inv_snew
add b880c3c3a experiments/2026-10-15_cut_sensitivity/results/inventory_ha25f_v2_2026-10-16_044849.json inv_ha25f
add 85447113f experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w1_v1_2026-10-16_025411.json inv_w1
add 102b3741e experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w23_v1_2026-10-16_033210.json inv_w23
add 06077a8a8 experiments/2026-10-15_cut_sensitivity/results/inventory_ep2_v1_2026-10-16_025227.json inv_ep2
add 06077a8a8 experiments/2026-10-15_cut_sensitivity/results/inventory_fill_v1_2026-10-16_025228.json inv_fill
FOLDS=experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json
add ec56720ad "$FOLDS" folds
add "$SC" "$SOLD" sold_config
add 59acf625e "$D4F/config/inputs_all_S_new_v1.json" snew_attack_config

# the fold list's own pinned cell files and S_paper493 facts (unchanged)
git show "ec56720ad:$FOLDS" | jq -r '.pins.cells[] | "\(.[0]) \(.[1])"' | while read -r c p; do add "$c" "$p" fold_pin; done
git show "ec56720ad:$FOLDS" | jq -r '.pins.sp493_facts | "\(.[0]) \(.[1])"' | while read -r c p; do add "$c" "$p" fold_pin; done

# routing records: the route C S_old config (+ its extraction records), 4f's S_new attack config (cross-check), every record cited by the honest rows
git show "$SC:$SOLD" | jq -r '.routing.records[] | "\(.commit) \(.path)"' >> "$TMP/routing.txt"
git show "59acf625e:$D4F/config/inputs_all_S_new_v1.json" | jq -r '.routing.records[] | "\(.commit) \(.path)"' >> "$TMP/routing.txt"
git show "$SC:$SOLD" | jq -r '.extraction_records[] | "\(.commit) \(.path)"' >> "$TMP/extract.txt"
{
  cited 09b070617 experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json '.task_set == "new mains"'
  cited b880c3c3a experiments/2026-10-15_cut_sensitivity/results/inventory_ha25f_v2_2026-10-16_044849.json '.set == "S_new"'
  cited 85447113f experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w1_v1_2026-10-16_025411.json '.set == "S_paper493"'
  cited 102b3741e experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w23_v1_2026-10-16_033210.json '.set == "S_paper493"'
  cited 06077a8a8 experiments/2026-10-15_cut_sensitivity/results/inventory_ep2_v1_2026-10-16_025227.json 'true'
  cited 06077a8a8 experiments/2026-10-15_cut_sensitivity/results/inventory_fill_v1_2026-10-16_025228.json 'true'
} | sed -E 's/^(.*) @ ([0-9a-f]+)$/\2 \1/' >> "$TMP/routing.txt"
git ls-tree --name-only "$REF:$RDIR" > "$TMP/names.txt"
sed "s|^|$REF $RDIR/|" "$TMP/names.txt" >> "$TMP/routing.txt"
sort -u "$TMP/routing.txt" | while read -r c p; do
  add "$c" "$p" routing
  case "$p" in *_attest.json) ;; *) ex_of "$c" "$p" >> "$TMP/extract.txt" ;; esac
done
sort -u "$TMP/extract.txt" | while read -r c p; do add "$c" "$p" extraction; done

jq -s --arg ref "$REF" --arg rdir "$RDIR" --rawfile names "$TMP/names.txt" \
  '{label: "ROUTE C RERUN (2026-10-18) honest all-tasks detector: committed files the remote input build reads (make_sources_v2.sh; git + jq)", routing_ref: $ref, routing_dir: $rdir,
    routing_names: ($names | split("\n") | map(select(length > 0))), files: unique_by(.commit + ":" + .path)}' "$TMP/files.jsonl" > "$OUT"
jq -c '{n_files: (.files | length), by_role: (.files | group_by(.role) | map({(.[0].role): length}) | add), names: (.routing_names | length)}' "$OUT"
# the change list against v1, by role and commit:path (keys only)
jq -n -c --slurpfile a "$HERE/config/sources_v1.json" --slurpfile b "$OUT" '
  def ks(x): [x.files[] | "\(.role) \(.commit[0:12]):\(.path)"]; (ks($a[0])) as $A | (ks($b[0])) as $B
  | {only_v1: ($A - $B), only_v2: ($B - $A), same_routing_ref: ($a[0].routing_ref == $b[0].routing_ref), same_names: ($a[0].routing_names == $b[0].routing_names)}'
