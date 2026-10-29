#!/bin/zsh -f
# Test of cover_check.jq on synthetic files (fixtures_cover/; no real record). Positive controls first (inputs, list), then one case per
# check, including a pre-flight entry with another sha, one path listed with two shas (a03f5d93 O-2) and .gz paths (2e). Run from the repo root:
#   zsh -f experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_cover_check.sh
T=experiments/2026-10-15_fpr_sweep_507_prep/tools_routec
X=$T/fixtures_cover
EP=experiments/2026-10-15_fpr_sweep_507_prep/results/routec_preflight
PF=$(jq -n -c --arg e "$EP" '[$e + "/routec_A_2026-10-21_100000.json@" + ("a" * 64), $e + "/routec_B_2026-10-21_101000.json@" + ("b" * 64), $e + "/routec_R37_2026-10-21_102000.json@" + ("c" * 64)]')
ok=1
check() {
  out=$(jq -c --arg mode "$2" --argjson pf "$PF" -f "$T/cover_check.jq" "$X/$1") || out="JQ_ERROR"
  if [ "$out" = "$3" ]; then echo "PASS $1 ($2) -> $out"; else echo "FAIL $1 ($2) -> $out (want $3)"; ok=0; fi
}
check inputs_ok.json inputs '{"ok":true,"failing":[]}'
check list_ok.json list '{"ok":true,"failing":[]}'
check inputs_placeholder.json inputs '{"ok":false,"failing":["template_placeholder_left","preflight_record_not_listed","noncanonical_path"]}'
check inputs_missing_r37.json inputs '{"ok":false,"failing":["preflight_record_not_listed"]}'
check list_unsorted.json list '{"ok":false,"failing":["not_sorted_unique"]}'
check list_other_sha.json list '{"ok":false,"failing":["preflight_entry_missing"]}'
check list_noncanonical.json list '{"ok":false,"failing":["entry_form"]}'
check list_duplicate.json list '{"ok":false,"failing":["not_sorted_unique"]}'
check list_path_two_shas.json list '{"ok":false,"failing":["path_twice"]}'
check inputs_gz_record.json inputs '{"ok":false,"failing":["gz_path"]}'
check inputs_gz_regression.json inputs '{"ok":false,"failing":["gz_path"]}'
check list_gz_entry.json list '{"ok":false,"failing":["gz_entry"]}'
check inputs_gz_upper_record.json inputs '{"ok":false,"failing":["gz_path"]}'
check inputs_gz_harness.json inputs '{"ok":false,"failing":["gz_path"]}'
check list_gz_upper_entry.json list '{"ok":false,"failing":["gz_entry"]}'
check inputs_gz_trailing_dot.json inputs '{"ok":false,"failing":["gz_path"]}'
check inputs_gz_pct2E.json inputs '{"ok":false,"failing":["noncanonical_path","gz_path"]}'
check inputs_gz_tgz.json inputs '{"ok":false,"failing":["gz_path"]}'
check inputs_gz_g_pct7A.json inputs '{"ok":false,"failing":["noncanonical_path"]}'
check list_ok.json inputs '{"ok":false,"failing":["not_an_object"]}'
check inputs_ok.json list '{"ok":false,"failing":["not_an_array"]}'
if [ "$ok" = 1 ]; then echo "ALL PASS"; else echo "FAILED"; exit 1; fi
