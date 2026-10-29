#!/bin/zsh -f
# Test of prov_final_preflight_equal.jq (GATES Amendment 8) on synthetic fixtures (fixtures_prov_equal/; no real record is read).
# Each case: the exact one-line output. Positive control first (the same content, another spec sha and time: equal), then one
# difference per checked field. Run from the repo root:  zsh -f experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_prov_final_preflight_equal.sh
T=experiments/2026-10-15_fpr_sweep_507_prep/tools_routec
X=$T/fixtures_prov_equal
ok=1
check() {
  out=$(jq -n -c --slurpfile p "$X/prov.json" --slurpfile f "$X/$1" -f "$T/prov_final_preflight_equal.jq") || out="JQ_ERROR"
  if [ "$out" = "$2" ]; then echo "PASS $1 -> $out"; else echo "FAIL $1 -> $out (want $2)"; ok=0; fi
}
check final_same.json '{"equal":true,"differing":[]}'
check final_drops.json '{"equal":false,"differing":["gate_view.drops"]}'
check final_nonown.json '{"equal":false,"differing":["zero_unit_by_file_nonown"]}'
check final_jobs.json '{"equal":false,"differing":["jobs"]}'
check final_files.json '{"equal":false,"differing":["zero_unit_by_file","file_set"]}'
check final_defect.json '{"equal":false,"differing":["status_not_both_CLEAR"]}'
check final_no_nonown.json '{"equal":false,"differing":["zero_unit_by_file_nonown"]}'
if [ "$ok" = 1 ]; then echo "ALL PASS"; else echo "FAILED"; exit 1; fi
