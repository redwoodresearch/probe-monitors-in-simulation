#!/bin/zsh -f
# Test of sd_preflight_consistent.jq (09's ruling (ii), SD/KAP vs A by key) on synthetic fixtures (fixtures_sd_consistent/; no real record).
# Positive control first: SD covers a subset of A's files and drops, with equal lists, so it is consistent (A holding more is fine).
# Then one case per checked field must FAIL: a file A lacks (an extra job's file), a zero-unit mismatch, a non-own mismatch, a non-own
# file A lacks, a drop A lacks, and a DEFECT record. Run from the repo root:
#   zsh -f experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_sd_preflight_consistent.sh
T=experiments/2026-10-15_fpr_sweep_507_prep/tools_routec
X=$T/fixtures_sd_consistent
ok=1
check() {
  out=$(jq -n -c --slurpfile s "$X/$1" --slurpfile a "$X/a_final.json" -f "$T/sd_preflight_consistent.jq") || out="JQ_ERROR"
  if [ "$out" = "$2" ]; then echo "PASS $1 -> $out"; else echo "FAIL $1 -> $out (want $2)"; ok=0; fi
}
check sd_ok.json '{"consistent":true,"differing":[]}'
check sd_file_not_in_a.json '{"consistent":false,"differing":["file_set"]}'
check sd_zero_unit_mismatch.json '{"consistent":false,"differing":["zero_unit_by_file"]}'
check sd_nonown_mismatch.json '{"consistent":false,"differing":["zero_unit_by_file_nonown"]}'
check sd_nonown_not_in_a.json '{"consistent":false,"differing":["file_set"]}'
check sd_extra_drop.json '{"consistent":false,"differing":["gate_view.drops"]}'
check sd_defect.json '{"consistent":false,"differing":["status_not_both_CLEAR"]}'
if [ "$ok" = 1 ]; then echo "ALL PASS"; else echo "FAILED"; exit 1; fi
