#!/bin/zsh -f
# Test of pf4_record_binds_s3.jq on synthetic records (fixtures_pf4_binds/; no real record). Positive control first, then a record whose
# spec sha differs from the pickle's (the case 09's addition targets), one naming another pickle, one with no spec, and a malformed
# --arg s3. Run from the repo root:
#   zsh -f experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_pf4_record_binds_s3.sh
T=experiments/2026-10-15_fpr_sweep_507_prep/tools_routec
X=$T/fixtures_pf4_binds
S3='experiments/2026-10-15_fpr_sweep_507_prep/results/routec_spec_A_preflight_2026-10-20_100000.pkl@bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
ok=1
check() {
  out=$(jq -c --arg s3 "$3" -f "$T/pf4_record_binds_s3.jq" "$X/$1") || out="JQ_ERROR"
  if [ "$out" = "$2" ]; then echo "PASS $1 -> $out"; else echo "FAIL $1 -> $out (want $2)"; ok=0; fi
}
check record_ok.json '{"ok":true,"failing":[]}' "$S3"
check record_other_sha.json '{"ok":false,"failing":["spec_sha256"]}' "$S3"
check record_other_local.json '{"ok":false,"failing":["spec_local"]}' "$S3"
check record_no_spec.json '{"ok":false,"failing":["no_spec"]}' "$S3"
check record_ok.json '{"ok":false,"failing":["s3_form","spec_sha256"]}' "${S3%@*}@bbbb"
if [ "$ok" = 1 ]; then echo "ALL PASS"; else echo "FAILED"; exit 1; fi
