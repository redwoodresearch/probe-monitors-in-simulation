#!/bin/zsh -f
# Test of eq4_release_marker.jq on synthetic markers for job A (fixtures_eq4_release/; no real marker). Two positive controls (the exact
# two entries; the two plus A's PROVISIONAL record), then one case per check, including another job's signoff, another job's record and
# this job's record with another sha. A wrong --arg j is the last case. Run from the repo root:
#   zsh -f experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_eq4_release_marker.sh
T=experiments/2026-10-15_fpr_sweep_507_prep/tools_routec
X=$T/fixtures_eq4_release
F4='results/routec_preflight/routec_A_2026-10-20_120000.json@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
S3='experiments/2026-10-15_fpr_sweep_507_prep/results/routec_spec_A_preflight_2026-10-20_100000.pkl@bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
ok=1
check() {
  out=$(jq -c --arg j "$3" --arg f4 "$F4" --arg s3 "$S3" -f "$T/eq4_release_marker.jq" "$X/$1") || out="JQ_ERROR"
  if [ "$out" = "$2" ]; then echo "PASS $1 (j=$3) -> $out"; else echo "FAIL $1 (j=$3) -> $out (want $2)"; ok=0; fi
}
check ok.json '{"ok":true,"failing":[]}' A
check ok_with_prov.json '{"ok":true,"failing":[]}' A
check bad_other_job_signoff.json '{"ok":false,"failing":["signoff"]}' A
check bad_signed.json '{"ok":false,"failing":["signed"]}' A
check bad_by.json '{"ok":false,"failing":["by"]}' A
check bad_items_fail.json '{"ok":false,"failing":["items"]}' A
check bad_items_extra.json '{"ok":false,"failing":["items"]}' A
check bad_no_time.json '{"ok":false,"failing":["time_or_notes"]}' A
check bad_missing_s3.json '{"ok":false,"failing":["covered_missing"]}' A
check bad_other_sha.json '{"ok":false,"failing":["covered_other_sha"]}' A
check bad_other_job_entry.json '{"ok":false,"failing":["covered_other_job"]}' A
check bad_covered_form.json '{"ok":false,"failing":["covered_form"]}' A
check ok.json '{"ok":false,"failing":["signoff","covered_other_job"]}' B
if [ "$ok" = 1 ]; then echo "ALL PASS"; else echo "FAILED"; exit 1; fi
