#!/bin/zsh -f
# Test of s1b7_marker.jq on synthetic markers (fixtures_s1b7/; no real marker). Positive control first (covered in another order than
# the pins), then one case per check. Run from the repo root:
#   zsh -f experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_s1b7_marker.sh
T=experiments/2026-10-15_fpr_sweep_507_prep/tools_routec
X=$T/fixtures_s1b7
PWM=xfer/r/pwm.json@1111111111111111111111111111111111111111111111111111111111111111
PSR=xfer/r/psr.json@2222222222222222222222222222222222222222222222222222222222222222
PWA=config/a.json@3333333333333333333333333333333333333333333333333333333333333333
ok=1
check() {
  out=$(jq -c --arg pwm "$PWM" --arg psr "$PSR" --arg pwa "$PWA" -f "$T/s1b7_marker.jq" "$X/$1") || out="JQ_ERROR"
  if [ "$out" = "$2" ]; then echo "PASS $1 -> $out"; else echo "FAIL $1 -> $out (want $2)"; ok=0; fi
}
check ok.json '{"ok":true,"failing":[]}'
check bad_signoff.json '{"ok":false,"failing":["signoff"]}'
check bad_signed.json '{"ok":false,"failing":["signed"]}'
check bad_by.json '{"ok":false,"failing":["by"]}'
check bad_items_extra.json '{"ok":false,"failing":["items"]}'
check bad_items_fail.json '{"ok":false,"failing":["items"]}'
check bad_no_notes.json '{"ok":false,"failing":["time_or_notes"]}'
check bad_covered_4.json '{"ok":false,"failing":["covered"]}'
check bad_covered_sha.json '{"ok":false,"failing":["covered"]}'
check bad_covered_dup.json '{"ok":false,"failing":["covered"]}'
if [ "$ok" = 1 ]; then echo "ALL PASS"; else echo "FAILED"; exit 1; fi
