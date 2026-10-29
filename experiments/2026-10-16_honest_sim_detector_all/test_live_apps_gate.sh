#!/usr/bin/env bash
# Lean fixture test of live_apps_gate.jq (synthetic app lists only; no Modal, no git). Run from this folder: bash test_live_apps_gate.sh
set -u
N=probe-shift_aswin-surya_sub-honsimdet_routec_kill-2026-11-15
pass=0; fail=0
t() {  # name expected_rc json
  printf '%s' "$3" | jq -e --arg n "$N" -f live_apps_gate.jq >/dev/null 2>&1; rc=$?
  if [ "$rc" -eq "$2" ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $1: rc $rc, expected $2"; fi
}
t "empty list" 0 '[]'
t "only stopped / other apps" 0 "[{\"description\":\"other\",\"state\":\"deployed\"},{\"description\":\"$N\",\"state\":\"stopped\"}]"
t "live deployed" 1 "[{\"description\":\"$N\",\"state\":\"deployed\"}]"
t "live ephemeral" 1 "[{\"description\":\"$N\",\"state\":\"ephemeral (detached)\"}]"
t "upper-case keys (schema change)" 5 "[{\"Description\":\"$N\",\"State\":\"deployed\"}]"
t "missing state" 5 "[{\"description\":\"$N\"}]"
t "not an array" 5 '{"description":"x","state":"deployed"}'
t "non-string state" 5 "[{\"description\":\"$N\",\"state\":null}]"
[ "$fail" -eq 0 ] && echo "PASS $pass/$((pass + fail))" || { echo "FAIL $fail/$((pass + fail))"; exit 1; }
