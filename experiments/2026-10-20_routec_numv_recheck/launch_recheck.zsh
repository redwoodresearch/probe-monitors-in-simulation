#!/bin/zsh
# Launch the independent recheck ONCE, after the review PASS and refresh3's arm.
#   zsh experiments/2026-10-20_routec_numv_recheck/launch_recheck.zsh <reviewed HEAD sha, 40 hex>
# Guards, in order: attached branch; HEAD == the reviewed sha; clean experiment folder; run script sha256; awake (AC
# power, then the client runs under caffeinate); no live app of the same name (live_app_gate.jq, fail closed). Then a
# single detached `modal run`, stdout+stderr to a fresh log; this prints only the RECHECK_RESULT line and the exit code.
set -euo pipefail
WANT=${1:?usage: launch_recheck.zsh <reviewed HEAD sha>}
WT=/Users/aswinsurya/probe-shift-numv-recheck-wt
EXP=experiments/2026-10-20_routec_numv_recheck
RUN_SHA=d050cbe15574ee4cd810c41f9a79a6f580d2afe3ee353f57d61c5416bedc37d4
APP=probe-shift_aswin-surya_sub-routec-numv_recheck_kill-2026-11-19
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
LOGD=/Users/aswinsurya/probe-shift-numv-recheck-logs

cd "${WT}"
[[ "$(git symbolic-ref -q HEAD)" == refs/heads/aswin/routec-numv-recheck ]] || { print -u2 "STOP: branch"; exit 1; }
[[ "$(git rev-parse HEAD)" == "${WANT}" ]] || { print -u2 "STOP: HEAD is not the reviewed sha"; exit 1; }
[[ -z "$(git status --porcelain -- "${EXP}")" ]] || { print -u2 "STOP: uncommitted changes in ${EXP}"; exit 1; }
[[ "$(shasum -a 256 "${EXP}/run_recheck_independent.py" | cut -d' ' -f1)" == "${RUN_SHA}" ]] \
  || { print -u2 "STOP: run script sha256"; exit 1; }
# awake: caffeinate cannot hold a closed lid on battery, so require AC power; the client then runs under caffeinate
pmset -g batt | head -1 | grep -q "AC Power" || { print -u2 "STOP: not on AC power"; exit 1; }
AL="$("${M}" app list --json)"
LIVE="$(jq --arg a "${APP}" -f "${EXP}/live_app_gate.jq" <<<"${AL}")"
[[ "${LIVE}" == 0 ]] || { print -u2 "STOP: live-app gate says ${LIVE}"; exit 1; }

mkdir -p "${LOGD}"
LOG="${LOGD}/recheck_$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S).log"
[[ ! -e "${LOG}" ]] || { print -u2 "STOP: log exists"; exit 1; }
print "launch $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z') HEAD ${WANT} log ${LOG}"
RC=0
caffeinate -i -s "${M}" run --detach "${EXP}/run_recheck_independent.py" >"${LOG}" 2>&1 || RC=${?}
grep '^RECHECK_RESULT ' "${LOG}" || true
print "client exit ${RC}"
exit ${RC}
