#!/usr/bin/env bash
# Move one phase's usable .eval files from the laptop's run cache to the Modal Volume (RULE.md amendment 6). MOVE ONLY:
# files are hashed, uploaded, verified on the Volume, then deleted; never opened, read, parsed or printed.
#   1. For each manifest file, one at a time: check free disk (>= MIN_FREE_GB), check its sha256 against the manifest,
#      `modal volume put` it to simh_evals_v1/<path>, and log name, bytes, sha256, destination, time.
#   2. Verify every file on the Volume (verify_simh_upload.py: count, bytes, sha256; Modal CPU, read-only).
#   3. Only after a PASS: delete each laptop copy, logging each deletion.
# The log stays outside git. Usage (laptop, from the repo root):
#   RUNCACHE=<…/.run_cache/phase2_attack_check/runs>
#   LOG=<file outside git> MIN_FREE_GB=20 upload_evals.sh <manifest.json>
set -euo pipefail
VOL=probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13  # pinned; verify_simh_upload.py checks the same Volume
M=$1; RC=${RUNCACHE:?}; LOG=${LOG:?}; MIN=${MIN_FREE_GB:-20}; HERE=$(cd "$(dirname "$0")" && pwd)
[[ ! -e $LOG ]] || { echo "$LOG exists: use a fresh log per upload (its PASS line is what allows deletion)" >&2; exit 1; }
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*" | tee -a "$LOG"; }
modal_() { perl -e 'alarm shift; exec @ARGV' "$1" uv run --no-project --python 3.12 --with modal==1.6.1 modal "${@:2}"; }
jq -r '.files[] | [.path, (.bytes|tostring), .sha256] | @tsv' "$M" > "$LOG.list"
while IFS=$'\t' read -r path bytes sha; do  # bash: `path` is an ordinary variable here
  free=$(df -g "$RC" | awk 'NR==2 {print $4}')
  (( free >= MIN )) || { log "STOP: ${free} GB free < ${MIN} GB"; exit 1; }
  src=$RC/$path
  [[ $(shasum -a 256 "$src" | cut -c1-64) == "$sha" ]] || { log "STOP: $path sha256 differs from the manifest"; exit 1; }
  force=(); [[ ${FORCE_PUT:-0} == 1 ]] && force=(--force)  # postrun_cont.sh's retry after a network death: same bytes
  modal_ 600 volume put ${force[@]+"${force[@]}"} "$VOL" "$src" "simh_evals_v1/$path" > /dev/null
  log "uploaded $path bytes=$bytes sha256=$sha -> $VOL:simh_evals_v1/$path"
done < "$LOG.list"
# judged by the verifier's own JSON line (the modal client can raise CancelledError while tearing down after printing it)
PYTHONPATH=src modal_ 900 run "$HERE/verify_simh_upload.py" --manifest "$M" >> "$LOG" 2>&1 || true
n=$(wc -l < "$LOG.list" | tr -d ' ')
grep -q "{\"files_on_volume\": $n, \"files_in_manifest\": $n, \"extra_on_volume\": 0, \"missing_or_mismatched\": 0, \"passes\": true}" "$LOG" \
  || { log "STOP: the Volume check did not pass; laptop copies kept"; exit 1; }
while IFS=$'\t' read -r fp bytes sha; do
  rm -- "${RC:?}/${fp:?}"; log "deleted laptop copy $RC/$fp (verified on the Volume)"
done < "$LOG.list"
log "done: $(wc -l < "$LOG.list") files uploaded, verified and deleted from the laptop"
