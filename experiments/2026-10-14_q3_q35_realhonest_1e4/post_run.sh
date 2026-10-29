#!/usr/bin/env bash
# After the relaunch (RULE.md, Amendments 13-14): one resumable script, armed before the relaunch ends, on the laptop:
#   nohup caffeinate -i bash post_run.sh > <state dir>/post_run.out 2>&1 &
# Each step runs once (a marker in the state dir); a rerun resumes where it left off. One copy at a time (a lock dir).
#   wait     the frozen teardown watcher has finished and every file the six upload manifests list is on the laptop with its
#            size (if not, and the devbox is stopped: start it, copy over SSM, stop it again, confirmed; if it is running,
#            copy only)
#   s3       one `aws s3 sync`; object count and bytes equal the local folder's
#   upload   upload_replicate.py per part (resumable: files an earlier run deleted are verified on the Volume first)
#   manifest make_rh_upload_list.py (both launches, world/stratum/launch_id/rerun_key per key), committed
#   stage    run_rh_stage.py --target rh, only from the pinned file (sha256 in the state dir's stage_pin); if the staged inputs
#            are already on the Volume (a dropped client), the c_gate alone; the passing record is kept with the staged sha256
#   format   the format check on 5 seeded runs (must pass) and its negative control (must show the bug on both runs)
#   repro    --repro-shards 32 unless present; both staged sha256s written into config/rh1e4_extract_v1.yaml and checked; committed
#   extract  extract_rh.py, attempts numbered across restarts, each only after a fresh billing read keeps the job under $405
#   compare  compare_repro.py --arm repro-c
#   cued     cue_d_rh.py --target rh
#   done     STATUS says DONE with the facts for the hand-off message
# Network rule: reads and transport failures are retried with backoff and never stop anything; a Volume listing that fails
# is an error to retry, never "absent". A gate failure ("GATE FAILED", a failed reference bar, a c_gate error) is final: it
# is recorded (gate_failed_<step>) and never retried into a pass. Nothing here stops a server or app; the only machine it
# may stop is a devbox it started itself.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
WT=$SP/wt_q35_realhonest
D=experiments/2026-10-14_q3_q35_realhonest_1e4
R=$WT/$D
E=$SP/rh_harness/experiments/2026-10-07_phase2_attack_check
RUN=2026-10-14_160335_honest_rh_relaunch
L=$SP/rh_harness/.run_cache/phase2_attack_check/runs/$RUN
WLOG=$SP/rh_watch/watch_and_stop_relaunch_2026-10-14_160335.log
S3=s3://individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs/$RUN
VOL=probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14
BOX=i-0abea93b8ad99ee41
CAP=405
STATE=$SP/rh_post
mkdir -p "$STATE"
UV=(uv run --no-project --python 3.12 --with modal==1.6.1 --with pyyaml)
export PYTHONPATH=$WT/src
cd "$WT" || exit 1  # every command runs from the worktree, in this shell (a gate failure exits the script, not a subshell)
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*" >> "$STATE/post_run.log"; }
status() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*" > "$STATE/STATUS.txt"; log "STATUS: $*"; }
done_() { [[ -e $STATE/done_$1 ]]; }
mark() { touch "$STATE/done_$1"; log "step $1 done"; }
fail() { status "STOPPED at step $1: $2 (see post_run.log); nothing else was stopped"; exit 1; }
gate_fail() { echo "$2" > "$STATE/gate_failed_$1"; fail "$1" "GATE FAILED (final, not retried): $2"; }
for g in "$STATE"/gate_failed_*; do [[ -e $g ]] && { status "STOPPED earlier on a failed gate: $(basename "$g"): $(cat "$g")"; exit 1; }; done
acquire() {  # Codex round 19: fail closed; a lock left by a dead run is removed by hand (rm -r <state dir>/lock), never here
  if mkdir "$STATE/lock" 2>/dev/null; then echo $$ > "$STATE/lock/pid"; return 0; fi
  echo "a lock exists ($STATE/lock, pid $(cat "$STATE/lock/pid" 2>/dev/null)); if that run is dead, remove the lock by hand"
  exit 1
}
acquire
trap '[[ $(cat "$STATE/lock/pid" 2>/dev/null) == "$$" ]] && rm -rf "$STATE/lock"' EXIT
while pgrep -f "modal run $D/" > /dev/null; do log "a Modal client of an earlier run is still running; waiting"; sleep 120; done
retry() {  # retry <tries> <command...>: backoff 60 s, 120 s, ... capped at 10 min
  local max=$1 n=0 rc; shift
  while :; do
    [[ " $* " == *run_rh_stage.py* ]] && pinned  # the pin, before every call to the staging script (retries included)
    "$@" >> "$STATE/post_run.log" 2>&1 && return 0
    rc=$?; n=$((n + 1)); log "attempt $n of $max failed (exit $rc): $*"
    ((n >= max)) && return "$rc"
    sleep $((n < 10 ? 60 * n : 600))
  done
}
GATES="GATE FAILED|fail the bar|reference coverage|staged keys without a meta|non-finite rows|c gate:|cue_D: errors|cue_D: [0-9]+ selected runs"
run_gate() {  # run_gate <step> <tries> <command...>: transport failures retried; output naming a gate failure is final
  local step=$1 max=$2 n=0 rc out; shift 2
  out=$STATE/last_$step.out
  while :; do
    [[ " $* " == *run_rh_stage.py* ]] && pinned  # the pin, before every call to the staging script (retries included)
    "$@" > "$out" 2>&1; rc=$?
    cat "$out" >> "$STATE/post_run.log"
    ((rc == 0)) && return 0
    grep -qE "$GATES" "$out" && gate_fail "$step" "$(grep -oE "($GATES)[^\"]{0,160}" "$out" | head -1)"
    n=$((n + 1)); log "$step: attempt $n of $max failed (exit $rc), not a gate failure"
    ((n >= max)) && return "$rc"
    sleep $((60 * n))
  done
}
vol_has() {  # vol_has <path>: 0 listed, 1 absent; a failed listing is retried (never read as absent), then the step fails
  local n=0 out
  while ((n < 8)); do
    out=$("${UV[@]}" modal volume ls "$VOL" "$(dirname "$1")" 2>&1) && { grep -qF "$1" <<< "$out"; return; }
    grep -qiE "no such file|not found|does not exist" <<< "$out" && return 1
    n=$((n + 1)); log "volume listing failed ($n of 8): $(tail -1 <<< "$out")"; sleep $((60 * n))
  done
  fail "${STEP:-?}" "the Volume could not be listed"
}
vol_sha() {  # retried; prints the sha256 or returns 1 after 6 failed reads
  local out n
  for n in 1 2 3 4 5 6; do out=$("${UV[@]}" python "$R/vol_sha.py" "$1" 2>/dev/null) && { awk '{print $1}' <<< "$out"; return 0; }; sleep $((60 * n)); done
  return 1
}
pushed() { [[ $(git -c url."https://github.com/".insteadOf="git@github.com:" ls-remote origin refs/heads/aswin/q3-q35-realhonest-1e4 2>/dev/null | cut -f1) == "$(git rev-parse HEAD)" ]]; }
commit() {  # commit <message> <paths...>: only those paths (none: nothing to commit); then push until the remote has HEAD
  local msg=$1; shift
  if (($# > 0)); then
    { git add -- "$@" && { git diff --cached --quiet -- "$@" || git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EaTTB7CWcfzNHhcqan35WH" -- "$@"; }; } >> "$STATE/post_run.log" 2>&1 || return 1
  else
    log "commit: no new files for: $msg"
  fi
  retry 20 git -c url."https://github.com/".insteadOf="git@github.com:" push -q origin aswin/q3-q35-realhonest-1e4 || return 1
  retry 5 pushed  # a resumed step whose earlier commit never reached the remote is pushed here too
}
pinned() {  # the staging script equals the pin (d9: one pinned file and sha, switchable once the shared gate fix is relayed)
  local pin; pin=$(cat "$STATE/stage_pin" 2>/dev/null)
  [[ -n $pin && $(shasum -a 256 "$R/run_rh_stage.py" | awk '{print $1}') == "$pin" ]] || fail "${STEP:-?}" "run_rh_stage.py does not match the pin in $STATE/stage_pin"
}
new_results() { (cd "$WT" && git ls-files --others --exclude-standard "$D/results" | grep -E "$1"); }
spent_job() {  # a fresh billing read: this job's Modal apps since 10-14 00:00 PT, plus the devbox's hours; prints dollars
  local out
  out=$("${UV[@]}" modal billing report --start 2026-10-14 --resolution h --tz America/Los_Angeles --json 2>/dev/null) || return 1
  python3 -I -c '
import json, sys, time
rows = json.loads(sys.argv[1])
mine = ("probe-shift_aswin-surya_sub-phase2_srvc34c93", "probe-shift_aswin-surya_sub-phase2_srvba41a2", "probe-shift_aswin-surya_sub-q3realhonest_")
modal_usd = sum(float(r["cost"]) for r in rows if str(r.get("description", "")).startswith(mine))
devbox_h = (time.time() - 1792006740) / 3600  # 12:39 PT 10-14, m7i.16xlarge, as if never stopped (an upper bound)
print(round(modal_usd + 3.2256 * devbox_h, 2))' "$out"
}
log "post_run.sh started (pid $$)"
status "armed: waiting for the relaunch to end"

# wait
STEP=wait
need() {  # every file the six upload manifests list is here with its size (parts already uploaded are skipped)
  [[ -s $L/launch.json && -s $L/gate1_final.json ]] || return 1
  (($(ls "$L"/upload_manifest_*.tsv 2>/dev/null | wc -l) == 6)) || return 1
  local part path size
  for part in common r1 r2 r3 r4 r5; do
    done_ "up_$part" && continue
    while IFS=$'\t' read -r path size _; do
      [[ -z $path ]] && continue
      [[ -f $L/$path && $(stat -f %z "$L/$path") == "$size" ]] || return 1
    done < "$L/upload_manifest_$part.tsv"
  done
}
if ! done_ wait; then
  until grep -qE "done: apps stopped|FAILED, stop or copy" "$WLOG" 2>/dev/null; do sleep 180; done
  log "teardown watcher finished: $(grep -E 'done: apps stopped|FAILED' "$WLOG" | tail -1)"
  if [[ -e $STATE/devbox_started_by_post_run ]] || ! need; then
    st=$(aws ec2 describe-instances --profile redwood-sandbox --region us-west-2 --instance-ids $BOX --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
    log "run folder incomplete on the laptop (devbox state: $st)"
    if [[ $st == stopped ]]; then
      touch "$STATE/devbox_started_by_post_run"
      retry 5 aws ec2 start-instances --profile redwood-sandbox --region us-west-2 --instance-ids $BOX || fail wait "devbox start"
      sleep 150
    fi
    retry 10 env DEVBOX=aswin@$BOX bash "$E/devbox.sh" rsync -az "box:phase2/runs_rh1e4/$RUN/" "$L/" || fail wait "copy over SSM"
    if [[ -e $STATE/devbox_started_by_post_run ]]; then
      stopped=""
      for i in 1 2 3 4 5 6 7 8; do
        aws ec2 stop-instances --profile redwood-sandbox --region us-west-2 --instance-ids $BOX >> "$STATE/post_run.log" 2>&1
        sleep $((60 * i))
        [[ $(aws ec2 describe-instances --profile redwood-sandbox --region us-west-2 --instance-ids $BOX --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null) =~ stopping|stopped ]] && { stopped=1; break; }
      done
      [[ -n $stopped ]] || fail wait "the devbox this script started is not confirmed stopping or stopped (marker kept; the on-box shutdown still applies)"
      rm -f "$STATE/devbox_started_by_post_run"
    fi
    need || fail wait "run folder still incomplete after the copy"
  fi
  grep -q '"launch_finished": null' "$L/gate1_final.json" && fail wait "the launch did not finish (gate1_final.json: launch_finished null)"
  mark wait
fi
status "running: the relaunch ended; relaying its files"

# s3
STEP=s3
if ! done_ s3; then
  retry 5 aws s3 sync --profile redwood-sandbox --only-show-errors "$L" "$S3" || fail s3 "aws s3 sync"
  local_n=$(find "$L" -type f | wc -l | tr -d ' '); local_b=$(find "$L" -type f -exec stat -f %z {} + | awk '{s += $1} END {print s}')
  remote=""
  for i in 1 2 3 4 5; do
    remote=$(aws s3 ls --profile redwood-sandbox --recursive --summarize "$S3/" 2>/dev/null | tail -2 | awk '{print $3}' | tr '\n' ' ') && [[ -n $remote ]] && break
    sleep $((60 * i))
  done
  log "S3: remote objects and bytes: $remote; local $local_n $local_b"
  [[ $remote == "$local_n $local_b " ]] || fail s3 "S3 count or bytes differ from the local folder"
  mark s3
fi

# upload
STEP=upload
for part in common r1 r2 r3 r4 r5; do
  done_ "up_$part" && continue
  if [[ ! -s $L/upload_manifest_$part.tsv ]]; then log "part $part: empty manifest, nothing to upload"; mark "up_$part"; continue; fi
  retry 5 "${UV[@]}" python "$R/upload_replicate.py" "$L" "$part" "$STATE/upload_log_relaunch.tsv" || fail upload "part $part"
  mark "up_$part"
done

# manifest
STEP=manifest
if ! done_ manifest; then
  M=$R/config/manifest_rh1e4_eval_upload_v1.json
  [[ -e $M ]] || retry 5 "${UV[@]}" modal run "$D/make_rh_upload_list.py" || fail manifest "make_rh_upload_list.py"
  python3 -I -c 'import json, sys; m = json.load(open(sys.argv[1])); assert m["files"] and m["relaunched_keys"] == 74' "$M" || fail manifest "the manifest is not whole"
  commit "1e-4 real honest runs: the counted-attempt manifest of both launches (post_run.sh)" "$D/config/manifest_rh1e4_eval_upload_v1.json" || fail manifest "commit or push"
  mark manifest
fi

# stage
STEP=stage
if ! done_ stage; then
  pinned
  if ! vol_has rh1e4_stage_v1/inputs/honest_b.jsonl; then
    run_gate stage 3 "${UV[@]}" modal run "$D/run_rh_stage.py" --target rh || vol_has rh1e4_stage_v1/inputs/honest_b.jsonl || fail stage "staging did not complete"
  fi
  if [[ -z $(new_results "rh_stage_rh_2026|rh_stage_rh_cgate_") ]]; then  # no local gate record (a dropped client): the c_gate alone
    run_gate stage 3 "${UV[@]}" modal run "$D/run_rh_stage.py" --target rh --c-gate-only || fail stage "c_gate could not run"
  fi
  sha=$(vol_sha rh1e4_stage_v1/inputs/honest_b.jsonl) || fail stage "could not hash the staged inputs"
  echo "$sha" > "$STATE/stage_ok"
  commit "1e-4 real honest runs: the relaunch's staging record (post_run.sh)" $(new_results "rh_stage_rh_") || fail stage "commit or push"
  mark stage
fi

# format
STEP=format
if ! done_ format; then
  pinned
  run_gate format 3 "${UV[@]}" modal run "$D/run_rh_stage.py" --target rh --format-n 5 || fail format "the format check could not run"
  run_gate format 3 "${UV[@]}" modal run "$D/run_rh_stage.py" --negative || fail format "the negative control could not run"
  commit "1e-4 real honest runs: format check on the relaunch's staging and its negative control (post_run.sh)" $(new_results "format_check_") || fail format "commit or push"
  mark format
fi

# repro
STEP=repro
if ! done_ repro; then
  if ! vol_has rh1e4_stage_v1/inputs/repro-c.jsonl; then
    pinned
    retry 3 "${UV[@]}" modal run "$D/run_rh_stage.py" --repro-shards 32 || vol_has rh1e4_stage_v1/inputs/repro-c.jsonl || fail repro "add_repro"
  fi
  hb=$(vol_sha rh1e4_stage_v1/inputs/honest_b.jsonl) || fail repro "could not hash honest_b.jsonl"
  rc=$(vol_sha rh1e4_stage_v1/inputs/repro-c.jsonl) || fail repro "could not hash repro-c.jsonl"
  [[ $hb == "$(cat "$STATE/stage_ok")" ]] || fail repro "honest_b.jsonl changed since its gate passed"
  Y=$R/config/rh1e4_extract_v1.yaml
  sed -i '' -e "s/honest_b.jsonl: TO_FILL_FROM_STAGING/honest_b.jsonl: $hb/" -e "s/repro-c.jsonl: TO_FILL_FROM_ADD_REPRO/repro-c.jsonl: $rc/" "$Y" || fail repro "sed"
  grep -q "honest_b.jsonl: $hb" "$Y" && grep -q "repro-c.jsonl: $rc" "$Y" && ! grep -q TO_FILL "$Y" || fail repro "the extraction config does not hold the staged sha256s"
  commit "1e-4 real honest runs: the extraction config's staged-input sha256s (post_run.sh)" "$D/config/rh1e4_extract_v1.yaml" || fail repro "commit or push"
  log "staged honest_b.jsonl $hb; repro-c.jsonl $rc"
  mark repro
fi

# extract
STEP=extract
if ! done_ extract; then
  until vol_has rh1e4_extract_v1/complete.json; do
    attempt=$(($(cat "$STATE/extract_attempt" 2>/dev/null || echo 0) + 1))
    ((attempt <= 4)) || fail extract "no complete.json after 4 attempts"
    spent=""
    for i in 1 2 3 4 5 6; do spent=$(spent_job) && break; log "billing read failed ($i of 6)"; sleep $((120 * i)); done
    [[ -n $spent ]] || fail extract "no fresh billing reading (nothing started)"
    prior=$((attempt - 1))  # every earlier attempt reserved at its worst ($85) until billing is beyond doubt: conservative
    python3 -I -c "import sys; sys.exit(0 if $spent + 35 + 85 * $prior + 85 <= $CAP else 1)" || fail extract "budget: about \$$spent spent + \$35 unbilled + \$85 x $prior earlier attempts + \$85 > \$$CAP (nothing started)"
    echo $attempt > "$STATE/extract_attempt"
    log "extraction attempt $attempt (job spend so far about \$$spent)"
    "${UV[@]}" modal run "$D/extract_rh.py" --config rh1e4_extract_v1.yaml --attempt $attempt >> "$STATE/post_run.log" 2>&1
    log "extraction attempt $attempt exit $?"
  done
  commit "1e-4 real honest runs: the relaunch's extraction records (post_run.sh)" $(new_results "extract_rh_rh1e4") || fail extract "commit or push"
  mark extract
fi

# compare
STEP=compare
if ! done_ compare; then
  run_gate compare 3 "${UV[@]}" modal run "$D/compare_repro.py" --run rh1e4_extract_v1 --arm repro-c || fail compare "compare_repro.py could not run"
  commit "1e-4 real honest runs: the per-shard reference gate of the relaunch's extraction (post_run.sh)" $(new_results "compare_repro_rh1e4") || fail compare "commit or push"
  mark compare
fi

# cued
STEP=cued
if ! done_ cued; then
  run_gate cued 3 "${UV[@]}" modal run "$D/cue_d_rh.py" --target rh || fail cued "cue_d_rh.py could not run"
  commit "1e-4 real honest runs: cue_D on the counted runs (post_run.sh)" $(new_results "cue_d_rh_") || fail cued "commit or push"
  mark cued
fi

head=$(cd "$WT" && git rev-parse --short=9 HEAD)
status "DONE @ $head: staged honest_b.jsonl $(cat "$STATE/stage_ok"); extraction rh1e4_extract_v1 on $VOL; see post_run.log"
log "post_run.sh finished"
