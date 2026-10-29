#!/usr/bin/env bash
# The continuation's post-run pipeline (RULE.md amendment 15; LAUNCH.md step 18). ONE resumable script, armed on the laptop
# under nohup + caffeinate before the continuation ends; it never stops an app, a launcher or the devbox, and never kills a
# process it did not start. Each step leaves a done marker in $ST (written atomically); a rerun resumes at the first step
# without one. A FAILED verdict is kept in $ST/FAILED: a rerun refuses to continue until a person removes it.
#   1 wait      launch.json "finished" set AND the launcher's pane gone (pane_state.sh GONE); polled every 150 s
#   2 copyback  the launch folder back to the laptop run cache (rsync, --bwlimit; free disk checked; amendment 6)
#   3 gate12    gates 1-2 on the devbox (--expect-keys cont_keys.txt); the record copied to results/
#   4 settings  the final settings validation (--new on the finished launch); the record copied to results/
#   5 manifests the upload manifest and the counted-key manifest (--world simh_full, 09's name), validated against gate12;
#               committed + pushed
#   6 upload    upload_evals.sh (sha-checked put, Volume verify, then the laptop copies deleted)
#   7 stage     the corrected staging, run_simh_stage_c.py, its sha256 PINNED (the shared cut-think patch, ef3e9f7f)
#   8 format    format_check_simh.py on every staged epoch
#   9 extract   a fresh config (validated, committed), extract_simh.py on H200: one shard per 8 epochs (at most 19)
#  10 commit    every record committed and pushed; STATUS = DONE
# Network rule: only a recognised transport failure (ssh 255, a deadline, a listed transport error, no record written) is
# retried, with backoff (120 s doubling to 600 s, at most MAXNET times); it never stops anything. Anything else, and every
# checker's own verdict (read from its record, never from a client's exit code), is a FAILURE. Records are reconciled
# before any paid launch, so a restart never repeats a launch whose verdict is already recorded. Commits happen in this
# script's own detached worktree $WT, rebased onto the fetched branch and pushed as HEAD:$BRANCH to the HTTPS URL.
# Usage (laptop): WT=<worktree> X=<launch scratch dir> ST=<state dir> STAGER_SHA256=<pin> nohup caffeinate -i bash postrun_cont.sh
set -uo pipefail
WT=${WT:?}; X=${X:?}; ST=${ST:?}; STAGER_SHA256=${STAGER_SHA256:?}
BRANCH=aswin/q3-q35-simhonest-purchase
URL=https://github.com/redwoodresearch/probe-distribution-shift-from-sims.git
REL=experiments/2026-10-14_q3_q35_simhonest_purchase
P=$WT/$REL
L=$X/h/experiments/2026-10-07_phase2_attack_check
RC=$X/h/.run_cache/phase2_attack_check/runs
export DEVBOX=aswin@i-05b1d910fd70016e6 GIT_TERMINAL_PROMPT=0
RUN=2026-10-14_160932_simh_cont
PANE=%0:1226862                                        # allin_cont.log: "bound simhcont to pane %0:1226862"
K=phase2/simh_checks
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
REFS="/home/aswin/phase2/runs/2026-10-09_000523_rerun_strict /home/aswin/phase2/runs/2026-10-09_173815_scaleup_canary /home/aswin/phase2/runs/2026-10-09_182505_scaleup /home/aswin/phase2/runs/2026-10-10_001314_scaleup_topup"
VOL=probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13
GROUP=cont
MODAL=(uv run --no-project --python 3.12 --with modal==1.6.1)
NETPAT='ConnectionError|nodename nor servname|Temporary failure in name resolution|Name or service not known|StreamTerminatedError|UNAVAILABLE|Connection reset|Broken pipe|Could not resolve host|ClientClosed'
GITNET='Could not resolve host|Failed to connect|Connection (timed out|reset|refused)|Operation timed out|early EOF|RPC failed|The requested URL returned error: 5[0-9][0-9]'
MAXNET=30
mkdir -p "$ST" || exit 1

now() { TZ=America/Los_Angeles date '+%F %T PT'; }
log() { echo "$(now) $*" >> "$ST/postrun.log"; echo "$(now) $*" >&2; }      # never stdout: callers capture stdout
put() { printf '%s\n' "$2" > "$1.tmp" && mv "$1.tmp" "$1" || { echo "$(now) cannot write $1" >&2; exit 1; }; }  # atomic
status() { put "$ST/STATUS" "$(now) $*"; log "STATUS $*"; }
fail() { put "$ST/FAILED" "$1: $2"; status "FAILED $1: $2"; exit 1; }
alive() { [[ -e $ST/FAILED ]] && exit 1; return 0; } # after every $(…) capture: a failure inside a subshell stops us too
done_() { [[ -s $ST/done_$1 ]]; }
mark() { put "$ST/done_$1" "${2:-ok}"; log "step $1 done: ${2:-ok}"; }
backoff() {  # backoff <label>: one more network retry, or FAILED after MAXNET
  NETN=$(( ${NETN:-0} + 1 )); (( NETN > MAXNET )) && fail "$1" "network retries exhausted ($MAXNET; not a checker verdict)"
  log "$1: network failure; retry $NETN in ${W}s"; sleep "$W"; W=$(( W * 2 > 600 ? 600 : W * 2 ))
}
reset_net() { W=120; NETN=0; }
bounded() {  # bounded <seconds> <cmd…>: the command in its own process group, killed at the deadline (exit 142)
  perl -e 'my $t = shift; my $pid = fork(); die "fork" unless defined $pid;
           if ($pid == 0) { setpgrp(0, 0); exec @ARGV or exit 127 }
           local $SIG{ALRM} = sub { kill "KILL", -$pid; waitpid($pid, 0); exit 142 }; alarm $t;
           waitpid($pid, 0); my $s = $?; kill "KILL", -$pid; exit($s & 127 ? 128 + ($s & 127) : $s >> 8)' "$@"
}
box() {  # box <label> <s> <remote command>: rerun on ssh 255 / deadline 142; stdout = the remote output; rc = its verdict
  local label=$1 t=$2 out rc; reset_net
  while :; do
    out=$(bounded "$t" "$L/devbox.sh" "$3" 2>> "$ST/$label.err"); rc=$?
    (( rc == 255 || rc == 142 )) || { printf '%s' "$out"; return $rc; }
    backoff "$label"
  done
}
rsync_() {  # rsync_ <label> <s> <rsync args…>: retried on transport codes only (255 ssh, 142 deadline, 12/30/35 protocol)
  local label=$1 t=$2 rc; shift 2; reset_net
  while :; do
    bounded "$t" "$L/devbox.sh" rsync "$@" 2>> "$ST/$label.err"; rc=$?
    (( rc == 0 )) && return 0
    (( rc == 255 || rc == 142 || rc == 12 || rc == 30 || rc == 35 )) || return $rc
    backoff "$label"
  done
}
json() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {'d': d}))" "$@"; }
whole() { python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2> /dev/null; }   # a complete JSON record
newest_whole() {  # newest_whole <prefix>: the newest complete record with that prefix; incomplete ones moved aside
  local f
  for f in $(ls -t "$P/results" | grep "^$1"); do
    if whole "$P/results/$f"; then echo "$f"; return 0; fi
    mkdir -p "$ST/incomplete" && mv "$P/results/$f" "$ST/incomplete/" && log "moved an incomplete record aside: $f"
  done
  return 0
}
mvol_has() {  # mvol_has <path>: 0 if the path exists on the Volume, 1 if not; transport failures retried (a read)
  local out rc; reset_net
  while :; do
    out=$(bounded 120 "${MODAL[@]}" modal volume ls "$VOL" "$1" 2>&1); rc=$?
    (( rc == 0 )) && return 0
    grep -q 'does not exist' <<< "$out" && return 1
    (( rc == 142 )) || grep -Eq "$NETPAT" <<< "$out" || fail volume "modal volume ls $1 failed: ${out:0:200}"
    backoff "volume ls $1"
  done
}
git_push() {  # git_push <message> <paths…>: idempotent commit in $WT; rebase onto the fetched branch; push HEAD:$BRANCH
  local msg=$1 out rc; shift; reset_net
  (cd "$WT" && { [[ -d $(git rev-parse --git-path rebase-merge) || -d $(git rev-parse --git-path rebase-apply) ]] && git rebase --abort; true; } \
    && git add -- "$@" && { git diff --cached --quiet || git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EaTTB7CWcfzNHhcqan35WH"; }) >> "$ST/git.log" 2>&1 || fail git "commit failed (git.log)"
  while :; do
    out=$(cd "$WT" && bounded 300 git fetch -q "$URL" "$BRANCH" 2>&1); rc=$?
    if (( rc == 0 )); then
      out=$(cd "$WT" && git rebase -q FETCH_HEAD 2>&1) || { (cd "$WT" && git rebase --abort) > /dev/null 2>&1; fail git "rebase onto $BRANCH failed: ${out:0:200}"; }
      out=$(cd "$WT" && bounded 300 git push -q "$URL" "HEAD:$BRANCH" 2>&1); rc=$?
      (( rc == 0 )) && break
      grep -Eq 'non-fast-forward|fetch first' <<< "$out" && { backoff "git push (the branch moved)"; continue; }
    fi
    printf '%s\n' "$out" >> "$ST/git.log"
    (( rc == 142 )) || grep -Eq "$GITNET" <<< "$out" || fail git "git failed (not the network): ${out:0:200}"
    backoff "git"
  done
  log "pushed $(cd "$WT" && git rev-parse HEAD)"
}

[[ -e $ST/FAILED ]] && { echo "$(now) refusing to run: $(cat "$ST/FAILED") (remove $ST/FAILED deliberately to retry)" >&2; exit 1; }
log "armed (pid $$); state $ST; worktree $WT @ $(cd "$WT" && git rev-parse HEAD)"
[[ $(shasum -a 256 "$P/run_simh_stage_c.py" | cut -c1-64) == "$STAGER_SHA256" ]] || fail arm "run_simh_stage_c.py differs from its pin"

# 1 wait: the launch finished and its launcher's pane is gone (both read fresh; an unreadable state waits)
if ! done_ wait; then
  while :; do
    f=$(box wait 120 "python3 -c 'import json; print(\"FINISHED\" if json.load(open(\"/home/aswin/phase2/runs/$RUN/launch.json\")).get(\"finished\") else \"RUNNING\")'"); alive
    p=$(box pane 60 "bash ~/$K/pane_state.sh $PANE"); alive
    [[ $f == FINISHED && $p == GONE ]] && break
    status "waiting: launch ${f:-unreadable}, launcher pane ${p:-unreadable}"; sleep 150
  done
  mark wait "launch finished, pane gone"
fi

# 2 copy back (resumable rsync; laptop disk checked first)
if ! done_ copyback; then
  status "copying the launch back"
  free=$(df -g "$RC" | awk 'NR==2 {print $4}'); (( free >= 20 )) || fail copyback "${free} GB free < 20 GB"
  rsync_ copyback 3600 -a --bwlimit=20000 "box:phase2/runs/$RUN/" "$RC/$RUN/" || fail copyback "rsync failed (rc $?; copyback.err)"
  mark copyback
fi

# 3-4: each checker writes to .tmp, renamed only once it parses as JSON, so a record on the devbox is always whole; the
# verdict is its "pass"
G12=gate12_cont_$RUN.json
CHK="python3 -c 'import json,sys; json.load(open(sys.argv[1]))'"
if ! done_ gate12; then
  status "gates 1-2"
  box gate12 900 "cd ~/$K && { test -e $G12 || { rm -f $G12.tmp; python3 gate12.py --out $G12.tmp --expect-keys cont_keys.txt ~/phase2/runs/$RUN > /dev/null; $CHK $G12.tmp && mv $G12.tmp $G12; }; }" > /dev/null \
    || { alive; fail gate12 "gate12 wrote no whole record"; }
  rsync_ gate12 600 -a "box:$K/$G12" "$P/results/$G12" || fail gate12 "could not copy $G12 back"
  [[ $(json "$P/results/$G12" 'd["pass"] is True and d["launch"].endswith("/'$RUN'")') == True ]] || fail gate12 "gates 1-2 did not pass (results/$G12)"
  mark gate12 "$(json "$P/results/$G12" '(d["usable"], d["selected"], d["gate2_cost_per_called_run_usd"])')"
fi
SET_OUT=sim_settings_cont_final_$RUN.json
if ! done_ settings; then
  status "final settings validation"
  box settings 3600 "cd ~/$K && { test -e $SET_OUT || { rm -f $SET_OUT.tmp; $PY check_sim_settings.py --reference $REFS --expect-reference 191 --new ~/phase2/runs/$RUN --out $SET_OUT.tmp > /dev/null; $CHK $SET_OUT.tmp && mv $SET_OUT.tmp $SET_OUT; }; }" > /dev/null \
    || { alive; fail settings "the settings check wrote no whole record"; }
  rsync_ settings 600 -a "box:$K/$SET_OUT" "$P/results/$SET_OUT" || fail settings "could not copy $SET_OUT back"
  [[ $(json "$P/results/$SET_OUT" 'd["pass"] is True') == True ]] || fail settings "settings validation did not pass (results/$SET_OUT)"
  mark settings
fi

# 5 manifests: generated to .tmp, renamed only on success; validated against gate12 and each other; committed
MAN=manifest_simh_eval_upload_$GROUP.json
KEYS=simh_counted_keys_$GROUP.json
if ! done_ manifests; then
  status "manifests"
  rsync_ mck 600 -a "$P/make_counted_keys.py" "box:$K/make_counted_keys.py" || fail manifests "could not copy make_counted_keys.py"
  s=$(box sha 60 "sha256sum ~/$K/make_counted_keys.py | cut -c1-64"); alive
  [[ $s == $(shasum -a 256 "$P/make_counted_keys.py" | cut -c1-64) ]] || fail manifests "make_counted_keys.py on the devbox differs from the reviewed copy"
  box man 900 "cd ~/$K && { test -e $MAN || { rm -f $MAN.tmp; python3 make_upload_manifest.py --gate12 $G12 --out $MAN.tmp ~/phase2/runs/$RUN > /dev/null && $CHK $MAN.tmp && mv $MAN.tmp $MAN; }; } && { test -e $KEYS || { rm -f $KEYS.tmp; python3 make_counted_keys.py --gate12 $G12 --out $KEYS.tmp --world simh_full ~/phase2/runs/$RUN > /dev/null && $CHK $KEYS.tmp && mv $KEYS.tmp $KEYS; }; }" > /dev/null; rc=$?
  alive; (( rc == 0 )) || fail manifests "a manifest generator failed (rc $rc)"
  rsync_ man 600 -a "box:$K/$MAN" "$P/results/$MAN" || fail manifests "could not copy $MAN back"
  rsync_ keys 600 -a "box:$K/$KEYS" "$P/results/$KEYS" || fail manifests "could not copy $KEYS back"
  ok=$(python3 - "$P/results/$G12" "$P/results/$MAN" "$P/results/$KEYS" "$RUN" <<'EOF'
import json, sys
g, m, k = (json.load(open(p)) for p in sys.argv[1:4]); run = sys.argv[4]
usable = {r["run"] for r in g["runs"] if r["usable"]}
files = {(f["launch"], f["task"], f["sha256"]) for f in m["files"]}
keys = k["keys"]
print(all([
    m["counted_attempts_without_eval"]["simh"] == [], {f["launch"] for f in m["files"]} == {run}, len(m["files"]) == g["usable"],
    len(files) == len(m["files"]), {f["arm_dir"] for f in m["files"]} == {"sim_honest@september"},
    k["launch"] == run, k["n"] == g["selected"] == len(keys), k["usable"] == g["usable"], {x["run_key"] for x in keys} == {r["run"] for r in g["runs"]},
    {x["run_key"] for x in keys if x["usable"]} == usable, {x["world"] for x in keys} == {"simh_full"}, {x["launch_id"] for x in keys} == {run},
    {(run, x["task"], x["eval_sha256"]) for x in keys if x["usable"]} == files]))
EOF
)
  [[ $ok == True ]] || fail manifests "the manifests disagree with gate12 or each other"
  git_push "Continuation: gates 1-2, final settings validation, upload manifest and counted-key manifest (postrun_cont.sh)" \
    "$REL/results/$G12" "$REL/results/$SET_OUT" "$REL/results/$MAN" "$REL/results/$KEYS"
  mark manifests "$(json "$P/results/$MAN" 'len(d["files"])') files; keys $(json "$P/results/$KEYS" '(d["n"], d["usable"])')"
fi

# 6 upload (amendment 6). Each attempt's manifest sha256 is persisted BEFORE it starts; only a PASS from an attempt with
# this manifest counts. A retry after a transport death re-puts with --force (same bytes; the Volume verify decides). A
# verifier "passes": false, a sha256 mismatch or low disk is a FAILURE. After a PASS, only the remaining laptop copies are
# deleted, each sha256-checked first; every deletion must succeed.
if ! done_ upload; then
  status "uploading the evals (amendment 6)"
  msha=$(shasum -a 256 "$P/results/$MAN" | cut -c1-64)
  passed() { local a; for a in "$ST"/upload_attempt_*.sha; do [[ -e $a ]] || continue; n_=${a##*_}; n_=${n_%.sha}
    [[ $(cat "$a") == "$msha" ]] && grep -q '"passes": true}' "$X/upload_cont_$n_.log" 2> /dev/null && return 0; done; return 1; }
  n=$(ls "$ST"/upload_attempt_*.sha 2> /dev/null | wc -l | tr -d ' '); reset_net
  while :; do
    if passed; then
      jq -r '.files[] | [.path, .sha256] | @tsv' "$P/results/$MAN" > "$ST/delete_list.tsv" || fail upload "could not read the manifest"
      bad=0
      while IFS=$'\t' read -r path sha; do
        f=$RC/$path; [[ -e $f ]] || continue
        if [[ $(shasum -a 256 "$f" | cut -c1-64) == "$sha" ]] && rm -- "$f"; then echo "$(now) deleted laptop copy $f (verified on the Volume)" >> "$ST/upload_finish.log"
        else bad=$(( bad + 1 )); fi
      done < "$ST/delete_list.tsv"
      (( bad == 0 )) || fail upload "$bad laptop copies could not be verified or deleted"
      break
    fi
    n=$(( n + 1 )); UL=$X/upload_cont_$n.log
    [[ -e $UL ]] && fail upload "$UL exists without its attempt record"
    put "$ST/upload_attempt_$n.sha" "$msha"
    RUNCACHE=$RC LOG=$UL MIN_FREE_GB=20 FORCE_PUT=$(( n > 1 ? 1 : 0 )) bounded 7200 "$P/upload_evals.sh" "$P/results/$MAN" > "$ST/upload_$n.out" 2>&1; rc=$?
    grep -q 'done: .*verified and deleted' "$UL" 2> /dev/null && break
    grep -q '"passes": false}' "$UL" 2> /dev/null && fail upload "the Volume verify did not pass ($UL)"
    grep -Eq 'STOP: .*(sha256 differs|GB free)' "$UL" 2> /dev/null && fail upload "$(grep 'STOP: ' "$UL" | tail -1)"
    (( rc == 142 )) || grep -Eq "$NETPAT" "$ST/upload_$n.out" "$UL" 2> /dev/null || fail upload "upload attempt $n exited $rc (upload_$n.out)"
    backoff "upload (attempt $n)"
  done
  mark upload "$msha"
fi

# 7 corrected staging: the group allocation is persisted; a whole record is reconciled (success or stop) before any launch
stage_ok() { [[ $(json "$P/results/$1" '(not d["simh"]["assembled"].get("STOP")) and "c_gate" in d["simh"] and d["git"] != ""') == True ]]; }
if ! done_ stage; then
  status "corrected staging"
  [[ $(shasum -a 256 "$P/run_simh_stage_c.py" | cut -c1-64) == "$STAGER_SHA256" ]] || fail stage "run_simh_stage_c.py differs from its pin"
  [[ -s $ST/stage_try ]] || put "$ST/stage_try" 1
  reset_net
  while :; do
    try=$(cat "$ST/stage_try") || fail stage "unreadable stage_try"; (( try > 3 )) && fail stage "three staging names used without a success"
    g=$GROUP; (( try > 1 )) && g=${GROUP}_r$try
    if (( try > 1 )); then                               # same inputs under a fresh output name; published idempotently
      [[ -e $P/results/manifest_simh_eval_upload_$g.json ]] || cp "$P/results/$MAN" "$P/results/manifest_simh_eval_upload_$g.json" || fail stage "cp"
      cmp -s "$P/results/$MAN" "$P/results/manifest_simh_eval_upload_$g.json" || fail stage "manifest_simh_eval_upload_$g.json differs from $MAN"
      git_push "Continuation: re-stage under $g after a transport failure (same manifest)" "$REL/results/manifest_simh_eval_upload_$g.json"
    fi
    rec=$(newest_whole "simh_stage_c_${g}_[0-9]")
    if [[ -n $rec ]]; then stage_ok "$rec" && break; fail stage "the staging stopped (results/$rec)"; fi
    slog=$ST/stage_${g}_$(date +%s).log
    (cd "$WT" && SIMH_GROUP=$g SIMH_SET=simh SIMH_LAUNCHES=$RUN PYTHONPATH=src bounded 5400 "${MODAL[@]}" modal run --detach \
      "$REL/run_simh_stage_c.py") > "$slog" 2>&1; rc=$?
    rec=$(newest_whole "simh_stage_c_${g}_[0-9]")
    if [[ -n $rec ]]; then stage_ok "$rec" && break; fail stage "the staging stopped (rc $rc; results/$rec)"; fi
    if grep -q 'FileExistsError' "$slog"; then put "$ST/stage_try" $(( try + 1 )); continue; fi   # assembled by a lost client
    (( rc == 142 )) || grep -Eq "$NETPAT" "$slog" || fail stage "the staging exited $rc without a record ($slog)"
    backoff "staging $g"
  done
  mark stage "$g $rec"
fi
read -r SG SREC < "$ST/done_stage" && [[ -n ${SREC:-} ]] || fail stage "unreadable done marker"
EPOCHS=$(json "$P/results/$SREC" 'd["simh"]["assembled"]["epochs"]["sim_honest@september"]') || fail stage "unreadable record"
SSHA=$(json "$P/results/$SREC" 'd["simh"]["assembled"]["staged_inputs_sha256"]["sim_honest@september.jsonl"]') || fail stage "unreadable record"

# 8 the format check on every staged epoch: a whole record is reconciled first; its "passes" is the verdict
if ! done_ format; then
  status "format check on $EPOCHS epochs"; reset_net
  while :; do
    frec=$(newest_whole "format_check_simh_${SG}_[0-9]")
    if [[ -n $frec ]]; then
      [[ $(json "$P/results/$frec" 'd["passes"] is True and len(d["rows"]) == '"$EPOCHS") == True ]] && break
      fail format "the format check failed (results/$frec)"
    fi
    flog=$ST/format_$(date +%s).log
    (cd "$WT" && SIMH_GROUP=$SG SIMH_LAUNCHES=$RUN PYTHONPATH=src bounded 5400 "${MODAL[@]}" modal run --detach \
      "$REL/format_check_simh.py" --format-n "$EPOCHS") > "$flog" 2>&1; rc=$?
    [[ -n $(newest_whole "format_check_simh_${SG}_[0-9]") ]] && continue   # reconciled at the top of the loop
    (( rc == 142 )) || grep -Eq "$NETPAT" "$flog" || fail format "the format check exited $rc without a record ($flog)"
    backoff "format check"
  done
  mark format "$frec"
fi

# 9 extraction: the config is written once, validated against the intended text, and published before any launch;
# attempts are persisted before each launch; records and the Volume are reconciled first (complete → done; a gate,
# check or coverage failure → FAILED; only a transport loss launches the next attempt, which keeps the done epochs)
CFG=simh_extract_${SG}_c.yaml
ERUN=simh_q4b_extract_v1_${SG}_c
if ! done_ extract; then
  status "extraction ($EPOCHS epochs)"
  shards=$(( (EPOCHS + 7) / 8 )); (( shards > 19 )) && shards=19
  want=$(printf '%s\n' "# RULE.md amendments 11-15: Qwen3.5-27B extraction of the continuation's CORRECTED staged simulated honest epochs" \
    "# (results/$SREC; written by postrun_cont.sh)" "run: $ERUN" "inputs: simh_q4f_v1_$SG" "inputs_sha256:" \
    "  sim_honest@september.jsonl: $SSHA" "model: Qwen/Qwen3.5-27B" "shards: $shards" "chunk_size: 8" "pad_to: 4096")
  [[ -e $P/config/$CFG ]] || put "$P/config/$CFG" "$want"
  [[ $(cat "$P/config/$CFG") == "$want" ]] || fail extract "config/$CFG differs from the intended configuration"
  git_push "Continuation: staging and format records, extraction config ($EPOCHS epochs, $shards H200 shards)" \
    "$REL/config/$CFG" "$REL/results/$SREC" "$REL/results/$(cat "$ST/done_format")"
  [[ -s $ST/extract_attempt ]] || put "$ST/extract_attempt" 0
  while :; do
    a=$(cat "$ST/extract_attempt") || fail extract "unreadable extract_attempt"
    if (( a > 0 )); then                                 # reconcile the last attempt before any new launch
      erec=$(newest_whole "extract_simh_${ERUN}_a${a}_")
      elog=$ST/extract_a$a.log
      nerr=0
      if [[ -n $erec ]]; then
        nerr=$(json "$P/results/$erec" 'sum("error" in s for s in d["shards"])') || fail extract "unreadable results/$erec"
        if (( nerr == 0 )) && mvol_has "$ERUN/complete.json"; then break; fi
        nbad=$(NETPAT=$NETPAT python3 -c 'import json, os, re, sys; d = json.load(open(sys.argv[1]))
print(sum(1 for s in d["shards"] if "error" in s and not re.search(os.environ["NETPAT"], s["error"])))' "$P/results/$erec") \
          || fail extract "unreadable results/$erec"
        (( nbad == 0 )) || fail extract "$nbad shard(s) failed with a non-transport error (results/$erec)"
      fi
      grep -q 'extraction incomplete' "$elog" 2> /dev/null && fail extract "coverage incomplete after attempt $a ($elog)"
      if (( nerr == 0 )); then                           # no shard error recorded: only a lost client may relaunch
        grep -Eq "$NETPAT" "$elog" 2> /dev/null || [[ $(cat "$ST/extract_rc_$a" 2> /dev/null) == 142 ]] \
          || fail extract "attempt $a ended without a complete extraction and without a transport failure ($elog)"
      fi
      (( a >= 3 )) && fail extract "three extraction attempts used (latest results/${erec:-none})"
      log "extraction attempt $a was lost to a transport failure; the next attempt keeps the done epochs"; sleep 120
    fi
    a=$(( a + 1 )); put "$ST/extract_attempt" "$a"     # persisted BEFORE the launch: a restart never reuses an attempt
    (cd "$WT" && PYTHONPATH=src bounded 5400 "${MODAL[@]}" --with pyyaml modal run --detach "$REL/extract_simh.py" \
      --config "$CFG" --attempt "$a") > "$ST/extract_a$a.log" 2>&1; put "$ST/extract_rc_$a" $?
  done
  mark extract "$erec"
fi

# 10 records committed; STATUS = DONE with the routing fields
if ! done_ commit; then
  git_push "Continuation: corrected extraction complete (postrun_cont.sh)" "$REL/results/$(cat "$ST/done_extract")" \
    "$REL/results/$(cat "$ST/done_format")" "$REL/results/$SREC" "$REL/config/$CFG"
  mark commit "$(cd "$WT" && git rev-parse HEAD)"
fi
status "DONE: staged $SG ($EPOCHS epochs, sha256 $SSHA; results/$SREC), format $(cat "$ST/done_format"), extraction $(cat "$ST/done_extract"), pushed @ $(cat "$ST/done_commit")"
