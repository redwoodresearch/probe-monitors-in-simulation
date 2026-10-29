#!/usr/bin/env bash
# Fresh-task honest runs (PLAN.md stage B/C): move one finished run folder off the devbox, as zoo C's were.
#   1. the devbox lists the runs (list_fresh_runs.py: run ids, eval files and sha256s, steps, errored; no text);
#   2. rsync over the SSM tunnel to <local root>, then every file's sha256 checked against the devbox's;
#   3. S3, write-once with a manifest (../2026-10-12_q3_llama70b_price/archive_run.py: If-None-Match '*', x-amz-meta-sha256),
#      keys phase2/runs/<run>/..., the launcher, watcher, docker-events and resource logs and the run list beside it;
#   4. `modal volume put` of the run folder and the run list to the work Volume, for stage_fresh_modal.py.
# The laptop only moves bytes and checks hashes. No transcript text is printed.
# Usage: DEVBOX=aswin@i-... transfer_run.sh <run name> <stage> <git sha> <local root>
set -euo pipefail
RUN=$1; STAGE=$2; GIT=$3; ROOT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
DB="$HERE/../2026-10-07_phase2_attack_check/devbox.sh"
WORK=probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13
RD=/home/aswin/phase2/runs/$RUN
OUT=/home/aswin/phase2/fresh_inputs/results
mkdir -p "$ROOT"
[[ ! -e "$ROOT/$RUN" ]] || { echo "$ROOT/$RUN exists" >&2; exit 1; }
LIST=$("$DB" "cd ~/phase2/attack_check_fresh && ~/sims/simulated-control-evals-sept/.venv/bin/python list_fresh_runs.py $RD $STAGE --git $GIT --out-dir $OUT | tail -1 | sed 's/^-> //'")
echo "run list: $LIST"
"$DB" rsync -a "box:$RD/" "$ROOT/$RUN/"
"$DB" rsync -a "box:$LIST" "$ROOT/"
mkdir -p "$ROOT/$RUN.logs"
for s in launcher.log policy_spend_watch.log docker_events.jsonl resources.log; do
  "$DB" rsync -a "box:$RD.$s" "$ROOT/$RUN.logs/" 2>/dev/null || echo "no $RUN.$s on the devbox"
done
"$DB" "cd $RD && find . -type f -print0 | sort -z | xargs -0 sha256sum" > "$ROOT/$RUN.devbox_sha256.txt"
(cd "$ROOT/$RUN" && find . -type f -print0 | sort -z | xargs -0 shasum -a 256) > "$ROOT/$RUN.local_sha256.txt"
diff <(awk '{print $1, $2}' "$ROOT/$RUN.devbox_sha256.txt" | LC_ALL=C sort -k2) <(awk '{print $1, $2}' "$ROOT/$RUN.local_sha256.txt" | LC_ALL=C sort -k2) \
  || { echo "the local copy differs from the devbox's" >&2; exit 1; }
echo "copy verified: $(wc -l < "$ROOT/$RUN.local_sha256.txt") files"
EXTRA=(--extra "$ROOT/$(basename "$LIST")=.run_list.json")
for f in "$ROOT/$RUN.logs"/*; do EXTRA+=(--extra "$f=.$(basename "$f" | sed "s/^$RUN\.//")"); done
AWS_PROFILE=redwood-sandbox uv run --no-project --python 3.12 --with boto3 python "$HERE/../2026-10-12_q3_llama70b_price/archive_run.py" \
  --root "$ROOT" --run "$RUN" "${EXTRA[@]}" --note "fresh-task honest runs (experiments/2026-10-13_q2_fresh_honest), stage $STAGE, git $GIT" \
  > "$ROOT/$RUN.archive.log" 2>&1 || { tail -5 "$ROOT/$RUN.archive.log" >&2; exit 1; }
tail -2 "$ROOT/$RUN.archive.log"
M="uv run --no-project --python 3.12 --with modal==1.6.1 modal"
$M volume create "$WORK" --version 2 >/dev/null 2>&1 || true
$M volume put "$WORK" "$ROOT/$RUN" "/$RUN" >/dev/null
$M volume put "$WORK" "$ROOT/$(basename "$LIST")" "/$(basename "$LIST")" >/dev/null
echo "on $WORK: /$RUN and /$(basename "$LIST")"
