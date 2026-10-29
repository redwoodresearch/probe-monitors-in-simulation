#!/bin/zsh -f
# COUNTSCHECK launcher (gate + run). Run it by path from the worktree, at the reviewed head:
#   zsh -f experiments/2026-10-20_routec_counts_check/launch_countscheck.zsh <reviewed 40-hex sha> [gate-only]
# "gate-only" runs every check (no remote call; `app list` is a read) and stops before `modal run`.
# Launch only after the review PASS and refresh3's ARMED line. It never reads or runs NOTE text.

H=$1
MODE=${2:-run}
W=/Users/aswinsurya/probe-shift-routec-countscheck-wt
M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
E=experiments/2026-10-20_routec_counts_check
APPN=probe-shift_aswin-surya_sub-rccount_pilothdr_kill-2026-11-19
L=/Users/aswinsurya/probe-shift-routec-countscheck-logs-2026-10-20

[[ "$H" =~ ^[0-9a-f]{40}$ ]] || { print "STOP: usage: launch_countscheck.zsh <40-hex reviewed sha> [gate-only]"; exit 1; }
[[ "$MODE" == run || "$MODE" == gate-only ]] || { print "STOP: mode must be run or gate-only"; exit 1; }
cd "$W" || { print "STOP: no worktree"; exit 1; }
[[ "$(git rev-parse HEAD)" == "$H" ]] || { print "STOP: HEAD != H"; exit 1; }
[[ "$(git rev-parse "HEAD:$E/launch_countscheck.zsh")" == "$(git hash-object "$E/launch_countscheck.zsh")" ]] \
  || { print "STOP: this launcher differs from its committed blob"; exit 1; }
GS=$(git status --porcelain --untracked-files=all -- "$E" ":(exclude)$E/results") || { print "STOP: git status failed"; exit 1; }
[[ -z "$GS" ]] || { print "STOP: uncommitted files in the experiment folder"; exit 1; }
git ls-remote origin refs/heads/aswin/routec-counts-check | grep -q "^$H" || { print "STOP: H is not the pushed head"; exit 1; }
[[ "$("$M" --version < /dev/null 2>&1 | tail -1)" == "modal client version: 1.6.1" ]] || { print "STOP: modal client is not 1.6.1"; exit 1; }
[[ "$("$M" profile current < /dev/null)" == "redwood-default" ]] || { print "STOP: modal profile is not redwood-default"; exit 1; }
AL=$("$M" app list --json < /dev/null) || { print "STOP: app list failed"; exit 1; }
print -rn -- "$AL" | jq -e --arg n "$APPN" -f "$E/no_live_app.jq" > /dev/null \
  || { print "STOP: a live $APPN app, or the app list schema changed"; exit 1; }
mkdir -p "$L" || { print "STOP: no log dir"; exit 1; }
ST=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
uv run --frozen python "$E/countscheck.py" --shape-only > "$L/shape_$ST.log" 2>&1 || { print "STOP: shape check (see $L/shape_$ST.log)"; exit 1; }
grep '^SHAPE_OK' "$L/shape_$ST.log" || { print "STOP: no SHAPE_OK line"; exit 1; }
print "COUNTSCHECK_GATE_OK $H"
[[ "$MODE" == gate-only ]] && exit 0

LOG="$L/run_$ST.log"
caffeinate -i "$M" run --detach "$W/$E/run_routec_counts_check.py" < /dev/null > "$LOG" 2>&1
RC=$?
print "rc=$RC log=$LOG"
grep -E '^(summary|record|bits) |STOP' "$LOG"
grep -A1 'View run at' "$LOG" | grep -o 'ap-[A-Za-z0-9]*' | head -1
exit $RC
