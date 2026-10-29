#!/usr/bin/env bash
# Copy the September approach texts the launcher restores, as OPAQUE files, to the q3-realhonest devbox.
# run_pilot.py (@ c42666d7, line 250) calls approaches.materialize() for every launch, honest-only included, so the
# folder must exist; honest runs never use the texts. This script never opens, prints, greps or diffs their contents:
# it takes the sha256s the tasks file references, checks each local <sha256>.txt hashes to its own name, copies only
# those files, re-hashes them on the devbox, and logs names and hashes only.
#   DEVBOX=aswin@<instance> bash copy_approaches.sh <local approaches dir> <tasks json> <log out>
set -euo pipefail
SRC=$1 TASKS=$2 LOG=$3
# The staging folder sits beside the log: refuse a log inside any git worktree, so no text can be staged by accident.
! git -C "$(dirname "$LOG")" rev-parse --git-dir >/dev/null 2>&1 || { echo "log folder is inside a git worktree" >&2; exit 1; }
H="$(cd "$(dirname "$0")" && pwd)/../2026-10-07_phase2_attack_check"
WANT=(); while read -r h; do WANT+=("$h"); done < <(jq -r '.cells[].approach_sha256' "$TASKS" | sort -u)
STAGE=$(mktemp -d "$(dirname "$LOG")/rh_appr.XXXXXX"); trap 'rm -rf "$STAGE"' EXIT
: > "$LOG"
for h in "${WANT[@]}"; do
  f="$SRC/$h.txt"; [[ -f $f ]] || { echo "missing $h" >&2; exit 1; }
  got=$(shasum -a 256 "$f" | cut -d' ' -f1); [[ $got == "$h" ]] || { echo "hash mismatch $h" >&2; exit 1; }
  cp "$f" "$STAGE/"; printf '%s.txt\t%s\n' "$h" "$got" >> "$LOG"
done
"$H/devbox.sh" 'mkdir -p ~/phase2/approaches'
"$H/devbox.sh" rsync -a "$STAGE/" box:phase2/approaches/
remote=$("$H/devbox.sh" 'cd ~/phase2/approaches && sha256sum *.txt' | awk '{print $2"\t"$1}' | sort)
[[ "$remote" == "$(sort "$LOG")" ]] || { echo "devbox hashes differ from the local check" >&2; exit 1; }
echo "copied ${#WANT[@]} files, names and hashes in $LOG"
