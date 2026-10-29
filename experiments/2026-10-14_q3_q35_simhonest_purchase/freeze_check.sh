#!/usr/bin/env bash
# Stage the reviewed harness on the devbox and check it file by file (RULE.md, Harness). From the probe-shift repo root:
#   DEVBOX=<user>@<id> freeze_check.sh <scratch dir>
# 1. git archive bbf0729's experiments/2026-10-07_phase2_attack_check into <scratch>/h.
# 2. rsync it WITHOUT results/ (89 tracked result files: not code, not copied) to ~/phase2/attack_check_simh (must not
#    exist yet), mkdir an empty results/, write LAUNCH_COMMIT = bbf0729's full sha.
# 3. sha256 every copied file on both sides (all but results/ and LAUNCH_COMMIT; the remote side under bash pipefail);
#    any difference, missing or extra file exits 1.
set -euo pipefail
X=$1; H=experiments/2026-10-07_phase2_attack_check; C=bbf0729
FULL=$(git rev-parse $C)
rm -rf "$X/h"; mkdir -p "$X/h"
git archive "$C" "$H" | tar -x -C "$X/h"
L="$X/h/$H"
"$L/devbox.sh" 'test ! -e ~/phase2/attack_check_simh' || { echo "attack_check_simh already exists on the devbox" >&2; exit 1; }
"$L/devbox.sh" rsync -az --exclude /results/ "$L/" box:phase2/attack_check_simh/
"$L/devbox.sh" "mkdir -p ~/phase2/attack_check_simh/results && echo $FULL > ~/phase2/attack_check_simh/LAUNCH_COMMIT"
(cd "$L" && find . -type f -not -path './results/*' -print0 | sort -z | xargs -0 shasum -a 256) > "$X/local.sha"
"$L/devbox.sh" "bash -c 'set -euo pipefail; cd ~/phase2/attack_check_simh; test -z \"\$(ls -A results)\"; find . -type f -not -path ./results/\\* -not -name LAUNCH_COMMIT -print0 | sort -z | xargs -0 sha256sum'" > "$X/remote.sha"
# compared in one collation (LC_ALL=C, by path): macOS and GNU `sort -z` order the same names differently
if diff <(sed 's/  */ /' "$X/local.sha" | LC_ALL=C sort -k2) <(sed 's/  */ /' "$X/remote.sha" | LC_ALL=C sort -k2) > "$X/freeze.diff"; then
  echo "freeze check passed: $(wc -l < "$X/local.sha") files equal bbf0729 ($FULL)"
else
  echo "freeze check FAILED: see $X/freeze.diff" >&2; exit 1
fi
