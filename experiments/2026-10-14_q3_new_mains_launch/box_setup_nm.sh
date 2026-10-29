#!/usr/bin/env bash
# DEPRECATED STEP (10-15 ~16:50 PT, 0d via d9): the on-box approaches copy below (`cp -a ... ~/phase2/approaches/. ~/phase2/nm/approaches/`)
# must not be run again: approach and attack files move ONLY through the reviewed relay_approaches.sh
# (origin/aswin/q3-simhonest-311 experiments/2026-10-15_q3_simhonest_311/), with its committed record. Kept for the record only.
# Laptop side: prepare the ISOLATED new-mains tree on one devbox (RULE.md, Isolation): ~/phase2/nm/{attack_check_nm,
# settings,approaches,runs,checks,pylib}. Copies ON THE BOX (read-only of the box's own ~/phase2/settings, minus the
# .compose-nobuild-* files real runs write, and of ~/phase2/approaches, opaque); pytest into nm/pylib (the shared venv is
# never modified); then the frozen harness from the laptop. Touches nothing outside ~/phase2/nm.
#   DEVBOX=aswin@<id> HN=<local nm harness dir> E=<dir with devbox.sh> box_setup_nm.sh
set -euo pipefail
: "${DEVBOX:?}" "${HN:?}" "${E:?}"
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [$DEVBOX] $*"; }
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$E/devbox.sh" "$2"; }
box 600 'set -e; mkdir -p ~/phase2/nm/runs ~/phase2/nm/checks ~/phase2/nm/approaches ~/phase2/nm/pylib ~/phase2/nm/settings
  rsync -a --exclude ".compose-nobuild-*" ~/phase2/settings/ ~/phase2/nm/settings/
  cp -a --update=none ~/phase2/approaches/. ~/phase2/nm/approaches/
  cd /tmp && uv pip install -q --python /home/aswin/sims/simulated-control-evals-sept/.venv/bin/python --target ~/phase2/nm/pylib pytest==8.3.3 >/tmp/nm_pytest.log 2>&1 || { tail -3 /tmp/nm_pytest.log; exit 1; }
  echo "settings $(du -sh ~/phase2/nm/settings | cut -f1); approaches $(ls ~/phase2/nm/approaches | wc -l); pytest $(ls ~/phase2/nm/pylib | grep -c pytest)"'
DEVBOX=$DEVBOX perl -e 'alarm 900; exec @ARGV' bash "$E/devbox.sh" rsync -a --checksum --exclude __pycache__ --exclude 'results/*' \
  "$HN/" box:phase2/nm/attack_check_nm/
box 60 'mkdir -p ~/phase2/nm/attack_check_nm/results && cd ~/phase2/nm/attack_check_nm && echo "harness files $(find . -type f ! -path "./results/*" | wc -l)"'
log "nm tree ready"
