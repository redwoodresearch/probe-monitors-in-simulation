#!/usr/bin/env bash
# Positive and negative controls for fit_env_gate.zsh (no Modal, no git, no repository access). Run from this folder: bash test_fit_env_gate.sh
# A fake `uv` (prints another version) is written into a fresh mktemp dir, which is left in place (no deletions).
set -u
pass=0; fail=0
CLEAN=(env -u PYTHONPATH -u PYTHONHOME -u UV_PROJECT_ENVIRONMENT)
t() {  # name expected_rc expected_first_word cmd...
  local name=$1 erc=$2 ew=$3; shift 3
  out=$("$@" zsh -f fit_env_gate.zsh 2>&1); rc=$?
  if [ "$rc" -eq "$erc" ] && [ "${out%% *}" = "$ew" ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $name: rc $rc out [$out]"; fi
}
FAKE=$(mktemp -d)
printf '#!/bin/sh\necho "uv 0.12.18 (fake)"\n' > "$FAKE/uv"; chmod +x "$FAKE/uv"
t "clean env -> FIT_ENV_OK" 0 FIT_ENV_OK "${CLEAN[@]}"
t "PYTHONPATH set -> STOP" 1 STOP: "${CLEAN[@]}" PYTHONPATH=/elsewhere/src
t "PYTHONPATH set but empty -> STOP" 1 STOP: "${CLEAN[@]}" PYTHONPATH=
t "PYTHONHOME set -> STOP" 1 STOP: "${CLEAN[@]}" PYTHONHOME=/elsewhere
t "UV_PROJECT_ENVIRONMENT set -> STOP" 1 STOP: "${CLEAN[@]}" UV_PROJECT_ENVIRONMENT=/elsewhere/.venv
t "other uv version -> STOP" 1 STOP: "${CLEAN[@]}" PATH="$FAKE:$PATH"
[ "$fail" -eq 0 ] && echo "PASS $pass/$((pass + fail))" || { echo "FAIL $fail/$((pass + fail))"; exit 1; }
