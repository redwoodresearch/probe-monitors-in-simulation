#!/bin/zsh -f
# Test of block_vars.py (static; it runs no block). Run from the repo root:
#   zsh -f experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_block_vars.sh
# 1. Positive control on the real blocks at 679f21f93 (a03f5d93's REQUIRED-A): PATHGATE3P/3B clobber exactly NA and NC; H and STEP_SPEC
#    have no block providing them (they were a pasted glue line).
# 2. Synthetic: assignments inside $( ) or ( ), a command-prefix TZ=, and a single-quoted '$Y' do not count; read and case-body
#    assignments do. VARS v -> SUB -> USE is OK; VARS v -> CLOB -> USE flags X and Y.
T=experiments/2026-10-15_fpr_sweep_507_prep/tools_routec
E=experiments/2026-10-15_fpr_sweep_507_prep
O=679f21f93
PP=$O:$E/routec_packages_abr37/PKG_routec_PROV1016_s3s4_ABR37.md
N=$O:$E/NOTE_routec_s346_2026-10-17.md
P6=$O:$E/routec_packages_abr37/PKG_routec_s6_ABR37.md
F=$T/fixtures_block_vars/blocks.md
ok=1
check() {
  out=$(uv run --offline --no-project --python 3.12 python -I $T/block_vars.py "$1" | grep '^  RESULT ')
  if [ "$out" = "$2" ]; then echo "PASS ${1%%=*}"; else echo "FAIL ${1%%=*}: $out"; ok=0; fi
}
check "old-prov=$PP#VARS p3,$PP#PATHGATE3P,$PP#S3DCS,$PP#S3LINEP" \
  '  RESULT old-prov: UNPROVIDED PATHGATE3P: H; UNPROVIDED PATHGATE3P: STEP_SPEC; COLLISION S3DCS: NC last set by PATHGATE3P (all: VARS p3, PATHGATE3P); COLLISION S3LINEP: NC last set by PATHGATE3P (all: VARS p3, PATHGATE3P); COLLISION S3LINEP: NA last set by PATHGATE3P (all: VARS p3, PATHGATE3P)'
check "old-s6=$N#VARS s3,$N#VARS s6,$P6#PATHGATE3B,$N#S6LINE,$N#D6PC,$N#D6LINE" \
  '  RESULT old-s6: UNPROVIDED PATHGATE3B: H; UNPROVIDED PATHGATE3B: STEP_SPEC; COLLISION S6LINE: NC last set by PATHGATE3B (all: VARS s3, PATHGATE3B)'
check "fix-ok=$F#VARS v,$F#SUB,$F#USE" '  RESULT fix-ok: OK'
check "fix-clob=$F#VARS v,$F#CLOB,$F#USE" \
  '  RESULT fix-clob: COLLISION USE: X last set by CLOB (all: VARS v, CLOB); COLLISION USE: Y last set by CLOB (all: VARS v, CLOB)'
S=$(uv run --offline --no-project --python 3.12 python -I $T/block_vars.py "sub=$F#VARS v,$F#SUB" | grep '^  SUB: ')
if [ "$S" = '  SUB: assigns 3, free 1 [P], shadows []' ]; then echo "PASS sub-scopes"; else echo "FAIL sub-scopes: $S"; ok=0; fi
if [ "$ok" = 1 ]; then echo "ALL PASS"; else echo "FAILED"; exit 1; fi
