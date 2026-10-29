#!/usr/bin/env bash
# Gated launch prep for the 360 new-main cells (experiments/2026-10-14_q3_new_mains_launch/RULE.md; the new-mains mode's
# RULE, "Launch prep"), ON THE DEVBOX in the isolated tree ~/phase2/nm/attack_check_nm; $0, no model call; keys, counts
# and hashes only (approach texts are written opaque by record_fleet_approaches.py and never printed). In order, each
# must succeed:
#   1. record_fleet_approaches.py: each cell's approach by the fleet rule, which must equal q2's hash and length;
#   2. record_provenance.py: the provenance hashes (recording mode, once) and the seeded fixtures' hashes;
#   3. test_new_mains_mode.py: the 196 build identically under this harness's legacy tasks.py (BASE_TASKS) and the
#      new one, with exactly the two known refusals; the fixtures reproduce; the new cells reproduce their provenance;
#   4. a preflight (build_real_samples.py, no_digest_lookup first on PATH) over every recorded cell: cells that fail
#      to build are DROPPED and COUNTED, each with its error class, in results/prep_nm_<stamp>.json.
#   bash prep_nm.sh
set -euo pipefail
C=$(cd "$(dirname "$0")" && pwd); cd "$C"
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
export PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src:$C CONTROL_TOWER_SETTINGS_DIR=/home/aswin/phase2/nm/settings
PYTEST_PATH=/home/aswin/phase2/nm/pylib  # pytest only, installed there by box_setup_nm.sh (the shared venv is never modified)
A=/home/aswin/phase2/nm/approaches
S=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
IN=config/new_mains_cells_v1.json; REC=config/nm_recorded.json; PROV=config/nm_prov.json; FIX=config/nm_fixtures.json
OK=config/nm_built.json
[[ ! -e $OK ]] || { echo "$OK exists" >&2; exit 1; }
# Steps 1-2 run ONCE: if a complete recording exists (every cell with its provenance, fixtures written), they are skipped.
RECORDED=0
if [[ -e $PROV ]]; then
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); f=json.load(open(sys.argv[2])); sys.exit(0 if all(len(c.get('provenance_system_sha256') or '')==64 for c in d['cells']) and f['fixtures'] else 1)" "$PROV" "$FIX" \
    && RECORDED=1 || { echo "$PROV exists but is incomplete" >&2; exit 1; }
fi
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
if (( RECORDED == 0 )); then
log "1. record_fleet_approaches"
"$PY" record_fleet_approaches.py --tasks "$IN" --out "$REC" --approaches "$A" > "results/prep_record_$S.log" 2>&1
"$PY" - "$REC" <<'PY'
import json, sys
cells = json.load(open(sys.argv[1]))["cells"]
bad = [c["cell"] for c in cells if (c["approach_sha256"], c["approach_chars"]) != (c["q2_approach_sha256"], c["q2_approach_chars"])]
print(f"recorded {len(cells)}; differ from q2's record: {len(bad)}")
sys.exit(1 if bad else 0)
PY
log "2. record_provenance"
"$PY" record_provenance.py "$REC" "$PROV" "$FIX" --approaches "$A" > "results/prep_provenance_$S.log" 2>&1
else
log "1-2. already recorded (complete): $PROV, $FIX"
fi
log "3. test_new_mains_mode"
BASE_TASKS=$C/tasks_legacy_h123.py LEGACY_TASKS=$C/config/pilot_tasks_amendment9_all196.json NEW_MAINS_CELLS=$C/$PROV \
  NEW_MAINS_FIXTURES=$C/$FIX APPROACHES=$A PYTHONPATH=$PYTHONPATH:$PYTEST_PATH "$PY" -m pytest -q -p no:cacheprovider test_new_mains_mode.py \
  > "results/prep_tests_$S.log" 2>&1 || { tail -5 "results/prep_tests_$S.log"; log "TESTS FAILED"; exit 1; }
tail -1 "results/prep_tests_$S.log"
log "4. preflight"
PATH=$C/no_digest_lookup:$PATH "$PY" build_real_samples.py --approaches "$A" "$PROV" "results/prep_preflight_$S.json" \
  > "results/prep_preflight_$S.log" 2>&1
"$PY" - "$PROV" "results/prep_preflight_$S.json" "$OK" "results/prep_nm_$S.json" <<'PY'
import hashlib, json, sys
prov, pf, ok, rep = sys.argv[1:]
data, rows = json.load(open(prov)), json.load(open(pf))["rows"]
bad = sorted({r["cell"] for r in rows if not r["built"]})
errors = {r["cell"] + ":" + r["mode"]: (r.get("error") or "")[:160] for r in rows if not r["built"]}
keep = [c for c in data["cells"] if c["cell"] not in bad]
data["cells"], data["dropped_at_preflight"] = keep, [{"cell": c, "stage": "preflight"} for c in bad]
open(ok, "w").write(json.dumps(data, indent=1))
r = {"cells_in": len(keep) + len(bad), "built_both_modes": len(keep), "dropped": bad, "errors": errors,
     "empty_approach_kept": sum(c["approach_chars"] == 0 for c in keep), "built_sha256": hashlib.sha256(open(ok, "rb").read()).hexdigest()}
open(rep, "w").write(json.dumps(r, indent=1))
print(json.dumps({k: v for k, v in r.items() if k != "errors"}))
PY
sha256sum "$IN" "$REC" "$PROV" "$FIX" "$OK"
log "PREP DONE"
