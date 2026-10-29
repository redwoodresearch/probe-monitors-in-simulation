#!/usr/bin/env bash
# ON THE DEVBOX (box3: L1, box4: L2), after the il_stage staging: build this launch's S_paper493 SIM-ATTACK task file in
# ~/phase2/sp493sim/, with the harness in ~/phase2/attack_check_311 untouched. DERIVED from the B fork's prep_sp493.sh @ f71377583 (= d2bb71640 except its box-C key)
# (aswin/q3-sp493; the +311's prep_tasks_311.sh rule): record_fleet_approaches.py (each cell's approach by the fleet rule,
# stored opaque in ~/phase2/approaches on this box; checked against q2's hash), add_template_hashes.py, the no-model-call
# preflight (build_real_samples.py with no_digest_lookup first on PATH; cells that fail to build are DROPPED and COUNTED).
# Changed: no add_r2.py / add_sp493_arms.py (sims only: one r1 entry per cell, no per-entry real_arms); the output keeps the
# input's top-level fields. Names, counts and hashes only: no approach text is printed.
#   bash ~/phase2/il_stage/prep_sp493_sim.sh <L1|L2>
set -euo pipefail
X=$1; [[ $X == L1 || $X == L2 ]]
S=/home/aswin/phase2/sp493sim C=/home/aswin/phase2/attack_check_311 ST=/home/aswin/phase2/il_stage
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
export PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src CONTROL_TOWER_SETTINGS_DIR=/home/aswin/phase2/settings
T=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
mkdir -p $S/results
IN=$S/tasks_sp493_sim_${X}_in.json REC=$S/tasks_sp493_sim_${X}_recorded.json FIN=$S/tasks_sp493_sim_${X}_final.json OUT=$S/tasks_sp493_sim_${X}.json
for f in $IN $REC $FIN $OUT; do [[ ! -e $f ]] || { echo "$f exists" >&2; exit 1; }; done
cp "$ST/tasks_sp493_sim_${X}_in.json" "$IN"
for f in record_fleet_approaches.py add_template_hashes.py build_real_samples.py no_digest_lookup; do
  [[ -e $C/$f ]] || { echo "$C/$f missing" >&2; exit 1; }; done
cd "$C"
"$PY" record_fleet_approaches.py --tasks "$IN" --out "$REC" --approaches /home/aswin/phase2/approaches > "$S/results/prep_record_${X}_$T.log" 2>&1
"$PY" add_template_hashes.py "$REC" "$FIN" > "$S/results/prep_template_${X}_$T.log" 2>&1
PATH=$C/no_digest_lookup:$PATH "$PY" build_real_samples.py --approaches /home/aswin/phase2/approaches "$FIN" \
  "$S/results/prep_preflight_${X}_$T.json" > "$S/results/prep_preflight_${X}_$T.log" 2>&1
"$PY" - "$FIN" "$S/results/prep_preflight_${X}_$T.json" "$OUT" "$S/results/prep_sp493_sim_${X}_$T.json" <<'PY'
import hashlib, json, sys
fin, pf, out, rep = sys.argv[1:]
data, rows = json.load(open(fin)), json.load(open(pf))["rows"]
bad = sorted({r["cell"] for r in rows if not r["built"]})
errors = {r["cell"] + ":" + r["mode"]: r.get("error", "")[:160] for r in rows if not r["built"]}
keep = [c for c in data["cells"] if c["cell"] not in bad]
for c in data["cells"]:
    if (c["approach_sha256"], int(c["approach_chars"])) != (c["q2_approach_sha256"], int(c["q2_approach_chars"])):
        sys.exit(f"{c['key']}: the recorded approach differs from q2's hash")
    if "real_arms" in c:
        sys.exit(f"{c['key']}: a per-entry real_arms in a sims-only launch")
data["cells"], data["dropped_at_preflight"] = keep, [c["key"] for c in json.load(open(fin))["cells"] if c["cell"] in bad]
if not keep:
    sys.exit("every cell failed preflight: refusing an empty plan")
open(out, "w").write(json.dumps(data, indent=1))
r = {"cells_in": len(keep) + len(bad), "built_both_modes": len(keep), "dropped": data["dropped_at_preflight"],
     "errors": errors, "empty_approach_kept": sum(int(c["approach_chars"]) == 0 for c in keep),
     "out_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
open(rep, "w").write(json.dumps(r, indent=1))
print("PREP " + json.dumps({k: v for k, v in r.items() if k != "errors"}))
PY
sha256sum "$IN" "$REC" "$FIN" "$OUT" | sed "s#$S/##"
