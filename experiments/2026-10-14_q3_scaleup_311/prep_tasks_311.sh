#!/usr/bin/env bash
# Build one box's launch tasks file ON THE DEVBOX (experiments/2026-10-14_q3_scaleup_311/RULE.md, Cells), as the canary
# built its own: record_fleet_approaches.py (each new cell's approach by the fleet rule, stored opaque, checked against
# q2's hash), add_template_hashes.py, a preflight over the cells (no_digest_lookup first on PATH; cells that fail to build
# are DROPPED and COUNTED in results/prep_311_box<N>_<stamp>.json), then add_r2.py. Names, counts and hashes only.
#   bash prep_tasks_311.sh <box 1|2>
set -euo pipefail
B=$1; C=$(cd "$(dirname "$0")" && pwd); cd "$C"
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
export PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src CONTROL_TOWER_SETTINGS_DIR=/home/aswin/phase2/settings
S=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
IN=config/tasks_311_box$B.json; REC=config/tasks_311_box${B}_recorded.json; FIN=config/tasks_311_box${B}_final.json
OK=config/tasks_311_box${B}_built.json; OUT=config/tasks_311_box${B}_launch.json
for f in $REC $FIN $OK $OUT; do [[ ! -e $f ]] || { echo "$f exists" >&2; exit 1; }; done
"$PY" record_fleet_approaches.py --tasks "$IN" --out "$REC" --approaches /home/aswin/phase2/approaches > "results/prep_record_box${B}_$S.log" 2>&1
"$PY" add_template_hashes.py "$REC" "$FIN" > "results/prep_template_box${B}_$S.log" 2>&1
PATH=$C/no_digest_lookup:$PATH "$PY" build_real_samples.py --approaches /home/aswin/phase2/approaches "$FIN" \
  "results/prep_preflight_box${B}_$S.json" > "results/prep_preflight_box${B}_$S.log" 2>&1
"$PY" - "$FIN" "results/prep_preflight_box${B}_$S.json" "$OK" "results/prep_311_box${B}_$S.json" <<'PY'
import hashlib, json, sys
fin, pf, ok, rep = sys.argv[1:]
data, rows = json.load(open(fin)), json.load(open(pf))["rows"]
bad = sorted({r["cell"] for r in rows if not r["built"]})
errors = {r["cell"] + ":" + r["mode"]: r.get("error", "")[:160] for r in rows if not r["built"]}
keep = [c for c in data["cells"] if c["cell"] not in bad]
for c in data["cells"]:
    assert (c["approach_sha256"], c["approach_chars"]) == (c["q2_approach_sha256"], c["q2_approach_chars"]), c["key"]
data["cells"], data["dropped_at_preflight"] = keep, [c["key"] for c in json.load(open(fin))["cells"] if c["cell"] in bad]
open(ok, "w").write(json.dumps(data, indent=1))
r = {"cells_in": len(keep) + len(bad), "built_both_modes": len(keep), "dropped": data["dropped_at_preflight"],
     "errors": errors, "empty_approach_kept": sum(c["approach_chars"] == 0 for c in keep),
     "built_sha256": hashlib.sha256(open(ok, "rb").read()).hexdigest()}
open(rep, "w").write(json.dumps(r, indent=1))
print(json.dumps({k: v for k, v in r.items() if k != "errors"}))
PY
"$PY" add_r2.py "$OK" "$OUT"
sha256sum "$IN" "$REC" "$FIN" "$OK" "$OUT"
