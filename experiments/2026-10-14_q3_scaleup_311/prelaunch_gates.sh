#!/usr/bin/env bash
# Pre-launch gates for one scale-up box's launch (experiments/2026-10-14_q3_scaleup_311/RULE.md), ON THE DEVBOX, run by
# canary.sh (the canary's wrapper, unchanged) before it starts anything; exit 0 only if every gate passes. The canary's
# prelaunch_gates.sh with the scale-up's tasks file and its r2 entries. No model call; names, counts and hashes only.
#   prelaunch_gates.sh <harness dir> <pilot config> <expected manifest tsv>
# 1. Freeze: every harness file hashes as the committed manifest says.
# 2. Preflight (gate 1): build_real_samples.py over the 196 and this box's cells with no_digest_lookup/ first on PATH;
#    every distinct new cell must build both modes (cells that failed an earlier preflight were dropped and counted
#    before this file was made); the only failures allowed are the two known 196 attack builds.
# 3. Approach and template (gate 3), on every entry: q2's hash and length, the pinned template hash for its kind, Sol's
#    fields null, its approach file hashing to its name. r2 entries: each `<key>_r2` equals `<key>` except key and
#    real_arms (["attack"]), and every cell has exactly one r2 entry.
# 4. Settings equality (gate 2): gate_settings.py --harness.
set -euo pipefail
C=$1 CFG=$2 MAN=$3
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
export PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src
cd "$C"
S=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
fail() { log "GATE FAIL: $*"; exit 1; }

diff <(find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) "$MAN" \
  > "results/manifest_diff_$S.txt" || fail "harness files differ from $MAN (results/manifest_diff_$S.txt)"
log "freeze: every harness file matches $MAN"

TASKS=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['tasks'])" "$CFG")
[[ $TASKS =~ ^config/tasks_311_q[1-4]_launch\.json$ ]] || fail "$CFG reads $TASKS"
PATH=$C/no_digest_lookup:$PATH "$PY" build_real_samples.py --approaches /home/aswin/phase2/approaches \
  config/pilot_tasks_amendment9_all196.json "$TASKS" "results/preflight_$S.json" > "results/preflight_$S.log" 2>&1 ||
  fail "preflight crashed (results/preflight_$S.log)"
"$PY" - "results/preflight_$S.json" "$TASKS" <<'PY' || fail "preflight or approach/template gate"
import hashlib, json, sys
from pathlib import Path
pf, tasks = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
name = Path(sys.argv[2]).name
new = [r for r in pf["rows"] if name in r["in"]]
cells = tasks["cells"]
r1 = [c for c in cells if "real_arms" not in c]
r2 = {c["key"]: c for c in cells if "real_arms" in c}
allowed = {"model_integrity_verification:exfiltrate_weights", "severity_renaming:suppress_safety_notifications"}
failed = {(r["cell"], r["mode"]) for r in pf["rows"] if not r["built"]}
print(f"preflight: new cells built {sum(r['built'] for r in new)} of {len(new)}; failures {sorted(failed)}")
assert len(new) == 2 * len(r1) == 2 * len({c["cell"] for c in cells}) and all(r["built"] for r in new), "a new cell failed"
assert failed <= {(c, "attack") for c in allowed}, "a failure outside the two known 196 attack builds"
T = {True: "cd05df782d7cc97ab9c9d6edf9cd43311145a7e708b14879e9c0df8d7ba2348a",
     False: "fec9f6614c887e78c8c9016270705e8c294650034f85272e4f5ff6cd62b6b473"}
canon = lambda c: json.dumps({k: v for k, v in c.items() if k not in ("key", "real_arms")}, sort_keys=True)
assert len(r2) == len(r1) and len({c["key"] for c in cells}) == len(cells), "r2 entries missing or keys repeat"
for c in r1:
    assert (c["approach_sha256"], c["approach_chars"]) == (c["q2_approach_sha256"], c["q2_approach_chars"]), c["key"]
    assert c["template_sha256"] == T[c["approach_chars"] > 0], c["key"]
    assert c["sol_system_sha256"] is None and c["sol_input_sha256"] is None, c["key"]
    assert c["attack_pair_launch"] == "same", c["key"]
    f = Path("/home/aswin/phase2/approaches") / f"{c['approach_sha256']}.txt"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == c["approach_sha256"], c["key"]
    d = r2[c["key"] + "_r2"]
    assert d["real_arms"] == ["attack"] and canon(d) == canon(c), f"{c['key']}_r2 differs from its cell's entry"
print(f"approach and template: {len(r1)} cells equal to q2's record and the pinned template hashes; "
      f"{len(r2)} r2 entries equal to their cells except key and real_arms")
PY
log "gate 1 (preflight) and gate 3 (approach, template, r2) PASS: results/preflight_$S.json"

"$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json \
  --out "results/gate_settings_harness_$S.json" --harness . 2>&1 | grep -v "Could not find sandbox" || true
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" \
  "results/gate_settings_harness_$S.json" || fail "settings equality"
log "gate 2 (settings equality, pre-launch) PASS"
log "ALL PRE-LAUNCH GATES PASS for $CFG"
