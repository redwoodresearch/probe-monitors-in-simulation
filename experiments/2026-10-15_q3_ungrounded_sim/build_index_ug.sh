#!/usr/bin/env bash
# build_index_ug.sh (RULE.md): ON q3-xbox1, after relay_refs_ug.sh, merge the 149 cells' pinned references into one
# --references-from folder with the interleaved launch's reviewed make_reference_index.py (byte-identical copy,
# sha256 in SHA256SUMS): the S_old index (34 old cells; files already on xbox1) + the 4 relayed Q_k mains (115 cells).
# --check-files re-hashes every reference file against its record. Refuses an existing output.
#   bash ~/phase2/ug_stage/build_index_ug.sh
set -euo pipefail
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
ST=/home/aswin/phase2/ug_stage; R=/home/aswin/phase2/refs_ug; OUT=/home/aswin/phase2/il_refindex_ug
TASKS=/home/aswin/phase2/attack_check_ug/config/tasks_ug_pilot.json
[[ ! -e $OUT && ! -L $OUT ]] || { echo "FAIL: $OUT exists"; exit 1; }
(cd "$ST" && sha256sum -c --quiet SHA256SUMS) || { echo "FAIL: staged files differ from SHA256SUMS"; exit 1; }
# the index SOURCES are pinned: the S_old index as il_old_box.sh pinned it, and each relayed Q_k main launch.json as
# hashed on its own box (2026-10-15 ~00:45 PT, d9's ungrounded fork), so a replaced source can't pass --check-files
sha256sum -c --quiet - <<'SUMS' || { echo "FAIL: an index source differs from its pin"; exit 1; }
79acbfc1ef4523f9a57cc19dc876252159c6c1d62521001f1ac71e2644036940  /home/aswin/phase2/il_refindex_old/launch.json
7e8227cd859502e60ad718fc60215410ece9d4b86eea04040bd7bee0a82c53ac  /home/aswin/phase2/refs_ug/q1_launch.json
fa489bf06d2be9c4a2b5c432ded7fb6065df49763618fb19e96a5f5d01cf03fb  /home/aswin/phase2/refs_ug/q2_launch.json
289ffb1030625f24666b1c9fb86a2ad577cf66a55407b9f18db125f0ed121701  /home/aswin/phase2/refs_ug/q3_launch.json
f12ed4cd86ed516036bcc47857c7a4d90ba95e21595a8c8f484653b9055a9ad6  /home/aswin/phase2/refs_ug/q4_launch.json
SUMS
"$PY" "$ST/make_reference_index.py" --source /home/aswin/phase2/il_refindex_old/launch.json \
  --source "$R/q1_launch.json=$R/q1" --source "$R/q2_launch.json=$R/q2" --source "$R/q3_launch.json=$R/q3" \
  --source "$R/q4_launch.json=$R/q4" --cells "$TASKS" --out "$OUT" --check-files
"$PY" - "$OUT/launch.json" "$TASKS" <<'PY'
import json, sys
refs = json.load(open(sys.argv[1]))["references"]
keys = [c["key"] for c in json.load(open(sys.argv[2]))["cells"]]
assert set(refs) == set(keys) and len(keys) == 149, (len(refs), len(keys), sorted(set(keys) ^ set(refs))[:5])
print(f"index: {len(refs)} references = the 149 cells")
PY
sha256sum "$OUT/launch.json"
