#!/usr/bin/env bash
# build_ug.sh (RULE.md): ON q3-xbox1, build the UNGROUNDED arm's two COPIES. The frozen September tree and the frozen
# harness are only read, never written (both are re-verified at the end). Fails closed; refuses if a target exists.
#   bash ~/phase2/ug_stage/build_ug.sh
# Needs in ~/phase2/ug_stage/: ug_edits.py, the committed patch, pilot_ug.yaml, tasks_ug_pilot.json, SHA256SUMS.
set -euo pipefail
FROZEN_TREE=/home/aswin/phase2/sims-sept-strict-off; FROZEN_TREE_SHA=1ea3772209663ef8a848effaf6677d23d5bab06c248a37c1fb61f0957efd7b8c
TREE=/home/aswin/phase2/sims-sept-ungrounded
SRC=/home/aswin/phase2/attack_check_311; DST=/home/aswin/phase2/attack_check_ug
RUN_PILOT_SHA=ea8d7253ba26fbd8be0f60b050fa17caf992b086317c6b00f7345537c4491d90
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
ST=/home/aswin/phase2/ug_stage; TMP=$(mktemp -d)
export PYTHONDONTWRITEBYTECODE=1  # imports below must not write __pycache__ into the frozen tree or harness
log() { echo "[ug build] $*"; }
fail() { echo "[ug build] FAIL: $*" >&2; exit 1; }
tree_sha() { (cd "$SRC" && "$PY" -c 'import sys; from pathlib import Path; sys.path.insert(0, "."); from run_attack_check import src_sha256; print(src_sha256(Path(sys.argv[1])))' "$1"); }
harness_list() {  # the frozen harness's hash list, as il311_box.sh/il_old_box.sh compute it (each producer's status checked)
  (cd "$1" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
    ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il_*' ! -path './config/tasks_il_*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum)
}
for p in "$TREE" "$DST"; do [[ ! -e $p && ! -L $p ]] || fail "$p exists"; done
(cd "$ST" && sha256sum -c --quiet SHA256SUMS) || fail "staged files differ from SHA256SUMS"
# 0. the frozen inputs are what we think they are
[[ $(tree_sha "$FROZEN_TREE") == "$FROZEN_TREE_SHA" ]] || fail "$FROZEN_TREE is not the pinned september tree"
harness_list "$SRC" > "$TMP/frozen_now.tsv" || fail "hashing $SRC"
grep -v '  ./config/harness_manifest' "$SRC/config/harness_manifest_fu.tsv" > "$TMP/frozen_manifest.tsv" || fail "reading the manifest"
cmp -s "$TMP/frozen_now.tsv" "$TMP/frozen_manifest.tsv" || fail "$SRC differs from its frozen manifest"
log "frozen inputs verified: tree $FROZEN_TREE_SHA; harness $(wc -l < "$TMP/frozen_now.tsv") files = harness_manifest_fu.tsv"
# 1. the ungrounded tree: a copy plus the one patch (two exact edits in src/runner/settings.py)
cp -a "$FROZEN_TREE" "$TREE"
"$PY" "$ST/ug_edits.py" tree "$TREE/src/runner/settings.py"
rc=0; diff -u --label a/src/runner/settings.py --label b/src/runner/settings.py \
  "$FROZEN_TREE/src/runner/settings.py" "$TREE/src/runner/settings.py" > "$TMP/ungrounded_no_honest_reference.patch" || rc=$?
[[ $rc == 1 ]] || fail "diff of settings.py: rc $rc (1 = differs is the only success)"
# the COMMITTED patch alone reproduces the tree from a fresh copy (as setup_devbox.sh's extra_patches would); the text of
# a regenerated diff depends on the diff implementation, so the check is on the RESULT, not the patch text
cp -a "$FROZEN_TREE/src" "$TMP/fresh_src"; (cd "$TMP" && mkdir fresh && mv fresh_src fresh/src && patch -s -p1 -d fresh < "$ST/ungrounded_no_honest_reference.patch") \
  || fail "the committed patch does not apply to a fresh copy"
cmp -s "$TMP/fresh/src/runner/settings.py" "$TREE/src/runner/settings.py" || fail "the committed patch does not reproduce the edited settings.py"
strays=$(find "$TMP/fresh/src" \( -name '*.orig' -o -name '*.rej' \) -print) || fail "find failed"
[[ -z $strays ]] || fail "the patch left .orig/.rej files (fuzz): $strays"
UG_SHA=$(tree_sha "$TREE"); [[ ${#UG_SHA} == 64 && $UG_SHA != "$FROZEN_TREE_SHA" ]] || fail "bad tree sha $UG_SHA"
rc=0; diff -rq "$FROZEN_TREE" "$TREE" > "$TMP/tree_diff.txt" || rc=$?
[[ $rc == 1 ]] || fail "diff -rq of the trees: rc $rc (1 = differs is the only success)"
[[ $(cat "$TMP/tree_diff.txt") == "Files $FROZEN_TREE/src/runner/settings.py and $TREE/src/runner/settings.py differ" ]] \
  || fail "the trees differ in more than src/runner/settings.py: $(head -c 400 "$TMP/tree_diff.txt")"
log "tree $TREE: src_sha256 $UG_SHA (frozen $FROZEN_TREE_SHA + settings.py only)"
# 2. unit check of the patched settings (no sample is built, so nothing reads or writes the settings folder)
for t in "$FROZEN_TREE" "$TREE"; do
  (cd "$t" && PYTHONPATH="$t/src" "$PY" - "$t" "$SRC/config/sim.yaml" <<'PY') || fail "settings unit check on $t"
import os, sys, types, yaml
tree, simyaml = sys.argv[1], sys.argv[2]
from runner.settings import SimulatedTaskSettings
base = {k: str(v) for k, v in yaml.safe_load(open(simyaml))["env"].items()} | {"BA_MODE": "attack", "BA_REAL_LOG": "/tmp/ref.eval", "BA_HONEST_LOG": "/tmp/ref.eval"}
arena = types.SimpleNamespace(default_max_steps=150)
def settings(extra):
    for k in [k for k in os.environ if k.startswith(("BA_", "PT_"))]:
        del os.environ[k]
    os.environ.update(base | extra)
    return SimulatedTaskSettings.from_environment(arena)
patched = tree.endswith("sims-sept-ungrounded")
off = settings({})
assert off.honest_logs == ("/tmp/ref.eval",) and "original_trajectory" in off.affordance_tokens(), (off.honest_logs, off.affordance_tokens())
on = settings({"BA_NO_HONEST_REFERENCE": "1"})
if patched:
    assert on.honest_logs == () and "original_trajectory" not in on.affordance_tokens(), (on.honest_logs, on.affordance_tokens())
    assert on.real_logs == ("/tmp/ref.eval",)
    try:
        settings({"BA_NO_HONEST_REFERENCE": "yes"}); raise SystemExit("a bad flag value was accepted")
    except AssertionError:
        pass
else:  # the frozen tree ignores the flag: the patch is what makes the arm ungrounded
    assert on.honest_logs == ("/tmp/ref.eval",), on.honest_logs
print(f"{tree}: flag unset -> {off.affordance_tokens()}; flag 1 -> {on.affordance_tokens()}")
PY
done
# 3. the harness copy: frozen files byte-identical except run.yaml, check_sim_samples.py and tasks.py, plus 3 new config files
rsync -a --exclude '/results/' --exclude '__pycache__/' --exclude '/attempt/results/' --exclude '/config/pilot_il_*' \
  --exclude '/config/tasks_il_*' "$SRC/" "$DST/" || fail "rsync"
mkdir "$DST/results"
[[ $(sha256sum < "$DST/run_pilot.py" | cut -c1-64) == "$RUN_PILOT_SHA" ]] || fail "run_pilot.py is not ea8d7253"
"$PY" "$ST/ug_edits.py" runyaml "$DST/config/run.yaml" "$UG_SHA"
"$PY" "$ST/ug_edits.py" preflight "$DST/check_sim_samples.py"
"$PY" "$ST/ug_edits.py" tasks "$DST/tasks.py"
cp "$ST/ungrounded_no_honest_reference.patch" "$ST/pilot_ug.yaml" "$ST/tasks_ug_pilot.json" "$DST/config/"  # the committed patch
(cd "$DST" && "$PY" -c 'import yaml; s = yaml.safe_load(open("config/run.yaml"))["simulators"]; u = s["september_ungrounded"]; assert u["env"] == {"BA_NO_HONEST_REFERENCE": "1"} and u["ungrounded"] is True and s["september"]["src_sha256"] == "'"$FROZEN_TREE_SHA"'", u; print("run.yaml parses; september unchanged; september_ungrounded:", u["src_sha256"])') \
  || fail "run.yaml"
harness_list "$DST" > "$DST/config/harness_manifest_ug.tsv" || fail "hashing $DST"
"$PY" - "$TMP/frozen_manifest.tsv" "$DST/config/harness_manifest_ug.tsv" <<'PY' || fail "the copy differs from the frozen harness in more than the intended files"
import sys
read = lambda p: {line[66:].rstrip("\n"): line[:64] for line in open(p)}
frozen, ug = read(sys.argv[1]), read(sys.argv[2])
changed = sorted(k for k in frozen if k in ug and frozen[k] != ug[k])
added, removed = sorted(set(ug) - set(frozen)), sorted(set(frozen) - set(ug))
assert changed == ["./check_sim_samples.py", "./config/run.yaml", "./tasks.py"], changed
assert added == ["./config/pilot_ug.yaml", "./config/tasks_ug_pilot.json", "./config/ungrounded_no_honest_reference.patch"], added
assert not removed, removed
print(f"copy manifest: {len(ug)} files; changed {changed}; added {added}")
PY
# 4. the frozen inputs are untouched
[[ $(tree_sha "$FROZEN_TREE") == "$FROZEN_TREE_SHA" ]] || fail "the frozen tree changed"
harness_list "$SRC" > "$TMP/frozen_after.tsv" && cmp -s "$TMP/frozen_after.tsv" "$TMP/frozen_manifest.tsv" || fail "the frozen harness changed"
sha256sum "$DST/config/harness_manifest_ug.tsv" "$DST/config/ungrounded_no_honest_reference.patch" "$DST/config/run.yaml" "$DST/check_sim_samples.py" "$DST/tasks.py"
log "BUILT: tree $TREE ($UG_SHA), harness $DST; frozen tree and harness unchanged"
