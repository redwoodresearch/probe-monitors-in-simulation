"""Both attack paths refuse a missing or mismatched approach (synthetic texts only; no real approach is read): phase2's
tasks.py path through approaches.materialize (tasks.py itself is unchanged), and the Astra launcher's attack path
through approach_guard (the snapshot builder, with its main-matching fallback).
"""

import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "2026-10-07_phase2_attack_check"), str(ROOT / "2026-10-06_astra_sim_honest")]
import approach_guard  # noqa: E402
import approaches  # noqa: E402

_sha = approaches.sha256


def test_tasks_path_refuses(tmp_path):
    store = tmp_path / "approaches"
    fields = approaches.store("synthetic approach A", store)
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"cells": [{"cell": "m:s"} | fields]}))
    restored = json.loads(approaches.materialize(task_file, tmp_path / "out.json", store).read_text())
    assert _sha(restored["cells"][0]["approach"]) == fields["approach_sha256"]
    (store / f"{fields['approach_sha256']}.txt").write_text("tampered")  # a text that no longer matches its hash
    with pytest.raises(AssertionError):
        approaches.materialize(task_file, tmp_path / "out2.json", store)
    task_file.write_text(json.dumps({"cells": [{"cell": "m:s", "approach_sha256": "0" * 64}]}))  # no stored text
    with pytest.raises(FileNotFoundError):
        approaches.materialize(task_file, tmp_path / "out3.json", store)
    first, other = SimpleNamespace(strategy="first"), SimpleNamespace(strategy="second")
    assert approaches.fleet_rule([first, other]) == "first" and approaches.fleet_rule([]) == ""


def _reference(path, metadata):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("samples/x_epoch_1.json", json.dumps({"id": "x", "metadata": metadata}))
    return path


def test_astra_path_refuses(tmp_path):
    text = "synthetic approach B"
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"cells": [
        {"cell": "m:s", "approach_sha256": _sha(text), "approach_chars": len(text)},
        {"cell": "m:empty", "approach_sha256": _sha(""), "approach_chars": 0},
        {"cell": "m:unrecorded"}]}))
    good = _reference(tmp_path / "good.eval", {"attack_approach": text})
    assert approach_guard.check(good, "m:s", tasks, 1) == _sha(text)
    bad = _reference(tmp_path / "bad.eval", {"attack_approach": text + " changed"})
    none = _reference(tmp_path / "none.eval", {})
    for ref, cell in [(bad, "m:s"), (none, "m:s"), (good, "m:unrecorded"), (good, "m:absent"), (good, "m:empty")]:
        with pytest.raises(ValueError):
            approach_guard.check(ref, cell, tasks, 1)
    # A recorded "" passes only if the launch registry has no strategy for the side task; one strategy (a stale
    # record, or a registry that changed) would let the builder's fallback fill it in.
    assert approach_guard.check(none, "m:empty", tasks, 0) == _sha("")
    with pytest.raises(ValueError):
        approach_guard.check(none, "m:empty", tasks, 1)


def test_strategy_counts_asks_the_launch_tree(tmp_path):
    """The counts come from the launch tree's own Python and registry (a synthetic one here), in the launch's env."""
    for d in (".venv/bin", "runner/arenas"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    (tmp_path / "runner/arenas/linuxarena.py").write_text(
        "import os\nfrom types import SimpleNamespace as N\n"
        "REG = {os.environ['CONTROL_TOWER_SETTINGS_DIR']: {'has': [N(strategy='x'), N(strategy='y')], 'none': []}}\n"
        "def linuxarena(env):\n"
        "    sides = {s: N(attack_strategies=v) for s, v in REG[env].items()}\n"
        "    return N(env=lambda: N(combinations=N(sides=lambda: sides)))\n")
    env = dict(os.environ, CONTROL_TOWER_SETTINGS_DIR="e")
    counts = approach_guard.strategy_counts(tmp_path, [("e", "has"), ("e", "none")], env)
    assert counts == {("e", "has"): 2, ("e", "none"): 0}
    with pytest.raises(RuntimeError):  # a side the registry doesn't know stops the launch
        approach_guard.strategy_counts(tmp_path, [("e", "missing")], env)
