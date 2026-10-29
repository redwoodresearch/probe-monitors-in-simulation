"""new_mains_v1 must leave the legacy (Sol-hash) construction exactly as it was, and reproduce its recorded provenance.
GATED, ON THE DEVBOX from the assembled harness folder (September venv, pinned settings), $0, no model call:

  BASE_TASKS=<bbf0729's tasks.py> LEGACY_TASKS=<the 196's task file> NEW_MAINS_CELLS=<recorded cells file>
  NEW_MAINS_FIXTURES=<fixtures json> APPROACHES=<approach dir> CONTROL_TOWER_SETTINGS_DIR=<settings> \\
      <tree>/.venv/bin/python -m pytest -q <this file>

A missing variable fails (nothing is skipped). The tests restore approach texts into a temporary folder they delete,
and compare and report hashes and cell keys only.
"""

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

import approaches
import yaml

HARNESS = Path.cwd()  # the assembled harness folder (tasks.py, approaches.py, config/run.yaml)
LEGACY = ("_sha", "_tell_septembers_attack", "la_real_attack_sept", "la_sim_attack_sept")
# The 196's two build refusals (registry approach differs from September's), the same under either tasks.py.
EXPECTED_REFUSALS = {"model_integrity_verification:exfiltrate_weights", "severity_renaming:suppress_safety_notifications"}
PREFIXES = ("BA_", "PT_", "LA_", "SCE_", "INSPECT_", "PHASE2_")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _defs(path: Path) -> dict:
    src = path.read_text()
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name in LEGACY}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell_env(cell: dict, cells_file: Path) -> None:
    for k in [k for k in os.environ if k.startswith(PREFIXES)]:
        del os.environ[k]
    prompt = yaml.safe_load((HARNESS / "config" / "run.yaml").read_text())["run"]["attacker_prompt"]
    os.environ.update({"LA_ENV": cell["env"], "BA_MODE": "attack", "BA_CELLS": cell["cell"],
                       "BA_ATTACKER_PROMPT": prompt, "PHASE2_CELLS": str(cells_file)})


def _restored(var: str, tmp: Path) -> Path:
    return approaches.materialize(Path(os.environ[var]), tmp / f"{var}.json", Path(os.environ["APPROACHES"]))


def _built(task_fn) -> str:
    """The built sample as a sha256 of its canonical JSON, or a refusal (hash of its message): never text."""
    try:
        (s,) = list(task_fn().dataset)
    except AssertionError as e:
        return "refused:" + _sha(str(e))
    return _sha(json.dumps([s.id, s.input, s.target, s.metadata], sort_keys=True, default=str))


def test_legacy_source_and_builds_unchanged():
    base_path = Path(os.environ["BASE_TASKS"])
    assert _defs(base_path) == _defs(HARNESS / "tasks.py") and len(_defs(base_path)) == len(LEGACY)
    base, new = _module(base_path, "tasks_base"), _module(HARNESS / "tasks.py", "tasks_new")
    tmp = Path(tempfile.mkdtemp(prefix="nm-legacy-"))
    try:
        cells_file = _restored("LEGACY_TASKS", tmp)
        cells = json.loads(cells_file.read_text())["cells"]
        assert len(cells) == 196, len(cells)
        refused = set()
        for cell in cells:
            _cell_env(cell, cells_file)
            a = _built(base.la_real_attack_sept)
            assert a == _built(new.la_real_attack_sept) == _built(new.attack_task("real")), cell["cell"]
            refused |= {cell["cell"]} if a.startswith("refused:") else set()
        assert refused == EXPECTED_REFUSALS, sorted(refused ^ EXPECTED_REFUSALS)
    finally:
        shutil.rmtree(tmp)


def test_fixtures_reproduce():
    import tasks

    raw = json.loads(Path(os.environ["NEW_MAINS_CELLS"]).read_text())
    assert raw["template_sha256"]["with_approach"] == tasks.TEMPLATE_196[True]
    assert raw["template_sha256"]["without_approach"] == tasks.TEMPLATE_196[False]
    fixtures = json.loads(Path(os.environ["NEW_MAINS_FIXTURES"]).read_text())["fixtures"]
    assert sorted(fixtures) == sorted(raw["fixture_sample"]) and fixtures
    tmp = Path(tempfile.mkdtemp(prefix="nm-fixtures-"))
    try:
        cells_file = _restored("NEW_MAINS_CELLS", tmp)
        cells = {c["cell"]: c for c in json.loads(cells_file.read_text())["cells"]}
        for name, want in fixtures.items():
            _cell_env(cells[name], cells_file)
            assert tasks.attack_task("real") is tasks.la_real_attack_new_mains_v1, name
            assert tasks.la_real_attack_new_mains_v1().provenance == want, name
    finally:
        shutil.rmtree(tmp)
