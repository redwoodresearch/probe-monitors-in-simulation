"""Launch prep for a new_mains_v1 cells file (GATED: runs only when a launch is approved). ON THE DEVBOX, from the
assembled harness folder (README: harness), with the September venv and pinned settings; $0, no model call. It builds
every cell's attack sample through tasks.py's new_mains_v1 builder (the code the launch runs) and records the rendered
prompt's sha256 as `provenance_system_sha256`, which every later build of that cell must reproduce, and writes the
fixture hashes for the cells file's seeded `fixture_sample`. Provenance, not equivalence validation. Hashes only.

  CONTROL_TOWER_SETTINGS_DIR=<settings> <tree>/.venv/bin/python record_provenance.py <recorded cells json> \\
      <out cells json> <fixtures json> [--approaches <dir>]
(<recorded cells json>: after record_fleet_approaches.py, whose approach_sha256 must equal q2_approach_sha256. Only
this script sets PHASE2_RECORD_PROVENANCE=1; the launcher refuses to start with it set, or with any cell unrecorded.)
"""

import argparse
import json
import os
from pathlib import Path

import approaches
import tasks
import yaml


def build(cell: dict, cells_file: Path, prompt: str) -> dict:
    for k in [k for k in os.environ if k.startswith(("BA_", "PT_", "LA_", "SCE_", "INSPECT_", "PHASE2_"))]:
        del os.environ[k]
    os.environ.update({"LA_ENV": cell["env"], "BA_MODE": "attack", "BA_CELLS": cell["cell"], "BA_ATTACKER_PROMPT": prompt,
                       "PHASE2_CELLS": str(cells_file), "PHASE2_RECORD_PROVENANCE": "1"})  # the one recording mode
    return tasks.la_real_attack_new_mains_v1().provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cells", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("fixtures", type=Path)
    p.add_argument("--approaches", type=Path, default=approaches.APPROACHES)
    a = p.parse_args()
    assert not a.out.exists() and not a.fixtures.exists(), "outputs exist (append-only)"
    data = json.loads(a.cells.read_text())
    assert data["construction"] == tasks.NEW_MAINS_V1 and all(c.get("provenance_system_sha256") is None for c in data["cells"])
    for c in data["cells"]:
        assert (c["approach_sha256"], c["approach_chars"]) == (c["q2_approach_sha256"], c["q2_approach_chars"]), c["cell"]
    prompt = yaml.safe_load(Path("config/run.yaml").read_text())["run"]["attacker_prompt"]
    restored = approaches.materialize(a.cells, a.out.with_suffix(".restored.json"), a.approaches.expanduser())
    try:
        prov = {c["cell"]: build(c, restored, prompt) for c in data["cells"]}
    finally:
        restored.unlink()  # restored texts never stay on disk next to the record
        os.environ.pop("PHASE2_RECORD_PROVENANCE", None)
    for c in data["cells"]:
        c["provenance_system_sha256"] = prov[c["cell"]]["system_sha256"]
    a.out.write_text(json.dumps(data, indent=1))
    a.fixtures.write_text(json.dumps({"note": "provenance, not equivalence validation", "construction": tasks.NEW_MAINS_V1,
                                      "fixtures": {x: prov[x] for x in data["fixture_sample"]}}, indent=1))
    print(f"{len(prov)} cells recorded; {len(data['fixture_sample'])} fixtures")


if __name__ == "__main__":
    main()
