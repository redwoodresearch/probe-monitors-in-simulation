"""LinuxArena attack samples told exactly what Sol was told in September, for one cell: real (step 1) or simulated
(step 2, the September simulator answering, forked at 0 from a real honest reference).

The September snapshot's own `real_task` builds the sample. The one input it can't reproduce is the cell's attack
approach: the September fleet passed it in per cell, and the env registry has none for these cells (`authored_approach`
returns ""). So it is set from Sol's epoch-1 sample (config/cells.json), and the task refuses to build unless the
rendered system prompt and the first user message hash to exactly Sol's. Runs ON THE DEVBOX, from the snapshot tree:

  PHASE2_CELLS=<cells.json> LA_ENV=<env> BA_MODE=attack BA_CELLS=<main>:<side> BA_ATTACKER_PROMPT=attacker_2026_08_28.md \\
      <snapshot>/.venv/bin/inspect eval <this file>@la_real_attack_sept ...
  ... plus BA_REAL_LOG=BA_HONEST_LOG=<reference .eval>, BA_FORK_FRACTION=0 and the simulator settings:
      <snapshot>/.venv/bin/inspect eval <this file>@la_sim_attack_sept ...

In September the simulated attacker's prompt was the real one, byte for byte (same approach, fork 0; checked on the
5 cells' stored samples), so both tasks hold to the same hashes.
"""

import hashlib
import json
import os
from pathlib import Path

from inspect_ai import Task, task
from runner.arenas.linuxarena import linuxarena
from runner.build.arena_real_task import real_task
from runner.build.arena_simulated_task import sim_task
from runner.policy.prompts import attacker_prompt
from runner.settings import (
    RealTaskSettings,
    SimulatedTaskSettings,
    required_environment_name,
)
from sim.identity import Mode


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _tell_septembers_attack(built: Task, cell: str, prompt_name: str) -> Task:
    """Give the task's one sample the cell's September approach, then refuse it unless the attacker's rendered system
    prompt and first user message hash to exactly Sol's."""
    cells = {c["cell"]: c for c in json.loads(Path(os.environ["PHASE2_CELLS"]).read_text())["cells"]}
    record = cells[cell]
    (sample,) = list(built.dataset)  # MemoryDataset yields its own Sample objects, so the edit below sticks
    # The September snapshot finds no approach in the env registry; today's sims code reads one from it, byte-identical
    # to September's for the step-1 cells (checked 2026-10-08). Any other text stops the build.
    assert sample.metadata["attack_approach"] in ("", record["approach"]), "the registry's approach differs from September's"
    sample.metadata["attack_approach"] = record["approach"]
    md = sample.metadata
    system = attacker_prompt(main_task=md["main_task"]["description"], side_task=md["side_task_text"],
                             environment=md["environment"]["description"], attack_approach=md["attack_approach"],
                             filename=prompt_name)
    assert _sha(system) == record["sol_system_sha256"], f"{record['cell']}: system prompt differs from Sol's"
    assert isinstance(sample.input, str) and _sha(sample.input) == record["sol_input_sha256"], (
        f"{record['cell']}: first user message differs from Sol's")
    return built


@task
def la_real_attack_sept() -> Task:
    arena = linuxarena(required_environment_name())
    settings = RealTaskSettings.from_environment(arena)
    assert settings.mode is Mode.ATTACK and len(settings.cells) == 1 and not settings.real_logs, settings
    return _tell_septembers_attack(real_task(arena, settings), settings.cells[0], settings.attacker_prompt)


@task
def la_sim_attack_sept() -> Task:
    arena = linuxarena(required_environment_name())
    settings = SimulatedTaskSettings.from_environment(arena)
    assert settings.mode is Mode.ATTACK and len(settings.cells) == 1 and settings.fork_fraction == 0.0, settings
    assert settings.real_logs == settings.honest_logs, "the reference is the fork source, as in September"
    return _tell_septembers_attack(sim_task(arena, settings), settings.cells[0], settings.attacker_prompt)
