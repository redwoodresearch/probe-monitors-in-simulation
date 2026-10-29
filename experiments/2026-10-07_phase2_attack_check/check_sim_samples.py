"""Step 2's preflight, no model calls: build each cell's simulated attack sample from its step-1 honest reference with
the September snapshot's own task code, run it up to the simulator's system prompt, and check what the simulator would
be given. Runs ON THE DEVBOX with the snapshot's Python:

  ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/attack_check/check_sim_samples.py --step1 <run dir>

For the pilot, either simulator of config/run.yaml, with the references given as a JSON {env: honest reference .eval}
(as run_attack_check.py --references takes them), run with that simulator's environment and tree, as a launch runs:

  PYTHONPATH=<tree>/src <venv>/bin/python check_sim_samples.py --simulator today --references <refs.json>

The simulator's own settings (run.yaml `env`, e.g. today's router mode and briefing) are added to sim.yaml's, and the
check refuses to run unless the sims code it imports is that simulator's tree (or, for September as step 2 ran it,
its base).

For the pilot (run_pilot.py runs it on each task's new reference, once per simulator, before that simulator's runs of
the task start): --tasks-file config/pilot_tasks.json, references keyed by task key, --only <task key>, --modes
attack,honest (the honest sample is the tree's own linuxarena_sim.py in honest mode, as sim_honest runs it), and --out.

Checks, per cell (writes results/sim_preflight_<stamp>.json; exits nonzero if any fails):
  - the sample builds from the real reference: one sample, the reference's environment digest matches our settings
    folder, and the attacker's system prompt and first message hash to Sol's (tasks.la_sim_attack_sept);
  - the reference's reasoning never reaches the simulator. Qwen3.5's reasoning is plain text, where Sol's (September)
    was encrypted, so a leak would be a new difference from Phase 1. Tested with markers: a copy of the reference gets
    one unique marker appended to every reasoning block and another to every tool output, the sample is built from the
    copy, and the simulator's system prompt must hold the output marker (the dump pastes outputs in: the positive
    control) and not the reasoning marker. Per live step the simulator gets only each call's function, description and
    arguments (runner/simulate/tools.py), never the agent's reasoning or text, so the system prompt is the only route.
It also records the real prompt's length, an input to the cost estimate.
"""

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config" / "run.yaml").read_text())
# control-tower reads the settings folder when it is first imported, so it is set before anything imports it.
os.environ["CONTROL_TOWER_SETTINGS_DIR"] = str(Path(CFG["settings_dir"]).expanduser())

from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from runner.simulate.config import config_for_sample
from sim.simulator import Simulator

sys.path.insert(0, str(HERE))
import runner
import tasks
from run_attack_check import src_sha256

RUN_PREFIXES = ("BA_", "PT_", "LA_", "SCE_", "INSPECT_")  # CONTROL_TOWER_SETTINGS_DIR (set above) is not one


def simulator_prompt(env: dict) -> str:
    """The simulator's system prompt for the one sample the task code builds under `env`: tasks.py's attack task, or
    the tree's own linuxarena_sim.py in honest mode."""
    for name in [n for n in os.environ if n.startswith(RUN_PREFIXES)]:
        del os.environ[name]
    os.environ.update(env)
    if env["BA_MODE"] == "honest":
        import runner.tasks.linuxarena_sim as linuxarena_sim

        (sample,) = list(linuxarena_sim.linuxarena_sim().dataset)
    else:
        (sample,) = list(tasks.la_sim_attack_sept().dataset)
    settings = tasks.SimulatedTaskSettings.from_environment(tasks.linuxarena(env["LA_ENV"]))
    config = config_for_sample(sample.metadata, model=settings.simulator_model, codebase_exec=False,
                               exec_source_authority=True, orig_trajectory_mode="dump", self_critique=None)
    return Simulator(config, client=object(), fallback=None).system_prompt()


def marked_copy(ref: Path, out: Path, reasoning_mark: str, output_mark: str) -> tuple[int, int]:
    """Write `ref` to `out` with a marker appended to every reasoning block and every string tool output."""
    log = read_eval_log(str(ref), resolve_attachments=True)
    (sample,) = log.samples
    blocks = outputs = 0
    for m in sample.messages:
        if isinstance(m, ChatMessageAssistant) and isinstance(m.content, list):
            for c in m.content:
                if c.type == "reasoning" and c.reasoning.strip():
                    c.reasoning += f" {reasoning_mark}"
                    blocks += 1
        elif isinstance(m, ChatMessageTool) and isinstance(m.content, str):
            m.content += f"\n{output_mark}"
            outputs += 1
    write_eval_log(log, str(out))
    return blocks, outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    refs = ap.add_mutually_exclusive_group(required=True)
    refs.add_argument("--step1", type=Path, help="the step-1 run folder (holds honest_a/<env>/*.eval)")
    refs.add_argument("--references", type=Path, help="JSON {task key, or env: real honest reference .eval}")
    ap.add_argument("--simulator", default="september", help="which of run.yaml's simulators")
    ap.add_argument("--tasks-file", type=Path, default=HERE / "config" / "cells.json")
    ap.add_argument("--approaches", type=Path, help="the approach texts for a task file that records only their "
                    "sha256s (default: approaches.py's data folder)")
    ap.add_argument("--only", help="only these task keys or cells, comma-separated")
    ap.add_argument("--modes", default="attack", help="attack, honest, or both (comma-separated)")
    ap.add_argument("--out", type=Path, help="default: results/sim_preflight_<stamp>.json")
    args = ap.parse_args()
    modes = args.modes.split(",")
    assert set(modes) <= {"attack", "honest"}, modes
    simulator = CFG["simulators"][args.simulator]
    imported = Path(runner.__file__).resolve().parents[1]  # <tree>/src
    # The patched tree, as every launch runs it. The unpatched base is accepted only for September, as step 2's
    # preflight ran it (its sample-building code is the same); today's must be the patched tree.
    allowed = {Path(simulator["tree"]).expanduser().resolve() / "src": simulator["src_sha256"]}
    if args.simulator == "september":
        allowed[Path(simulator["base_tree"]).expanduser().resolve() / "src"] = simulator["base_src_sha256"]
    assert imported in allowed, f"sims code imported from {imported}, not {args.simulator}'s tree (set PYTHONPATH)"
    simulator_src = Path(sys.modules[Simulator.__module__].__file__).resolve().parents[1]  # sim is its own package
    assert simulator_src == imported, f"the simulator was imported from {simulator_src}, the runner from {imported}"
    tree_hash = src_sha256(imported.parent)
    assert tree_hash == allowed[imported], f"{imported} hashes to {tree_hash}, not {allowed[imported]}"
    given = json.loads(args.references.read_text()) if args.references else None
    sim = yaml.safe_load((HERE / "config" / "sim.yaml").read_text())
    import approaches  # a pilot task file records approaches as sha256s: restore them (checked) into a temp copy

    restored = approaches.materialize(args.tasks_file, Path(tempfile.mkdtemp(prefix="phase2-cells-")) / "cells.json",
                                      (args.approaches or approaches.APPROACHES).expanduser())
    cells = json.loads(restored.read_text())["cells"]
    if args.only:
        wanted = set(args.only.split(","))
        cells = [c for c in cells if c.get("key") in wanted or c["cell"] in wanted]
        assert cells, f"no task matches {sorted(wanted)}"
    results, failed = [], []
    with tempfile.TemporaryDirectory(prefix="phase2-preflight-") as tmp:
        for cell, mode in ((c, m) for c in cells for m in modes):
            row = {"cell": cell["cell"], "env": cell["env"], "task": cell.get("key"), "mode": mode}
            base = sim["env"] | simulator.get("env", {}) | {"LA_ENV": cell["env"]} | (
                {"BA_MODE": "attack", "BA_CELLS": cell["cell"], "PHASE2_CELLS": str(restored)}
                if mode == "attack" else {"BA_MODE": "honest"})
            try:
                refs = [Path(given[cell.get("key", cell["env"])]).expanduser().resolve()] if given else \
                    sorted((args.step1.resolve() / "honest_a" / cell["env"]).glob("*.eval"))
                assert len(refs) == 1, f"need exactly one reference .eval, found {[str(r) for r in refs]}"
                (ref,) = refs
                row["reference"] = str(ref)
                prompt = simulator_prompt(base | {"BA_REAL_LOG": str(ref), "BA_HONEST_LOG": str(ref)})
                r_mark, o_mark = f"PHASE2-REASONING-{uuid.uuid4().hex}", f"PHASE2-OUTPUT-{uuid.uuid4().hex}"
                copy = Path(tmp) / f"{cell.get('key', cell['env'])}_{mode}.eval"
                blocks, outputs = marked_copy(ref, copy, r_mark, o_mark)
                marked = simulator_prompt(base | {"BA_REAL_LOG": str(copy), "BA_HONEST_LOG": str(copy)})
                row |= {"built": True, "simulator_prompt_chars": len(prompt), "reasoning_blocks_marked": blocks,
                        "tool_outputs_marked": outputs, "reasoning_marker_in_prompt": marked.count(r_mark),
                        "output_marker_in_prompt": marked.count(o_mark)}
                # No reasoning blocks, or no output marker found: the test would prove nothing, so it doesn't pass.
                if row["reasoning_marker_in_prompt"] or not blocks or not row["output_marker_in_prompt"]:
                    failed.append(f"{cell['cell']} ({mode})")
            except Exception as e:  # noqa: BLE001 — recorded per cell, then the whole check fails below
                row |= {"built": False, "error": repr(e)[:2000]}
                failed.append(f"{cell['cell']} ({mode})")
            results.append(row)
            print(json.dumps(row), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = args.out or HERE / "results" / f"sim_preflight_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump({"step1": str(args.step1) if args.step1 else None, "references": given, "modes": modes,
                   "tasks_file": str(args.tasks_file),
                   "simulator": args.simulator, "sims_src": str(imported), "sims_src_sha256": tree_hash,
                   "results": results, "failed": failed}, f, indent=1)
    print(f"-> {out}")
    if failed:
        raise SystemExit(f"preflight failed for {failed}")


if __name__ == "__main__":
    main()
