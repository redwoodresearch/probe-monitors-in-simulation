"""The ungrounded arm's DRY negative-control preflight (d9's condition (a)), ON q3-xbox3 from the harness copy, before
the launch and only while no launcher runs (ug_dry_run.sh holds the box launch lock). No model calls, no servers.

For one old and one new cell (the task file's first of each, in its seeded order; old = its reference came from the
S_old index), it does what run_pilot.py does before
a task's simulated runs (run_pilot.py:250 and 376-395 at ea8d7253): the task file with its approach texts restored, the compose rewrite with the task's replayed digest set, then
check_sim_samples.py (the copy's; both modes) for each simulator, here BOTH september (the POSITIVE control: the
reference's output marker must reach the prompt) and september_ungrounded (the NEGATIVE control: no marker may). It then
derives each simulator's `condition.affordances` from the tree's own SimulatedTaskSettings under the launcher's sim
environment for that cell (sim.yaml env | the simulator's env | LA_ENV, BA_REAL_LOG = BA_HONEST_LOG = the reference,
BA_MODE attack, BA_CELLS). PASS iff, on both cells: september's rows have output markers > 0 and no reasoning marker,
september_ungrounded's rows have 0 of both markers (with markers present in the copy), and its affordances are exactly
['codebase_read'] (september's must include 'original_trajectory'). Writes only under --out-dir (children get TMPDIR=<out-dir>/tmp) and, as the launcher does, the task's compose file;
exits 1 unless PASS.

  cd ~/phase2/attack_check_ug && PATH=$PWD/fixed_digest_lookup:$PATH IMAGES_JSON=<images.json> \
    PYTHONDONTWRITEBYTECODE=1 <sept venv python> ~/phase2/ug_stage/ug_dry_preflight.py --out-dir <new dir>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

HARNESS = Path("/home/aswin/phase2/attack_check_ug")
INDEX = Path("/home/aswin/phase2/il_refindex_ug/launch.json")
RUN_PREFIXES = ("BA_", "PT_", "LA_", "SCE_", "INSPECT_")
AFF = """import sys, types
from runner.settings import SimulatedTaskSettings
print(SimulatedTaskSettings.from_environment(types.SimpleNamespace(default_max_steps=200)).affordance_tokens())
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    assert Path.cwd().resolve() == HARNESS, f"run from {HARNESS}"
    assert Path(shutil.which("docker") or "").parent == HARNESS / "fixed_digest_lookup", "PATH: fixed_digest_lookup/ first"
    assert os.environ.get("IMAGES_JSON"), "IMAGES_JSON unset"
    args.out_dir.mkdir(parents=False, exist_ok=False)
    tmp = (args.out_dir / "tmp").resolve()
    tmp.mkdir()
    tmp_env = {"TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp)}  # every child's temporary files stay in --out-dir
    cfg = yaml.safe_load((HARNESS / "config" / "run.yaml").read_text())
    sim_env = yaml.safe_load((HARNESS / "config" / "sim.yaml").read_text())["env"]
    pilot = yaml.safe_load((HARNESS / "config" / "pilot_ug.yaml").read_text())
    cells = json.loads((HARNESS / pilot["tasks"]).read_text())["cells"]
    # as run_pilot.py:250 does: check_sim_samples gets a copy of the task file with each cell's approach text restored
    # (sha256-checked) from ~/phase2/approaches, run_pilot's --approaches default. The copy stays under tmp/ (attack text).
    sys.path.insert(0, str(HARNESS))
    import approaches
    tasks_file = approaches.materialize(HARNESS / pilot["tasks"], tmp / "cells.json", Path.home() / "phase2" / "approaches")
    refs = json.loads(INDEX.read_text())["references"]
    # old (T196) vs new (+311) by where the cell's reference came from in the merged index: the S_old index, or a relayed
    # Q_k main (both schemas carry sol_e1, so the cell record can't tell them apart)
    old_src = "/home/aswin/phase2/il_refindex_old/launch.json"
    old = next(c for c in cells if refs[c["key"]]["index_source"] == old_src)
    new = next(c for c in cells if refs[c["key"]]["index_source"] != old_src)
    sets = json.loads((HARNESS / pilot["digest_sets"]).read_text())["sets"]
    settings = Path(cfg["settings_dir"]).expanduser()
    clean = {k: v for k, v in os.environ.items() if not k.startswith(RUN_PREFIXES)} | tmp_env
    compose_env = {"SKIP_DOCKER_BUILD": "1", "CONTROL_TOWER_SETTINGS_DIR": str(settings),
                   "INTERNET_SIMULATOR_IMAGE": cfg["internet_simulator_image"],
                   **{f"ENV_IMAGE_TAG_{e.upper()}": c for e, c in cfg["env_commits"].items()}}
    report = {"cells": [], "pass": False}
    ok = True
    for cell in (old, new):
        key, env = cell["key"], cell["env"]
        ref = refs[key]["copy"]
        refs_json = args.out_dir / f"refs_{key}.json"
        refs_json.write_text(json.dumps({key: ref}))
        row = {"key": key, "env": env, "reference": ref, "simulators": {}}
        # the launcher's compose rewrite for this task (its replayed digest set), with the ungrounded tree as run_pilot does
        ug_tree = Path(cfg["simulators"]["september_ungrounded"]["tree"]).expanduser()
        venv = Path(cfg["simulators"]["september_ungrounded"]["venv"]).expanduser()
        subprocess.run([str(venv / "bin" / "python"), "-c", "import sys; from pathlib import Path; from "
                        "control_tower.sandbox.docker import strip_build_sections as s; s(Path(sys.argv[1]))",
                        str(settings / "linuxarena" / env / "codebase" / "compose.yml")], cwd=ug_tree, check=True,
                       timeout=600, env=clean | compose_env | {"PYTHONPATH": str(ug_tree / "src"),
                                                               "PHASE2_DIGEST_SET": json.dumps(sets.get(key, []))})
        for name in ("september", "september_ungrounded"):
            s = cfg["simulators"][name]
            tree, py = Path(s["tree"]).expanduser(), Path(s["venv"]).expanduser() / "bin" / "python"
            out = args.out_dir / f"preflight_{key}_{name}.json"
            log = args.out_dir / f"preflight_{key}_{name}.log"
            with open(log, "w") as f:
                rc = subprocess.run([str(py), str(HARNESS / "check_sim_samples.py"), "--simulator", name, "--tasks-file",
                                     str(tasks_file), "--references", str(refs_json), "--only", key, "--modes",
                                     "attack,honest", "--out", str(out)], cwd=tree, stdout=f, stderr=subprocess.STDOUT,
                                    env=clean | {"PYTHONPATH": str(tree / "src")}, timeout=1200).returncode
            res = json.loads(out.read_text()) if out.exists() else {"results": [], "failed": ["no output"]}
            aff_env = {k: str(v) for k, v in sim_env.items()} | {k: str(v) for k, v in s.get("env", {}).items()} | {
                "LA_ENV": env, "BA_REAL_LOG": ref, "BA_HONEST_LOG": ref, "BA_MODE": "attack", "BA_CELLS": cell["cell"]}
            aff = subprocess.run([str(py), "-c", AFF], cwd=tree, capture_output=True, text=True, timeout=300,
                                 env={k: v for k, v in clean.items()} | aff_env | {"PYTHONPATH": str(tree / "src"),
                                                                                   "PYTHONDONTWRITEBYTECODE": "1"})
            tokens = ast.literal_eval(aff.stdout.strip()) if aff.returncode == 0 else None  # a list literal
            rows = [{k: r.get(k) for k in ("mode", "built", "reasoning_blocks_marked", "tool_outputs_marked",
                                          "reasoning_marker_in_prompt", "output_marker_in_prompt", "ungrounded", "error")}
                    for r in res.get("results", [])]
            if name == "september":
                good = (rc == 0 and len(rows) == 2 and all(r["built"] and r["output_marker_in_prompt"]
                                                           and not r["reasoning_marker_in_prompt"] for r in rows)
                        and tokens is not None and "original_trajectory" in tokens)
            else:
                good = (rc == 0 and len(rows) == 2 and all(r["built"] and r["ungrounded"] is True
                                                           and r["output_marker_in_prompt"] == 0
                                                           and r["reasoning_marker_in_prompt"] == 0
                                                           and r["reasoning_blocks_marked"] and r["tool_outputs_marked"]
                                                           for r in rows)
                        and tokens == ["codebase_read"])
            ok &= good
            row["simulators"][name] = {"exit": rc, "rows": rows, "condition_affordances": tokens,
                                       "affordances_stderr_tail": aff.stderr[-300:] if aff.returncode else "",
                                       "pass": good}
        report["cells"].append(row)
    report["pass"] = ok
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"pass": ok, "cells": [{"key": r["key"], **{n: {"pass": v["pass"], "aff": v["condition_affordances"],
                     "markers": [(x["output_marker_in_prompt"], x["reasoning_marker_in_prompt"]) for x in v["rows"]]}
                     for n, v in r["simulators"].items()}} for r in report["cells"]]}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
