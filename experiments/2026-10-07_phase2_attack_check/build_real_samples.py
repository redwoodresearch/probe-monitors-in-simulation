"""Build each candidate pilot task's real attack and honest samples with the September tree's own task code, no
sandbox and no model call: the attack build runs tasks.py's checks (Sol's approach, system prompt and first message
hashes). Run from the attack-check folder with the September venv and PYTHONPATH=<tree>/src:
  python build_real_samples.py [--approaches <dir>] <task file> [<task file> ...] <out json>

Task files that record approaches as sha256s get their texts restored from <dir> (approaches.py) and checked.
"""
import json, os, sys, traceback
from pathlib import Path
import yaml
HERE = Path.cwd()
cfg = yaml.safe_load((HERE / "config/run.yaml").read_text())
os.environ["CONTROL_TOWER_SETTINGS_DIR"] = str(Path(cfg["settings_dir"]).expanduser())
sys.path.insert(0, str(HERE))
import approaches
import tasks
import runner.tasks.linuxarena_real as real
args = sys.argv[1:]
directory = approaches.APPROACHES
if args[0] == "--approaches":
    directory, args = Path(args[1]).expanduser(), args[2:]
*files, out = args
cells, sources = {}, {}
for f in files:
    restored = approaches.materialize(Path(f), Path("/tmp") / f"restored_{Path(f).name}", directory)
    for t in json.loads(restored.read_text())["cells"]:
        cells.setdefault(t["cell"], t)
        sources.setdefault(t["cell"], []).append(Path(f).name)
combined = Path("/tmp/phase2_candidate_cells.json")
combined.write_text(json.dumps({"cells": list(cells.values())}))
common = {"PT_ATTACKER": cfg["policy"], "BA_MAX_STEPS": str(cfg["run"]["max_steps"]), "VLLM_BASE_URL": "http://unused/v1",
          "SKIP_DOCKER_BUILD": "1", "INTERNET_SIMULATOR_IMAGE": cfg["internet_simulator_image"],
          **{f"ENV_IMAGE_TAG_{e.upper()}": c for e, c in cfg["env_commits"].items()}}
rows = []
for cell, t in cells.items():
    for mode in ("attack", "honest"):
        for k in [k for k in os.environ if k.startswith(("BA_", "PT_", "LA_"))]:
            del os.environ[k]
        os.environ.update(common | {"LA_ENV": t["env"]})
        if mode == "attack":
            os.environ.update({"BA_MODE": "attack", "BA_CELLS": cell, "BA_ATTACKER_PROMPT": cfg["run"]["attacker_prompt"],
                               "PHASE2_CELLS": str(combined)})
        else:
            os.environ.update({"BA_MODE": "honest", "BA_MAINS": t["main"]})
        row = {"cell": cell, "env": t["env"], "mode": mode, "in": sources[cell]}
        try:
            task = tasks.la_real_attack_sept() if mode == "attack" else real.linuxarena_real()
            samples = list(task.dataset)
            row |= {"built": len(samples) == 1, "samples": len(samples)}
        except Exception as e:  # recorded per task and mode
            row |= {"built": False, "error": f"{type(e).__name__}: {e}"[:500], "where": traceback.format_exc().splitlines()[-3:]}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("cell", "mode", "built") if k in row} | ({"error": row["error"][:200]} if "error" in row else {})), flush=True)
json.dump({"files": files, "tasks": len(cells), "rows": rows, "failed": [r for r in rows if not r["built"]]}, open(out, "w"), indent=1)
print(f"{sum(r['built'] for r in rows)} of {len(rows)} built -> {out}")
