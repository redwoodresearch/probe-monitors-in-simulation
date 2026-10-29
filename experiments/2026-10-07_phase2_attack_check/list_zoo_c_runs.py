"""Zoo C, after each stage: the run list Aswin labels and q2's guard joins (eval file, sample ID, task, run order; IDs
only, no text), and the errored-run count per batch the orchestrator asked for (retry_on_error differs from overlap800).
Reads a local copy of the stage's run folder: launch.json for each run's attempts and outcome, and each .eval's header
for its sample ID and status. $0, local.

Prediction for stage 1 (written before it ended): 100 runs, every one with exactly one log and one sample; at most 3
errored; no run hits the 150-step limit more than 10 times.

  uv run --no-project --python 3.14 python list_zoo_c_runs.py <local run folder> <stage>
"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
TASKS = HERE / "config" / "pilot_tasks_zoo_c_oct11c.json"


def main(run: Path, stage: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes: commit them first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    spec = json.loads(TASKS.read_bytes())
    cells = {c["key"]: c for c in spec["cells"]}
    lj_raw = (run / "launch.json").read_bytes()
    lj = json.loads(lj_raw)
    if lj["input_sha256"].get("config/" + TASKS.name) != hashlib.sha256(TASKS.read_bytes()).hexdigest():
        raise SystemExit("the local task file is not the one this launch recorded (input_sha256)")
    keys = spec["stages"][stage]
    if sorted(r["task"] for r in lj["runs"]) != sorted(keys):
        raise SystemExit("the launch's runs are not this stage's keys")
    rows = []
    for r in sorted(lj["runs"], key=lambda r: keys.index(r["task"])):
        c = cells[r["task"]]
        a = r["attempts"][-1] if r["attempts"] else {}  # the counted attempt (earlier ones were pre-start redraws)
        evs = sorted((run / "honest_b" / Path(a["log_dir"]).name).glob("*.eval")) if a else []
        row = {"key": r["task"], "batch": spec["batch_tag"], "stage": stage, "position": c["position"], "run_order": c["run"],
               "task": c["main"], "env": c["env"], "state": r["state"], "attempts": len(r["attempts"]), "logs": len(evs)}
        if len(evs) == 1:
            try:
                with zipfile.ZipFile(evs[0]) as z:
                    h = json.loads(z.read("header.json"))
            except (zipfile.BadZipFile, KeyError, OSError, ValueError):
                h = None  # an unreadable log: counted as errored, with a fixed reason
            o = a.get("outcome")  # None: the launcher recorded no outcome, so errors are unknown (counted as errored)
            pc = (o or {}).get("policy_calls") or [{}]
            row |= {"eval_file": evs[0].name, "eval_sha256": hashlib.sha256(evs[0].read_bytes()).hexdigest(),
                    "sample_id": (h["eval"]["dataset"].get("sample_ids") or [None])[0] if h else None,
                    "status": h.get("status") if h else "unreadable log",
                    "sample_errors": len(o.get("sample_errors") or []) if o else None, "clean": o.get("clean") if o else None,
                    "ended": pc[0].get("ended"), "steps": pc[0].get("steps"), "step_limit_hit": bool(pc[0].get("step_limit_hits"))}
        rows.append(row)
    errored = [x["key"] for x in rows if x.get("status") != "success" or x.get("sample_errors") != 0 or x.get("clean") is not True]
    out = {"label": f"zoo C batch {spec['batch_tag']} stage {stage}: run list (IDs only) and errored runs", "git": git,
           "inputs": {"launch.json": hashlib.sha256(lj_raw).hexdigest(), TASKS.name: hashlib.sha256(TASKS.read_bytes()).hexdigest()},
           "run_folder": run.name, "runs": len(rows), "with_one_log": sum(x["logs"] == 1 for x in rows),
           "errored": len(errored), "errored_keys": errored, "step_limit_hits": sum(x.get("step_limit_hit", False) for x in rows),
           "ended": dict(Counter(x.get("ended") for x in rows)), "rows": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"zoo_c_runs_{spec['batch_tag']}_stage{stage}_{stamp}.json"
    with dst.open("x") as fh:
        fh.write('{"head": ' + json.dumps({k: v for k, v in out.items() if k != "rows"}) + ',\n"rows": [\n'
                 + ",\n".join(json.dumps(x) for x in rows) + "\n]}\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("stage")
    a = ap.parse_args()
    main(a.run, a.stage)
