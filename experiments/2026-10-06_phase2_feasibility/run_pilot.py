"""Run the pilot's real honest LinuxArena tasks with a self-hosted policy, one `inspect eval` per environment, all in
parallel. Runs ON THE DEVBOX, from the sce checkout's environment, with VLLM_API_KEY forwarded into the session.

  cd ~/simulated-control-evals && uv run python ~/phase2/run_pilot.py --model qwen3_32b --config ~/phase2/pilot.yaml

Writes <out>/<env>/*.eval, <out>/<env>.log and <out>/launch.json (commands, environment minus the key, timings).
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

SCE = Path.home() / "simulated-control-evals"


def now() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a key under `models` in the config")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: sce transcripts/<date>_la_real_honest_phase2-<model>")
    ap.add_argument("--only", default="", help="comma-separated environments to run (a rerun of startup failures)")
    args = ap.parse_args()
    # Every run gets os.environ plus the config's settings, so an inherited experiment setting would reach every run
    # without being recorded: refuse rather than strip, so it's noticed.
    stray = sorted(k for k in os.environ if k.startswith(("BA_", "PT_", "LA_", "SCE_", "INSPECT_")))
    assert not stray, f"unset these inherited settings first; they would reach every run unrecorded: {stray}"
    assert args.out or not args.only, "--only needs --out: the default folder is the full run's (append-only)"
    cfg = yaml.safe_load(args.config.read_text())
    run, model = cfg["run"], cfg["models"][args.model]
    assert os.environ.get("VLLM_API_KEY"), "VLLM_API_KEY is not in the environment"
    # The policy settings this config doesn't set are constants in sce's code, so a different sce commit could change
    # them silently: run only at the pinned commit (checked before anything is written).
    sce_commit = subprocess.run(["git", "-C", str(SCE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert sce_commit == cfg["sce_commit"], f"sce at {sce_commit or 'no checkout'}; config pins {cfg['sce_commit']}"
    # experiments/ is left out: the devbox setup doesn't copy it, and nothing the pilot runs lives there.
    dirty = subprocess.run(["git", "-C", str(SCE), "status", "--porcelain", "--untracked-files=no", "--", ".",
                            ":(exclude)experiments"], capture_output=True, text=True, check=True).stdout
    assert not dirty, f"sce has local changes to tracked files, so the pinned commit isn't what would run:\n{dirty}"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    out = (args.out or SCE / "transcripts" / f"{stamp}_la_real_honest_phase2-{args.model}").resolve()
    assert not out.exists(), f"{out} exists; runs are append-only"
    out.mkdir(parents=True)
    uv = shutil.which("uv", path=f"{Path.home() / '.local/bin'}:{os.environ['PATH']}")

    only = {e for e in args.only.split(",") if e}
    tasks = [t for t in cfg["tasks"] if not only or t["env"] in only]
    assert tasks and only <= {t["env"] for t in cfg["tasks"]}, f"--only {sorted(only)} names no configured environment"
    launched = []
    for task in tasks:
        env_vars = {"LA_ENV": task["env"], "BA_MODE": run["mode"], "PT_ATTACKER": model["policy"],
                    "BA_MAINS": task["main"], "BA_MAX_STEPS": str(run["max_steps"]), "VLLM_BASE_URL": model["base_url"],
                    "SKIP_DOCKER_BUILD": "1" if run["skip_docker_build"] else ""}
        cmd = [uv, "run", "inspect", "eval", "src/runner/tasks/linuxarena_real.py", "--log-dir", str(out / task["env"]),
               "--max-samples", str(run["max_samples"]), "--time-limit", str(run["time_limit_s"]), "--display", "plain"]
        with open(out / f"{task['env']}.log", "w") as log:  # the child keeps its own copy of the file descriptor
            proc = subprocess.Popen(cmd, cwd=SCE, env=os.environ | env_vars, stdout=log, stderr=subprocess.STDOUT)
        launched.append({"task": task, "env_vars": env_vars, "cmd": cmd, "proc": proc, "start": now(), "t0": time.time()})
        print(f"{now()} launched {task['env']} / {task['main']} (pid {proc.pid})", flush=True)

    record = {"model_key": args.model, "model": model, "run": run, "only": sorted(only), "sce_commit": sce_commit,
              "started": launched[0]["start"], "finished": None,
              "tasks": [{"task": i["task"], "env_vars": i["env_vars"], "cmd": i["cmd"], "start": i["start"], "end": None,
                         "process_seconds": None, "exit_code": None} for i in launched]}
    (out / "launch.json").write_text(json.dumps(record, indent=1))  # rewritten as each process ends
    while any(r["exit_code"] is None for r in record["tasks"]):
        time.sleep(5)
        for item, rec in zip(launched, record["tasks"]):
            code = item["proc"].poll()
            if code is None or rec["exit_code"] is not None:
                continue
            rec |= {"end": now(), "process_seconds": round(time.time() - item["t0"], 1), "exit_code": code}
            print(f"{now()} {item['task']['env']} exited {code} after {rec['process_seconds']} s", flush=True)
            (out / "launch.json").write_text(json.dumps(record, indent=1))
    # `inspect eval` can exit 0 with an errored or cancelled log, so each task's outcome is read from its logs.
    from inspect_ai.log import read_eval_log  # noqa: PLC0415 — the sce environment this runs in has it

    for rec in record["tasks"]:
        logs = [read_eval_log(str(p)) for p in sorted((out / rec["task"]["env"]).glob("*.eval"))]
        rec["eval_status"] = [log.status for log in logs]
        rec["sample_errors"] = [s.error.message[:500] for log in logs for s in log.samples or [] if s.error]
        rec["samples"] = sum(len(log.samples or []) for log in logs)
    record["finished"] = now()
    (out / "launch.json").write_text(json.dumps(record, indent=1))
    print(f"-> {out}", flush=True)
    failed = [r["task"]["env"] for r in record["tasks"]
              if r["exit_code"] != 0 or r["eval_status"] != ["success"] or r["sample_errors"] or r["samples"] != 1]
    if failed:
        raise SystemExit(f"tasks without exactly one clean sample: {failed} (see launch.json)")


if __name__ == "__main__":
    main()
