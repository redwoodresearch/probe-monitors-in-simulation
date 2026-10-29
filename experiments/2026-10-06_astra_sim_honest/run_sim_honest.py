"""Launch Astra simulated honest runs: one `inspect eval` per main task, from the September snapshot of the sims code.

Each task gets its own reference (built by run_build_references.py) as both BA_REAL_LOG and BA_HONEST_LOG, and its
own log folder `<out>/runs/<main>/`, as the September fleet ran one job per cell. Settings: `--config`
(default config/astra_honest.json). With `BA_MODE=attack` in the config, each task also gets its design cell
(`BA_CELLS=<main>:<side>`, the cell its reference came from).
Each task folder records the exact inputs it was launched with (`inputs.json`: settings, Inspect arguments, the
reference's path and sha256, the sims tree's commit). Resumable: a task whose folder already holds a finished
one-sample log is skipped, but only if its recorded inputs equal the current ones, and the output folder's
`protocol.json` (shared settings, written by the first launch) must match too; otherwise nothing is launched rather
than mix two protocols in one cohort. Every reference is checked to exist before anything is launched. Per-task Inspect output
goes to `<out>/logs/<main>.log`; one progress line per task goes to `<out>/progress.jsonl`, including failures.
Exits nonzero if any task failed. Ctrl+C, SIGTERM, or any failure of the launcher itself stops the batch: queued tasks are cancelled and running
`inspect eval` processes get SIGINT and, 60 s later, a kill, so nothing keeps spending after the launcher stops.
Inspect may not finish its own cleanup in that window (tested 2026-10-06 without API keys: it kept retrying the
errored sample until killed). Afterwards, on a dedicated machine, remove leftover scratch containers with
`docker ps -aq --filter name=inspect-la- | xargs -r docker rm -f`.

Settings come only from the config: inherited BA_/PT_/LA_/SCE_/INSPECT_ variables are dropped, and the launch stops if
Inspect could auto-load a `.env` (it searches the working folder and its parents). Keys come from the environment,
so run the whole launcher under one `op run`:
  op run --env-file=<.env.tpl> -- uv run --project ~/simulated-control-evals-sept python \\
    experiments/2026-10-06_astra_sim_honest/run_sim_honest.py --references <results/references_*.json> \\
    --refs-dir <built references> --out <run dir> --mains ct_rewrite_in_go,content_encoding_detector --workers 2
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from inspect_ai.log import list_eval_logs, read_eval_log

HERE = Path(__file__).parent
CONFIG: dict = {}  # set from --config in main()
PT = ZoneInfo("America/Los_Angeles")
# Variables that configure the sims code or Inspect. Only the config sets them; an inherited value is dropped.
RUN_PREFIXES = ("BA_", "PT_", "LA_", "SCE_", "INSPECT_")
STOP = threading.Event()  # set when the launcher is interrupted: no new task starts after it
RUNNING: set[subprocess.Popen] = set()
LOCK = threading.Lock()


def finished(run_dir: Path) -> bool:
    """A log with exactly one sample and no error. Judged on the sample, not `results.completed_samples`, which
    counts scored samples and the honest scorer may score nothing."""
    for info in list_eval_logs(str(run_dir)) if run_dir.exists() else []:
        log = read_eval_log(info)
        if log.status == "success" and len(log.samples or []) == 1 and log.samples[0].error is None:
            return True
    return False


def reference_path(ref: dict, refs_dir: Path) -> Path:
    return refs_dir / ref["environment"] / ref["fork_run"] / f"{ref['source_sample'].replace(':', '_')}.eval"


def inputs(main: str, ref: dict, args) -> dict:
    """Everything that defines this task's run. Compared on resume; a mismatch is a different protocol."""
    reference = reference_path(ref, args.refs_dir)
    sims_commit = subprocess.run(["git", "-C", str(args.sims_tree), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
    sims_diff = subprocess.run(["git", "-C", str(args.sims_tree), "diff", "HEAD", "--", "src"],
                               capture_output=True, text=True).stdout
    return {"main": main, "env": CONFIG["env"], "inspect_args": CONFIG["inspect_args"],
            "cell": ref["cell"] if CONFIG["env"]["BA_MODE"] == "attack" else None,
            "settings_dir": str(args.settings_dir),
            "reference": str(reference), "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "sims_commit": sims_commit, "sims_src_diff_sha256": hashlib.sha256(sims_diff.encode()).hexdigest()}


def with_defaults(record: dict, args) -> dict:
    """A record written before a field existed, read with that field's historical value (the first launcher had no
    --config, --settings-dir or attack mode)."""
    return {"config": "astra_honest.json", "cell": None, "settings_dir": str((args.sims_tree / ".settings").resolve())
            } | record


def run_one(main: str, ref: dict, args) -> dict:
    run_dir = args.out / "runs" / main
    want = inputs(main, ref, args)
    recorded = run_dir / "inputs.json"
    if recorded.exists():
        had = with_defaults(json.loads(recorded.read_text()), args)
        had.pop("config", None)
        if had != want:
            changed = sorted(k for k in want if had.get(k) != want[k])
            return {"main": main, "status": f"failed (inputs changed since the existing run: {changed})"}
        if finished(run_dir):
            return {"main": main, "status": "skipped (finished)"}
    else:
        assert not finished(run_dir), f"{run_dir} has a finished log but no inputs.json; cannot tell what produced it"
        run_dir.mkdir(parents=True, exist_ok=True)
        recorded.write_text(json.dumps(want, indent=1))
    inherited = {k: v for k, v in os.environ.items() if not k.startswith(RUN_PREFIXES)}
    env = inherited | CONFIG["env"] | {
        "LA_ENV": ref["environment"], "BA_REAL_LOG": want["reference"], "BA_HONEST_LOG": want["reference"],
        "CONTROL_TOWER_SETTINGS_DIR": str(args.settings_dir),
    } | ({"BA_CELLS": ref["cell"]} if CONFIG["env"]["BA_MODE"] == "attack" else {})
    # The snapshot's own Inspect, started directly (not through `uv run`) so an interrupt reaches it.
    cmd = [str(args.sims_tree / ".venv" / "bin" / "inspect"), "eval", "src/runner/tasks/linuxarena_sim.py",
           "--log-dir", str(run_dir), *CONFIG["inspect_args"]]
    start = time.time()
    with open(args.out / "logs" / f"{main}.log", "a") as log, LOCK:
        if STOP.is_set():
            return {"main": main, "status": "failed (launcher interrupted before this task started)"}
        proc = subprocess.Popen(cmd, cwd=args.sims_tree, env=env, stdout=log, stderr=subprocess.STDOUT)
        RUNNING.add(proc)
    code = proc.wait()
    with LOCK:
        RUNNING.discard(proc)
    return {"main": main, "status": "finished" if code == 0 and finished(run_dir) else f"failed (exit {code})",
            "seconds": round(time.time() - start), "reference": want["reference"]}


def stop_batch(pool: ThreadPoolExecutor) -> None:
    """No queued task may start; running evals get SIGINT, then a kill at one shared 60 s deadline."""
    STOP.set()
    with LOCK:
        running = list(RUNNING)
    for proc in running:  # SIGINT lets Inspect write its log and remove its sandbox containers
        proc.send_signal(signal.SIGINT)
    deadline = time.time() + 60
    for proc in running:
        try:
            proc.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            pass
    for proc in running:
        if proc.poll() is None:
            proc.kill()
    pool.shutdown(wait=True, cancel_futures=True)


def interrupted(*_):
    """SIGTERM handler: stop the batch the same way Ctrl+C does."""
    raise KeyboardInterrupt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--refs-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mains", help="comma-separated main task ids; default: every covered main task")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--sims-tree", type=Path, default=Path.home() / "simulated-control-evals-sept")
    p.add_argument("--settings-dir", type=Path, help="env checkouts (default: <sims tree>/.settings)")
    p.add_argument("--config", type=Path, default=HERE / "config" / "astra_honest.json")
    args = p.parse_args()
    CONFIG.update(json.loads(args.config.read_text()))
    args.settings_dir = (args.settings_dir or args.sims_tree / ".settings").resolve()
    # Inspect runs in the sims tree, so every path it is handed must be absolute.
    args.refs_dir, args.out, args.sims_tree = args.refs_dir.resolve(), args.out.resolve(), args.sims_tree.resolve()
    refs = json.loads(args.references.read_text())["references"]
    mains = args.mains.split(",") if args.mains else sorted(refs)
    mode = CONFIG["env"]["BA_MODE"]
    assert mode in ("honest", "attack"), mode
    dotenvs = [d / ".env" for d in [args.sims_tree, *args.sims_tree.parents] if (d / ".env").exists()]
    assert not dotenvs, f"Inspect would auto-load {dotenvs[0]} and could change the protocol; move it away first"
    missing = [m for m in mains if not reference_path(refs[m], args.refs_dir).exists()]
    assert not missing, f"{len(missing)} task(s) have no built reference, so nothing was launched: {missing[:10]}"
    # One run, one protocol. The cohort's shared settings are fixed by the first launch into this folder
    # (protocol.json), so a later launch of another subset of tasks cannot change them either.
    protocol = {k: v for k, v in inputs(mains[0], refs[mains[0]], args).items()
                if k in ("env", "inspect_args", "sims_commit", "sims_src_diff_sha256", "settings_dir")
                } | {"config": str(args.config.name)}
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        had = with_defaults(json.loads(protocol_path.read_text()), args)
        had.pop("cell", None)
        assert had == protocol, (f"{protocol_path} records another protocol, so nothing was launched: "
                                 f"{sorted(k for k in protocol if had.get(k) != protocol[k])} differ")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(protocol, indent=1))
    # And each task folder must match its own recorded inputs (its reference included).
    recorded = {m: args.out / "runs" / m / "inputs.json" for m in mains}
    changed = [m for m, path in recorded.items()
               if path.exists() and {k: v for k, v in with_defaults(json.loads(path.read_text()), args).items()
                                     if k != "config"} != inputs(m, refs[m], args)]
    assert not changed, f"{len(changed)} task(s) were run with other inputs, so nothing was launched: {changed[:10]}"
    (args.out / "logs").mkdir(parents=True, exist_ok=True)

    # The requested task list, recorded before anything is dispatched, so a task that never started still counts.
    with open(args.out / "requested.jsonl", "a") as f:
        f.write(json.dumps({"at": datetime.now(PT).isoformat(timespec="seconds"), "references": str(args.references),
                            "mains": mains}) + "\n")
    signal.signal(signal.SIGTERM, interrupted)
    failed = 0
    pool = ThreadPoolExecutor(args.workers)
    try:
        with open(args.out / "progress.jsonl", "a") as progress:
            futures = {pool.submit(run_one, m, refs[m], args): m for m in mains}
            for future in as_completed(futures):
                try:  # a worker that raised is recorded as a failed task, and reporting goes on for the rest
                    row = future.result()
                except Exception as e:
                    row = {"main": futures[future], "status": f"failed ({type(e).__name__}: {e})"}
                row |= {"at": datetime.now(PT).isoformat(timespec="seconds")}
                failed += row["status"].startswith("failed")
                progress.write(json.dumps(row) + "\n")
                progress.flush()
                print(json.dumps(row), flush=True)
    except BaseException as e:  # an interrupt, or the coordinator itself failing: either way, stop all spending
        stop_batch(pool)
        print(f"stopped ({type(e).__name__}): queued tasks cancelled, running evals terminated", flush=True)
        raise
    pool.shutdown(wait=True)
    print(f"{len(mains)} tasks, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
