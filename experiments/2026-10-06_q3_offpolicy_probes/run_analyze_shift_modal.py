"""Run analyze_shift.py on Modal: one CPU container, jobs one after another, each on all its cores, on score and call
tables saved on the Q3 Volume (scores/, call_tables/). The container gets analyze_shift.py, config/q3.yaml and the exclusions file from the
local checkout, which must be committed: that commit goes into every result. Results, logs and a run record (jobs,
commit, input sha256s, package versions, exit codes) are written to the Volume under analysis/<stamp>/attempt<k>/, a
new folder per run and per retried container; each job's result is saved as soon as it finishes. Copy the result
files into results/ afterwards.

  uv run --no-project --python 3.13 --with 'modal>=1.2' --with pyyaml modal run --detach \
      experiments/2026-10-06_q3_offpolicy_probes/run_analyze_shift_modal.py --jobs config/shift_jobs.yaml
  uv run --no-project --python 3.13 --with 'modal>=1.2' modal volume get \
      probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06 analysis/<stamp> <local dir>
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3_analyze-shift_kill-2026-11-06"
VOL = "probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06"
EXP = "experiments/2026-10-06_q3_offpolicy_probes"
CODE = [f"{EXP}/analyze_shift.py", f"{EXP}/config/q3.yaml", "splits/exclusions_v3.json"]  # q3.yaml names exclusions_v3
REPO = Path("/root/repo")  # the repo layout analyze_shift.py expects, in the container
FOLDER = {"scores_": "scores", "call_table_": "call_tables"}  # data/ file prefix -> Volume folder
# The versions the committed shift results were computed with (local uv environment, 2026-10-06).
PINS = ["numpy==2.5.3", "pandas==3.0.6", "pyarrow==25.0.1", "scikit-learn==1.7.2", "scipy==1.18.1", "pyyaml==6.0.3",
        "joblib==1.6.0"]
CPU, MEMORY = 32, 32768
TIMEOUT = 30 * 60  # all 5 jobs took 5.2 min on 32 cores (2026-10-07_122136); a hang stops at about $1

image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(*PINS)
for rel in CODE:
    image = image.add_local_file(HERE.parent.parent / rel, str(REPO / rel))
app = modal.App(APP)
vol = modal.Volume.from_name(VOL)


@app.function(image=image, volumes={"/vol": vol}, cpu=CPU, memory=MEMORY, timeout=TIMEOUT)
def analyze(jobs: list[dict], commit: str, stamp: str, swap_prob: float) -> dict:
    import hashlib
    import os
    import sys
    import time
    from importlib.metadata import version

    exp, logs = REPO / EXP, Path("/tmp/logs")
    for d in (exp / "data", exp / "results", logs):
        d.mkdir()
    base = Path("/vol/analysis") / stamp
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"attempt{len(list(base.iterdir()))}"  # a retried (preempted) container starts a new attempt folder
    out.mkdir()  # never overwrite: fails if it exists
    inputs = {}
    for job in jobs:
        for rel in (job[k] for k in ("scores", "calls", "turn_mean") if job.get(k)):
            if rel not in inputs:
                src = Path("/vol") / next(f for p, f in FOLDER.items() if Path(rel).name.startswith(p)) / Path(rel).name
                (exp / rel).symlink_to(src)
                inputs[rel] = {"volume_file": f"{VOL}/{src.relative_to('/vol')}", "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}

    # The run record is written before any job starts and rewritten as each job ends, so a partial attempt keeps it.
    record = {"stamp": stamp, "attempt": out.name, "git_commit": commit, "app": APP, "cpu": CPU, "memory_mb": MEMORY,
              "swap_prob": swap_prob, "schedule": "jobs one at a time, each on all cores",
              "python": sys.version.split()[0], "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS},
              "code_sha256": {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in CODE},
              "inputs": inputs, "minutes": None, "runs": [{"job": job, "exit_code": None} for job in jobs]}

    def save():
        (out / f"analyze_shift_run_{stamp}.json").write_text(json.dumps(record, indent=1))
        vol.commit()

    save()
    # One job at a time: each one spreads its (slice, layer) analyses over all CPU cores (joblib n_jobs=-1, capped at
    # the container's cores). Each finished job's log and result are saved before the next starts.
    env = os.environ | {"Q3_GIT_COMMIT": commit, "LOKY_MAX_CPU_COUNT": str(CPU)}
    runs, t0 = record["runs"], time.time()
    for i, job in enumerate(jobs):
        args = [sys.executable, str(exp / "analyze_shift.py"), "--swap-prob", str(swap_prob)]
        for k in ("scores", "calls", "turn_mean", "kind"):
            args += [f"--{k.replace('_', '-')}", job[k]] if job.get(k) else []
        args += ["--subset"] if job.get("subset") else []
        with open(logs / f"job{i}.log", "w") as log:  # the child process keeps its own handle
            p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=exp, env=env)
        while p.poll() is None:
            time.sleep(15)
            print(f"{(time.time() - t0) / 60:.1f} min: job {i} of {len(jobs)} running", flush=True)
        text = (logs / f"job{i}.log").read_text()
        (out / f"job{i}.log").write_text(text)
        last = text.strip().splitlines()[-1:]
        result = Path(last[0].removeprefix("-> ")).name if p.returncode == 0 and last and last[0].startswith("-> ") else None
        if result:
            (out / result).write_bytes((exp / "results" / result).read_bytes())
        runs[i] |= {"exit_code": p.returncode, "log": f"job{i}.log", "result": result, "minutes": round((time.time() - t0) / 60, 1)}
        save()
        print(f"{(time.time() - t0) / 60:.1f} min: job {i} done, exit {p.returncode}, {result}", flush=True)
    record["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    failed = [r for r in runs if not r["result"]]
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(runs)} jobs failed; logs in {VOL}/analysis/{stamp}/{out.name}/: {failed}")
    return record


@app.local_entrypoint()
def main(jobs: str = "config/shift_jobs.yaml", swap_prob: float = 0.5):
    """--swap-prob 0: the paired null's identity check (analyze_shift.paired_null)."""
    import yaml

    root = HERE.parent.parent
    files = CODE + [str((HERE / jobs).relative_to(root)), f"{EXP}/{Path(__file__).name}"]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *files], capture_output=True, text=True, cwd=root, check=True).stdout
    if dirty:
        raise RuntimeError(f"commit these first, the results record the commit:\n{dirty}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root, check=True).stdout.strip()
    spec = yaml.safe_load((HERE / jobs).read_text())["jobs"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"{len(spec)} jobs at {commit[:8]}, swap_prob {swap_prob} -> {VOL}/analysis/{stamp}/", flush=True)
    print(json.dumps(analyze.remote(spec, commit, stamp, swap_prob), indent=1))
