"""Fill the matched null's missing columns on the deployed runner (POST HOC, RULE.md).

The first launch (runner d710f66f, tag d710f66f) fit 7 columns per H100 at once; some threads ran out of GPU memory (a
column holds about 10 GB) and the laptop client then stopped the app on an exception it could not unpickle. Its finished
columns are checkpointed. This calls the deployed runner, unchanged since d710f66f, with each job's missing columns in
chunks of 3, and passes the runner file's last commit, so every column carries the first launch's provenance.

  MNULL_FAMILY=wd NULL_IN=<folder with d9's scores npz> PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' \\
      modal deploy experiments/2026-10-13_matched_world_null_POSTHOC/run_matched_null.py
  MNULL_FAMILY=wd uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy python \\
      experiments/2026-10-13_matched_world_null_POSTHOC/fill_matched_null.py <tag>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_matched_null.py"
FAMILY = os.environ["MNULL_FAMILY"]
APP = f"probe-shift_aswin-surya_sub-mnull-{FAMILY}_fit_kill-2026-11-13"
JOBS = {"wd": ("wd48", "wd63"), "q1b": ("q1b_avg", "q1b_lt")}
PER, N_COLS = 3, 21


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()


if __name__ == "__main__":
    tag = sys.argv[1]
    if git("status", "--porcelain", "--", str(RUNNER), __file__):
        raise SystemExit("commit the runner and this launcher first")
    commit = git("log", "-1", "--format=%H", "--", str(RUNNER))  # the code that runs, as the first launch recorded it
    found = modal.Function.from_name(APP, "summarize").remote(tag, FAMILY)
    todo = []
    for job in JOBS[FAMILY]:
        miss = [c for c in range(N_COLS) if c not in found[job]["columns_found"]]
        todo += [(job, miss[i:i + PER]) for i in range(0, len(miss), PER)]
    run = modal.Function.from_name(APP, "run")
    calls = [(job, ch, run.spawn(job, ch, tag, commit)) for job, ch in todo]
    print(json.dumps({"app": APP, "commit": commit, "calls": [[job, ch, c.object_id] for job, ch, c in calls]}), flush=True)
    failed = 0
    for job, ch, c in calls:
        try:
            print(json.dumps(c.get()), flush=True)
        except Exception as e:  # a remote error (e.g. CUDA OOM) may not unpickle without torch here: count and print each, exit 1
            failed += 1
            print(f"FAILED {job} {ch}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    print(json.dumps({"failed_calls": failed}), flush=True)
    sys.exit(1 if failed else 0)
