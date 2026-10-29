"""G3 step 7's barrier with its values pinned (09, about 11:3x PT 10-13): run before every simulated-arm launch (render,
extraction, pilots included), and it exits 1 unless the step-6 commit 6d7b1f50 is an ancestor of HEAD and forecast.json, at
that commit and in the working tree, has the acknowledged sha256 4243f2cd...; it then prints the extractor's barrier arguments.
run_extract_g3.py is frozen by sha256 in forecast.json, so it is not edited: it gets these same values on its command line
(--forecast-commit, --forecast-sha256), checks them again in its entrypoint (barrier()) before anything is spawned, and its
workers refuse simulated work without the barrier record and forecast.json bytes that hash to it (sim_gate()).

  python3 experiments/2026-10-13_g3_new_reader/g3_step7_barrier.py && <launch with --forecast-commit 6d7b1f50... --forecast-sha256 4243f2cd...>
"""
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = "6d7b1f50c2c08a5887962b5066681684d6e1ae63"
SHA = "4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77"
root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip())
rel = str((HERE / "forecast.json").relative_to(root))
if subprocess.run(["git", "merge-base", "--is-ancestor", COMMIT, "HEAD"], cwd=root).returncode:
    sys.exit(f"barrier: {COMMIT} is not an ancestor of HEAD")
at = subprocess.run(["git", "show", f"{COMMIT}:{rel}"], cwd=root, capture_output=True, check=True).stdout
now = (HERE / "forecast.json").read_bytes()
if hashlib.sha256(at).hexdigest() != SHA or hashlib.sha256(now).hexdigest() != SHA:
    sys.exit("barrier: forecast.json does not have the acknowledged sha256")
if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=root).returncode:
    sys.exit("barrier: forecast.json differs from HEAD")
import json  # noqa: E402
ext = HERE / "run_extract_g3.py"  # the frozen extractor (Codex): unchanged since the freeze
if hashlib.sha256(ext.read_bytes()).hexdigest() != json.loads(now)["frozen"]["files sha256"]["extractor"]:
    sys.exit("barrier: run_extract_g3.py differs from the frozen extractor")
print(f"barrier ok: {COMMIT[:8]} is an ancestor of HEAD; forecast.json sha256 {SHA[:8]}…; extractor args: --forecast-commit {COMMIT} --forecast-sha256 {SHA}")
