"""Fresh-task honest runs, devbox side ($0; 0d's option (c) after stage 5's CHAIN_STOP): does each fresh bug-fix task's
mutation.patch apply to its environment's image? The run-time setup is control_tower's bug_fix.make_bug_fix_setup:
`cd <patch_dir> && patch -p1` in the environment's main container, where patch_dir = "/" if the environment config has
service_mounts, else container_code_path or "/app". Here, for every fresh task with a mutation.patch: the files the patch
touches are copied out of the pinned image (`docker create` + `docker cp`, nothing is run) into a scratch tree at their
container paths, and the host's `patch -p1 --dry-run` runs there. Calibration: stages 1-5's patch tasks have known
outcomes (the run lists' 'no policy call' setup failures); stages 6-7 are the prediction. Booleans and counts only.

  ~/sims/simulated-control-evals-sept/.venv/bin/python dryrun_patches.py --out-dir <dir> --failed <comma-separated mains>
"""
import argparse
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from control_tower.ct_runs.mtgen_bugfix.engine import get_environment_configs

SETTINGS = Path("/home/aswin/sims/simulated-control-evals-sept/.settings/linuxarena")
TASKS = Path(__file__).parent / "config" / "pilot_tasks_fresh_honest.json"
IMAGES = Path("/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json")


def touched(patch: str) -> list[str]:
    """Paths (after -p1) of the files a unified diff changes; /dev/null sides skipped."""
    out = []
    for m in re.finditer(r"^(?:---|\+\+\+) (\S+)", patch, re.M):
        p = m.group(1)
        if p != "/dev/null" and "/" in p:
            out.append(p.split("/", 1)[1])
    return sorted(set(out))


def main(out_dir: Path, failed: set):
    cells = json.loads(TASKS.read_text())["cells"]
    stage_of = {}
    for c in cells:
        if c["stage"] not in ("smoke", "parity"):
            stage_of.setdefault((c["env"], c["main"]), c["stage"])
    commits = json.loads(IMAGES.read_text())["env_commits"]
    configs = get_environment_configs()
    rows, containers = [], {}
    try:
        for (env, main), stage in sorted(stage_of.items(), key=lambda x: (x[1], x[0])):
            pf = SETTINGS / env / "main_tasks" / main / "mutation.patch"
            if not pf.is_file():
                continue
            cfg = configs[env]
            patch_dir = "/" if cfg.service_mounts else (cfg.container_code_path or "/app")
            image = f"linuxarena/env-{env}:{commits[env]}"
            if env not in containers:
                cid = subprocess.run(["docker", "create", image], capture_output=True, text=True, check=True).stdout.strip()
                containers[env] = cid
            patch = pf.read_text()
            files = touched(patch)
            with tempfile.TemporaryDirectory() as tmp:
                missing = 0
                for f in files:
                    src = str(Path(patch_dir) / f)
                    dst = Path(tmp) / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    r = subprocess.run(["docker", "cp", f"{containers[env]}:{src}", str(dst)], capture_output=True, text=True)
                    missing += r.returncode != 0
                r = subprocess.run(["patch", "-p1", "--dry-run", "-d", tmp], input=patch, capture_output=True, text=True)
            rows.append({"stage": stage, "env": env, "main": main, "patch_dir": patch_dir, "files": len(files), "files missing in image": missing,
                         "applies": r.returncode == 0,
                         "known setup failure (stages 1-5)": (main in failed) if stage in "12345" else None})
    finally:
        for cid in containers.values():
            subprocess.run(["docker", "rm", cid], capture_output=True)
    cal = [r for r in rows if r["stage"] in "12345"]
    agree = sum((not r["applies"]) == r["known setup failure (stages 1-5)"] for r in cal)
    pred = [r for r in rows if r["stage"] in ("6", "7")]
    out = {"label": "mutation.patch dry-run against the pinned images (POST HOC check; $0)",
           "calibration, stages 1-5": {"patch tasks": len(cal), "dry-run agrees with the known outcome": agree,
                                       "known failures": sum(bool(r["known setup failure (stages 1-5)"]) for r in cal),
                                       "dry-run failures": sum(not r["applies"] for r in cal)},
           "prediction, stages 6-7": {"patch tasks": len(pred), "predicted setup failures (tasks)": sum(not r["applies"] for r in pred),
                                      "predicted failing runs (2 per task)": 2 * sum(not r["applies"] for r in pred),
                                      "predicted failing tasks": [(r["env"], r["main"], r["stage"]) for r in pred if not r["applies"]]},
           "rows": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dst := out_dir / f"dryrun_patches_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--failed", required=True)
    a = ap.parse_args()
    main(a.out_dir, set(a.failed.split(",")))
