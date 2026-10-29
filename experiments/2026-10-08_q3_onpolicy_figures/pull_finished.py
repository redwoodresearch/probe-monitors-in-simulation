"""Copy each finished run of a Phase 2 run folder from the devbox to this laptop, read-only, as it finishes.

Every `--interval` seconds: copy launch.json (rewritten in place on the devbox, so a copy that does not parse is retried),
then rsync each attempt folder of every run whose state is terminal ("done", "no policy call", "stopped", "not run"),
once. Pulled folders are listed in `<dest>/<name>/.pulled.json`. Stops when every run is terminal. Uses the a1 session's
devbox.sh (SSM ssh with the laptop's own session, no agent forwarding); nothing is written on the devbox.

  DEVBOX=<user>@<instance> python pull_finished.py --devbox-sh <path>/devbox.sh --name <run folder> --dest <dir>
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

TERMINAL = {"done", "no policy call", "stopped", "not run"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devbox-sh", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    if "DEVBOX" not in os.environ:
        raise SystemExit("set DEVBOX=<user>@<instance>")
    dest = Path(args.dest) / args.name
    dest.mkdir(parents=True, exist_ok=True)
    done_file = dest / ".pulled.json"
    pulled = set(json.loads(done_file.read_text())) if done_file.exists() else set()
    rsync = lambda src, dst: subprocess.run([args.devbox_sh, "rsync", "-az", src, dst], capture_output=True, text=True)
    while True:
        tmp = dest / ".launch.json.tmp"
        r = rsync(f"box:phase2/runs/{args.name}/launch.json", str(tmp))
        try:
            if r.returncode:
                raise RuntimeError(r.stderr.strip()[-300:])
            launch = json.loads(tmp.read_text())
        except (RuntimeError, json.JSONDecodeError) as e:
            print(f"{time.strftime('%H:%M:%S')} launch.json not read ({e}); retrying", flush=True)
            time.sleep(30)
            continue
        tmp.replace(dest / "launch.json")
        runs = launch["runs"]
        for run in runs:
            if run.get("state") not in TERMINAL:
                continue
            for a in run.get("attempts") or []:
                ld = a.get("log_dir") or ""
                if f"/{args.name}/" not in ld:
                    continue
                rel = ld.split(f"/{args.name}/", 1)[1].rstrip("/")
                if rel in pulled:
                    continue
                (dest / rel).mkdir(parents=True, exist_ok=True)
                r = rsync(f"box:phase2/runs/{args.name}/{rel}/", f"{dest / rel}/")
                if r.returncode:
                    print(f"{time.strftime('%H:%M:%S')} pull failed {rel}: {r.stderr.strip()[-300:]}", flush=True)
                    continue
                pulled.add(rel)
                done_file.write_text(json.dumps(sorted(pulled)))
                print(f"{time.strftime('%H:%M:%S')} pulled {rel} ({run['arm']} {run['task']} {run['state']})", flush=True)
        n_term = sum(r.get("state") in TERMINAL for r in runs)
        print(f"{time.strftime('%H:%M:%S')} {n_term}/{len(runs)} runs terminal, {len(pulled)} attempt folders pulled", flush=True)
        if n_term == len(runs):
            print("all runs terminal", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
