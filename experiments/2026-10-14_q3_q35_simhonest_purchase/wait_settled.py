"""The barrier between A and the continuation (RULE.md, amendment 7): exit 0 only when A can no longer write a compose
file. Runs ON THE DEVBOX BEFORE the continuation's servers and watchers start.

A launcher writes an environment's compose file in its reference setup, and each run's sandbox writes it at start; a run
is redrawn (started again) only after a pre-start failure, never once its policy has been called (D5, run_pilot.py:
468-473). So A is settled when its launcher has FINISHED (launch.json "finished" set), or when every simulated run of A
is either ended (state done, no policy call, or not run) or running with at least one policy model call in its live
sample buffer (its sandbox is up: compose written, no redraw possible). A waiting or ready run (a redraw included) is
not settled. If A's tmux session has ended but launch.json is not finished, the launcher died: exit 2 (report; nothing
starts). Exit 2 on timeout too. Prints counts only; no transcript content is read beyond event types and model names.
  python3 wait_settled.py <A run dir> <A tmux session> [timeout s, default 7200]
"""

import glob
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

BUFFERS = Path.home() / ".local" / "share" / "inspect_ai" / "samplebuffer"  # as run_pilot.py:62


def policy_called_live(log_dir: Path, policy: str) -> bool:
    for ev in log_dir.glob("*.eval"):
        for db in glob.glob(str(BUFFERS / "*" / f"{glob.escape(ev.stem)}*.db")):
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                try:
                    rows = con.execute("select data from events").fetchall()
                finally:
                    con.close()
            except sqlite3.Error:
                continue
            for (data,) in rows:
                d = json.loads(data)
                if d.get("event") == "model" and d.get("model") == policy:
                    return True
    return False


run, session = Path(sys.argv[1]).expanduser(), sys.argv[2]
deadline = time.time() + (float(sys.argv[3]) if len(sys.argv) > 3 else 7200)
while time.time() < deadline:
    try:
        rec = json.loads((run / "launch.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):  # not written yet, or caught mid-write
        rec = None
    if rec and rec.get("finished"):
        print(f"{run.name}: the launcher has finished: settled")
        sys.exit(0)
    tmux = subprocess.run(["tmux", "-L", "phase2", "has-session", "-t", f"={session}"], capture_output=True)
    live = tmux.returncode == 0
    if not live:
        print(f"{run.name}: session {session} is gone but launch.json is not finished: NOT settled (report)")
        sys.exit(2)
    if rec and rec["runs"]:
        sims = [r for r in rec["runs"] if not r["real"]]
        waiting = sum(r["state"] in ("waiting", "ready") for r in sims)
        running = [r for r in sims if r["state"] == "running"]
        started = sum(policy_called_live(Path(r["attempts"][-1]["log_dir"]), rec["policy"]) for r in running)
        if not waiting and started == len(running):
            print(f"{run.name}: settled ({len(sims)} simulated runs; {len(running)} running, each with a policy call; "
                  f"none waiting or ready)")
            sys.exit(0)
    time.sleep(30)
print(f"{run.name}: not settled in time (report; nothing starts)")
sys.exit(2)
