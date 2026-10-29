"""ON BOX 4, read-only: one line of JSON on 4r's main and follow-up for s4r_census_hold.sh: real runs not yet terminal,
whether the main session snms4r is live, DEADLINE_STOP, the census hold marker, other PAUSED markers, the follow-up
session, the run_pilot.py processes whose working directory is the nm harness (one launcher per settings folder), the
harness PAUSE_SIMS, the follow-up's first simulated attempt start, end_main's last log line, and the servers_end argument
of every running end_main 4r process (the release deadline is derived from it: LAST_START - 15 min)."""

import json
import os
import subprocess
from pathlib import Path

MAIN = Path("/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r")
FU = Path("/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r_fu")
HARNESS = "/home/aswin/phase2/nm/attack_check_nm"
EL = Path("/home/aswin/phase2/nm/checks/end_main_s4r.log")
EM = ["/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python", "/home/aswin/phase2/nm/checks/end_main.py", "4r"]


def live(sock: str, name: str) -> bool:
    return subprocess.run(["tmux", "-L", sock, "has-session", "-t", f"={name}"], stderr=subprocess.DEVNULL).returncode == 0


d = json.loads((MAIN / "launch.json").read_text())
pids = subprocess.run(["pgrep", "-f", "run_pilot[.]py"], capture_output=True, text=True).stdout.split()


def launcher(p: str) -> bool:
    """A launcher process proper: an argument that IS run_pilot.py (not the bash -c wrapper whose command string
    mentions it), with the nm harness as its working directory."""
    try:
        argv = Path(f"/proc/{p}/cmdline").read_bytes().decode().split("\0")
        return any(a.endswith("/run_pilot.py") or a == "run_pilot.py" for a in argv) and os.readlink(f"/proc/{p}/cwd") == HARNESS
    except (FileNotFoundError, ProcessLookupError):  # the process exited between pgrep and the read: not a launcher
        return False  # any other error (permission, I/O) propagates: the state is unreadable and the hold stays (Codex r34)


in_harness = [p for p in pids if launcher(p)]
em_ends = []
for p in subprocess.run(["pgrep", "-f", "end_main[.]py 4r "], capture_output=True, text=True).stdout.split():
    try:
        argv = Path(f"/proc/{p}/cmdline").read_bytes().decode().split("\0")[:-1]
    except OSError:
        continue
    if argv[:3] == EM and len(argv) == 8:
        em_ends.append(int(float(argv[6])))
starts = []
if (FU / "launch.json").exists():
    starts = sorted(a["start"] for r in json.loads((FU / "launch.json").read_text())["runs"] if not r["real"]
                    for a in r["attempts"] if a.get("start"))
print(json.dumps({
    "open_real": sum(1 for r in d["runs"] if r["real"] and r["state"] not in ("done", "no policy call", "not run")),
    "main_live": live("nm", "snms4r"), "deadline_stop": (MAIN / "DEADLINE_STOP.txt").exists(),
    "hold": (MAIN / "PAUSED_census_hold.txt").exists(),
    "other_paused": sorted(p.name for p in MAIN.glob("PAUSED_*.txt") if p.name != "PAUSED_census_hold.txt"),
    "fu_live": live("nm", "fnms4r"), "launchers_in_harness": len(in_harness),
    "harness_pause_sims": os.path.exists(f"{HARNESS}/results/PAUSE_SIMS"), "fu_first_sim_start": starts[0] if starts else None,
    "em_last": EL.read_text().splitlines()[-1] if EL.exists() else "", "em_servers_end": sorted(em_ends)}))
