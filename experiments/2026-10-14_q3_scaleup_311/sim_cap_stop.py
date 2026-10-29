"""sim_cap_stop.py (RULE.md Amendment 7, variant iii; Codex round 12: the launcher's admission guard reserves but never
enforces a per-run ceiling). ON THE DEVBOX, tmux -L guards sc311q<N>, every 120 s, no laptop: reads the follow-up run's
simulator spend with spend_now.py (the canary's, reviewed: priced spend; running attempts at max(a fresh reading, the
estimate); a running attempt without a fresh read for 300 s at the worst case) and, ONLY on a fresh reading (the
launcher's last snapshot <= 300 s old) whose total WITHOUT the worst-case charges of unread attempts is at or above CAP
(an unread attempt never blocks the stop and never triggers it), writes the pause files
(<harness>/results/PAUSE_SIMS and <run>/PAUSE_SIMS): NO NEW simulated start; in-flight simulated runs FINISH (0d, 17:15:
a cap stop is clean, never a Ctrl+C for spend). Its threshold is the launcher's admission limit + a margin, a backstop
behind the admission guard, which already admits no new task past the limit. A missing, failed or stale reading is logged and retried, never an
action. Started before the follow-up launch; waits while the run folder has no launch.json.
  <sept venv python> sim_cap_stop.py <follow-up run dir> <follow-up session> <cap usd>
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
HARNESS, CHECKS = Path("/home/aswin/phase2/attack_check_311"), Path("/home/aswin/phase2/s311_checks")
PY, TREE = "/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python", "/home/aswin/phase2/sims-sept-strict-off/src"
run, session, cap = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
log_f = CHECKS / f"sim_cap_stop_{session}.log"


def note(msg: str) -> None:
    with open(log_f, "a") as f:
        print(datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T PT"), msg, file=f, flush=True)


note(f"start: run {run}, session {session}, cap ${cap:.0f}")
seen = False
while True:
    try:
        r = subprocess.run([PY, str(CHECKS / "spend_now.py"), str(run), str(HARNESS), str(CHECKS)], capture_output=True,
                           text=True, timeout=120, env={"PYTHONPATH": TREE, "PATH": "/usr/bin:/bin"})
        out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        assert r.returncode == 0, f"exit {r.returncode}"
    except Exception as e:  # noqa: BLE001 - a failed read is never an action
        note(f"read failed ({e!r}); retrying")
        time.sleep(120)
        continue
    if out == "absent":
        live = subprocess.run(["tmux", "-L", "phase2", "has-session", "-t", f"={session}"], stderr=subprocess.DEVNULL).returncode == 0
        if seen and not live:
            note("the follow-up session is gone; done")
            break
        time.sleep(120)
        continue
    try:
        usd, age, running, worst_charged = out.split()
        usd, age, worst_charged = float(usd), int(float(age)), int(worst_charged)
    except Exception as e:  # noqa: BLE001
        note(f"unparseable reading {out!r} ({e!r}); retrying")
        time.sleep(120)
        continue
    seen = True
    try:  # the worst case spend_now.py charged each unread attempt (the launcher's own, >= the configured $75)
        worst = float(json.loads((run / "launch.json").read_text())["spend"][-1]["worst_per_sim_run_usd"])
    except Exception as e:  # noqa: BLE001
        note(f"no worst-case figure in launch.json ({e!r}); retrying")
        time.sleep(120)
        continue
    # Codex round 13: an unread attempt must not disable the stop. Each one is charged max(live, worst) >= worst, so
    # usd - worst_charged x worst never counts unread spend (a lower bound for those attempts) and acts on the rest.
    known = usd - worst_charged * worst
    fresh = age <= 300
    note(f"simulator ${usd:.2f} (snapshot {age} s old, {running} running, {worst_charged} at the ${worst:.0f} worst case; "
         f"counted without them ${known:.2f}){'' if fresh else ' NOT FRESH'}")
    if fresh and known >= cap:
        for f in (HARNESS / "results" / "PAUSE_SIMS", run / "PAUSE_SIMS"):
            f.touch()
        (run / "SIM_CAP_PAUSE.txt").write_text(f"simulator ${known:.2f} >= ${cap:.0f} (fresh): no new simulated start; "
                                               f"in-flight runs finish\n")
        note(f"SIM CAP PAUSE: ${known:.2f} >= ${cap:.0f}: pause files written; no Ctrl+C, in-flight runs finish")
        break
    if not subprocess.run(["tmux", "-L", "phase2", "has-session", "-t", f"={session}"], stderr=subprocess.DEVNULL).returncode == 0:
        note("the follow-up session is gone; done")
        break
    time.sleep(120)
