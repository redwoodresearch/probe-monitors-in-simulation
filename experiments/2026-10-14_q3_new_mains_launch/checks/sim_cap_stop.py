"""sim_cap_stop.py (RULE.md Amendment 7, variant iii; Codex round 12: the launcher's admission guard reserves but never
enforces a per-run ceiling). ON THE DEVBOX, tmux -L nmguards scnms<N>, every 120 s, no laptop: reads the follow-up run's
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
HARNESS, CHECKS = Path("/home/aswin/phase2/nm/attack_check_nm"), Path("/home/aswin/phase2/nm/checks")
PY, TREE = "/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python", "/home/aswin/phase2/sims-sept-strict-off/src"
run, session, cap = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
# New-mains copy (Codex r1 and r2 item 6; 0d 18:42). Two additions, both on top of the clean pause above:
#  - HARD ceiling: on a FRESH snapshot whose total INCLUDING unread attempts (each at max(its last reading, the worst
#    case), the figure the launcher's own admission guard reserves for it) is at or above HARD, one graceful Ctrl+C to the
#    follow-up session (in-flight runs cut, recorded "stopped": a deliberate exception to "clean stops only"). Never on a
#    stale, failed or missing snapshot (0d: those are never a Ctrl+C).
#  - BLIND pause, reversible: while no fresh reading has come for BLIND_S (stale snapshot; failed or unparseable reads,
#    from the first one once the run folder exists; launch.json gone after a reading), or while a fresh one has BLIND_N or
#    more running attempts unread for 1,200 s, <run>/PAUSE_SIMS: no new simulated start (in-flight runs continue). It
#    LATCHES (Amendment 4): never removed here, since the usable gate's pause shares the file; a person may lift it. Before the run folder exists nothing can start, so
#    nothing is paused (and the folder is never created here: the launch requires it absent). Every reading is box-local:
#    spend_now.py reads launch.json and the harness's run_spend() (the attempts' Inspect logs and sample buffers on this box).
hard = float(sys.argv[4])
assert hard > cap, (cap, hard)
BLIND_S, BLIND_N = 900, 3
paused = blind_paused = False
blind_since = None
log_f = CHECKS / f"sim_cap_stop_{session}.log"


def note(msg: str) -> None:
    with open(log_f, "a") as f:
        print(datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T PT"), msg, file=f, flush=True)


def blind(why: str) -> None:  # this loop's reading can't be trusted: never an action except, after BLIND_S, no new start
    global blind_since, blind_paused
    if not run.is_dir():  # no launch yet: nothing can start
        return
    blind_since = blind_since or time.time()
    if not blind_paused and not paused and (time.time() - blind_since >= BLIND_S or why.startswith("fresh")):
        (run / "PAUSE_SIMS").touch()
        (run / "SIM_BLIND_PAUSE.txt").write_text(f"{why}: no new simulated start until a fresh, read reading\n")
        note(f"SIM BLIND PAUSE ({why}): <run>/PAUSE_SIMS written; in-flight runs continue")
        blind_paused = True


def seeing() -> None:  # Amendment 4 (Codex r5): the blind pause LATCHES. <run>/PAUSE_SIMS is shared with the usable
    global blind_since  # gate's permanent pause, so this script never removes it (a person may); no race on that file
    blind_since = None


note(f"start: run {run}, session {session}, pause at ${cap:.0f}, hard stop at ${hard:.0f}")
seen = False
while True:
    try:
        r = subprocess.run([PY, str(CHECKS / "spend_now.py"), str(run), str(HARNESS), str(CHECKS)], capture_output=True,
                           text=True, timeout=120, env={"PYTHONPATH": TREE, "PATH": "/usr/bin:/bin"})
        out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        assert r.returncode == 0, f"exit {r.returncode}"
    except Exception as e:  # noqa: BLE001 - a failed read is never a stop
        note(f"read failed ({e!r}); retrying")
        blind("reads failing")
        time.sleep(120)
        continue
    if out == "absent":
        live = subprocess.run(["tmux", "-L", "nm", "has-session", "-t", f"={session}"], stderr=subprocess.DEVNULL).returncode == 0
        if seen and not live:
            note("the follow-up session is gone; done")
            break
        if seen:  # launch.json gone after a reading: no reading at all
            blind("launch.json missing")
        time.sleep(120)
        continue
    try:
        usd, age, running, worst_charged, long_unread = out.split()
        usd, age, worst_charged, long_unread = float(usd), int(float(age)), int(worst_charged), int(long_unread)
    except Exception as e:  # noqa: BLE001
        note(f"unparseable reading {out!r} ({e!r}); retrying")
        blind("unparseable readings")
        time.sleep(120)
        continue
    seen = True
    try:  # the worst case spend_now.py charged each unread attempt (the launcher's own, >= the configured $75)
        worst = float(json.loads((run / "launch.json").read_text())["spend"][-1]["worst_per_sim_run_usd"])
    except Exception as e:  # noqa: BLE001
        note(f"no worst-case figure in launch.json ({e!r}); retrying")
        blind("no worst-case figure")
        time.sleep(120)
        continue
    # Codex round 13: an unread attempt must not disable the clean pause. Each one is charged max(live, worst) >= worst,
    # so usd - worst_charged x worst never counts unread spend (a lower bound for those attempts) and acts on the rest.
    known = usd - worst_charged * worst
    fresh = age <= 300
    note(f"simulator ${usd:.2f} (snapshot {age} s old, {running} running, {worst_charged} at the ${worst:.0f} worst case, "
         f"{long_unread} unread >= 1,200 s; counted without them ${known:.2f}){'' if fresh else ' NOT FRESH'}")
    if not fresh:
        blind("stale snapshot")
    elif long_unread >= BLIND_N:
        blind(f"fresh, {long_unread} attempts unread >= 1,200 s")
    else:
        seeing()
    if fresh and known >= cap and not paused:
        for f in (HARNESS / "results" / "PAUSE_SIMS", run / "PAUSE_SIMS"):
            f.touch()
        (run / "SIM_CAP_PAUSE.txt").write_text(f"simulator ${known:.2f} >= ${cap:.0f} (fresh): no new simulated start; "
                                               f"in-flight runs finish\n")
        note(f"SIM CAP PAUSE: ${known:.2f} >= ${cap:.0f}: pause files written; in-flight runs continue up to ${hard:.0f}")
        paused = True
    if fresh and usd >= hard:
        (run / "SIM_HARD_STOP.txt").write_text(f"simulator ${usd:.2f} (unread attempts at >= ${worst:.0f}) >= ${hard:.0f} "
                                               f"(fresh): one Ctrl+C to {session}; cut runs are cap_stopped\n")
        subprocess.run(["tmux", "-L", "nm", "send-keys", "-t", f"={session}:", "C-c"])
        note(f"SIM HARD STOP: ${usd:.2f} >= ${hard:.0f}: one Ctrl+C sent to {session}")
        break
    if not subprocess.run(["tmux", "-L", "nm", "has-session", "-t", f"={session}"], stderr=subprocess.DEVNULL).returncode == 0:
        note("the follow-up session is gone; done")
        break
    time.sleep(120)
