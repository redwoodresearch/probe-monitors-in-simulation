"""gate_pause_311_rl.py: gate_pause_311.py (sha256 aeb22a757d334441..., aswin/q3-scaleup-311 @ ba21dea16, the copy in
~/phase2/s311_checks on the IL boxes) with ONE change, for the IL completion launches only (d9 02:50, 0d's GO; 09 registers):
a simulated run whose failure is an infrastructure rate limit under 09's Append 35 evidence rule - it has attempts, EVERY
attempt has terminal sample errors, and every terminal error is RateLimitError (the class a RetryError wraps, as
sim_ratelimit_list.py @ 36a2047fa reads it) - is counted as "infra_rl" and left out of that arm's ended and usable counts.
No other error is ever excluded; real runs are never excluded. Logged per arm as "infra_rl N". Everything else verbatim.

gate_pause_311.py (RULE.md Amendment 10; 0d via d9, about 19:38 PT: in-job gates PAUSE and never kill): the follow-ups'
usable gate, replacing gate1_311b.py there. ON THE DEVBOX, tmux -L guards gp311q<N>, every 120 s while the follow-up's
launcher session exists and its launch hasn't finished: per arm, k usable of n ended runs (gate1_311b.py's counts(),
copied verbatim); any arm with >= 20 ended runs and usable <= 75% (inclusive, integers: 4 x usable <= 3 x ended) ->
<run>/PAUSE_SIMS (the patched launcher starts no new simulated run while it exists) and <run>/GATE_PAUSE.txt. Never a
Ctrl+C: in-flight runs finish. It never removes a pause. One line per check to <run>/gate_pause_watch.log.
  <sept venv python> gate_pause_311.py <follow-up run dir> <follow-up session>
"""

import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ENDED = ("done", "no policy call")
run, session = Path(sys.argv[1]), sys.argv[2]


def terminal_class(e: str) -> str:
    """sim_ratelimit_list.py's cls(): the exception a RetryError wraps ("raised X"), else the error's class."""
    m = re.search(r"raised (\w+)", e)
    return (m.group(1) if m else e.split("(")[0])[:60]


def infra_rl(r: dict) -> bool:
    """Append 35: a simulated run whose every attempt failed with terminal sample errors that are all RateLimitError."""
    if r["real"] or not r.get("attempts"):
        return False
    errs = [[terminal_class(str(e)) for e in ((a.get("outcome") or {}).get("sample_errors") or [])] for a in r["attempts"]]
    return all(errs_a and set(errs_a) == {"RateLimitError"} for errs_a in errs)


# arm_of, run_usable and counts: copied verbatim from gate1_311b.py (the same usable definition), except counts' infra_rl
def arm_of(r: dict) -> str:
    if r["arm"] == "attack":
        return "attack r2" if r["task"].endswith("_r2") else "attack r1"
    return r["arm"] if r["real"] else f"{r['arm']}@{r['simulator']}"


def run_usable(r: dict) -> bool:
    finals = [a for a in r["attempts"] if a.get("outcome") and not a.get("redrawn_because")]
    calls = finals[-1]["outcome"]["policy_calls"] if finals else []
    return bool(calls) and all(c["usable"] for c in calls)


def counts(launch: dict) -> dict:
    """Per arm: runs, distinct ended runs ("ended") and the usable ones among them ("usable")."""
    out = {}
    for r in launch["runs"]:
        c = out.setdefault(arm_of(r), Counter(runs=0, ended=0, usable=0, infra_rl=0))  # explicit zeros survive dict()
        c["runs"] += 1
        if r["state"] in ENDED and not run_usable(r) and infra_rl(r):
            c["infra_rl"] += 1  # Append 35 infrastructure: neither ended nor usable for the gate
        elif r["state"] in ENDED:
            c["ended"] += 1
            c["usable"] += run_usable(r)
    return {k: dict(v) for k, v in sorted(out.items())}


def note(msg: str) -> None:
    """Logging that can never stop the watch (Codex round 19): a failed log write falls back to stdout, then to nothing."""
    line = f"{datetime.now(ZoneInfo('America/Los_Angeles')):%F %T PT} {msg}"
    try:
        with open(run / "gate_pause_watch.log", "a") as f:
            print(line, file=f, flush=True)
    except Exception:  # noqa: BLE001
        try:
            print(line, flush=True)
        except Exception:  # noqa: BLE001
            pass


def alive() -> bool:
    """True while the launcher's session exists. Only tmux's confirmed absence ("can't find session" or "no server
    running", exit 1) ends the watch; any other result (an exception, a signal, another exit code) is logged and retried
    in 30 s (Codex rounds 20 and 22)."""
    while True:
        try:
            p = subprocess.run(["tmux", "-L", "phase2", "has-session", "-t", f"={session}"], capture_output=True, text=True,
                               timeout=60)
            if p.returncode == 0:
                return True
            if p.returncode == 1 and ("can't find session" in p.stderr or "no server running" in p.stderr):
                return False
            note(f"liveness probe inconclusive (exit {p.returncode}: {p.stderr.strip()[:120]}); retrying in 30 s")
        except Exception as e:  # noqa: BLE001
            note(f"liveness probe failed ({type(e).__name__}); retrying in 30 s")
        time.sleep(30)


paused = False
while alive():
    try:  # any read or count error is a wait, never an exit (Codex round 19): the watch keeps guarding
        launch = json.loads((run / "launch.json").read_text())
        finished = bool(launch.get("finished"))
        c = {} if finished else counts(launch)
    except Exception as e:  # noqa: BLE001 - not written yet, caught mid-write, or unreadable: next check
        note(f"launch.json unreadable or not countable ({type(e).__name__}); next check")
        time.sleep(30)
        continue
    if finished:
        note("launch finished")
        break
    note("; ".join(f"{k}: ended {v['ended']} usable {v['usable']} infra_rl {v['infra_rl']}" for k, v in c.items()) + (" (PAUSED)" if paused else ""))
    low = [k for k, v in c.items() if v["ended"] >= 20 and 4 * v["usable"] <= 3 * v["ended"]]
    if low and not paused:
        why = f"usable <= 75% in {low}: " + "; ".join(f"{k} {c[k]['usable']} of {c[k]['ended']}" for k in low)
        try:
            (run / "PAUSE_SIMS").touch()
            paused = True
        except Exception as e:  # noqa: BLE001 - not paused: retried at the next check
            note(f"GATE PAUSE FAILED to write PAUSE_SIMS ({e!r}); retrying at the next check")
        if paused:
            try:
                (run / "GATE_PAUSE.txt").write_text(f"{datetime.now(ZoneInfo('America/Los_Angeles')):%F %T PT} {why}\n")
            except Exception as e:  # noqa: BLE001 - the pause itself is in place
                note(f"could not write GATE_PAUSE.txt ({e!r})")
            note(f"GATE PAUSE: {why}; PAUSE_SIMS written, no Ctrl+C, in-flight runs finish")
    time.sleep(120)
note(f"session {session} gone or launch finished")
