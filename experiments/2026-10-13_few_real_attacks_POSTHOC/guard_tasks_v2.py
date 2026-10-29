"""Container guard v2 (G1, 10-13; after v1 stopped step 4 on a one-poll task count of 9 that no container log explained).
It reads the app's state and running task count by app ID (the v2.5 billing watcher's own `state()`). Consecutive polls
start at least 60 s apart (time.monotonic; a slow poll pushes the next one later, never closer).
- Every over-count (tasks > the cap) is logged at once, with the time and the count. Then a second line lists the app's
  containers with their start times (`modal container list --json`, filtered by app ID, 15 s timeout). A failed or slow
  listing is logged, never fatal.
- The guard stops the app only when two consecutive polls, at least 60 s apart, are both over the cap. A poll at or
  under the cap resets it.
- A failed or erroring poll fails closed: ALARM, then stop. This covers an app missing from the list and a task count
  that is not an integer.
- Every stop is the watcher's own `stop()`: `modal app stop <app id> --yes`, retried until the app shows as stopped.
It exits once the app is stopped.

  python3 guard_tasks_v2.py <app id> <max tasks>
"""
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("watch_billing_strict", HERE / "watch_billing_strict.py")  # the v2.5 copy beside it
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)
APART_S, LIST_TIMEOUT_S = 60.0, 15


def stamp() -> str:
    return datetime.now(w.PT).strftime("%H:%M:%S")


def containers(app: str) -> list[str]:
    """This app ID's live containers with their start times, for the log; a failed listing is reported, never raised."""
    try:
        rows = json.loads(w.cli("container", "list", "--json", timeout=LIST_TIMEOUT_S))
        return [f"{c['container_id']} (start {c.get('start_time')})" for c in rows
                if isinstance(c, dict) and c.get("app_id") == app and isinstance(c.get("container_id"), str)]
    except Exception as e:
        return [f"container list failed ({type(e).__name__})"]


def main(app: str, cap: str):
    cap, prev_over = int(cap), None  # prev_over: the start time of the previous poll, if it was over the cap
    while True:
        t = time.monotonic()  # this poll's start
        try:
            st, tasks = w.state(app)
            n = int(tasks)
        except Exception as e:  # fail closed
            w.say(f"{stamp()} ALARM: poll failed ({type(e).__name__}); stopping the app")
            w.stop(app, f"guard: failed poll ({type(e).__name__})")
        w.say(f"{stamp()} state={st} tasks={n}")
        if st in ("stopped", "gone"):
            sys.exit(0)
        if n > cap:
            w.say(f"{stamp()} OVER-COUNT: {n} tasks > {cap}")
            w.say(f"{stamp()} OVER-COUNT containers of {app}: {containers(app)}")
            if prev_over is not None and t - prev_over >= APART_S:
                w.say(f"{stamp()} ALARM: over the cap at two consecutive polls {t - prev_over:.0f} s apart; stopping the app")
                w.stop(app, f"guard: {n} tasks > {cap} at two consecutive polls {t - prev_over:.0f} s apart")
            prev_over = t
        else:
            prev_over = None
        time.sleep(max(0.0, t + APART_S - time.monotonic()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
