"""Container guard v2.7 (q2, 10-14, after the 15:19 incident where a stale -> stop rule killed a healthy server). Same CLI
as v2.6c. It stops the app ONLY on fresh evidence: two consecutive SUCCESSFUL polls, 60-600 s apart on the wall clock,
both reading a task count over the max. A failed, slow or unparseable poll is logged (ALERT), never stops the app, and
breaks the over-count pair (so a laptop sleep or network outage between two readings can never complete it). Like v2.6c it
has no duplicate-writer logic of its own: that check (two LIVE containers on one input) reads the app logs, outside it.
This guard is a CONVENIENCE: spend is bounded by the job's `timeout=` and in-job gates; laptop sleep must never stop a job.
It uses the v2.7 billing watcher beside it for state() and stop() (by app ID, retried until stopped).
  python3 guard_tasks_v27.py <app id> <max tasks>"""
import importlib.util
import json
import sys
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("watch_v27", Path(__file__).resolve().parent / "watch_billing_strict_v27.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)
APART_S, PAIR_MAX_S, POLL_S, LIST_TIMEOUT_S, FRESH_MAX_S = 60.0, 600.0, 120.0, 15, 90.0  # a read slower than 90 s is stale


def containers(app: str) -> list[str]:
    """This app ID's live containers with start times, for the log; a failed listing is reported, never raised."""
    try:
        rows = json.loads(w.cli("container", "list", "--json", timeout=LIST_TIMEOUT_S))
        return [f"{c['container_id']} (start {c.get('start_time')})" for c in rows
                if isinstance(c, dict) and c.get("app_id") == app and isinstance(c.get("container_id"), str)]
    except Exception as e:
        return [f"container list failed ({type(e).__name__})"]


def main(app, cap, *, state_fn=w.state, stop_fn=w.stop, sleep_fn=time.sleep, now_fn=time.time, list_fn=containers,
         max_polls=None):
    cap, prev_over, polls = int(cap), None, 0  # prev_over: wall time of the previous SUCCESSFUL over-count poll
    while max_polls is None or polls < max_polls:
        polls += 1
        t0 = now_fn()
        try:
            st, tasks = state_fn(app)
            n = int(tasks)
            t = now_fn()  # the reading's time is taken AFTER it returns, so a sleep inside the call can't make it look fresh
            if t - t0 > FRESH_MAX_S:
                raise TimeoutError(f"state read took {t - t0:.0f} s (> {FRESH_MAX_S:.0f}): stale")
        except Exception as e:
            prev_over = None  # a failed poll breaks the pair: only consecutive fresh readings can stop the app
            w.say(f"ALERT: guard poll failed ({type(e).__name__}); no stop on a missing reading; polling again")
            sleep_fn(POLL_S)
            continue
        w.say(f"state={st} tasks={n} max={cap}")
        if w.ended(st):
            return "ended"
        if n > cap:
            w.say(f"OVER-COUNT: {n} tasks > {cap}")
            if prev_over is not None and APART_S <= t - prev_over <= PAIR_MAX_S:  # decided before any slow diagnostic call
                w.say(f"ALARM: {n} > {cap} at two consecutive fresh polls {t - prev_over:.0f} s apart; stopping {app}")
                def recheck():  # each stop retry needs a fresh over-count read
                    r0 = now_fn()
                    st2, n2 = state_fn(app)
                    return None if now_fn() - r0 > FRESH_MAX_S else (int(n2) > cap and not w.ended(st2))
                if stop_fn(app, f"guard: {n} tasks > {cap} at two consecutive fresh polls {t - prev_over:.0f} s apart",
                           recheck) != "abandoned":
                    return "stopped"
                w.say(f"stop abandoned (a fresh read no longer met the stop); the guard keeps polling {app}")
                prev_over = None  # a new pair of fresh over-counts is needed before any further stop
                sleep_fn(POLL_S)
                continue
            prev_over = t
            w.say(f"OVER-COUNT containers of {app}: {list_fn(app)}")
        else:
            prev_over = None
        sleep_fn(POLL_S)
    return "max_polls"


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest-tolerance"]:  # v2.6-compatible flag: the offline tests (cheap; run them on Modal)
        import subprocess
        sys.exit(subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "test_watchers_v27.py")]).returncode)
    r = main(*sys.argv[1:3])
    sys.exit(0 if r == "ended" else 4)
