"""Container guard v2.6: a copy of G1's guard v2 (d742d0df, sha256 42f2a21f...) whose ONLY change is the failed-poll rule
(0d, 10-13 about 18:27 PT, after a laptop DNS drop stopped healthy apps; 0d: "the v2.6 guard gets the same tolerance of 3
consecutive failed polls, and a successful poll over max (or a duplicate writer) still stops at once"): a failed or erroring
poll is tolerated, with an ALERT line, while fewer than TOLERATE (3) polls in a row have failed and the last good poll is under
MAX_GAP_S (300 s) old; the next failed poll (the 4th in a row) or a 300 s gap stops the app as before. A failed poll never
resets the over-count state, so a later good over-count poll still completes the two-poll rule. It loads the v2.6 billing
watcher beside it (watch_billing_strict_v26.py) for state() and stop(). This guard has no duplicate-writer logic of its own
(that check reads the app logs, outside the guard); nothing about it changes.
  python3 guard_tasks_v26.py <app id> <max tasks>
  python3 guard_tasks_v26.py --selftest-tolerance   (offline: the failed-poll rule's cases, PASS/FAIL each)

Guard v2's own description follows.
Container guard v2 (G1, 10-13; after v1 stopped step 4 on a one-poll task count of 9 that no container log explained).
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
_spec = importlib.util.spec_from_file_location("watch_billing_strict_v26", HERE / "watch_billing_strict_v26.py")  # v2.6 copy beside it
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)
APART_S, LIST_TIMEOUT_S = 60.0, 15
TOLERATE, MAX_GAP_S = 3, 300.0  # v2.6: failed polls tolerated in a row; longest gap since the last good poll


def tolerate_failure(fails: int, gap_s) -> tuple[bool, str]:
    """v2.6: (True, why) to keep polling after `fails` failed polls in a row (this one included), else (False, why).
    gap_s is None before the first good poll: no failure is tolerated then (as v2)."""
    if gap_s is None:
        return False, "no good poll yet"
    if fails > TOLERATE:
        return False, f"{fails} failed polls in a row (tolerates {TOLERATE})"
    if gap_s > MAX_GAP_S:
        return False, f"{gap_s:.0f} s since the last good poll (> {MAX_GAP_S:.0f})"
    return True, f"failure {fails} of {TOLERATE} tolerated"


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
    fails, good_t = 0, None  # v2.6: failed polls in a row, the last good poll's time (None before the first)
    while True:
        t = time.monotonic()  # this poll's start
        try:
            st, tasks = w.state(app)
            n = int(tasks)
            fails, good_t = 0, t
        except Exception as e:  # v2.6: tolerated or stops, per the rule (prev_over is kept either way)
            fails += 1
            keep, why = tolerate_failure(fails, None if good_t is None else time.monotonic() - good_t)
            if keep:
                w.say(f"{stamp()} ALERT: poll failed ({type(e).__name__}); {why}; polling again")
                time.sleep(max(0.0, t + APART_S - time.monotonic()))
                continue
            w.say(f"{stamp()} ALARM: poll failed ({type(e).__name__}); {why}; stopping the app")
            w.stop(app, f"guard: failed poll ({type(e).__name__}); {why}")
        w.say(f"{stamp()} state={st} tasks={n}")
        if w.ended(st):  # v2.6b: the watcher's ended(): "gone" or any state starting with "stopped"
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


def selftest_tolerance() -> bool:
    """v2.6, offline: the failed-poll rule on fixed cases; prints PASS/FAIL per case."""
    cases = [("2 failures: keep", 2, 120.0, True), ("3 failures: keep (the 3rd is tolerated)", 3, 180.0, True),
             ("4th failure in a row: stop", 4, 240.0, False), ("1 failure, 301 s since the last good poll: stop", 1, 301.0, False),
             ("1 failure before any good poll: stop", 1, None, False)]
    ok = True
    for name, f, g, want in cases:
        got = tolerate_failure(f, g)[0]
        ok &= got == want
        print(f"{'PASS' if got == want else 'FAIL'}: {name}")
    for s, want in [("ephemeral (detached)", False), ("stopped", True), ("gone", True)]:
        good = w.ended(s) == want
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}: state {s!r}: ended={w.ended(s)}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest-tolerance"]:
        sys.exit(0 if selftest_tolerance() else 6)
    main(*sys.argv[1:3])
