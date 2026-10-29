"""Fake-CLI tests for guard_tasks_v2.py (nothing real is called or stopped). A fake monotonic clock advances by every
sleep and by each fake CLI call's simulated duration, and every poll's start time is recorded.

The tests cover:
- one-poll spikes don't stop the app; each over-count is logged, with its containers line right after it;
- two consecutive over-count polls stop it, and only when their starts are at least 60 s apart;
- poll starts are always at least 60 s apart, including after slow calls;
- a slow or failed container listing doesn't block or prevent the stop;
- a failed poll, an app missing from the list, or a non-integer count fail closed;
- every stop is by app ID with --yes."""
import importlib.util
import io
import json
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP, OTHER = "ap-GUARDTEST000000000000", "ap-SOMEONEELSE0000000000"


def load():
    spec = importlib.util.spec_from_file_location("guard_tasks_v2", HERE / "guard_tasks_v2.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); return g


def run(tasks_seq, poll_s=1.0, list_s=1.0, list_fails=False, max_polls=12):
    """tasks_seq: per poll, a task count, "ERR" (the list call fails), "MISSING", "STOPPED" or a non-integer string.
    poll_s / list_s: the simulated duration of each app-list / container-list call. After a successful `app stop` every
    later list shows the app stopped. Returns (exit code, stop calls, log, poll start times)."""
    g = load(); calls, clock, i, state, starts = [], {"t": 1000.0}, {"poll": 0}, {"stopped": False}, []

    def cli(*args, timeout=60):
        calls.append((args, timeout))
        if args[:2] == ("app", "stop"):
            state["stopped"] = True; return ""
        if args[:2] == ("container", "list"):
            assert timeout == g.LIST_TIMEOUT_S
            clock["t"] += list_s
            if list_fails:
                raise subprocess.TimeoutExpired("modal container list", timeout)
            return json.dumps([{"container_id": "ta-AAA", "app_id": APP, "start_time": "02:43:22"},
                               {"container_id": "ta-BBB", "app_id": APP, "start_time": "02:59:10"},
                               {"container_id": "ta-ZZZ", "app_id": OTHER, "start_time": "01:00:00"}])
        if args[:2] == ("app", "list"):
            if state["stopped"]:
                return json.dumps([{"app_id": APP, "state": "stopped", "tasks": "0"}])
            k = i["poll"]; i["poll"] += 1
            if k >= min(len(tasks_seq), max_polls):
                raise KeyboardInterrupt
            starts.append(clock["t"]); clock["t"] += poll_s
            v = tasks_seq[k]
            if v == "ERR":
                raise subprocess.CalledProcessError(1, "modal app list")
            if v == "MISSING":
                return json.dumps([{"app_id": OTHER, "state": "ephemeral", "tasks": "1"}])
            if v == "STOPPED":
                return json.dumps([{"app_id": APP, "state": "stopped", "tasks": "0"}])
            return json.dumps([{"app_id": APP, "state": "ephemeral", "tasks": str(v)}])
        raise AssertionError(args)

    real_sleep, real_mono = time.sleep, time.monotonic
    time.sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
    time.monotonic = lambda: clock["t"]
    g.w.cli = cli
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            g.main(APP, "8"); code = None
    except SystemExit as e:
        code = e.code
    except KeyboardInterrupt:
        code = "looping"
    finally:
        time.sleep, time.monotonic = real_sleep, real_mono
    stops = [a for a, _ in calls if a[:2] == ("app", "stop")]
    assert all(a == ("app", "stop", APP, "--yes") for a in stops), stops
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(x >= 60 for x in gaps), gaps  # poll starts are always at least 60 s apart
    lines = buf.getvalue().splitlines()
    for j, l in enumerate(lines):  # every over-count line is followed at once by its containers line
        if "OVER-COUNT:" in l:
            assert "OVER-COUNT containers of" in lines[j + 1], lines[j:j + 2]
    return code, len(stops), buf.getvalue(), starts


code, stops, log, _ = run([8, 9, 8, 8, "STOPPED"])
assert stops == 0 and code == 0 and log.count("OVER-COUNT: 9 tasks > 8") == 1, (code, stops)
assert "ta-AAA" in log and "ta-BBB" in log and "ta-ZZZ" not in log, log
print("one-poll spike (8, 9, 8, 8): no stop; one over-count line, then this app's containers only (ta-AAA, ta-BBB)")

code, stops, log, starts = run([8, 9, 9, 8])
assert stops >= 1 and code == 3 and log.count("OVER-COUNT:") == 2 and "two consecutive polls 60 s apart" in log, (code, stops, log)
print(f"two consecutive over-count polls (8, 9, 9), starts {[s - starts[0] for s in starts]} s: stopped by app ID with --yes at the second")

code, stops, log, starts = run([8, 9, 9, 8], poll_s=5.0, list_s=14.0)
assert stops >= 1 and code == 3 and log.count("OVER-COUNT:") == 2, (code, stops, log)
print(f"slow calls (5 s polls, 14 s listings): poll starts {[s - starts[0] for s in starts]} s, still at least 60 s apart; stopped at the second over-count")

code, stops, log, starts = run([9, 9, 8], poll_s=70.0)
assert stops >= 1 and code == 3, (code, stops, log)
print(f"a 70 s poll: the next starts at once (starts {[s - starts[0] for s in starts]} s); stopped at the second over-count")

code, stops, log, _ = run([9, 8, 9, 8, 9, 8, "STOPPED"])
assert stops == 0 and code == 0 and log.count("OVER-COUNT:") == 3, (code, stops)
print("alternating spikes (9, 8, 9, 8, 9, 8): each logged with its containers, never stopped (a poll under the cap resets)")

code, stops, log, _ = run([8, 9, 9], list_fails=True)
assert stops >= 1 and code == 3 and "container list failed (TimeoutExpired)" in log, (code, stops, log)
print("failed container listing: logged, and the stop still happens at the second over-count")

for bad, why in (("ERR", "CalledProcessError"), ("MISSING", "AppMissing"), ("nine", "ValueError")):
    code, stops, log, _ = run([8, bad])
    assert stops >= 1 and code == 3 and f"ALARM: poll failed ({why})" in log, (bad, code, stops, log)
    print(f"poll {bad!r}: fails closed (ALARM: poll failed ({why}); stop by app ID)")
print("all guard v2 tests passed")
