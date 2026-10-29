"""Lean offline tests for the v2.7 watcher and guard (no network: poll, stop, sleep and clock are injected).
Run: python3 test_watchers_v27.py   (prints PASS/FAIL per test; exit 0 only if all pass)"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wat, grd = load("watch_billing_strict_v27"), load("guard_tasks_v27")


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def fail(*_):
    raise TimeoutError("network down")


def test_failed_polls_never_stop():
    """10+ minutes of failed bill and state polls (a network outage) must not stop the app."""
    c, stops = Clock(), []
    r = wat.main("ap-x", "5", c.t + 3600, bill_fn=fail, state_fn=fail, stop_fn=lambda a, why, *r: stops.append(a),
                 sleep_fn=c.sleep, now_fn=c.now, max_polls=12)
    assert stops == [], stops
    assert c.t - 1_000_000.0 >= 600, c.t  # at least 10 minutes of failed polls elapsed
    return f"no stop after {c.t - 1_000_000.0:.0f} s of failed polls (result {r})"


def test_fresh_over_cap_stops_once():
    """A failed poll, then a fresh reading over the stop: exactly one stop, by app ID."""
    c, stops, reads = Clock(), [], iter([None, 6.0])

    def bill_fn(app):
        v = next(reads)
        if v is None:
            raise ConnectionError("blip")
        return v

    r = wat.main("ap-x", "5", c.t + 3600, bill_fn=bill_fn, state_fn=lambda a: ("ephemeral (detached)", "3"),
                 stop_fn=lambda a, why, *r: stops.append(a), sleep_fn=c.sleep, now_fn=c.now, max_polls=5)
    assert stops == ["ap-x"] and r == "stopped", (stops, r)
    return "one stop by app ID on the fresh $6.00 >= $5 reading"


def test_guard_stale_never_stops_fresh_pair_does():
    """Guard: failed reads between over-counts never stop; two consecutive fresh over-counts do."""
    c, stops = Clock(), []
    seq = iter([("ephemeral", "9"), None, ("ephemeral", "9"), None, None, ("ephemeral", "9")])

    def state_fn(app):
        v = next(seq)
        if v is None:
            raise TimeoutError("down")
        return v

    grd.main("ap-x", "5", state_fn=state_fn, stop_fn=lambda a, why, *r: stops.append(a), sleep_fn=c.sleep, now_fn=c.now,
             list_fn=lambda a: [], max_polls=6)
    assert stops == [], stops
    seq2, stops2 = iter([("ephemeral", "9"), ("ephemeral", "9")]), []
    r = grd.main("ap-x", "5", state_fn=lambda a: next(seq2), stop_fn=lambda a, why, *r: stops2.append(a), sleep_fn=c.sleep,
                 now_fn=c.now, list_fn=lambda a: [], max_polls=2)
    assert stops2 == ["ap-x"] and r == "stopped", (stops2, r)
    return "no stop across failed reads; one stop on two consecutive fresh over-counts"


def test_over_cap_needs_fresh_running_state():
    """An over-cap bill with the app ended, or with the state read failing, must not stop; the real parser rejects inf/NaN."""
    for st_fn in (lambda a: ("stopped", "0"), fail):
        c, stops = Clock(), []
        wat.main("ap-x", "5", c.t + 3600, bill_fn=lambda a: 6.0, state_fn=st_fn, stop_fn=lambda a, why, *r: stops.append(a),
                 sleep_fn=c.sleep, now_fn=c.now, max_polls=3)
        assert stops == [], stops
    for bad in ('[{"object_id": "ap-x", "cost": "Infinity"}]', '[{"object_id": "ap-x", "cost": "NaN"}]',
                '[{"object_id": "ap-x", "cost": "1e309"}]'):
        try:
            wat.parse_bill(bad, "ap-x")
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    assert wat.parse_bill('[{"object_id": "ap-x", "cost": "1.5"}, {"object_id": "ap-y", "cost": "9"}]', "ap-x") == 1.5
    return "no stop when ended or state unread; inf/NaN/1e309 rejected; the real parser sums only this app ID"


def test_guard_stale_timing_never_stops():
    """Guard: a reading that returns after a long sleep, or a >600 s gap between over-counts, never completes the pair."""
    c, stops = Clock(), []
    seq = iter([0, 900, 0])  # the second state call returns 900 s late (a laptop sleep inside the call)

    def slow_state(app):
        c.t += next(seq)
        return ("ephemeral", "9")

    grd.main("ap-x", "5", state_fn=slow_state, stop_fn=lambda a, why, *r: stops.append(a), sleep_fn=c.sleep, now_fn=c.now,
             list_fn=lambda a: [], max_polls=3)
    assert stops == [], stops  # the late reading neither stops nor seeds a pair with the next one
    c2, stops2 = Clock(), []
    grd.main("ap-x", "5", state_fn=lambda a: ("ephemeral", "9"), stop_fn=lambda a, why, *r: stops2.append(a),
             sleep_fn=lambda s: c2.sleep(s + 700), now_fn=c2.now, list_fn=lambda a: [], max_polls=3)
    assert stops2 == [], stops2
    c3, stopped = Clock(), []
    calls = iter([TimeoutError("down"), ("ephemeral", "2")])

    def st3(app):
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v
    def offline_stop(a):
        stopped.append(a)
        raise OSError("offline")
    r = wat.stop("ap-x", "test", recheck=lambda: int(st3("ap-x")[1]) > 5, stop_cmd=offline_stop,
                 state_fn=lambda a: ("ephemeral", "9"), sleep_fn=c3.sleep, max_attempts=3)
    assert r == "abandoned" and stopped == ["ap-x"], (r, stopped)  # one stop command, none after the fresh recovery read
    return "no stop after a 900 s-late reading (nor a pair seeded by it), with 820 s gaps, or on a retry after recovery"


def test_abandoned_stop_keeps_watching():
    """An abandoned stop must not end the watcher or guard: they keep polling and stop on the next fresh evidence."""
    c, calls, res = Clock(), [], iter(["abandoned", None])
    r = wat.main("ap-x", "5", c.t + 3600, bill_fn=lambda a: 6.0, state_fn=lambda a: ("ephemeral (detached)", "3"),
                 stop_fn=lambda a, why, *rc: (calls.append(a), next(res))[1], sleep_fn=c.sleep, now_fn=c.now, max_polls=5)
    assert r == "stopped" and calls == ["ap-x", "ap-x"], (r, calls)
    c2, polls, at, res2 = Clock(), [0], [], iter(["abandoned", None])

    def st2(a):
        polls[0] += 1
        return ("ephemeral", "9")
    r2 = grd.main("ap-x", "5", state_fn=st2, stop_fn=lambda a, why, *rc: (at.append(polls[0]), next(res2))[1],
                  sleep_fn=c2.sleep, now_fn=c2.now, list_fn=lambda a: [], max_polls=6)
    assert r2 == "stopped" and at == [2, 4], (r2, at)  # stops on polls 2 and 4: the abandoned stop reset the pair
    return "after an abandoned stop both keep polling and stop again only on new fresh evidence"


if __name__ == "__main__":
    ok = True
    for t in (test_failed_polls_never_stop, test_fresh_over_cap_stops_once, test_guard_stale_never_stops_fresh_pair_does,
              test_over_cap_needs_fresh_running_state, test_guard_stale_timing_never_stops,
              test_abandoned_stop_keeps_watching):
        try:
            print(f"PASS: {t.__name__}: {t()}")
        except Exception as e:
            ok = False
            print(f"FAIL: {t.__name__}: {type(e).__name__} {e}")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
