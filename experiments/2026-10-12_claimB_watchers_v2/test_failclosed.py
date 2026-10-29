"""Negative tests for the v2 watchers with a fake Modal CLI (nothing real is called or stopped)."""
import importlib.util, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # v2.4: the watchers are loaded from this file's folder, whatever the working directory

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def run_strict(app_list, tasks_seq=None, max_sleeps=6):
    w = load("watch_billing_strict"); calls = []; n = {"i": 0, "t": 0}
    def cli(*args, timeout=60):
        calls.append(args)
        if args[:2] == ("billing", "report"): return "[]"
        if args[:2] == ("app", "stop"): return ""
        if args[:2] == ("app", "list"):
            v = app_list(n)
            if isinstance(v, Exception): raise v
            return v
        raise AssertionError(args)
    def sleep(s):
        n["i"] += 1
        if n["i"] > max_sleeps: raise KeyboardInterrupt
    w.cli, w.time.sleep = cli, sleep
    try: w.main("ap-FAKE", "10", str(time.time() + 3600)); code = None
    except SystemExit as e: code = e.code
    except KeyboardInterrupt: code = "looping"
    return code, sum(a[:2] == ("app", "stop") for a in calls)

listed = lambda tasks: json.dumps([{"app_id": "ap-FAKE", "state": "deployed", "tasks": tasks}])
assert run_strict(lambda n: "[]") == (3, 1); print("strict missing app: stops")
o = run_strict(lambda n: subprocess.CalledProcessError(1, "x")); assert o[0] == "looping" and o[1] >= 1, o; print("strict list error: stop issued, retried")
o = run_strict(lambda n: json.dumps([{"app_id": "ap-FAKE"}])); assert o[0] == "looping" and o[1] >= 1, o; print("strict missing field: stop issued, retried")
def idle_then(n):
    n["t"] += 1
    return listed("1") if n["t"] == 1 else listed("0")
o = run_strict(idle_then); assert o[1] >= 1, o; print("strict idle 3 polls: stop issued")

def run_cost(container_out, bill_rc=0, max_sleeps=30, stop_res=(0, "", ""), extra=(), calls=None, app_state="ephemeral"):
    c = load("watch_cost"); calls = [] if calls is None else calls; n = {"i": 0}
    def sh(args, timeout=120):
        calls.append(args)
        if args[:2] == ["container", "list"]:
            v = container_out(n)
            return v if isinstance(v, subprocess.CompletedProcess) else subprocess.CompletedProcess(args, 0, v, "")
        if args[:2] == ["app", "stop"]: return subprocess.CompletedProcess(args, *stop_res)
        if args[:2] == ["app", "list"]: return subprocess.CompletedProcess(args, 0, json.dumps([{"app_id": "ap-" + "F" * 22, "state": app_state}] if app_state else []), "")
        if args[:2] == ["billing", "report"]: return subprocess.CompletedProcess(args, bill_rc, "[]", "")
        raise AssertionError(args)
    def sleep(s):
        n["i"] += 1
        if n["i"] > max_sleeps: raise KeyboardInterrupt
    c.sh, c.time.sleep = sh, sleep
    sys.argv = ["watch_cost.py", "--app-id", "ap-" + "F" * 22, "--stop", "5", "--log", "/dev/null", *extra]
    try: c.main(); out = "returned"
    except KeyboardInterrupt: out = "looping"
    return out, sum(a[:2] == ["app", "stop"] for a in calls)

assert run_cost(lambda n: subprocess.CompletedProcess(["c"], 1, "", "")) == ("returned", 1); print("cost list rc!=0: stops")
assert run_cost(lambda n: "not json") == ("returned", 1); print("cost malformed json: stops")
assert run_cost(lambda n: "{}") == ("returned", 1); print("cost wrong schema: stops")
def work_then_idle(n):
    n["k"] = n.get("k", 0) + 1
    return json.dumps([{"app_name": "fake", "app_id": "ap-" + "F" * 22, "container_id": "c1", "start_time": time.time()}]) if n["k"] <= 2 else "[]"
assert run_cost(work_then_idle) == ("returned", 1); print("cost work then idle: stops")
assert run_cost(lambda n: "not json", stop_res=(1, "", "App is already stopped.")) == ("returned", 1)
assert run_cost(lambda n: "not json", stop_res=(1, "App is already stopped. (2026-10-12 17:24)\n", "")) == ("returned", 1)
assert run_cost(lambda n: "not json", stop_res=(2, "", "App is already stopped."), max_sleeps=8)[0] == "looping"
print("cost stop, app already stopped: returns after one stop (stderr and stdout); other rc keeps retrying")
assert run_cost(lambda n: "not json", stop_res=(1, "", "Error: something else"), max_sleeps=8)[0] == "looping"
print("cost stop, other stop error (rc 1): keeps retrying")

def billing_read(stdout, now="2026-10-13T23:59:00+00:00"):  # v2.2: exact --end from a frozen clock, exact app match
    c = load("watch_cost"); got = []
    class Frozen(c.datetime):
        @classmethod
        def now(cls, tz=None): return c.datetime.fromisoformat(now)
    c.datetime = Frozen
    c.sh = lambda args, timeout=120: got.append(args) or subprocess.CompletedProcess(args, 0, stdout, "")
    return c.billed("ap-X", "2026-10-12"), got[0]
rows = [{"object_id": "ap-X", "cost": "0.25"}, {"object_id": "ap-X", "cost": "0.10"}, {"object_id": "ap-Y", "description": "ap-X", "cost": "9"}]
v, args = billing_read(json.dumps(rows))
assert abs(v - 0.35) < 1e-12, v
assert args[args.index("--start") + 1] == "2026-10-12" and args[args.index("--end") + 1] == "2026-10-15", args
assert billing_read(json.dumps({"data": rows}))[0] == v
assert billing_read("{}")[0] is None and billing_read("[1]")[0] is None  # malformed: failed read
assert billing_read("[]")[0] == 0 and billing_read(json.dumps([{"object_id": "ap-Y", "cost": "1"}]))[0] == 0  # no rows yet (lag): $0
assert billing_read(json.dumps([{"object_id": "ap-X", "cost": "nan"}]))[0] is None
assert billing_read(json.dumps([{"object_id": "ap-X"}]))[0] is None
print("cost billing read: --start fixed, --end = UTC date + 2, exact object_id match, malformed or non-finite -> failed read")

def strict_stop(stop_seq, list_seq):  # v2.2: stop() checks state after every attempt; missing = gone only after a success
    w = load("watch_billing_strict"); calls = []; n = {"i": 0, "s": 0, "l": 0}
    def cli(*args, timeout=60):
        calls.append(args)
        if args[:2] == ("app", "stop"):
            ok = stop_seq[min(n["s"], len(stop_seq) - 1)]; n["s"] += 1
            if not ok: raise subprocess.CalledProcessError(1, "modal app stop", "error")
            return ""
        if args[:2] == ("app", "list"):
            v = list_seq[min(n["l"], len(list_seq) - 1)]; n["l"] += 1
            if v == "ERR": raise subprocess.CalledProcessError(1, "modal app list")
            return "[]" if v == "MISSING" else json.dumps([{"app_id": "ap-FAKE", "state": v, "tasks": "0"}])
        raise AssertionError(args)
    def sleep(s):
        n["i"] += 1
        if n["i"] > 5: raise KeyboardInterrupt
    w.cli, w.time.sleep = cli, sleep
    try: w.stop("ap-FAKE", "test"); code = None
    except SystemExit as e: code = e.code
    except KeyboardInterrupt: code = "looping"
    return code, sum(a[:2] == ("app", "stop") for a in calls)
assert strict_stop([False], ["stopped"]) == (3, 1)  # the reported race: stop errors on an already-stopped app
assert strict_stop([True], ["stopped"]) == (3, 1)
assert strict_stop([False], ["deployed"])[0] == "looping"  # a live app keeps being stopped
assert strict_stop([False], ["MISSING"])[0] == "looping"  # failed stop + not listed: not proof it stopped
assert strict_stop([False], ["ERR"])[0] == "looping"  # a failed state check retries
assert strict_stop([True], ["MISSING"]) == (3, 1)  # successful stop + not listed: gone
assert strict_stop([True, False], ["deployed", "MISSING"]) == (3, 2)  # success, stale live state, failed stop, then missing
print("strict stop: stopped exits; missing exits only after a successful stop; live, failed checks and unconfirmed misses retry")

# v2.3: by app ID only (a fresh ID counts all its containers; --since only filters when given); the stop names the app ID
AID, OTHER = "ap-" + "F" * 22, "ap-" + "O" * 22
def busy(app_id, offset):  # a container that started `offset` seconds before each poll
    return lambda n: json.dumps([{"app_name": "same-name", "app_id": app_id, "container_id": "c-" + app_id, "start_time": time.time() - offset}])
calls = []
out = run_cost(busy(AID, 3600), extra=["--app-id", AID, "--stop", "1"], calls=calls)  # 1 h at $5.4/h is past $1
assert out == ("returned", 1) and ["app", "stop", AID, "--yes"] in calls, (out, calls[-1])
assert run_cost(busy(OTHER, 3600), extra=["--app-id", AID, "--stop", "1"], max_sleeps=4)[0] == "looping"  # another app: not counted
assert run_cost(busy(AID, 3600), extra=["--app-id", AID, "--stop", "1", "--since", str(time.time())], max_sleeps=4)[0] == "looping"  # before --since
assert run_cost(busy(AID, 3600), extra=["--app-id", AID, "--stop", "1", "--since", str(time.time() - 7200)], max_sleeps=4) == ("returned", 1)
for bad in (["--app-id", "my-app-name"], ["--app-id", "ap-short"], ["--app-id", AID, "--since", "nan"], ["--app-id", AID, "--stop", "inf"],
            ["--app-id", AID, "--startup-polls", "0"], ["--app-id", AID, "--startup-polls", "-3"], ["--app-id", AID, "--rate", "0"],
            ["--app-id", AID, "--rate", "-1"], ["--app-id", AID, "--since", str(time.time() + 3600)]):
    c = load("watch_cost"); sys.argv = ["watch_cost.py", "--stop", "1", "--log", "/dev/null", *bad]
    try: c.main(); raise AssertionError(f"accepted {bad}")
    except SystemExit as e: assert e.code not in (0, None), bad
print("cost v2.3: counts only its app ID (and --since when given); stops by app ID; rejects names, short IDs, non-finite args")
for st, want in (("stopped", "returned"), ("ephemeral", "looping")):  # after a failed stop: only "stopped" is done
    got = run_cost(lambda n: "not json", stop_res=(1, "", "Error: something else"), app_state=st, max_sleeps=8)[0]
    assert got == want, (st, got)
print("cost v2.3: a failed stop is done only when the list shows the app stopped (live or missing keeps retrying)")

def unit_stop(app_state):  # v2.4: stop_until_done alone; a failed stop on an app missing from the list keeps retrying
    c = load("watch_cost"); n = {"i": 0}
    def sh(args, timeout=120):
        if args[:2] == ["app", "stop"]: return subprocess.CompletedProcess(args, 1, "", "Error: something else")
        if args[:2] == ["app", "list"]: return subprocess.CompletedProcess(args, 0, json.dumps([{"app_id": AID, "state": app_state}] if app_state else []), "")
        raise AssertionError(args)
    def sleep(s):
        n["i"] += 1
        if n["i"] > 5: raise KeyboardInterrupt
    c.sh, c.time.sleep = sh, sleep
    try: c.stop_until_done(AID, "/dev/null", "test"); return "returned"
    except KeyboardInterrupt: return "looping"
assert unit_stop(None) == "looping" and unit_stop("stopped") == "returned"
assert run_cost(lambda n: "[]", max_sleeps=40) == ("returned", 1)  # listed app, no container ever seen: stopped after 10 polls
c = load("watch_cost"); c.sh = lambda args, timeout=120: subprocess.CompletedProcess(args, 0, "[]", ""); clock = {"now": 0.0}
c.time.sleep, c.time.monotonic = lambda s: clock.__setitem__("now", clock["now"] + s), lambda: clock["now"]  # fake clock: no 2-min wait
sys.argv = ["watch_cost.py", "--app-id", AID, "--stop", "1", "--log", "/dev/null"]
try: c.main(); raise AssertionError("watched an app ID that is not listed")
except SystemExit as e: assert e.code == 2, e.code
print("cost v2.4: an unlisted app ID exits 2 at start; a listed app with no container in 10 polls is stopped")

def startup(list_seq, lookup_s=0.0, gate_containers=False):  # v2.4: app list replies in order (the last repeats)
    c = load("watch_cost"); n = {"l": 0}; t = {"now": 0.0}
    def sh(args, timeout=120):
        if args[:2] == ["app", "list"]:
            t["now"] += lookup_s
            v = list_seq[min(n["l"], len(list_seq) - 1)]; n["l"] += 1
            body = {"LISTED": [{"app_id": AID, "state": "ephemeral"}], "ABSENT": [], "BAD": [{}]}.get(v)
            return subprocess.CompletedProcess(args, 1, "", "err") if v == "ERR" else subprocess.CompletedProcess(args, 0, json.dumps(body), "")
        if args[:2] == ["container", "list"]:
            rows = [{"app_id": AID, "container_id": "c1", "start_time": t["now"]}] if gate_containers else []
            return subprocess.CompletedProcess(args, 0, json.dumps(rows), "")
        if args[:2] == ["billing", "report"]: return subprocess.CompletedProcess(args, 0, "[]", "")
        raise AssertionError(args)
    def sleep(s):
        if s >= 60: raise KeyboardInterrupt  # the main loop's first sleep: it is watching
        t["now"] += s
    c.sh, c.time.sleep, c.time.monotonic = sh, sleep, lambda: t["now"]
    sys.argv = ["watch_cost.py", "--app-id", AID, "--stop", "1000", "--log", "/dev/null"]
    try: c.main(); out = "returned"
    except SystemExit as e: out = f"exit {e.code}"
    except KeyboardInterrupt: out = "watching"
    return out, t["now"], n["l"]
o = startup(["ABSENT"]); assert o[0] == "exit 2" and 120 <= o[1] <= 121, o  # a working list never shows it: exit 2 at about 2 min
o = startup(["ABSENT"], lookup_s=30); assert o[0] == "exit 2" and o[1] <= 150, o  # slow lookups: at most one call past the deadline
assert startup(["ERR"])[0] == "watching"  # failed lookups are not absence: it watches (and fails closed there)
assert startup(["BAD"])[0] == "watching"  # malformed rows are a failed lookup too
assert startup(["ABSENT", "LISTED"])[0] == "watching"
assert startup(["LISTED", "ABSENT"]) == ("watching", 0.0, 1)  # one sighting is enough: no second lookup
assert startup(["ABSENT"], gate_containers=True) == ("watching", 0.0, 0)  # list lags, containers run: watch at once
assert run_cost(busy(AID, 3600), extra=["--app-id", AID, "--stop", "1000", "--since", str(time.time())], max_sleeps=25)[0] == "looping"  # only pre-since containers: alive, not stopped
assert startup(["ABSENT", "ERR"])[0] == "exit 2"  # absence seen by a working list holds through later failed lookups
assert startup(["ERR", "ABSENT", "BAD"])[0] == "exit 2"
calls = []
def alternating(n):  # empty successful polls alternate with failed ones: the bound counts the successful ones
    n["a"] = n.get("a", 0) + 1
    return "[]" if n["a"] % 2 else subprocess.CompletedProcess(["c"], 1, "", "")
assert run_cost(alternating, max_sleeps=40, calls=calls) == ("returned", 1)
print("cost v2.4: startup exits 2 only when a working list never shows the ID; failed lookups go on watching; alternating empty and failed polls still stop")

def gate_then_gone():  # v2.4: a container seen at the gate, gone by the first poll, still counts toward the stop
    c = load("watch_cost"); n = {"cl": 0}; stops = []
    def sh(args, timeout=120):
        if args[:2] == ["container", "list"]:
            n["cl"] += 1
            return subprocess.CompletedProcess(args, 0, json.dumps([{"app_id": AID, "container_id": "c1", "start_time": time.time() - 3600}] if n["cl"] == 1 else []), "")
        if args[:2] == ["app", "list"]: return subprocess.CompletedProcess(args, 0, json.dumps([{"app_id": AID, "state": "ephemeral"}]), "")
        if args[:2] == ["billing", "report"]: return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:2] == ["app", "stop"]: stops.append(args); return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)
    def sleep(s): raise KeyboardInterrupt
    c.sh, c.time.sleep = sh, sleep
    sys.argv = ["watch_cost.py", "--app-id", AID, "--stop", "1", "--log", "/dev/null"]  # 1 h at $5.4/h is past $1
    try: c.main(); return "returned", len(stops)
    except KeyboardInterrupt: return "watching", len(stops)
assert gate_then_gone() == ("returned", 1)
print("cost v2.4: containers running during the startup gate start cost watching at once, and count even if gone by the first poll")

def gate_timeouts():  # v2.4: every startup-gate CLI call is capped at 15 s, so two hung calls blind it for 30 s at most
    c = load("watch_cost"); seen_t = []; t = {"now": 0.0}
    def sh(args, timeout=120):
        if args[:2] in (["container", "list"], ["app", "list"]):
            seen_t.append(timeout); t["now"] += timeout  # every call hangs until its timeout
            return subprocess.CompletedProcess(args, 124, "", "timeout")
        if args[:2] == ["billing", "report"]: return subprocess.CompletedProcess(args, 0, "[]", "")
        raise AssertionError(args)
    def sleep(s):
        if s >= 60: raise KeyboardInterrupt
        t["now"] += s
    c.sh, c.time.sleep, c.time.monotonic = sh, sleep, lambda: t["now"]
    sys.argv = ["watch_cost.py", "--app-id", AID, "--stop", "1000", "--log", "/dev/null"]
    try: c.main()
    except (KeyboardInterrupt, SystemExit): pass  # either way the gate has finished; only its call timeouts are checked
    return seen_t
ts_ = gate_timeouts()
assert ts_ and max(ts_[:-1]) <= 15, ts_  # the gate's calls (the last entry may be the main loop's own poll)
print("cost v2.4: startup-gate CLI calls are capped at 15 s")

def frozen_after_finish():  # v2.4: a container that finished keeps its cost frozen through later failed polls
    import tempfile
    c = load("watch_cost"); clock = {"now": 1_000_000.0}; n = {"cl": 0}
    seq = [[{"app_id": AID, "container_id": "c1", "start_time": 1_000_000.0 - 600}], [], "ERR", "ERR", [], "STOP"]
    def sh(args, timeout=120):
        if args[:2] == ["container", "list"]:
            v = seq[min(n["cl"], len(seq) - 1)]; n["cl"] += 1
            if v == "STOP": raise KeyboardInterrupt
            return subprocess.CompletedProcess(args, 1, "", "") if v == "ERR" else subprocess.CompletedProcess(args, 0, json.dumps(v), "")
        if args[:2] == ["app", "list"]: return subprocess.CompletedProcess(args, 0, json.dumps([{"app_id": AID, "state": "ephemeral"}]), "")
        if args[:2] == ["billing", "report"]: return subprocess.CompletedProcess(args, 0, "[]", "")
        raise AssertionError(args)
    real_time = c.time.time
    def sleep(s): clock["now"] += s
    log = Path(tempfile.mkdtemp()) / "w.log"
    c.sh, c.time.sleep, c.time.time, c.time.monotonic = sh, sleep, lambda: clock["now"], lambda: clock["now"]
    sys.argv = ["watch_cost.py", "--app-id", AID, "--stop", "1000", "--log", str(log)]
    try: c.main()
    except KeyboardInterrupt: pass
    finally: c.time.time = real_time
    return [json.loads(x)["est_usd"] for x in log.read_text().splitlines() if "est_usd" in x]
ests = frozen_after_finish()
assert len(ests) >= 4 and ests[1] == ests[2] == ests[3], ests  # finished after poll 1: no growth through the failed polls
print("cost v2.4: rejects non-positive rate/stop/cap and a future --since; finished containers stay frozen through failed polls")

c = load("watch_cost"); real_mono = __import__("time").perf_counter; t0 = real_mono()
r = c.run_group(["bash", "-c", "(sleep 30) & sleep 30"], timeout=1)  # v2.5: a grandchild holding the pipes open
assert r.returncode == 124 and real_mono() - t0 < 10, (r.returncode, real_mono() - t0)
assert c.run_group(["bash", "-c", "echo ok"], timeout=10).stdout.strip() == "ok"
assert c.run_group(["/nonexistent/binary"], timeout=5).returncode == 125
print("cost v2.5: a hung CLI call with a grandchild returns in about 1 s (the whole process group is killed)")
