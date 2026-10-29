"""Tests for 0d's refined duplicate-writer rule: leases in run_sce_fixfull.py and alerts in launch_sce_fixfull.py."""
import importlib.util, os, sys, tempfile, threading, time
from pathlib import Path
WT = Path(sys.argv[1]); D = WT / "experiments/2026-10-13_q3_sce_reply_probe"
sys.path.insert(0, str(D))
import run_sce_fixfull as X
X.BEAT, X.FRESH, X.WAIT, X.KILL_AFTER = 0.2, 1.5, 0.8, 0.8
store = {}
# 1. one writer: no alert, a done marker
with X.Lease(store, "s", "fit:48:a") as L:
    pass
assert L.alerts == [] and any(k.startswith("done|s|fit:48:a|") for k in store), "single writer"
print("1 ok: one writer, done marker written")
# 2. a preempted predecessor: a lease that stopped beating 5 s ago -> alert, no stop
store["s|pack:63|old-task"] = time.time() - 5
with X.Lease(store, "s", "pack:63") as L2:
    pass
assert L2.alerts == ["old-task"], L2.alerts
print("2 ok: stale predecessor ->", L2.alerts, "(alert, no stop)")
# 3. two live writers: writer A keeps beating while B enters -> B stops (Gate)
A = X.Lease(store, "s", "fit:63:a")
A.__enter__()
B = X.Lease(store, "s", "fit:63:a")
try:
    B.__enter__()
    raise SystemExit("FAILED: two live writers were not stopped")
except X.Gate as e:
    assert ("conflict|" + B.key) in store, "sticky conflict marker"
    print("3 ok: two live writers stop ->", e, "| conflict marker written")
A.stop.set()
# 3b. final_check: the conflict marker fails the run even though A completes normally
A.__exit__(None, None, None)
ok, w = X.final_check(store, "s", {"fit:48:a", "pack:63", "fit:63:a"})
assert not ok and w["conflict_markers"] == 1, w
print("3b ok: final check fails on the conflict ->", w)
# 3c. final_check passes a clean run, fails a lease still beating
st2 = {}
for i in ("pack:48", "fit:48:a"):
    with X.Lease(st2, "t", i):
        pass
ok, w = X.final_check(st2, "t", {"pack:48", "fit:48:a"})
assert ok, w
C = X.Lease(st2, "t", "fit:48:a"); C.__enter__()  # a duplicate of a completed input, still running
ok, w = X.final_check(st2, "t", {"pack:48", "fit:48:a"})
assert not ok and w["live_leases_at_end"] == 1, w
C.stop.set()
print("3c ok: clean run passes; a still-running duplicate fails ->", w)
# 3d. a writer whose heartbeat fails, or whose write blocks, kills itself after KILL_AFTER (< FRESH), not later
class Bad(dict):
    mode = None
    def __setitem__(self, k, v):
        if self.mode == "raise":
            raise ConnectionError("dict down")
        if self.mode == "block":
            time.sleep(60)
        super().__setitem__(k, v)
for mode in ("raise", "block"):
    exits, t0 = [], time.monotonic()
    X.os._exit = lambda code, exits=exits, t0=t0: exits.append((code, round(time.monotonic() - t0, 2)))
    bad = Bad(); E = X.Lease(bad, "u", "pack:48"); E.__enter__(); bad.mode = mode
    time.sleep(X.FRESH)
    assert exits and exits[0][0] == 3 and X.KILL_AFTER <= exits[0][1] < X.FRESH, (mode, exits)
    E.stop.set()
    print(f"3d ok ({mode}): the writer exits after {exits[0][1]} s, before its lease looks stale ({X.FRESH} s)")
# 3e. slow successful writes: the deadline runs from before the last good write, so the writer dies before its stored
# stamp goes stale (exit wall time - last stored stamp < FRESH)
class Slow(dict):
    calls = 0
    def __setitem__(self, k, v):
        self.calls += 1
        if self.calls == 2:
            time.sleep(0.5)  # a slow but successful beat
        elif self.calls > 2:
            raise ConnectionError("dict down")
        super().__setitem__(k, v)
exits = []
X.os._exit = lambda code: exits.append((code, time.time()))
sl = Slow(); F = X.Lease(sl, "v", "pack:63"); F.__enter__()
time.sleep(2.5)
F.stop.set()
stamp = sl[F.key]
assert exits and exits[0][1] - stamp < X.KILL_AFTER + 0.2, (exits, stamp)  # the old post-write anchor gives ~0.5 + 0.8 = 1.3 s
print(f"3e ok: slow good write; the writer exits {exits[0][1] - stamp:.2f} s after its last stored stamp (< KILL_AFTER + 0.2)")
# 3f. a first write slower than KILL_AFTER: the lease is refused before any work starts
class SlowFirst(dict):
    def __setitem__(self, k, v):
        time.sleep(X.KILL_AFTER + 0.2)
        super().__setitem__(k, v)
try:
    X.Lease(SlowFirst(), "w", "pack:48").__enter__()
    raise SystemExit("FAILED: a stale first acquisition started work")
except X.Gate as e:
    print("3f ok: slow first acquisition refused ->", e)
# 4. launcher: a preemption line is an ALERT (logged), not an Abort
spec = importlib.util.spec_from_file_location("L", D / "launch_sce_fixfull.py"); Lm = importlib.util.module_from_spec(spec); spec.loader.exec_module(Lm)
J = Path(tempfile.mkdtemp()); Lm.state["log"] = J / "launch.log"
(J / "run.log").write_text("View run at https://x\nRunner preempted; restarting input on a new container\nWARNING for FunctionGetOutputs (RESOURCE_EXHAUSTED). Retrying...\n")
n = Lm.alert(J, 0)
assert n == 1 and "ALERT" in (J / "launch.log").read_text(), "launcher alert"
assert Lm.alert(J, n) == 1 and (J / "launch.log").read_text().count("ALERT") == 1, "no repeat alert"
print("4 ok: preemption line ->", (J / "launch.log").read_text().strip()[:120])
