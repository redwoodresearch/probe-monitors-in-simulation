"""Frozen watcher for the plant-validation app (POST HOC). Every ~45 s: billed for this app (modal billing report) and an
integrated container-time estimate from the measured elapsed time (running tasks x $0.46/h = $0.32/h for 4 CPU + 16 GiB
x1.45). Polls about every 20 s (CLI timeouts 20 s, so at most about 60 s between decisions). Stops when
max(billed, integrated) + 1.5 minutes of the current burn would pass the stop: `modal app stop <id> -y`, then polls
`modal app list` until the app shows stopped; ALARM if not stopped within 2 minutes. A failed or malformed query is an
ALARM; elapsed burn keeps integrating at the last known task count (64 before any is known); 3 failed state queries in a
row stop the app; an app missing from the list is treated as unknown.
Usage: python plant2_watch.py <app_id> <stop_usd> <start_usd> <log>"""
import json, subprocess, sys, time
APP, STOP, integ, LOG = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
M = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal>=1.4.3", "modal"]
RATE = 0.46


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + msg + "\n")


def cli(args, timeout=20):
    try:
        r = subprocess.run(M + args, capture_output=True, text=True, timeout=max(1, timeout))
        return r.stdout if r.returncode == 0 else None
    except subprocess.TimeoutExpired:
        return None


def state(timeout=20):
    try:
        a = next((a for a in json.loads(cli(["app", "list", "--json"], timeout)) if a["app_id"] == APP), None)
        return None if a is None else (a["state"], int(a["tasks"]))
    except Exception:
        return None


def billed():
    try:
        return round(sum(float(r["cost"]) for r in json.loads(cli(["billing", "report", "--for", "today", "--resolution", "h", "--json"]))
                         if r["object_id"] == APP), 3)
    except Exception:
        return None


def stop(reason):
    deadline = time.time() + 120  # one deadline from before the stop command; every wait is capped by what remains
    log(f"STOP ({reason}): modal app stop {APP} -y")
    cli(["app", "stop", APP, "-y"], min(20, deadline - time.time()))
    while time.time() < deadline:
        st = state(min(20, deadline - time.time()))
        log(f"after stop: {st}")
        if st and st[0] == "stopped":
            sys.exit(0)
        time.sleep(max(0, min(10, deadline - time.time())))
    log(f"ALARM: {APP} not stopped within 2 minutes of issuing modal app stop -y")
    sys.exit(1)


fails, last, tasks = 0, time.time(), 64
while True:
    st, b = state(), billed()
    now = time.time()
    if st is None:
        fails += 1
        log(f"ALARM: app state query failed or app missing ({fails} in a row)")
    else:
        fails = 0
    integ += tasks * RATE * (now - last) / 3600  # the burn since the last poll, at the last known task count
    last = now
    if st is not None:
        tasks = st[1]
    if fails >= 3:
        stop("state unknown 3 times")
    if b is None:
        log("ALARM: billing query failed")
    est = max(b or 0.0, integ)
    head = max(tasks, 1) * RATE * 1.5 / 60
    log(f"state {st} billed ${b} integrated ${integ:.3f} with headroom ${est + head:.3f}")
    if est + head > STOP:
        stop(f"${est:.3f} + 1.5 min of burn > ${STOP}")
    if st and st[0] == "stopped":
        log("app stopped; watcher exits")
        sys.exit(0)
    time.sleep(20)
