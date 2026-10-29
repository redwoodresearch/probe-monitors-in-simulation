"""Container guard for G3's launches (0d, 02:30 PT 10-13; the 10-12 duplicate-shard lesson): every 30 s it reads the app's
running containers with the frozen v2.2 strict watcher's state() (watch_billing_strict.py here, by app id) and stops the
app, through that watcher's stop(), the first time the count exceeds the step's shard count. Fail closed, as the strict
watcher: a failed or erroring poll stops the app too. It runs beside the billing watcher (which holds the $ stop).
Exits 0 when the app is stopped, 3 after a stop it issued.

  python3 guard_containers.py <app id> <max containers (the step's shard count)> <deadline, epoch s>
"""
import sys
import time

import watch_billing_strict as W


def main(app: str, cap: str, deadline: str):
    cap, deadline, peak = int(cap), float(deadline), 0
    while True:
        try:
            st, tasks = W.state(app)
            n = int(tasks)
        except Exception as e:  # fail closed
            W.say(f"{time.strftime('%H:%M:%S')} GUARD ALARM: poll failed ({type(e).__name__}); stopping the app")
            W.stop(app, f"guard: failed poll ({type(e).__name__})")
        peak = max(peak, n)
        W.say(f"{time.strftime('%H:%M:%S')} guard state={st} containers={n} peak={peak} max={cap}")
        if n > cap:
            W.say(f"{time.strftime('%H:%M:%S')} GUARD ALARM: {n} containers, more than the step's {cap} shards; stopping the app")
            W.stop(app, f"guard: {n} containers > {cap}")
        if time.time() >= deadline:
            W.stop(app, "guard: deadline passed")
        if st in ("stopped", "gone"):
            sys.exit(0)
        time.sleep(30)


if __name__ == "__main__":
    main(*sys.argv[1:4])
