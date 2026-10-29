"""Predicted wall time of the 980-run launch (RULE.md, Amendment 3). Replays the frozen launcher's start rules
(run_pilot.py launch() @ c42666d7: new tasks strictly in file order, an environment's launches stagger_s = 90 s apart,
at most max_real_runs = 60 running) with run durations bootstrapped from batch C's attempts (seed 0, 1,000 draws), and
the infinite-capacity floor (stagger and order only). Run on the devbox:
    ~/sims/simulated-control-evals-sept/.venv/bin/python sim_wall_time.py <batch C launch.json> <980-cell tasks file>"""
import heapq
import json
import random
import sys
from datetime import datetime

T = lambda s: datetime.fromisoformat(s).timestamp()  # noqa: E731
durs = [T(a["end"]) - T(a["start"]) for r in json.load(open(sys.argv[1]))["runs"] for a in r["attempts"]]
envs = [c["env"] for c in json.load(open(sys.argv[2]))["cells"]]
STAGGER, CAP = 90.0, 60


def schedule(dur):
    last, t, running, starts, ends = {}, 0.0, [], [], []
    for env, d in zip(envs, dur):
        t = max(t, last.get(env, -1e9) + STAGGER)  # file order: not before the previous task's start
        if dur is not None:
            while len(running) >= CAP or (running and running[0] <= t):
                if len(running) >= CAP:
                    t = max(t, heapq.heappop(running))
                else:
                    heapq.heappop(running)
            heapq.heappush(running, t + d)
            ends.append(t + d)
        last[env] = t
        starts.append(t)
    return starts[-1], max(ends) if ends else None


floor, _ = schedule([0.0] * len(envs))
print(f"{len(envs)} tasks, {len(set(envs))} environments; floor (stagger and order only): last start {floor / 3600:.2f} h")
rng = random.Random(0)
res = sorted(schedule([rng.choice(durs) for _ in envs]) for _ in range(1000))
last_end = sorted(e for _, e in res)
q = lambda xs, p: xs[int(p * (len(xs) - 1))] / 3600  # noqa: E731
print(f"cap {CAP}, bootstrapped durations: last start median {q(sorted(s for s, _ in res), .5):.2f} h; "
      f"wall (last end) median {q(last_end, .5):.2f} h, p90 {q(last_end, .9):.2f} h, max {last_end[-1] / 3600:.2f} h")
