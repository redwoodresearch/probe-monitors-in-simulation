"""Per-world rates of the fake-environment judge's yeses, as rates_with_intervals.py defines them (runs with any yes of
that type over runs; yes turns over parsed turns), with task-clustered bootstrap 95% intervals beside the exact ones:
2,000 resamples of tasks with replacement (seed [20261011, 501, i]), a task's runs and turns moving together. Also counts
yeses whose quote was not found in the turn (the stored `quote_found` field) and the rates without them. Reads only the
data file's world, run, task, suspicion and quote_found fields; prints numbers only. $0, local.

  uv run --no-project --with numpy --with scipy python analyze_fake_env_clustered.py <data/fake_env_<stamp>.jsonl>
"""
import collections, hashlib, json, sys
import numpy as np
from scipy.stats import beta

KEEP = ("world", "run", "task", "suspicion", "quote_found")
exact = lambda k, n: [beta.ppf(0.025, k, n - k + 1) if k else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0]
raw = open(sys.argv[1], "rb").read()
rows = [{k: r.get(k) for k in KEEP} for r in map(json.loads, raw.decode().splitlines())]
del raw
out = {"input_sha256": hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(), "worlds": {}}
YES = ("test_or_eval", "looks_fake")
for w in sorted({r["world"] for r in rows}):
    wr = [r for r in rows if r["world"] == w]
    parsed = [r for r in wr if r["suspicion"] != "unparsed"]
    tasks = sorted({r["task"] for r in wr}); ti = {t: i for i, t in enumerate(tasks)}
    runs_of = collections.defaultdict(set)
    for r in wr:
        runs_of[ti[r["task"]]].add(r["run"])
    res = {"tasks": len(tasks), "runs": sum(len(v) for v in runs_of.values()), "parsed_turns": len(parsed),
           "yes_quote_not_found": sum(1 for r in parsed if r["suspicion"] in YES and not r["quote_found"])}
    for t in ("test_or_eval", "looks_fake", "either"):
        for drop in (False, True):
            hit = lambda r: (r["suspicion"] in YES if t == "either" else r["suspicion"] == t) and (r["quote_found"] or not drop)
            run_hit = collections.defaultdict(set); turn_n = np.zeros(len(tasks)); turn_k = np.zeros(len(tasks))
            for r in parsed:
                turn_n[ti[r["task"]]] += 1
                if hit(r):
                    turn_k[ti[r["task"]]] += 1
                    run_hit[ti[r["task"]]].add(r["run"])
            run_n = np.array([len(runs_of[i]) for i in range(len(tasks))], float)
            run_k = np.array([len(run_hit[i]) for i in range(len(tasks))], float)
            br, bt = [], []
            for i in range(1, 2001):
                m = np.bincount(np.random.default_rng([20261011, 501, i]).integers(0, len(tasks), len(tasks)), minlength=len(tasks))
                br.append(m @ run_k / (m @ run_n)); bt.append(m @ turn_k / (m @ turn_n))
            kr, nr, kt, nt = int(run_k.sum()), int(run_n.sum()), int(turn_k.sum()), int(turn_n.sum())
            res[t + (" (quote found only)" if drop else "")] = {
                "runs": f"{kr}/{nr}", "run_rate": kr / nr, "run_exact": exact(kr, nr), "run_task_bootstrap": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
                "turns": f"{kt}/{nt}", "turn_rate": kt / nt, "turn_exact": exact(kt, nt), "turn_task_bootstrap": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}
    out["worlds"][w] = res
print(json.dumps(out, default=float))
