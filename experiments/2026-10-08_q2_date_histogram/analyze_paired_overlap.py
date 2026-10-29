"""The paired version of the prediction's overlap test (local, no Modal): does a simulated run's most common date match
the most common date of the real run of the same task, less often than two real runs of the same task match each
other? Reads a date-histogram result's per-run most common dates (runs with at least one counted date). Task = the
sample ID without its trailing hash (main task:side task for attack runs, main task for honest runs). Per task, the
share of matching (run, run) pairs between the two arms (within one arm, pairs of two different runs); tasks are
weighted equally. Each difference is taken on the tasks both comparisons share, with a 95% interval that resamples
main tasks (all of a main task's cells together; 2,000 draws, seed 0). Real honest runs come from two recording batches
(file prefixes base200 and overlap800); the batch-matched line keeps only overlap800 real honest runs in both
comparisons. --leave-out-tied drops runs whose most common date is tied (results that record them, from 2026-10-09 on)
and prints only the differences.

  uv run --no-project python experiments/2026-10-08_q2_date_histogram/analyze_paired_overlap.py experiments/2026-10-08_q2_date_histogram/results/dates_visible_2026-10-08_164041.json [--leave-out-tied]
"""

import json
import random
import sys
from collections import defaultdict

SEED, DRAWS = 0, 2000


def task(key: str) -> str:
    return key.rsplit("__", 2)[1].rsplit(":", 1)[0]


def by_task(modes: dict, batch: str | None = None) -> dict[str, list[str]]:
    out = defaultdict(list)
    for k, day in modes.items():
        if batch is None or k.startswith(batch + "_"):
            out[task(k)].append(day)
    return out


def match(a: dict, b: dict, same_arm: bool = False) -> dict[str, float]:
    """Per shared task: the share of (run in a, run in b) pairs with the same most common date."""
    out = {}
    for t in sorted(set(a) & set(b)):
        pairs = [
            (x, y)
            for i, x in enumerate(a[t])
            for j, y in enumerate(b[t])
            if not same_arm or i < j
        ]
        if pairs:
            out[t] = sum(x == y for x, y in pairs) / len(pairs)
    return out


def paired(m1: dict, m2: dict) -> str:
    tasks = sorted(set(m1) & set(m2))
    clusters = defaultdict(list)
    for t in tasks:
        clusters[t.split(":")[0]].append(m1[t] - m2[t])
    keys, rng, boot = sorted(clusters), random.Random(SEED), []
    for _ in range(DRAWS):
        d = [x for k in (rng.choice(keys) for _ in keys) for x in clusters[k]]
        boot.append(sum(d) / len(d))
    boot.sort()
    a, b = (sum(m[t] for t in tasks) / len(tasks) for m in (m1, m2))
    return (
        f"{len(tasks)} ({len(keys)} main tasks) | {a:.1%} vs {b:.1%} | {a - b:+.1%} "
        f"[{boot[int(0.025 * DRAWS)]:+.1%}, {boot[int(0.975 * DRAWS) - 1]:+.1%}]"
    )


with open(sys.argv[1]) as fh:
    r = json.load(fh)
modes = r["most common day per run"]
untied = "--leave-out-tied" in sys.argv
if untied:
    tied = r["runs whose most common date is tied"]
    modes = {
        a: {k: v for k, v in d.items() if k not in set(tied.get(a, []))}
        for a, d in modes.items()
    }
arms = {a: by_task(m) for a, m in modes.items()}
honest_800 = by_task(modes["real-honest"], "overlap800")
cmp = {
    "sim-attack vs real-attack-e1": match(arms["sim-attack"], arms["real-attack-e1"]),
    "sim-attack vs real-attack-e2": match(arms["sim-attack"], arms["real-attack-e2"]),
    "real-attack-e2 vs real-attack-e1 (real vs real)": match(
        arms["real-attack-e2"], arms["real-attack-e1"]
    ),
    "sim-honest vs real-honest": match(arms["sim-honest"], arms["real-honest"]),
    "real-honest, two runs of one task (real vs real)": match(
        arms["real-honest"], arms["real-honest"], same_arm=True
    ),
    "sim-honest vs real-honest (overlap800 only)": match(
        arms["sim-honest"], honest_800
    ),
    "real-honest, two runs of one task (overlap800 only)": match(
        honest_800, honest_800, same_arm=True
    ),
}
print(sys.argv[1].rsplit("/", 1)[-1] + (", tied runs left out" if untied else ""))
if not untied:
    print("comparison | tasks | same most common date (tasks weighted equally)")
    for name, m in cmp.items():
        print(f"{name} | {len(m)} | {sum(m.values()) / max(len(m), 1):.1%}")
print(
    ("" if untied else "\n")
    + "simulated vs real, minus real vs real, on shared tasks | tasks | the two | difference [95% interval]"
)
for a, b in (
    ("sim-attack vs real-attack-e1", "real-attack-e2 vs real-attack-e1 (real vs real)"),
    ("sim-honest vs real-honest", "real-honest, two runs of one task (real vs real)"),
    (
        "sim-honest vs real-honest (overlap800 only)",
        "real-honest, two runs of one task (overlap800 only)",
    ),
):
    print(f"{a} | {paired(cmp[a], cmp[b])}")
