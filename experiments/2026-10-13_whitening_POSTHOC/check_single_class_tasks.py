"""Counts only: how many labelled sim runs sit in main tasks with a single label class (all sim attack or all sim
honest), whose labels a within-task shuffle cannot move."""
import json
import sys
from collections import Counter, defaultdict

R36 = sys.argv[1]
runs = json.load(open(f"{R36}/run_scores_2026-10-06_181548/runs.json")) + [
    r for r in json.load(open(f"{R36}/sim_honest_scores_2026-10-07_124717/runs.json")) if r["arm"] == "sim-honest"]
assert len(runs) == 4244
c = defaultdict(Counter)
for r in runs:
    if r["arm"] in ("sim-attack", "sim-honest"):
        c[r["main_task"]][r["arm"]] += 1
only_a = [t for t in c if c[t]["sim-honest"] == 0]
only_h = [t for t in c if c[t]["sim-attack"] == 0]
mixed = [t for t in c if c[t]["sim-honest"] and c[t]["sim-attack"]]
tot = sum(sum(x.values()) for x in c.values())
print(f"tasks with sim runs: {len(c)} | only sim attack: {len(only_a)} | only sim honest: {len(only_h)} | both: {len(mixed)}")
print(f"labelled sim runs: {tot} | in attack-only tasks: {sum(c[t]['sim-attack'] for t in only_a)} | in honest-only tasks: {sum(c[t]['sim-honest'] for t in only_h)} | "
      f"in mixed tasks: {sum(sum(c[t].values()) for t in mixed)} (attack {sum(c[t]['sim-attack'] for t in mixed)}, honest {sum(c[t]['sim-honest'] for t in mixed)})")
print(f"share of labelled runs whose label no within-task shuffle can move: {(sum(c[t]['sim-attack'] for t in only_a) + sum(c[t]['sim-honest'] for t in only_h)) / tot:.3f}")
print("mixed-task compositions (sim attack, sim honest): count", sorted(Counter((c[t]["sim-attack"], c[t]["sim-honest"]) for t in mixed).items()))
