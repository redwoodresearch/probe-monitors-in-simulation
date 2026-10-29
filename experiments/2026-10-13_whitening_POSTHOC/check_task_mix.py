"""Check (1) after Amendment 4's null, counts only: per main task, how many real honest, real attack (e2) and simulated attack / honest runs the
whitening runs table holds, and how the shares line up across tasks. Tabulation of the stored runs tables, no scores."""
import json
import sys
from collections import Counter, defaultdict
from statistics import median

R36 = sys.argv[1]
base = json.load(open(f"{R36}/run_scores_2026-10-06_181548/runs.json"))
sh = [r for r in json.load(open(f"{R36}/sim_honest_scores_2026-10-07_124717/runs.json")) if r["arm"] == "sim-honest"]
runs = base + sh
assert len(runs) == 4244
c = defaultdict(Counter)
fold_of = {}
for r in runs:
    c[r["main_task"]][r["arm"]] += 1
    fold_of.setdefault(r["main_task"], set()).add(r["fold"])
print("arms:", dict(Counter(r["arm"] for r in runs)))
print("tasks:", len(c), "| tasks in more than one fold:", sum(len(s) > 1 for s in fold_of.values()))
H, E2, SA, SH = "real-honest", "real-attack-e2", "sim-attack", "sim-honest"
t_e2 = [t for t in c if c[t][E2]]
t_h = [t for t in c if c[t][H]]
print(f"tasks with e2: {len(t_e2)} | with real honest: {len(t_h)} | with both: {len(set(t_e2) & set(t_h))}")
print(f"e2 runs in tasks with no real honest run: {sum(c[t][E2] for t in t_e2 if not c[t][H])} of {sum(c[t][E2] for t in c)}")
print(f"real honest runs in tasks with no e2 run: {sum(c[t][H] for t in t_h if not c[t][E2])} of {sum(c[t][H] for t in c)}")
both = [t for t in c if c[t][E2] or c[t][H]]
e2_share = {t: c[t][E2] / (c[t][E2] + c[t][H]) for t in both}
sa_share = {t: c[t][SA] / (c[t][SA] + c[t][SH]) if c[t][SA] + c[t][SH] else float("nan") for t in both}
print(f"per-task e2 share of (e2 + real honest): median {median(e2_share.values()):.3f}, min {min(e2_share.values()):.3f}, max {max(e2_share.values()):.3f}")
ok = [t for t in both if sa_share[t] == sa_share[t]]
print(f"per-task sim-attack share of (sim attack + sim honest), tasks with sims: {len(ok)}: median {median(sa_share[t] for t in ok):.3f}, "
      f"min {min(sa_share[t] for t in ok):.3f}, max {max(sa_share[t] for t in ok):.3f}")


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


a, b = rank([sa_share[t] for t in ok]), rank([e2_share[t] for t in ok])
ma, mb = sum(a) / len(a), sum(b) / len(b)
rho = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
print(f"Spearman across tasks, sim-attack share vs e2 share: {rho:.3f} (n = {len(ok)} tasks)")
print("per fold: tasks, real honest, e2, sim attack, sim honest, e2 share")
for f in range(5):
    ts = [t for t in c if f in fold_of[t]]
    n = {a_: sum(c[t][a_] for t in ts) for a_ in (H, E2, SA, SH)}
    print(f"  fold {f}: {len(ts)} tasks | {n[H]} | {n[E2]} | {n[SA]} | {n[SH]} | {n[E2] / (n[E2] + n[H]):.3f}")
