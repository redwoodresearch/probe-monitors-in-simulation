"""Per fold model of own_frac{100,25,10}_L48 (09 / 0d's question via 4f, 2026-10-15): distinct T196 cells ("tasks"), distinct main tasks, runs (run1 + honest_a epochs) and rows it
trained on, rebuilt from pass 1's spec (sha-pinned) with run_famexp._train_tasks' rule @ b5a0586fa, and checked against the fit's own
group records (train_tasks, n). Keys and counts only.

  python3 -I check_fold_training_counts.py <spec_2026-10-14_163027.json> <groups dir> <repo> <out.json>
(spec and groups/own_{R,C}_L48_f{0..4}.json from probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13:/famexp/2026-10-14_163027/)
"""
import hashlib
import json
import math
import random
import subprocess
import sys

spec_p, gdir, repo = sys.argv[1:4]
sb = open(spec_p, "rb").read()
assert hashlib.sha256(sb).hexdigest() == "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"
spec = json.loads(sb)
t196 = json.loads(subprocess.run(["git", "-C", repo, "show", "4d1eec572:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"], capture_output=True, check=True).stdout)
main_of = {c["key"]: c["main"] for c in t196["cells"]}
fot = spec["fot"]
out = {}
for ri, readout in enumerate(("R", "C")):
    keys = [k for k, (w, t) in spec["map"].items() if w in ("run1", "honest_a") and spec["n_rows"][k][ri] > 0]
    tasks_all = sorted({spec["map"][k][1] for k in keys})
    for f in range(5):
        t = sorted({spec["map"][k][1] for k in keys if fot[spec["map"][k][1]] != f})
        random.Random(f"famexp|own_a|f{f}").shuffle(t)
        g = json.load(open(f"{gdir}/own_{readout}_L48_f{f}.json"))
        for v in (1.0, 0.25, 0.1):
            tt = set(t[: math.ceil(v * len(t))])
            ks = [k for k in keys if spec["map"][k][1] in tt]
            name = f"own_frac{int(round(v * 100))}_L48"
            fit = [x for x in g["fits"] if x["variant"] == name and x["fold"] == f]
            assert len(fit) == 1
            out.setdefault((name, readout), []).append({"fold": f, "cells": len(tt), "mains": len({main_of[c] for c in tt}), "runs": len(ks),
                "runs_run1": sum(spec["map"][k][0] == "run1" for k in ks), "runs_honest_a": sum(spec["map"][k][0] == "honest_a" for k in ks),
                "rows": sum(spec["n_rows"][k][ri] for k in ks), "fit_train_tasks": fit[0]["train_tasks"], "fit_n": fit[0]["n"]})
    print(f"readout {readout}: cells with run1/honest_a rows = {len(tasks_all)}, their mains = {len({main_of[c] for c in tasks_all})}, runs = {len(keys)}")
for (name, readout), rows in out.items():
    def mm(k, rows=rows):
        return f"{min(r[k] for r in rows)}–{max(r[k] for r in rows)}"
    print(f"{name} {readout}: cells {mm('cells')} | mains {mm('mains')} | runs {mm('runs')} (run1 {mm('runs_run1')}, honest_a {mm('runs_honest_a')}) | rows {mm('rows')}"
          f" | cells == fit train_tasks: {all(r['cells'] == r['fit_train_tasks'] for r in rows)} | rows == fit n: {all(r['rows'] == r['fit_n'] for r in rows)}")
meta = {"label": "own_frac{100,25,10}_L48 training sets per fold model, by key: T196 cells, distinct main tasks, runs (run1 + honest_a epochs), rows; keys and counts only",
        "sources": {"spec": {"volume": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", "path": "/famexp/2026-10-14_163027/spec_2026-10-14_163027.json", "sha256": hashlib.sha256(sb).hexdigest(),
                             "pinned_by": "results/prep_2026-10-14_163027.json result.spec_sha256", "keys_read": ["map", "n_rows", "fot"]},
                    "rule": "run_famexp._train_tasks and _group @ b5a0586fa (the fit's commit): cells in folds != f with run1 or honest_a rows in the readout, seeded order, first ceil(p x n)",
                    "fit_check": {"path": "/famexp/2026-10-14_163027/run_2026-10-14_175636/groups/own_{R,C}_L48_f{0..4}.json", "keys_read": ["fits[].variant", "fits[].fold", "fits[].train_tasks", "fits[].n"]},
                    "mains": "T196 list @ 4d1eec572 cells[].main"},
        "units": {"cells": "T196 task cells (main x side task)", "mains": "distinct main tasks of those cells", "runs": "run1 + honest_a epochs (one epoch per run)", "rows": "training units at the readout (R: reply rows; C: content-only command rows)"}}
json.dump(meta | {"fits": {f"{n}|{r}": v for (n, r), v in out.items()}}, open(sys.argv[4], "x"), indent=1)
