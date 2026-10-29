"""Routing-evidence addendum (09's ruling via 4f, 10-15 ~12:1x PT): DT-scored runs whose main is in no DT fit cohort are UNMAPPED under revised
Append 21b. Lists them by key from the committed records (keys and counts only; no npz is read or rewritten) and writes one append-only record.
Usage (from the DT worktree): python3 -I write_routing_unmapped.py <manifests_dt dir> <ref manifests dir> <out>"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

MD, RD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
R = "experiments/2026-10-14_p2_dt_scaleup/results"
show = lambda s: subprocess.run(["git", "show", s], capture_output=True, check=True).stdout  # noqa: E731
fc_src = f"0c20bf47e:{R}/dt_unseen_check_2026-10-15_120424.json"
fc = json.loads(show(fc_src))
ex = {d: sorted(v["excluded_tasks"]) for d, v in fc["check"]["designs"].items()}
E = set(next(iter(ex.values())))
assert all(set(v) == E for v in ex.values()) and len(E) == 5, "the four designs' fits exclude different tasks"
t196_src = f"ba4331cbf:{R}/dt_q4b_t196_t196_2026-10-14_174544.json"
t196 = json.loads(show(t196_src))
rl_src = f"675f2a451:{R}/dt_q4b_relaunch_rh1e4_relaunch_2026-10-15_115637.json"
rl = json.loads(show(rl_src))
reason = "never in any DT fit cohort"
rows = {"t196_q4b": sorted(k for k in t196["runs_main"] if k.split("|")[1] in E),
        "ref/relaunch/dt": sorted(k for k in rl["runs"] if k.split("|")[1] in E)}
scanned = {}
for p in sorted(MD.glob("*.json")) + sorted(RD.glob("*.json")):
    m = json.loads(p.read_text())
    hit = sorted(j for j in m["runs"] if j.split("|")[1] in E)
    scanned[p.stem] = len(hit)
    if hit and p.stem != "ref_relaunch_dt":
        rows[p.stem] = hit
rec = {"label": "PRELIMINARY; routing-evidence addendum (keys only; no npz read or rewritten)",
       "ruling": "09 via 4f (10-15 ~12:1x PT): treat these runs as UNMAPPED under revised Append 21b. No DT fold model trained on their mains, so the "
                 "designated-fold `score` (and `fold`, `layer`) in their npz rows is not a held-out score. The analysis uses the five-model ensemble "
                 "from `score_by_fold`, re-standardized on the unseen-main pool, as for every fold -1 main.",
       "evidence": {"fold_check": fc_src, "fold_check_sha256": hashlib.sha256(show(fc_src)).hexdigest(), "fits": fc["fits"],
                    "excluded_tasks_by_design": ex, "cohort_tasks_by_design": {d: v["cohort_tasks"] for d, v in fc["check"]["designs"].items()},
                    "held_out_runs_failing_trained_elsewhere": {d: fc["check"]["held_runs"] - v["held_out_mains_in_some_other_model_training"]
                                                                for d, v in fc["check"]["designs"].items()}},
       "records": {"t196_q4b": t196_src, "ref/relaunch/dt": rl_src},
       "unmapped_runs": {f: {k: {"reason": reason, "route": "revised 21b five-model ensemble (score_by_fold), not the designated-fold score"} for k in ks}
                         for f, ks in rows.items()},
       "counts": {f: len(ks) for f, ks in rows.items()},
       "scanned_folder_manifests": {"n": len(scanned), "with_hits": {k: v for k, v in scanned.items() if v}}}
OUT.open("x").write(json.dumps(rec, indent=1))
print(json.dumps({"counts": rec["counts"], "scanned": rec["scanned_folder_manifests"], "excluded": sorted(E),
                  "failing": rec["evidence"]["held_out_runs_failing_trained_elsewhere"]}, indent=1))
