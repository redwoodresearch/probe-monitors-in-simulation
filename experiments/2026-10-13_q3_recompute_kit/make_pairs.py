"""Five transcript pairs for Aswin to read himself: the real attack (second epoch, e2) and the simulated attack on the
same task (cell = main task : side task), drawn with a stated seed from #36's 939 paired cells, the runs the SCE
reply-probe headline cells score. IDs and paths only: this script opens no transcript.

Sources (git, sha256-checked):
- #36's run list runs.json @ c24106ca (arm, key, cell per scored run);
- the extraction's run table runs_2026-10-06_123859.jsonl @ 5de727c6 (each run's cohort, .eval file, sample id, epoch).
Cohort S3 prefixes: ../simulated-control-evals src/paper/configs/real_runs.yml (la_public10_real_attack_reroll_n1000,
la_public10_sim_astra_attack_n1000); the .eval file sits under <prefix>logs/ (probe_shift.transcripts.file_samples).

Run from the repo (after `git fetch origin main`): python3 -I make_pairs.py [out.csv]
"""

import csv
import hashlib
import json
import random
import subprocess
import sys

SEED = 20261013
RUNS = ("c24106ca0897b228980ad31d8e415a18dff2ff6c",
        "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json",
        "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")
TABLE = ("5de727c66b2d258d4de8c5128ff0e30d54af3da4",
         "experiments/2026-10-06_activation_extraction/results/runs_2026-10-06_123859.jsonl",
         "539c8d7d657d8541ffacc2a222a08940b762ade1310978064e8bd423d4520a1a")
COHORTS = {"la_public10_real_attack_reroll_n1000": "s3://rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch2-n1000/",
           "la_public10_sim_astra_attack_n1000": "s3://rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000/"}
ARMS = ("real-attack-e2", "sim-attack")


def git_file(commit, path, sha):
    b = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"sha256 mismatch: {commit[:8]}:{path}")
    return b


def main(out):
    runs = json.loads(git_file(*RUNS))
    table = [json.loads(line) for line in git_file(*TABLE).splitlines()]
    by_cell = {a: {} for a in ARMS}
    for r in runs:
        if r["arm"] in ARMS:
            if r["cell"] in by_cell[r["arm"]]:
                raise SystemExit(f"two {r['arm']} runs for {r['cell']}")
            by_cell[r["arm"]][r["cell"]] = r
    paired = sorted(set(by_cell[ARMS[0]]) & set(by_cell[ARMS[1]]))
    print(f"paired cells: {len(paired)}; seed {SEED}")
    rows = []
    for i, cell in enumerate(random.Random(SEED).sample(paired, 5), 1):
        for arm in ARMS:
            r = by_cell[arm][cell]
            # the run list's key is the extraction's output stem; match it to exactly one table row of the same arm and cell
            m = [t for t in table if t["arm"] == arm and t["cell"] == cell and t["key"].startswith(r["key"])]
            if len(m) != 1:
                raise SystemExit(f"{arm} {cell}: {len(m)} table rows match")
            t = m[0]
            rows.append({"pair": i, "cell": cell, "arm": arm, "run_key": t["key"], "cohort": t["cohort"],
                         "eval_s3": f"{COHORTS[t['cohort']]}logs/{t['file']}", "sample_id": t["sample_id"], "epoch": t["epoch"]})
    with open(out, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["pair"], r["arm"], r["cell"], r["eval_s3"], r["sample_id"], r["epoch"], sep="\t")
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"pairs_seed{SEED}.csv")
