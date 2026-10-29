"""The batch spec for C's new real runs (PREREG.md @ a12f23b3; sizes from results/c_sizing_2026-10-11_150659.json): one seeded
order of the frozen 274 unpaired simulated-cohort tasks (default_rng(20261011).permutation), so that stage 1 (the first
50 tasks, ~100 runs) and either size (166 or 233 tasks) are prefixes of one list. Task IDs only; $0.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_new_real_runs/make_batch_spec.py <runs_phase1_honest.csv>
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
RUNS_SHA = "75381ffed70ddc96ddf59b55eff3c7cf9df77db82063cea009a49bd5d37a83ce"
UNPAIRED_SHA = "b735a5e06016a59df327b1ad6b7b373209a8a1dfb891853d943cf9eb87414c3f"
SEED, STAGE1, SIZES = 20261011, 50, {"per-comparison Kish": 166, "largest Kish (as Amendment 1)": 233}


def main(runs_p):
    if hashlib.sha256(Path(runs_p).read_bytes()).hexdigest() != RUNS_SHA:
        raise SystemExit("run list differs from PREREG's")
    runs = list(csv.DictReader(open(runs_p)))
    order = next(c for c in runs[0] if c.startswith("real_run_order"))
    batch = lambda rid: "overlap800" if "|overlap800_" in rid else "base200" if "|base200_" in rid else "other"
    sim = sorted({r["main_task"] for r in runs if r["arm"] != "real-honest"})
    f2 = defaultdict(dict)
    for r in runs:
        if r["arm"] == "real-honest" and r[order] in ("1", "2"):
            f2[r["main_task"]][r[order]] = r["run_id"]
    same = {t for t, d in f2.items() if len(d) == 2 and all(batch(v) == "overlap800" for v in d.values())}
    unpaired = sorted(set(sim) - same)
    if hashlib.sha256("\n".join(unpaired).encode()).hexdigest() != UNPAIRED_SHA:
        raise SystemExit("unpaired list differs from PREREG's")
    perm = [unpaired[i] for i in np.random.default_rng(SEED).permutation(len(unpaired))]
    out = {"prereg": "PREREG.md @ a12f23b3", "unpaired sha256": UNPAIRED_SHA, "seed": SEED, "runs per task": 2,
           "stage 1 (first tasks)": STAGE1, "sizes (first tasks)": SIZES, "task order": perm,
           "task order sha256": hashlib.sha256("\n".join(perm).encode()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"batch_spec_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(out["task order sha256"], perm[:3], dest)


if __name__ == "__main__":
    main(sys.argv[1])
