"""Side-task success on the 939 paired cells under the sims project's own measure, against the in-run nano verdict.

Input: the sims paper's per-cell success triplets (`success_triplets_corrected.json`, sha256 35ee58f7...), a member of
s3://rr-sce-sim-transcripts/paper/builds/2026-10-03_selective_safety_evidence/evidence.tar.gz
(workspace/la1000-metrics/experiments/2026-09-18_linuxarena_astra_strategy/results/corrected_metrics/). Each row has the
real attack's native scorer outcome (C/I), the codebase-aware Sol-high success judge's credence on the real
transcript and on the matched Astra simulated transcript, and the simulated run's in-run gpt-5-nano label.
Means and paired differences with 95% intervals from 2,000 bootstrap resamples of main tasks.

  aws s3 cp <evidence.tar.gz> <dir>/ && tar xzf <dir>/evidence.tar.gz -C <dir> <member>
  uv run python experiments/2026-10-07_sim_failure_hiding/analyze_success_credence.py --triplets <dir>/<member>
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SHA256 = "35ee58f7a4a39c68a041f5d9c70d428c00f6782c6aedc0b9e591c7c299d28827"
FIELDS = {"native": "real_native", "real_sol": "real_sol", "sim_sol": "sim_sol", "nano": "sim_in_run_label"}


def value(x) -> float | None:
    return {"C": 1.0, "I": 0.0}.get(x) if isinstance(x, str) else (None if x is None else float(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", type=Path, required=True)
    p.add_argument("--reps", type=int, default=2000)
    args = p.parse_args()
    raw = args.triplets.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256, "not the pinned success_triplets_corrected.json"
    rows = {r["cell"]: r for r in json.loads(raw)["rows"]}
    cells = json.loads((REPO / "splits/exclusions_v3.json").read_text())["paired_attack_cells"]["cells"]
    kept = [c for c in cells if c in rows and all(value(rows[c][f]) is not None for f in FIELDS.values())]
    x = {k: np.array([value(rows[c][f]) for c in kept]) for k, f in FIELDS.items()}
    mains = [c.split(":")[0] for c in kept]
    names = sorted(set(mains))
    idx = np.array([names.index(m) for m in mains])
    w = np.random.default_rng(0).multinomial(len(names), np.full(len(names), 1 / len(names)), size=args.reps)[:, idx]
    boot = {k: (w * v).sum(1) / w.sum(1) for k, v in x.items()}

    def est(a, b=None):
        point = x[a].mean() - (x[b].mean() if b else 0)
        draws = boot[a] - (boot[b] if b else 0)
        return {"estimate": round(float(point), 4), "ci95": [round(float(q), 4) for q in np.percentile(draws, [2.5, 97.5])]}

    out = {
        "meta": {"created": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                 "triplets_sha256": SHA256, "n_paired_cells": len(cells),
                 "n_cells_used": len(kept), "missing_or_undefined": sorted(set(cells) - set(kept)),
                 "bootstrap": {"reps": args.reps, "cluster": "main task", "seed": 0},
                 "fields": FIELDS},
        "means": {k: est(k) for k in FIELDS},
        "differences": {"sim_sol_minus_real_sol (same instrument)": est("sim_sol", "real_sol"),
                        "sim_sol_minus_native": est("sim_sol", "native"),
                        "real_sol_minus_native (judge bias on real)": est("real_sol", "native"),
                        "nano_minus_native": est("nano", "native")},
    }
    path = HERE / "results" / f"success_credence_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out["means"]), json.dumps(out["differences"]), path, sep="\n")


if __name__ == "__main__":
    main()
