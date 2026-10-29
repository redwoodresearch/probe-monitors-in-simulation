"""Draw for RULE.md (this folder): the top 10% of the 437 fix-full simulated honest runs by the mixed fix-full probe (c″)
at L63, then 10 of them with seed 41013, each with its real honest counterpart. Numbers and keys only; no transcript text.

  uv run --no-project --python 3.12 --with numpy==2.3.3 python experiments/2026-10-13_claimB_read10/draw_read10.py
"""

import csv
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SCORES = (
    "42c3737a",
    "experiments/2026-10-13_q3_sce_reply_probe/results/simhonest_fullfix_c_full_runscores_2026-10-13_152618.csv",
)
SCORES_SHA_PREFIX = "0cf45b7e"
REAL = (
    "88b3ae092887fad5de2e0b68259b3df8c13bbb4a",
    "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json",
)
REAL_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
LAYER, SEED, N_DRAW, N_RUNS = 63, 41013, 10, 437


def git_show(rev, path):
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"], capture_output=True, check=True, cwd=HERE
    ).stdout


def main():
    raw = git_show(*SCORES)
    sha = hashlib.sha256(raw).hexdigest()
    assert sha.startswith(SCORES_SHA_PREFIX), (
        f"scores sha256 {sha} does not start with {SCORES_SHA_PREFIX}"
    )
    rows = [
        r for r in csv.DictReader(io.StringIO(raw.decode())) if int(r["layer"]) == LAYER
    ]
    assert len(rows) == N_RUNS and len({r["run_key"] for r in rows}) == N_RUNS, (
        len(rows),
        "L63 rows",
    )
    for r in rows:
        r["score_mean"] = float(r["score_mean"])
        assert math.isfinite(r["score_mean"]), r["run_key"]
    ranked = sorted(rows, key=lambda r: (-r["score_mean"], r["run_key"]))
    k = math.ceil(0.1 * N_RUNS)
    top = ranked[:k]
    perm = np.random.default_rng(SEED).permutation(k).tolist()
    real_raw = git_show(*REAL)
    assert hashlib.sha256(real_raw).hexdigest() == REAL_SHA, (
        "counterpart manifest changed"
    )
    real = [r for r in json.loads(real_raw) if r["arm"] == "real-honest"]
    assert len(real) == 990, len(real)
    by_task = {}
    for r in sorted(real, key=lambda r: r["key"]):
        by_task.setdefault(r["main_task"], r["key"])

    def entry(i):
        r = top[i]
        return {
            "rank": i + 1,
            "run_key": r["run_key"],
            "main_task": r["main_task"],
            "score_mean": r["score_mean"],
            "caught_1pct": float(r["caught_1pct"]),
            "counterpart": by_task.get(r["main_task"], "no counterpart"),
        }

    out = {
        "label": "read10 draw (POST HOC, descriptive); RULE.md in this folder",
        "scores": {"commit": SCORES[0], "path": SCORES[1], "sha256": sha},
        "real_runs": {"rev": REAL[0], "path": REAL[1], "sha256": REAL_SHA},
        "layer": LAYER,
        "n_runs": len(rows),
        "top_k": k,
        "boundary_score": top[-1]["score_mean"],
        "next_score_below": ranked[k]["score_mean"],
        "top_caught_1pct_gt0": sum(float(r["caught_1pct"]) > 0 for r in top),
        "seed": SEED,
        "permutation": perm,
        "drawn": [entry(i) for i in perm[:N_DRAW]],
        "replacement_order": [entry(i) for i in perm[N_DRAW:]],
        "git": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=HERE,
            check=True,
        ).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule_sha256": hashlib.sha256((HERE / "RULE.md").read_bytes()).hexdigest(),
        "numpy": np.__version__,
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"draw_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(
        path.name,
        "top_k",
        k,
        "caught_1pct>0 in top",
        out["top_caught_1pct_gt0"],
        "no counterpart",
        sum(e["counterpart"] == "no counterpart" for e in out["drawn"]),
    )


if __name__ == "__main__":
    main()
