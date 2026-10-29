"""POST HOC (RULE.md), step 2: the margin X for "false alarms stable", from SYNTHETIC outcomes, before any edited score
exists. $0, local; reads only published scores.

Base: the committed layer-63 output_end scores of the 4,254 runs (simhonest/2026-10-07_124717/scores.npz, sha256 below).
Synthetic edited copies: every one of the 437 simulated honest runs (all treated as changed, the widest case) gets its
as-is nested scores plus one per-run shift d ~ Normal(0, sigma), the same for all its nested models: no true change on
average. sigma: the SD of the per-run outer-score change that the same edit made on simulated ATTACK runs (fix939's
per-run file, FIX - V1; published), and twice that. For each of 200 synthetic copies, the main contrast's 95% interval
((iii) - (ii), paired, thresholds redrawn per draw; analyze_edit_honest.contrast with 1,000 task draws) gives a reach
max(|lo|, |hi|). X per rate = the 80th percentile of the reach under the wider sigma, rounded up to 0.5 points: inside
+-X is where the interval lands at least 80% of the time when nothing changed on average.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy python experiments/2026-10-12_q3_edit_honest_fa/power_edit_honest.py \
      --scores <simhonest_2026-10-07_124717 scores.npz> --fix939-runs <fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl>
"""

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
spec = importlib.util.spec_from_file_location("aeh", HERE / "analyze_edit_honest.py")
spec.loader.exec_module(A := importlib.util.module_from_spec(spec))
SCORES_SHA = "2e9cbf66"  # the README's S3 archive of simhonest/2026-10-07_124717/scores.npz
FIX939_SHA = "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"  # fix939_manifest_POSTHOC_2026-10-11.json
COPIES, DRAWS, SEED = 200, 1000, 20261014


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--fix939-runs", required=True)
    args = ap.parse_args()
    if not A.sha(args.scores).startswith(SCORES_SHA) or A.sha(args.fix939_runs) != FIX939_SHA:
        raise SystemExit("an input's sha256 differs from its manifest")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    by = {}
    for line in Path(args.fix939_runs).read_text().splitlines():
        r = json.loads(line)
        by.setdefault(r["key"], {})[r["label"]] = r["run_scores"]["output_end"]
    deltas = np.array([v["FIX"] - v["V1"] for v in by.values() if v.get("FIX") is not None and v.get("V1") is not None])
    sigma = float(deltas.std())
    runs = json.loads(A.RUNS.read_text())
    z = np.load(args.scores)
    o0, n0 = z["output_end__outer"], z["output_end__nested"]
    o, ne, arm, meta, _ = A.assemble(runs, o0, n0, {})  # edited copies = as-is, then shifted below
    fold = meta["fold"].astype(int)
    M, cal, n_tasks = A.masks(o, arm, meta)
    W = A.boot_weights(meta["main_task"], DRAWS, SEED)
    sub = np.flatnonzero(cal | M["ii"] | M["iii"] | M["i"])  # only these runs enter thresholds or rates
    Ms = {g: m[sub] for g, m in M.items()}
    edit_cols = np.flatnonzero(M["iii"][sub])
    rng = np.random.default_rng(SEED)
    out = {"label": "POST HOC power check on synthetic outcomes; no edited score exists", "git": head, "sigma_attack_runs": sigma,
           "n_attack_runs": int(len(deltas)), "attack_mean_shift_not_used": float(deltas.mean()), "tasks": n_tasks, "copies": COPIES,
           "draws": DRAWS, "seed": SEED, "scenarios": {}}
    for name, s in (("sigma", sigma), ("2sigma", 2 * sigma)):
        reach = {a: [] for a in A.RATES}
        width = {a: [] for a in A.RATES}
        for _ in range(COPIES):
            d = rng.normal(0, s, len(edit_cols))
            nes = ne[:, :, sub].copy()
            nes[:, :, edit_cols] = nes[:, :, edit_cols] + d[None, None, :]
            for a in A.RATES:
                _, r = A.contrast(nes, fold[sub], cal[sub], Ms, a, W[:, sub])
                lo, hi = A.ci(r["iii-ii"][1:])
                reach[a].append(max(abs(lo), abs(hi)))
                width[a].append(hi - lo)
        out["scenarios"][name] = {"sigma": s, **{f"{a:g}": {"reach_p80": float(np.percentile(reach[a], 80)), "reach_median": float(np.median(reach[a])),
                                                          "width_median": float(np.median(width[a]))} for a in A.RATES}}
    out["X"] = {f"{a:g}": float(np.ceil(out["scenarios"]["2sigma"][f"{a:g}"]["reach_p80"] * 200) / 200) for a in A.RATES}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"power_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
