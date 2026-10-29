"""POST HOC (RULE.md, this folder): the margin X for (A) under one reader, from SYNTHETIC outcomes, before any edited
honest score is read. $0, local. A copy of d9's ../2026-10-12_q3_edit_honest_fa/power_edit_honest.py with only the
inputs swapped: the reader's composite detector (reader_inputs.py) in place of #36's layer 63, and sigma from the
reader's stored edited attack scores.

sigma (RULE.md): the SD, over the 823 changed attack runs, of (the mean of the run's four test-role composite nested
scores, edited - as simulated); q2's edited attack files hold nested scores only, so this departs from Qwen's outer-score
sigma. Synthetic edited copies: every simulated honest run (all treated as changed, the widest case) gets its
as-simulated nested scores plus one per-run shift d ~ Normal(0, sigma), the same for all its nested models. Only the
as-simulated honest arrays are read: assemble() is given no edited rows. X per rate = the 80th percentile of the main
contrast's reach under sigma, rounded up to 0.5 points; the 2-sigma value beside it.

  PYTHONPATH=src uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q3_reader_matched_fa/power_reader.py \
      --reader olmo --fit-dir <fit run copy> --sim-runs <sim_honest_scores runs.json> --scores <simhonest_scores_*.npz> --fix-scores <fix_scores_*.npz>
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
spec = importlib.util.spec_from_file_location("aeh", REPO / "experiments/2026-10-12_q3_edit_honest_fa/analyze_edit_honest.py")
spec.loader.exec_module(A := importlib.util.module_from_spec(spec))
spec = importlib.util.spec_from_file_location("ri", HERE / "reader_inputs.py")
spec.loader.exec_module(RI := importlib.util.module_from_spec(spec))
COPIES, DRAWS, SEED = 200, 1000, 20261014


def main():
    ap = argparse.ArgumentParser()
    for a in ("--reader", "--fit-dir", "--sim-runs", "--scores", "--fix-scores", "--stage"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    RI.frozen(__file__, HERE / "reader_inputs.py", HERE / "RULE.md")
    stage_changed = sorted(k for k, v in json.loads(Path(args.stage).read_text())["per_run"].items() if v["changed_messages"])
    D = RI.load(args.reader, args.fit_dir, args.sim_runs, args.scores, args.fix_scores, stage_changed, edited=False)  # no edited honest array is read
    # sigma from the stored edited attack scores: per changed run, mean of its four test-role nested scores, edited - as is
    deltas = []
    for i, row in D["fix"].items():
        f = int(D["fold"][i]); m = np.arange(5) != f
        deltas.append(float(np.mean(row[m]) - np.mean(D["base_C"][f, m, i])))
    deltas = np.array(deltas)
    if len(deltas) != 823 or not np.isfinite(deltas).all():
        raise SystemExit("not 823 finite edited attack changes")
    sigma = float(deltas.std())
    o, ne, arm, meta, _ = A.assemble(D["hruns"], D["h_outer"], D["h_nested"], {})  # no edited rows: copies = as is
    fold = meta["fold"].astype(int)
    M, cal, n_tasks = A.masks(o, arm, meta)
    W = A.boot_weights(meta["main_task"], DRAWS, SEED)
    sub = np.flatnonzero(cal | M["ii"] | M["iii"] | M["i"])
    Ms = {g: m[sub] for g, m in M.items()}
    edit_cols = np.flatnonzero(M["iii"][sub])
    rng = np.random.default_rng(SEED)
    out = {"label": "POST HOC power check on synthetic outcomes; no edited honest score is read", "git": head, "reader": args.reader,
           "layer_per_fold": D["layer"], "sigma_attack_runs (nested mean, composite)": sigma, "n_attack_runs": int(len(deltas)),
           "attack_mean_shift_not_used": float(deltas.mean()), "tasks": n_tasks, "copies": COPIES, "draws": DRAWS, "seed": SEED,
           "inputs sha256": {"scores": RI.sha(args.scores), "fix_scores": RI.sha(args.fix_scores), "sim_runs": RI.sha(args.sim_runs)},
           "scenarios": {}}
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
    out["X"] = {f"{a:g}": float(np.ceil(out["scenarios"]["sigma"][f"{a:g}"]["reach_p80"] * 200) / 200) for a in A.RATES}
    out["X_2sigma"] = {f"{a:g}": float(np.ceil(out["scenarios"]["2sigma"][f"{a:g}"]["reach_p80"] * 200) / 200) for a in A.RATES}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with (HERE / "results" / f"power_{args.reader}_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
