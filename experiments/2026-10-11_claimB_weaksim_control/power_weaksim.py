"""$0 power check for the weak-simulator positive control (PLAN.md): how often the registered rule gives a clear answer.

The rule (PLAN.md): A = the cell-weighted mean of per-environment AUROCs over pairs of DIFFERENT main tasks (R0 vs la18
real; each environment has its own frozen probe, so runs are only ranked against runs scored by the same probe; a main
task's own real-simulated pair is left out because the two share the task), run score = mean output_end score, interval
from a main-task bootstrap stratified by environment (la18 has one cell per main task, checked below). PASS if the
one-sided lower bound is >= BAR; FAIL if the one-sided 95% upper bound (95th percentile) is < BAR; otherwise TOO FEW.
The lower bound's percentile is calibrated here: P(PASS) is reported for the 5th, 4th, 3rd and 2.5th percentiles, and the
plan registers the highest percentile whose P(PASS) at a true A of BAR stays at or below 0.05 in every setting.

Model per world (binormal, unit variance per arm): environment e's separation is d_e = d0 + u_e, u_e ~ N(0, tau_e^2);
d0 is solved in each world so that the world's own population AUROC (cell-weighted mean of Phi(d_e / sqrt 2)) equals
the target, so every row is power at a fixed cohort AUROC. tau_e comes from a latent between-environment AUROC SD, a
sensitivity knob: 0; 0.030 (step A's level-1 spread over its 5 test environments, SD 0.042 in
claimB_A_per_env_2026-10-10_180020.json on S3 manifest v13, minus a typical sampling SE of about 0.029); and the raw
0.042. A cell effect shared by a main task's real and simulated run has correlation rho (0; 0.3 as a sensitivity).
Also reports P(lower bound >= 0.822), step A's own fixed-sim AUROC, for information only, and the Monte Carlo SE. Only
the AUROC bound is simulated, not the rule's baseline conditions (they can only lower P(PASS)).

  uv run --no-project --python 3.12 --with numpy --with scipy python \\
      experiments/2026-10-11_claimB_weaksim_control/power_weaksim.py results/la18_cohort_<stamp>.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).parent
BAR, STEP_A = 0.75, 0.822
TARGETS = (0.65, 0.70, 0.75, 0.80, 0.822, 0.85, 0.90, 0.95)
SETTINGS = ((0.0, 0.0), (0.030, 0.0), (0.042, 0.0), (0.030, 0.3))  # (latent between-environment AUROC SD, rho)
WORLDS, BOOT, SEED = 1000, 500, 20261011
PCTS = (5.0, 4.0, 3.0, 2.5)  # candidate one-sided lower-bound percentiles


def env_counts(cohort: dict, shared_only: bool) -> list[int]:
    """Cells (real and R0 both present) per environment: the shared environments only, or all of them."""
    per = cohort["cells_per_environment"]["real_R0"]
    shared = set(cohort["cells"]["environments_shared_with_claimB"])
    return [n for e, n in sorted(per.items()) if e in shared or not shared_only]


def world(rng, counts, target, env_sd, rho):
    """One world at a fixed population AUROC: (point, lower bounds at PCTS..., 95th percentile) of the bootstrap."""
    w = np.array(counts) / sum(counts)
    tau = env_sd * np.sqrt(2) / norm.pdf(norm.ppf(target))  # delta method: dA/dd = phi(d / sqrt 2) / sqrt 2
    u = rng.normal(0, tau, len(counts)) if env_sd > 0 else np.zeros(len(counts))
    d0 = brentq(lambda d: float(w @ norm.cdf((d + u) / np.sqrt(2))) - target, -20, 20)
    point, boot = 0.0, np.zeros(BOOT)
    for n, ue, we in zip(counts, u, w):
        c = rng.normal(0, np.sqrt(rho), n)
        real = c + rng.normal(0, np.sqrt(1 - rho), n)
        sim = c + d0 + ue + rng.normal(0, np.sqrt(1 - rho), n)
        K = (sim[:, None] > real[None, :]) + 0.5 * (sim[:, None] == real[None, :])
        W = rng.multinomial(n, np.full(n, 1 / n), size=BOOT).astype(float)  # main-task weights within the environment
        bad = (W > 0).sum(1) < 2  # a draw holding a single task has no cross-task pair: redraw it (the analysis does too)
        while bad.any():
            W[bad] = rng.multinomial(n, np.full(n, 1 / n), size=int(bad.sum()))
            bad = (W > 0).sum(1) < 2
        point += we * (K.sum() - np.trace(K)) / (n * n - n)  # pairs of different main tasks only
        num = np.einsum("bi,ij,bj->b", W, K, W) - (W**2) @ np.diag(K)  # drops a task's pairs with itself or its copies
        boot += we * num / (n * n - (W**2).sum(1))
    if not np.isfinite(boot).all():
        raise ValueError("non-finite bootstrap AUROC")
    return (point, *np.percentile(boot, PCTS), np.percentile(boot, 95))


def main():
    cohort_path = HERE / sys.argv[1]
    cohort = json.loads(cohort_path.read_text())
    if cohort["cells"]["real_R0_cells_per_main_task_max"] != 1:
        raise ValueError("more than one cell per main task: the bootstrap must resample main tasks, not cells")
    rng = np.random.default_rng(SEED)
    out = {"label": "power for the weak-simulator positive control (planning estimate)", "bar": BAR, "step_a": STEP_A,
           "worlds": WORLDS, "boot": BOOT, "seed": SEED, "rows": []}
    for name, shared_only in (("shared_environments", True), ("all_environments", False)):
        counts = env_counts(cohort, shared_only)
        for env_sd, rho in SETTINGS:
            for t in TARGETS:
                res = np.array([world(rng, counts, t, env_sd, rho) for _ in range(WORLDS)])
                lbs, ub = res[:, 1:1 + len(PCTS)], res[:, -1]
                by = {str(q): {"p_pass": float((lbs[:, k] >= BAR).mean()),
                               "p_too_few": float(((lbs[:, k] < BAR) & (ub >= BAR)).mean()),
                               "p_lower_bound_ge_step_a": float((lbs[:, k] >= STEP_A).mean()),
                               "median_width_lb_to_ub": float(np.median(ub - lbs[:, k]))} for k, q in enumerate(PCTS)}
                out["rows"].append({
                    "reading": name, "cells": int(sum(counts)), "environments": len(counts), "env_sd": env_sd,
                    "rho": rho, "true_auroc": t, "mean_point": float(res[:, 0].mean()),
                    "p_fail": float((ub < BAR).mean()), "by_lower_bound_percentile": by,
                    "mc_se_max": float(np.sqrt(0.25 / WORLDS)),
                })
    out["inputs"] = {str(cohort_path.relative_to(HERE)): hashlib.sha256(cohort_path.read_bytes()).hexdigest()}
    out["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_weaksim_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for r in out["rows"]:
        print(r["reading"], r["cells"], "sd", r["env_sd"], "rho", r["rho"], "A", r["true_auroc"], "point",
              round(r["mean_point"], 3), "pass", {k: round(v["p_pass"], 3) for k, v in r["by_lower_bound_percentile"].items()},
              "fail", round(r["p_fail"], 3))
    print("->", dst)


if __name__ == "__main__":
    main()
