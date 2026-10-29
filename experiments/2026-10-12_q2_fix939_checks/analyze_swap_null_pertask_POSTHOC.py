"""fix939 (J), POST HOC (Bugbot on #951; after the reading, no verdict): the world-swap null with whole tasks swapped.
The registered null (analyze_own_world_auroc.py, PLAN.md addendum J) swaps each cell's edited and e2 attack runs and each
task's two honest runs independently. Here one coin per main task swaps all of that task's attack cells and its honest
run together, so swaps respect any correlation between cells of a task. 200 copies, seed 20261017, the analysis's own
code and bootstrap (2,000 task draws, seed 20261012). A copy fires if edited - real's 95% interval excludes 0. Local, $0.

  uv run --no-project --with numpy python experiments/2026-10-12_q2_fix939_checks/analyze_swap_null_pertask_POSTHOC.py \\
      <#36 scores dir> <fix939 per-run jsonl> <d9 scoring json> <d9 scores npz>
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_own_world_auroc import B, SEED, ci, draws, guard, load, save, separation, sha  # noqa: E402  the analysis's own code

N_NULL, NULL_SEED = 200, 20261017


def main(d, fix_p, scoring_p, z_p):
    head = guard()  # the imported analysis script and PLAN.md match HEAD
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("this script differs from HEAD: commit it before running")
    dz, _, _ = load(d, fix_p, scoring_p, z_p)
    W = draws(dz["n_tasks"], B, SEED)
    g, fires = np.random.default_rng(NULL_SEED), 0
    for _ in range(N_NULL):
        ft = g.integers(0, 2, dz["n_tasks"]).astype(bool)  # one coin per main task
        fa, fh = ft[dz["At"]["e2"]], ft[dz["Ht"]]
        sw = lambda x, y, m: (np.where(m[:, None], y, x), np.where(m[:, None], x, y))  # noqa: E731
        (ae, ar), (he, hr) = sw(dz["A"]["edited"], dz["A"]["e2"], fa), sw(dz["H"]["iii"], dz["H"]["i"], fh)
        d_ = separation(ae, dz["Af"]["e2"], dz["At"]["e2"], he, dz["Hf"], dz["Ht"], W)[0] - separation(ar, dz["Af"]["e2"], dz["At"]["e2"], hr, dz["Hf"], dz["Ht"], W)[0]
        lo_, hi_ = ci(d_); fires += not (lo_ <= 0 <= hi_)
    save("swap_null_pertask_POSTHOC", {"label": "POST HOC (Bugbot on #951; no verdict)", "git": head,
                                       "inputs sha256": {"fix939 per-run file": sha(fix_p), "d9 scoring record": sha(scoring_p), "d9 scores": sha(z_p)},
                                       "copies": N_NULL, "seed": NULL_SEED, "fired": fires, "rate": fires / N_NULL,
                                       "registered per-cell null (for comparison)": "fired 11 of 200 (own_world_auroc_2026-10-12_033546.json)"})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
