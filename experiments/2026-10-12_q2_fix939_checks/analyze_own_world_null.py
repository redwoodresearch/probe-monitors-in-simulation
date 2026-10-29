"""fix939 (J), the per-world label-shuffle null (added on request after the reading: Codex on #951, the orchestrator's
decision, 10-12; committed before its run). A level check, descriptive, no verdict. The world-swap null in
analyze_own_world_auroc.py stays the contrast's calibration. Local, $0.

For each world, within each nested model's test fold, the attack and honest labels are shuffled among that fold's runs
(counts kept), and the pooled AUROC is recomputed exactly as the analysis computes it (its separation(), unit weights).
1,000 shuffles per world, seed 20261016. Gate: the unshuffled AUROCs reproduce own_world_auroc_2026-10-12_033546.json's.

  uv run --no-project --with numpy python experiments/2026-10-12_q2_fix939_checks/analyze_own_world_null.py \\
      <#36 scores dir> <fix939 per-run jsonl> <d9 scoring json> <d9 scores npz> <own_world_auroc_2026-10-12_033546.json>
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_own_world_auroc import guard, load, save, separation, sha  # noqa: E402  the analysis's own code

RESULT_SHA = "cf9b905737c4f58c"  # prefix of own_world_auroc_2026-10-12_033546.json's sha256
N_SHUFFLE, SEED = 1000, 20261016
WORLDS = {"real (e2 vs i)": ("e2", "i"), "original sim (sim vs ii)": ("sim", "ii"), "edited sim (edited vs iii)": ("edited", "iii"), "e1 vs i": ("e1", "i")}


def main(d, fix_p, scoring_p, z_p, res_p):
    head = guard()  # the imported analysis script and PLAN.md match HEAD
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("this script differs from HEAD: commit it before running")
    if sha(res_p)[:16] != RESULT_SHA:
        raise SystemExit("not the committed own-world AUROC result")
    committed = json.loads(Path(res_p).read_text())["worlds"]
    dz, _, _ = load(d, fix_p, scoring_p, z_p)
    one = np.ones((1, dz["n_tasks"]))
    out = {"label": "added on request after the reading (Codex on #951; the orchestrator's decision); descriptive, no verdict",
           "git": head, "inputs sha256": {Path(res_p).name: sha(res_p), "fix939 per-run file": sha(fix_p), "d9 scoring record": sha(scoring_p),
                                          "d9 scores": sha(z_p), "#36 runs.json": sha(Path(d) / "runs.json")},
           "shuffles per world": N_SHUFFLE, "seed": SEED, "worlds": {}}
    g = np.random.default_rng(SEED)
    for name, (a, h) in WORLDS.items():
        A, Af, At, H, Hf, Ht = dz["A"][a], dz["Af"][a], dz["At"][a], dz["H"][h], dz["Hf"], dz["Ht"]
        obs = float(separation(A, Af, At, H, Hf, Ht, one)[0][0])
        if not np.isfinite(obs) or abs(obs - committed[name]["AUROC"]) > 1e-12:
            raise SystemExit(f"{name}: the AUROC {obs} doesn't reproduce the committed {committed[name]['AUROC']}")
        rows, folds, tasks = np.vstack([A, H]), np.concatenate([Af, Hf]), np.concatenate([At, Ht])
        lab = np.r_[np.ones(len(A), bool), np.zeros(len(H), bool)]
        null = []
        for _ in range(N_SHUFFLE):
            sh = lab.copy()
            for f in range(5):  # shuffle within each test fold, counts kept
                ix = np.flatnonzero(folds == f); sh[ix] = g.permutation(lab[ix])
            null.append(float(separation(rows[sh], folds[sh], tasks[sh], rows[~sh], folds[~sh], tasks[~sh], one)[0][0]))
        null = np.array(null)
        if not np.isfinite(null).all():
            raise SystemExit(f"{name}: a non-finite shuffled AUROC")
        out["worlds"][name] = {"AUROC": obs, "null mean": float(null.mean()), "null 95th / 99th percentile": [float(x) for x in np.percentile(null, [95, 99])],
                               "null max": float(null.max()), "shuffles at or above the AUROC": int((null >= obs).sum())}
    save("own_world_null", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
