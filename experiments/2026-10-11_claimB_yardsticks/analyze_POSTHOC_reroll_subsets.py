"""POST HOC (0d's GO, relayed after 23:40 PT 10-11; $0, stored scores only, no Modal compute): is I3's frozen-probe
R2-vs-R1 AUROC (0.625 on its 97 cells) a small-subset fluke, and are I3's cells an unusual draw for S1 vs R1 (0.873)?

Rule, fixed before running:
- Scores: the weak-sim job's stored frozen-probe per-run scores for claim B runs (claimB/weaksim_v1/scores.json,
  `claimB_shared`; the meaning control's refits, each environment's probe trained without it). They cover 4 of the 5
  test environments (clinical_trial, media_processing, shared_dotfiles, user_tracking); tiktok is not stored, so this
  check uses those 4 only and I3's 78 cells in them. Layer gate: for every (cell, arm) of I3's 78 cells, the stored score
  must equal I3's own L63 score (or, failing that, its step-A-layer score) within 1e-6; the matching layer is the one
  checked, and if neither matches for all, the script stops. The other layer has no stored all-cell scores (not run).
- Population: every test-environment cell in those 4 environments with e1, e2 and the fixed sim stored.
- Reference: 2,000 random subsets (default_rng(20261011)) drawn per environment without replacement with I3's
  per-environment counts (I3's cells are the reroll study's preselected ~20 per environment, so the draw is stratified by
  environment). Per subset: AUROC(e2 over e1) and AUROC(S1 over e1).
- Reported: the full-population AUROCs; I3's 78-cell AUROCs from the same stored scores; each one's percentile in the
  subset distribution and two-sided p = min(1, 2 * (1 + #tail) / 2001), tail on the side the value falls. "Unusual draw"
  if either p < 0.05.
- S2 has no reference distribution (it exists only on I3's cells). The subset SD of AUROC(S1 over e1) is reported as how
  far a ~78-97-cell AUROC moves with cell choice. Added, labelled post hoc: a paired main-task bootstrap (2,000 draws) of
  AUROC(S1 over R1) - AUROC(S2 over R1) on I3's 97 cells from I3's own scores (L63 and step A's layer), the direct
  within-cell comparison.

  uv run --no-project --python 3.13 --with numpy --with scikit-learn python \\
      experiments/2026-10-11_claimB_yardsticks/analyze_POSTHOC_reroll_subsets.py <weaksim scores.json> <I3 reroll_scores json>
"""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
WS, I3 = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
ENVS = ["clinical_trial", "media_processing", "shared_dotfiles", "user_tracking"]
ARMS = {"R1": "real-attack-e1", "R2": "real-attack-e2", "S1": "sim-attack-fixed"}
SEED, N_SUB, N_BOOT, TOL = 20261011, 2000, 2000, 1e-6


def auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def place(value, dist):
    dist = np.asarray(dist)
    lo, hi = int((dist <= value).sum()), int((dist >= value).sum())
    return {"value": value, "percentile": float(100 * (dist < value).mean() + 50 * (dist == value).mean()),
            "p_two_sided": min(1.0, 2 * (1 + min(lo, hi)) / (len(dist) + 1))}


def main():
    ws_raw, i3_raw = WS.read_bytes(), I3.read_bytes()
    shared, i3 = json.loads(ws_raw)["claimB_shared"], json.loads(i3_raw)["runs"]
    score = {}
    for r in shared:
        if r["environment"] in ENVS and r["arm"] in ARMS.values():
            if (r["cell"], r["arm"]) in score:
                raise ValueError(f"duplicate {r['cell']} {r['arm']}")
            score[(r["cell"], r["arm"])] = r["score"]
    cells = sorted({c for c, a in score if all((c, x) in score for x in ARMS.values())})
    env_of = {r["cell"]: r["environment"] for r in shared if r["cell"] in set(cells)}
    task_of = {r["cell"]: r["main_task"] for r in shared if r["cell"] in set(cells)}
    i3_cells = sorted(c for c, v in i3["L63"].items() if v["environment"] in ENVS)
    if not set(i3_cells) <= set(cells):
        raise ValueError("an I3 cell has no stored scores")
    layer = None
    for lk in ("L63", "A"):
        if all(abs(score[(c, ARMS[k])] - i3[lk][c][k]) <= TOL for c in i3_cells for k in ARMS):
            layer = lk
            break
    if layer is None:
        raise SystemExit("stored scores match neither of I3's layers; stop")
    per_env = Counter(env_of[c] for c in i3_cells)
    by_env = {e: [c for c in cells if env_of[c] == e] for e in ENVS}

    def aurocs(cs):
        g = lambda k: np.array([score[(c, ARMS[k])] for c in cs])  # noqa: E731
        return {"R2_vs_R1": auc(g("R1"), g("R2")), "S1_vs_R1": auc(g("R1"), g("S1"))}

    def aurocs_i3(runs):
        cs = sorted(runs)
        g = lambda k: np.array([runs[c][k] for c in cs])  # noqa: E731
        return {"R2_vs_R1": auc(g("R1"), g("R2")), "S1_vs_R1": auc(g("R1"), g("S1"))}

    rng = np.random.default_rng(SEED)
    subs = [aurocs([c for e in ENVS for c in rng.choice(by_env[e], per_env[e], replace=False)]) for _ in range(N_SUB)]
    dist = {k: [s[k] for s in subs] for k in ("R2_vs_R1", "S1_vs_R1")}
    own = aurocs(i3_cells)
    out = {"label": "POST HOC: subset check of I3's frozen-probe AUROCs (stored scores; 4 test environments, tiktok not stored)",
           "layer_matched": layer, "population_cells": len(cells), "i3_cells_in_these_envs": len(i3_cells),
           "per_env_draw": dict(per_env), "full_population": aurocs(cells), "i3_78": own,
           "subset_reference": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                                    "pct2.5_97.5": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                                for k, v in dist.items()},
           "i3_78_placement": {k: place(own[k], dist[k]) for k in dist},
           "i3_97_placement_matched_layer": {k: place(v, dist[k]) for k, v in aurocs_i3(i3[layer]).items()}
                                            | {"note": "I3's 97 cells at the matched layer, from I3's own scores; they include "
                                               "tiktok, so this is against the 4-environment reference for scale only. The other "
                                               "layer has no stored all-cell scores, so its 97-cell values are not placed."}}
    out["unusual_draw"] = any(v["p_two_sided"] < 0.05 for v in out["i3_78_placement"].values())
    # post hoc, within-cell: AUROC(S1 over R1) - AUROC(S2 over R1) on I3's 97 cells, paired main-task bootstrap
    paired = {}
    for lk in ("L63", "A"):
        cs = sorted(i3[lk])
        R1, S1, S2 = (np.array([i3[lk][c][k] for c in cs]) for k in ("R1", "S1", "S2"))
        tasks = np.array([i3[lk][c]["main_task"] for c in cs])
        names = sorted(set(tasks))
        members = [np.flatnonzero(tasks == t) for t in names]
        brng, diffs = np.random.default_rng(SEED), []
        for _ in range(N_BOOT):
            i = np.concatenate([members[j] for j in brng.choice(len(names), len(names))])
            diffs.append(auc(R1[i], S1[i]) - auc(R1[i], S2[i]))
        paired[lk] = {"S1_vs_R1": auc(R1, S1), "S2_vs_R1": auc(R1, S2), "difference": auc(R1, S1) - auc(R1, S2),
                      "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    out["posthoc_paired_S1_minus_S2_vs_R1"] = paired
    out |= {"inputs": {str(WS): hashlib.sha256(ws_raw).hexdigest(), str(I3): hashlib.sha256(i3_raw).hexdigest()},
            "seed": SEED, "subsets": N_SUB,
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"reroll_subsets_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "git", "script_sha256")}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
