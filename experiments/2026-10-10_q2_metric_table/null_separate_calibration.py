"""Separate-calibration refit: a fixed-probe label-shuffle null for its AUROCs (local, free; Codex on #478). Labels are
shuffled within each run (counts kept) on the stored test scores, 1,000 draws (seed 0), and the fold-mean AUROC is
recomputed per world. The probes were fit on the real labels and are not refit, so this is not selection-aware; a
selection-aware null would refit each draw (new paid work).

  uv run --no-project --with numpy --with scikit-learn python experiments/2026-10-10_q2_metric_table/null_separate_calibration.py <recal output>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent


def fold_auroc(y, s, world, fold, w):
    a = []
    for k in range(5):
        m = (world == w) & (fold == k)
        if len(set(y[m])) == 2:
            a.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(a))


def main(path: Path):
    full = json.loads(path.read_text()); rows = full["rows"]
    world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows]); run = np.array([r["run"] for r in rows])
    groups = [np.flatnonzero(run == r) for r in np.unique(run)]
    out = {"input sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "null": "fixed-probe, labels shuffled within run, 1,000 draws, seed 0; not selection-aware"}
    for name, ls in full["label sets"].items():
        y = np.array([r[name] for r in rows]); s = np.array(ls["test scores"], dtype=float)
        obs = {w: fold_auroc(y, s, world, fold, w) for w in (0, 1)}
        rng = np.random.default_rng(0); null = {0: [], 1: []}
        for _ in range(1000):
            yp = y.copy()
            for g in groups:
                yp[g] = rng.permutation(yp[g])
            for w in (0, 1):
                null[w].append(fold_auroc(yp, s, world, fold, w))
        out[name] = {wn: {"AUROC": obs[w], "null mean": float(np.mean(null[w])), "null 95th percentile": float(np.percentile(null[w], 95)),
                          "p": float((1 + np.sum(np.array(null[w]) >= obs[w])) / 1001)} for w, wn in ((0, "real"), (1, "simulated"))}
    dest = HERE / "results" / f"separate_calibration_null_{full.get('stamp', path.stem)}.json"
    with open(dest, "x") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
