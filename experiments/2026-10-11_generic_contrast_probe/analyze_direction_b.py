"""PLAN.md amendment 1 (A1.2), local: direction B's layer rule and validity check (keys and numbers only).

Layer: among layers 16-48, the highest between/within variance ratio of the 8 layer-choice pairs' danger vs benign
projections onto B; ties nearest 32, then the lower layer. Validity: on the 8 validity pairs (used once), danger above
benign with AUROC >= 0.90 along B at that layer. The read token is the same "." in both classes, so answer polarity
cannot pass this check.

  python analyze_direction_b.py <run_contrast_b output json>
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

LAYER_RANGE, GATE = range(16, 49), 0.90


def auc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def main(path):
    d = json.loads(Path(path).read_text())
    pr = {k: np.array(v) for k, v in d["proj"].items()}  # [pair, layer]
    if len({r["token_id"] for r in d["items"]}) != 1:
        raise SystemExit("read token differs between items")
    if not all(np.isfinite(v).all() for v in pr.values()):
        raise SystemExit("non-finite projections")
    ratio = {}
    for L in LAYER_RANGE:
        a, b = pr["layer_danger"][:, L], pr["layer_benign"][:, L]
        within, between = (a.var(ddof=1) + b.var(ddof=1)) / 2, (a.mean() - b.mean()) ** 2
        ratio[L] = float(between / within) if within > 0 else (float("inf") if between > 0 else 0.0)  # no separation: 0, never inf
    best = max(ratio.values())
    chosen = sorted([L for L, x in ratio.items() if x == best], key=lambda L: (abs(L - 32), L))[0]
    val = auc(pr["validity_danger"][:, chosen], pr["validity_benign"][:, chosen])
    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "tracked_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())}
    print(json.dumps({"git": git, "written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
                      "contrast_file": Path(path).name, "split": d["split"], "read_token_id": d["items"][0]["token_id"],
                      "chosen_layer": chosen, "ratio": {str(L): round(x, 4) for L, x in ratio.items()},
                      "validity_auroc_at_chosen": val, "validity_auroc_at_32": auc(pr["validity_danger"][:, 32], pr["validity_benign"][:, 32]),
                      "gate": GATE, "valid": bool(val >= GATE)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
