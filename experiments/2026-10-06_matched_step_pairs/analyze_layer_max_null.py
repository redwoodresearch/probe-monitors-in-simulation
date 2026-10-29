"""A null for "the best single layer" of the fresh within-pairs probe, from saved scores (CPU, seconds; no refits).

analyze_probes.fresh refit the leave-one-environment-out probe at every layer for 100 within-pair label swaps (seed 0).
For each draw, take the highest AUROC over all 64 layers against that draw's swapped labels; compare the real labels'
best layer with that distribution. This is a max-statistic permutation test: the same pick-the-best-layer step runs on
real and swapped labels. It is NOT a null for the nested score (each held-out environment at its inner-CV layer), which
would need the inner CV refit for every draw; the nested rows' Null column stays the per-layer (fixed-layer) null.

  python experiments/2026-10-06_matched_step_pairs/analyze_layer_max_null.py --probes data/probe_scores/probe_scores_<stamp>.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    args = ap.parse_args()
    P = json.loads(args.probes.read_text())
    out = {"probes_file": args.probes.name, "pairs_file": P["pairs_file"], "readouts": {}}
    for kind in ("output_end", "call_end"):
        r = P["fresh"][kind]
        pair = np.array([p for p, _ in r["items"]])
        y = np.array([float(v == "sim") for _, v in r["items"]])
        layers = sorted(r["layers"], key=int)
        real = {l: roc_auc_score(y, r["layers"][l]["scores"]) for l in layers}
        best_layer = max(real, key=real.get)
        rng = np.random.default_rng(0)  # analyze_probes.fresh's draws: one per null column, pairs in sorted order
        up = np.unique(pair)
        draws = []
        for d in range(len(r["layers"][layers[0]]["null_scores"])):
            flip = dict(zip(up, rng.random(len(up)) < 0.5))
            yn = np.where([flip[p] for p in pair], 1 - y, y)
            draws.append(max(roc_auc_score(yn, r["layers"][l]["null_scores"][d]) for l in layers))
        draws = np.array(draws)
        out["readouts"][kind] = {"best_layer": int(best_layer), "observed_auroc": round(float(real[best_layer]), 4),
                                 "null_mean": round(float(draws.mean()), 4), "null_p95": round(float(np.percentile(draws, 95)), 4),
                                 "p_value": round(float((1 + (draws >= real[best_layer]).sum()) / (1 + len(draws))), 4),
                                 "n_draws": len(draws)}
        print(kind, out["readouts"][kind])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"layer_max_null_{stamp}.json", "x") as f:
        f.write(json.dumps(out, indent=1))
    print(f"wrote results/layer_max_null_{stamp}.json")


if __name__ == "__main__":
    main()
