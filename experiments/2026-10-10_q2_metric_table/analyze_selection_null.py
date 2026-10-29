"""Selection-aware shuffled-label null for the corrected destructive probe, the analysis (local, free; committed with
its reading rule before the job launched).

Reading rule (Aswin 2026-10-10: a secondary robustness check, not confirmation; the main conclusion stays on the
real-vs-simulated catch gap):
- Statistic: fold-mean AUROC per world on `destructive_raw`, from run_selection_null.py (20 within-run shuffles, seeds
  1-20, layer and C chosen again in every draw), against the real-label refit's values in
  `results/separate_calibration_2026-10-10_225753.json` (0.920 real, 0.924 simulated).
- p per world = (1 + draws with null AUROC >= observed) / 21. The smallest possible p is 1/21 (0.048).
- If p = 1/21 in both worlds: "the corrected-label AUROC exceeds all 20 selection-aware shuffled-label draws
  (p = 1/21)". Otherwise report the draws that reach it and flag the result.
- The draws' catch at target 1% and 5% (thresholds from each test fold's calibration fold, as in the refit) and the
  real minus simulated gap are reported descriptively, with no p: a label shuffle tests whether activations predict
  the label, not whether the worlds differ.
- Prediction: null AUROC means 0.60 ± 0.05 in each world (fixed-probe null 0.60 and 0.62; the old labels'
  selection-aware draws 0.58-0.64), every draw below 0.70, p = 1/21 in both worlds.
- All 100 (draw, fold) fits must be present, every world and fold must hold both classes in every draw, and the labels
  must match the refit's (same labels sha256 and positives); otherwise it raises.

  uv run --no-project --with numpy --with scikit-learn python experiments/2026-10-10_q2_metric_table/analyze_selection_null.py <null output>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from analyze_separate_calibration import metrics

HERE = Path(__file__).parent
OBSERVED = HERE / "results" / "separate_calibration_2026-10-10_225753.json"
LABEL = "destructive_raw"


def main(path: Path):
    full = json.loads(path.read_text())
    rows = full["rows"]
    world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows])
    obs_all = json.loads(OBSERVED.read_text()); obs = obs_all[LABEL]
    y0 = np.array([r[LABEL] for r in rows])
    pos = [int(((world == w) & (y0 == 1)).sum()) for w in (0, 1)]
    if full.get("labels sha256") != obs_all["labels sha256"] or pos != obs["positives real, simulated"]:
        raise ValueError(f"labels differ from the refit's: sha256 {full.get('labels sha256')} vs {obs_all['labels sha256']}, positives {pos} vs {obs['positives real, simulated']}")
    if sorted(int(d) for d in full["draws"]) != list(range(1, 21)):
        raise ValueError(f"draws {sorted(full['draws'])}, expected 1-20")
    draws = []
    for d, v in sorted(full["draws"].items(), key=lambda kv: int(kv[0])):
        y = np.array([int(c) for c in v["labels"]]); s = np.array(v["test scores"], dtype=float)
        if np.isnan(s).any() or len(v["calibration negatives"]) != 5:
            raise ValueError(f"draw {d}: {int(np.isnan(s).sum())} rows without a test score, {len(v['calibration negatives'])} calibration folds")
        one_class = [(w, k) for w in (0, 1) for k in range(5) if len(set(y[(world == w) & (fold == k)])) < 2]
        if one_class or (y.sum(), len(y)) != (y0.sum(), len(y0)):
            raise ValueError(f"draw {d}: single-class (world, fold) {one_class}, or its label count differs from the real labels'")
        cal = {int(k): (np.array(c["rows"]), np.array(c["scores"])) for k, c in v["calibration negatives"].items()}
        m = metrics(y, s, cal, world, fold, np.ones(len(y), int))
        if m is None or not all(np.isfinite(x) for x in m.values()):
            raise ValueError(f"draw {d}: metrics missing or not finite: {m}")
        draws.append({"draw": int(d), **m, "picks": v["picks"]})
    out = {"input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "observed": str(OBSERVED), "label": LABEL}
    for wn in ("real", "simulated"):
        k = f"AUROC {wn}"; null = np.array([d[k] for d in draws])
        out[k] = {"observed": obs[k]["point"], "null mean": float(null.mean()), "null max": float(null.max()),
                  "draws at or above observed": int((null >= obs[k]["point"]).sum()), "p": float((1 + (null >= obs[k]["point"]).sum()) / (1 + len(null)))}
    for q in (1, 5):
        for kind in ("catch real", "catch simulated", "catch gap real minus simulated", "achieved false alarms real", "achieved false alarms simulated"):
            k = f"target {q}% {kind}"; null = np.array([d[k] for d in draws])
            out[f"{k} (descriptive)"] = {"observed": obs[k]["point"], "null mean": float(null.mean()), "null range": [float(null.min()), float(null.max())]}
    out["draws"] = draws
    dest = HERE / "results" / f"selection_null_{full.get('stamp', path.stem)}.json"
    with open(dest, "x") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    for wn in ("real", "simulated"):
        o = out[f"AUROC {wn}"]
        print(f"AUROC {wn}: observed {o['observed']:.3f}; null mean {o['null mean']:.3f}, max {o['null max']:.3f}; {o['draws at or above observed']} of 20 at or above; p {o['p']:.4f}")
    for q in (1, 5):
        o = out[f"target {q}% catch gap real minus simulated (descriptive)"]
        print(f"target {q}% gap (descriptive): observed {o['observed']:+.3f}; null mean {o['null mean']:+.3f}, range [{o['null range'][0]:+.3f}, {o['null range'][1]:+.3f}]")
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
