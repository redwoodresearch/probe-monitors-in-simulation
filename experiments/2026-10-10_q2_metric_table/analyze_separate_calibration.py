"""Separate-calibration recompute, the analysis (local, free; committed before the scores): for each label set
(destructive on the corrected rule; transfer as frozen), AUROC (fold mean) per world, and the share caught at target 1%
and 5% false alarms with each test fold's threshold from its calibration fold's real negatives (the frozen
convention: negatives sorted high to low, the value at index floor(q * n)). Reported: catch and achieved false alarms
per world, the real minus simulated catch gap, all with a task bootstrap (2,000 draws, seed 0) that re-estimates the
thresholds from the resampled calibration negatives (fitted probes held fixed).

  uv run --no-project --with numpy --with scikit-learn python experiments/2026-10-10_q2_metric_table/analyze_separate_calibration.py <recal output>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
N_BOOT = 2000


def pct(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def topk(neg, q, w):
    v = np.sort(np.repeat(neg, w))[::-1]
    return v[int(np.floor(q * len(v)))] if len(v) else None


def metrics(y, s, cal, world, fold, wt):
    """AUROC per world (fold mean) and catch / achieved false alarms at target 1% and 5%; None if a calibration fold is
    empty in this draw."""
    res = {}
    for world_i, wn in ((0, "real"), (1, "simulated")):
        a = []
        for k in range(5):
            m = (world == world_i) & (fold == k) & (wt > 0)
            if len(set(y[m])) == 2:
                a.append(roc_auc_score(y[m], s[m], sample_weight=wt[m]))
        res[f"AUROC {wn}"] = float(np.mean(a))
    for q in (0.01, 0.05):
        c = np.zeros(len(y), bool)
        for k in range(5):
            ci, cs = cal[k]
            thr = topk(cs, q, wt[ci])
            if thr is None:
                return None
            c[fold == k] = s[fold == k] > thr
        for world_i, wn in ((0, "real"), (1, "simulated")):
            for lab, kind in ((1, "catch"), (0, "achieved false alarms")):
                m = (world == world_i) & (y == lab)
                res[f"target {int(q * 100)}% {kind} {wn}"] = float((wt[m] * c[m]).sum() / wt[m].sum())
        res[f"target {int(q * 100)}% catch gap real minus simulated"] = res[f"target {int(q * 100)}% catch real"] - res[f"target {int(q * 100)}% catch simulated"]
    return res


def main(path: Path):
    full = json.loads(path.read_text())
    rows = full["rows"]
    world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows]); task = np.array([r["task"] for r in rows])
    names = sorted(set(task)); ti = np.searchsorted(names, task)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)) for _ in range(N_BOOT)])
    out = {"input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "labels sha256": full.get("labels sha256")}
    for name, ls in full["label sets"].items():
        y = np.array([r[name] for r in rows]); s = np.array(ls["test scores"], dtype=float)
        if np.isnan(s).any():
            raise ValueError(f"{name}: {int(np.isnan(s).sum())} rows without a test score")
        cal = {int(k): (np.array(v["rows"]), np.array(v["scores"])) for k, v in ls["calibration negatives"].items()}

        point = metrics(y, s, cal, world, fold, np.ones(len(y), int))
        boot = [b for w in W if (b := metrics(y, s, cal, world, fold, w[ti])) is not None]
        out[name] = {"positives real, simulated": [int(((world == 0) & (y == 1)).sum()), int(((world == 1) & (y == 1)).sum())],
                     "bootstrap draws used": len(boot), "picks": ls["picks"], **{k: {"point": v, "95% (task bootstrap, thresholds re-estimated)": pct([b[k] for b in boot])} for k, v in point.items()}}
    dest = HERE / "results" / f"separate_calibration_{full.get('stamp', path.stem)}.json"
    with open(dest, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for name in full["label sets"]:
        o = out[name]
        g = lambda k, o=o: f"{o[k]['point']:.3f} [{o[k]['95% (task bootstrap, thresholds re-estimated)'][0]:.3f}, {o[k]['95% (task bootstrap, thresholds re-estimated)'][1]:.3f}]"
        print(f"{name}: positives {o['positives real, simulated']}; AUROC real {g('AUROC real')}, sim {g('AUROC simulated')}")
        for q in (1, 5):
            print(f"  target {q}%: catch real {g(f'target {q}% catch real')}, sim {g(f'target {q}% catch simulated')}, gap {g(f'target {q}% catch gap real minus simulated')}; "
                  f"false alarms real {g(f'target {q}% achieved false alarms real')}, sim {g(f'target {q}% achieved false alarms simulated')}")
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
