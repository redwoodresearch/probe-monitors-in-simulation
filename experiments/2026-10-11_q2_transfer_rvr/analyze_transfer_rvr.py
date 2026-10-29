"""Data-transfer monitor on Sol's honest runs: real run 1 vs run 2, per task and pooled, top-5 share, smallest detectable
gap, and the pairs needed for a ±5-point range (DEFINITIONS.md, committed d736315b before any number; exploratory, $0).
From the separate-calibration refit's full output; 2,000 task resamples (seed 0) re-estimate each fold's threshold.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_transfer_rvr/analyze_transfer_rvr.py <recal output json>
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
N_BOOT, TARGETS, MDE_Z = 2000, (0.01, 0.05), 1.96 + 0.84


def thresholds(cs, w, q):
    """Per draw, the frozen convention on weighted negatives (as Q2.5): sorted high to low, the value at index floor(q n)."""
    order = np.argsort(-cs, kind="stable"); cs, w = cs[order], w[:, order]
    c = np.cumsum(w, axis=1); idx = np.floor(q * c[:, -1])
    return cs[np.argmax(c > idx[:, None], axis=1)]


def main(path: Path):
    full = json.loads(path.read_text()); rows = full["rows"]
    world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows]); task = np.array([r["task"] for r in rows])
    y = np.array([r["transfer"] for r in rows]) == 1; s = np.array(full["label sets"]["transfer"]["test scores"], float)
    cal = {int(k): (np.array(v["rows"]), np.array(v["scores"])) for k, v in full["label sets"]["transfer"]["calibration negatives"].items()}
    runs = {}
    for r in rows:
        if r["world"] == 0:
            runs.setdefault(r["task"], set()).add(r["run"])
    first2 = {t: sorted(v)[:2] for t, v in runs.items() if len(v) >= 2}
    batch = lambda run: "overlap800" if "|overlap800_" in run else "base200" if "|base200_" in run else "other"
    same = {t for t, (a, b) in first2.items() if batch(a) == batch(b) == "overlap800"}
    which = {run: i + 1 for t, v in first2.items() for i, run in enumerate(v)}
    rk = np.array([which.get(r["run"], 0) if r["world"] == 0 else 0 for r in rows])
    names = sorted(set(task)); ti = np.searchsorted(names, task); T = len(names)
    rng = np.random.default_rng(0)
    W = np.vstack([np.ones(T, int), np.stack([np.bincount(rng.integers(0, T, T), minlength=T) for _ in range(N_BOOT)])])
    out = {"input": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tasks": T, "real tasks with two runs": len(first2),
           "same-batch pairs (both overlap800)": len(same), "real tasks with one run": sum(len(v) == 1 for v in runs.values())}
    for q in TARGETS:
        thr = np.stack([thresholds(cal[k][1], W[:, ti[cal[k][0]]], q) for k in range(5)], axis=1)  # draws x folds
        hit = s[None, :] > thr[:, fold]  # draws x rows
        def per_task(m):  # draws x tasks: weighted caught and positives of mask m
            c = np.zeros((N_BOOT + 1, T)); n = np.zeros(T)
            idx = np.flatnonzero(m & y)
            np.add.at(c.T, ti[idx], hit[:, idx].T); np.add.at(n, ti[idx], 1)
            return c, n
        res = {}
        for label, tasks in (("same-batch pairs", same), ("all pairs", set(first2))):
            inset = np.isin(task, list(tasks))
            c1, n1 = per_task(inset & (rk == 1)); c2, n2 = per_task(inset & (rk == 2))
            pooled = (W * c1).sum(1) / (W * n1).sum(1) - (W * c2).sum(1) / (W * n2).sum(1)
            both = (n1 > 0) & (n2 > 0)
            per = ((W * np.where(both, c1 / np.maximum(n1, 1) - c2 / np.maximum(n2, 1), 0)).sum(1) / (W * both).sum(1))
            contrib = c1[0] / n1.sum() - c2[0] / n2.sum()
            sign = np.sign(pooled[0]) or 1.0
            top = np.argsort(-sign * contrib)[:5]
            res[label] = {"pooled gap": {"point": float(pooled[0]), "95%": [float(v) for v in np.percentile(pooled[1:], [2.5, 97.5])],
                                         "smallest detectable (2.80 SD)": float(MDE_Z * pooled[1:].std())},
                          "per-task gap": {"point": float(per[0]), "95%": [float(v) for v in np.percentile(per[1:], [2.5, 97.5])], "tasks": int(both.sum()),
                                           "smallest detectable (2.80 SD)": float(MDE_Z * per[1:].std())},
                          "positives run 1, run 2": [int(n1.sum()), int(n2.sum())],
                          "top-5 share of the pooled gap (contributions with its sign)": float(contrib[top].sum() / pooled[0]) if pooled[0] else None,
                          "top-5 absolute contributions (points)": [round(100 * float(x), 2) for x in np.sort(np.abs(contrib))[::-1][:5]]}
            if label == "same-batch pairs":
                h = (res[label]["pooled gap"]["95%"][1] - res[label]["pooled gap"]["95%"][0]) / 2
                need = int(np.ceil(len(tasks) * (h / 0.05) ** 2))
                res[label]["pairs needed for ±5 (95% half-width)"] = {"current half-width (points)": round(100 * h, 2), "pairs now": len(tasks),
                                                                      "pairs needed": need, "more pairs": max(0, need - len(tasks))}
        cr, nr = per_task(world == 0); cs_, ns = per_task(world == 1)
        pooled_rs = (W * cr).sum(1) / (W * nr).sum(1) - (W * cs_).sum(1) / (W * ns).sum(1)
        both = (nr > 0) & (ns > 0)
        per_rs = (W * np.where(both, cr / np.maximum(nr, 1) - cs_ / np.maximum(ns, 1), 0)).sum(1) / (W * both).sum(1)
        pooled_117 = (W * cr * both).sum(1) / (W * nr * both).sum(1) - (W * cs_ * both).sum(1) / (W * ns * both).sum(1)
        res["real (both runs) vs simulated"] = {"pooled gap": {"point": float(pooled_rs[0]), "95%": [float(v) for v in np.percentile(pooled_rs[1:], [2.5, 97.5])],
                                                               "smallest detectable (2.80 SD)": float(MDE_Z * pooled_rs[1:].std()), "tasks": "every task (as Q2.5)"},
                                                "pooled gap, tasks with positives in both worlds (fix, Codex on #718)": {
                                                    "point": float(pooled_117[0]), "95%": [float(v) for v in np.percentile(pooled_117[1:], [2.5, 97.5])],
                                                    "smallest detectable (2.80 SD)": float(MDE_Z * pooled_117[1:].std()), "tasks": int(both.sum())},
                                                "per-task gap": {"point": float(per_rs[0]), "95%": [float(v) for v in np.percentile(per_rs[1:], [2.5, 97.5])],
                                                                 "tasks": int(both.sum()), "smallest detectable (2.80 SD)": float(MDE_Z * per_rs[1:].std())}}
        out[f"target {int(100 * q)}%"] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"transfer_rvr_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps(out, indent=1))
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
