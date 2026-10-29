"""Q2.5 (post hoc, PLAN_POSTHOC.md): destructive (corrected labels) and transfer monitors against ±5 and ±10-point
margins at target 1% and 5% false alarms, with Holm over the 4 gaps, the real-vs-real control beside each, and a plant
sweep (2-15 points) as sensitivity. From the separate-calibration refit's stored scores; the task bootstrap (2,000
draws, seed 0) re-estimates the thresholds in every draw, as analyze_separate_calibration.py does, and the stored
point estimates and intervals are reproduced before anything else is computed.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_margins_POSTHOC/analyze_q25_POSTHOC.py <recal output>
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.margins import LEVEL, difference_p, equivalence_p, holm, inside  # noqa: E402
from probe_shift.planted import detected  # noqa: E402

HERE = Path(__file__).parent
STORED = HERE.parent / "2026-10-10_q2_metric_table" / "results" / "separate_calibration_2026-10-10_225753.json"
N_BOOT, MARGINS, TARGETS, SIZES, COPIES = 2000, (0.05, 0.10), (0.01, 0.05), range(2, 16), 200


def pct(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5], axis=-1)]


def thresholds(cs, w, q):
    """Per draw, the frozen convention on weighted negatives: sorted high to low, the value at index floor(q * n)."""
    order = np.argsort(-cs, kind="stable"); cs, w = cs[order], w[:, order]
    c = np.cumsum(w, axis=1); idx = np.floor(q * c[:, -1])
    if (c[:, -1] == 0).any():
        raise ValueError("a bootstrap draw left a calibration fold empty")
    return cs[np.argmax(c > idx[:, None], axis=1)]


def first_second(rows):
    """Real rows of tasks with two or more real runs: 1 for the first run (sorted run key), 2 for the second, else 0."""
    runs = {}
    for r in rows:
        if r["world"] == 0:
            runs.setdefault(r["task"], set()).add(r["run"])
    order = {run: i + 1 for rs in runs.values() if len(rs) >= 2 for i, run in enumerate(sorted(rs)[:2])}
    return np.array([order.get(r["run"], 0) if r["world"] == 0 else 0 for r in rows])


def main(path: Path):
    full = json.loads(path.read_text()); rows = full["rows"]; stored = json.loads(STORED.read_text())
    world = np.array([r["world"] for r in rows]); fold = np.array([r["fold"] for r in rows]); task = np.array([r["task"] for r in rows])
    names = sorted(set(task)); ti = np.searchsorted(names, task); fs = first_second(rows)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)) for _ in range(N_BOOT)])
    W = np.vstack([np.ones(len(names), int), W])  # row 0: the point estimate
    out = {"input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "stored result": str(STORED),
           "margins (points)": [100 * m for m in MARGINS], "level for inside after Holm": LEVEL}
    gaps, family = {}, {}
    for name in ("destructive_raw", "transfer"):
        y = np.array([r[name] for r in rows]); s = np.array(full["label sets"][name]["test scores"], float)
        cal = {int(k): (np.array(v["rows"]), np.array(v["scores"])) for k, v in full["label sets"][name]["calibration negatives"].items()}
        for q in TARGETS:
            thr = np.stack([thresholds(cal[k][1], W[:, ti[cal[k][0]]], q) for k in range(5)], axis=1)  # draws x folds
            def parts(m):
                idx = np.flatnonzero(m); wc = W[:, ti[idx]] * (s[idx][None, :] > thr[:, fold[idx]])
                return idx, wc, W[:, ti[idx]].sum(1)

            def rate(m):
                _, wc, den = parts(m)
                return wc.sum(1) / den
            pos = y == 1
            g = rate((world == 0) & pos) - rate((world == 1) & pos)
            rvr = rate((fs == 1) & pos) - rate((fs == 2) & pos)
            key = f"{name} target {int(q * 100)}%"
            ref = stored[name][f"target {int(q * 100)}% catch gap real minus simulated"]
            if abs(g[0] - ref["point"]) > 1e-9 or np.abs(np.array(pct(g[1:])) - ref["95% (task bootstrap, thresholds re-estimated)"]).max() > 1e-9:
                raise ValueError(f"{key}: does not reproduce the stored gap {ref}")
            gaps[key] = g; family[key] = g[1:]
            real = rate((world == 0) & pos); simpos, wc, den = parts((world == 1) & pos); num = wc.sum(1)
            caught0 = np.flatnonzero(wc[0] > 0)  # columns of simulated positives caught at the point thresholds
            sweep = {}
            for size in SIZES:
                k = round(size * len(simpos) / 100); hits = 0
                for c in range(COPIES):
                    kill = np.random.default_rng(c).choice(caught0, k, replace=False)
                    gp = real - (num - wc[:, kill].sum(1)) / den
                    hits += detected(gp[0], tuple(pct(gp[1:])), g[0], k / len(simpos))
                sweep[size] = hits / COPIES
            out[key] = {"gap": {"point": float(g[0]), "95%": pct(g[1:])},
                        "real-vs-real (run 1 minus run 2)": {"point": float(rvr[0]), "95%": pct(rvr[1:]),
                                                             "inside": {f"±{int(100 * m)}": inside(pct(rvr[1:]), m) for m in MARGINS},
                                                             "tasks": len(set(task[fs == 1])), "positives run 1, run 2": [int(((fs == 1) & pos).sum()), int(((fs == 2) & pos).sum())]},
                        "plant sweep (detection rate per size in points)": sweep,
                        "smallest size detected in at least 80% of copies": next((z for z, r in sweep.items() if r >= 0.8), None)}
    for m in MARGINS:
        eq = holm({k: equivalence_p(v, m) for k, v in family.items()})
        for k, v in family.items():
            out[k][f"margin ±{int(100 * m)}"] = {"inside (95% interval)": inside(out[k]["gap"]["95%"], m), "equivalence p": equivalence_p(v, m),
                                                 "Holm-adjusted equivalence p": eq[k], "inside after Holm": eq[k] < LEVEL}
    dp = holm({k: difference_p(v) for k, v in family.items()})
    for k, v in family.items():
        out[k]["difference from 0"] = {"p": difference_p(v), "Holm-adjusted p": dp[k]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"q25_margins_POSTHOC_{stamp}.json"; dest.parent.mkdir(exist_ok=True)
    with open(dest, "x") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    for k in family:
        o = out[k]; f = lambda d: f"{100 * d['point']:+.1f} [{100 * d['95%'][0]:+.1f}, {100 * d['95%'][1]:+.1f}]"
        print(f"{k}: gap {f(o['gap'])}; real-vs-real {f(o['real-vs-real (run 1 minus run 2)'])}; "
              + "; ".join(f"±{int(100 * m)}: inside {o[f'margin ±{int(100 * m)}']['inside (95% interval)']}, Holm p {o[f'margin ±{int(100 * m)}']['Holm-adjusted equivalence p']:.4f}" for m in MARGINS)
              + f"; diff Holm p {o['difference from 0']['Holm-adjusted p']:.3f}; 80% plant {o['smallest size detected in at least 80% of copies']}")
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
