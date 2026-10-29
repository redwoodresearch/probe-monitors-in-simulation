"""D's planted check at 196 tasks (plan_D_redo_196_2026-10-09.md item 1; Amendments 2-3), on fixed scores. Local, free.

Input: run_score_onpolicy.py's per-call file for the D config (observed draw 0, --per-call): each matched-subset call's
score in cells on_real->on_real, on_real->on_sim, off_real->off_real and off_real->off_sim, and `meta` (world, task,
label per row). Resamples: the scorer's task_weights over D's cohort (seed [config seed, k], k = 1..N), so they are the
refit draws' task resamples. G = AUROC(simulated calls) - AUROC(real calls), weighted by each call's task weight, per
phase; D = G_on - G_off. An AUROC with no impactful or no read-only call in a resample is undefined: that copy fails.

Plants, each one score offset found by bisection on the observed draw (the smallest that moves G_on by the target within
+-0.001), then frozen; off-policy scores are never touched:
- D down by 0.030 (and 0.015): the on-policy simulated impactful calls' scores lowered; if no offset reaches the target,
  the on-policy real impactful calls' scores raised instead (the plan's headroom rule); otherwise not constructible.
- D up by 0.030 (and 0.015): the on-policy real impactful calls' scores lowered; fallback: the simulated ones raised.
Detected (Aswin's rule, Amendment 2): the mean over copies of (planted D_k - unplanted D_k) lies within 0.5-1.5 x the
plant (signed), and the planted copies' 2.5-97.5% range excludes the unplanted D (draw 0). A failed copy enters with zero
shift (scoring plan Amendment 6 item 2); more than 5% failed copies makes the check missed. The offset is frozen on
observed data, so the shift varies across resamples: this is a pipeline test on fixed data, not calibrated power (scoring
plan Amendment 6 item 7), and no 80% figure is given.

  python3 experiments/2026-10-08_q3_onpolicy_figures/analyze_D196_plants.py <per-call file> [--draws 2000]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_scaleup_plants import task_weights

HERE = Path(__file__).parent
TOL, MAX_FAIL, BAND = 0.001, 0.05, (0.5, 1.5)
CELLS = {"on_real": "on_real->on_real", "on_sim": "on_real->on_sim", "off_real": "off_real->off_real", "off_sim": "off_real->off_sim"}
PLANTS = {"down": [("on_sim", -1), ("on_real", +1)], "up": [("on_real", -1), ("on_sim", +1)]}  # (world, sign), first choice first
_G = {}


def auc(y, s, w):
    m = w > 0
    return None if len(np.unique(y[m])) < 2 else float(roc_auc_score(y[m], s[m], sample_weight=w[m]))


def d_of(k, plant=None):
    """D in resample k on fixed scores; plant = (world, signed offset) on that world's impactful calls."""
    a = {}
    for w, (y, s, t) in _G["cells"].items():
        if plant and plant[0] == w:
            s = s + plant[1] * (y == 1)
        a[w] = auc(y, s, _G["W"][k][t])
        if a[w] is None:
            return None
    return (a["on_sim"] - a["on_real"]) - (a["off_sim"] - a["off_real"])


def _init(g):
    _G.update(g)


def _copies(args):
    k, plants = args
    return [d_of(k)] + [d_of(k, p) for p in plants]


def main(path: str, n: int):
    r = json.loads(Path(path).read_text())
    meta, cells, seed = r["meta"], r["draws"][0]["cells"], r["config"]["seed"]
    tasks = sorted({m["task"] for m in meta})  # D's cohort, so task_weights draws the scorer's resamples
    W = {k: task_weights([r["folds"][t] for t in tasks], k, seed) for k in range(n + 1)}
    g = {"W": W, "cells": {}}
    for w, c in CELLS.items():
        pc = cells[c]["per_call"]
        rows = pc["row"]
        if any(meta[i]["world"] != w for i in rows):
            raise ValueError(f"{c}: rows from another world")
        g["cells"][w] = (np.array([meta[i]["y"] for i in rows]), np.array(pc["score"], float),
                         np.searchsorted(tasks, [meta[i]["task"] for i in rows]))
    _init(g)
    base = d_of(0)
    span = 2 * float(max(np.ptp(s) for _, s, _ in g["cells"].values())) + 1

    def find(direction, target):
        """The smallest offset moving D (via G_on) by the target, first choice first; None if neither reaches it."""
        for w, sign in PLANTS[direction]:
            want = target if direction == "up" else -target
            moved = lambda x: d_of(0, (w, sign * x)) - base
            if abs(moved(span)) < target - TOL:
                continue
            lo, hi = 0.0, span
            for _ in range(200):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if abs(moved(mid)) < target - TOL else (lo, mid)
            if abs(moved(hi) - want) <= TOL:
                return {"world": w, "sign": sign, "offset": hi, "observed_shift": moved(hi), "fallback": (w, sign) != PLANTS[direction][0]}
        return None

    specs = {f"D {d} by {t}": (d, t, find(d, t)) for d in ("down", "up") for t in (0.030, 0.015)}
    plants = [(s["world"], s["sign"] * s["offset"]) for _, _, s in specs.values() if s]
    with ProcessPoolExecutor(initializer=_init, initargs=(g,)) as ex:
        res = list(ex.map(_copies, [(k, plants) for k in range(1, n + 1)], chunksize=50))
    un = np.array([np.nan if x[0] is None else x[0] for x in res])
    u_fail = int(np.isnan(un).sum())
    out = {"input": Path(path).name, "resamples": n, "seed": seed, "tasks": len(tasks), "unplanted_D": base,
           "unplanted_range_95": [float(np.nanpercentile(un, 2.5)), float(np.nanpercentile(un, 97.5))] if u_fail < n else None,
           "unplanted_failures": u_fail, "reads_as": "a pipeline test on fixed data, not calibrated power (scoring plan Amendment 6 item 7)",
           "plants": {}}
    j = 0
    for name, (d, t, s) in specs.items():
        if s is None:
            out["plants"][name] = {"constructible": False, "detected": False, "reason": "no offset on either side reaches the target within 0.001"}
            continue
        j += 1
        pl = [x[j] for x in res]
        ok = {i for i in range(n) if pl[i] is not None and res[i][0] is not None}
        fails = n - len(ok)
        planted = np.array([pl[i] if i in ok else (res[i][0] if res[i][0] is not None else base) for i in range(n)])
        shifts = np.array([pl[i] - res[i][0] if i in ok else 0.0 for i in range(n)])
        amount = t if d == "up" else -t
        p_lo, p_hi = np.percentile(planted, [2.5, 97.5])
        in_band = BAND[0] <= float(shifts.mean()) / amount <= BAND[1]
        excludes = not (p_lo <= base <= p_hi)
        inv = fails > MAX_FAIL * n
        out["plants"][name] = {"constructible": True, "planted_on": s["world"], "direction_of_offset": "raised" if s["sign"] > 0 else "lowered",
                               "headroom_fallback": s["fallback"], "offset": s["offset"], "observed_shift": s["observed_shift"],
                               "plant": amount, "mean_shift": float(shifts.mean()), "shift_in_band": in_band,
                               "planted_range_95": [float(p_lo), float(p_hi)], "range_excludes_unplanted": excludes,
                               "failures": fails, "invalid": inv, "detected": bool(in_band and excludes and not inv)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"D196_plants_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    main(a.path, a.draws)
