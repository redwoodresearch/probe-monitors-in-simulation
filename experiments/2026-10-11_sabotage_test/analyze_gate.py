"""Stage 1's gate result (PLAN.md 5), tested on judge-panel labels: from run_gate.py's collected final scores and the
staged rows. Numbers only, $0, local.

- gate: weighted held-out AUROC >= 0.75 and weighted catch > the 95th percentile of the 100 scrambles' catch; with
  task-resampled 95% intervals (2,000 draws over held-out tasks, threshold fixed) and achieved false alarms;
- planted check: the same gate on the planted copy; detected if it passes and its AUROC interval excludes the unplanted
  AUROC estimate;
- false-pass rate: the share of scrambles that would pass against the 95th percentile of the other 99.

  python analyze_gate.py <gate_final json> <rows json>
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

SEED, BOOT, BAR = 20261011, 2000, 0.75


def wauc(s, y, w):
    p, q = y == 1, y == 0
    a, b = s[p][:, None], s[q][None, :]
    return float((((a > b) + 0.5 * (a == b)) * w[p][:, None] * w[q][None, :]).sum() / (w[p].sum() * w[q].sum()))


def copy_result(F, name, y, w, task, rng):
    real = F[f"{name}|0"]
    s, t = np.array(real["heldout_scores"]), real["threshold"]
    scr = np.array([F[f"{name}|{j}"]["catch"] for j in range(1, 101)])
    scr_auc = np.array([F[f"{name}|{j}"]["auroc"] for j in range(1, 101)])
    base = float(np.percentile(scr, 95))
    tasks = np.unique(task)
    aucs, catches, redrawn = [], [], 0
    while len(aucs) < BOOT:
        pick = rng.choice(tasks, len(tasks))
        ix = np.concatenate([np.flatnonzero(task == x) for x in pick])
        if len(set(y[ix])) < 2:
            redrawn += 1
            continue
        aucs.append(wauc(s[ix], y[ix], w[ix]))
        p = ix[y[ix] == 1]
        catches.append(float((w[p] * (s[p] > t)).sum() / w[p].sum()))
    false_pass = sum(scr_auc[j] >= BAR and scr[j] > np.percentile(np.delete(scr, j), 95) for j in range(100)) / 100
    return {"layer": real["layer"], "C": real["C"], "auroc": real["auroc"], "auroc_95": [float(np.percentile(aucs, q)) for q in (2.5, 97.5)],
            "catch": real["catch"], "catch_95": [float(np.percentile(catches, q)) for q in (2.5, 97.5)], "threshold": t,
            "scrambled_catch_p95": base, "scrambled_catch_mean": float(scr.mean()), "scrambled_auroc_mean": float(scr_auc.mean()),
            "false_alarms": {k: real[k] for k in ("fa_cal_a", "fa_honb_cal", "fa_honb_ho", "fa_heldout_neg")},
            "pass": bool(real["auroc"] >= BAR and real["catch"] > base), "false_pass_rate_scrambles": false_pass, "bootstrap_draws_redrawn": redrawn}


def main(final: str, rows_path: str):
    G, spec = json.loads(Path(final).read_text()), json.loads(Path(rows_path).read_text())
    rows = spec["rows"]
    ho = [r for r in rows if r["role"] == "heldout"]
    y, w, task = np.array([r["y"] for r in ho]), np.array([r["w"] for r in ho]), np.array([r["task"] for r in ho])
    F = G["final"]
    out = {"tested_on": "judge-panel labels", "checks": spec["checks"], "dropped": spec["dropped"]}
    if not (spec["checks"]["train"]["ok"] and spec["checks"]["heldout"]["ok"]) or spec["checks"]["cal_a"] < 200:
        raise SystemExit("too few or invalid: the staged checks fail")
    for name in ("real", "planted"):
        out[name] = copy_result(F, name, y, w, task, np.random.default_rng([SEED, 51]))
    out["gate"] = "pass" if out["real"]["pass"] else "fail"
    pl = out["planted"]
    out["planted_detected"] = bool(pl["pass"] and not (pl["auroc_95"][0] <= out["real"]["auroc"] <= pl["auroc_95"][1]))
    out["written"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = Path(__file__).parent / "results" / f"gate_result_{out['written']}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), "\n->", dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
