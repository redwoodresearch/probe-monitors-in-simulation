"""POST HOC: the redesigned planted check's sizing choice and validation result (PLANT_REDESIGN.md at eb44f8d3), tested on
judge-panel labels. Numbers only, $0, local.

  size <plant2_sizing json>: the smallest s > 0 whose mean (over outer folds) catch rise over s = 0 is >= 0.20.
  valid <plant2_valid json> <gate_final json> <rows json>: detection = the planted gate passes (AUROC >= 0.75 and catch >
    the 95th percentile of its 100 scrambles) AND catch rise over the recorded unplanted catch >= 0.10 AND a paired
    task-bootstrap 95% interval of the rise (2,000 draws over held-out tasks, both copies' thresholds fixed) excludes 0;
    null plants: rules 2 and 3 per copy, with an exact binomial 95% interval; the scrambles' false-pass rate.
"""

import json
import sys
from datetime import datetime
from math import comb
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

SEED, BOOT, TARGET, RISE, BAR = 20261011, 2000, 0.20, 0.10, 0.75


def exact_ci(k, n, a=0.05):
    cdf = lambda p, j: sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(j + 1))  # noqa: E731

    def bis(f):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            m = (lo + hi) / 2
            lo, hi = (lo, m) if f(m) else (m, hi)
        return (lo + hi) / 2

    return [0.0 if k == 0 else bis(lambda p: 1 - cdf(p, k - 1) >= a / 2), 1.0 if k == n else bis(lambda p: cdf(p, k) <= a / 2)]


def size(path):
    d = json.loads(Path(path).read_text())
    folds = sorted({int(k.split("|")[0]) for k in d})
    sizes = sorted({float(k.split("|")[1]) for k in d if float(k.split("|")[1]) > 0})
    rise = {s: float(np.mean([d[f"{f}|{s:g}"]["catch"] - d[f"{f}|0"]["catch"] for f in folds])) for s in sizes}
    chosen = next((s for s in sizes if rise[s] >= TARGET), None)
    return {"mean_rise_by_size": rise, "unplanted_mean_catch": float(np.mean([d[f"{f}|0"]["catch"] for f in folds])), "chosen_size": chosen}


def valid(path, gate_final, rows_path):
    V, Gf, spec = json.loads(Path(path).read_text()), json.loads(Path(gate_final).read_text()), json.loads(Path(rows_path).read_text())
    ho = [r for r in spec["rows"] if r["role"] == "heldout"]
    y, w, task = np.array([r["y"] for r in ho]), np.array([r["w"] for r in ho]), np.array([r["task"] for r in ho])
    U = Gf["final"]["real|0"]
    su, tu = np.array(U["heldout_scores"]), U["threshold"]
    F = V["final"]
    rng = np.random.default_rng([SEED, 101])
    tasks, draws = np.unique(task), []
    while len(draws) < BOOT:
        ix = np.concatenate([np.flatnonzero(task == x) for x in rng.choice(tasks, len(tasks))])
        if (y[ix] == 1).any():
            draws.append(ix)

    def catch(s, t, ix):
        p = ix[y[ix] == 1]
        return float((w[p] * (s[p] > t)).sum() / w[p].sum())

    def rise_ci(sp, tp):
        r = [catch(sp, tp, ix) - catch(su, tu, ix) for ix in draws]
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]

    P = F["planted|0"]
    sp = np.array(P["heldout_scores"])
    scr = np.array([F[f"planted|{j}"]["catch"] for j in range(1, 101)])
    scr_auc = np.array([F[f"planted|{j}"]["auroc"] for j in range(1, 101)])
    base = float(np.percentile(scr, 95))
    rise, ci = P["catch"] - U["catch"], rise_ci(sp, P["threshold"])
    out = {"tested_on": "judge-panel labels", "post_hoc": True, "s": V["s"], "unplanted": {"catch": U["catch"], "auroc": U["auroc"], "layer": U["layer"], "C": U["C"]},
           "planted": {"catch": P["catch"], "auroc": P["auroc"], "layer": P["layer"], "C": P["C"], "scrambled_catch_p95": base,
                       "gate_pass": bool(P["auroc"] >= BAR and P["catch"] > base), "rise": rise, "rise_95": ci,
                       "false_alarms": {k: P[k] for k in ("fa_cal_a", "fa_honb_cal", "fa_honb_ho", "fa_heldout_neg")}}}
    out["detected"] = bool(out["planted"]["gate_pass"] and rise >= RISE and ci[0] > 0)
    out["scrambles_false_pass_rate"] = sum(scr_auc[j] >= BAR and scr[j] > np.percentile(np.delete(scr, j), 95) for j in range(100)) / 100
    nulls = []
    for j in range(20):
        N = F[f"null{j}|0"]
        r, c = N["catch"] - U["catch"], rise_ci(np.array(N["heldout_scores"]), N["threshold"])
        nulls.append({"catch": N["catch"], "rise": r, "rise_95": c, "fires_rules_2_3": bool(r >= RISE and c[0] > 0)})
    k = sum(x["fires_rules_2_3"] for x in nulls)
    out["null_plants"] = {"fired": k, "of": 20, "rate_exact_95": exact_ci(k, 20), "copies": nulls}
    return out


if __name__ == "__main__":
    res = size(sys.argv[2]) if sys.argv[1] == "size" else valid(*sys.argv[2:5])
    res["written"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = Path(__file__).parent / "results" / f"plant2_{sys.argv[1]}_result_POSTHOC_{res['written']}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "null_plants"} | ({"null_plants": {k: res["null_plants"][k] for k in ("fired", "of", "rate_exact_95")}}
                     if "null_plants" in res else {}), indent=1), "\n->", dst)
