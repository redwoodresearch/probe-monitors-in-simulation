"""Figure 3's numbers from run_score_onpolicy.py's draws. $0, local.

- Cells (trained W → tested V): the observed value (draw 0) and a 95% interval over draws 1..B.
- Gaps, signed simulated − real (amendment 43), per simulator:
  - main estimand: (real → sim) − (real → real);
  - build-in-sim: (sim → real) − (real → real);
  - threshold transfer (§4 secondary): catch and false alarms on simulated calls at the real-calibrated threshold, minus
    real → real.
- Primary result (amendment 47): the AUROC and standardized-pAUC gaps with their intervals, as absolute numbers.
- Failed draws (amendments 28, 30, 34, 38, 42): kept, at the scale's floor for the lower bound and its ceiling for the
  upper; counted per metric and by cause; more than 2.5% failed makes an interval "unreliable at this size".
- Catch and false-alarm rates are "unestimable at this size" when a calibration fold of the observed data holds fewer
  than 100 read-only calls (amendment 28): their numbers are withheld, in cells and in every gap that uses them.
- Verdicts (§4 at ±0.03, a labelled secondary under amendment 47; amendment 40's half-width rule for Δ_AUROC): only with
  --final-cohort, never at pilot size (§4, amendment 14). The joint verdict needs both measures equivalent and both
  arms passing amendment 9's label-shuffle null (--null-real, --null-sim); otherwise "uninformative (signal not
  established)", or pending without the nulls. Decisions use unrounded bounds.
- The draws must be exactly 0..B, each once; otherwise the analysis is marked partial and gives no verdict.

  python analyze_onpolicy_2x2.py results/onpolicy_2x2_<label>_draws0-2000_<stamp>.json [--final-cohort]
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
METRICS = ["auroc", "pauc", "catch", "false_alarm", "catch_frozen", "false_alarm_frozen"]
THRESHOLDED = {"catch": "calibrated", "false_alarm": "calibrated", "catch_frozen": "frozen", "false_alarm_frozen": "frozen"}
PRIMARY, DELTA = ["auroc", "pauc"], 0.03


def interval(vals, causes, lo_scale, hi_scale):
    """95% interval; failed draws (None) at the floor for the lower bound and the ceiling for the upper."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    failed = np.isnan(v)
    if not len(v):
        return {"ci95": None, "draws": 0}
    lo = float(np.percentile(np.where(failed, lo_scale, v), 2.5))
    hi = float(np.percentile(np.where(failed, hi_scale, v), 97.5))
    by_cause = Counter(c for cs, f in zip(causes, failed) if f for c in set(cs))
    r = {"ci95_exact": [lo, hi], "ci95": [round(lo, 4), round(hi, 4)], "draws": len(v), "failed": int(failed.sum()),
         "failed_share": round(float(failed.mean()), 4), "failed_by_cause": dict(by_cause)}
    if failed.mean() > 0.025:
        r["status"] = "unreliable at this size"
    return r


def verdict(metric, lo, hi):
    """§4's per-measure verdict at ±0.03 (amendment 40: a Δ_AUROC half-width above 0.03 is inconclusive)."""
    if metric == "auroc" and (hi - lo) / 2 > DELTA:
        return "inconclusive"
    if -DELTA <= lo and hi <= DELTA:
        return "equivalent"
    if lo > DELTA or hi < -DELTA:
        return "different"
    return "inconclusive"


def joint(verdicts: dict, gate):
    """§4's joint verdict over Δ_AUROC and the co-primary (the standardized pAUC gap, amendment 43): "different" if
    either is; "equivalent" only if both are AND both arms pass amendment 9's label-shuffle null, otherwise
    "uninformative (signal not established)"; without the nulls, pending; anything else inconclusive."""
    if "different" in verdicts.values():
        return "different"
    if all(v == "equivalent" for v in verdicts.values()):
        return "pending the label-shuffle null (amendment 9)" if gate is None else ("equivalent" if gate else "uninformative (signal not established)")
    return "inconclusive"


def main(path, final_cohort, null_real=None, null_sim=None):
    R = json.loads(Path(path).read_text())
    ids = [d["draw"] for d in R["draws"]]
    draws = {d["draw"]: d for d in R["draws"]}
    if 0 not in draws:
        raise ValueError("the observed draw (0) is missing")
    complete = sorted(ids) == list(range(max(ids) + 1)) and max(ids) >= 2000
    obs, boots = draws[0], [draws[b] for b in sorted(draws) if b > 0]
    simulator = R["config"]["worlds"]["sim"]
    out = {"input": Path(path).name, "label": R["label"],
           "simulator": simulator + (" (primary, amendment 46)" if simulator == "sim_attack@september" else " (secondary)"),
           "amendments": R["amendments"], "git_of_scoring": R["git"], "draws": len(boots), "bootstrap_complete": complete,
           "verdicts": "given" if final_cohort and complete else
           ("none: pilot size (§4, amendment 14)" if not final_cohort else "none: partial bootstrap"),
           "summary": R["summary"], "cells": {}, "gaps": {}}
    withheld = {}  # (cell, metric) -> reason

    def raw(d, cell, m):
        return d["cells"][cell]["values"][m]

    for cell, o in obs["cells"].items():
        support = {k: min((f["calibration_read_only_calls"] for f in o.get("folds") or [] if f["threshold_kind"] == k), default=0)
                   for k in ("calibrated", "frozen")}
        res = {}
        for m in METRICS:
            r = {"value": o["values"][m], **interval([raw(d, cell, m) for d in boots],
                                                     [d["cells"][cell]["failed"].get(m, []) for d in boots], 0.0, 1.0)}
            reason = None
            if o["failed"].get(m):
                reason = f"unestimable at this size ({', '.join(o['failed'][m])})"
            elif m in THRESHOLDED and support[THRESHOLDED[m]] < 100:
                reason = f"unestimable at this size (a calibration fold has {support[THRESHOLDED[m]]} read-only calls; 100 needed)"
            if reason:
                withheld[(cell, m)] = reason
                r = {"value": None, "ci95": None, "status": reason}
            res[m] = r
        out["cells"][cell] = res | {"folds": o.get("folds"), "eval_calls": o.get("eval_calls")}
    main_ = "main estimand: (real→sim) − (real→real)"
    pairs = {main_: [(m, "real->sim", m, "real->real", m) for m in ("auroc", "pauc", "catch")]
             + [("threshold transfer: catch at the real-calibrated threshold", "real->sim", "catch_frozen", "real->real", "catch"),
                ("threshold transfer: false alarms at the real-calibrated threshold", "real->sim", "false_alarm_frozen", "real->real", "false_alarm")],
             "build-in-sim: (sim→real) − (real→real)": [(m, "sim->real", m, "real->real", m) for m in ("auroc", "pauc", "catch")]}
    for name, rows in pairs.items():
        g = {}
        for label, a, ma, b, mb in rows:
            reasons = [withheld[k] for k in ((a, ma), (b, mb)) if k in withheld]
            if reasons:
                g[label] = {"value": None, "ci95": None, "status": "; ".join(sorted(set(reasons)))}
                continue
            diff = lambda d, a=a, ma=ma, b=b, mb=mb: None if raw(d, a, ma) is None or raw(d, b, mb) is None else raw(d, a, ma) - raw(d, b, mb)
            causes = [d["cells"][a]["failed"].get(ma, []) + d["cells"][b]["failed"].get(mb, []) for d in boots]
            o = diff(obs)
            r = {"value": None if o is None else round(o, 4), **interval([diff(d) for d in boots], causes, -1.0, 1.0)}
            if label in PRIMARY and name == main_:
                if out["verdicts"] == "given":
                    v = "inconclusive" if "status" in r else verdict(label, *r["ci95_exact"])
                    r["secondary_verdict_at_0.03"] = {"verdict": v, "label": "borrowed margin, not consequence-based (amendment 47)"}
                else:
                    r["secondary_verdict_at_0.03"] = {"verdict": None, "why": out["verdicts"]}
            g[label] = r
        out["gaps"][name] = g
    out["primary_result"] = {m: out["gaps"][main_][m] for m in PRIMARY}
    if out["verdicts"] == "given":  # the joint verdict, gated by both arms' nulls when they are given
        gate = None
        if null_real and null_sim:
            nr = json.loads(Path(null_real).read_text())
            ns = json.loads(Path(null_sim).read_text())["nulls"][simulator]
            gate = bool(nr["passes_at_0.05"]) and bool(ns["passes_at_0.05"])
            out["label_shuffle_nulls"] = {"real": {k: nr[k] for k in ("p", "passes_at_0.05")}, "simulated": ns}
        per = {m: out["primary_result"][m]["secondary_verdict_at_0.03"]["verdict"] for m in PRIMARY}
        out["joint_verdict_at_0.03"] = {"verdict": joint(per, gate), "per_measure": per,
                                        "label": "borrowed margin, not consequence-based (amendment 47); secondary"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"onpolicy_2x2_analysis_{R['label']}_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("label", "simulator", "draws", "bootstrap_complete", "verdicts")}))
    for m, r in out["primary_result"].items():
        print("primary", m, {k: r.get(k) for k in ("value", "ci95", "failed", "status")})
    for cell, r in out["cells"].items():
        print(cell, {m: (r[m]["value"] if r[m]["value"] is None else round(r[m]["value"], 4), r[m].get("ci95"), r[m].get("status", ""))
                     for m in ("auroc", "pauc", "catch")})
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--final-cohort", action="store_true", help="the registered final cohort; verdicts are given only then")
    ap.add_argument("--null-real", help="analyze_a9_null.py's real-arm result (amendment 9)")
    ap.add_argument("--null-sim", help="the simulated-arm null summary, keyed by simulator (amendment 9)")
    a = ap.parse_args()
    main(a.path, a.final_cohort, a.null_real, a.null_sim)
