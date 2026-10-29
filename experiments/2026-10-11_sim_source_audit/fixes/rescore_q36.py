"""POST HOC fix (a), row 1 (POSTHOC_a_row1_q36.md): #36's real-vs-sim numbers on the overlap800 and base200 cells, from
the stored per-run scores and caught fractions (no refit, #36's own frozen thresholds). $0, laptop, numbers only.

The metric code mirrors analyze_q3_attack.metrics (same pairing rule, probe_shift.attack_eval helpers, seed 0, 1,000
main-task resamples over all runs). Gate: on all cells it must reproduce the stored metrics' `slices.all_runs` to
within rounding (0.002; the stored scores are rounded to 4 decimals, caught fractions to 3), or it stops.

  uv run --no-project --python 3.12 --with numpy python rescore_q36.py --run-scores <headline_run_scores_*.jsonl> \
      --metrics <metrics_*.json> --exclusions splits/exclusions_v3.json --out results/
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from probe_shift.attack_eval import E1, E2, HONEST, SIM, auc, boot_weights, rate, summary  # noqa: E402

SHORT = {E1: "e1", E2: "e2", SIM: "sim"}
GAPS = [("sim_minus_e1", SIM, E1), ("sim_minus_e2", SIM, E2), ("e2_minus_e1", E2, E1)]
READOUTS = {"probe_content_mean": "probe_call_end_content_mean", "probe_allcalls_mean": "probe_call_end_mean"}
FPRS = (0.01, 0.05)
TOL = 0.002


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(arm, score, caught, W, keep):
    """Draw-wise values [n_boot + 1] for every reported statistic, on runs `keep` (#36's metrics())."""
    out = {}
    aucs = {}
    for a in (E1, E2, SIM):
        m = keep & np.isin(arm, [HONEST, a])
        aucs[a] = auc(score[m], (arm[m] == a).astype(float), W[:, m])
        out[f"auroc_{SHORT[a]}"] = aucs[a]
    for name, a, b in GAPS:
        out[f"auroc_gap_{name}"] = aucs[a] - aucs[b]
    for f in FPRS:
        rates = {a: rate(caught[f][keep & (arm == a)], W[:, keep & (arm == a)]) for a in (HONEST, E1, E2, SIM)}
        for a in (E1, E2, SIM):
            out[f"catch_{f:g}_{SHORT[a]}"] = rates[a]
        out[f"false_alarm_{f:g}_honest"] = rates[HONEST]
        for name, a, b in GAPS:
            out[f"catch_gap_{f:g}_{name}"] = rates[a] - rates[b]
    return out


def main(run_scores: str, metrics: str, exclusions: str, out: str):
    rows = [json.loads(line) for line in Path(run_scores).read_text().splitlines()]
    M = json.loads(Path(metrics).read_text())
    ref = json.loads(Path(exclusions).read_text())["paired_attack_cells"]["sim_reference"]
    arm = np.array([r["arm"] for r in rows])
    cell = np.array([r["cell"] for r in rows])
    task = np.array([r["main_task"] for r in rows])
    W = boot_weights(task, M["n_bootstrap"], 0)
    is_att = arm != HONEST
    unknown = sorted(set(cell[is_att]) - set(ref))
    if unknown:
        raise SystemExit(f"{len(unknown)} attack cells without a sim_reference label")
    src = np.array([ref.get(c, "honest") if a else "honest" for c, a in zip(cell, is_att)])
    res = {"inputs": {"run_scores": run_scores, "run_scores_sha256": sha(run_scores), "metrics": metrics,
                      "metrics_sha256": sha(metrics), "exclusions_sha256": sha(exclusions), "seed": 0,
                      "n_bootstrap": M["n_bootstrap"]}, "readouts": {}}
    for short, long in READOUTS.items():
        score = np.array([np.nan if r[short] is None else r[short] for r in rows], float)
        caught = {f: np.array([np.nan if r[f"{short}_caught{f:g}"] is None else r[f"{short}_caught{f:g}"] for r in rows], float)
                  for f in FPRS}
        ok = ~np.isnan(score) & np.all([~np.isnan(c) for c in caught.values()], 0)
        failing = set(cell[is_att & ~ok])
        base = np.where(~is_att, ok, ~np.isin(cell, list(failing)))  # analyze_q3_attack.Ctx.paired
        slices = {"all_cells": base, "overlap800": base & (~is_att | (src == "overlap800")),
                  "base200": base & (~is_att | (src == "base200"))}
        vals = {"all_cells": stats(arm, score, caught, W, slices["all_cells"])}
        # reproduction gate on all cells against the stored metrics, before any subset is computed
        S = M["readouts"][long]["task"]["slices"]["all_runs"]
        want = {f"auroc_{a}": S["auroc_vs_honest"][a] for a in ("e1", "e2", "sim")}
        want |= {f"auroc_gap_{n}": S["auroc_gap"][n] for n, _, _ in GAPS}
        for f in FPRS:
            want |= {f"catch_{f:g}_{a}": S["catch_rate"][f"fpr_{f:g}"][a] for a in ("e1", "e2", "sim")}
            want[f"false_alarm_{f:g}_honest"] = S["catch_rate"][f"fpr_{f:g}"]["honest_false_alarm_rate"]
            want |= {f"catch_gap_{f:g}_{n}": S["gap"][f"fpr_{f:g}"][n] for n, _, _ in GAPS}
        n_want = S["n_runs"]
        n_got = {SHORT.get(a, "honest"): int((slices["all_cells"] & (arm == a)).sum()) for a in (HONEST, E1, E2, SIM)}
        gate = {"n_runs_match": n_got == n_want, "n_runs": n_got, "max_abs_diff": 0.0, "fails": []}
        for k, w in want.items():
            g = summary(vals["all_cells"][k])
            for gv, wv in zip([g["value"], *g["ci95"]], [w["value"], *w["ci95"]]):
                d = abs(gv - wv)
                gate["max_abs_diff"] = max(gate["max_abs_diff"], round(d, 6))
                if d > TOL:
                    gate["fails"].append({k: {"got": g, "want": w}})
        gate["passed"] = gate["n_runs_match"] and not gate["fails"]
        if not gate["passed"]:
            res["readouts"][short] = {"gate_all_cells_vs_stored": gate}
            res["stopped"] = f"reproduction gate failed for {short}"
            break
        vals |= {k: stats(arm, score, caught, W, m) for k, m in slices.items() if k != "all_cells"}
        r = {"gate_all_cells_vs_stored": gate,
             "n_runs": {k: {SHORT.get(a, "honest"): int((m & (arm == a)).sum()) for a in (HONEST, E1, E2, SIM)} for k, m in slices.items()},
             "slices": {k: {s: summary(v) for s, v in vv.items()} for k, vv in vals.items()},
             "contrast_overlap800_minus_base200": {s: summary(vals["overlap800"][s] - vals["base200"][s])
                                                   for s in vals["all_cells"] if "gap" in s}}
        res["readouts"][short] = r
    res["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res["written"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = Path(out) / f"rescore_q36_{Path(run_scores).stem.split('__')[0].replace('headline_run_scores_', '')}_{res['written']}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: v["gate_all_cells_vs_stored"] for k, v in res["readouts"].items()}, indent=1))
    print("->", dst)
    if "stopped" in res:
        raise SystemExit(res["stopped"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-scores", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--exclusions", required=True)
    ap.add_argument("--out", required=True)
    main(**vars(ap.parse_args()))
