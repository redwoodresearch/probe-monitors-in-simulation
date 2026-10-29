"""POST HOC (PLAN_POSTHOC.md): split the frozen L63 probe's remaining sim − real e1 gap by output category, from the
per-output records run_q11_rescore_modal.py wrote (gate passed). Local, numbers only.

  uv run --no-project --with numpy python experiments/2026-10-11_q3_q11_remainder/analyze_q11_remainder.py \\
      --records <records json> --gate <results/q11_rescore_gate_POSTHOC_*.json>
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SIM, REAL, E2 = "sim-attack-fixed", "real-attack-e1", "real-attack-e2"
KINDS = ["shell", "view", "create", "str_replace", "insert", "editor_other", "other"]
B, SEED, BAR, MOST, LITTLE = 2000, 20261011, 0.80, 0.50, 0.25
STORED = {"gap_before": 0.46700197863895043, "gap_after": 0.24362830689954867, "planted_shift": 0.290859129233213}


def table(recs, cat, cats, key):
    """Per run (rows): n_r, and per category the count share n_rt / n_r and the score sum / n_r."""
    runs = sorted({(x["cell"], x["arm"]) for x in recs})
    ri = {r: i for i, r in enumerate(runs)}
    ci = {c: j for j, c in enumerate(cats)}
    n, W, S = np.zeros(len(runs)), np.zeros((len(runs), len(cats))), np.zeros((len(runs), len(cats)))
    for x in recs:
        if x[key] is None:
            continue
        i, j = ri[(x["cell"], x["arm"])], ci[cat(x)]
        n[i] += 1
        W[i, j] += 1
        S[i, j] += x[key]
    keep = n > 0
    meta = {}
    for x in recs:
        meta.setdefault((x["cell"], x["arm"]), x)
    arm = np.array([r[1] for r in runs])[keep]
    task = np.array([meta[r]["main_task"] for r in runs])[keep]
    return arm, task, W[keep] / n[keep, None], S[keep] / n[keep, None]


def split(arm, w, Wt, St, a, b):
    """Exact contributions of each category to the gap mean(a) − mean(b) of run scores, with run weights w."""
    def stats(x):
        m = (arm == x) * w
        return (m @ Wt) / m.sum(), (m @ St) / m.sum()
    pa, Ca = stats(a)
    pb, Cb = stats(b)
    G = Ca.sum() - Cb.sum()
    k = (Ca.sum() + Cb.sum()) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = Ca / pa, Cb / pb
    ma, mb = np.where(pa > 0, ma, mb), np.where(pb > 0, mb, ma)
    ma, mb = np.nan_to_num(ma), np.nan_to_num(mb)
    comp = (pa - pb) * ((ma + mb) / 2 - k)
    within = (pa + pb) / 2 * (ma - mb)
    return {"G": G, "contribution": comp + within, "composition": comp, "per_output": within, "p_a": pa, "p_b": pb}


def run_split(recs, cat, cats, key, a, b, tasks, draws):
    arm, task, Wt, St = table(recs, cat, cats, key)
    obs = split(arm, np.ones(len(arm)), Wt, St, a, b)
    tix = {t: i for i, t in enumerate(tasks)}
    ti = np.array([tix[t] for t in task])
    boot = [split(arm, d[ti], Wt, St, a, b) for d in draws]

    def ci(f):
        v = np.array([f(x) for x in boot])
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {"gap": float(obs["G"]), "gap_ci95": ci(lambda x: x["G"]), "runs": {x: int((arm == x).sum()) for x in (a, b)}, "categories": {}}
    for j, c in enumerate(cats):
        out["categories"][c] = {
            "share_of_outputs_" + a: float(obs["p_a"][j]), "share_of_outputs_" + b: float(obs["p_b"][j]),
            **{f: float(obs[f][j]) for f in ("contribution", "composition", "per_output")},
            **{f + "_ci95": ci(lambda x, f=f, j=j: x[f][j]) for f in ("contribution", "composition", "per_output")},
            "share": float(obs["contribution"][j] / obs["G"]), "share_ci95": ci(lambda x, j=j: x["contribution"][j] / x["G"])}
    return out


def change_split(recs, cat, cats, a, b, tasks, draws):
    """Per category, contribution after minus contribution before (the part of the gap the evening removed), same draws."""
    arm, task, Wt, Sb = table(recs, cat, cats, "before")
    arm2, _, Wt2, Sa = table(recs, cat, cats, "after")
    if not (np.array_equal(arm, arm2) and np.array_equal(Wt, Wt2)):
        raise ValueError("before and after tables disagree on runs or outputs")
    tix = {t: i for i, t in enumerate(tasks)}
    ti = np.array([tix[t] for t in task])

    def f(w):
        x, y = split(arm, w, Wt, Sa, a, b), split(arm, w, Wt, Sb, a, b)
        return x["contribution"] - y["contribution"], x["G"] - y["G"]

    d0, g0 = f(np.ones(len(arm)))
    boot = [f(dr[ti]) for dr in draws]
    out = {"gap_change": float(g0), "gap_change_ci95": [float(np.percentile([x[1] for x in boot], q)) for q in (2.5, 97.5)], "categories": {}}
    for j, c in enumerate(cats):
        out["categories"][c] = {"change": float(d0[j]), "change_ci95": [float(np.percentile([x[0][j] for x in boot], q)) for q in (2.5, 97.5)],
                                "share_of_removed": float(d0[j] / g0)}
    return out


def planted_split(recs, cat, cats, tasks, draws):
    """The planted shift (planted − evened, mean over matched outputs per run, mean over runs) split by category."""
    rows = [x for x in recs if x["arm"] == SIM and x["planted"] is not None]
    runs = sorted({x["cell"] for x in rows})
    ri, ci_ = {r: i for i, r in enumerate(runs)}, {c: j for j, c in enumerate(cats)}
    n, D = np.zeros(len(runs)), np.zeros((len(runs), len(cats)))
    for x in rows:
        i = ri[x["cell"]]
        n[i] += 1
        D[i, ci_[cat(x)]] += x["planted"] - x["after"]
    D /= n[:, None]
    task = np.array([next(x["main_task"] for x in rows if x["cell"] == r) for r in runs])
    tix = {t: i for i, t in enumerate(tasks)}
    ti = np.array([tix[t] for t in task])

    def f(w):
        d = (w[:, None] * D).sum(0) / w.sum()
        return d, d.sum()

    d0, s0 = f(np.ones(len(runs)))
    boot = [f(dr[ti]) for dr in draws]
    out = {"runs": len(runs), "shift": float(s0), "shift_ci95": [float(np.percentile([b[1] for b in boot], q)) for q in (2.5, 97.5)], "categories": {}}
    for j, c in enumerate(cats):
        out["categories"][c] = {"term": float(d0[j]), "term_ci95": [float(np.percentile([b[0][j] for b in boot], q)) for q in (2.5, 97.5)],
                                "share": float(d0[j] / s0), "share_ci95": [float(np.percentile([b[0][j] / b[1] for b in boot], q)) for q in (2.5, 97.5)]}
    return out


def reading(share_ci):
    return "carries most" if share_ci[0] > MOST else "little" if share_ci[1] < LITTLE else "partly / unresolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--gate", required=True)
    args = ap.parse_args()
    gate = json.loads(Path(args.gate).read_text())
    raw = Path(args.records).read_bytes()
    if not gate["gate"]["passed"] or hashlib.sha256(raw).hexdigest() != gate["records_local_copy_sha256"]:
        raise SystemExit("gate not passed, or the records file is not the gated one")
    recs = json.loads(raw)
    tasks = sorted({x["main_task"] for x in recs})
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)).astype(float) for _ in range(B)]
    kind = (lambda x: x["kind"], KINDS)
    touched = (lambda x: "touched" if x["touched"] else "untouched", ["touched", "untouched"])
    family = (lambda x: "str_replace+insert" if x["kind"] in ("str_replace", "insert") else "rest", ["str_replace+insert", "rest"])
    res = {"records_sha256": gate["records_local_copy_sha256"], "gate_file": Path(args.gate).name, "seed": SEED, "draws": B,
           "agent_calls_side": "not covered: B-dev stored only the reply readout (output_end); no call-end readout exists",
           "planted": {name: planted_split(recs, c, cs, tasks, draws) for name, (c, cs) in
                       (("by_kind", kind), ("by_plant_edited", (lambda x: "edited" if x["plant_edited"] else "not edited", ["edited", "not edited"])))}}
    sr = res["planted"]["by_kind"]["categories"]["str_replace"]
    res["planted"]["bar"] = {"rule": "str_replace share >= 0.80 and its term's 95% interval above 0", "share": sr["share"],
                             "term_ci95": sr["term_ci95"], "passed": bool(sr["share"] >= BAR and sr["term_ci95"][0] > 0)}
    for state in ("after", "before"):
        res[state] = {name: run_split(recs, c, cs, state, SIM, REAL, tasks, draws) for name, (c, cs) in
                      (("by_kind", kind), ("by_touched", touched), ("by_family", family))}
    res["change_before_to_after"] = {name: change_split(recs, c, cs, SIM, REAL, tasks, draws) for name, (c, cs) in
                                     (("by_kind", kind), ("by_touched", touched), ("by_family", family))}
    res["control_e2_minus_e1_after"] = {"by_kind": run_split(recs, *kind, "after", E2, REAL, tasks, draws)}
    res["checks"] = {"gap_after_vs_stored": res["after"]["by_kind"]["gap"] - STORED["gap_after"],
                     "gap_before_vs_stored": res["before"]["by_kind"]["gap"] - STORED["gap_before"],
                     "planted_shift_vs_stored": res["planted"]["by_kind"]["shift"] - STORED["planted_shift"]}
    a = res["after"]
    res["readings"] = {"fix939_family": reading(a["by_family"]["categories"]["str_replace+insert"]["share_ci95"]),
                       "touched": reading(a["by_touched"]["categories"]["touched"]["share_ci95"]),
                       "leading_type": max(KINDS, key=lambda c: a["by_kind"]["categories"][c]["contribution"]),
                       "caveat": None if res["planted"]["bar"]["passed"] else
                       "planted bar failed: the type split shows where the score difference sits, not what causes it"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"q11_remainder_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({"checks": res["checks"], "planted_bar": res["planted"]["bar"], "readings": res["readings"],
                      "after_by_kind": {c: (round(v["contribution"], 4), [round(x, 4) for x in v["contribution_ci95"]], round(v["share"], 3))
                                        for c, v in a["by_kind"]["categories"].items()}}, indent=1))
    print(f"q11_remainder_POSTHOC_{stamp}.json")


if __name__ == "__main__":
    main()
