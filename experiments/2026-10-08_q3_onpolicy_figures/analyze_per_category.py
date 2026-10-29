"""Figure 3 by labeler category, and two exploratory cuts. $0, local; reads the observed draw's per-call scores.

From run_score_onpolicy.py --per-call (draw 0 only: the real-trained probe's out-of-fold score and threshold flags for
every evaluated call), joined to make_call_categories.py's table by (request id, call index):
- Per-category AUROC (primary for "does it only work on obvious commands?"): each impactful category's calls against all
  read-only calls of the same test world, scored by the real-trained probe; real (real → real) and simulated
  (real → sim). Categories under 20 calls: counts only.
- Per-category recall at the frozen 1% threshold (the real-calibrated threshold, flag_frozen): DESCRIPTIVE, OUTSIDE THE
  REGISTERED RULES (amendment 28: the calibration folds hold 63–93 read-only calls).
- Exploratory, outside the registration: (1) mix reweighting: impactful calls reweighted to the off-policy category mix
  (Q2's per-category table, real → real counts), AUROC and standardized pAUC recomputed; (2) the localhost rule's
  refit (a separate --per-call run under label_rule localhost_reads_read_only): AUROC and pAUC.
- Amendment 9's simulated-arm null: real → sim scores fixed, labels permuted within each simulated transcript, 1,000
  draws; p = (1 + #{null ≥ observed}) / 1,001.
Intervals: 2,000 fold-stratified task resamples of the fixed scores (seed 0), as the off-policy per-category table.

  python analyze_per_category.py --percall <september percall json> [--percall <today percall json>] \
      --categories <phase2_call_categories json> [--localhost <september localhost percall json>]
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
Q2_TABLE = "experiments/2026-10-06_q2_impactful_call_transfer/results/q2_final_table_2026-10-07_095958_056930.md"
MIN_N = 20


def offpolicy_mix(path: Path):
    """Impactful category counts of the off-policy real → real cell (call_end rows of Q2's per-category table, a committed
    results file read from the working tree; its sha256 goes in the output)."""
    md = path.read_text()
    counts = {}
    for m in re.finditer(r"^\| call_end \| (\w+) \| recall \| [\d.na]+ \(n=(\d+)\)", md, re.M):
        counts[m.group(1)] = int(m.group(2))
    if "file_write" not in counts:
        raise ValueError("Q2's per-category table did not parse")
    return counts


def task_weights(fold_of_task, b, seed=0):
    f = np.asarray(fold_of_task)
    if b == 0:
        return np.ones(len(f))
    rng, w = np.random.default_rng([seed, b]), np.zeros(len(f))
    for k in range(5):
        idx = np.flatnonzero(f == k)
        np.add.at(w, rng.choice(idx, len(idx), replace=True), 1)
    return w


def ci(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) else None


def frame(path, cats):
    """Per evaluated call of each real-trained cell: world, task, fold, y, score, frozen flag, category."""
    R = json.loads(Path(path).read_text())
    meta, d0 = R["meta"], next(d for d in R["draws"] if d["draw"] == 0)
    by = {(c["irid"], c["call"]): c for c in cats}
    out = {}
    for cell in ("real->real", "real->sim"):
        pc = d0["cells"][cell]["per_call"]
        rows = []
        for i, s, ff in zip(pc["row"], pc["score"], pc["flag_frozen"]):
            m = meta[i]
            c = by[(m["irid"], m["call"])]
            rows.append({"task": m["task"], "fold": R["folds"][m["task"]], "y": m["y"], "score": s, "flag": ff,
                         "category": c["category"], "label": c["label"]})
        out[cell] = rows
    return out, R


def per_category(rows, tasks, fold_of_task, mix=None):
    """AUROC per impactful category vs all read-only, descriptive 1% recall, and (with mix) the reweighted pooled
    AUROC and pAUC, each with fold-stratified task-bootstrap intervals over fixed scores."""
    ti = {t: i for i, t in enumerate(tasks)}
    tix = np.array([ti[r["task"]] for r in rows])
    y = np.array([r["y"] for r in rows])
    s = np.array([r["score"] for r in rows])
    flag = np.array([r["flag"] for r in rows])
    cat = np.array([r["category"] for r in rows])
    res = {}
    draws = [task_weights(fold_of_task, b)[tix] for b in range(2001)]
    for c in sorted(set(cat[y == 1])):
        m = (cat == c) & (y == 1)
        n, ntasks = int(m.sum()), len(set(tix[m]))
        if n < MIN_N:
            res[c] = {"n": n, "tasks": ntasks, "note": f"under {MIN_N} calls: count only"}
            continue
        sel = m | (y == 0)

        def auc(w):
            ww = w[sel]
            return None if ww[y[sel] == 1].sum() == 0 or ww[y[sel] == 0].sum() == 0 else roc_auc_score(y[sel], s[sel], sample_weight=ww)

        def rec(w):
            return None if w[m].sum() == 0 else float((w * flag)[m].sum() / w[m].sum())
        res[c] = {"n": n, "tasks": ntasks, "auroc": round(auc(draws[0]), 4), "auroc_ci95": ci([auc(w) for w in draws[1:]]),
                  "recall_at_frozen_1pct_DESCRIPTIVE": round(rec(draws[0]), 4), "recall_ci95": ci([rec(w) for w in draws[1:]])}
    if mix:
        share_on = Counter(cat[y == 1])
        tot_off = sum(v for k, v in mix.items() if k in share_on)
        cw = {k: (mix.get(k, 0) / tot_off) / (share_on[k] / (y == 1).sum()) for k in share_on}  # absent off-policy: weight 0
        mw = np.array([cw[c] if yy == 1 else 1.0 for c, yy in zip(cat, y)])

        def both(w):
            ww = w * mw
            return roc_auc_score(y, s, sample_weight=ww), roc_auc_score(y, s, sample_weight=ww, max_fpr=0.05)
        o = both(draws[0])
        bs = [both(w) for w in draws[1:]]
        res["_mix_reweighted_EXPLORATORY"] = {"weights": {k: round(v, 3) for k, v in cw.items()}, "auroc": round(o[0], 4),
                                             "auroc_ci95": ci([b[0] for b in bs]), "pauc": round(o[1], 4), "pauc_ci95": ci([b[1] for b in bs])}
        o = (roc_auc_score(y, s), roc_auc_score(y, s, max_fpr=0.05))
        res["_unweighted"] = {"auroc": round(o[0], 4), "pauc": round(o[1], 4)}
    return res


def sim_null(rows, seed=0, n=1000):
    """Amendment 9's simulated-arm null: real-trained scores fixed, labels permuted within each simulated transcript."""
    y = np.array([r["y"] for r in rows])
    s = np.array([r["score"] for r in rows])
    t = np.array([r["task"] for r in rows])
    obs = roc_auc_score(y, s)
    rng = np.random.default_rng([seed, 9, 1])
    null = []
    for _ in range(n):
        yy = y.copy()
        for task in np.unique(t):
            m = np.flatnonzero(t == task)
            yy[m] = rng.permutation(yy[m])
        null.append(roc_auc_score(yy, s))
    null = np.array(null)
    return {"observed": round(float(obs), 4), "null_mean": round(float(null.mean()), 4), "null_max": round(float(null.max()), 4),
            "p": round(float((1 + (null >= obs).sum()) / (n + 1)), 4), "passes_at_0.05": bool((1 + (null >= obs).sum()) / (n + 1) <= 0.05)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percall", nargs="+", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--localhost")
    ap.add_argument("--q2-table", default=str(HERE.parents[1] / Q2_TABLE), help="Q2's per-category table (the off-policy mix)")
    args = ap.parse_args()
    cats = json.loads(Path(args.categories).read_text())["rows"]
    q2 = Path(args.q2_table)
    mix = offpolicy_mix(q2)
    out = {"inputs": [Path(p).name for p in args.percall], "categories": Path(args.categories).name, "offpolicy_mix_counts": mix,
           "offpolicy_mix_source": {"path": Q2_TABLE, "sha256": hashlib.sha256(q2.read_bytes()).hexdigest()},
           "labels": {"recall": "DESCRIPTIVE, OUTSIDE THE REGISTERED RULES (A28: calibration folds hold 63–93 read-only calls)",
                      "mix_and_localhost": "EXPLORATORY, outside the registration"}, "simulators": {}}
    for p in args.percall:
        F, R = frame(p, cats)
        sim = R["config"]["worlds"]["sim"]
        tasks = sorted({r["task"] for rows in F.values() for r in rows})
        fot = [R["folds"][t] for t in tasks]
        out["simulators"][sim] = {"real (real → real)": per_category(F["real->real"], tasks, fot, mix),
                                  "simulated (real → sim)": per_category(F["real->sim"], tasks, fot, mix),
                                  "a9_sim_arm_null": sim_null(F["real->sim"])}
    if args.localhost:
        F, R = frame(args.localhost, cats)
        out["localhost_rule_EXPLORATORY"] = {cell: {"auroc": round(roc_auc_score([r["y"] for r in rows], [r["score"] for r in rows]), 4),
                                                    "pauc": round(roc_auc_score([r["y"] for r in rows], [r["score"] for r in rows], max_fpr=0.05), 4),
                                                    "impactful": sum(r["y"] for r in rows), "read_only": sum(1 - r["y"] for r in rows)}
                                             for cell, rows in F.items()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"per_category_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for sim, d in out["simulators"].items():
        print("==", sim)
        for world, res in d.items():
            if world == "a9_sim_arm_null":
                print("  A9 simulated-arm null:", res)
                continue
            for c, r in res.items():
                print("  ", world, "|", c, "|", r)
    if "localhost_rule_EXPLORATORY" in out:
        print("== localhost rule (exploratory):", out["localhost_rule_EXPLORATORY"])
    print("->", path)


if __name__ == "__main__":
    main()
