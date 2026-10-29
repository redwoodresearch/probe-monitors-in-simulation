"""K6 2x2 analysis: {mean difference, logistic regression} x {raw, standardized} (run_k6_learning_curves.py with four
estimators). Does raw mean difference fail because of the estimator or because of raw coordinate scale?

  1. Reproduction: the two estimators the first run had must give the same numbers (every metric of every fit).
  2. Full size, per layer and training family: held-out AUROC per family (mean over the 5 folds, 95% interval from
     resampling held-out tasks within each fold, 1,000 draws), pair accuracy, midpoint sd over pair gap, label-swap
     null mean; logistic-regression convergence.
  3. Small sizes: AUROC at 25/50/100 pairs and stability (direction cosine, held-out score Spearman between draws).

  python analyze_k6_2x2.py --stamp <2x2 stamp> --first <2026-10-07_172023> --pairs <eval_awareness_pairs_*.jsonl>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
EST = ["mean_difference", "mean_difference_standardized", "logistic_regression_raw", "logistic_regression"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--layer", type=int, default=48)
    args = ap.parse_args()
    res = HERE / "results"
    load = lambda s, n: pd.read_json(res / f"k6_learning_curves_{s}_{n}.jsonl", lines=True)
    rows, stab, nulls, held = (load(args.stamp, n) for n in ("rows", "stability", "nulls", "heldout"))
    first = load(args.first, "rows")
    keys = ["layer", "fold", "train_family", "size", "draw", "estimator"]
    m = first.merge(rows, on=keys, suffixes=("_a", "_b"))
    cols = [c for c in ("auroc_explicit", "auroc_implicit", "pair_acc_explicit", "pair_acc_implicit")]
    worst = max(float((m[f"{c}_a"] - m[f"{c}_b"]).abs().max()) for c in cols)
    if len(m) != len(first) or worst > 1e-9:
        raise ValueError(f"first run not reproduced: {len(m)}/{len(first)} fits matched, max difference {worst}")
    print(f"reproduced all {len(first)} fits of the first run (max difference {worst:.1e})")
    dial = [json.loads(l) for l in Path(args.pairs).read_text().splitlines()]
    label = np.array([d["label"] for d in dial]); family = np.array([d["family"] for d in dial]); task = np.array([d["task"] for d in dial])
    rng = np.random.default_rng(0)
    out = {"stamp": args.stamp, "reproduced_first": args.first, "full": [], "small": []}
    full = rows[rows["size"].astype(str) == "all"]
    for (layer, fam, est), g in full.groupby(["layer", "train_family", "estimator"]):
        rec = {"layer": layer, "train_family": fam, "estimator": est}
        h = held[(held.layer == layer) & (held.train_family == fam) & (held.estimator == est)]
        for f in ("explicit", "implicit"):
            per_fold, boots = [], []
            for _, x in h.iterrows():
                idx, s = np.array(x.dialogues), np.array(x.scores)
                k = family[idx] == f
                idx, s = idx[k], s[k]
                tks = np.unique(task[idx])
                by = {t: np.flatnonzero(task[idx] == t) for t in tks}
                per_fold.append(roc_auc_score(label[idx], s))
                b = []
                for _ in range(1000):
                    j = np.concatenate([by[t] for t in rng.choice(tks, len(tks))])
                    b.append(roc_auc_score(label[idx][j], s[j]) if 0 < label[idx][j].mean() < 1 else np.nan)
                boots.append(b)
            boot = np.nanmean(np.array(boots), 0)
            rec[f"auroc_{f}"] = round(float(np.mean(per_fold)), 4)
            rec[f"auroc_{f}_ci95"] = [round(float(v), 4) for v in np.nanpercentile(boot, [2.5, 97.5])]
            rec[f"pair_acc_{f}"] = round(float(g[f"pair_acc_{f}"].mean()), 4)
            rec[f"midpoint_sd_over_gap_{f}"] = round(float(g[f"midpoint_sd_over_gap_{f}"].mean()), 3)
            nl = nulls[(nulls.layer == layer) & (nulls.train_family == fam) & (nulls.estimator == est)]
            rec[f"null_auroc_{f}"] = round(float(nl[f"auroc_{f}"].mean()), 4)
        rec["lr_converged"] = None if g.lr_converged.isna().all() else round(float(g.lr_converged.mean()), 3)
        out["full"].append(rec)
    small = rows[rows["size"].astype(str) != "all"].groupby(["layer", "train_family", "size", "estimator"])[["auroc_explicit", "auroc_implicit"]].mean()
    st = stab.groupby(["layer", "train_family", "size", "estimator"])[["cosine_mean", "spearman_mean"]].mean()
    sm = small.join(st).reset_index()
    out["small"] = sm.round(4).to_dict(orient="records")
    fr = pd.DataFrame(out["full"])
    fmt = lambda r, f: f"{r[f'auroc_{f}']:.3f} [{r[f'auroc_{f}_ci95'][0]:.3f}, {r[f'auroc_{f}_ci95'][1]:.3f}]"
    for fam in ("combined", "explicit", "implicit"):
        print(f"\n== full size, trained on {fam}, layer {args.layer}")
        for est in EST:
            r = fr[(fr.layer == args.layer) & (fr.train_family == fam) & (fr.estimator == est)].iloc[0]
            print(f"  {est:30s} AUROC explicit {fmt(r, 'explicit')} implicit {fmt(r, 'implicit')} | pair acc {r.pair_acc_explicit:.3f}/{r.pair_acc_implicit:.3f} "
                  f"| midpoint/gap {r.midpoint_sd_over_gap_explicit:.2f}/{r.midpoint_sd_over_gap_implicit:.2f} | null {r.null_auroc_explicit:.3f}/{r.null_auroc_implicit:.3f} | conv {r.lr_converged}")
    print("\n== combined training, explicit AUROC at full size, by layer")
    print(fr[fr.train_family == "combined"].pivot(index="layer", columns="estimator", values="auroc_explicit")[EST].round(3).to_string())
    print("\n== transfer at full size (explicit->implicit / implicit->explicit), by layer")
    tr = pd.concat([fr[fr.train_family == "explicit"].assign(d="explicit->implicit", v=lambda x: x.auroc_implicit),
                    fr[fr.train_family == "implicit"].assign(d="implicit->explicit", v=lambda x: x.auroc_explicit)])
    print(tr.pivot_table(index=["d", "layer"], columns="estimator", values="v")[EST].round(3).to_string())
    print(f"\n== small sizes, combined training, layer {args.layer}: explicit AUROC / direction cosine")
    s48 = sm[(sm.layer == args.layer) & (sm.train_family == "combined")]
    print(s48.pivot(index="size", columns="estimator", values="auroc_explicit")[EST].round(3).to_string())
    print(s48.pivot(index="size", columns="estimator", values="cosine_mean")[EST].round(3).to_string())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (res / f"k6_2x2_analysis_{args.stamp}__{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
