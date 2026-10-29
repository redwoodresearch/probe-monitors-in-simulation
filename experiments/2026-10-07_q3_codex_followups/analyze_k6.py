"""K6 analysis: mean difference vs logistic regression by number of training pairs (run_k6_learning_curves.py output).

Per layer, training family, size and estimator: held-out AUROC and pair accuracy on each family (mean over the 5 folds
and the draws; ± its standard deviation over draws averaged across folds, the spread one would see between two
probe-building attempts), stability (cosine between directions, Spearman between held-out scores, two draws of the same
size), and the label-swap null at full size. In-family = the family trained on; transfer = the other family.

  python analyze_k6.py --stamp <k6 stamp> [--layer 48]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--layer", type=int, default=48, help="the #41 held-out primary layer")
    args = ap.parse_args()
    res = HERE / "results"
    rows = pd.read_json(res / f"k6_learning_curves_{args.stamp}_rows.jsonl", lines=True)
    stab = pd.read_json(res / f"k6_learning_curves_{args.stamp}_stability.jsonl", lines=True)
    nulls = pd.read_json(res / f"k6_learning_curves_{args.stamp}_nulls.jsonl", lines=True)
    rows["size"] = rows["size"].astype(str)
    stab["size"] = stab["size"].astype(str)
    out = {"stamp": args.stamp, "layer": args.layer, "tables": {}}
    for fam in ("combined", "explicit", "implicit"):
        r = rows[(rows.train_family == fam)]
        metrics = ["auroc_explicit", "auroc_implicit", "pair_acc_explicit", "pair_acc_implicit"]
        g = r.groupby(["layer", "size", "estimator"])
        mean = g[metrics].mean()
        spread = r.groupby(["layer", "size", "estimator", "fold"])[metrics].std().groupby(["layer", "size", "estimator"]).mean().add_suffix("_sd")
        t = mean.join(spread).join(stab[stab.train_family == fam].groupby(["layer", "size", "estimator"])[["cosine_mean", "spearman_mean"]].mean())
        nl = nulls[nulls.train_family == fam].groupby(["layer", "estimator"])[["auroc_explicit", "auroc_implicit"]].mean().add_prefix("null_")
        t = t.reset_index().merge(nl.reset_index(), on=["layer", "estimator"], how="left")
        out["tables"][fam] = t.round(4).to_dict(orient="records")
        show = t[t.layer == args.layer].drop(columns="layer").set_index(["size", "estimator"])
        cols = [c for c in ("auroc_explicit", "auroc_explicit_sd", "auroc_implicit", "auroc_implicit_sd", "pair_acc_explicit", "pair_acc_implicit",
                            "cosine_mean", "spearman_mean", "null_auroc_explicit", "null_auroc_implicit") if c in show]
        print(f"\n== trained on {fam}, layer {args.layer}")
        print(show[cols].round(3).to_string())
    # across layers: transfer at full size, by estimator
    full = rows[rows["size"] == "all"]
    tr = pd.concat([full[full.train_family == "explicit"].assign(transfer=lambda d: d.auroc_implicit, direction="explicit->implicit"),
                    full[full.train_family == "implicit"].assign(transfer=lambda d: d.auroc_explicit, direction="implicit->explicit")])
    tt = tr.groupby(["direction", "layer", "estimator"]).transfer.mean().unstack("estimator").round(3)
    print("\n== transfer AUROC at full size, mean over folds\n" + tt.to_string())
    out["transfer_full"] = tt.reset_index().to_dict(orient="records")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (res / f"k6_analysis_{args.stamp}__{stamp}.json").open("x") as f:
        f.write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
