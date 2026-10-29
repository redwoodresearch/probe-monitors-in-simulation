"""Plot run AUROC against k (first k or random k outputs per run), with the independent-evidence prediction and the
real-vs-real control. Rows: held-out environment, main-task folds; columns: level 0, level 7.

  uv run --with matplotlib python experiments/2026-10-07_what_each_method_sees/plot_adds_up.py --results results/adds_up_<stamp>.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
COLOR = {"tfidf": "#2a78d6", "probe": "#eb6834"}  # reference palette slots 1-2
NAME = {"tfidf": "TF-IDF", "probe": "probe (Qwen3-32B)"}
SPLIT = {"env": "held-out environment", "task": "main-task folds"}
ALL_X = 40  # where "all outputs" sits on the log axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    path = Path(args.results) if Path(args.results).is_absolute() else HERE / args.results
    d = json.loads(path.read_text())["results"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    for row, split in enumerate(("env", "task")):
        for col, level in enumerate(("L0", "L7")):
            ax = axes[row, col]
            for m in ("tfidf", "probe"):
                r = d[f"{level}|{split}|real_vs_sim"]["by_method"][m]
                ks = list(r["first_k"])
                x = [ALL_X if k == "all" else int(k) for k in ks]
                first = [r["first_k"][k]["auroc"] for k in ks]
                lo = [r["first_k"][k]["ci95"][0] for k in ks]
                hi = [r["first_k"][k]["ci95"][1] for k in ks]
                ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.15, linewidth=0)
                ax.plot(x, first, "-o", color=COLOR[m], lw=2, ms=5, label=f"{NAME[m]}: first k")
                rk = [k for k in ks if k in r["random_k"]]
                ax.plot([int(k) for k in rk] + [ALL_X], [r["random_k"][k]["auroc_mean"] for k in rk] + [first[-1]], "--",
                        color=COLOR[m], lw=2, label=f"{NAME[m]}: random k")
                ax.plot(x, [r["first_k"][k]["independent"] for k in ks], ":", color=COLOR[m], lw=1.5,
                        label=f"{NAME[m]}: independent outputs (predicted)")
                c = d[f"{level}|{split}|null_e1_vs_e2"]["by_method"][m]
                ax.plot(x, [c["first_k"][k]["auroc"] for k in ks], "-x", color=COLOR[m], lw=1, ms=5, alpha=0.45,
                        label=f"{NAME[m]}: real vs real control, first k")
                ax.annotate(f"{first[-1]:.3f}", (ALL_X, first[-1]), xytext=(6, -3 if m == "probe" else 6),
                            textcoords="offset points", fontsize=8, color="#52514e")
            ax.axhline(0.5, color="#b8b7b2", lw=1)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 3, 5, 10, 20, ALL_X])
            ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "all"])
            ax.minorticks_off()
            ax.grid(axis="y", color="#e6e5e0", lw=1)
            ax.set_ylim((0.45, 1.0) if level == "L0" else (0.45, 0.72))
            ax.set_title(f"Level {level[1]} ({'original text' if level == 'L0' else 'after the cleanup ladder'}), {SPLIT[split]}",
                         fontsize=10)
            if col == 0:
                ax.set_ylabel("run AUROC (higher = more separable)")
            if row == 1:
                ax.set_xlabel("outputs per run used (k, log scale; a run with fewer uses all it has)")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle("Real e1 vs simulated runs: mean of per-output log-odds over k outputs per run (bands: 95% CI, main-task bootstrap)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    out = path.with_name(path.name.replace("adds_up_", "adds_up_plot_").replace(".json", ".png"))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
