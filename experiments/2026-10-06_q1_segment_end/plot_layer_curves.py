"""Per-layer curves for the all-layers Q1 analyses: run-level AUROC (real epoch 1 vs simulated) at every layer, one line
per text level, for both readouts and both splits; a markdown table beside the figure. For interpretation only: the
headline number comes from the frozen rule in protocol_level7.md.

  uv run --with matplotlib python experiments/2026-10-06_q1_segment_end/plot_layer_curves.py \
      --results original=results/q1_gpu_<id>.json level5=results/q1_gpu_<id>.json level7=results/q1_gpu_<id>.json \
      [--variant all_outputs] [--out results/layer_curves_<stamp>]

Each figure also writes <out>_inputs.json: the per-layer numbers it plots, which `--inputs` replots from without the
full analysis files.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def curve(res: dict, pos: str, split: str, variant: str):
    """[(layer, auroc, lo, hi)] for one readout, split ("main_task" or "held_out_environment") and variant."""
    out = []
    for l in sorted(res["by_layer"], key=int):
        if pos not in res["by_layer"][l]["results"]:  # e.g. a call_end-only rerun
            continue
        r = res["by_layer"][l]["results"][pos]["real_vs_sim"]
        d = (r["held_out_environment"][variant] if split == "held_out_environment" else r[variant])["run"]
        if d:
            out.append((int(l), d["auroc"], d["ci95"][0], d["ci95"][1]))
    return out


def null_p95(res: dict, pos: str, split: str, variant: str):
    """[(layer, p95)] of the per-layer label-shuffle null. It exists for main-task folds only; held-out environments have
    no per-layer null (their check is the selection-aware null), and a main-fold null is not shown in their place."""
    if split != "main_task":
        return []
    out = []
    for l in sorted(res["by_layer"], key=int):
        if pos not in res["by_layer"][l]["results"]:
            continue
        n = ((res["by_layer"][l]["results"][pos]["real_vs_sim"][variant]["run"] or {}).get("label_shuffle_null") or {})
        if "p95" in n:
            out.append((int(l), n["p95"]))
    return out


def prune(res: dict, variant: str) -> dict:
    """The per-layer fields the plot reads, in the analysis file's own layout (small enough to commit)."""
    keep = lambda d: {"run": d["run"] and {k: d["run"][k] for k in ("auroc", "ci95", "label_shuffle_null") if k in d["run"]}}
    out = {}
    for l, x in res["by_layer"].items():
        for pos in ("output_end", "call_end"):
            if pos in x["results"]:
                r = x["results"][pos]["real_vs_sim"]
                out.setdefault(l, {"results": {}})["results"][pos] = {"real_vs_sim": {
                    variant: keep(r[variant]), "held_out_environment": {variant: keep(r["held_out_environment"][variant])}}}
    return {"by_layer": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", help="label=path, in text-level order")
    ap.add_argument("--inputs", type=Path, help="an <out>_inputs.json written by an earlier plot, instead of --results")
    ap.add_argument("--variant", help="all_outputs (default) or no_editor_outputs; with --inputs, the saved file's")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if bool(args.results) == bool(args.inputs):
        ap.error("give --results or --inputs")
    outputs = [args.out.with_suffix(".png"), args.out.with_suffix(".md"), args.out.parent / f"{args.out.name}_inputs.json"]
    if any(o.exists() for o in outputs):  # results/ is append-only: never overwrite a plot or its saved inputs
        ap.error(f"{[str(o) for o in outputs if o.exists()]} exist; pick a new --out")
    if args.inputs:
        saved = json.loads(args.inputs.read_text())
        if args.variant and saved["variant"] != args.variant:
            ap.error(f"{args.inputs} holds {saved['variant']}, not {args.variant}")
        args.variant = saved["variant"]
        runs = [tuple(x) for x in saved["runs"]]
    else:
        args.variant = args.variant or "all_outputs"
        runs = [(x.split("=", 1)[0], prune(json.loads(Path(x.split("=", 1)[1]).read_text()), args.variant)) for x in args.results]
    # Ordered text levels: one sequential hue, light (original text) to dark (most normalized).
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(len(runs) - 1, 1)) for i in range(len(runs))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    md = []
    for row, pos in enumerate(("output_end", "call_end")):
        for col, split in enumerate(("held_out_environment", "main_task")):
            ax = axes[row, col]
            ax.axhline(0.5, color="0.6", lw=1, ls=":")
            for (label, res), c in zip(runs, colors):
                pts = curve(res, pos, split, args.variant)
                if pts:
                    ls, a, lo, hi = zip(*pts)
                    ax.plot(ls, a, color=c, lw=2, label=label)
                    ax.fill_between(ls, lo, hi, color=c, alpha=0.12, lw=0)
                nl = null_p95(res, pos, split, args.variant)
                if nl:
                    ax.plot(*zip(*nl), color=c, lw=1, ls="--", alpha=0.8)
            if split == "held_out_environment":
                ax.text(0.01, 0.02, "no per-layer shuffle null for this split (see the selection-aware null)",
                        transform=ax.transAxes, fontsize=8, color="0.4")
            ax.set_title(f"{pos}, {split.replace('_', ' ')}")
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel("run AUROC, real vs simulated (higher = more separable)")
    handles = {}  # every series, including those only some panels have (call_end-only reruns)
    for ax in axes.flat:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    order = [label for label, _ in runs if label in handles]
    axes[0, 0].legend([handles[k] for k in order], order, title="text level", fontsize=9)
    fig.suptitle(f"Probe AUROC by layer ({args.variant.replace('_', ' ')}); shaded: 95% task-bootstrap interval; "
                 "dashed: label-shuffle null p95 (main-task folds)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=130)
    for pos in ("output_end", "call_end"):
        for split in ("held_out_environment", "main_task"):
            curves = {label: {l: a for l, a, _, _ in curve(res, pos, split, args.variant)} for label, res in runs}
            layers = sorted({l for c in curves.values() for l in c})
            md += [f"### {pos}, {split.replace('_', ' ')} ({args.variant.replace('_', ' ')})", "",
                   "| Layer | " + " | ".join(curves) + " |", "|---|" + "---|" * len(curves)]
            md += [f"| {l} | " + " | ".join(f"{c[l]:.3f}" if l in c else "–" for c in curves.values()) + " |" for l in layers]
            md.append("")
    args.out.with_suffix(".md").write_text("\n".join(md))
    inputs = outputs[2]
    sources = args.results or saved["sources"]  # a replot keeps the analyses its numbers came from
    inputs.write_text(json.dumps({"variant": args.variant, "sources": sources,
                                  "replotted_from": str(args.inputs) if args.inputs else None, "runs": runs}))
    print(f"-> {args.out.with_suffix('.png')}, {args.out.with_suffix('.md')}, {inputs}")


if __name__ == "__main__":
    main()
