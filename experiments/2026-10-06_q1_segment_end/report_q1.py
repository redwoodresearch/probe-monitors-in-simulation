"""Q1 report: one summary table and a Figure 1 from the segment-end analysis, plus rows from other tracks.

Reads the newest results/q1_segment_end_*.json (or --results) and, optionally, --external JSON files with rows
from other tracks (TF-IDF, LLM judge), each {"rows": [{"name", "metric" ("auroc" or "pairwise accuracy"), "run",
"step", "no_editor_run", "base200_run", "overlap800_run", "under_32k_run", "null_e1_vs_e2_run", "note"}]} where
every value is {"auroc" or "accuracy", "ci95": [lo, hi]} or null. --summary is a short plain-language paragraph for the
top ("What this means"). Writes results.md and a figure next to it.

  uv run --with matplotlib python experiments/2026-10-06_q1_segment_end/report_q1.py [--external path.json ...] [--summary text.md]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
# Reference palette slots 1-3 (dataviz skill), validated as a set; grays for baselines and nulls.
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
RVS, NULL = "real_vs_sim", "null_e1_vs_e2"
WHAT = {"output_end": "tool outputs (last content token of each output): do the outputs look simulated?",
        "call_end": "agent tool calls (last content token of each call): does the agent's own text read differently "
                    "after it has seen simulated outputs?"}


def val(d):
    if not d:
        return None
    return d.get("auroc", d.get("accuracy"))


def held(r, variant="all_outputs"):
    """Held-out-environment results: one per variant (GPU analysis) or a single all-outputs entry (CPU analysis)."""
    h = r["held_out_environment"]
    return h.get(variant) if variant in h else (h if variant == "all_outputs" and "run" in h else None)


def fmt(d):
    v = val(d)
    return "–" if v is None else f"{v:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path)
    ap.add_argument("--external", type=Path, nargs="*", default=[])
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--baselines", type=Path, help="baselines JSON (length, perplexity, position) when --results lacks it")
    ap.add_argument("--headline", type=Path, help="a headline file (e.g. q1_gpu_headline_max48_*.json) overriding --results'")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="markdown sections appended before the figure")
    args = ap.parse_args()
    full = sorted((p for p in (HERE / "results").glob("q1_*.json")
                   if not p.stem.endswith("_test") and ("q1_gpu_2" in p.name or "q1_segment_end_2" in p.name)),
                  key=lambda p: re.search(r"\d{4}-\d{2}-\d{2}_\d{6}", p.stem).group())  # by date and time (a config name may follow)
    path = args.results.resolve() if args.results else full[-1]
    res = json.loads(path.read_text())
    n_dirs = f"{res['config']['random_directions']:,}"
    if "baselines" not in res:
        res["baselines"] = json.loads(args.baselines.read_text())
    head = json.loads(args.headline.read_text())["headline"] if args.headline else res.get("headline")
    head_layers = json.loads(args.headline.read_text()).get("layers") if args.headline else None
    layers = sorted(int(l) for l in res["by_layer"])
    positions = list(res["by_layer"][str(layers[0])]["results"])
    R = lambda l, pos: res["by_layer"][str(l)]["results"][pos]
    # Each probe's row uses its layer with the best real-vs-sim run-level AUROC: chosen on the same data, so slightly
    # optimistic; every layer is in the per-layer tables and the figure.
    best = {pos: max(layers, key=lambda l: val(R(l, pos)[RVS]["all_outputs"]["run"])) for pos in positions}
    if head:  # the headline layer, chosen by inner CV on training folds; per-layer rows use the layer chosen most often
        from collections import Counter

        best = {pos: Counter(head[pos][RVS]["all_outputs"]["chosen_layer_by_fold"].values()).most_common(1)[0][0]
                for pos in positions}
    rows = []
    for pos in positions:
        l = best[pos]
        r, n = R(l, pos)[RVS], R(l, pos)[NULL]
        sh = r["all_outputs"]["run"].get("label_shuffle_null")
        if head:
            h, hn = head[pos][RVS], head[pos][NULL]
            rows.append({"name": f"Probe, {pos}, headline layer", "metric": "auroc", "run": h["all_outputs"]["run"],
                         "no_editor_run": h["no_editor_outputs"]["run"],
                         "no_errors_run": (h.get("no_tool_error_outputs") or {}).get("run"),
                         "null_e1_vs_e2_run": hn["all_outputs"]["run"],
                         "note": (f"layers {min(head_layers)}-{max(head_layers)} only; " if head_layers else "")
                                 + "layer chosen per fold by inner CV on training folds: "
                                 + ", ".join(f"fold {f}: L{v}" for f, v in h["all_outputs"]["chosen_layer_by_fold"].items())})
        rows.append({"name": f"Probe, {pos}, layer {l}", "metric": "auroc", "run": r["all_outputs"]["run"],
                     "step": r["all_outputs"]["step"], "no_editor_run": r["no_editor_outputs"]["run"],
                     "no_errors_run": (r.get("no_tool_error_outputs") or {}).get("run"),
                     "base200_run": r["all_outputs"]["base200"]["run"], "overlap800_run": r["all_outputs"]["overlap800"]["run"],
                     "under_32k_run": r["all_outputs"]["under_32k"]["run"], "null_e1_vs_e2_run": n["all_outputs"]["run"],
                     "note": ("most-chosen headline layer" if head else "best layer (post hoc)")
                             + (f"; shuffle null p95 {sh['p95']:.3f}" if sh else "")
                             + ("; TF-IDF on call text alone is about 0.53 per run, so separation here comes from the "
                                "preceding outputs the probe has read" if pos == "call_end" else "")})
        rows.append({"name": f"Probe, {pos}, held-out environment, layer {l}", "metric": "auroc",
                     "run": held(r)["run"], "step": held(r)["step"],
                     "no_editor_run": (held(r, "no_editor_outputs") or {}).get("run"),
                     "no_errors_run": (held(r, "no_tool_error_outputs") or {}).get("run"),
                     "null_e1_vs_e2_run": held(n)["run"], "note": "trained on 9 environments, tested on the 10th"})
        rd = lambda x: (x.get("best_random_direction") or {})  # absent in lite (all-layers) analyses
        if rd(r["all_outputs"]):
            rows.append({"name": f"Best of {n_dirs} random directions, {pos}, layer {l}", "metric": "auroc",
                         "run": rd(r["all_outputs"]).get("run"), "step": rd(r["all_outputs"]).get("step"),
                         "no_editor_run": rd(r["no_editor_outputs"]).get("run"),
                         "null_e1_vs_e2_run": rd(n["all_outputs"]).get("run"), "note": "picked on the training folds"})
    lo = best[positions[0]]
    for kind, s in res["by_layer"][str(lo)].get("sensitivity", {}).items():
        rows.append({"name": f"Probe at {kind}, layer {lo}", "metric": "auroc", "run": s["run"], "step": s["step"],
                     "note": "template token: sensitivity check; sees the trailing newline" if kind == "output_marker"
                     else "template token: sensitivity check"})
    b, bn = res["baselines"]["results"][RVS], res["baselines"]["results"][NULL]
    for step_key, total_key, name in (("output_length", "total_output_tokens", "Output length"),
                                      ("perplexity", "run_perplexity", "Tool-output perplexity (Qwen3-32B)")):
        r, rn, n = b["all_outputs"][step_key], b["no_editor_outputs"][step_key], bn["all_outputs"][step_key]
        rows.append({"name": name, "metric": "auroc", "run": r["run"], "step": r["step"], "no_editor_run": rn["run"],
                     "base200_run": r["base200"]["run"], "overlap800_run": r["overlap800"]["run"],
                     "under_32k_run": r["under_32k"]["run"], "null_e1_vs_e2_run": n["run"],
                     "note": f"run total: {fmt(b['all_outputs'][total_key]['run'])}"})
    rows.append({"name": "Token position only", "metric": "auroc", "step": b["all_outputs"]["position"]["step"], "note": "step level"})
    for p in args.external:
        rows += json.loads(p.read_text())["rows"]

    cols = [("run", "Run-level"), ("step", "Step-level"), ("no_editor_run", "Run, no editor outputs"),
            ("no_errors_run", "Run, no tool-error outputs"),
            ("base200_run", "Run, base200"), ("overlap800_run", "Run, overlap800"), ("under_32k_run", "Run, under 32k"),
            ("null_e1_vs_e2_run", "Null: real e1 vs e2 (run)")]
    md = ["# Q1: can a probe tell real LinuxArena attack runs from simulated ones?", "", "## What this means", "",
          args.summary.read_text().strip() if args.summary else "_(to be written once the numbers are in)_", "",
          "## Setup", "",
          f"Source: `{path.relative_to(HERE) if path.is_relative_to(HERE) else path}`. Real attack epoch 1 vs simulated attack on the paired cells of "
          "`splits/exclusions_v3.json` (transcripts cut to their first 32k tokens), shared main-task folds; 95% intervals "
          "resample main tasks. Higher is better at telling real from simulated; 0.5 is chance.", "",
          "What the probe reads: Qwen3-32B's internal state (the residual stream at one layer) at the last content token of a "
          "tool output or tool call, which reflects everything Qwen has read up to that point in the transcript, not just "
          "that segment. Two probes are trained separately:"] + [f"- `{p}`: {WHAT.get(p, p)}" for p in positions] + [
          "", "Step level: one example per tool output (or tool call). Run level: a run's score is the mean of its step "
          "scores. Caveat: per step, the probe has read the whole transcript up to that point, while the text baselines see "
          "only that one output, so per-step gains over them can come from context; the per-run comparison is the fairer one.",
          "", "## Summary", "",
          "| Method | " + " | ".join(c for _, c in cols) + " | Note |", "|" + "---|" * (len(cols) + 2)]
    for r in rows:
        unit = " (pairwise accuracy)" if r.get("metric") == "pairwise accuracy" else ""
        md.append(f"| {r['name']}{unit} | " + " | ".join(fmt(r.get(k)) for k, _ in cols) + f" | {r.get('note', '')} |")
    for pos in positions:
        md += ["", f"## Probe by layer: {pos} (run level)", "",
               "| Layer | Real vs sim | No editor outputs | Held-out environment | Random directions | Null: e1 vs e2 | Shuffle null p95 | Step-level |",
               "|---|---|---|---|---|---|---|---|"]
        for l in layers:
            r, n = R(l, pos)[RVS], R(l, pos)[NULL]
            sh = r["all_outputs"]["run"].get("label_shuffle_null") or {}
            md.append(f"| {l} | {fmt(r['all_outputs']['run'])} | {fmt(r['no_editor_outputs']['run'])} | "
                      f"{fmt(held(r)['run'])} | {fmt((r['all_outputs'].get('best_random_direction') or {}).get('run'))} | "
                      f"{fmt(n['all_outputs']['run'])} | {sh.get('p95', float('nan')):.3f} | {fmt(r['all_outputs']['step'])} |")
        l = best[pos]
        md += ["", f"Held-out environment, {pos}, layer {l} (run level):", "",
               "| Environment held out | Real vs sim | Null: e1 vs e2 | Runs |", "|---|---|---|---|"]
        for e, v in sorted(held(R(l, pos)[RVS])["per_environment"].items()):
            vn = held(R(l, pos)[NULL])["per_environment"][e]
            md.append(f"| {e} | {fmt(v['run'])} | {fmt(vn['run'])} | {v['n_runs']} |")
        md += ["", f"Step-level AUROC by token position, {pos}, layer {l}:", "", "| Band | Probe | Position only |", "|---|---|---|"]
        for k, v in R(l, pos)[RVS]["all_outputs"].items():
            if k.startswith("band_"):
                md.append(f"| {k[5:].replace('_', '-')} | {fmt(v['step'])} | {fmt(v['position_only'])} |")

    # Figure 1: (a) every method's run-level score with its interval and the e1-vs-e2 null; (b) the probes by layer.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 0.42 * len(rows) + 2.4), gridspec_kw={"width_ratios": [1.3, 1]})
    shown = [r for r in rows if val(r.get("run"))]
    for i, r in enumerate(reversed(shown)):
        v, (lo_, hi_) = val(r["run"]), r["run"]["ci95"]
        ax.plot([lo_, hi_], [i, i], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(v, i, "o", color=BLUE, ms=8, mec="white", mew=2)
        ax.text(hi_ + 0.01, i, f"{v:.2f}", va="center", color=INK, fontsize=9)
        if val(r.get("null_e1_vs_e2_run")):
            ax.plot(val(r["null_e1_vs_e2_run"]), i, "D", color=GRAY, ms=7, mfc="white", mew=2)
    ax.set_yticks(range(len(shown)), [r["name"] + (" (pairwise acc.)" if r.get("metric") == "pairwise accuracy" else "")
                                      for r in reversed(shown)], fontsize=8.5, color=INK)
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xlim(0.35, 1.06)
    ax.set_xlabel("Run-level AUROC, real vs simulated (higher = more separable)", color=INK)
    ax.set_title("(a) Methods: dot = real vs sim, ◇ = real e1 vs e2 null", loc="left", fontsize=10, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    styles = {"output_end": (BLUE, "o", "Probe, tool outputs"), "call_end": (ORANGE, "s", "Probe, tool calls")}
    for pos in positions:
        color, marker, label = styles.get(pos, (AQUA, "^", pos))
        bx.fill_between(layers, [R(l, pos)[RVS]["all_outputs"]["run"]["ci95"][0] for l in layers],
                        [R(l, pos)[RVS]["all_outputs"]["run"]["ci95"][1] for l in layers], color=color, alpha=0.15, lw=0)
        bx.plot(layers, [val(R(l, pos)[RVS]["all_outputs"]["run"]) for l in layers], f"-{marker}", color=color, lw=2, ms=6,
                mec="white", mew=1.5, label=label)
        bx.plot(layers, [val(R(l, pos)[RVS]["no_editor_outputs"]["run"]) for l in layers], f":{marker}", color=color, lw=1.5,
                ms=5, mfc="white", label=f"{label}, no editor")
        bx.plot(layers, [val(held(R(l, pos)[RVS])["run"]) for l in layers], "--", color=color, lw=1.2,
                label=f"{label}, held-out environment")
    if R(layers[0], positions[0])[RVS]["all_outputs"].get("best_random_direction"):
        bx.plot(layers, [val(R(l, positions[0])[RVS]["all_outputs"]["best_random_direction"]["run"]) for l in layers], "--",
                color=GRAY, lw=1.5, label=f"Best of {n_dirs} random directions (outputs)")
    bx.plot(layers, [val(R(l, positions[0])[NULL]["all_outputs"]["run"]) for l in layers], "-D", color=GRAY, lw=2, ms=6,
            mfc="white", label="Null: real e1 vs e2 (outputs)")
    bx.axhline(0.5, color=MUTED, lw=1, ls=":")
    bx.set_xticks(layers)
    bx.set_xlabel("Qwen3-32B layer (of 64)", color=INK)
    bx.set_ylabel("Run-level AUROC (higher = more separable)", color=INK)
    bx.set_title("(b) Probes by layer", loc="left", fontsize=10, color=INK)
    bx.legend(frameon=False, fontsize=8, loc="center right")
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    fig.tight_layout()
    fig_path = HERE / "results" / f"figure1_{path.stem.removeprefix('q1_segment_end_')}.png"
    fig.savefig(fig_path, dpi=150)
    for p in args.extra:
        md += ["", p.read_text().strip()]
    md += ["", f"![Figure 1]({fig_path.relative_to(HERE)})"]
    (HERE / "results.md").write_text("\n".join(md) + "\n")
    print(f"wrote {HERE / 'results.md'} and {fig_path}")


if __name__ == "__main__":
    main()
