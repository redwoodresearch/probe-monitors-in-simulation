"""Figure 3: the on-policy 2×2 as literal 2×2 grids (q2's figure 2 style), with the off-policy grid beside it. $0, local.

Rows: AUROC, standardized pAUC (false alarms 0–5%), catch at 1% false alarms. Left: on-policy (analyze_onpolicy_2x2.py);
right: off-policy, final figure 2 (q2's refit bootstrap, Qwen3.5-27B reader: refit_<stamp>.json and refit_pauc_<stamp>.json,
PR #260/#267), shown side by side only: this is not amendment 15's D. In each grid, rows are trained on real /
simulated and columns tested on real / simulated; each cell shows the value and its 95% interval, or why it has none.
Under each on-policy grid: the gaps, simulated − real (amendment 43), with 95% intervals, and for AUROC and pAUC the
secondary verdict at ±0.03 ("borrowed margin, not consequence-based", amendment 47).

  python3 plot_figure3.py results/onpolicy_2x2_analysis_<label>_<stamp>.json <refit json> <refit pauc json>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
W = ("real", "sim")
MEASURES = [("auroc", "AUROC", (0.9, 1.0)), ("pauc", "Standardized pAUC, false alarms 0–5%", (0.75, 1.0)),
            ("catch", "Catch at 1% false alarms (noisy)", (0.4, 1.0))]


def fmt(metric, v):
    return "—" if v is None else (f"{v:.1%}" if metric.startswith("catch") else f"{v:.3f}")


def dfmt(metric, v):
    return "—" if v is None else (f"{v * 100:+.1f} pts" if metric.startswith("catch") else f"{v:+.4f}")


def grid(ax, cells, metric, title, lo, hi, nonull=False):
    vals = [[(cells[f"{t}->{te}"][metric]["value"] if cells[f"{t}->{te}"][metric]["value"] is not None else lo) for te in W] for t in W]
    ax.imshow(vals, cmap="Blues", vmin=lo, vmax=hi)
    for i, t in enumerate(W):
        for j, te in enumerate(W):
            c = cells[f"{t}->{te}"][metric]
            dark = c["value"] is not None and (c["value"] - lo) / (hi - lo) > 0.55
            col = "white" if dark else "black"
            if c.get("status"):
                short = "unestimable\nat this size" if c["status"].startswith("unestimable") else c["status"]
                ax.text(j, i, short, ha="center", va="center", fontsize=9, color=col)
                continue
            ax.text(j, i - 0.08, fmt(metric, c["value"]), ha="center", va="center", fontsize=16, weight="bold", color=col)
            if c.get("ci95"):
                ax.text(j, i + 0.22, f"[{fmt(metric, c['ci95'][0])}, {fmt(metric, c['ci95'][1])}]", ha="center", va="center", fontsize=9, color=col)
            if nonull and (metric == "pauc" or (t == "sim" and metric == "auroc")):
                ax.text(j, i + 0.38, "no shuffle null run", ha="center", va="center", fontsize=8, style="italic", color=col)
    ax.set_xticks([0, 1], ["tested on\nreal", "tested on\nsimulated"])
    ax.set_yticks([0, 1], ["trained on\nreal", "trained on\nsimulated"])
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=10, pad=30)


def main(on_path: Path, refit_path: Path, pauc_path: Path):
    A = json.loads(on_path.read_text())
    refit, pauc = (json.loads(p.read_text())["readers"]["qwen3.5"] for p in (refit_path, pauc_path))
    if refit["fingerprint"] != pauc["fingerprint"]:
        raise ValueError("the refit and pAUC files come from different replicates")
    off = {c: {"auroc": refit[c]["auroc"], "pauc": pauc[c], "catch": refit[c]["catch, calibrated in test world"]}
           for c in ("real->real", "real->sim", "sim->real", "sim->sim")}
    fig, axes = plt.subplots(3, 2, figsize=(17, 18))
    for row, (m, name, (lo, hi)) in enumerate(MEASURES):
        grid(axes[row, 0], A["cells"], m, f"On-policy: {name} (higher is better)", lo, hi, nonull=True)
        grid(axes[row, 1], off, m, f"Off-policy, final figure 2 (side by side, not D): {name}", lo, hi)
        lines = []
        for gname, g in A["gaps"].items():
            if m in g:
                d = g[m]
                v = d.get("secondary_verdict_at_0.03")
                if d.get("status", "").startswith("unestimable"):
                    lines.append(f"{gname}: unestimable at this size (a calibration fold under 100 read-only calls)")
                    continue
                lines.append(f"{gname}: {dfmt(m, d['value'])} "
                             f"[{dfmt(m, d['ci95'][0]) if d.get('ci95') else '—'}, {dfmt(m, d['ci95'][1]) if d.get('ci95') else '—'}]"
                             + (f"  {d['status']}" if d.get("status") else "")
                             + (f"  (±0.03, borrowed margin: {v['verdict']})" if v and v.get("verdict") else "  (no verdict at pilot size)" if v else ""))
        axes[row, 0].text(0.5, -0.06, "\n".join(lines), transform=axes[row, 0].transAxes, ha="center", va="top", fontsize=8)
    sim = A.get("simulator", "")
    fig.suptitle(f"{A['label']}: on-policy 2×2, impactful-command probe, Qwen3.5-27B acting; simulator {sim}", fontsize=12, y=0.995)
    fig.text(0.01, 0.005, f"Preliminary. Attack runs, analyzable pairs ({A['summary']['analyzable_pairs']} tasks); readout: the last token of each command "
             f"(rule 1). Gaps are simulated − real. On-policy intervals: {A['draws']} task resamples, each refitting, reselecting and recalibrating; "
             f"verdicts: {A['verdicts']}.\nOff-policy panels (final figure 2, q2's refit bootstrap): Qwen3.5-27B reading Sol's transcripts; matched calls, layer "
             f"and C re-picked in every draw, held-out calibration. Shown side by side only; amendment 15's D is computed separately.\n"
             f"Data: {on_path.name}; off-policy: {refit_path.name}, {pauc_path.name}.\nAmendments: " + "; ".join(f"{k}: {v}" for k, v in A["amendments"].items()), fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    out = HERE / "figures" / on_path.with_suffix(".png").name.replace("onpolicy_2x2_analysis", "figure3")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print("->", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
