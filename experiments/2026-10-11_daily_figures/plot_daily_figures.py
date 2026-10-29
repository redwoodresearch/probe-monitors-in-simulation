"""Monday daily figures (Aswin's spec, about 13:57 PT on 2026-10-11, via 0d): committed result files only, house colours,
plain-language takeaway titles, no status text, file names or seeds on the images.

  uv run --no-project --with matplotlib --with numpy python experiments/2026-10-11_daily_figures/plot_daily_figures.py --out <dir>

Every number is read with `git show <commit>:<path>` and checked against the value written here, or the script stops.
"""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
SIM, REAL2, GRAY = "#e8833a", "#8fb3e0", "#5a6371"


def show(ref, path):
    return json.loads(subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout)


def dig(obj, key):
    for k in key:
        obj = obj[k]
    return obj


def checked(body, key, want, scale=1, tol=5e-4):
    got = dig(body, key)
    got = [g * scale for g in got] if isinstance(got, list) else got * scale
    for g, w in zip(got if isinstance(got, list) else [got], want if isinstance(want, list) else [want]):
        if abs(g - w) > tol:
            raise SystemExit(f"{key}: file gives {g}, spec says {w}")
    return got


def fig_probe_vs_text(out):
    """Probe vs TF-IDF, fixed sim vs real e1 and the e2-vs-e1 control: both AUROCs on the same task draws."""
    ref, path = "25a89e34", "experiments/2026-10-11_daily_figures/results/probe_vs_tfidf_bars_2026-10-11_141329.json"  # the clean rerun on #729
    b = show(ref, path)["rows"]
    bars = []  # (label, colour, hatch, value, [lo, hi])
    for comp, colour, spec in (("fixed_vs_e1", SIM, (0.8222, [0.7951, 0.8474], 0.6498, [0.6250, 0.6764])),
                               ("e2_vs_e1", REAL2, (0.5278, [0.4972, 0.5587], 0.4875, [0.4694, 0.5068]))):
        p, pci, t, tci = spec
        bars += [("Probe", colour, None, checked(b[comp], ["auroc_probe"], p), checked(b[comp], ["auroc_probe_ci95"], pci)),
                 ("Text\nclassifier", colour, "///", checked(b[comp], ["auroc_tfidf"], t), checked(b[comp], ["auroc_tfidf_ci95"], tci))]
    d = {c: (checked(b[c], ["difference"], v), checked(b[c], ["difference_ci95"], ci))
         for c, v, ci in (("fixed_vs_e1", 0.1724, [0.1391, 0.2048]), ("e2_vs_e1", 0.0403, [0.0026, 0.0774]))}
    # the probe's shuffled-label null: step A's own search repeated on shuffled labels (b7; the same blob at d194b192 and aa4e57be)
    null = show("aa4e57be", "experiments/2026-10-11_claimB_q11_posthoc/results/stepA_catch_null_POSTHOC_2026-10-11_040431.json")["rows"]
    if [r["comparison"] for r in null[:2]] != ["fixed_vs_e1", "e2_vs_e1"]:
        raise SystemExit("step A null rows are not in the expected order")
    nulls = {0: checked(null[0], ["gate", "auroc_null_p95"], 0.5267), 2: checked(null[1], ["gate", "auroc_null_p95"], 0.5285)}  # bar index
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    x = np.array([0, 1, 2.6, 3.6])
    for xi, (lab, col, hatch, v, (lo, hi)) in zip(x, bars):
        ax.bar(xi, v, width=0.8, color=col, alpha=1.0 if hatch is None else 0.55, hatch=hatch, edgecolor="white" if hatch else col, zorder=2)
        ax.errorbar(xi, v, yerr=[[v - lo], [hi - v]], color="black", capsize=5, lw=1.4, zorder=3)
        ax.annotate(f"[{lo:.3f}, {hi:.3f}]", (xi, hi), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        ax.annotate(f"{v:.3f}", (xi, hi), xytext=(0, 17), textcoords="offset points", ha="center", va="bottom", fontsize=13, fontweight="bold")
    for i, y in nulls.items():  # an arrowed tick just left of the probe bar (the open side), pointing at it
        ax.hlines(y, x[i] - 0.66, x[i] - 0.44, colors="black", lw=2.2, zorder=4)
        ax.plot(x[i] - 0.44, y, marker=">", ms=6, color="black", zorder=4)
    ax.axhline(0.5, ls="--", lw=1, color=GRAY, zorder=1)
    ax.text(1.0, 0.5, " coin flip", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=GRAY)
    ax.set_xticks(x, [b_[0] for b_ in bars], fontsize=10)
    for cx, lab in ((0.5, "Fixed simulated runs vs real runs"), (3.1, "Two real runs (control)")):
        ax.annotate(lab, (cx, 0), xycoords=("data", "axes fraction"), xytext=(0, -42), textcoords="offset points", ha="center", va="top",
                    fontsize=10.5, fontweight="bold")
    ax.set_xlim(-0.8, 4.3)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("How well it ranks the two groups apart\n(0.5 = coin flip, 1 = perfect; higher is better)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    (fd, (flo, fhi)), (cd, (clo, chi)) = d["fixed_vs_e1"], d["e2_vs_e1"]
    fig.suptitle("After the formatting fixes, the probe still tells fixed simulated runs from real ones\nbetter than a text classifier does",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.155, f"Probe minus text classifier, fixed simulated vs real runs:  {fd:+.3f} [{flo:+.3f}, {fhi:+.3f}]\n"
                         f"Probe minus text classifier, two real runs:  {cd:+.3f} [{clo:+.3f}, {chi:+.3f}]", ha="center", va="center",
             fontsize=10.5, linespacing=1.6)
    fig.text(0.5, 0.025, "Black lines: 95% intervals from resampling tasks (each comparison: 946 runs over the same 276 tasks; the same draws for every bar and difference).\n"
                         "Arrowed tick: the probe's shuffled-label score (95th percentile of its own search, labels swapped within each cell,\n"
                         "a narrower null than swapping whole tasks). The text classifier has none.",
             ha="center", fontsize=8, color=GRAY, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.215, 0.97, 0.91))
    png = out / "probe_vs_text_classifier.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    return png, f"{path}@{ref}"


def fig_monitor_2x2(out):
    """196-task command monitor, Qwen acting: catch at the frozen threshold targeting 1% false alarms, trained x tested."""
    T = "1% target (refit draws 1-500)"
    real = ("a5de5364", "experiments/2026-10-08_q3_onpolicy_figures/results/q22_levels_POSTHOC_2026-10-11_003931.json")
    sim = ("ce8c9f97", "experiments/2026-10-08_q3_onpolicy_figures/results/sim_trained_levels_POSTHOC_2026-10-11_140215.json")
    ver = ("0552737b", "experiments/2026-10-08_q3_onpolicy_figures/results/scaleup196_verdicts_2026-10-10_164622.json")
    cells = {  # (row, col): (source, arm, catch, catch ci95, achieved false alarms), percentages
        (0, 0): (real, "real catch (real → real)", 89.16, [74.53, 91.67], 1.14),
        (0, 1): (real, "simulated catch (real → sim)", 88.70, [75.71, 91.45], 2.92),
        (1, 0): (sim, "sim-trained on real runs (sim → real)", 90.12, [74.10, 91.63], 1.71),
        (1, 1): (sim, "sim-trained on simulated runs (sim → sim)", 90.35, [74.72, 92.30], 1.16)}
    vals = {}
    for rc, (src, arm, c, ci, fa) in cells.items():
        body = show(*src)[T][arm]
        vals[rc] = (checked(body, ["catch_frozen", "observed"], c, 100, 0.01), checked(body, ["catch_frozen", "ci95"], ci, 100, 0.01),
                    checked(body, ["false_alarm_frozen", "observed"], fa, 100, 0.01))
    v = show(*ver)["endpoints"]
    gap = (checked(v, ["main_tpr_gap_real_minus_sim", "observed"], 0.46, 100, 0.01), checked(v, ["main_tpr_gap_real_minus_sim", "ci95"], [-2.32, 2.50], 100, 0.01))
    ctl = (checked(v, ["catch_control_run2_minus_run1", "observed"], -1.98, 100, 0.01), checked(v, ["catch_control_run2_minus_run1", "ci95"], [-4.41, 1.72], 100, 0.01))
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    M = np.array([[vals[(r, c)][0] for c in (0, 1)] for r in (0, 1)])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    for (r, c), (cv, (lo, hi), fa) in vals.items():
        ax.text(c, r - 0.16, f"{cv:.1f}%", ha="center", va="center", fontsize=24, fontweight="bold", color="white")
        ax.text(c, r + 0.08, f"[{lo:.1f}, {hi:.1f}]", ha="center", va="center", fontsize=11, color="white")
        ax.text(c, r + 0.30, f"false alarms {fa:.1f}%", ha="center", va="center", fontsize=10.5, color="white")
    ax.set_xticks([0, 1], ["tested on\nreal runs", "tested on\nsimulated runs"], fontsize=12)
    ax.set_yticks([0, 1], ["trained on\nreal runs", "trained on\nsimulated runs"], fontsize=12)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fmt = lambda x: f"{x:+.1f}".replace("-", "\u2212")  # noqa: E731
    fig.suptitle("With Qwen doing 196 tasks, the monitor trained on real runs\ncatches real and simulated impactful commands about equally:\na 10-point gap is ruled out",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.14, "Share of impactful commands caught at a threshold targeting 1% false alarms, set on each row's own training runs\n(false alarms achieved on the test runs shown in each cell)",
             ha="center", fontsize=9.5, color=GRAY)
    fig.text(0.5, 0.075, f"Trained on real runs, real minus simulated:  {fmt(gap[0])} points [{fmt(gap[1][0])}, {fmt(gap[1][1])}]\n"
                         f"Trained on real runs, real run 2 minus real run 1:  {fmt(ctl[0])} points [{fmt(ctl[1][0])}, {fmt(ctl[1][1])}]",
             ha="center", va="center", fontsize=11, linespacing=1.6)
    fig.text(0.5, 0.018, "Brackets: 95% intervals over 500 refits on resampled tasks.", ha="center", fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, 0.17, 1, 0.87))
    png = out / "monitor_196_tasks_2x2.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    return png, "; ".join(f"{p_}@{r_}" for r_, p_ in (real, sim, ver))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for png, src in [fig_monitor_2x2(args.out), fig_probe_vs_text(args.out)]:
        print(png, "<-", src)


if __name__ == "__main__":
    main()
