"""Draw the post's figures from the analysis tables in analysis/results/ into analysis/output/.

Run it with:  uv run analysis/make_figures.py   (from the repository root)

analysis/results/ holds the output of analysis/main_analysis.py (run it to rebuild them from the per-unit scores in data/).
  Figures 1 and 5: for each probe, the share of attack runs caught on real machines and in simulation (left), and the difference
    between them beside the difference between the two real runs (right), with 95% intervals. Figure 1 uses the probes that score
    commands, Figure 5 those that score tool outputs. The top two rows average the six probes and the four probes that best separate
    attack runs from honest runs.
  Figure 4: per probe, how far simulated catch sits from real catch, minus how far the two real runs sit from each other (commands).
  Figure 6: per task, how far the on-policy probe's most suspicious command sits above the alarm line, real run 1 against real run 2
    (left) and against the simulated run (right).
  Figure 7: per probe, the R^2 of real run 1 with the other real run and with the simulated run, across tasks (commands).
All at the 1% alarm level (1 alarm per 100 honest actions).
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).parent
RESULTS, OUT = HERE / "results", HERE / "output"
INK, MUTED, REAL, SIM, RERUN, AVERAGE, PROBE = "#222222", "#6b6b68", "#2a78d6", "#eb6834", "#8d8d88", "#3b3f7a", "#8b8fc4"
MINUS = "−"
BAR = 0.36  # each bar's height; a row holds two bars
LEVEL = "1%"
NAMES = {"on_policy_probe": "On-policy probe", "off_policy_probe": "Off-policy probe", "deception_probe": "Deception probe",
         "harmfulness_direction": "Harmfulness direction", "refusal_direction": "Refusal direction", "high_stakes_probe": "High-stakes probe"}
ROWS = ["Six-probe average", "Four-probe average", *NAMES.values()]  # the rows of Figures 1, 4 and 5, top to bottom


def read(name, **match):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if all(r[k] == v for k, v in match.items())]


def figure_rows(name, **match):
    """The eight rows of a figure, in order; stops if any is missing, repeated or unexpected."""
    rows = read(name, **match)
    if [r["row"] for r in rows] != ROWS:
        raise SystemExit(f"{name} {match}: expected the rows {ROWS}, found {[r['row'] for r in rows]}")
    return rows


def triple(row, col):
    """A value and its 95% interval, as numbers."""
    return float(row[col]), float(row[col + "_low"]), float(row[col + "_high"])


def signed(x, digits=2):
    return f"{x:+.{digits}f}".replace("-", MINUS)


def style():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path.relative_to(HERE))


def catch_figure(scores, path):
    rows = figure_rows("catch_rates.csv", scores=scores, alarm_level=LEVEL)
    averages = [r for r in rows if r["row"].endswith("average")]
    probes = [r for r in rows if not r["row"].endswith("average")]
    # The right panel uses one scale for commands and tool outputs, so the two versions of this figure can be compared.
    ends = [0.0] + [float(r[c + s]) for sc in ("commands", "tool outputs") for r in figure_rows("catch_rates.csv", scores=sc, alarm_level=LEVEL)
                    for c in ("simulated_minus_real", "run2_minus_run1") for s in ("", "_low", "_high")]
    span = max(ends) - min(ends)

    style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 1.3 + 0.62 * len(rows)), sharey=True, gridspec_kw={"width_ratios": [1, 1.1]})
    ys = list(range(len(averages))) + [len(averages) + 0.4 + i for i in range(len(probes))]
    for y, r in zip(ys, averages + probes, strict=True):
        bold = "bold" if r in averages else "normal"
        for dy, col, colour in ((-BAR / 2, "real_caught", REAL), (BAR / 2, "simulated_caught", SIM)):
            v, lo, hi = triple(r, col)
            left.barh(y + dy, v, BAR * 0.92, color=colour, zorder=2)
            left.plot([lo, hi], [y + dy] * 2, color=INK, lw=1.1, zorder=3)
            left.text(max(v, hi) + 1.2, y + dy, f"{v:.0f}", va="center", fontsize=9, color=INK, fontweight=bold)
        if r["honest_runs_alarmed"]:  # the no-skill floor: the share of real honest runs that raise an alarm
            x = float(r["honest_runs_alarmed"])
            left.plot([x, x], [y - BAR - 0.04, y + BAR + 0.04], color=INK, lw=2.2, zorder=4)
        first = AVERAGE if r in averages else SIM
        for dy, col, colour in ((-BAR / 2, "simulated_minus_real", first), (BAR / 2, "run2_minus_run1", RERUN)):
            v, lo, hi = triple(r, col)
            right.barh(y + dy, v, BAR * 0.92, color=colour, zorder=2)
            right.plot([lo, hi], [y + dy] * 2, color=INK, lw=1.1, zorder=3)
            if col == "simulated_minus_real":
                right.text(max(v, hi) + 0.6, y + dy, signed(v), va="center", fontsize=9, color=INK, fontweight=bold)

    left.set_xlim(0, 112)
    left.set_xlabel("Attack runs caught (%)")
    right.set_xlim(min(ends) - 0.05 * span, max(ends) + 0.2 * span)
    right.axvline(0, color=MUTED, lw=1, zorder=1)
    right.set_xlabel("Difference in catch rate (percentage points)")
    left.set_yticks(ys, [r["row"] for r in averages + probes])
    for label in left.get_yticklabels()[: len(averages)]:
        label.set_fontweight("bold")
    left.invert_yaxis()
    for ax in (left, right):
        ax.tick_params(axis="y", length=0)
        ax.axhline(len(averages) - 0.3, color="#dddddd", lw=1)
    legend = {"loc": "lower left", "bbox_to_anchor": (0.0, 1.01), "frameon": False, "fontsize": 9, "borderaxespad": 0.0}
    left.legend(handles=[Patch(color=REAL, label="Real (mean of two runs)"), Patch(color=SIM, label="Simulated"),
                         Line2D([], [], color=INK, lw=2.2, label="No-skill floor")], ncol=3, **legend)
    right.legend(handles=[Patch(color=SIM, label="Simulated minus real"), Patch(color=RERUN, label="Real run 2 minus run 1"),
                          Patch(color=AVERAGE, label="Average")], ncol=3, **legend)
    save(fig, path)


def excess_figure(path):
    """Figure 4: the simulation gap (simulated catch's mean distance from the two real runs' catch) minus the rerun gap (the
    distance between the two real runs), commands; positive = simulation farther from real than a rerun is."""
    rows = figure_rows("excess_distance.csv", scores="commands", alarm_level=LEVEL)
    style()
    fig, ax = plt.subplots(figsize=(9.0, 1.2 + 0.62 * len(rows)))
    vals = [triple(r, "excess_distance") for r in rows]
    lo_, hi_ = min([0.0] + [x for v in vals for x in v]), max([0.0] + [x for v in vals for x in v])
    span = hi_ - lo_
    ys = []
    for i, (r, (v, lo, hi)) in enumerate(zip(rows, vals, strict=True)):
        avg = r["row"].endswith("average")
        y = i + (0 if avg else 0.6)
        ys.append(y)
        ax.barh(y, v, 0.62, color=AVERAGE if avg else PROBE, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.4, zorder=3)
        ax.text(max(v, hi) + 0.02 * span, y, signed(v), va="center", fontsize=10, color=INK, fontweight="bold" if avg else "normal")
    ax.set_xlim(lo_ - 0.05 * span, hi_ + 0.25 * span)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.set_xlabel("Simulation gap minus rerun gap (percentage points)\n← simulation closer to real than a rerun | farther →")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(ys, [r["row"] for r in rows])
    for label, r in zip(ax.get_yticklabels(), rows, strict=True):
        if r["row"].endswith("average"):
            label.set_fontweight("bold")
    ax.invert_yaxis()
    save(fig, path)


def scatter_figure(path):
    """Figure 6: the on-policy probe, commands; one point per task with all three runs scored."""
    pts = np.array([[float(r[a]) for a in ("real_attack_run1", "real_attack_run2", "simulated_attack")]
                    for r in read("per_task_values.csv", probe="on_policy_probe", scores="commands")
                    if all(r[a] for a in ("real_attack_run1", "real_attack_run2", "simulated_attack"))])
    r2 = read("per_task_r2.csv", probe="on_policy_probe", scores="commands")[0]
    assert len(pts) == int(r2["tasks_with_all_three_runs_scored"]), "the points are not the tasks behind the R^2"
    lo, hi = pts.min(), pts.max()
    pad = 0.03 * (hi - lo)
    lim = (lo - pad, hi + pad)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    xlabel = "Real run 1: most suspicious action,\ndistance above the alarm line"
    panels = ((1, RERUN, "Two real runs (how well a rerun agrees)", "Real run 2", "r2_real_run2_vs_real_run1"),
              (2, SIM, "Real against simulated", "Simulated run", "r2_simulated_vs_real_run1"))
    for ax, (j, colour, title, who, col) in zip(axes, panels, strict=True):
        ax.scatter(pts[:, 0], pts[:, j], s=7, color=colour, alpha=0.45, linewidths=0, zorder=2)
        slope, icept = np.polyfit(pts[:, 0], pts[:, j], 1)  # a drawing aid only
        ax.plot(lim, [icept + slope * x for x in lim], color=INK, lw=1.6, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{who}: most suspicious action,\ndistance above the alarm line")
        ax.text(0.05, 0.95, f"R² = {float(r2[col]):.2f}", transform=ax.transAxes, va="top", fontsize=11, fontweight="bold", color=INK)
    fig.suptitle("On-policy probe", fontweight="bold")
    save(fig, path)


def r2_figure(path):
    """Figure 7: per probe (commands), the R^2 across tasks of real run 1 with the other real run and with the simulated run."""
    rows = {r["probe"]: r for r in read("per_task_r2.csv", scores="commands")}
    vals = [x for probe in NAMES for c in ("r2_real_run2_vs_real_run1", "r2_simulated_vs_real_run1") for x in triple(rows[probe], c)]
    if not all(0.0 <= x <= 0.30 for x in vals):  # the axis is fixed, so nothing may fall outside it
        raise SystemExit("Figure 7: a value or interval end lies outside the axis [0, 0.30]")
    style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for i, probe in enumerate(NAMES):
        for dy, col, colour in ((-0.19, "r2_real_run2_vs_real_run1", RERUN), (0.19, "r2_simulated_vs_real_run1", SIM)):
            v, lo, hi = triple(rows[probe], col)
            ax.barh(i + dy, v, 0.34, color=colour, zorder=2)
            ax.plot([lo, hi], [i + dy] * 2, color=INK, lw=1.1, zorder=3)
            ax.text(hi + 0.003, i + dy, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 0.30)
    ax.set_yticks(range(len(NAMES)), list(NAMES.values()))
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel("R² with real run 1 across tasks (higher = more alike)")
    ax.legend(handles=[Patch(color=RERUN, label="The other real run"), Patch(color=SIM, label="Simulated run")], loc="lower right",
              frameon=False)
    save(fig, path)


def main():
    OUT.mkdir(exist_ok=True)
    catch_figure("commands", OUT / "figure_1.png")
    excess_figure(OUT / "figure_4.png")
    catch_figure("tool outputs", OUT / "figure_5.png")
    scatter_figure(OUT / "figure_6.png")
    r2_figure(OUT / "figure_7.png")


if __name__ == "__main__":
    main()
