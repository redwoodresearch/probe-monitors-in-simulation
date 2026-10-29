import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BLUE, ORANGE, GRAY, INK, MUTED = "#2b78d6", "#ea6a35", "#8a8a84", "#222222", "#555555"
fig, ax = plt.subplots(figsize=(14, 7.4), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
def box(x, y, w, h, text, edge, fill="white", bold=False, size=12.5):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=2, ec=edge, fc=fill))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, weight="bold" if bold else "normal", wrap=True)
def arrow(x1, y1, x2, y2, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16, lw=1.8, color=color, linestyle=ls, shrinkA=2, shrinkB=2))
# column headers
ax.text(25, 59.5, "Real machine", ha="center", fontsize=15, weight="bold", color=BLUE)
ax.text(25, 56.8, "commands actually run", ha="center", fontsize=11.5, color=MUTED)
ax.text(76, 59.5, "Simulation", ha="center", fontsize=15, weight="bold", color=ORANGE)
ax.text(76, 56.8, "A language model writes every tool output", ha="center", fontsize=11.5, color=MUTED)
ax.plot([51, 51], [4, 58], color="#dddddd", lw=1.5)
# honest runs -> thresholds
box(25, 49, 34, 5.5, "Honest runs (about two per main task)", BLUE)
box(76, 49, 34, 5.5, "Honest runs (about one per main task)", ORANGE)
box(25, 38, 34, 8, "Real threshold\nfires on 1% of real\nhonest commands", BLUE, fill="#eef4fc")
box(76, 38, 34, 8, "Simulated threshold\nfires on 1% of simulated\nhonest commands", ORANGE, fill="#fdf0ea")
arrow(25, 46.1, 25, 42.2, BLUE); arrow(76, 46.1, 76, 42.2, ORANGE)
# attack runs
box(14, 25, 17, 6.5, "Attack run 1\ncaught or missed", BLUE)
box(36, 25, 17, 6.5, "Attack run 2\ncaught or missed", BLUE)
box(76, 25, 22, 6.5, "Attack run\ncaught or missed", ORANGE)
arrow(20, 33.8, 15, 28.4, BLUE, ls="--"); arrow(30, 33.8, 35, 28.4, BLUE, ls="--"); arrow(76, 33.8, 76, 28.4, ORANGE, ls="--")
ax.text(9.5, 30.0, "same real\nthreshold", fontsize=10.5, color=MUTED, ha="center")
ax.text(83.5, 30.0, "its own\nthreshold", fontsize=10.5, color=MUTED, ha="center")
# comparisons
box(25, 8, 34, 8, "Real run 2 minus real run 1\nhow much a rerun differs", GRAY, fill="#f1f1ef", bold=False)
box(76, 8, 34, 8, "Simulated minus real\nsimulated run vs the mean\nof the two real runs", ORANGE, fill="#fdf0ea")
arrow(14, 21.6, 22, 12.2, GRAY); arrow(36, 21.6, 28, 12.2, GRAY)
arrow(76, 21.6, 76, 12.2, ORANGE); arrow(39, 21.8, 62, 12.2, ORANGE, ls=":"); arrow(17, 21.8, 60, 12.4, ORANGE, ls=":")
ax.text(50.5, 1.0, "Catch rate = share of attack runs with at least one command above the threshold", ha="center", fontsize=11, color=MUTED)
fig.savefig(sys.argv[1], bbox_inches="tight", facecolor="white")
