"""Method schematic in the style of the SCE paper's Figures 1-2: execute twice, simulate once, compare catch rates."""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

S = sys.argv[1]
SIM, REAL, INK, MUTED, SIMBG = "#eb6834", "#2a78d6", "#222222", "#6b6b68", "#fbeee6"
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11); ax.set_ylim(-0.35, 4.6); ax.axis("off")

def box(x, y, w, h, text, color, bg="white", size=11, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=bg, ec=color, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)

def arrow(x0, y0, x1, y1, color):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6))

# tasks
box(0.2, 1.7, 1.9, 1.2, "LinuxArena tasks\nmain task +\nhidden side task", INK, size=10)
# three pipelines
ys = [3.55, 2.35, 0.75]
labels = [("Execute (run 1)", REAL, "white"), ("Execute (run 2)", REAL, "white"), ("Simulate\nGPT-6-Astra writes\neach tool output,\nguided by a real\nhonest run", SIM, SIMBG)]
for y, (lab, col, bg) in zip(ys, labels):
    h = 0.75 if "Execute" in lab else 1.3
    yy = y - h / 2
    arrow(2.1, 2.3, 2.9, y, col)
    box(2.9, yy, 2.0, h, lab, col, bg, size=10 if "Execute" in lab else 9.5)
    arrow(4.9, y, 5.5, y, col)
    box(5.5, y - 0.33, 2.3, 0.66, "probes read Qwen3.5's\nactivations", MUTED, size=10.5, weight="normal")
    arrow(7.8, y, 8.35, y, col)
names = ["real\ncatch 1", "real\ncatch 2", "simulated\ncatch"]
for y, n, col in zip(ys, names, [REAL, REAL, SIM]):
    ax.add_patch(Circle((8.75, y), 0.38, fc="white", ec=col, lw=1.8))
    ax.text(8.75, y, n, ha="center", va="center", fontsize=9, color=col)
# deltas
ax.annotate("", xy=(9.35, ys[0]), xytext=(9.35, ys[1]), arrowprops=dict(arrowstyle="<->", color=REAL, lw=1.6))
ax.text(9.5, (ys[0] + ys[1]) / 2, "rerun gap", color=REAL, fontsize=12, va="center")
ax.annotate("", xy=(9.35, ys[2]), xytext=(9.35, ys[1] - 0.2), arrowprops=dict(arrowstyle="<->", color=SIM, lw=1.6))
ax.text(9.5, (ys[1] - 0.2 + ys[2]) / 2, "simulation gap", color=SIM, fontsize=12, va="center")
ax.text(9.5, (ys[1] - 0.2 + ys[2]) / 2 - 0.42, "average distance\nfrom each real run", color=SIM, fontsize=8.5, va="center")
ax.text(8.75, 4.3, "catch rate", ha="center", fontsize=10, color=MUTED)
ax.text(5.5, -0.25, "Each pipeline sets its own alarm line on its own honest runs (1 in 100 honest actions).", fontsize=9.5, color=MUTED, ha="center")
fig.tight_layout(); fig.savefig(os.path.join(S, "figs_simple", "fig0_method.png"), dpi=200); plt.close(fig)
print("ok")
