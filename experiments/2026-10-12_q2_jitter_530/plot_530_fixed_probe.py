"""POST HOC figure (10-13, Aswin via 0d: "yes do that, just put as new figuers? idk why you're refitting it properly"): the
530-task off-policy 2x2 at 1% false alarms with FIXED-probe intervals, beside the paired real - simulated difference.

Reads jitter_530_2026-10-12_114651.json's "fixed-score task bootstrap (2000 draws, seed 20261012)" block, variant
"fixed" (replicate 0's probe and its fixed 1% thresholds; tasks resampled only). The JSON is on S3 behind
results/s3_manifest_v1.json; its first copy in git (the manifest's "first copy in git") is read with `git show` and
checked against the manifest's sha256 before use. The old refit-aggregate figure is untouched.

Cells are named "<training world>-><test world>" as in the JSON. The caption's false-alarm range is the four cells'
"held-out FA of the frozen thresholds", read by key. The difference's point is real->real minus real->sim
(the two cells' "point" values, same block); its interval is the block's paired "real - sim (real-trained), 95%".

  uv run --no-project --python 3.12 --with matplotlib --with numpy python experiments/2026-10-12_q2_jitter_530/plot_530_fixed_probe.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NAME = "jitter_530_2026-10-12_114651.json"
BLOCK, VARIANT = "fixed-score task bootstrap (2000 draws, seed 20261012)", "fixed"
CELLS = ("real->real", "real->sim", "sim->real", "sim->sim")
DIFF = "real - sim (real-trained), 95%"
COLOR = {"real": "#2a78d6", "sim": "#eb6834"}  # by training world; blue/orange, no red or green
INK, INK2 = "#0b0b0b", "#52514e"
CAPTION = ("POST HOC. Bars: fixed probe and fixed 1% threshold, resampling tasks only. Retraining on each resample "
           "(recipe uncertainty) widens single cells to about 7–9 points; see the appendix. "
           "1% is the calibration target; achieved held-out false-alarm rates {fa_lo}–{fa_hi}.")
FA = "held-out FA of the frozen thresholds"  # per cell: "held-out FA"


def load() -> dict:
    man = json.loads((HERE / "results" / "s3_manifest_v1.json").read_text())["files"][NAME]
    first = man["what"].split("first copy in git at ")[1].split()[0].rstrip(";,.")
    rel = str((HERE / "results" / NAME).relative_to(REPO))
    data = subprocess.run(["git", "show", f"{first}:{rel}"], cwd=REPO, capture_output=True, check=True).stdout
    if hashlib.sha256(data).hexdigest() != man["sha256"]:
        raise SystemExit(f"{NAME} at {first} does not match the manifest's sha256")
    d = json.loads(data)
    if BLOCK not in d or VARIANT not in d[BLOCK]:
        raise SystemExit(f"missing {BLOCK!r} / {VARIANT!r}")
    b = d[BLOCK][VARIANT]
    for c in CELLS:
        if c not in b or "point" not in b[c] or "95%" not in b[c]:
            raise SystemExit(f"missing cell {c!r} or its point / 95%")
    if DIFF not in b or len(b[DIFF]) != 2:
        raise SystemExit(f"missing {DIFF!r}")
    fa = [d[FA][c]["held-out FA"] for c in CELLS]  # the four cells' achieved held-out false-alarm rates
    lo_s, hi_s = f"{100 * min(fa):.2f}%", f"{100 * max(fa):.2f}%"
    if (lo_s, hi_s) != ("0.85%", "1.23%"):  # 0d: the caption's range must be the file's, or stop
        raise SystemExit(f"achieved held-out false-alarm range is {lo_s}-{hi_s}, not 0.85%-1.23%")
    return {"first": first, "sha256": man["sha256"], "block": b, "fa": (lo_s, hi_s)}


def main():
    src = load(); b = src["block"]
    pts = {c: b[c]["point"] for c in CELLS}
    lo = {c: b[c]["95%"][0] for c in CELLS}; hi = {c: b[c]["95%"][1] for c in CELLS}
    dpt = pts["real->real"] - pts["real->sim"]; dlo, dhi = b[DIFF]

    plt.rcParams.update({"font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [3, 1.25]})
    x = range(len(CELLS))
    for i, c in enumerate(CELLS):
        train = c.split("->")[0]
        ax.bar(i, pts[c], width=0.62, color=COLOR[train], edgecolor="white", linewidth=2)
        ax.errorbar(i, pts[c], yerr=[[pts[c] - lo[c]], [hi[c] - pts[c]]], fmt="none", ecolor=INK, elinewidth=2, capsize=8)
        ax.text(i, hi[c] + 0.012, f"{pts[c]:.3f}\n[{lo[c]:.3f}, {hi[c]:.3f}]", ha="center", va="bottom", fontsize=13, color=INK)
    ax.set_xticks(list(x), [f"{c.replace('->', '→')}\ntrain: {c.split('->')[0]}\ntest: {c.split('->')[1]}" for c in CELLS])  # one item per line
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of impactful (state-changing) calls caught\nat the 1% false-alarm target (higher = better)")  # 0d's wording, wrapped
    ax.set_title("530-task off-policy 2×2, catch at 1% (fixed probe)", color=INK, fontsize=14)
    ax.grid(axis="y", color="#e6e6e3", linewidth=1); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[w]) for w in ("real", "sim")]
    ax.legend(handles, ["trained on real", "trained on sim"], loc="lower right", frameon=False)

    ax2.axhline(0, color=INK2, linewidth=1.5, zorder=1)
    ax2.errorbar(0, 100 * dpt, yerr=[[100 * (dpt - dlo)], [100 * (dhi - dpt)]], fmt="o", color=COLOR["real"], ecolor=INK,
                 elinewidth=2, capsize=8, markersize=10, zorder=3)
    ax2.text(0.12, 100 * dpt, f"{100 * dpt:+.2f}\n[{100 * dlo:+.2f}, {100 * dhi:+.2f}]", ha="left", va="center", fontsize=13, color=INK)
    ax2.set_xlim(-0.5, 1.1); ax2.set_xticks([0], ["real->real minus\nreal->sim"])
    ax2.set_ylim(-3, 3)
    ax2.set_ylabel("real − simulated catch, real-trained\n(percentage points)")
    ax2.set_title("paired difference", color=INK, fontsize=14)
    ax2.grid(axis="y", color="#e6e6e3", linewidth=1); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.text(0.5, 0.01, CAPTION.format(fa_lo=src["fa"][0], fa_hi=src["fa"][1]), ha="center", va="bottom", fontsize=11.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.canvas.draw()  # r_nc9 (#1311): no two tick labels may overlap at this figure size
    r = fig.canvas.get_renderer()
    for axis in (ax, ax2):
        boxes = [lab.get_window_extent(r) for lab in axis.get_xticklabels() if lab.get_text()]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i].overlaps(boxes[j]):
                    raise SystemExit(f"x tick labels {i} and {j} overlap")
    out = HERE / "results" / "fig_530_fixed_probe_2026-10-13_v2"  # append-only: new files beside the first render
    if out.with_suffix(".png").exists() or out.with_suffix(".pdf").exists():
        raise SystemExit(f"{out.name} exists; results are append-only")
    fig.savefig(out.with_suffix(".png"), dpi=150); fig.savefig(out.with_suffix(".pdf"))
    shown = {"caption": CAPTION.format(fa_lo=src["fa"][0], fa_hi=src["fa"][1]), "source": f"{NAME} @ {src['first']} (sha256 {src['sha256']})", "block": f"{BLOCK} / {VARIANT}",
             "cells (point, 95%)": {c: [pts[c], [lo[c], hi[c]]] for c in CELLS},
             "real->real - real->sim (point from the cells; 95% from the block's paired draws)": [dpt, [dlo, dhi]]}
    print(json.dumps(shown, indent=1)); print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
