"""New figure (POST HOC): the 196-task on-policy 2x2 catch at 1% false alarms with fixed-probe intervals. $0, local.

Aswin, 2026-10-13, relayed by the orchestrator: "yes do that, just put as new figuers? idk why you're refitting it
properly". Left: catch per cell with method (b)'s interval from width_196 (same probe, same draw-0 threshold, 2,000 task
resamples). Right: real - simulated and first - second real run, with their (b) intervals. The old refit figures stay as
they are. Only the title, axis labels, values, intervals and one method line go on the image; status (POST HOC), the
source file and the explanations are printed as the caption, which goes beside the figure (Aswin's figure rule).
Every number is read by key from the width_196 result JSON (sha256 checked).

  uv run --no-project --python 3.14 --with matplotlib python -I plot_width_196_fixed_POSTHOC.py <width_196 json> <out.png>
"""

import hashlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHA = "c21444b8245aa574b3e49a45f805b9ffa078adb124c45de61ca037503a4468ff"
CELLS = [("real->real", "trained on real,\ntested on real"), ("real->sim", "trained on real,\ntested on simulated"),
         ("sim->real", "trained on simulated,\ntested on real"), ("sim->sim", "trained on simulated,\ntested on simulated"),
         ("real->run2", "trained on real,\ntested on second\nreal run")]
DIFFS = [("real_minus_sim", "real − simulated\n(trained on real)"), ("run1_minus_run2", "first real run −\nsecond real run\n(trained on real)")]
REAL, SIM, DIFF = "#3b6fb6", "#e08a2c", "#7a7a7a"  # blue / orange / grey: no red-green
FOOTER = "Fixed probe and fixed 1% threshold; error bars: 95% interval over 2,000 task resamples."
CAPTION = ("POST HOC. Bars: fixed probe and fixed 1% threshold, resampling tasks only (2,000 draws). Retraining the probe on each "
           "resample (recipe uncertainty) widens single cells to about 15–18 points; see the appendix. 1% is the calibration "
           "target; achieved held-out false-alarm rates for the probe trained on real runs: real {fa_real}%, simulated {fa_sim}% "
           "({fa_sim_wo}% without the 4 stuck runs). Error bars: 2.5th–97.5th percentiles of the 2,000 task resamples (main tasks "
           "resampled within each fold). Source: width_196_POSTHOC_2026-10-13_101005.json, method (b).")


def main(src: str, out: str):
    raw = open(src, "rb").read()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise SystemExit("width_196 result: sha256 differs")
    R = json.loads(raw)
    b = R["bootstrap"]["b"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [5, 2.3]})
    for i, (c, lab) in enumerate(CELLS):
        s = b[c]["catch"]
        p, (lo, hi) = 100 * s["point"], (100 * v for v in s["ci95"])
        ax.bar(i, p, color=REAL if c.startswith("real") else SIM, width=0.7)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="black", capsize=6, lw=1.4)
        ax.text(i, p / 2, f"{p:.1f}%", ha="center", va="center", color="white", fontsize=15, fontweight="bold")
        ax.text(i, hi + 1.5, f"[{lo:.1f}, {hi:.1f}]", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(len(CELLS)), [lab for _, lab in CELLS], fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("share of impactful (state-changing) calls caught\nat the 1% false-alarm target (higher = better)", fontsize=10)
    ax.set_title("Catch per cell (fixed probe, fixed threshold)", fontsize=12)
    for i, (d, lab) in enumerate(DIFFS):
        s = b[d]
        p, (lo, hi) = 100 * s["point"], (100 * v for v in s["ci95"])
        bx.bar(i, p, color=DIFF, width=0.6)
        bx.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="black", capsize=6, lw=1.4)
        bx.text(i, hi + 0.3, f"{p:+.1f}\n[{lo:+.1f}, {hi:+.1f}]", ha="center", va="bottom", fontsize=10)
    bx.axhline(0, color="black", lw=1)
    bx.set_xticks(range(len(DIFFS)), [lab for _, lab in DIFFS], fontsize=9.5)
    bx.set_xlim(-0.6, 1.6)
    bx.set_ylim(-6, 7)
    bx.set_ylabel("difference in catch (percentage points)", fontsize=10)
    bx.set_title("Paired differences", fontsize=12)
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    lo, hi = min(100 * b[c]["catch"]["point"] for c, _ in CELLS), max(100 * b[c]["catch"]["point"] for c, _ in CELLS)
    rs = b["real_minus_sim"]
    fig.suptitle(f"Catch is {lo:.0f}–{hi:.0f}% in every cell; real − simulated {100 * rs['point']:+.1f} points "
                 f"[{100 * rs['ci95'][0]:+.1f}, {100 * rs['ci95'][1]:+.1f}]", fontsize=13)
    fig.text(0.01, 0.01, FOOTER, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=160)
    S, sw = R["sensitivity_empty_replies"]["draw0_with_vs_without"], 100 * R["bootstrap"]["b"]["real->real"]["fa"]["point"]
    print("CAPTION (goes beside the figure, not on it):", CAPTION.format(fa_real=f"{sw:.2f}", fa_sim=f"{100 * S['real->sim']['false_alarm'][0]:.2f}",
                                                                          fa_sim_wo=f"{100 * S['real->sim']['false_alarm'][1]:.2f}"))
    print({c: [round(100 * b[c]["catch"]["point"], 2)] + [round(100 * v, 2) for v in b[c]["catch"]["ci95"]] for c, _ in CELLS}
          | {d: [round(100 * b[d]["point"], 2)] + [round(100 * v, 2) for v in b[d]["ci95"]] for d, _ in DIFFS})


if __name__ == "__main__":
    main(*sys.argv[1:3])
