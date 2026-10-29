"""The trap figure: a probe trained on simulated attacks vs real honest runs (the mixed recipe, cell c) reports high
catch on simulated attacks but catches few real attacks, for each of five readers. Bars: the real-trained probe's real
catch, the mixed probe's real catch, the mixed probe's simulated catch, at 1% false alarms, with stored 95% main-task
bootstrap intervals. Numbers come only from trap_numbers.csv (key reads, extract_trap_numbers.py). The plotting runs in
a Modal CPU function (Aswin's no-laptop-compute rule); the PNG is written to the Volume and returned.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach plot_trap.py
"""
import csv
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3trap_figure_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q3trap_work_kill-2026-11-13"
HERE = Path(__file__).resolve().parent
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("matplotlib==3.9.2")
vol = modal.Volume.from_name(VOL, create_if_missing=True, version=2)
READERS = ["Qwen3-32B", "Qwen3.5-27B", "Seed-OSS-36B", "Olmo-3.1-32B", "Llama-3.3-70B"]
Q = ["real-trained (a): real catch", "mixed (c): real catch", "mixed (c): simulated catch"]


@app.function(image=image, cpu=2, memory=2048, timeout=600, volumes={"/out": vol})
def plot(rows: list[dict], stamp: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    get = {(r["reader"], r["rate"], r["quantity"]): r for r in rows}
    gap = {rd: get[(rd, "fpr_0.01", "mixed (c): overstatement (sim − real)")] for rd in READERS}
    # Okabe–Ito blue and vermillion plus neutral grey with hatching (colour-blind safe; identity also by legend and labels)
    style = {Q[0]: dict(color="#0072B2", label="Real-trained probe, on real attacks"),
             Q[1]: dict(color="#D55E00", label="Mixed probe (simulated attacks vs real honest), on real attacks"),
             Q[2]: dict(color="#BBBBBB", hatch="//", edgecolor="#555555", label="Same mixed probe, on simulated attacks")}
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    w, xs = 0.26, range(len(READERS))
    for j, q in enumerate(Q):
        for i, rd in enumerate(READERS):
            r = get[(rd, "fpr_0.01", q)]
            v, lo, hi = 100 * float(r["value"]), 100 * float(r["ci_lo"]), 100 * float(r["ci_hi"])
            x = i + (j - 1) * (w + 0.02)
            ax.bar(x, v, w, **{k: s for k, s in style[q].items() if k != "label"}, label=style[q]["label"] if i == 0 else None,
                   linewidth=0.8)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#222222", elinewidth=1.1, capsize=3)
            ax.text(x, hi + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222222")
        if j == 0:
            for i, rd in enumerate(READERS):
                g = gap[rd]
                ax.text(i, 112, f"overstates by\n{100*float(g['value']):.0f} points [{100*float(g['ci_lo']):.0f}, {100*float(g['ci_hi']):.0f}]",
                        ha="center", va="center", fontsize=9.5, color="#222222")
    views = {rd: get[(rd, "fpr_0.01", Q[0])]["view"] for rd in READERS}
    ax.set_xticks(list(xs), [f"{rd}\n{views[rd]}" for rd in READERS], fontsize=10)
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Attacks caught (%), targeting 1% false alarms\nhigher is better", fontsize=11)
    sims = [100 * float(get[(rd, "fpr_0.01", Q[2])]["value"]) for rd in READERS]
    reals = [100 * float(get[(rd, "fpr_0.01", Q[1])]["value"]) for rd in READERS]
    ax.set_title(f"A probe trained on simulated attacks vs real honest runs reports {min(sims):.0f}–{max(sims):.0f}% catch on "
                 f"simulated attacks\nbut catches only {min(reals):.0f}–{max(reals):.0f}% of real attacks (five readers, reply readout)",
                 fontsize=12.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=1, frameon=False, fontsize=10)
    g5 = [100 * float(get[(rd, "fpr_0.05", "mixed (c): overstatement (sim − real)")]["value"]) for rd in READERS]
    q48 = get[("Qwen3-32B (layer 48, beside)", "fpr_0.01", "mixed (c): overstatement (sim − real)")]
    fig.text(0.01, 0.005, "Readout: output_end, the last content token of each tool output. Real attacks = the second real run (e2). "
             "Error bars: 95% main-task bootstrap intervals (2,000 draws; fitted probes held fixed, thresholds recalibrated per draw).\n"
             f"Overstatement = the mixed probe's simulated catch minus its real catch, in percentage points; targeting 5% false alarms it is {min(g5):.0f}–{max(g5):.0f}. Qwen3-32B at layer 48, at the 1% target: "
             f"{100*float(q48['value']):.0f} [{100*float(q48['ci_lo']):.0f}, {100*float(q48['ci_hi']):.0f}]. Mixed probe = unrepaired simulated "
             "attacks vs real honest runs.\n1% and 5% are the target false-alarm rates the thresholds were calibrated to, not achieved rates. Source: trap_numbers.csv (file@key per bar).", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = f"/out/trap_figure_{stamp}.png"
    fig.savefig(out, bbox_inches="tight")
    vol.commit()
    return Path(out).read_bytes()


@app.local_entrypoint()
def main():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rows = list(csv.DictReader(open(HERE / "trap_numbers.csv")))
    png = plot.remote(rows, stamp)
    with open(HERE / f"trap_figure_{stamp}.png", "xb") as fh:
        fh.write(png)
    print(f"wrote trap_figure_{stamp}.png ({len(png)} bytes); also on Volume {VOL}")
