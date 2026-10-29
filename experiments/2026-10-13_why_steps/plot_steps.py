"""Steps 1 and 2 figures (0d item 7b): one bar chart each, Qwen3-32B, values read by key from the metrics files; the
drawing runs on Modal CPU and only the PNGs come back.
- Step 1: the mixed recipe's command-readout (call_end content-only) overstatement at 1%, fix-full vs placeholder
  (fix-full-matched positions), L48 and L63; raw simulated / real catch printed on each bar.
- Step 2: S2 (the score shift of the 30 simulator-rewritten honest runs over the honest real-vs-simulated gap), call_end
  L48 / L63 (primary view) and output_end L63 (secondary); caught-at-1% printed.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-13_why_steps/plot_steps.py --step1 <why_metrics_steps_2026-10-13_175358.json> --step2 <why_step2_2026-10-13_182312.json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-whysteps_plot_kill-2026-11-13"
BLUE, ORANGE = "#0072B2", "#E69F00"  # Okabe-Ito, colour-blind safe

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("matplotlib==3.10.6", "numpy==2.3.3")


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0)
def draw(s1: list[dict], s2: list[dict]) -> dict:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def png(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()

    fig, ax = plt.subplots(figsize=(10, 6.2))
    for i, layer in enumerate(("48", "63")):
        for j, (world, color, name) in enumerate((("ff", BLUE, "replies kept (fix-full)"), ("ph", ORANGE, "replies replaced by a placeholder"))):
            p = next(q for q in s1 if q["layer"] == layer and q["world"] == world)
            x = i + (j - 0.5) * 0.36
            ax.bar(x, p["y"], 0.34, color=color, label=name if i == 0 else None)
            ax.errorbar(x, p["y"], yerr=[[p["y"] - p["ci"][0]], [p["ci"][1] - p["y"]]], color="0.2", capsize=5, lw=1.4)
            ax.text(x, max(p["ci"][1], 0) + 0.006, f"{p['y']:+.3f}", ha="center", fontsize=13, fontweight="bold")
            ax.text(x, -0.031, f"sim {p['sim']:.2f}\nreal {p['real']:.2f}", ha="center", fontsize=10, color="0.25")
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks([0, 1], ["Layer 48", "Layer 63"], fontsize=13)
    ax.set_ylim(-0.045, 0.16)
    ax.set_ylabel("Overstatement: simulated − real catch at 1%\n(closer to 0 = more faithful grade)", fontsize=12)
    ax.set_title("Without reply content, the mixed probe's command readout\nno longer measurably overstates", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    fig.text(0.5, 0.012, "Qwen3-32B, mixed recipe (simulated attacks vs real honest), command readout (last content token of each call).\n"
             "Bars: 95% intervals, 2,000 main-task resamples.", ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    one = png(fig)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    for i, p in enumerate(s2):
        color = BLUE if p["primary"] else "#56B4E9"
        ax.bar(i, p["y"], 0.55, color=color)
        ax.errorbar(i, p["y"], yerr=[[p["y"] - p["ci"][0]], [p["ci"][1] - p["y"]]], color="0.2", capsize=6, lw=1.4)
        ax.text(i + 0.31, p["y"], f"{p['y']:.2f}", va="center", fontsize=13, fontweight="bold")
        ax.text(i, -0.17, f"caught at 1%:\nrewritten {100 * p['caught_rew']:.0f}%\noriginal {100 * p['caught_orig']:.0f}%\n{p['pairs']} pairs",
                ha="center", va="top", fontsize=10, color="0.25")
    ax.axhline(1, color="0.4", lw=1.2, ls="--")
    ax.text(-0.45, 1.04, "1 = the whole honest real-vs-simulated gap", ha="left", fontsize=10, color="0.3")
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks(range(len(s2)), [p["label"] for p in s2], fontsize=12)
    ax.set_ylim(-0.75, 2.1)
    ax.set_ylabel("Score shift toward 'simulated'\n(share of the honest gap; 0 = none)", fontsize=12)
    ax.set_title("Simulator-rewritten replies move the probe's scores\nby about the whole honest gap", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012, "Qwen3-32B, fix-full mixed probe, 30 real honest runs with the September simulator's replies (real commands kept).\n"
             "Bars: 95% intervals, 2,000 resamples of the pairs.", ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return {"step1": one, "step2": png(fig)}


@app.local_entrypoint()
def main(step1: str, step2: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    m1, m2 = json.loads(Path(step1).read_bytes()), json.loads(Path(step2).read_bytes())
    s1 = []
    for L in ("48", "63"):  # reads by key only
        for world, key in (("ff", f"L{L}__ff_call_end__c"), ("ph", f"L{L}__ph_call_end_mff__c")):
            y = m1["rows"][key]["fpr_0.01"]
            s1.append({"layer": L, "world": world, "key": key, "y": y["overstatement_fake_minus_e2"]["value"],
                       "ci": y["overstatement_fake_minus_e2"]["ci95"], "sim": y["fake_attack_catch"]["value"],
                       "real": y["real_attack_catch_e2"]["value"]})
    s2 = []
    for key, label, primary in (("ff_call_end_L48_c", "command readout\nlayer 48", True), ("ff_call_end_L63_c", "command readout\nlayer 63", True),
                                ("ff_output_end_L63_c", "reply readout\nlayer 63 (secondary)", False)):
        v = m2["readouts"][key]["changed"]
        s2.append({"key": key, "label": label, "primary": primary, "y": v["S2"]["value"], "ci": v["S2"]["ci95"], "pairs": v["pairs"],
                   "caught_rew": v["caught_1pct_full_run_thresholds"]["rewritten"], "caught_orig": v["caught_1pct_full_run_thresholds"]["original"]})
    out = draw.remote(s1, s2)
    side = {"git": git("rev-parse", "HEAD"), "step1_input": Path(step1).name, "step1_sha256": hashlib.sha256(Path(step1).read_bytes()).hexdigest(),
            "step2_input": Path(step2).name, "step2_sha256": hashlib.sha256(Path(step2).read_bytes()).hexdigest(), "step1": s1, "step2": s2}
    for k in ("step1", "step2"):
        p = HERE / "results" / f"fig_{k}_qwen3-32b_2026-10-13_v3.png"
        with p.open("xb") as fh:
            fh.write(out[k])
        side[f"{k}_png_sha256"] = hashlib.sha256(out[k]).hexdigest()
        print(f"wrote {p.name} {side[f'{k}_png_sha256'][:12]}", flush=True)
    with (HERE / "results" / "fig_steps_qwen3-32b_2026-10-13_v3.json").open("x") as fh:
        fh.write(json.dumps(side, indent=1))
