"""Figure for the command-readout sim-vs-real detector (POST HOC, PRELIMINARY): run-level AUROC with task-bootstrap 95% intervals, per
fit x layer, command readout beside reply readout, the within-task shuffle null's 95th percentile as a tick. Rendered on Modal (CPU).
Every plotted number and every number in the title is read by key from the two records; captions.json lists each key path.

  <modal> run experiments/2026-10-15_cmd_sim_detector/plot_detector.py --honest results/honest_detector_<stamp>.json --attack results/attack_heldout_<stamp>.json
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-cmdsimdet_kill-2026-11-14")
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("matplotlib==3.10.6", "numpy==2.3.3")
A2_NULL = {"commit": "db69919f6ad12ec5d2861713a02039f7b83f8be2", "path": "experiments/2026-10-14_cmd_readout_simh_control/results/controls_a2_2026-10-15_004048.json",
           "sha256": "8182a82008d2069c1bdf391d5e1118e972a34d7dd4fef1c126d8fd5106da7caf"}
LAYERS = (48, 63)
# (group label, record, problem key or held-out contrast, explanation)
GROUPS = [("honest: sim vs real B ∪ C", "H", "simh_vs_BuC"),
          ("honest: sim vs real B,\nsame main tasks", "H", "simh_vs_Bmatched"),
          ("control: real honest\nC vs B", "H", "c_vs_b"),
          ("attack: sim vs real run 1", "A", "c_simatk_vs_run1"),
          ("attack: sim vs held-out\nrun 2 (sim-vs-run1 model)", "A", "hold:sim_vs_heldout_run2"),
          ("control: held-out run 2\nvs run 1 (same model)", "A", "hold:heldout_run2_vs_run1"),
          ("control: real run 2\nvs run 1 (own fit)", "A", "d_run2_vs_run1")]


def values(H: dict, A: dict, a2: dict) -> dict:
    """{readout: {group: {L: (auroc, lo, hi, null_p95 or None, key path)}}}, read by key."""
    out = {}
    for ro in ("command", "reply"):
        out[ro] = {}
        for lab, src, key in GROUPS:
            out[ro][key] = {}
            for L in LAYERS:
                cell = f"{ro}|L{L}"
                if src == "H":
                    r = H["cells"][cell]["problems"][key]
                    path = f"honest.cells['{cell}'].problems['{key}']"
                    v = (r["observed"]["auroc"], *r["observed"]["auroc_boot"]["ci95"], r.get("null", {}).get("p95"),
                         f"{path}.observed.auroc / .observed.auroc_boot.ci95 / .null.p95")
                elif key.startswith("hold:"):
                    k = key.split(":", 1)[1]
                    r = A["cells"][cell]["heldout_run2"]["contrasts"][k]
                    v = (r["auroc"], *r["boot"]["ci95"], None, f"attack.cells['{cell}'].heldout_run2.contrasts['{k}'].auroc / .boot.ci95 (no null)")
                else:
                    r = A["cells"][cell]["problems"][key]
                    nul = a2["cells"][cell]["problems"][key]["null"]["run_level_auroc_summary"]["p95"]
                    v = (r["run_level"]["auroc"], *r["run_auroc_boot"]["ci95"], nul,
                         f"attack.cells['{cell}'].problems['{key}'].run_level.auroc / .run_auroc_boot.ci95; null p95 = A2.cells['{cell}'].problems['{key}'].null.run_level_auroc_summary.p95")
                out[ro][key][L] = v
    return out


@app.function(image=img, cpu=2, memory=4096, timeout=300)
def render(vals: dict, title: str, caption: str) -> bytes:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    col = {48: "#4C78A8", 63: "#F58518"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, ro in zip(axes, ("command", "reply")):
        x = np.arange(len(GROUPS))
        for i, L in enumerate(LAYERS):
            for g, (lab, _src, key) in enumerate(GROUPS):
                a, lo, hi, nul, _ = vals[ro][key][str(L)] if str(L) in vals[ro][key] else vals[ro][key][L]
                xx = g + (i - 0.5) * 0.36
                ax.bar(xx, a - 0.5, bottom=0.5, width=0.34, color=col[L], label=f"layer {L}" if g == 0 else None)
                ax.errorbar(xx, a, yerr=[[a - lo], [hi - a]], color="black", capsize=3, lw=1)
                ax.text(xx, hi + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
                if nul is not None:
                    ax.plot([xx - 0.17, xx + 0.17], [nul, nul], color="#666666", ls="--", lw=1.2, label="shuffle null 95th pct." if (g == 0 and i == 0) else None)
        ax.axhline(0.5, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in GROUPS], rotation=30, ha="right", fontsize=8.5)
        ax.set_title(f"{ro} readout ({'content-only end of call' if ro == 'command' else 'averaged reply'})", fontsize=11)
        ax.set_ylim(0.35, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("run-level AUROC (higher = runs more separable)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(title, fontsize=11.5, x=0.01, ha="left")
    fig.text(0.01, 0.005, caption, fontsize=8, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


@app.local_entrypoint()
def main(honest: str, attack: str):
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    hb, ab = (HERE / honest).read_bytes(), (HERE / attack).read_bytes()
    H, A = json.loads(hb), json.loads(ab)
    a2b = subprocess.run(["git", "show", f"{A2_NULL['commit']}:{A2_NULL['path']}"], capture_output=True, check=True, cwd=HERE).stdout
    if sha(a2b) != A2_NULL["sha256"]:
        raise SystemExit("A2 record sha256 differs")
    if not A["reproduction_of_a2"]["pass"]:
        raise SystemExit("the attack refit does not reproduce A2: no figure")
    v = values(H, A, json.loads(a2b))
    c, r = v["command"], v["reply"]
    title = (f"Command readout (content-only end of call): sim vs real run-level AUROC {c['simh_vs_BuC'][63][0]:.2f} for honest runs and "
             f"{c['c_simatk_vs_run1'][63][0]:.2f} for attacks (layer 63),\nvs {c['c_vs_b'][63][0]:.2f} and {c['d_run2_vs_run1'][63][0]:.2f} for the command-readout real-vs-real controls; "
             f"the averaged reply readout has higher observed AUROC: {r['simh_vs_BuC'][48][0]:.2f} honest, {r['c_simatk_vs_run1'][48][0]:.2f} attack (layer 48)")
    nH = H["cells"]["command|L63"]["problems"]
    caption = (f"POST HOC, PRELIMINARY. Qwen3.5-27B's own T196 runs. L2 logistic regression (C = 0.01) on one row per readout unit, 5-fold CV grouped by main task (#36 folds); "
               f"run score = mean row logit. Honest: sim honest = A25's {nH['simh_vs_BuC']['n_runs']['class1']} runs vs real honest B ∪ C "
               f"({nH['simh_vs_BuC']['n_runs']['class0']} runs); matched = B on the same {nH['simh_vs_Bmatched']['n_mains']} mains. Attack: sim 191 vs run 1 193; run 2 (192) "
               f"also scored held-out by the sim-vs-run1 fold models. Bars: 95% task-bootstrap intervals (2,000 draws, scores fixed). Dashed: 95th percentile of a "
               f"within-task label-shuffle null (honest: 50 refits; attack: A2's 100 refits of these exact fits; none for held-out contrasts). "
               f"No paired test of reply vs command. At the reply readout the real-vs-real honest control (C vs B, layer 48) is "
               f"{r['c_vs_b'][48][0]:.2f}, above its null 95th percentile {r['c_vs_b'][48][3]:.2f}, so reply separation is not all simulation-specific.")
    png = render.remote({ro: {k: {str(L): list(x) for L, x in d.items()} for k, d in v[ro].items()} for ro in v}, title, caption)
    stamp = json.loads(ab)["stamp"]
    out = HERE / "results" / f"detector_auroc_{stamp}.png"
    out.open("xb").write(png)
    caps = {"figure": out.name, "figure_sha256": sha(png), "title": title, "caption": caption,
            "sources": {"honest": {"path": honest, "sha256": sha(hb)}, "attack": {"path": attack, "sha256": sha(ab)}, "a2_null": A2_NULL},
            "values": {ro: {k: {str(L): {"auroc": x[0], "ci95": [x[1], x[2]], "null_p95": x[3], "key": x[4]} for L, x in d.items()} for k, d in v[ro].items()} for ro in v}}
    (HERE / "results" / f"detector_auroc_{stamp}_captions.json").open("x").write(json.dumps(caps, indent=1))
    print(out.name)
    print(title)
