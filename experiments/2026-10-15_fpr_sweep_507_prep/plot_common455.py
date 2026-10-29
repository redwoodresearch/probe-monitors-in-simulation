"""Figures A and B and the readings tables for the 455 common comparison, from a run455 record (Modal CPU; no laptop compute).

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_fpr_sweep_507_prep/plot_common455.py --record <name> --sha <sha256> --titles config/titles455.json
"""

import hashlib
import io
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")
MAIN_R = ["Sol (a) R", "FIBS R", "harmfulness R", "refusal R", "high-stakes R", "own_a R", "own_frac100_L48 R", "c_asis R"]
FOOT = ("455 attempted cells (S_old 183 + S_new 272); one honest pool: B ∪ C (S_old) + B (S_new), + relaunch for Sol (a)/c_asis; "
        "pointwise 95% main-level bootstrap, conditional on the threshold")


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, scaledown_window=2)
def render(blob: bytes, titles: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rec = json.loads(blob)
    P = rec["probes"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6"})
    ORANGE, BLUE, GREY = "#e6550d", "#3182bd", "#6f6f6f"
    out = {}
    reg = {k: v["registered"] for k, v in P.items()}

    def band(ax, x, s, col, lab, alpha=0.25, ls="-"):
        y = np.array([100 * q["point"] for q in s])
        lo = np.array([100 * q["ci95"][0] if q.get("ci95") else np.nan for q in s])
        hi = np.array([100 * q["ci95"][1] if q.get("ci95") else np.nan for q in s])
        ax.fill_between(x, lo, hi, color=col, alpha=alpha, lw=0)
        ax.plot(x, y, color=col, ls=ls, lw=1.6, label=lab)

    # ---- Figure A: unit-level sweep, pooled block, registered comparator; S_old / S_new dotted
    labs = [lb for lb in MAIN_R if lb in P and isinstance(P[lb].get("unit"), list)]
    fig, axes = plt.subplots(2, len(labs), figsize=(3.1 * len(labs), 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    for j, lb in enumerate(labs):
        pts = [q for q in P[lb]["unit"] if q["thresholds"]["union"]["estimable"]]
        x = [q["fpr_target"] for q in pts]
        comb = [q["blocks"]["combined"] for q in pts]
        ax = axes[0, j]
        band(ax, x, [b["tpr"]["real_mean"] for b in comb], BLUE, "real (mean of run 1, run 2)")
        band(ax, x, [b["tpr"]["sim"] for b in comb], ORANGE, "sim")
        ax.set_title(lb.replace(" R", ", reply") + (" (stress test)" if lb.startswith("c_asis") else ""), fontsize=10, loc="left")
        ax.set_ylim(0, 100)
        if j == 0:
            ax.set_ylabel("catch (%)")
        ax = axes[1, j]
        key = reg[lb]
        ax.axhspan(-5, 5, color="#f2f2f2", zorder=0)
        for yv in (-1, 1):
            ax.axhline(yv, color="#bdbdbd", lw=0.7, ls=":")
        ax.axhline(0, color="black", lw=0.8)
        band(ax, x, [b["diff"]["run2_minus_run1"] for b in comb], GREY, "run 2 − run 1", alpha=0.2, ls="--")
        band(ax, x, [b["diff"][key] for b in comb], ORANGE, {"sim_minus_run1": "sim − run 1", "sim_minus_run2": "sim − run 2"}[key] + " (registered, pooled)")
        for o, ls in (("S_old", ":"), ("S_new", "-.")):
            ax.plot(x, [100 * q["blocks"][o]["diff"][key]["point"] for q in pts], color="black", ls=ls, lw=1.0, label=f"{o} (point)")
        sh = [q["fpr_target"] for q in pts if q["thresholds"]["union"]["shaded_lt10_above"]]
        if sh:
            ax.axvspan(min(x) / 1.15, max(sh) * 1.15, facecolor="none", hatch="////", edgecolor="#bdbdbd", lw=0)
        ax.set_xscale("log")
        ax.set_xlim(1e-4 / 1.2, 1.2e-1)
        ax.set_ylim(-30, 30)
        if j == 0:
            ax.set_ylabel("catch difference (points)")
        ax.set_xlabel("target unit FPR on honest units")
        sec = axes[0, j].secondary_xaxis("top")
        at = [1e-4, 1e-3, 1e-2, 1e-1]
        lab_ = []
        for xx in at:
            q = min(pts, key=lambda q_: abs(np.log10(q_["fpr_target"]) - np.log10(xx)))
            lab_.append(f"{100 * q['thresholds']['union']['realized_run_false_alarm_pool']:.1f}%" if abs(np.log10(q["fpr_target"]) - np.log10(xx)) < 1e-9 else "")
        sec.set_xticks(at)
        sec.set_xticklabels(lab_, fontsize=7.5)
        sec.set_xlabel("honest runs flagged", fontsize=7.5)
    h, lb_ = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lb_:
                h.append(hh), lb_.append(ll)
    fig.legend(h, lb_, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(titles["A"], x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, FOOT, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    out["common455_figA_unit.png"] = buf.getvalue()
    plt.close(fig)
    # ---- Figure B: all probes at run level 1/2/5/10%, registered comparator, pooled with 95% intervals; S_old and S_new points
    labs = [lb for lb in P if isinstance(P[lb].get("run"), dict) and "points" in P[lb]["run"]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 0.32 * len(labs) + 2.2), sharey=True)
    for i, a in enumerate((0.01, 0.02, 0.05, 0.1)):
        ax = axes[i]
        ax.axvspan(-5, 5, color="#f2f2f2", zorder=0)
        ax.axvline(0, color="black", lw=0.8)
        for r, lb in enumerate(labs):
            q = next(q for q in P[lb]["run"]["points"] if abs(q["alpha"] - a) < 1e-12)
            d = q["blocks"]["all"]["diff"][reg[lb]]
            yy = len(labs) - r
            if d.get("ci95"):
                ax.plot([100 * d["ci95"][0], 100 * d["ci95"][1]], [yy, yy], color=ORANGE, lw=1.6)
            ax.plot(100 * d["point"], yy, "o", color=ORANGE, ms=4)
            for o, mk in (("S_old", "^"), ("S_new", "s")):
                ax.plot(100 * q["blocks"][o]["diff"][reg[lb]]["point"], yy, mk, color="black", ms=3, mfc="none")
            g = q["blocks"]["all"]["diff"]["run2_minus_run1"]
            ax.plot(100 * g["point"], yy, "|", color=GREY, ms=8)
        ax.set_title(f"run-level FPR {int(100 * a)}%", fontsize=10)
        ax.set_xlabel("registered gap (points)")
        ax.set_xlim(-40, 40)
    axes[0].set_yticks(range(len(labs), 0, -1))
    axes[0].set_yticklabels([f"{lb}  [{'sim−run2' if reg[lb] == 'sim_minus_run2' else 'sim−run1'}]" for lb in labs], fontsize=7.5)
    fig.suptitle(titles["B"], x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, FOOT + "; ▲ S_old, ■ S_new points; | run 2 − run 1 (pooled)", fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    out["common455_figB_run.png"] = buf.getvalue()
    plt.close(fig)
    return out


@app.local_entrypoint()
def main(record: str, sha: str, titles: str):
    b = (HERE / "results" / record).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit("record sha256 differs")
    out = render.remote(b, json.loads((HERE / titles).read_text()))
    stem = record.removesuffix(".json")
    for fn, blob in out.items():
        (HERE / "results" / f"{stem}_{fn}").open("xb").write(blob)
        print(f"{stem}_{fn}", hashlib.sha256(blob).hexdigest())
