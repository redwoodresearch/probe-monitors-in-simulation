"""Headline figures H1 (diverging lines), H2 (bars), H3 (Sol (a) by origin x strategy) for the 455 common comparison, titled by
script from the records (0d via 2e, ~10:3x). Modal CPU; writes PNGs and a captions JSON next to the headlines record.

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_fpr_sweep_507_prep/plot_headlines455.py --run455 <name>@<sha> --headlines <name>@<sha>
"""

import hashlib
import io
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")
PLAIN = {"Sol (a) R": "Sol-trained probe (a)", "FIBS R": "FIBS (public)", "harmfulness R": "harmfulness direction", "refusal R": "refusal direction",
         "high-stakes R": "high-stakes probe", "own_a R": "own-real probe (own_a)", "own_frac10_L48 R": "own-real, 10% data", "own_frac25_L48 R": "own-real, 25% data",
         "own_frac100_L48 R": "own-real, 100% data", "c_asis R": "stress test (sim-trained c_asis)"}
BLUE, ORANGE, GREY = "#3182bd", "#e6550d", "#7f7f7f"


def titles_and_captions(r455, hd):
    """Every title is chosen by rule from the numbers (no hand-written claim)."""
    import numpy as np

    out = {}
    h1 = [q for q in hd["H1"]["points"] if "difference_sim_minus_rerun" in q]
    pos = [q for q in h1 if q["difference_sim_minus_rerun"]["ci95"] and q["difference_sim_minus_rerun"]["ci95"][0] > 0]
    neg = [q for q in h1 if q["difference_sim_minus_rerun"]["ci95"] and q["difference_sim_minus_rerun"]["ci95"][1] < 0]
    s = [100 * q["mean_abs_sim_minus_real"]["point"] for q in h1]
    g = [100 * q["mean_abs_run2_minus_run1"]["point"] for q in h1]
    rng = f"(mean |gap| {min(s):.1f}–{max(s):.1f} points vs {min(g):.1f}–{max(g):.1f} for rerunning)"
    inside = all(q["difference_sim_minus_rerun"]["ci95"] and -0.02 <= q["difference_sim_minus_rerun"]["ci95"][0] and q["difference_sim_minus_rerun"]["ci95"][1] <= 0.02 for q in h1)
    wide = max((100 * max(abs(q["difference_sim_minus_rerun"]["ci95"][0]), abs(q["difference_sim_minus_rerun"]["ci95"][1])) for q in h1
                if q["difference_sim_minus_rerun"]["ci95"]), default=float("nan"))
    if not pos and not neg and inside:
        t1 = f"Simulation adds about as much catch-rate error as rerunning the real eval at all {len(h1)} readable FPRs (difference within ±2 points) {rng}"
    elif not pos and not neg:
        t1 = f"No clear difference between simulation's and rerunning's catch-rate error at {len(h1)} readable FPRs (difference intervals up to ±{wide:.1f} points) {rng}"
    elif pos and not neg:
        t1 = f"Simulation adds more catch-rate error than rerunning the real eval at {len(pos)} of {len(h1)} readable FPRs {rng}"
    elif neg and not pos:
        t1 = f"Simulation adds less catch-rate error than rerunning the real eval at {len(neg)} of {len(h1)} readable FPRs {rng}"
    else:
        t1 = f"Simulation adds more error than rerunning at {len(pos)} and less at {len(neg)} of {len(h1)} readable FPRs {rng}"
    nP = len(hd["H1"]["probes"])
    hi_ = [q for q in h1 if q["fpr_target"] >= 4e-3 - 1e-12]
    above = float(np.mean([100 * q["difference_sim_minus_rerun"]["point"] for q in hi_])) if hi_ else float("nan")
    out["H1"] = {"title": t1, "caption": (f"{t1}. From about 4e-3 upward the sim line sits on average {above:.1f} points above the rerun line, while "
                                          f"{len(pos) + len(neg)} of {len(h1)} paired difference intervals exclude 0. Mean over {nP} probes (stress test excluded; reply readout; "
                                          f"registered comparator) of |sim − real| catch, and of |run 2 − run 1|, "
                                          f"on 455 attempted tasks (183 + 272); x = target per-action FPR on honest units (one pool: B ∪ C + B, + relaunch for Sol (a)); "
                                          f"95% bands from the shared main-level task bootstrap (10,000 draws), conditional on the thresholds; 'about as much' requires every "
                                          f"paired difference interval inside ±2 points; |gap| is biased upward by "
                                          f"noise for both lines, which is why the grey rerun line is the comparison. Hatched: fewer than all {nP} probes readable "
                                          f"(< 10 honest units above the threshold).")}
    labs = [lb for lb in PLAIN if lb in r455["probes"]]
    for a, nm in ((0.05, "5%"), (0.01, "1%")):
        D_ = {lb: next(q for q in r455["probes"][lb]["run"]["points"] if abs(q["alpha"] - a) < 1e-12)["blocks"]["all"]["diff"] for lb in labs}
        mon = [lb for lb in labs if not lb.startswith("c_asis")]
        regk = {lb: r455["probes"][lb]["registered"] for lb in labs}
        inside = lambda s: bool(s.get("ci95")) and -0.05 <= s["ci95"][0] and s["ci95"][1] <= 0.05  # noqa: E731
        resolved = lambda s: bool(s.get("ci95")) and (s["ci95"][0] > 0 or s["ci95"][1] < 0)  # noqa: E731
        kr = sum(inside(D_[lb][regk[lb]]) for lb in mon)
        km = sum(inside(D_[lb]["sim_minus_real_mean"]) for lb in mon)
        ky = sum(inside(D_[lb]["run2_minus_run1"]) for lb in mon)
        rr_, ry_ = sum(resolved(D_[lb][regk[lb]]) for lb in mon), sum(resolved(D_[lb]["run2_minus_run1"]) for lb in mon)
        pts_ = [100 * D_[lb][regk[lb]]["point"] for lb in mon]
        out[f"H2_{nm}"] = {"title": (f"At {nm} honest-run alarm, sim − real (registered comparator) is within ±5 points for {kr} of {len(mon)} probes, vs {ky} of "
                                     f"{len(mon)} for rerunning (run 2 − run 1); point gaps {min(pts_):+.1f} to {max(pts_):+.1f}; stress test "
                                     f"{100 * D_['c_asis R'][regk['c_asis R']]['point']:+.1f}"),
                           "counts": {"registered_within_5": kr, "real_mean_within_5": km, "rerun_within_5": ky, "registered_resolved": rr_, "rerun_resolved": ry_,
                                      "n_probes": len(mon)},
                           "note": f"resolved (interval excludes 0): registered {rr_} of {len(mon)}, rerun {ry_} of {len(mon)}; sim − real (mean) within ±5: {km} of {len(mon)}"}
    out["H2"] = {"caption": (out["H2_5%"]["title"] + "; " + out["H2_5%"]["note"] + ". Bars: real catch (mean of run 1 and run 2) vs sim; 455 attempted tasks pooled; "
                             "9 probes (stress test excluded) plus the stress test; honest-run alarm = share of the pooled honest calibration runs "
                             "(B ∪ C + B, + relaunch for Sol (a)) above the run-level threshold; error bars = 95% main-level task bootstrap.")}
    h3 = hd["H3_sol_by_origin_strategy"]
    q = next(q for q in h3["points"] if abs(q["alpha"] - 0.05) < 1e-12)
    cells = [b for b in q["blocks"] if b != "all"]
    re_ = [100 * q["blocks"][b]["tpr"]["real_mean"]["point"] for b in cells]
    si = [100 * q["blocks"][b]["tpr"]["sim"]["point"] for b in cells]
    which = "sim catch varies less across task sets than real catch" if np.ptp(si) < np.ptp(re_) else "sim catch varies at least as much across task sets as real catch"
    t3 = f"Sol (a) reply at 5%: real catch ranges {min(re_):.0f}–{max(re_):.0f}% across task sets, sim {min(si):.0f}–{max(si):.0f}%; {which}"
    out["H3"] = {"title": t3, "caption": t3 + (f". Task sets: S_old (T196) and S_new (+311) cohorts, each split by whether the task has a strategy "
                                               f"(approach_chars > 0); n per bar in the labels; 5% = share of pooled honest calibration runs alarmed; error bars = "
                                               f"95% main-level task bootstrap.")}
    return out


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, scaledown_window=2)
def render(b455: bytes, bhd: bytes, figA: bytes) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    r455, hd = json.loads(b455), json.loads(bhd)
    tc = titles_and_captions(r455, hd)
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6"})
    out = {"captions.json": json.dumps(tc, indent=1).encode()}
    # ---- H1
    pts = hd["H1"]["points"]
    ok = [q for q in pts if "mean_abs_sim_minus_real" in q]
    x = [q["fpr_target"] for q in ok]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for key, col, lab in (("mean_abs_run2_minus_run1", GREY, "|run 2 − run 1| (rerunning the real eval)"), ("mean_abs_sim_minus_real", ORANGE, "|sim − real| (registered comparator)")):
        y = np.array([100 * q[key]["point"] for q in ok])
        lo = np.array([100 * q[key]["ci95"][0] for q in ok])
        hi = np.array([100 * q[key]["ci95"][1] for q in ok])
        ax.fill_between(x, lo, hi, color=col, alpha=0.22, lw=0)
        ax.plot(x, y, color=col, lw=2.2, label=lab)
    for yv, ls in ((5, "--"), (1, ":")):
        ax.axhline(yv, color="black", lw=0.8, ls=ls)
    bad = [q["fpr_target"] for q in pts if "mean_abs_sim_minus_real" not in q]
    if bad:
        ax.axvspan(1e-4 / 1.2, max(bad) * 1.1, facecolor="none", hatch="////", edgecolor="#bdbdbd", lw=0, label="too few probes readable")
    ax.set_xscale("log")
    ax.set_xlim(1e-4 / 1.2, 1.2e-1)
    ax.set_ylim(0, None)
    ax.set_xlabel("honest per-action (unit) FPR")
    ax.set_ylabel("|gap| (points), lower = sim closer to real")
    sec = ax.secondary_xaxis("top")
    at = [1e-4, 1e-3, 1e-2, 1e-1]
    labs_ = []
    for xx in at:
        q = min(pts, key=lambda q_: abs(np.log10(q_["fpr_target"]) - np.log10(xx)))
        labs_.append(f"{100 * q['mean_run_fa_pool']:.1f}%" if abs(np.log10(q["fpr_target"]) - np.log10(xx)) < 1e-9 else "")
    sec.set_xticks(at)
    sec.set_xticklabels(labs_, fontsize=8.5)
    sec.set_xlabel("share of honest runs alarmed (mean over probes)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    import textwrap

    fig.suptitle(textwrap.fill(tc["H1"]["title"], 95), x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    out["H1_gap_lines.png"] = buf.getvalue()
    plt.close(fig)
    # ---- H2
    labs = [lb for lb in PLAIN if lb in r455["probes"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, (a, nm) in zip(axes, ((0.05, "5%"), (0.01, "1%"))):
        xs = np.arange(len(labs))
        for off, w, col, lab in ((-0.2, "real_mean", BLUE, "real (mean of run 1, run 2)"), (0.2, "sim", ORANGE, "sim")):
            vals, los, his = [], [], []
            for lb in labs:
                s = next(q for q in r455["probes"][lb]["run"]["points"] if abs(q["alpha"] - a) < 1e-12)["blocks"]["all"]["tpr"][w]
                vals.append(100 * s["point"])
                los.append(100 * (s["point"] - s["ci95"][0]))
                his.append(100 * (s["ci95"][1] - s["point"]))
            bars = ax.bar(xs + off, vals, 0.4, color=col, label=lab, yerr=[los, his], capsize=2, error_kw={"lw": 0.8})
            for bb, v in zip(bars, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="white", rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([PLAIN[lb] for lb in labs], rotation=40, ha="right", fontsize=8.5)
        ax.set_title(textwrap.fill(tc[f"H2_{nm}"]["title"], 70), fontsize=9.5, loc="left")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("catch (%) at the honest-run alarm rate")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    out["H2_catch_bars.png"] = buf.getvalue()
    plt.close(fig)
    # ---- H3
    q = next(q for q in hd["H3_sol_by_origin_strategy"]["points"] if abs(q["alpha"] - 0.05) < 1e-12)
    cells = ["S_old|strategy", "S_old|no_strategy", "S_new|strategy", "S_new|no_strategy"]
    nice = {"S_old|strategy": "T196, with strategy", "S_old|no_strategy": "T196, no strategy", "S_new|strategy": "+311, with strategy", "S_new|no_strategy": "+311, no strategy"}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = np.arange(len(cells))
    for off, w, col, lab in ((-0.2, "real_mean", BLUE, "real (mean of run 1, run 2)"), (0.2, "sim", ORANGE, "sim")):
        vals = [100 * q["blocks"][c]["tpr"][w]["point"] for c in cells]
        lo = [100 * (q["blocks"][c]["tpr"][w]["point"] - q["blocks"][c]["tpr"][w]["ci95"][0]) for c in cells]
        hi = [100 * (q["blocks"][c]["tpr"][w]["ci95"][1] - q["blocks"][c]["tpr"][w]["point"]) for c in cells]
        bars = ax.bar(xs + off, vals, 0.4, color=col, label=lab, yerr=[lo, hi], capsize=3)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{nice[c]}\n(n = {q['blocks'][c]['n_cells']} tasks)" for c in cells], fontsize=9)
    ax.set_ylabel("catch (%) at 5% honest-run alarm")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(textwrap.fill(tc["H3"]["title"], 85), x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    out["H3_sol_origin_strategy.png"] = buf.getvalue()
    plt.close(fig)
    return out


@app.local_entrypoint()
def main(run455: str, headlines: str):
    blobs = []
    for spec in (run455, headlines):
        nm, sha = spec.rsplit("@", 1)
        b = (HERE / "results" / nm).read_bytes()
        if hashlib.sha256(b).hexdigest() != sha:
            raise SystemExit(f"{nm}: sha256 differs")
        blobs.append(b)
    out = render.remote(blobs[0], blobs[1], b"")
    stem = headlines.rsplit("@", 1)[0].removesuffix(".json")
    for fn, blob in out.items():
        (HERE / "results" / f"{stem}_{fn}").open("xb").write(blob)
        print(f"{stem}_{fn}", hashlib.sha256(blob).hexdigest())
