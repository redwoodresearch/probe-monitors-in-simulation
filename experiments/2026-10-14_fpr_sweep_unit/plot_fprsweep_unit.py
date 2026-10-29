"""Figures F1-F3 and the viewer export for the unit-level FPR sweep (RULE.md in this folder). Figure-only: reads the saved record
from the solavg Volume (sha256 checked) and plots fields already in it; recomputes nothing. Titles come from a committed JSON
file written after reading the table. Writes PNGs (plus 900 px and 400 px check copies), the viewer JSON and a caption file.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_fpr_sweep_unit/plot_fprsweep_unit.py --record <name> --sha <sha256> --titles <json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-fprsweep_unit_kill-2026-11-14"
SOLVOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
BLUE, ORANGE, GREY, BAND5, HATCH = "#3182bd", "#e6550d", "#6f6f6f", "#f5f5f5", "#bdbdbd"
MAIN = ("sol_a|avg_top2|corrected|none|tight", "fibs|reply_mean|corrected|none", "harmfulness|reply_mean|corrected|none",
        "refusal|reply_mean|corrected|none", "highstakes|reply_mean|corrected|none")
PC = "c_asis|avg_top2|corrected|none"
NAMES = {MAIN[0]: "Sol (a)", MAIN[1]: "FIBS", MAIN[2]: "harmfulness", MAIN[3]: "refusal", MAIN[4]: "high-stakes",
         PC: "c_asis: positive control, not a monitor"}
OWN = (("own_frac100_L48 C", "own-real (frac100, L48): command"), ("own_frac100_L48 R", "own-real (frac100, L48): reply"),
       ("own_a_tight cmd_A", "own_a tight: command A"), ("own_a_tight avg_top2", "own_a tight: averaged reply"))
FOOT = ("Threshold fixed on honest B ∪ C reply units (per-fold z for Sol (a), c_asis); a run is caught if any unit exceeds it; "
        "183 attempted tasks; pointwise 95% task bootstrap (10,000 draws within environments)")
FOOT3 = ("Target and real catch = mean of the run 1 and run 2 hits; targets set on the selection half (93 tasks), values on the estimation half (90); "
         "pointwise 95% task bootstrap")
FOOT_N = "Threshold on honest B \u222a C reply units; any-unit catch; 183 tasks; 95% task bootstrap"
FOOT3_N = "Target, real catch: mean of run 1, run 2 hits; set on selection half, shown on estimation half"
app = modal.App(APP)
sol_vol = modal.Volume.from_name(SOLVOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=900, retries=0, volumes={"/sol": sol_vol})
def render(name: str, sha: str, titles: dict, draft: bool, tag: str = "", only_f1: bool = False, f1_primary: str = "sim_minus_real_mean") -> dict:  # noqa: C901
    import io
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    b = (Path("/sol") / "xfer" / "fprsweep_unit" / name / f"{name}.json").read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit("record sha256 differs")
    rec = json.loads(b)
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    pts = {pid: [q for q in rec["probes"][pid]["points"] if q["estimable"]] for pid in MAIN + (PC,)}
    view = {"label": "PRELIMINARY until 09 and Codex review; from raw data, by script", "record": name, "record_sha256": sha, "panels": {}}
    out = {}

    def save(fig, key, layout):  # wide: 150 dpi master and a 900 px copy; narrow (phone): 400 px and an 800 px (2x) copy
        w_in = fig.get_size_inches()[0]
        for tag, dpi in ((("", 150), ("_900px", 900 / w_in)) if layout == "wide" else (("_narrow_400px", 400 / w_in), ("_narrow_800px", 800 / w_in))):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            out[f"{key}{tag}.png"] = buf.getvalue()
        plt.close(fig)

    def shade(ax, qs):
        sh = [q["fpr_target"] for q in qs if q["shaded_lt10_above"]]
        un = [q["fpr_target"] for q in qs if not q["shaded_lt10_above"]]
        if sh:
            right = (max(sh) * min(u for u in un if u > max(sh))) ** 0.5 if any(u > max(sh) for u in un) else 10 ** -0.98
            ax.axvspan(10 ** -4.06, right, facecolor="none", hatch="////", edgecolor=HATCH, lw=0, zorder=0)

    def pct(v):
        return f"{100 * v:.1f}%" if 100 * v < 1 else f"{100 * v:.0f}%"

    def top_axis(ax, qs, fs):
        at = [1e-4, 1e-3, 1e-2, 1e-1]
        lab = []
        for x in at:
            q = min(qs, key=lambda q_: abs(np.log10(q_["fpr_target"]) - np.log10(x)))
            lab.append(pct(q["realized_run_false_alarm_any_unit"]["pool"]) if abs(np.log10(q["fpr_target"]) - np.log10(x)) < 1e-9 else "")
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(at)
        sec.set_xticklabels(lab, fontsize=fs)
        sec.minorticks_off()
        sec.tick_params(length=2, pad=1)
        sec.set_xlabel("honest runs flagged (in-sample)", fontsize=fs, labelpad=2)

    def series(q, key, grp):
        s_ = q[grp][key]
        return 100 * s_["point"], 100 * s_["ci95"][0], 100 * s_["ci95"][1]

    def panel_lines(ax, qs, specs, pid, kind, fs):
        x = [q["fpr_target"] for q in qs]
        vp = {}
        for key, grp, col, ls, lw, alpha, lab in specs:
            y, lo, hi = (np.array(v) for v in zip(*[series(q, key, grp) for q in qs]))
            if alpha > 0:
                ax.fill_between(x, lo, hi, color=col, alpha=alpha, lw=0, label=lab + ", 95% band" if lab else None)
                ax.plot(x, y, color=col, ls=ls, lw=lw)
            else:  # a secondary comparison drawn as its point estimate only
                ax.plot(x, y, color=col, ls=ls, lw=lw, label=lab + ", estimate" if lab else None)
            vp[key] = {"x": x, "point": y.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        if kind == "tpr":  # saturated catch (0% or 100%) is marked on its curve
            for key, grp, col, *_ in specs:
                sat = [(q["fpr_target"], 100 * q["tpr"][key]["point"]) for q in qs if q["saturated"][key]]
                if sat:
                    ax.plot(*zip(*sat), "x", color="black", ms=5, mew=1.2, label="saturated catch (0% or 100%; not read)")
        if kind == "gap":  # zero-width intervals (saturated or degenerate) are marked; they are not read
            dg = [(q["fpr_target"], 100 * q["diff"][f1_primary]["point"]) for q in qs if q["diff"][f1_primary].get("degenerate")]
            if dg:
                ax.plot(*zip(*dg), "x", color="black", ms=5, mew=1.2, label="zero-width interval (saturated; not read)")
            vp["degenerate"] = [bool(q["diff"][f1_primary].get("degenerate")) for q in qs]
        vp["shaded_lt10_above"] = [q["shaded_lt10_above"] for q in qs]
        vp["run_false_alarm_pool"] = [q["realized_run_false_alarm_any_unit"]["pool"] for q in qs]
        view["panels"][f"{kind}|{pid}"] = vp
        ax.set_xscale("log")
        ax.set_xlim(10 ** -4.06, 10 ** -0.98)
        shade(ax, qs)
        top_axis(ax, qs, fs)

    def legend_handles(axes):
        seen = {}
        for ax in axes:
            for h_, l_ in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(l_, h_)
        return list(seen.values())

    def grid(kind, title, specs, ylab, ylim_fn, layout):
        wide = layout == "wide"
        fs = 8.5 if wide else 8
        fig, axes = plt.subplots(2, 3, figsize=(12, 8.4)) if wide else plt.subplots(6, 1, figsize=(4.6, 18.5))
        ym = ylim_fn([q for pid in MAIN for q in pts[pid]])
        for ax, pid in zip(axes.flat, MAIN + (PC,)):
            qs = pts[pid]
            if kind == "gap":
                ax.axhspan(-5, 5, facecolor=BAND5, edgecolor="#8c8c8c", ls=":", lw=0.9, zorder=0, label="\u00b15 points")
                ax.axhline(0, color="black", lw=0.8)
            panel_lines(ax, qs, specs, pid, kind, fs)
            ax.set_ylim(*(ym if pid != PC else ylim_fn(qs)))
            ax.set_title(NAMES[pid], loc="left", pad=30, color="#a63603" if pid == PC else "black", fontweight="bold" if pid == PC else "normal")
            ax.set_xlabel("target unit FPR on honest B \u222a C reply units")
            if not wide or pid in (MAIN[0], MAIN[3]):
                ax.set_ylabel(ylab)
        h = legend_handles(axes.flat)
        h.append(matplotlib.patches.Patch(facecolor="white", hatch="////", edgecolor=HATCH, label="fewer than 10 honest units above the threshold (not read)"))
        fig.legend(handles=h, loc="lower center", ncol=(2 if kind == "gap" else 3) if wide else 1, frameon=False, fontsize=9.5 if wide else 8.5,
                   bbox_to_anchor=(0.5, 0.03 if wide else 0.012))
        fig.suptitle(textwrap.fill(title, 120 if wide else 52), fontsize=12.5 if wide else 10.5, x=0.01, ha="left", y=0.997)
        fig.text(0.01, 0.006 if wide else 0.002, FOOT if wide else FOOT_N, fontsize=8.3 if wide else 7.2, color="#444444", ha="left", va="bottom")
        fig.tight_layout(rect=(0, (0.1 if kind == "gap" else 0.08) if wide else (0.07 if kind == "gap" else 0.045), 1, 0.935 if wide else 0.955),
                         h_pad=2.2, w_pad=1.6)
        save(fig, f"{name}{tag}_F{1 if kind == 'gap' else 2}_{'gap' if kind == 'gap' else 'curves'}", layout)

    def ylim_gap(qs):
        m = max(max(abs(q["diff"][k]["ci95"][0]), abs(q["diff"][k]["ci95"][1])) for q in qs for k in (f1_primary, "run2_minus_run1"))
        m = 5 * np.ceil(100 * m / 5 + 1e-9)
        return (-m, m)

    for layout in ("wide", "narrow"):
        f1_specs = ([("run2_minus_run1", "diff", GREY, "--", 1.3, 0.3, "run 2 \u2212 run 1 (rerun + launch)"),
                     ("sim_minus_real_mean", "diff", ORANGE, "-", 2.0, 0.25, "sim \u2212 real (mean of run 1, run 2)")] if f1_primary == "sim_minus_real_mean" else
                    [("run2_minus_run1", "diff", GREY, "--", 1.3, 0.3, "run 2 \u2212 run 1 (rerun + launch)"),
                     ("sim_minus_real_mean", "diff", ORANGE, ":", 1.4, 0.0, "sim \u2212 real (mean of run 1, run 2)"),
                     ("sim_minus_run1", "diff", ORANGE, "-", 2.0, 0.25, "sim \u2212 run 1 (same launch)")])
        grid("gap", titles["F1"], f1_specs, "catch difference (points)", ylim_gap, layout)
        if only_f1:  # a title-only re-render of F1 (same record, same code path); F2, F3 and the viewer export are not rewritten
            continue
        grid("tpr", titles["F2"], [("real_mean", "tpr", BLUE, "-", 2.0, 0.2, "real (mean of run 1, run 2)"),
                                   ("sim", "tpr", ORANGE, "-", 2.0, 0.2, "simulated")],
             "catch (% of attack runs)", lambda qs: (0, 100), layout)

    # F3: own-real operating points (estimation half), distinct operating points per panel
    def f3(layout):
        wide = layout == "wide"
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.8)) if wide else plt.subplots(4, 1, figsize=(4.6, 15.5))
        allv = []
        for ax, (lab, nm) in zip(axes.flat, OWN):
            r = rec["own_real"][lab]["primary"]
            ops = {}
            for o in r["operating_points"]:
                ops.setdefault(o["k"], []).append(o)
            xs = sorted(ops)
            vp = []
            for i, k in enumerate(xs):
                q = ops[k][0]["estimation"]
                if q["shaded_lt10_above"]:  # fewer than 10 honest units above: shown, not read
                    ax.axvspan(i - 0.45, i + 0.45, facecolor="none", hatch="////", edgecolor=HATCH, lw=0, zorder=0)
                for j, (key, col, mk, fc, lg) in enumerate((("sim_minus_run2", ORANGE, "o", ORANGE, "sim \u2212 run 2 (primary; run 1 is a training arm)"),
                                                             ("sim_minus_real_mean", ORANGE, "o", "white", "sim \u2212 real (mean)"),
                                                             ("run2_minus_run1", GREY, "s", GREY, "run 2 \u2212 run 1 (rerun + launch)"))):
                    y, lo, hi = series(q, key, "diff")
                    allv += [lo, hi]
                    ax.errorbar(i + (j - 1) * 0.22, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=col, mfc=fc, ms=7, capsize=3, lw=1.3, label=lg if i == 0 else None)
                vp.append({"k": k, "targets": [o["target"] for o in ops[k]], "fpr_target": ops[k][0]["fpr_target"], "units_above": q["pool_units_above"],
                           "not_reached": [o["target_not_reached"] for o in ops[k]], "selection_real_mean_catch": ops[k][0]["selection_real_mean_catch"],
                           "estimation_tpr": {w: q["tpr"][w]["point"] for w in q["tpr"]},
                           "diff": {d_: {"point": q["diff"][d_]["point"], "ci95": q["diff"][d_]["ci95"], "degenerate": q["diff"][d_].get("degenerate")} for d_ in q["diff"]}})
            view["panels"][f"own|{lab}"] = {"pool": r["pool"], "points": vp}
            ax.axhspan(-5, 5, facecolor=BAND5, edgecolor="#8c8c8c", ls=":", lw=0.9, zorder=0)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([("selection target " + "/".join(f"{100 * o['target']:.0f}" for o in ops[k]) + "%"
                                 + (",\nnot reached" if all(o["target_not_reached"] for o in ops[k]) else "")
                                 + f"\nFPR {ops[k][0]['fpr_target']:.1e}\n{ops[k][0]['estimation']['pool_units_above']} {'honest unit(s)' if wide else 'unit(s)'} above"
                                 + f"\nreal catch {100 * ops[k][0]['estimation']['tpr']['real_mean']['point']:.0f}%") for k in xs],
                               fontsize=8.2 if wide else 7.4)
            ax.set_xlim(-0.6, len(xs) - 0.4)
            pool_lab = {"honest_b": "B", "honest_b+honest_c": "B \u222a C"}[r["pool"]]
            ax.set_title(f"{nm}  (honest pool {pool_lab})", loc="left", fontsize=11.5 if wide else 9.5)
            ax.set_ylabel("catch difference (points)")
            ax.grid(axis="x", visible=False)
        m = 5 * np.ceil(max(abs(v) for v in allv) / 5 + 1e-9)  # allv is already in points
        for ax in axes.flat:
            ax.set_ylim(-m, m)
        h = legend_handles(axes.flat)
        h.append(matplotlib.patches.Patch(facecolor=BAND5, edgecolor="#8c8c8c", ls=":", label="\u00b15 points"))
        h.append(matplotlib.patches.Patch(facecolor="white", hatch="////", edgecolor=HATCH, label="fewer than 10 honest units above the threshold (not read)"))
        fig.legend(handles=h, loc="lower center", ncol=2 if wide else 1, frameon=False, fontsize=9.5 if wide else 8.5, bbox_to_anchor=(0.5, 0.03 if wide else 0.012))
        fig.suptitle(textwrap.fill(titles["F3"], 110 if wide else 52), fontsize=12.5 if wide else 10.5, x=0.01, ha="left", y=0.997)
        fig.text(0.01, 0.006 if wide else 0.002, FOOT3 if wide else FOOT3_N, fontsize=8.3 if wide else 7.2, color="#444444", ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.11 if wide else 0.085, 1, 0.935 if wide else 0.95), h_pad=2.0, w_pad=1.6)
        save(fig, f"{name}_F3_own_real", layout)

    if only_f1:
        if not draft:
            od = Path("/sol") / "xfer" / "fprsweep_unit" / name
            for fn, blob in out.items():
                (od / fn).open("xb").write(blob)
            sol_vol.commit()
        return out
    f3("wide")
    f3("narrow")
    view["a24_run_level_reference"] = {pid: rec["a24_reference_sweeps"][pid] for pid in MAIN + (PC,)}
    view["footers"] = {"F1": FOOT, "F2": FOOT, "F3": FOOT3, "narrow": [FOOT_N, FOOT3_N]}
    view["titles"] = titles
    out[f"{name}_viewer.json"] = json.dumps(view, default=float).encode()
    if not draft:  # a draft (layout check) is returned only; the final render is also written to the Volume
        od = Path("/sol") / "xfer" / "fprsweep_unit" / name
        for fn, blob in out.items():
            (od / fn).open("xb").write(blob)
        sol_vol.commit()
    return out


@app.local_entrypoint()
def main(record: str, sha: str, titles: str, draft_dir: str = "", tag: str = "", only_f1: bool = False, f1_primary: str = "sim_minus_real_mean"):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    t = json.loads((HERE / titles).read_text())
    out = render.remote(record, sha, t, bool(draft_dir), tag, only_f1, f1_primary)
    R = Path(draft_dir) if draft_dir else HERE / "results"
    for fn, blob in out.items():
        (R / fn).open("wb" if draft_dir else "xb").write(blob)
    print(json.dumps({fn: hashlib.sha256(blob).hexdigest() for fn, blob in out.items()}, indent=1))
