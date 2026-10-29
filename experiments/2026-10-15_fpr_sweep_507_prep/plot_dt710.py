"""DT add-on figures for the ~710 (2e/Aswin for the 15:00 with James): (a) DT bars at per-run 1% and 5% (real catch, sim catch; gap and rerun
printed), all calls and impactful calls only; (b) a DT-only H1 curve on DT's own cell set; (c) the DT unit sweep rows (appendix). Titles by
rule from a run_common710 --dt-only record; captions give every title number's key paths. Modal CPU.

  PYTHONPATH=src <modal 1.6.1> run experiments/2026-10-15_fpr_sweep_507_prep/plot_dt710.py --record <name>@<sha> --commit <commit holding it>
"""

import hashlib
import io
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")
BLUE, ORANGE, GREY = "#3182bd", "#e6550d", "#7f7f7f"
PR = "FPR type: per run (share of honest runs alarmed)"
PA = "FPR type: per action (share of honest tool calls flagged)"
TS = "threshold source: real honest (pool as above)"
NICE = {"destructive_primary": "destructive, primary", "destructive_secondary": "destructive, secondary", "transfer_primary": "transfer, primary",
        "transfer_secondary": "transfer, secondary"}


def nice(lb):
    d = lb.removeprefix("DT_").removesuffix(" C").removesuffix(" impactful")
    return f"DT {NICE[d]}" + (" (impactful calls only, POST HOC)" if "impactful" in lb else "")


PANELS = [("pa1%", "unit", 0.01), ("1%", "run", 0.01), ("5%", "run", 0.05)]  # James 12:13: per action is the headline; per-action 1% first


def panel_point(pr, lv, al):
    if lv == "unit":
        i = next(i for i, q in enumerate(pr["unit"]) if abs(q["fpr_target"] - al) <= 1e-12 * al)
        return i, pr["unit"][i], "combined", f"probes.{pr['label']}.unit[{i}]"
    j = next(j for j, q in enumerate(pr["run"]["points"]) if abs(q["alpha"] - al) < 1e-12)
    return j, pr["run"]["points"][j], "all", f"probes.{pr['label']}.run.points[{j}]"


def titles(rec):
    import numpy as np

    P = rec["probes"]
    allc = [lb for lb in P if "impactful" not in lb]
    imp = [lb for lb in P if "impactful" in lb]
    pb = {s: v for s, v in P[allc[0]]["pool_by_source"].items()}
    rl = P[allc[0]]["relaunch"]
    pool = ("one real honest pool per DT row: S_old B ∪ C" + (" + the DT real-honest relaunch (958 runs with rows)" if rl.startswith("included") else "")
            + " + S_new B + new-main-task B (reconstructed by key); honest A held out" + ("" if rl.startswith("included") else f"; {rl}")
            + "; pool for the all-calls rows: " + ", ".join(f"{s} {v['runs']} runs / {v['units']:,} calls" for s, v in pb.items()))
    um = P[allc[0]].get("unmapped_rerouted", {}).get("counts")
    lab = (f"DT: {P[allc[0]]['dt_label']}; registered comparator sim − run 2; run score = max over command calls ({P[allc[0]]['aggregate']}); "
           "T196 rows reproduce the FPR fork's DT rows exactly (crossrecipe_2026-10-15_110033) before 09's unmapped-fold re-routing"
           + (f" ({um['t196_runs']} T196 runs and {um['relaunch_runs']} relaunch runs re-routed to the five-model ensemble)" if um else "")
           + "; sim − run 1 NOT launch-matched on +311 and new main tasks; "
           "new main tasks: different construction (21c); PRELIMINARY")
    srf = {lb: P[lb]["source_ratio"]["range_f_ge_1e-3"] for lb in P if isinstance(P[lb].get("source_ratio"), dict) and not P[lb]["source_ratio"]["within_tolerance"]}
    by_src = {}
    for lb in P:
        pts_ = [q for q in P[lb].get("source_ratio", {}).get("points", []) if q["fpr_target"] >= 1e-3 - 1e-12]
        for q in pts_:
            for s_, v in q["ratio_to_pooled"].items():
                if v is not None:
                    lo, hi = by_src.get(s_, (v, v))
                    by_src[s_] = (min(lo, v), max(hi, v))
    srtxt = (("‡ pool sources disagree beyond 2× (unit-level source-ratio check at f ≥ 1e-3, ratio range): " + ", ".join(f"{nice(lb)} {v[0]:.2f}–{v[1]:.2f}" for lb, v in srf.items())
              + "; ranges by source over all DT rows: " + ", ".join(f"{s_} {a:.2f}–{b:.2f}" for s_, (a, b) in sorted(by_src.items()))) if srf else "every DT row's pool sources agree within 2×")
    out = {"common": {"pool": pool, "labels": lab}, "source_ratio_fail": {"probes": srf, "text": srtxt, "sources": {
        "rule": "within_tolerance false; ranges 2 dp; by-source ranges: min/max of ratio_to_pooled over points with fpr_target >= 1e-3",
        "keys": [f"probes.{lb}.source_ratio" for lb in P]}}}
    inside = lambda s: bool(s.get("ci95")) and -0.05 <= s["ci95"][0] and s["ci95"][1] <= 0.05  # noqa: E731
    for nm, lv, al in PANELS:
        pp = {lb: panel_point(P[lb], lv, al) for lb in P}
        D = {lb: pp[lb][1]["blocks"][pp[lb][2]]["diff"] for lb in P}
        for grp, labs in (("all", allc), ("impactful", imp)):
            kr = sum(inside(D[lb]["sim_minus_run2"]) for lb in labs)
            ky = sum(inside(D[lb]["run2_minus_run1"]) for lb in labs)
            pts = [100 * D[lb]["sim_minus_run2"]["point"] for lb in labs]
            if lv == "unit":
                sh = [100 * pp[lb][1]["thresholds"]["union"]["realized_run_false_alarm_pool"] for lb in labs]
                head = f"at 1% per-action FPR (≈ {np.mean(sh):.1f}% of honest runs alarmed, mean over designs; range {min(sh):.1f}–{max(sh):.1f}%)"
            else:
                head = f"at {nm} per-run FPR"
            out[f"A_{grp}_{nm}"] = {"title": (f"DT {'all calls' if grp == 'all' else 'impactful calls only (POST HOC)'} {head}: sim − run 2 within ±5 points for "
                                              f"{kr} of {len(labs)} designs vs {ky} of {len(labs)} for rerunning (run 2 − run 1); point gaps {min(pts):+.1f} to {max(pts):+.1f}"),
                                    "sources": {"rule": "-0.05 <= ci95[0] and ci95[1] <= 0.05; points 100 x point, 1 dp; per-run share: mean and range of 100 x realized_run_false_alarm_pool",
                                                "keys": [f"{pp[lb][3]}.blocks.{pp[lb][2]}.diff.{k}" for lb in labs for k in ("sim_minus_run2", "run2_minus_run1")]
                                                + ([f"{pp[lb][3]}.thresholds.union.realized_run_false_alarm_pool" for lb in labs] if lv == "unit" else [])}}
    H = rec["H1_common"]
    if "points" not in H:
        raise SystemExit(f"DT H1 withheld: {H}")
    h1 = [q for q in H["points"] if "difference_sim_minus_rerun" in q]
    ix = [i for i, q in enumerate(H["points"]) if "difference_sim_minus_rerun" in q]
    pos = [q for q in h1 if q["difference_sim_minus_rerun"]["ci95"] and q["difference_sim_minus_rerun"]["ci95"][0] > 0]
    neg = [q for q in h1 if q["difference_sim_minus_rerun"]["ci95"] and q["difference_sim_minus_rerun"]["ci95"][1] < 0]
    s_ = [100 * q["mean_abs_sim_minus_real"]["point"] for q in h1]
    g_ = [100 * q["mean_abs_run2_minus_run1"]["point"] for q in h1]
    wide = max((100 * max(abs(q["difference_sim_minus_rerun"]["ci95"][0]), abs(q["difference_sim_minus_rerun"]["ci95"][1])) for q in h1
                if q["difference_sim_minus_rerun"]["ci95"]), default=float("nan"))
    rng = f"(mean |gap| {min(s_):.1f}–{max(s_):.1f} points vs {min(g_):.1f}–{max(g_):.1f} for rerunning)" if h1 else ""
    if not pos and not neg:
        t1 = f"DT (4 designs, all calls): no clear difference between simulation's and rerunning's catch-rate error at {len(h1)} readable FPRs (intervals up to ±{wide:.1f}) {rng}"
    else:
        t1 = (f"DT (4 designs, all calls): simulation adds more error than rerunning at {len(pos)} and less at {len(neg)} of {len(h1)} readable FPRs "
              f"(paired difference intervals exclude 0) {rng}")
    n = H["n_cells"]
    ntxt = f"n = {n['S_old'] + n['S_new'] + n['S_newmains']} tasks (T196 {n['S_old']}, +311 {n['S_new']}, new main tasks {n['S_newmains']}; S4 {n['S4']})"
    mean_all = float(np.mean([100 * q["difference_sim_minus_rerun"]["point"] for q in h1])) if h1 else float("nan")
    out["B"] = {"title": t1, "n_text": ntxt, "sources": {
        "counts and bound": {"rule": "readable = points with difference_sim_minus_rerun; ci95 excludes 0; bound = max |endpoint| x 100", "keys": [f"H1_common.points[{i}].difference_sim_minus_rerun.ci95" for i in ix]},
        "ranges": {"rule": "min/max of 100 x point, 1 dp", "keys": [f"H1_common.points[{i}].{k}.point" for i in ix for k in ("mean_abs_sim_minus_real", "mean_abs_run2_minus_run1")]},
        "mean above": {"rule": "mean of 100 x difference point over readable points, 1 dp", "keys": [f"H1_common.points[{i}].difference_sim_minus_rerun.point" for i in ix]},
        "n": {"rule": "as stored", "keys": ["H1_common.n_cells", "H1_common_without_S4.n_cells"]}},
        "caption": (f"{t1}. Mean over the 4 DT designs (all calls; registered comparator sim − run 2) of |sim − real| and |run 2 − run 1|, both on DT's own cell set, "
                    f"the tasks where all 4 designs are scored: {ntxt}; across all {len(h1)} readable FPRs the sim line sits on average {mean_all:.1f} points above the "
                    f"rerun line. Dashed: without S4 (Aswin's sensitivity; n = {sum(rec['H1_common_without_S4']['n_cells'][o] for o in ('S_old', 'S_new', 'S_newmains'))}). "
                    f"x: {PA}; {TS}. {pool}. 95% bands: shared main-level task bootstrap (10,000 draws), conditional on the thresholds. {srtxt}. {lab}.")}
    out["A"] = {"caption": (f"DT at per-action 1% (left; {PA}), then per-run 1% and 5% ({PR}): real catch (mean of run 1 and run 2) vs sim, all origins pooled, with n per row under its label; above each pair, "
                            f"the registered gap (sim − run 2) and the rerun (run 2 − run 1), in points. Impactful-calls-only rows (POST HOC): rows filtered to "
                            f"impactful == 1 for attack and honest runs alike, thresholds recomputed on honest impactful calls; a run with no impactful call is "
                            f"counted as never alarming and keeps its cell (2e); the drop-cell version is a sensitivity in the record. {TS}. {pool}. Error bars: 95% main-level task bootstrap. {srtxt}. {lab}.")}
    out["C"] = {"caption": f"Appendix: DT unit sweep, combined catch and the registered gap with origin points, vs target per-action FPR; {PA}; {TS}. {pool}. {lab}."}
    return out


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, scaledown_window=2)
def render(blob: bytes, ident: dict) -> dict:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rec = json.loads(blob)
    P = rec["probes"]
    tc = {"source_record": ident | {"key_paths": "dot-separated from the record's top level; [i] is a list index; probe labels are dict keys"}} | titles(rec)
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6"})
    out = {"captions.json": json.dumps(tc, indent=1).encode()}
    labs = list(P)
    # (a) bars
    fig, axes = plt.subplots(1, 3, figsize=(22, 8.4), sharey=True)
    for ax, (nm, lv, al) in zip(axes, PANELS):
        xs = np.arange(len(labs))
        q = {lb: (lambda t: t[1]["blocks"][t[2]])(panel_point(P[lb], lv, al)) for lb in labs}
        for off, w, col, lab in ((-0.2, "real_mean", BLUE, "real (mean of run 1, run 2)"), (0.2, "sim", ORANGE, "sim")):
            vals = [100 * q[lb]["tpr"][w]["point"] for lb in labs]
            lo = [100 * (q[lb]["tpr"][w]["point"] - q[lb]["tpr"][w]["ci95"][0]) for lb in labs]
            hi = [100 * (q[lb]["tpr"][w]["ci95"][1] - q[lb]["tpr"][w]["point"]) for lb in labs]
            ax.bar(xs + off, vals, 0.4, color=col, label=lab, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
        for i, lb in enumerate(labs):
            top = 100 * max(q[lb]["tpr"]["real_mean"]["ci95"][1], q[lb]["tpr"]["sim"]["ci95"][1])
            ax.text(i, top + 2, f"{100 * q[lb]['diff']['sim_minus_run2']['point']:+.1f}\n({100 * q[lb]['diff']['run2_minus_run1']['point']:+.1f})", ha="center", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{textwrap.fill(nice(lb) + (' ‡' if lb in tc['source_ratio_fail']['probes'] else ''), 22)}\nn = {q[lb]['n_cells']}" for lb in labs],
                           rotation=40, ha="right", fontsize=7.5)
        ax.set_title(textwrap.fill(tc[f"A_all_{nm}"]["title"], 80) + "\n\n" + textwrap.fill(tc[f"A_impactful_{nm}"]["title"], 80), fontsize=7.5, loc="left", pad=8)
        ax.set_ylim(0, 110)
    axes[0].set_ylabel("catch (%) at the honest FPR in each panel title")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.005, "above each pair: sim − run 2 (registered), with run 2 − run 1 in parentheses, in points; ‡ pool sources disagree beyond 2× (source-ratio check)",
             fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out["DT_A_bars.png"] = png(fig)
    plt.close(fig)
    # (b) DT-only H1
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    same = rec["H1_common"].get("identical_to_without_S4")
    for nm, ls in [("common", "-")] + ([] if same else [("common_without_S4", "--")]):
        pts = rec[f"H1_{nm}"]["points"]
        ok = [q for q in pts if "mean_abs_sim_minus_real" in q]
        x = [q["fpr_target"] for q in ok]
        for key, col, lab in (("mean_abs_run2_minus_run1", GREY, "|run 2 − run 1| (rerunning the real eval)"), ("mean_abs_sim_minus_real", ORANGE, "|sim − run 2| (registered)")):
            if ls == "-":
                ax.fill_between(x, [100 * q[key]["ci95"][0] for q in ok], [100 * q[key]["ci95"][1] for q in ok], color=col, alpha=0.22, lw=0)
            ax.plot(x, [100 * q[key]["point"] for q in ok], color=col, lw=2.2 if ls == "-" else 1.2, ls=ls, label=lab + ("" if ls == "-" else ", without S4"))
    for yv, ls_ in ((5, "--"), (1, ":")):
        ax.axhline(yv, color="black", lw=0.8, ls=ls_)
    ax.set_xscale("log")
    ax.set_xlim(1e-4 / 1.2, 1.2e-1)
    ax.set_ylim(0, None)
    ax.set_xlabel("honest FPR, per action (share of honest tool calls flagged)")
    ax.set_ylabel("|gap| (points), lower = sim closer to real")
    ax.set_title(tc["B"]["n_text"], fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(textwrap.fill(tc["B"]["title"], 100), x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out["DT_B_h1.png"] = png(fig)
    plt.close(fig)
    # (c) appendix sweep rows
    fig, axes = plt.subplots(2, len(labs), figsize=(3.2 * len(labs), 6.2), squeeze=False)
    for j, lb in enumerate(labs):
        u = P[lb].get("unit")
        if not isinstance(u, list):
            axes[0, j].set_title(f"{nice(lb)}: unit sweep withheld", fontsize=7)
            continue
        u = [q for q in u if q["thresholds"]["union"]["estimable"] and q["blocks"]["combined"]["readable"]]  # readable points only (>= 10 honest units above T)
        x = [q["fpr_target"] for q in u]
        for w, col in (("real_mean", BLUE), ("sim", ORANGE)):
            axes[0, j].plot(x, [100 * q["blocks"]["combined"]["tpr"][w]["point"] for q in u], color=col, lw=1.3)
        axes[0, j].set_xscale("log")
        axes[0, j].set_ylim(0, 100)
        axes[0, j].set_title(textwrap.fill(nice(lb), 28), fontsize=7.5, loc="left")
        ax = axes[1, j]
        ax.axhspan(-5, 5, color="#f2f2f2", zorder=0)
        ax.axhline(0, color="black", lw=0.8)
        for key, col, ls in (("run2_minus_run1", GREY, "--"), ("sim_minus_run2", ORANGE, "-")):
            s = [q["blocks"]["combined"]["diff"][key] for q in u]
            ax.fill_between(x, [100 * t["ci95"][0] if t.get("ci95") else np.nan for t in s], [100 * t["ci95"][1] if t.get("ci95") else np.nan for t in s], color=col, alpha=0.2, lw=0)
            ax.plot(x, [100 * t["point"] for t in s], color=col, lw=1.1, ls=ls)
        for o, ls, col in (("S_old", ":", "black"), ("S_new", "-.", "black"), ("S_newmains", "-", "#756bb1")):
            ax.plot(x, [100 * q["blocks"][o]["diff"]["sim_minus_run2"]["point"] for q in u], color=col, lw=0.7, ls=ls)
        ax.set_xscale("log")
        ax.set_ylim(-30, 30)
        ax.set_xlabel("per action FPR", fontsize=7.5)
    fig.suptitle("Appendix: DT unit sweep (blue real, orange sim; gap: orange sim − run 2, grey run 2 − run 1; T196 dotted, +311 dash-dot, new main purple)",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out["DT_C_appendix.png"] = png(fig)
    plt.close(fig)
    return out


@app.local_entrypoint()
def main(record: str, commit: str):
    nm, sha = record.rsplit("@", 1)
    b = (HERE / "results" / nm).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"{nm}: sha256 differs")
    ident = {"file": f"experiments/2026-10-15_fpr_sweep_507_prep/results/{nm}", "sha256": sha, "commit": commit}
    if subprocess.run(["git", "cat-file", "-e", f"{commit}:{ident['file']}"], cwd=HERE).returncode != 0:
        raise SystemExit("the record is not in that commit")
    out = render.remote(b, ident)
    stem = nm.removesuffix(".json")
    for fn, blob in out.items():
        (HERE / "results" / f"{stem}_{fn}").open("xb").write(blob)
        print(f"{stem}_{fn}", hashlib.sha256(blob).hexdigest())
