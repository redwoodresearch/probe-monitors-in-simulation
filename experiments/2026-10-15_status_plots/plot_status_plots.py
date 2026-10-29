"""Two template-style status-page plots (Aswin via the viewer, ~10:5x PT 10-15), every value read by key from committed records.
PRELIMINARY. Modal CPU only; the laptop runs `git show` and writes the returned files.

P1 "error bars then vs now": U (approximate 95% upper limit on X_err) for sim − real mean and for run 2 − run 1, with-strategy,
   then = X_err v6 (T196 estimation half) vs now = scale-up X_err Revision 6s (old and new task sets, and both pooled).
P2 "Sol (a) by origin under the registered calibration": real (mean of runs 1, 2) vs sim catch per task set x strategy, reply,
   per-run 5%, from the Revision 6s record, with the record's sim − real-mean gap and its 95% interval printed.

  modal run experiments/2026-10-15_status_plots/plot_status_plots.py \
      --then <path>@<commit> --now <path>@<commit>
"""

import io
import json
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-statusplots_render_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")
BLUE, ORANGE, GREY = "#3182bd", "#e6550d", "#7f7f7f"
ROWS = [  # (label, v6 row key, 6s row key)
    ("reply, 5%", "R_reply_meanpooled|mechanical|alpha=0.05", "R|0.05"),
    ("reply, 1%", "R_reply_meanpooled|mechanical|alpha=0.01", "R|0.01"),
    ("command, 5%", "C_anycommand_max|mechanical|alpha=0.05", "C|0.05"),
]
GROUPS = [  # (label, record, key path under the row; n key path)
    ("then: v6\n(old tasks, est. half)", "then", "primary_eligible_estimation_half/{c}/with_strategy"),
    ("now: old tasks\n(est. half)", "now", "primary/{c}/S_old|with_strategy"),
    ("now: new tasks", "now", "primary/{c}/S_new|with_strategy"),
    ("now: both pooled\n('strata differ')", "now", "primary/{c}/combined|with_strategy"),
]
COMPS = [("sim_minus_mean_real", ORANGE, "sim − real (mean of runs 1, 2)"), ("reexecution_run2_minus_run1", GREY, "run 2 − run 1 (rerunning the real eval)")]
STRATA = [("S_old|with_strategy", "old tasks,\nwith strategy"), ("S_old|without_strategy", "old tasks,\nno strategy"),
          ("S_new|with_strategy", "new tasks,\nwith strategy"), ("S_new|without_strategy", "new tasks,\nno strategy")]


def get(d, path):
    for k in path.split("/"):
        d = d[k]
    return d


def extract(then, now):
    """Every plotted number with its key path. Values stay in the record's units (U in catch points; catch in %; gaps in points)."""
    out = {"P1": [], "P2": {}}
    for lab, k6, ks in ROWS:
        rec = {"row": lab, "groups": []}
        for glab, which, tmpl in GROUPS:
            base = ["result", "results", k6 if which == "then" else ks]
            g = {"group": glab.replace("\n", " "), "record": which}
            for c, _, _ in COMPS:
                p = "/".join(base + [tmpl.format(c=c), "pooled" if which == "now" else "", "U_approx_upper_95"]).replace("//", "/")
                src = then if which == "then" else now
                g[c] = {"U": get(src, p), "key": p}
                if which == "now":
                    g[c]["n_mains"] = get(src, "/".join(base + [tmpl.format(c=c), "mains"]))
                    g[c]["n_mains_key"] = "/".join(base + [tmpl.format(c=c), "mains"])
            rec["groups"].append(g)
        p6 = "/".join(["result", "results", k6, "eligibility_append13_4", "eligible"])
        pn = "/".join(["result", "results", ks, "primary", "eligible_v6_frozen"])
        rec["eligible_then"], rec["eligible_now"] = get(then, p6), get(now, pn)
        rec["eligible_keys"] = [p6, pn]
        rec["pools_then"] = {pr: get(then, f"result/results/{k6}/pools/{pr}") for pr in rec["eligible_then"]}
        rec["pools_then_key"] = f"result/results/{k6}/pools/<probe>"
        rec["combined_label_now"] = get(now, f"result/results/{ks}/primary/combined_label")
        rec["schemes_now"] = get(now, f"result/results/{ks}/primary/schemes")
        rec["schemes_key"] = f"result/results/{ks}/primary/schemes"
        out["P1"].append(rec)
    n_then = get(then, "result/results/R_reply_meanpooled|mechanical|alpha=0.05/registered_strata_pooling_test_tost5/primary_eligible_estimation_half/n_with")
    out["P1_then_n_with"] = {"value": n_then, "key": "result/results/R_reply_meanpooled|mechanical|alpha=0.05/registered_strata_pooling_test_tost5/primary_eligible_estimation_half/n_with"}
    out["P1_draws"] = {"then": get(then, "result/draws"), "now": {"n": get(now, "result/design/draws"), "seed": get(now, "result/design/seed")}}
    cal = get(now, "result/calibration_records/primary/sol_a|R|0.05|primary|BuC|B")
    out["P2_calibration"] = {"key": "result/calibration_records/primary/sol_a|R|0.05|primary|BuC|B", "pool_old": cal["pool_old"], "pool_new": cal["pool_new"],
                             "n_pool_old": cal["n_pool_old"], "n_pool_new": cal["n_pool_new"], "primary": cal["primary"]}
    for st, _ in STRATA:
        cp = f"result/results/R|0.05/primary/catch_percent/{st}"
        gp = f"result/results/R|0.05/primary/sim_minus_mean_real/{st}/pooled/per_probe/sol_a"
        out["P2"][st] = {"real_mean": get(now, cp + "/catch_mean_real/sol_a"), "run1": get(now, cp + "/catch_run1/sol_a"), "run2": get(now, cp + "/catch_run2/sol_a"),
                         "sim": get(now, cp + "/catch_sim/sol_a"), "catch_key": cp + "/<catch_mean_real|catch_run1|catch_run2|catch_sim>/sol_a",
                         "gap": get(now, gp + "/delta"), "gap_ci95": get(now, gp + "/ci95"), "gap_key": gp + "/<delta|ci95>",
                         "n_mains": get(now, f"result/results/R|0.05/primary/sim_minus_mean_real/{st}/mains"),
                         "n_key": f"result/results/R|0.05/primary/sim_minus_mean_real/{st}/mains"}
    return out


def titles(ex):
    r5 = ex["P1"][0]["groups"]
    t, o, n, p = (g["sim_minus_mean_real"]["U"] for g in r5)
    rr = [g["reexecution_run2_minus_run1"]["U"] for g in r5]
    rerun = f"Rerunning the real eval: {rr[0]:.1f} → {rr[3]:.1f} pooled"
    if p < t - 1.0 and max(o, n) >= t - 1.0:
        t1 = (f"The bound on sim − real catch error falls from {t:.1f} to {p:.1f} points (reply, per-run 5%) only when old and new tasks are pooled; "
              f"within each task set it stays near {o:.1f} (old) and {n:.1f} (new). {rerun}")
    elif p < t - 1.0:
        t1 = (f"The bound on sim − real catch error fell from {t:.1f} to {p:.1f} points (reply, per-run 5%) with 196 → 507 tasks, and within each "
              f"task set too ({o:.1f} old, {n:.1f} new). {rerun}")
    else:
        t1 = (f"The bound on sim − real catch error moved from {t:.1f} to {p:.1f} points (reply, per-run 5%) with 196 → 507 tasks "
              f"({o:.1f} old, {n:.1f} new). {rerun}")

    def phrase(st, lab):
        s = ex["P2"][st]
        lo, hi = s["gap_ci95"]
        word = "more" if s["gap"] > 0 else ("less" if s["gap"] < 0 else "the same")
        res = "interval excludes 0" if (lo > 0 or hi < 0) else "interval includes 0"
        return f"{word} on {lab} ({s['gap']:+.1f}, {res})"

    t2 = (f"Sol (a), reply, per-run 5%, registered calibration (each task set's own real-honest pool): sim catches "
          f"{phrase('S_old|with_strategy', 'old tasks with strategy')}, {phrase('S_new|with_strategy', 'new tasks with strategy')} and "
          f"{phrase('S_new|without_strategy', 'new tasks without strategy')} than real")
    return t1, t2


def captions(ex, t1, t2):
    rows = ex["P1"]
    el = "; ".join(f"{r['row']}: {', '.join(r['eligible_now'])}" for r in rows)
    same = all(r["eligible_then"] == r["eligible_now"] for r in rows)
    odd = [f"{r['row']}: {pr} ({sc})" for r in rows for pr, sc in r["schemes_now"].items() if sc != "stratum_specific"]
    pex = [f"{r['row']} {pr} on honest {pl}" for r in rows for pr, pl in r["pools_then"].items() if pl != "BuC"]
    pools_txt = "per-probe pool honest B ∪ C" + (f", except {'; '.join(pex)}" if pex else "")
    odd_txt = (f" Exception, per the record's schemes: {'; '.join(odd)} uses one union pool across both task sets." if odd else "")
    cal = ex["P2_calibration"]
    pretty = {"BuC": "B ∪ C", "B": "B"}
    c1 = (f"{t1}. U is the upper end of an approximate 95% bootstrap interval on X_err, the typical size of the sim − real catch gap across the "
          f"eligible probes after removing sampling noise, in catch points; lower means a tighter bound (registered estimator, Append 19; conditional on "
          f"fixed thresholds). FPR type: per run (share of honest runs alarmed); threshold source: real honest. Tasks with a strategy only. "
          f"Then = X_err v6 on the T196 estimation half ({ex['P1_then_n_with']['value']} tasks with strategy; {pools_txt}). Now = scale-up "
          f"Revision 6s: old tasks = the same estimation half, new tasks = the second task batch, each calibrated on its own real-honest pool.{odd_txt} 'Both pooled' "
          f"is the equal-task-weighted mixture, which the record labels '{rows[0]['combined_label_now']}'. Eligible probes "
          f"{'are the same then and now' if same else 'DIFFER then vs now'} ({el}). Both use 10,000 draws, seed {ex['P1_draws']['now']['seed']}. "
          f"Run 2 is a different launch from run 1 on old tasks and the same launch on new tasks.")
    c2 = (f"{t2}. Real = mean of runs 1 and 2. Thresholds: old tasks on honest {pretty.get(cal['pool_old'], cal['pool_old'])} ({cal['n_pool_old']} runs), "
          f"new tasks on new honest {pretty.get(cal['pool_new'], cal['pool_new'])} ({cal['n_pool_new']} runs), Sol (a) z-scored per fold; FPR type: per run (share of honest runs alarmed); threshold "
          f"source: real honest. Old tasks are the estimation half only. Printed: the record's sim − real gap with its 95% main-level bootstrap "
          f"interval; the record stores intervals for gaps, not for each catch, so bars carry none. Contrast: the common-calibration H3 "
          f"(common_headlines_2026-10-15_102843 @ 3a50d2e88) sets one threshold on the pooled real-honest calibration runs (old B ∪ C, new B and "
          f"the relaunch) and uses all old tasks; the two differ in both calibration and task set. Old tasks without strategy have n = {ex['P2']['S_old|without_strategy']['n_mains']}.")
    return c1, c2


def text_numbers(ex):
    """Every number printed in a title or caption: (figure, what, value as printed, source record, key path)."""
    rows = []
    g = ex["P1"][0]["groups"]
    for i, what in ((0, "then"), (3, "now pooled"), (1, "now old tasks"), (2, "now new tasks")):
        x = g[i]["sim_minus_mean_real"]
        rows.append({"figure": "P1 title/caption", "what": f"U sim − real mean, reply 5%, {what}", "printed": f"{x['U']:.1f}",
                     "record": "then" if i == 0 else "now", "key": x["key"]})
    for i, what in ((0, "then"), (3, "now pooled")):
        x = g[i]["reexecution_run2_minus_run1"]
        rows.append({"figure": "P1 title/caption", "what": f"U run 2 − run 1, reply 5%, {what}", "printed": f"{x['U']:.1f}",
                     "record": "then" if i == 0 else "now", "key": x["key"]})
    rows.append({"figure": "P1 caption", "what": "then tasks with strategy", "printed": str(ex["P1_then_n_with"]["value"]), "record": "then",
                 "key": ex["P1_then_n_with"]["key"]})
    rows.append({"figure": "P1 caption", "what": "draws and seed (now; then equals)", "printed": f"{ex['P1_draws']['now']['n']}, {ex['P1_draws']['now']['seed']}",
                 "record": "now and then", "key": "now: result/design/draws, result/design/seed; then: result/draws"})
    for st, lab in (("S_old|with_strategy", "old with"), ("S_new|with_strategy", "new with"), ("S_new|without_strategy", "new without")):
        s = ex["P2"][st]
        rows.append({"figure": "P2 title/caption", "what": f"Sol (a) gap sim − real mean, {lab}", "printed": f"{s['gap']:+.1f}", "record": "now",
                     "key": s["gap_key"].replace("<delta|ci95>", "delta")})
        lo, hi = s["gap_ci95"]
        rows.append({"figure": "P2 title/caption", "what": f"Sol (a) gap 95% interval, {lab}",
                     "printed": "interval excludes 0" if (lo > 0 or hi < 0) else "interval includes 0", "value": s["gap_ci95"], "record": "now",
                     "key": s["gap_key"].replace("<delta|ci95>", "ci95")})
    cal = ex["P2_calibration"]
    for f_ in ("n_pool_old", "n_pool_new", "pool_old", "pool_new"):
        rows.append({"figure": "P2 caption", "what": f_, "printed": str(cal[f_]), "record": "now", "key": f"{cal['key']}/{f_}"})
    rows.append({"figure": "P2 caption", "what": "old tasks without strategy n", "printed": str(ex["P2"]["S_old|without_strategy"]["n_mains"]), "record": "now",
                 "key": ex["P2"]["S_old|without_strategy"]["n_key"]})
    ids = [
        {"printed": "Append 19", "where": "now record U_label text", "key": "result/results/R|0.05/primary/sim_minus_mean_real/combined|with_strategy/pooled/U_label"},
        {"printed": "v6", "where": "then record label", "key": "result/label"},
        {"printed": "T196 (old tasks)", "where": "then record key name", "key": "result/attempted_tasks_per_arm_t196"},
        {"printed": "Revision 6s", "where": "commit message of the now record's commit 2bf70468d ('Scale-up X_err FINAL on Revision 6s')", "key": None},
        {"printed": "common_headlines_2026-10-15_102843 @ 3a50d2e88",
         "where": "file experiments/2026-10-15_fpr_sweep_507_prep/results/common_headlines_2026-10-15_102843.json at commit 3a50d2e88 (cited, not read)", "key": None},
    ]
    levels = [
        {"printed": "5% / 1% (per-run alarm rate)", "record": "both", "key": "row keys: then result/results/<...|alpha=0.05|0.01>; now result/results/<R|0.05, R|0.01, C|0.05>"},
        {"printed": "95% (U and gap intervals)", "record": "both", "key": "U_approx_upper_95 (U_label: 'approximate bootstrap upper limit'); per_probe/<probe>/ci95"},
        {"printed": "run 1 / run 2 (real arms; real = their mean)", "record": "both",
         "key": "comparator keys sim_minus_mean_real, reexecution_run2_minus_run1; now catch_percent/<stratum>/<catch_run1|catch_run2|catch_mean_real>"},
    ]
    return {"numbers": rows, "identifiers": ids, "levels": levels}


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=600, retries=0, scaledown_window=2)
def render(b_then: bytes, b_now: bytes) -> dict:
    import hashlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    then, now = json.loads(b_then), json.loads(b_now)
    ex = extract(then, now)
    ex["source_sha256"] = {k: hashlib.sha256(b).hexdigest() for k, b in (("then", b_then), ("now", b_now))}
    t1, t2 = titles(ex)
    c1, c2 = captions(ex, t1, t2)
    ex["title_and_caption_numbers"] = text_numbers(ex)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                         "axes.axisbelow": True})
    files = {}

    def top(title, wrap, fs, height):
        """Fraction of the figure height left for axes below a wrapped suptitle."""
        lines = len(textwrap.wrap(title, wrap))
        return 1 - (lines * fs * 1.3 / 72 + 0.1) / height

    def legend_and_layout(fig, ax, title, wrap, fs, height):
        """Suptitle, then a one-row legend under it, then the axes."""
        t = top(title, wrap, fs, height)
        fig.tight_layout(rect=(0, 0, 1, t - (0.3 if wrap > 60 else 0.45) / height))  # before the suptitle, so its space is not reserved twice
        fig.suptitle(textwrap.fill(title, wrap), x=0.01, y=0.995, ha="left", va="top", fontsize=fs)
        h, lab = ax.get_legend_handles_labels()
        fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, t), ncol=2 if wrap > 60 else 1, frameon=False, fontsize=8)

    def p1(width, narrow):
        n = len(ex["P1"])
        height = 3.0 * n + 1.4 if narrow else 5.4
        fig, axes = plt.subplots(n if narrow else 1, 1 if narrow else n, figsize=(width, height), sharey=True, squeeze=False)
        axes = axes.ravel()
        ymax = max(g[c]["U"] for r in ex["P1"] for g in r["groups"] for c, _, _ in COMPS)
        for ax, r in zip(axes, ex["P1"]):
            xs = np.arange(len(r["groups"]))
            for off, (c, col, lab) in zip((-0.2, 0.2), COMPS):
                vals = [g[c]["U"] for g in r["groups"]]
                bars = ax.bar(xs + off, vals, 0.4, color=col, label=lab)
                for bb, v in zip(bars, vals):
                    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(["then (v6)\nold, est. half", "now: old\nest. half", "now: new", "now: pooled\n('strata\ndiffer')"] if width < 12
                               else [g for g, _, _ in GROUPS], fontsize=7 if width < 12 else 7.5)
            ax.set_title(r["row"], fontsize=10, loc="left")
            ax.set_ylim(0, ymax * 1.18)
            if narrow:
                ax.set_ylabel("U (points)", fontsize=8.5)
        if not narrow:
            axes[0].set_ylabel("U: bound on catch error (points)\nlower = tighter")
        wrap, fs = (52, 8.5) if narrow else ((105, 10) if width < 12 else (140, 11))
        legend_and_layout(fig, axes[0], t1, wrap, fs, height)
        return fig

    def p2(width, narrow):
        height = 6.0 if narrow else 5.4
        fig, ax = plt.subplots(figsize=(width, height))
        xs = np.arange(len(STRATA))
        for off, key, col, lab in ((-0.2, "real_mean", BLUE, "real (mean of run 1, run 2)"), (0.2, "sim", ORANGE, "sim")):
            vals = [ex["P2"][st][key] for st, _ in STRATA]
            bars = ax.bar(xs + off, vals, 0.4, color=col, label=lab)
            for bb, v in zip(bars, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        for x, (st, _) in zip(xs, STRATA):
            s = ex["P2"][st]
            ax.text(x, max(s["real_mean"], s["sim"]) + 2, f"gap {s['gap']:+.1f}\n[{s['gap_ci95'][0]:+.1f}, {s['gap_ci95'][1]:+.1f}]",
                    ha="center", va="bottom", fontsize=7.5 if narrow else 8.5)
        ax.set_xticks(xs)
        short = {"S_old|with_strategy": "old,\nstrategy", "S_old|without_strategy": "old,\nnone", "S_new|with_strategy": "new,\nstrategy",
                 "S_new|without_strategy": "new,\nnone"}
        ax.set_xticklabels([f"{short[st] if narrow else lab}\n(n = {ex['P2'][st]['n_mains']})" for st, lab in STRATA], fontsize=7 if narrow else 9)
        ax.set_ylim(0, 118)
        ax.set_ylabel("Sol (a) catch (%) at per-run 5%")
        wrap, fs = (52, 8.5) if narrow else ((105, 10) if width < 10 else (120, 11))
        legend_and_layout(fig, ax, t2, wrap, fs, height)
        return fig

    for name, fn in (("P1_error_bound_then_vs_now", p1), ("P2_sol_by_origin_registered", p2)):
        for suffix, width, dpi, narrow in (("", 14 if name.startswith("P1") else 11, 150, False), ("_900px", 9, 100, False), ("_narrow_400px", 4, 100, True)):
            fig = fn(width, narrow)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            plt.close(fig)
            files[f"{name}{suffix}.png"] = buf.getvalue()
    files["captions.json"] = json.dumps({"P1": {"title": t1, "caption": c1}, "P2": {"title": t2, "caption": c2}}, indent=1, ensure_ascii=False).encode()
    return {"files": files, "values": ex}


def git_bytes(spec: str) -> bytes:
    path, commit = spec.rsplit("@", 1)
    return subprocess.run(["git", "show", f"{commit}:{path}"], check=True, capture_output=True).stdout


@app.local_entrypoint()
def main(then: str, now: str):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    out = render.remote(git_bytes(then), git_bytes(now))
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    for fname, b in out["files"].items():
        (res / f"status_plots_{stamp}_{fname}").write_bytes(b)
    record = {"label": "PRELIMINARY; values read by key from the source records (no recomputation); status-page plots for the viewer",
              "git": head, "stamp_pt": stamp, "app": app.name, "sources": {"then": then, "now": now},
              "how_to_check": "every plotted value sits under values with its key path; every number printed in a title or caption is listed in "
                              "values.title_and_caption_numbers.numbers (identifiers under .identifiers, rate and interval levels under .levels) with its record ('then' or 'now', see sources) and key path",
              "values": out["values"]}
    (res / f"status_plots_{stamp}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print("wrote", sorted(p.name for p in res.glob(f"status_plots_{stamp}*")))
