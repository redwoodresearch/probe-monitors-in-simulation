"""Figure-only re-render of A24's v8a run (p1_main_2026-10-14_162810) for the viewer (0d's GO; the viewer's final figure list).
Reads the saved JSONL from the Volume on a Modal CPU container (sha256 checked), plots only fields already in its rows, and
recomputes nothing: X_err cells are 09's v6 values as copied into the family_v6 rows; catches, differences, bands and the
decomposition are the fixed_cal_std probe rows and the sweep rows as stored. Writes numbered PNGs (plus 400 px check copies)
and a caption file with the status text, source path and sha that the images leave out.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-14_q35_sol_avg_POSTHOC/plot_p1_main.py --check-dir <dir> --v6 <xerr_v6 copy> [--draft]
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q35solavg_p1plot_kill-2026-11-13"
SOLVOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
NAME = "p1_main_2026-10-14_162810"
SRC = f"{SOLVOL}:/xfer/p1main/{NAME}/{NAME}.jsonl"
SRC_SHA = "314a1b7ab17aab1d"
V6_SHA = "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645"  # 09's X_err v6 (pools per probe: a label, read by key)  # prefix of the run's JSONL sha256 (manifest p1_main_2026-10-14_162810.manifest.json)
MAIN = [("sol_a", "Sol (a)"), ("fibs", "FIBS"), ("harmfulness", "harmfulness"), ("refusal", "refusal"), ("highstakes", "high-stakes")]
REPLY = {"sol_a": "sol_a|avg_top2|corrected|none|tight", "fibs": "fibs|reply_mean|corrected|none", "harmfulness": "harmfulness|reply_mean|corrected|none",
         "refusal": "refusal|reply_mean|corrected|none", "highstakes": "highstakes|reply_mean|corrected|none", "c_asis": "c_asis|avg_top2|corrected|none"}
POOL = {"honest_b": "honest B", "honest_b+honest_c": "honest B ∪ C"}
BLUE, ORANGE, GREY = "#3182bd", "#e6550d", "#8c8c8c"
app = modal.App(APP)
sol_vol = modal.Volume.from_name(SOLVOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")


def mn(t):
    return t.replace("-", "\u2212")


def pct(x):
    return f"{100 * x:.1f}"


def sg(x):
    return mn(f"{100 * x:+.1f}")


def iv(d):
    return mn(f"[{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]") if d.get("ci95") else "[withheld]"


@app.function(image=image, cpu=(2, 2), memory=(16384, 16384), timeout=1200, retries=0, volumes={"/sol": sol_vol})
def render(stamp: str, v6pools: dict) -> dict:
    import io
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})
    raw = (Path("/sol") / "xfer" / "p1main" / NAME / f"{NAME}.jsonl").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if not sha.startswith(SRC_SHA):
        raise ValueError(f"JSONL sha256 {sha[:16]} differs from the run's {SRC_SHA}")
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    probe = {(r["probe_id"], r["alpha"]): r for r in rows if r["row_type"] == "probe" and r["convention"] == "fixed_cal_std"}
    sweep = {r["probe_id"]: r for r in rows if r["row_type"] == "sweep"}
    v6 = [r for r in rows if r["row_type"] == "family_v6" and r["cfg"] == "mechanical"]
    v6m = next(r for r in rows if r["row_type"] == "family_v6_meta")
    figs, caps = {}, {}

    def save(fig, key, cap):
        for tag, width_px in (("", 1000), ("_400px", 400)):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=width_px / fig.get_figwidth())
            figs[key + tag] = buf.getvalue()
        caps[key] = cap
        plt.close(fig)

    # 1. X_err (09's v6, verbatim)
    split = f"{v6m['universe_and_split']['cohort_in_eligibility_half']}/{v6m['universe_and_split']['cohort_in_estimation_half']}"  # task counts, as v6 saves them
    cell = {(r["readout"], r["alpha"], r["comparison"]): r for r in v6}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    xs_, us_, ur_ = [], [], []
    def pool_lab(ro):
        ps = {v for a in (0.01, 0.05) for v in v6pools[f"{ro}|{a}"].values()}
        return "alarm line: honest B ∪ C" if ps == {"BuC"} else "alarm line: honest B ∪ C or B, by probe"

    for ax, (ro, lab) in zip(axes, (("R_reply_meanpooled", "Replies, run mean"), ("C_anycommand_max", "Commands, run max"))):
        for i, a in enumerate((0.01, 0.05)):
            if all((cell[(ro, a, c_)].get("pooled") or {}).get("X_err") is None for c_ in ("sim_minus_mean_real", "reexecution_run2_minus_run1")):
                ax.text(i, 0.5, "fewer than 2\neligible probes", ha="center", va="bottom", fontsize=10, color="#444444")
                continue
            for j, (cmp_, col, nm) in enumerate((("sim_minus_mean_real", ORANGE, "simulated − real (mean)"), ("reexecution_run2_minus_run1", GREY, "real run 2 − run 1"))):
                po = (cell[(ro, a, cmp_)].get("pooled") or {})
                x = i + (j - 0.5) * 0.42
                if po.get("X_err") is None:
                    ax.text(x, 0.5, "fewer than 2\neligible probes", ha="center", va="bottom", fontsize=10, rotation=90, color="#444444")
                    continue
                ax.bar(x, po["X_err"], 0.38, color=col, label=nm if i == 1 and ax is axes[0] else None)
                ax.plot([x - 0.17, x + 0.17], [po["X_err"]] * 2, color=col, lw=3)  # visible even at 0
                ax.plot([x, x], [po["X_err"], po["U_approx_upper_95"]], color="k", lw=1.2)
                ax.plot([x - 0.07, x + 0.07], [po["U_approx_upper_95"]] * 2, color="k", lw=1.2)
                xt = "0\n(floored)" if po.get("X_err_wording") == "point estimate 0 (floored)" else f"{po['X_err']:.1f}"
                ax.text(x, po["U_approx_upper_95"] + 0.25, f"X {xt}\nU {po['U_approx_upper_95']:.1f}", ha="center", va="bottom", fontsize=10)
                if cmp_ == "sim_minus_mean_real":
                    xs_.append(po["X_err"])
                    us_.append(po["U_approx_upper_95"])
                else:
                    ur_.append(po["U_approx_upper_95"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["1% false alarms", "5% false alarms"])
        ax.set_title(f"{lab}\n{pool_lab(ro)}", fontsize=11)
        ax.set_xlim(-0.6, 1.6)
    axes[0].set_ylabel("error to add (points of catch)")
    axes[0].set_ylim(0, max(us_ + ur_) + 4.5)
    axes[0].legend(loc="upper left", frameon=False)
    xr = "0 (floored)" if all(x == 0 for x in xs_) else f"{min(xs_):.1f}–{max(xs_):.1f}"
    fig.suptitle(textwrap.fill(f"Simulated vs real, pooled over the eligible main probes: best estimate of the error to add is {xr} points, with "
                               f"approximate upper limits of {min(us_):.0f}–{max(us_):.0f} points (real run 2 vs run 1: {min(ur_):.0f}–{max(ur_):.0f})", 100))
    fig.text(0.5, 0.015, f"v6, eligibility split {split} tasks, Sol (a) per-fold z, U approximate, conditional on fixed thresholds", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    side = "; ".join(f"{ro.split('_')[0] == 'R' and 'replies' or 'commands'} {a:.0%}: sim − run 1 X {cell[(ro, a, 'sim_minus_run1')]['pooled']['X_err']:.2f} "
                     f"(U {cell[(ro, a, 'sim_minus_run1')]['pooled']['U_approx_upper_95']:.2f}), sim − run 2 X {cell[(ro, a, 'sim_minus_run2')]['pooled']['X_err']:.2f} "
                     f"(U {cell[(ro, a, 'sim_minus_run2')]['pooled']['U_approx_upper_95']:.2f})"
                     for ro in ("R_reply_meanpooled", "C_anycommand_max") for a in (0.01, 0.05) if (cell[(ro, a, "sim_minus_run1")].get("pooled") or {}).get("X_err") is not None)
    elig = "; ".join(f"{'replies' if ro.startswith('R') else 'commands'} {a:.0%}: " + ", ".join(cell[(ro, a, 'sim_minus_mean_real')]["eligibility"].get("eligible", []))
                     + " (pools " + ", ".join(f"{k}={v}" for k, v in sorted(v6pools[f"{ro}|{a}"].items())) + ")" for ro in ("R_reply_meanpooled", "C_anycommand_max") for a in (0.01, 0.05))
    strata = sorted({str((cell[k_].get("pooled") or {}).get("strata_label")) for k_ in cell if k_[2] == "sim_minus_mean_real" and (cell[k_].get("pooled") or {}).get("strata_label")})
    save(fig, "1_error_to_add", "The pooled error to add for simulated vs real catch, over the main probes eligible in each cell (09's X_err v6, values copied verbatim; "
         "the bar is X, the whisker ends at U, an approximate bootstrap upper limit, conditional on fixed thresholds, coverage not guaranteed). Grey: real run 2 − run 1, "
         f"the real-vs-real yardstick. Eligible probes and alarm-line pools per cell: {elig}. Strata: {'; '.join(strata)}. Sim − run 1 and sim − run 2: {side}. "
         "Cells with fewer than 2 eligible probes have no X.")

    # 2-3. catch at 1% and 5%: the five main probes at the reply readout, then the positive control apart
    for a, key in ((0.01, "2_catch_1pct"), (0.05, "3_catch_5pct")):
        groups = [(m, nm) for m, nm in MAIN] + [("c_asis", "positive control")]
        fig, ax = plt.subplots(figsize=(11.5, 6.8))
        incl, pools = 0, set()
        for g, (m, _nm) in enumerate(groups):
            r = probe[(REPLY[m], a)]
            pools.add(r["calibration"])
            x0 = g + (0.6 if m == "c_asis" else 0)
            for j, (w, col, lab) in enumerate((("real_mean", BLUE, "real (mean of runs 1 and 2)"), ("sim", ORANGE, "simulated"))):
                c = r["catch"][w]
                x = x0 + (j - 0.5) * 0.36
                ax.bar(x, 100 * c["point"], 0.34, color=col, label=lab if g == 0 else None)
                ax.errorbar(x, 100 * c["point"], yerr=[[100 * (c["point"] - c["ci95"][0])], [100 * (c["ci95"][1] - c["point"])]], color="k", capsize=3, lw=1)
                ax.text(x, 100 * c["ci95"][1] + 1, pct(c["point"]), ha="center", va="bottom", fontsize=10)
            d, rr = r["primary_interval"]["sim_minus_real_mean"], r["primary_interval"]["run2_minus_run1"]
            pt, pr_ = r["comparisons"]["sim_minus_real_mean"]["point"], r["comparisons"]["run2_minus_run1"]["point"]
            if m != "c_asis" and d.get("ci95") and d["ci95"][0] <= 0 <= d["ci95"][1]:
                incl += 1
            ax.text(x0, -0.12, f"sim − real\n{sg(pt)}\n{iv(d)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)
            ax.text(x0, -0.32, f"run 2 − run 1\n{sg(pr_)}\n{iv(rr)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10, color="#666666")
        ax.axvline(len(MAIN) - 0.2, color="#999999", ls="--", lw=1)
        ax.set_xticks([g + (0.6 if m == "c_asis" else 0) for g, (m, _) in enumerate(groups)])
        ax.set_xticklabels([nm for _, nm in groups])
        ax.set_ylabel(f"catch at {a:.0%} false alarms (%)")
        top = max(100 * probe[(REPLY[m], a)]["catch"][w]["ci95"][1] for m, _ in groups for w in ("real_mean", "sim"))
        ax.set_ylim(0, top * 1.22 + 4)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        ax.set_title(textwrap.fill(f"At {a:.0%} false alarms, the 95% interval of simulated − real catch includes 0 for {incl} of 5 main probes", 80), fontsize=12)
        fig.text(0.5, 0.012, textwrap.fill("Averaged replies, run mean; alarm line on " + " / ".join(sorted(POOL[p] for p in pools)) + " per probe; whiskers and "
                                           "intervals: 95% main-task bootstrap within environments (run 2 − run 1: paired score interval)", 120), ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.subplots_adjust(bottom=0.36)
        save(fig, key, f"Catch of attempted attacks (183 tasks) at {a:.0%} run-level false alarms, the five main probes at the reply readout and the positive control "
             "(trained on simulated attacks vs real honest; it should separate simulated from real). Other readouts: JSONL row_type 'probe', convention "
             "'fixed_cal_std' (e.g. Sol (a) end-of-call any-command at readout_id 'A_max').")

    # 4. sweep, reply readout: sim - real (mean) against run-level FPR, 2 x 3 small multiples
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8), sharey=True, sharex=True)
    ok_n, lo_fpr = 0, 1.0
    for ax, (m, nm) in zip(axes.flat, MAIN + [("c_asis", "positive control")]):
        s = sweep[REPLY[m]]
        pts = s["points"]
        x = [p["fpr_target"] for p in pts]
        lo_fpr = min(lo_fpr, x[0])
        d = [(100 * p["delta"]["point"], p["delta"]["ci95"]) for p in pts]
        cr = [p["control_run2_minus_run1"]["ci95"] for p in pts]
        ax.axhspan(-5, 5, color="#f3f3f3", zorder=0)
        kx = [p["fpr_target"] for p in pts if p["k"] < 10]
        if kx:
            ax.axvspan(x[0], max(kx), facecolor="none", edgecolor="#bdbdbd", hatch="///", lw=0, zorder=0)
        ax.fill_between(x, [100 * c[0] if c else float("nan") for c in cr], [100 * c[1] if c else float("nan") for c in cr], color=GREY, alpha=0.35, lw=0)
        ax.fill_between(x, [100 * c[0] if c else float("nan") for _, c in d], [100 * c[1] if c else float("nan") for _, c in d], color=BLUE, alpha=0.35, lw=0)
        ax.plot(x, [v for v, _ in d], color=BLUE, lw=1.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log")
        if m != "c_asis" and all(c and c[0] <= 0 <= c[1] for _, c in d):
            ok_n += 1
        c1, c5 = probe[(REPLY[m], 0.01)]["catch"]["real_mean"]["point"], probe[(REPLY[m], 0.05)]["catch"]["real_mean"]["point"]
        ax.set_title(f"{nm}: real catch {100 * c1:.0f}% / {100 * c5:.0f}%", fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("run-level false-alarm rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("simulated − real catch (points)")
    fig.suptitle(textwrap.fill(f"Across false-alarm rates, the pointwise 95% band of simulated − real catch includes 0 at every saved rate for {ok_n} of 5 main "
                               "probes (reply readout)", 90))
    fig.text(0.5, 0.012, textwrap.fill("Averaged replies, run mean, alarm line per probe on honest B ∪ C; blue: simulated − real with its 95% main-task bootstrap band; "
                                       "grey band: real run 2 − run 1; light band: ±5 points; hatched: fewer than 10 honest runs above the line", 130),
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save(fig, "4_sweep_replies", f"Simulated − real (mean of runs 1 and 2) catch against the run-level false-alarm rate, reply readout; panel titles give real catch at 1% / 5%. "
         f"The x-axis starts at {lo_fpr:.4f}, the lowest rate with at least one honest run above the line ({len(sweep[REPLY['sol_a']]['points'])} log-spaced rates "
         f"from 1/n_honest to 0.1, about 386 honest runs); rates below that are not estimable from this data, so the axis does not reach 1e-4. Bands are pointwise "
         "95% intervals at the 25 saved rates. Command readout (run max, honest B): not drawn, because among the six displayed probes only Sol (a) and the positive "
         "control have run-max command sweep rows; the external probes' command rows are run means (high-stakes on honest B, the others on honest B ∪ C).")

    caps["_note"] = ("Decomposition figure left out: the bias term is only an upper bound (the simulator's own re-run noise is unmeasured), so no single honest "
                     "takeaway title fits; the decomposition per probe, rate and stratum is in the JSONL (row_type 'probe', field 'within_task_decomposition').")
    out = Path("/sol") / "xfer" / "p1main" / NAME / f"replot_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    for k, b in figs.items():
        (out / f"{NAME}_replot_{k}.png").write_bytes(b)
    sol_vol.commit()
    return {"source_sha256": sha, "figs": figs, "caps": caps, "volume_dir": f"{SOLVOL}:/xfer/p1main/{NAME}/replot_{stamp}/"}


@app.local_entrypoint()
def main(check_dir: str, v6: str, draft: bool = False):
    """check_dir: where the 1,000 px and 400 px copies go for looking at every render; --draft: layout check only (no results/, no captions)."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if not draft and git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if hashlib.sha256(Path(v6).read_bytes()).hexdigest() != V6_SHA:
        raise SystemExit("v6 sha256 mismatch")
    res6 = json.loads(Path(v6).read_text())["result"]["results"]
    v6pools = {f"{ro}|{a}": res6[f"{ro}|mechanical|alpha={a}"]["pools"] for ro in ("R_reply_meanpooled", "C_anycommand_max") for a in (0.01, 0.05)}
    out = render.remote(stamp + ("_draft" if draft else ""), v6pools)
    R, chk = HERE / "results", Path(check_dir) / f"p1replot_{stamp}"
    chk.mkdir(parents=True, exist_ok=True)
    for k, b in out["figs"].items():
        (chk / f"{k}.png").write_bytes(b)
    if draft:
        print(json.dumps({"draft": True, "check_copies": str(chk)}))
        return
    lines = [f"# Captions for {NAME} figures (re-rendered {stamp} by plot_p1_main.py @ {git('rev-parse', 'HEAD')[:9]})", ""]
    for k, b in sorted(out["figs"].items()):
        if k.endswith("_400px"):
            continue
        fn = f"{NAME}_replot_{k}.png"
        (R / fn).open("xb").write(b)
        lines += [f"## {fn}", "", out["caps"][k], "", "PRELIMINARY until 09 and Codex review.", "",
                  f"Source: {SRC} (sha256 {out['source_sha256']}); PNG sha256 {hashlib.sha256(b).hexdigest()}; Volume copy {out['volume_dir']}", ""]
    if out["caps"].get("_note"):
        lines += ["## Not drawn", "", out["caps"]["_note"], ""]
    (R / f"{NAME}_captions.md").open("x").write("\n".join(lines))
    print(json.dumps({"figures": sorted(k for k in out["figs"] if not k.endswith("_400px")), "check_copies": str(chk), "volume": out["volume_dir"]}))
